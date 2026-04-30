# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Integration tests verifying ``RDMATransport`` plugs into ``Manager`` correctly.

These tests exercise the wiring (``Manager.__init__`` accepts the transport,
metadata flows through the quorum RPC, ``should_commit`` drives the
disallow/send/disallow lifecycle) without requiring RDMA hardware. The transport
runs in fallback mode on this machine, so the underlying transfers go through
``HTTPTransport`` -- but the contract being verified (the API surface the
Manager actually calls) is the same as the RDMA path.
"""

from datetime import timedelta
from typing import Optional
from unittest import TestCase
from unittest.mock import create_autospec, MagicMock, patch

import torch
from torch.distributed import TCPStore
from torchft._torchft import QuorumResult
from torchft.checkpointing.http_transport import HTTPTransport
from torchft.checkpointing.rdma_transport import (
    _RDMA_META_PREFIX,
    RDMATransport,
)
from torchft.manager import Manager, MANAGER_ADDR_KEY, REPLICA_ID_KEY
from torchft.process_group import ProcessGroup


def _mock_should_commit(
    rank: int, step: int, should_commit: bool, timeout: timedelta
) -> bool:
    return should_commit


class TestRDMAManagerWiring(TestCase):
    """Verify RDMATransport satisfies the ``CheckpointTransport`` contract Manager calls."""

    store: TCPStore
    manager: Optional[Manager] = None

    def tearDown(self) -> None:
        if self.manager is not None:
            self.manager.shutdown(wait=False)
            self.manager = None

    def _create_manager_with_rdma_transport(
        self,
        timeout: timedelta = timedelta(seconds=10),
    ) -> Manager:
        pg = create_autospec(ProcessGroup)
        pg.errored.return_value = None

        self.store = TCPStore(
            host_name="localhost", port=0, is_master=True, wait_for_workers=False
        )
        self.store.set(MANAGER_ADDR_KEY, "dummy")
        self.store.set(REPLICA_ID_KEY, "dummy_id")

        transport: RDMATransport = RDMATransport(
            device=torch.device("cpu"),
            timeout=timeout,
        )

        self.load_state_dict_mock = MagicMock()

        with patch(
            "os.environ",
            {
                "MASTER_ADDR": "localhost",
                "MASTER_PORT": str(self.store.port),
                "RANK": "1",
                "WORLD_SIZE": "2",
            },
        ):
            manager = Manager(
                pg=pg,
                min_replica_size=2,
                load_state_dict=self.load_state_dict_mock,
                state_dict=lambda: {"weight": torch.tensor([1.0, 2.0, 3.0])},
                use_async_quorum=False,
                timeout=timeout,
                init_sync=True,
                checkpoint_transport=transport,
            )
        self.manager = manager
        return manager

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_rdma_transport_accepted_by_manager_init(
        self, client_mock: MagicMock
    ) -> None:
        """Manager.__init__ accepts an RDMATransport instance and uses it."""
        manager = self._create_manager_with_rdma_transport()
        self.assertIsInstance(manager._checkpoint_transport, RDMATransport)
        # On this CPU-only machine the transport runs in fallback mode.
        self.assertIsNotNone(manager._checkpoint_transport._fallback)
        self.assertIsInstance(
            manager._checkpoint_transport._fallback, HTTPTransport
        )

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_metadata_flows_through_quorum_rpc(
        self, client_mock: MagicMock
    ) -> None:
        """Manager passes the transport's metadata into the quorum RPC."""
        manager = self._create_manager_with_rdma_transport()
        client_mock().should_commit = _mock_should_commit

        quorum = QuorumResult()
        quorum.quorum_id = 1
        quorum.replica_rank = 1
        quorum.replica_world_size = 2
        quorum.recover_src_manager_address = "manager address"
        quorum.store_address = f"localhost:{self.store.port}"
        quorum.max_step = 1
        quorum.max_replica_rank = 1
        quorum.max_world_size = 2
        quorum.heal = False
        client_mock()._quorum.return_value = quorum

        manager.start_quorum()
        manager.wait_quorum()

        kwargs = client_mock()._quorum.call_args.kwargs
        sent_metadata = kwargs["checkpoint_metadata"]
        # In fallback mode the URL is the HTTPTransport's URL; in RDMA mode
        # it would carry the rdma: prefix. Either way it must equal what the
        # transport currently advertises.
        self.assertEqual(sent_metadata, manager._checkpoint_transport.metadata())
        # Sanity: the metadata isn't empty and reflects fallback (no rdma: prefix).
        self.assertTrue(sent_metadata.startswith("http://"))
        self.assertFalse(sent_metadata.startswith(_RDMA_META_PREFIX))

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_recovering_replica_fetches_state_via_transport(
        self, client_mock: MagicMock
    ) -> None:
        """A healing quorum drives recv_checkpoint through the transport."""
        manager = self._create_manager_with_rdma_transport()
        client_mock().should_commit = _mock_should_commit

        # First publish a checkpoint we can recover from -- this is what a peer
        # primary would do via the same transport API the Manager uses.
        published_state = {
            "torchft": {"step": 7, "batches_committed": 14},
            "user": {"default": {"weight": torch.tensor([4.2, 4.3, 4.4])}},
        }
        manager._checkpoint_transport.send_checkpoint(
            dst_ranks=[],
            step=7,
            state_dict=published_state,
            timeout=timedelta(seconds=10),
        )
        client_mock()._checkpoint_metadata.return_value = (
            manager._checkpoint_transport.metadata()
        )

        quorum = QuorumResult()
        quorum.quorum_id = 1
        quorum.replica_rank = 1
        quorum.replica_world_size = 2
        quorum.recover_src_manager_address = "manager address"
        quorum.recover_src_replica_rank = 0
        quorum.store_address = f"localhost:{self.store.port}"
        quorum.max_step = 7
        quorum.max_replica_rank = None
        quorum.max_world_size = 2
        quorum.heal = True
        client_mock()._quorum.return_value = quorum

        manager.start_quorum()

        # With ``use_async_quorum=False``, ``start_quorum`` runs
        # ``_apply_pending_state_dict`` synchronously, which feeds the
        # recovered user state into the user's load_state_dict callback and
        # then clears ``_pending_state_dict``. We verify the callback was
        # invoked with the data the transport pulled back.
        self.assertEqual(manager.current_step(), 7)
        self.assertEqual(manager.batches_committed(), 14)
        self.load_state_dict_mock.assert_called_once()
        recovered_user_state = self.load_state_dict_mock.call_args.args[0]
        torch.testing.assert_close(
            recovered_user_state["weight"],
            published_state["user"]["default"]["weight"],
        )

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_should_commit_drives_disallow_lifecycle(
        self, client_mock: MagicMock
    ) -> None:
        """``should_commit`` calls ``disallow_checkpoint`` on the transport."""
        manager = self._create_manager_with_rdma_transport()
        client_mock().should_commit = _mock_should_commit

        quorum = QuorumResult()
        quorum.quorum_id = 1
        quorum.replica_rank = 1
        quorum.replica_world_size = 2
        quorum.recover_src_manager_address = "manager address"
        quorum.store_address = f"localhost:{self.store.port}"
        quorum.max_step = 1
        quorum.max_replica_rank = 1
        quorum.max_world_size = 2
        quorum.heal = False
        client_mock()._quorum.return_value = quorum

        # First send to leave the transport in the allowed state.
        manager._checkpoint_transport.send_checkpoint(
            dst_ranks=[],
            step=1,
            state_dict={"torchft": {"step": 1, "batches_committed": 2}},
            timeout=timedelta(seconds=10),
        )
        # In fallback mode, _disallowed mirrors the underlying HTTPTransport.
        self.assertFalse(manager._checkpoint_transport._fallback._disallowed)

        manager.start_quorum()
        manager.allreduce(torch.tensor([1.0])).wait()
        self.assertTrue(manager.should_commit())

        # After should_commit the transport is back in the disallowed state.
        self.assertTrue(manager._checkpoint_transport._fallback._disallowed)

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_full_send_disallow_send_cycle(self, client_mock: MagicMock) -> None:
        """Send -> disallow -> send round trip via the manager's transport handle."""
        manager = self._create_manager_with_rdma_transport()
        transport = manager._checkpoint_transport

        # Cycle 1.
        transport.send_checkpoint(
            dst_ranks=[],
            step=1,
            state_dict={"torchft": {"step": 1, "batches_committed": 2}},
            timeout=timedelta(seconds=10),
        )
        self.assertFalse(transport._fallback._disallowed)
        meta1 = transport.metadata()
        self.assertTrue(meta1.startswith("http://"))

        transport.disallow_checkpoint()
        self.assertTrue(transport._fallback._disallowed)

        # Cycle 2.
        transport.send_checkpoint(
            dst_ranks=[],
            step=2,
            state_dict={"torchft": {"step": 2, "batches_committed": 4}},
            timeout=timedelta(seconds=10),
        )
        self.assertFalse(transport._fallback._disallowed)
        # Metadata is stable across sends in fallback mode (same URL).
        self.assertEqual(transport.metadata(), meta1)

        transport.disallow_checkpoint()
        self.assertTrue(transport._fallback._disallowed)

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_manager_shutdown_shuts_down_transport(
        self, client_mock: MagicMock
    ) -> None:
        """Manager.shutdown propagates to the underlying RDMATransport."""
        manager = self._create_manager_with_rdma_transport()
        transport = manager._checkpoint_transport
        # Spy on the transport's shutdown to confirm Manager calls it.
        with patch.object(
            transport, "shutdown", wraps=transport.shutdown
        ) as shutdown_spy:
            manager.shutdown(wait=True)
            self.manager = None  # already shut down -> skip teardown shutdown
            shutdown_spy.assert_called_once()
