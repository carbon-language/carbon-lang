"""
Test lldb-gdbserver operation
"""

import unittest2
import pexpect
import socket
import sys
from lldbtest import *
from lldbgdbserverutils import *
import logging

class LldbGdbServerTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    port = 12345

    _TIMEOUT_SECONDS = 5

    _GDBREMOTE_KILL_PACKET = "$k#6b"

    _LOGGING_LEVEL = logging.WARNING
#    _LOGGING_LEVEL = logging.DEBUG

    def setUp(self):
        TestBase.setUp(self)
        self.lldb_gdbserver_exe = get_lldb_gdbserver_exe()
        if not self.lldb_gdbserver_exe:
            self.skipTest("lldb_gdbserver exe not specified")

        FORMAT = '%(asctime)-15s %(levelname)-8s %(message)s'
        logging.basicConfig(format=FORMAT)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self._LOGGING_LEVEL)

    def create_socket(self):
        sock = socket.socket()
        logger = self.logger

        def shutdown_socket():
            if sock:
                try:
                    # send the kill packet so lldb-gdbserver shuts down gracefully
                    sock.sendall(LldbGdbServerTestCase._GDBREMOTE_KILL_PACKET)
                except:
                    logger.warning("failed to send kill packet to debug monitor: {}; ignoring".format(sys.exc_info()[0]))

                try:
                    sock.close()
                except:
                    logger.warning("failed to close socket to debug monitor: {}; ignoring".format(sys.exc_info()[0]))

        self.addTearDownHook(shutdown_socket)

        sock.connect(('localhost', self.port))
        return sock

    def start_server(self):
        # start the server
        server = pexpect.spawn("{} localhost:{}".format(self.lldb_gdbserver_exe, self.port))

        # Turn on logging for what the child sends back.
        if self.TraceOn():
            server.logfile_read = sys.stdout

        # Schedule lldb-gdbserver to be shutting down during teardown.
        logger = self.logger
        def shutdown_lldb_gdbserver():
            try:
                server.close()
            except:
                logger.warning("failed to close pexpect server for debug monitor: {}; ignoring".format(sys.exc_info()[0]))

        self.addTearDownHook(shutdown_lldb_gdbserver)

        # Wait until we receive the server ready message before continuing.
        server.expect_exact('Listening for a connection on localhost:{}'.format(self.port))

        # Create a socket to talk to the server
        self.sock = self.create_socket()

        return server

    def create_no_ack_remote_stream(self):
       return [
            "lldb-gdbserver <  19> read packet: +",
            "lldb-gdbserver <  19> read packet: $QStartNoAckMode#b0",
            "lldb-gdbserver <   1> send packet: +",
            "lldb-gdbserver <   6> send packet: $OK#9a",
            "lldb-gdbserver <   1> read packet: +"]


    def test_exe_starts(self):
        server = self.start_server()

    def test_start_no_ack_mode(self):
        server = self.start_server()
        self.assertIsNotNone(server)

        log_lines = self.create_no_ack_remote_stream()

        expect_lldb_gdbserver_replay(self, self.sock, log_lines, True,
                                     self._TIMEOUT_SECONDS, self.logger)

    @unittest2.expectedFailure()
    def test_thread_suffix_supported(self):
        server = self.start_server()
        self.assertIsNotNone(server)

        log_lines = self.create_no_ack_remote_stream()
        log_lines.append([
            "lldb-gdbserver <  26> read packet: $QThreadSuffixSupported#e4",
            "lldb-gdbserver <   6> send packet: $OK#9a"])

        expect_lldb_gdbserver_replay(self, self.sock, log_lines, True,
                                     self._TIMEOUT_SECONDS, self.logger)

    @unittest2.expectedFailure()
    def test_list_threads_in_stop_reply_supported(self):
        server = self.start_server()
        self.assertIsNotNone(server)

        log_lines = self.create_no_ack_remote_stream()
        log_lines.append([
            "lldb-gdbserver <  27> read packet: $QListThreadsInStopReply#21",
            "lldb-gdbserver <   6> send packet: $OK#9a"])

        expect_lldb_gdbserver_replay(self, self.sock, log_lines, True,
                                     self._TIMEOUT_SECONDS, self.logger)


if __name__ == '__main__':
    unittest2.main()
