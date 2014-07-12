import unittest2

# Add the directory above ours to the python library path since we
# will import from there.
import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import gdbremote_testcase
import os
import select
import tempfile
import time
from lldbtest import *

class TestStubNamedPipeTestCase(gdbremote_testcase.GdbRemoteTestCaseBase):
    def create_named_pipe(self):
        temp_dir = tempfile.mkdtemp()
        named_pipe_path = os.path.join(temp_dir, "stub_port_number")
        try:
            os.mkfifo(named_pipe_path)
        except OSError, e:
            # print "Failed to create named pipe: %s" % e
            raise e
        return named_pipe_path

    def get_port_from_named_pipe(self):
        # Set port to 0
        self.port = 0

        # Don't turn on any kind of logging
        self.debug_monitor_extra_args = ""

        # Create the named pipe that we're reading on.
        self.named_pipe_path = self.create_named_pipe()
        self.assertIsNotNone(self.named_pipe_path)
        # print "using named pipe:{}".format(self.named_pipe_path)
        try:
            # print "launching server..."
            server = self.launch_debug_monitor()
            # print "server launched..."
            self.assertIsNotNone(server)
            self.assertTrue(server.isalive())
            server.expect("(debugserver|lldb-gdbserver)", timeout=10)

            # print "about to open named pipe..."
            # Open the read side of the pipe in non-blocking mode.  This will return right away, ready or not.
            fd = os.open(self.named_pipe_path, os.O_RDONLY | os.O_NONBLOCK)
            named_pipe = os.fdopen(fd, "r")
            self.assertIsNotNone(named_pipe)

            # print "waiting on content from the named pipe..."
            # Wait for something to read with a max timeout.
            (ready_readers, _, _) = select.select([fd], [], [], 5)
            self.assertIsNotNone(ready_readers, "write side of pipe has not written anything - stub isn't writing to pipe.")
            self.assertNotEqual(len(ready_readers), 0, "write side of pipe has not written anything - stub isn't writing to pipe.")

            try:
                # Read the port from the named pipe.
                stub_port_raw = named_pipe.read()
                self.assertIsNotNone(stub_port_raw)
                self.assertNotEqual(len(stub_port_raw), 0, "no content to read on pipe")

                # Trim null byte, convert to int.
                stub_port_raw = stub_port_raw[:-1]
                stub_port = int(stub_port_raw)
                self.assertTrue(stub_port > 0)
            finally:
                named_pipe.close()
            # print "stub is listening on port: {} (from text '{}')".format(stub_port, stub_port_raw)
        finally:
            temp_dir = os.path.dirname(self.named_pipe_path)
            try:
                os.remove(self.named_pipe_path)
            except:
                # Not required.
                None
            os.rmdir(temp_dir)

    @debugserver_test
    def test_get_port_from_named_pipe_debugserver(self):
        self.init_debugserver_test()
        self.set_inferior_startup_launch()
        self.get_port_from_named_pipe()

    @llgs_test
    def test_get_port_from_named_pipe_llgs(self):
        self.init_llgs_test()
        self.set_inferior_startup_launch()
        self.get_port_from_named_pipe()
