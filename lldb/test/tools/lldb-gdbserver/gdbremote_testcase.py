"""
Base class for gdb-remote test cases.
"""

import errno
import unittest2
import pexpect
import platform
import sets
import signal
import socket
import subprocess
import sys
import time
from lldbtest import *
from lldbgdbserverutils import *
import logging
import os.path

class GdbRemoteTestCaseBase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    port = 12345

    _TIMEOUT_SECONDS = 5

    _GDBREMOTE_KILL_PACKET = "$k#6b"

    _LOGGING_LEVEL = logging.WARNING
    # _LOGGING_LEVEL = logging.DEBUG

    _STARTUP_ATTACH = "attach"
    _STARTUP_LAUNCH = "launch"

    # GDB Signal numbers that are not target-specific used for common exceptions
    TARGET_EXC_BAD_ACCESS      = 0x91
    TARGET_EXC_BAD_INSTRUCTION = 0x92
    TARGET_EXC_ARITHMETIC      = 0x93
    TARGET_EXC_EMULATION       = 0x94
    TARGET_EXC_SOFTWARE        = 0x95
    TARGET_EXC_BREAKPOINT      = 0x96

    def setUp(self):
        TestBase.setUp(self)
        FORMAT = '%(asctime)-15s %(levelname)-8s %(message)s'
        logging.basicConfig(format=FORMAT)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self._LOGGING_LEVEL)
        self.test_sequence = GdbRemoteTestSequence(self.logger)
        self.set_inferior_startup_launch()

        # Uncomment this code to force only a single test to run (by name).
        #if not re.search(r"P_", self._testMethodName):
        #    self.skipTest("focusing on one test")

    def reset_test_sequence(self):
        self.test_sequence = GdbRemoteTestSequence(self.logger)

    def init_llgs_test(self):
        self.debug_monitor_exe = get_lldb_gdbserver_exe()
        if not self.debug_monitor_exe:
            self.skipTest("lldb_gdbserver exe not found")
        self.debug_monitor_extra_args = " -c 'log enable -T -f process-{}.log lldb break process thread' -c 'log enable -T -f packets-{}.log gdb-remote packets'".format(self.id(), self.id(), self.id())

    def init_debugserver_test(self):
        self.debug_monitor_exe = get_debugserver_exe()
        if not self.debug_monitor_exe:
            self.skipTest("debugserver exe not found")
        self.debug_monitor_extra_args = " --log-file=/tmp/packets-{}.log --log-flags=0x800000".format(self._testMethodName)

    def create_socket(self):
        sock = socket.socket()
        logger = self.logger

        def shutdown_socket():
            if sock:
                try:
                    # send the kill packet so lldb-gdbserver shuts down gracefully
                    sock.sendall(GdbRemoteTestCaseBase._GDBREMOTE_KILL_PACKET)
                except:
                    logger.warning("failed to send kill packet to debug monitor: {}; ignoring".format(sys.exc_info()[0]))

                try:
                    sock.close()
                except:
                    logger.warning("failed to close socket to debug monitor: {}; ignoring".format(sys.exc_info()[0]))

        self.addTearDownHook(shutdown_socket)

        sock.connect(('localhost', self.port))
        return sock

    def set_inferior_startup_launch(self):
        self._inferior_startup = self._STARTUP_LAUNCH

    def set_inferior_startup_attach(self):
        self._inferior_startup = self._STARTUP_ATTACH

    def launch_debug_monitor(self, attach_pid=None):
        # Create the command line.
        commandline = "{}{} localhost:{}".format(self.debug_monitor_exe, self.debug_monitor_extra_args, self.port)
        if attach_pid:
            commandline += " --attach=%d" % attach_pid

        # Start the server.
        server = pexpect.spawn(commandline)

        # Turn on logging for what the child sends back.
        if self.TraceOn():
            server.logfile_read = sys.stdout

        return server

    def connect_to_debug_monitor(self, attach_pid=None):
        server = self.launch_debug_monitor(attach_pid=attach_pid)

        # Wait until we receive the server ready message before continuing.
        server.expect_exact('Listening to port {} for a connection from localhost'.format(self.port))

        # Schedule debug monitor to be shut down during teardown.
        logger = self.logger
        def shutdown_debug_monitor():
            try:
                server.close()
            except:
                logger.warning("failed to close pexpect server for debug monitor: {}; ignoring".format(sys.exc_info()[0]))
        self.addTearDownHook(shutdown_debug_monitor)

        attempts = 0
        MAX_ATTEMPTS = 20

        while attempts < MAX_ATTEMPTS:
            # Create a socket to talk to the server
            try:
                self.sock = self.create_socket()
                return server
            except socket.error as serr:
                # We're only trying to handle connection refused
                if serr.errno != errno.ECONNREFUSED:
                    raise serr

                # Increment attempts.
                print("connect to debug monitor on port %d failed, attempt #%d of %d" % (self.port, attempts + 1, MAX_ATTEMPTS))
                attempts += 1

                # And wait a second before next attempt.
                time.sleep(1)

        raise Exception("failed to create a socket to the launched debug monitor after %d tries" % attempts)

    def launch_process_for_attach(self,inferior_args=None, sleep_seconds=3):
        # We're going to start a child process that the debug monitor stub can later attach to.
        # This process needs to be started so that it just hangs around for a while.  We'll
        # have it sleep.
        exe_path = os.path.abspath("a.out")

        args = [exe_path]
        if inferior_args:
            args.extend(inferior_args)
        if sleep_seconds:
            args.append("sleep:%d" % sleep_seconds)

        return subprocess.Popen(args)

    def prep_debug_monitor_and_inferior(self, inferior_args=None, inferior_sleep_seconds=3):
        """Prep the debug monitor, the inferior, and the expected packet stream.

        Handle the separate cases of using the debug monitor in attach-to-inferior mode
        and in launch-inferior mode.

        For attach-to-inferior mode, the inferior process is first started, then
        the debug monitor is started in attach to pid mode (using --attach on the
        stub command line), and the no-ack-mode setup is appended to the packet
        stream.  The packet stream is not yet executed, ready to have more expected
        packet entries added to it.

        For launch-inferior mode, the stub is first started, then no ack mode is
        setup on the expected packet stream, then the verified launch packets are added
        to the expected socket stream.  The packet stream is not yet executed, ready
        to have more expected packet entries added to it.

        The return value is:
        {inferior:<inferior>, server:<server>}
        """
        inferior = None
        attach_pid = None

        if self._inferior_startup == self._STARTUP_ATTACH:
            # Launch the process that we'll use as the inferior.
            inferior = self.launch_process_for_attach(inferior_args=inferior_args, sleep_seconds=inferior_sleep_seconds)
            self.assertIsNotNone(inferior)
            self.assertTrue(inferior.pid > 0)
            attach_pid = inferior.pid

        # Launch the debug monitor stub, attaching to the inferior.
        server = self.connect_to_debug_monitor(attach_pid=attach_pid)
        self.assertIsNotNone(server)

        if self._inferior_startup == self._STARTUP_LAUNCH:
            # Build launch args
            launch_args = [os.path.abspath('a.out')]
            if inferior_args:
                launch_args.extend(inferior_args)

        # Build the expected protocol stream
        self.add_no_ack_remote_stream()
        if self._inferior_startup == self._STARTUP_LAUNCH:
            self.add_verified_launch_packets(launch_args)

        return {"inferior":inferior, "server":server}

    def add_no_ack_remote_stream(self):
        self.test_sequence.add_log_lines(
            ["read packet: +",
             "read packet: $QStartNoAckMode#b0",
             "send packet: +",
             "send packet: $OK#9a",
             "read packet: +"],
            True)

    def add_verified_launch_packets(self, launch_args):
        self.test_sequence.add_log_lines(
            ["read packet: %s" % build_gdbremote_A_packet(launch_args),
             "send packet: $OK#00",
             "read packet: $qLaunchSuccess#a5",
             "send packet: $OK#00"],
            True)

    def add_thread_suffix_request_packets(self):
        self.test_sequence.add_log_lines(
            ["read packet: $QThreadSuffixSupported#00",
             "send packet: $OK#00",
            ], True)

    def add_process_info_collection_packets(self):
        self.test_sequence.add_log_lines(
            ["read packet: $qProcessInfo#00",
              { "direction":"send", "regex":r"^\$(.+)#[0-9a-fA-F]{2}$", "capture":{1:"process_info_raw"} }],
            True)

    _KNOWN_PROCESS_INFO_KEYS = [
        "pid",
        "parent-pid",
        "real-uid",
        "real-gid",
        "effective-uid",
        "effective-gid",
        "cputype",
        "cpusubtype",
        "ostype",
        "vendor",
        "endian",
        "ptrsize"
        ]

    def parse_process_info_response(self, context):
        # Ensure we have a process info response.
        self.assertIsNotNone(context)
        process_info_raw = context.get("process_info_raw")
        self.assertIsNotNone(process_info_raw)

        # Pull out key:value; pairs.
        process_info_dict = { match.group(1):match.group(2) for match in re.finditer(r"([^:]+):([^;]+);", process_info_raw) }

        # Validate keys are known.
        for (key, val) in process_info_dict.items():
            self.assertTrue(key in self._KNOWN_PROCESS_INFO_KEYS)
            self.assertIsNotNone(val)

        return process_info_dict

    def add_register_info_collection_packets(self):
        self.test_sequence.add_log_lines(
            [ { "type":"multi_response", "query":"qRegisterInfo", "append_iteration_suffix":True,
              "end_regex":re.compile(r"^\$(E\d+)?#[0-9a-fA-F]{2}$"),
              "save_key":"reg_info_responses" } ],
            True)

    def parse_register_info_packets(self, context):
        """Return an array of register info dictionaries, one per register info."""
        reg_info_responses = context.get("reg_info_responses")
        self.assertIsNotNone(reg_info_responses)

        # Parse register infos.
        return [parse_reg_info_response(reg_info_response) for reg_info_response in reg_info_responses]

    def expect_gdbremote_sequence(self, timeout_seconds =None):
        if not timeout_seconds:
            timeout_seconds = self._TIMEOUT_SECONDS
        return expect_lldb_gdbserver_replay(self, self.sock, self.test_sequence, timeout_seconds, self.logger)

    _KNOWN_REGINFO_KEYS = [
        "name",
        "alt-name",
        "bitsize",
        "offset",
        "encoding",
        "format",
        "set",
        "gcc",
        "dwarf",
        "generic",
        "container-regs",
        "invalidate-regs"
    ]

    def assert_valid_reg_info(self, reg_info):
        # Assert we know about all the reginfo keys parsed.
        for key in reg_info:
            self.assertTrue(key in self._KNOWN_REGINFO_KEYS)

        # Check the bare-minimum expected set of register info keys.
        self.assertTrue("name" in reg_info)
        self.assertTrue("bitsize" in reg_info)
        self.assertTrue("offset" in reg_info)
        self.assertTrue("encoding" in reg_info)
        self.assertTrue("format" in reg_info)

    def find_pc_reg_info(self, reg_infos):
        lldb_reg_index = 0
        for reg_info in reg_infos:
            if ("generic" in reg_info) and (reg_info["generic"] == "pc"):
                return (lldb_reg_index, reg_info)
            lldb_reg_index += 1

        return (None, None)

    def add_lldb_register_index(self, reg_infos):
        """Add a "lldb_register_index" key containing the 0-baed index of each reg_infos entry.

        We'll use this when we want to call packets like P/p with a register index but do so
        on only a subset of the full register info set.
        """
        self.assertIsNotNone(reg_infos)

        reg_index = 0
        for reg_info in reg_infos:
            reg_info["lldb_register_index"] = reg_index
            reg_index += 1

    def add_query_memory_region_packets(self, address):
        self.test_sequence.add_log_lines(
            ["read packet: $qMemoryRegionInfo:{0:x}#00".format(address),
             {"direction":"send", "regex":r"^\$(.+)#[0-9a-fA-F]{2}$", "capture":{1:"memory_region_response"} }],
            True)

    def parse_key_val_dict(self, key_val_text):
        self.assertIsNotNone(key_val_text)
        kv_dict = {}
        for match in re.finditer(r";?([^:]+):([^;]+)", key_val_text):
            kv_dict[match.group(1)] = match.group(2)
        return kv_dict

    def parse_memory_region_packet(self, context):
        # Ensure we have a context.
        self.assertIsNotNone(context.get("memory_region_response"))

        # Pull out key:value; pairs.
        mem_region_dict = self.parse_key_val_dict(context.get("memory_region_response"))

        # Validate keys are known.
        for (key, val) in mem_region_dict.items():
            self.assertTrue(key in ["start", "size", "permissions", "error"])
            self.assertIsNotNone(val)

        # Return the dictionary of key-value pairs for the memory region.
        return mem_region_dict

    def assert_address_within_memory_region(self, test_address, mem_region_dict):
        self.assertIsNotNone(mem_region_dict)
        self.assertTrue("start" in mem_region_dict)
        self.assertTrue("size" in mem_region_dict)

        range_start = int(mem_region_dict["start"], 16)
        range_size = int(mem_region_dict["size"], 16)
        range_end = range_start + range_size

        if test_address < range_start:
            self.fail("address 0x{0:x} comes before range 0x{1:x} - 0x{2:x} (size 0x{3:x})".format(test_address, range_start, range_end, range_size))
        elif test_address >= range_end:
            self.fail("address 0x{0:x} comes after range 0x{1:x} - 0x{2:x} (size 0x{3:x})".format(test_address, range_start, range_end, range_size))

    def add_threadinfo_collection_packets(self):
        self.test_sequence.add_log_lines(
            [ { "type":"multi_response", "first_query":"qfThreadInfo", "next_query":"qsThreadInfo",
                "append_iteration_suffix":False, "end_regex":re.compile(r"^\$(l)?#[0-9a-fA-F]{2}$"),
              "save_key":"threadinfo_responses" } ],
            True)

    def parse_threadinfo_packets(self, context):
        """Return an array of thread ids (decimal ints), one per thread."""
        threadinfo_responses = context.get("threadinfo_responses")
        self.assertIsNotNone(threadinfo_responses)

        thread_ids = []
        for threadinfo_response in threadinfo_responses:
            new_thread_infos = parse_threadinfo_response(threadinfo_response)
            thread_ids.extend(new_thread_infos)
        return thread_ids

    def wait_for_thread_count(self, thread_count, timeout_seconds=3):
        start_time = time.time()
        timeout_time = start_time + timeout_seconds

        actual_thread_count = 0
        while actual_thread_count < thread_count:
            self.reset_test_sequence()
            self.add_threadinfo_collection_packets()

            context = self.expect_gdbremote_sequence()
            self.assertIsNotNone(context)

            threads = self.parse_threadinfo_packets(context)
            self.assertIsNotNone(threads)

            actual_thread_count = len(threads)

            if time.time() > timeout_time:
                raise Exception(
                    'timed out after {} seconds while waiting for theads: waiting for at least {} threads, found {}'.format(
                        timeout_seconds, thread_count, actual_thread_count))

        return threads

    def add_set_breakpoint_packets(self, address, do_continue=True, breakpoint_kind=1):
        self.test_sequence.add_log_lines(
            [# Set the breakpoint.
             "read packet: $Z0,{0:x},{1}#00".format(address, breakpoint_kind),
             # Verify the stub could set it.
             "send packet: $OK#00",
             ], True)

        if (do_continue):
            self.test_sequence.add_log_lines(
                [# Continue the inferior.
                 "read packet: $c#00",
                 # Expect a breakpoint stop report.
                 {"direction":"send", "regex":r"^\$T([0-9a-fA-F]{2})thread:([0-9a-fA-F]+);", "capture":{1:"stop_signo", 2:"stop_thread_id"} },
                 ], True)        

    def add_remove_breakpoint_packets(self, address, breakpoint_kind=1):
        self.test_sequence.add_log_lines(
            [# Remove the breakpoint.
             "read packet: $z0,{0:x},{1}#00".format(address, breakpoint_kind),
             # Verify the stub could unset it.
             "send packet: $OK#00",
            ], True)

    def add_qSupported_packets(self):
        self.test_sequence.add_log_lines(
            ["read packet: $qSupported#00",
             {"direction":"send", "regex":r"^\$(.*)#[0-9a-fA-F]{2}", "capture":{1: "qSupported_response"}},
            ], True)

    _KNOWN_QSUPPORTED_STUB_FEATURES = [
        "augmented-libraries-svr4-read",
        "PacketSize",
        "QStartNoAckMode",
        "QThreadSuffixSupported",
        "qXfer:auxv:read",
        "qXfer:libraries:read",
        "qXfer:libraries-svr4:read",
    ]

    def parse_qSupported_response(self, context):
        self.assertIsNotNone(context)

        raw_response = context.get("qSupported_response")
        self.assertIsNotNone(raw_response)

        # For values with key=val, the dict key and vals are set as expected.  For feature+, feature- and feature?, the
        # +,-,? is stripped from the key and set as the value.
        supported_dict = {}
        for match in re.finditer(r";?([^=;]+)(=([^;]+))?", raw_response):
            key = match.group(1)
            val = match.group(3)

            # key=val: store as is
            if val and len(val) > 0:
                supported_dict[key] = val
            else:
                if len(key) < 2:
                    raise Exception("singular stub feature is too short: must be stub_feature{+,-,?}")
                supported_type = key[-1]
                key = key[:-1]
                if not supported_type in ["+", "-", "?"]:
                    raise Exception("malformed stub feature: final character {} not in expected set (+,-,?)".format(supported_type))
                supported_dict[key] = supported_type 
            # Ensure we know the supported element
            if not key in self._KNOWN_QSUPPORTED_STUB_FEATURES:
                raise Exception("unknown qSupported stub feature reported: %s" % key)

        return supported_dict

    def run_process_then_stop(self, run_seconds=1):
        # Tell the stub to continue.
        self.test_sequence.add_log_lines(
             ["read packet: $vCont;c#00"],
             True)
        context = self.expect_gdbremote_sequence()

        # Wait for run_seconds.
        time.sleep(run_seconds)

        # Send an interrupt, capture a T response.
        self.reset_test_sequence()
        self.test_sequence.add_log_lines(
            ["read packet: {}".format(chr(03)),
             {"direction":"send", "regex":r"^\$T([0-9a-fA-F]+)([^#]+)#[0-9a-fA-F]{2}$", "capture":{1:"stop_result"} }],
            True)
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)
        self.assertIsNotNone(context.get("stop_result"))

        return context

    def select_modifiable_register(self, reg_infos):
        """Find a register that can be read/written freely."""
        PREFERRED_REGISTER_NAMES = sets.Set(["rax",])

        # First check for the first register from the preferred register name set.
        alternative_register_index = None

        self.assertIsNotNone(reg_infos)
        for reg_info in reg_infos:
            if ("name" in reg_info) and (reg_info["name"] in PREFERRED_REGISTER_NAMES):
                # We found a preferred register.  Use it.
                return reg_info["lldb_register_index"]
            if ("generic" in reg_info) and (reg_info["generic"] == "fp"):
                # A frame pointer register will do as a register to modify temporarily.
                alternative_register_index = reg_info["lldb_register_index"]

        # We didn't find a preferred register.  Return whatever alternative register
        # we found, if any.
        return alternative_register_index

    def extract_registers_from_stop_notification(self, stop_key_vals_text):
        self.assertIsNotNone(stop_key_vals_text)
        kv_dict = self.parse_key_val_dict(stop_key_vals_text)

        registers = {}
        for (key, val) in kv_dict.items():
            if re.match(r"^[0-9a-fA-F]+", key):
                registers[int(key, 16)] = val
        return registers

    def gather_register_infos(self):
        self.reset_test_sequence()
        self.add_register_info_collection_packets()

        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        reg_infos = self.parse_register_info_packets(context)
        self.assertIsNotNone(reg_infos)
        self.add_lldb_register_index(reg_infos)

        return reg_infos

    def find_generic_register_with_name(self, reg_infos, generic_name):
        self.assertIsNotNone(reg_infos)
        for reg_info in reg_infos:
            if ("generic" in reg_info) and (reg_info["generic"] == generic_name):
                return reg_info
        return None

        