import unittest2

import gdbremote_testcase
from lldbtest import *

class TestGdbRemote_qThreadStopInfo(gdbremote_testcase.GdbRemoteTestCaseBase):

    def gather_stop_replies_via_qThreadStopInfo(self, thread_count):
        # Set up the inferior args.
        inferior_args=[]
        for i in range(thread_count - 1):
            inferior_args.append("thread:new")
        inferior_args.append("sleep:10")
        procs = self.prep_debug_monitor_and_inferior(inferior_args=inferior_args)

        # Assumes test_sequence has anything added needed to setup the initial state.
        # (Like optionally enabling QThreadsInStopReply.)
        self.test_sequence.add_log_lines([
            "read packet: $c#00"
            ], True)
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Give threads time to start up, then break.
        time.sleep(1)
        self.reset_test_sequence()
        self.test_sequence.add_log_lines([
            "read packet: {}".format(chr(03)),
            {"direction":"send", "regex":r"^\$T([0-9a-fA-F]+)([^#]+)#[0-9a-fA-F]{2}$", "capture":{1:"stop_result", 2:"key_vals_text"} },
            ], True)
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Wait until all threads have started.
        threads = self.wait_for_thread_count(thread_count, timeout_seconds=3)
        self.assertIsNotNone(threads)
        self.assertEquals(len(threads), thread_count)

        # Grab stop reply for each thread via qThreadStopInfo{tid:hex}.
        for thread in threads:
            # Run the qThreadStopInfo command.
            self.reset_test_sequence()
            self.test_sequence.add_log_lines([
                "read packet: $qThreadStopInfo{:x}#00".format(thread),
                {"direction":"send", "regex":r"^\$T([0-9a-fA-F]+)([^#]+)#[0-9a-fA-F]{2}$", "capture":{1:"stop_result", 2:"key_vals_text"} },
                ], True)
            context = self.expect_gdbremote_sequence()
            self.assertIsNotNone(context)

            # Parse stop reply contents.
            key_vals_text = context.get("key_vals_text")
            self.assertIsNotNone(key_vals_text)
            kv_dict = self.parse_key_val_dict(key_vals_text)
            self.assertIsNotNone(kv_dict)

            # Verify there is a thread.
            kv_thread = kv_dict.get("thread")
            self.assertIsNotNone(kv_thread)
            self.assertEquals(int(kv_thread, 16), thread)

        return threads

    def qThreadStopInfo_works_for_multiple_threads(self, thread_count):
        # Gather threads from stop notification when QThreadsInStopReply is enabled.
        # stop_reply_threads = self.gather_stop_replies_via_qThreadStopInfo(self.ENABLE_THREADS_IN_STOP_REPLY_ENTRIES, thread_count)
        stop_reply_threads = self.gather_stop_replies_via_qThreadStopInfo(thread_count)
        self.assertEquals(len(stop_reply_threads), thread_count)

    @debugserver_test
    @dsym_test
    def test_qThreadStopInfo_works_for_multiple_threads_debugserver_dsym(self):
        self.init_debugserver_test()
        self.buildDsym()
        self.set_inferior_startup_launch()
        self.qThreadStopInfo_works_for_multiple_threads(5)

    @llgs_test
    @dwarf_test
    def test_qThreadStopInfo_works_for_multiple_threads_llgs_dwarf(self):
        self.init_llgs_test()
        self.buildDwarf()
        self.set_inferior_startup_launch()
        self.qThreadStopInfo_works_for_multiple_threads(5)


if __name__ == '__main__':
    unittest2.main()
