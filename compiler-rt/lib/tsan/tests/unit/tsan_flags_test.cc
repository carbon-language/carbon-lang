//===-- tsan_flags_test.cc ------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
//===----------------------------------------------------------------------===//
#include "tsan_flags.h"
#include "tsan_rtl.h"
#include "gtest/gtest.h"
#include <string>

namespace __tsan {

TEST(Flags, Basic) {
  // At least should not crash.
  Flags f;
  InitializeFlags(&f, 0);
  InitializeFlags(&f, "");
}

TEST(Flags, DefaultValues) {
  Flags f;

  f.enable_annotations = false;
  f.exitcode = -11;
  InitializeFlags(&f, "");
  EXPECT_EQ(66, f.exitcode);
  EXPECT_EQ(true, f.enable_annotations);
}

static const char *options1 =
  " enable_annotations=0"
  " suppress_equal_stacks=0"
  " suppress_equal_addresses=0"
  " suppress_java=0"
  " report_bugs=0"
  " report_thread_leaks=0"
  " report_destroy_locked=0"
  " report_signal_unsafe=0"
  " report_atomic_races=0"
  " force_seq_cst_atomics=0"
  " suppressions=qwerty"
  " print_suppressions=0"
  " print_benign=0"
  " exitcode=111"
  " halt_on_error=0"
  " atexit_sleep_ms=222"
  " profile_memory=qqq"
  " flush_memory_ms=444"
  " flush_symbolizer_ms=555"
  " memory_limit_mb=666"
  " stop_on_start=0"
  " running_on_valgrind=0"
  " history_size=5"
  " io_sync=1"
  " die_after_fork=true"

  " symbolize=0"
  " external_symbolizer_path=asdfgh"
  " allow_addr2line=true"
  " strip_path_prefix=zxcvb"
  " fast_unwind_on_fatal=0"
  " fast_unwind_on_malloc=0"
  " handle_ioctl=0"
  " malloc_context_size=777"
  " log_path=aaa"
  " verbosity=2"
  " detect_leaks=0"
  " leak_check_at_exit=0"
  " allocator_may_return_null=0"
  " print_summary=0"
  "";

static const char *options2 =
  " enable_annotations=true"
  " suppress_equal_stacks=true"
  " suppress_equal_addresses=true"
  " suppress_java=true"
  " report_bugs=true"
  " report_thread_leaks=true"
  " report_destroy_locked=true"
  " report_signal_unsafe=true"
  " report_atomic_races=true"
  " force_seq_cst_atomics=true"
  " suppressions=aaaaa"
  " print_suppressions=true"
  " print_benign=true"
  " exitcode=222"
  " halt_on_error=true"
  " atexit_sleep_ms=123"
  " profile_memory=bbbbb"
  " flush_memory_ms=234"
  " flush_symbolizer_ms=345"
  " memory_limit_mb=456"
  " stop_on_start=true"
  " running_on_valgrind=true"
  " history_size=6"
  " io_sync=2"
  " die_after_fork=false"

  " symbolize=true"
  " external_symbolizer_path=cccccc"
  " allow_addr2line=false"
  " strip_path_prefix=ddddddd"
  " fast_unwind_on_fatal=true"
  " fast_unwind_on_malloc=true"
  " handle_ioctl=true"
  " malloc_context_size=567"
  " log_path=eeeeeee"
  " verbosity=0"
  " detect_leaks=true"
  " leak_check_at_exit=true"
  " allocator_may_return_null=true"
  " print_summary=true"
  "";

void VerifyOptions1(Flags *f) {
  EXPECT_EQ(f->enable_annotations, 0);
  EXPECT_EQ(f->suppress_equal_stacks, 0);
  EXPECT_EQ(f->suppress_equal_addresses, 0);
  EXPECT_EQ(f->suppress_java, 0);
  EXPECT_EQ(f->report_bugs, 0);
  EXPECT_EQ(f->report_thread_leaks, 0);
  EXPECT_EQ(f->report_destroy_locked, 0);
  EXPECT_EQ(f->report_signal_unsafe, 0);
  EXPECT_EQ(f->report_atomic_races, 0);
  EXPECT_EQ(f->force_seq_cst_atomics, 0);
  EXPECT_EQ(f->suppressions, std::string("qwerty"));
  EXPECT_EQ(f->print_suppressions, 0);
  EXPECT_EQ(f->print_benign, 0);
  EXPECT_EQ(f->exitcode, 111);
  EXPECT_EQ(f->halt_on_error, 0);
  EXPECT_EQ(f->atexit_sleep_ms, 222);
  EXPECT_EQ(f->profile_memory, std::string("qqq"));
  EXPECT_EQ(f->flush_memory_ms, 444);
  EXPECT_EQ(f->flush_symbolizer_ms, 555);
  EXPECT_EQ(f->memory_limit_mb, 666);
  EXPECT_EQ(f->stop_on_start, 0);
  EXPECT_EQ(f->running_on_valgrind, 0);
  EXPECT_EQ(f->history_size, 5);
  EXPECT_EQ(f->io_sync, 1);
  EXPECT_EQ(f->die_after_fork, true);

  EXPECT_EQ(f->symbolize, 0);
  EXPECT_EQ(f->external_symbolizer_path, std::string("asdfgh"));
  EXPECT_EQ(f->allow_addr2line, true);
  EXPECT_EQ(f->strip_path_prefix, std::string("zxcvb"));
  EXPECT_EQ(f->fast_unwind_on_fatal, 0);
  EXPECT_EQ(f->fast_unwind_on_malloc, 0);
  EXPECT_EQ(f->handle_ioctl, 0);
  EXPECT_EQ(f->malloc_context_size, 777);
  EXPECT_EQ(f->log_path, std::string("aaa"));
  EXPECT_EQ(f->verbosity, 2);
  EXPECT_EQ(f->detect_leaks, 0);
  EXPECT_EQ(f->leak_check_at_exit, 0);
  EXPECT_EQ(f->allocator_may_return_null, 0);
  EXPECT_EQ(f->print_summary, 0);
}

void VerifyOptions2(Flags *f) {
  EXPECT_EQ(f->enable_annotations, true);
  EXPECT_EQ(f->suppress_equal_stacks, true);
  EXPECT_EQ(f->suppress_equal_addresses, true);
  EXPECT_EQ(f->suppress_java, true);
  EXPECT_EQ(f->report_bugs, true);
  EXPECT_EQ(f->report_thread_leaks, true);
  EXPECT_EQ(f->report_destroy_locked, true);
  EXPECT_EQ(f->report_signal_unsafe, true);
  EXPECT_EQ(f->report_atomic_races, true);
  EXPECT_EQ(f->force_seq_cst_atomics, true);
  EXPECT_EQ(f->suppressions, std::string("aaaaa"));
  EXPECT_EQ(f->print_suppressions, true);
  EXPECT_EQ(f->print_benign, true);
  EXPECT_EQ(f->exitcode, 222);
  EXPECT_EQ(f->halt_on_error, true);
  EXPECT_EQ(f->atexit_sleep_ms, 123);
  EXPECT_EQ(f->profile_memory, std::string("bbbbb"));
  EXPECT_EQ(f->flush_memory_ms, 234);
  EXPECT_EQ(f->flush_symbolizer_ms, 345);
  EXPECT_EQ(f->memory_limit_mb, 456);
  EXPECT_EQ(f->stop_on_start, true);
  EXPECT_EQ(f->running_on_valgrind, true);
  EXPECT_EQ(f->history_size, 6);
  EXPECT_EQ(f->io_sync, 2);
  EXPECT_EQ(f->die_after_fork, false);

  EXPECT_EQ(f->symbolize, true);
  EXPECT_EQ(f->external_symbolizer_path, std::string("cccccc"));
  EXPECT_EQ(f->allow_addr2line, false);
  EXPECT_EQ(f->strip_path_prefix, std::string("ddddddd"));
  EXPECT_EQ(f->fast_unwind_on_fatal, true);
  EXPECT_EQ(f->fast_unwind_on_malloc, true);
  EXPECT_EQ(f->handle_ioctl, true);
  EXPECT_EQ(f->malloc_context_size, 567);
  EXPECT_EQ(f->log_path, std::string("eeeeeee"));
  EXPECT_EQ(f->verbosity, 0);
  EXPECT_EQ(f->detect_leaks, true);
  EXPECT_EQ(f->leak_check_at_exit, true);
  EXPECT_EQ(f->allocator_may_return_null, true);
  EXPECT_EQ(f->print_summary, true);
}

static const char *test_default_options;
extern "C" const char *__tsan_default_options() {
  return test_default_options;
}

TEST(Flags, ParseDefaultOptions) {
  Flags f;

  test_default_options = options1;
  InitializeFlags(&f, "");
  VerifyOptions1(&f);

  test_default_options = options2;
  InitializeFlags(&f, "");
  VerifyOptions2(&f);
}

TEST(Flags, ParseEnvOptions) {
  Flags f;

  InitializeFlags(&f, options1);
  VerifyOptions1(&f);

  InitializeFlags(&f, options2);
  VerifyOptions2(&f);
}

TEST(Flags, ParsePriority) {
  Flags f;

  test_default_options = options2;
  InitializeFlags(&f, options1);
  VerifyOptions1(&f);

  test_default_options = options1;
  InitializeFlags(&f, options2);
  VerifyOptions2(&f);
}

}  // namespace __tsan
