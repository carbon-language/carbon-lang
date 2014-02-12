//===-- tsan_flags.cc -----------------------------------------------------===//
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

#include "sanitizer_common/sanitizer_flags.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "tsan_flags.h"
#include "tsan_rtl.h"
#include "tsan_mman.h"

namespace __tsan {

Flags *flags() {
  return &CTX()->flags;
}

// Can be overriden in frontend.
#ifdef TSAN_EXTERNAL_HOOKS
extern "C" const char* __tsan_default_options();
#else
extern "C" SANITIZER_INTERFACE_ATTRIBUTE
const char *WEAK __tsan_default_options() {
  return "";
}
#endif

static void ParseFlags(Flags *f, const char *env) {
  ParseFlag(env, &f->enable_annotations, "enable_annotations");
  ParseFlag(env, &f->suppress_equal_stacks, "suppress_equal_stacks");
  ParseFlag(env, &f->suppress_equal_addresses, "suppress_equal_addresses");
  ParseFlag(env, &f->suppress_java, "suppress_java");
  ParseFlag(env, &f->report_bugs, "report_bugs");
  ParseFlag(env, &f->report_thread_leaks, "report_thread_leaks");
  ParseFlag(env, &f->report_destroy_locked, "report_destroy_locked");
  ParseFlag(env, &f->report_signal_unsafe, "report_signal_unsafe");
  ParseFlag(env, &f->report_atomic_races, "report_atomic_races");
  ParseFlag(env, &f->force_seq_cst_atomics, "force_seq_cst_atomics");
  ParseFlag(env, &f->suppressions, "suppressions");
  ParseFlag(env, &f->print_suppressions, "print_suppressions");
  ParseFlag(env, &f->print_benign, "print_benign");
  ParseFlag(env, &f->exitcode, "exitcode");
  ParseFlag(env, &f->halt_on_error, "halt_on_error");
  ParseFlag(env, &f->atexit_sleep_ms, "atexit_sleep_ms");
  ParseFlag(env, &f->profile_memory, "profile_memory");
  ParseFlag(env, &f->flush_memory_ms, "flush_memory_ms");
  ParseFlag(env, &f->flush_symbolizer_ms, "flush_symbolizer_ms");
  ParseFlag(env, &f->memory_limit_mb, "memory_limit_mb");
  ParseFlag(env, &f->stop_on_start, "stop_on_start");
  ParseFlag(env, &f->running_on_valgrind, "running_on_valgrind");
  ParseFlag(env, &f->history_size, "history_size");
  ParseFlag(env, &f->io_sync, "io_sync");
  ParseFlag(env, &f->die_after_fork, "die_after_fork");
}

void InitializeFlags(Flags *f, const char *env) {
  internal_memset(f, 0, sizeof(*f));

  // Default values.
  f->enable_annotations = true;
  f->suppress_equal_stacks = true;
  f->suppress_equal_addresses = true;
  f->suppress_java = false;
  f->report_bugs = true;
  f->report_thread_leaks = true;
  f->report_destroy_locked = true;
  f->report_signal_unsafe = true;
  f->report_atomic_races = true;
  f->force_seq_cst_atomics = false;
  f->suppressions = "";
  f->print_suppressions = false;
  f->print_benign = false;
  f->exitcode = 66;
  f->halt_on_error = false;
  f->atexit_sleep_ms = 1000;
  f->profile_memory = "";
  f->flush_memory_ms = 0;
  f->flush_symbolizer_ms = 5000;
  f->memory_limit_mb = 0;
  f->stop_on_start = false;
  f->running_on_valgrind = false;
  f->history_size = kGoMode ? 1 : 2;  // There are a lot of goroutines in Go.
  f->io_sync = 1;
  f->die_after_fork = true;

  SetCommonFlagsDefaults(f);
  // Override some common flags defaults.
  f->allow_addr2line = true;

  // Let a frontend override.
  ParseFlags(f, __tsan_default_options());
  ParseCommonFlagsFromString(f, __tsan_default_options());
  // Override from command line.
  ParseFlags(f, env);
  ParseCommonFlagsFromString(f, env);

  // Copy back to common flags.
  *common_flags() = *f;

  // Sanity check.
  if (!f->report_bugs) {
    f->report_thread_leaks = false;
    f->report_destroy_locked = false;
    f->report_signal_unsafe = false;
  }

  if (f->history_size < 0 || f->history_size > 7) {
    Printf("ThreadSanitizer: incorrect value for history_size"
           " (must be [0..7])\n");
    Die();
  }

  if (f->io_sync < 0 || f->io_sync > 2) {
    Printf("ThreadSanitizer: incorrect value for io_sync"
           " (must be [0..2])\n");
    Die();
  }
}

}  // namespace __tsan
