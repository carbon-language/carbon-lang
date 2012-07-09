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
void WEAK OverrideFlags(Flags *f) {
  (void)f;
}

void InitializeFlags(Flags *f, const char *env) {
  internal_memset(f, 0, sizeof(*f));

  // Default values.
  f->enable_annotations = true;
  f->suppress_equal_stacks = true;
  f->suppress_equal_addresses = true;
  f->report_thread_leaks = true;
  f->report_signal_unsafe = true;
  f->force_seq_cst_atomics = false;
  f->strip_path_prefix = "";
  f->suppressions = "";
  f->exitcode = 66;
  f->log_fileno = 2;
  f->atexit_sleep_ms = 1000;
  f->verbosity = 0;
  f->profile_memory = "";
  f->flush_memory_ms = 0;
  f->stop_on_start = false;
  f->running_on_valgrind = false;
  f->use_internal_symbolizer = false;

  // Let a frontend override.
  OverrideFlags(f);

  // Override from command line.
  ParseFlag(env, &f->enable_annotations, "enable_annotations");
  ParseFlag(env, &f->suppress_equal_stacks, "suppress_equal_stacks");
  ParseFlag(env, &f->suppress_equal_addresses, "suppress_equal_addresses");
  ParseFlag(env, &f->report_thread_leaks, "report_thread_leaks");
  ParseFlag(env, &f->report_signal_unsafe, "report_signal_unsafe");
  ParseFlag(env, &f->force_seq_cst_atomics, "force_seq_cst_atomics");
  ParseFlag(env, &f->strip_path_prefix, "strip_path_prefix");
  ParseFlag(env, &f->suppressions, "suppressions");
  ParseFlag(env, &f->exitcode, "exitcode");
  ParseFlag(env, &f->log_fileno, "log_fileno");
  ParseFlag(env, &f->atexit_sleep_ms, "atexit_sleep_ms");
  ParseFlag(env, &f->verbosity, "verbosity");
  ParseFlag(env, &f->profile_memory, "profile_memory");
  ParseFlag(env, &f->flush_memory_ms, "flush_memory_ms");
  ParseFlag(env, &f->stop_on_start, "stop_on_start");
  ParseFlag(env, &f->use_internal_symbolizer, "use_internal_symbolizer");
}

}  // namespace __tsan
