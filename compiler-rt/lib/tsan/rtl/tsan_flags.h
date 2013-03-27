//===-- tsan_flags.h --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
// NOTE: This file may be included into user code.
//===----------------------------------------------------------------------===//

#ifndef TSAN_FLAGS_H
#define TSAN_FLAGS_H

// ----------- ATTENTION -------------
// ThreadSanitizer user may provide its implementation of weak
// symbol __tsan::OverrideFlags(__tsan::Flags). Therefore, this
// header may be included in the user code, and shouldn't include
// other headers from TSan or common sanitizer runtime.

namespace __tsan {

struct Flags {
  // Enable dynamic annotations, otherwise they are no-ops.
  bool enable_annotations;
  // Supress a race report if we've already output another race report
  // with the same stack.
  bool suppress_equal_stacks;
  // Supress a race report if we've already output another race report
  // on the same address.
  bool suppress_equal_addresses;
  // Suppress weird race reports that can be seen if JVM is embed
  // into the process.
  bool suppress_java;
  // Turns off bug reporting entirely (useful for benchmarking).
  bool report_bugs;
  // Report thread leaks at exit?
  bool report_thread_leaks;
  // Report destruction of a locked mutex?
  bool report_destroy_locked;
  // Report violations of async signal-safety
  // (e.g. malloc() call from a signal handler).
  bool report_signal_unsafe;
  // Report races between atomic and plain memory accesses.
  bool report_atomic_races;
  // If set, all atomics are effectively sequentially consistent (seq_cst),
  // regardless of what user actually specified.
  bool force_seq_cst_atomics;
  // Strip that prefix from file paths in reports.
  const char *strip_path_prefix;
  // Suppressions filename.
  const char *suppressions;
  // Print matched suppressions at exit.
  bool print_suppressions;
  // Override exit status if something was reported.
  int exitcode;
  // Write logs to "log_path.pid".
  // The special values are "stdout" and "stderr".
  // The default is "stderr".
  const char *log_path;
  // Sleep in main thread before exiting for that many ms
  // (useful to catch "at exit" races).
  int atexit_sleep_ms;
  // Verbosity level (0 - silent, 1 - a bit of output, 2+ - more output).
  int verbosity;
  // If set, periodically write memory profile to that file.
  const char *profile_memory;
  // Flush shadow memory every X ms.
  int flush_memory_ms;
  // Flush symbolizer caches every X ms.
  int flush_symbolizer_ms;
  // Stops on start until __tsan_resume() is called (for debugging).
  bool stop_on_start;
  // Controls whether RunningOnValgrind() returns true or false.
  bool running_on_valgrind;
  // Path to external symbolizer.
  const char *external_symbolizer_path;
  // Per-thread history size, controls how many previous memory accesses
  // are remembered per thread.  Possible values are [0..7].
  // history_size=0 amounts to 32K memory accesses.  Each next value doubles
  // the amount of memory accesses, up to history_size=7 that amounts to
  // 4M memory accesses.  The default value is 2 (128K memory accesses).
  int history_size;
  // Controls level of synchronization implied by IO operations.
  // 0 - no synchronization
  // 1 - reasonable level of synchronization (write->read)
  // 2 - global synchronization of all IO operations
  int io_sync;
};

Flags *flags();
void InitializeFlags(Flags *flags, const char *env);
}  // namespace __tsan

#endif  // TSAN_FLAGS_H
