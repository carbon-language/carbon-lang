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

#include "sanitizer_common/sanitizer_flags.h"
#include "sanitizer_common/sanitizer_deadlock_detector_interface.h"

namespace __tsan {

struct Flags : DDFlags {
  // Enable dynamic annotations, otherwise they are no-ops.
  bool enable_annotations;
  // Suppress a race report if we've already output another race report
  // with the same stack.
  bool suppress_equal_stacks;
  // Suppress a race report if we've already output another race report
  // on the same address.
  bool suppress_equal_addresses;
  // Turns off bug reporting entirely (useful for benchmarking).
  bool report_bugs;
  // Report thread leaks at exit?
  bool report_thread_leaks;
  // Report destruction of a locked mutex?
  bool report_destroy_locked;
  // Report incorrect usages of mutexes and mutex annotations?
  bool report_mutex_bugs;
  // Report violations of async signal-safety
  // (e.g. malloc() call from a signal handler).
  bool report_signal_unsafe;
  // Report races between atomic and plain memory accesses.
  bool report_atomic_races;
  // If set, all atomics are effectively sequentially consistent (seq_cst),
  // regardless of what user actually specified.
  bool force_seq_cst_atomics;
  // Print matched "benign" races at exit.
  bool print_benign;
  // Override exit status if something was reported.
  int exitcode;
  // Exit after first reported error.
  bool halt_on_error;
  // Sleep in main thread before exiting for that many ms
  // (useful to catch "at exit" races).
  int atexit_sleep_ms;
  // If set, periodically write memory profile to that file.
  const char *profile_memory;
  // Flush shadow memory every X ms.
  int flush_memory_ms;
  // Flush symbolizer caches every X ms.
  int flush_symbolizer_ms;
  // Resident memory limit in MB to aim at.
  // If the process consumes more memory, then TSan will flush shadow memory.
  int memory_limit_mb;
  // Stops on start until __tsan_resume() is called (for debugging).
  bool stop_on_start;
  // Controls whether RunningOnValgrind() returns true or false.
  bool running_on_valgrind;
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
  // Die after multi-threaded fork if the child creates new threads.
  bool die_after_fork;
};

Flags *flags();
void InitializeFlags(Flags *flags, const char *env);
}  // namespace __tsan

#endif  // TSAN_FLAGS_H
