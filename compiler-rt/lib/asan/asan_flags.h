//===-- asan_flags.h -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer, an address sanity checker.
//
// ASan runtime flags.
//===----------------------------------------------------------------------===//

#ifndef ASAN_FLAGS_H
#define ASAN_FLAGS_H

#include "sanitizer_common/sanitizer_internal_defs.h"

// ASan flag values can be defined in four ways:
// 1) initialized with default values at startup.
// 2) overriden during compilation of ASan runtime by providing
//    compile definition ASAN_DEFAULT_OPTIONS.
// 3) overriden from string returned by user-specified function
//    __asan_default_options().
// 4) overriden from env variable ASAN_OPTIONS.

namespace __asan {

struct Flags {
  // Size (in bytes) of quarantine used to detect use-after-free errors.
  // Lower value may reduce memory usage but increase the chance of
  // false negatives.
  int  quarantine_size;
  // Minimal size (in bytes) of redzones around heap objects.
  // Requirement: redzone >= 16, is a power of two.
  int  redzone;
  // Maximal size (in bytes) of redzones around heap objects.
  int  max_redzone;
  // If set, prints some debugging information and does additional checks.
  bool debug;
  // Controls the way to handle globals (0 - don't detect buffer overflow
  // on globals, 1 - detect buffer overflow, 2 - print data about registered
  // globals).
  int  report_globals;
  // If set, attempts to catch initialization order issues.
  bool check_initialization_order;
  // If set, uses custom wrappers and replacements for libc string functions
  // to find more errors.
  bool replace_str;
  // If set, uses custom wrappers for memset/memcpy/memmove intinsics.
  bool replace_intrin;
  // Used on Mac only.
  bool mac_ignore_invalid_free;
  // Enables stack-use-after-return checking at run-time.
  bool detect_stack_use_after_return;
  // The minimal and the maximal fake stack size log.
  int min_uar_stack_size_log;
  int max_uar_stack_size_log;
  // Use mmap with 'norserve' flag to allocate fake stack.
  bool uar_noreserve;
  // ASan allocator flag. max_malloc_fill_size is the maximal amount of bytes
  // that will be filled with malloc_fill_byte on malloc.
  int max_malloc_fill_size, malloc_fill_byte;
  // Override exit status if something was reported.
  int  exitcode;
  // If set, user may manually mark memory regions as poisoned or unpoisoned.
  bool allow_user_poisoning;
  // Number of seconds to sleep between printing an error report and
  // terminating application. Useful for debug purposes (when one needs
  // to attach gdb, for example).
  int  sleep_before_dying;
  // Allow the users to work around the bug in Nvidia drivers prior to 295.*.
  bool check_malloc_usable_size;
  // If set, explicitly unmaps (huge) shadow at exit.
  bool unmap_shadow_on_exit;
  // If set, calls abort() instead of _exit() after printing an error report.
  bool abort_on_error;
  // Print various statistics after printing an error message or if atexit=1.
  bool print_stats;
  // Print the legend for the shadow bytes.
  bool print_legend;
  // If set, prints ASan exit stats even after program terminates successfully.
  bool atexit;
  // If set, coverage information will be dumped at shutdown time if the
  // appropriate instrumentation was enabled.
  bool coverage;
  // By default, disable core dumper on 64-bit - it makes little sense
  // to dump 16T+ core.
  bool disable_core;
  // Allow the tool to re-exec the program. This may interfere badly with the
  // debugger.
  bool allow_reexec;
  // If set, prints not only thread creation stacks for threads in error report,
  // but also thread creation stacks for threads that created those threads,
  // etc. up to main thread.
  bool print_full_thread_history;
  // Poison (or not) the heap memory on [de]allocation. Zero value is useful
  // for benchmarking the allocator or instrumentator.
  bool poison_heap;
  // If true, poison partially addressable 8-byte aligned words (default=true).
  // This flag affects heap and global buffers, but not stack buffers.
  bool poison_partial;
  // Report errors on malloc/delete, new/free, new/delete[], etc.
  bool alloc_dealloc_mismatch;
  // If true, assume that memcmp(p1, p2, n) always reads n bytes before
  // comparing p1 and p2.
  bool strict_memcmp;
  // If true, assume that dynamic initializers can never access globals from
  // other modules, even if the latter are already initialized.
  bool strict_init_order;
  // If true, ASan tweaks a bunch of other flags (quarantine, redzone, heap
  // poisoning) to reduce memory consumption as much as possible, and restores
  // them to original values when the first instrumented module is loaded into
  // the process. This is mainly intended to be used on Android.
  bool start_deactivated;
  // If non-zero, try to detect operations like <, <=, >, >= and - on invalid
  // pointer pairs (e.g. when pointers belong to different objects).
  // The bigger the value the harder we try.
  int detect_invalid_pointer_pairs;
};

extern Flags asan_flags_dont_use_directly;
inline Flags *flags() {
  return &asan_flags_dont_use_directly;
}
void InitializeFlags(Flags *f, const char *env);

}  // namespace __asan

#endif  // ASAN_FLAGS_H
