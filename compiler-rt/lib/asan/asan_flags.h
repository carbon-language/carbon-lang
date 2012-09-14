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

#include "sanitizer/common_interface_defs.h"

// ASan flag values can be defined in three ways:
// 1) initialized with default values at startup.
// 2) overriden from string returned by user-specified function
//    __asan_default_options().
// 3) overriden from env variable ASAN_OPTIONS.

namespace __asan {

struct Flags {
  // Size (in bytes) of quarantine used to detect use-after-free errors.
  // Lower value may reduce memory usage but increase the chance of
  // false negatives.
  int  quarantine_size;
  // If set, uses in-process symbolizer from common sanitizer runtime.
  bool symbolize;
  // Verbosity level (0 - silent, 1 - a bit of output, 2+ - more output).
  int  verbosity;
  // Size (in bytes) of redzones around heap objects.
  // Requirement: redzone >= 32, is a power of two.
  int  redzone;
  // If set, prints some debugging information and does additional checks.
  bool debug;
  // Controls the way to handle globals (0 - don't detect buffer overflow
  // on globals, 1 - detect buffer overflow, 2 - print data about registered
  // globals).
  int  report_globals;
  // If set, attempts to catch initialization order issues.
  bool check_initialization_order;
  // Max number of stack frames kept for each allocation.
  int  malloc_context_size;
  // If set, uses custom wrappers and replacements for libc string functions
  // to find more errors.
  bool replace_str;
  // If set, uses custom wrappers for memset/memcpy/memmove intinsics.
  bool replace_intrin;
  // Used on Mac only. See comments in asan_mac.cc and asan_malloc_mac.cc.
  bool replace_cfallocator;
  // Used on Mac only.
  bool mac_ignore_invalid_free;
  // ASan allocator flag. See asan_allocator.cc.
  bool use_fake_stack;
  // ASan allocator flag. Sets the maximal size of allocation request
  // that would return memory filled with zero bytes.
  int  max_malloc_fill_size;
  // Override exit status if something was reported.
  int  exitcode;
  // If set, user may manually mark memory regions as poisoned or unpoisoned.
  bool allow_user_poisoning;
  // Number of seconds to sleep between printing an error report and
  // terminating application. Useful for debug purposes (when one needs
  // to attach gdb, for example).
  int  sleep_before_dying;
  // If set, registers ASan custom segv handler.
  bool handle_segv;
  // If set, uses alternate stack for signal handling.
  bool use_sigaltstack;
  // Allow the users to work around the bug in Nvidia drivers prior to 295.*.
  bool check_malloc_usable_size;
  // If set, explicitly unmaps (huge) shadow at exit.
  bool unmap_shadow_on_exit;
  // If set, calls abort() instead of _exit() after printing an error report.
  bool abort_on_error;
  // If set, prints ASan exit stats even after program terminates successfully.
  bool atexit;
  // By default, disable core dumper on 64-bit - it makes little sense
  // to dump 16T+ core.
  bool disable_core;
  // Allow the tool to re-exec the program. This may interfere badly with the
  // debugger.
  bool allow_reexec;
  // Strips this prefix from file paths in error reports.
  const char *strip_path_prefix;
  // If set, prints not only thread creation stacks for threads in error report,
  // but also thread creation stacks for threads that created those threads,
  // etc. up to main thread.
  bool print_full_thread_history;
  // ASan will write logs to "log_path.pid" instead of stderr.
  const char *log_path;
};

Flags *flags();
void InitializeFlags(Flags *f, const char *env);

}  // namespace __asan

#endif  // ASAN_FLAGS_H
