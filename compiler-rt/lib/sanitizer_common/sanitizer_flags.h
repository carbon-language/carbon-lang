//===-- sanitizer_flags.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer/AddressSanitizer runtime.
//
//===----------------------------------------------------------------------===//

#ifndef SANITIZER_FLAGS_H
#define SANITIZER_FLAGS_H

#include "sanitizer_internal_defs.h"

namespace __sanitizer {

void ParseFlag(const char *env, bool *flag, const char *name);
void ParseFlag(const char *env, int *flag, const char *name);
void ParseFlag(const char *env, const char **flag, const char *name);

struct CommonFlags {
  // If set, use the online symbolizer from common sanitizer runtime to turn
  // virtual addresses to file/line locations.
  bool symbolize;
  // Path to external symbolizer. If it is NULL, symbolizer will be looked for
  // in PATH. If it is empty (or if "symbolize" is false), external symbolizer
  // will not be started.
  const char *external_symbolizer_path;
  // If set, allows online symbolizer to run addr2line binary to symbolize
  // stack traces (addr2line will only be used if llvm-symbolizer binary is not
  // available.
  bool allow_addr2line;
  // Strips this prefix from file paths in error reports.
  const char *strip_path_prefix;
  // Use fast (frame-pointer-based) unwinder on fatal errors (if available).
  bool fast_unwind_on_fatal;
  // Use fast (frame-pointer-based) unwinder on malloc/free (if available).
  bool fast_unwind_on_malloc;
  // Intercept and handle ioctl requests.
  bool handle_ioctl;
  // Max number of stack frames kept for each allocation/deallocation.
  int malloc_context_size;
  // Write logs to "log_path.pid".
  // The special values are "stdout" and "stderr".
  // The default is "stderr".
  const char *log_path;
  // Verbosity level (0 - silent, 1 - a bit of output, 2+ - more output).
  int  verbosity;
  // Enable memory leak detection.
  bool detect_leaks;
  // Invoke leak checking in an atexit handler. Has no effect if
  // detect_leaks=false, or if __lsan_do_leak_check() is called before the
  // handler has a chance to run.
  bool leak_check_at_exit;
  // If false, the allocator will crash instead of returning 0 on out-of-memory.
  bool allocator_may_return_null;
  // If false, disable printing error summaries in addition to error reports.
  bool print_summary;
  // Check printf arguments.
  bool check_printf;
  // If set, registers the tool's custom SEGV handler (both SIGBUS and SIGSEGV
  // on OSX).
  bool handle_segv;
  // If set, allows user to register a SEGV handler even if the tool registers
  // one.
  bool allow_user_segv_handler;
  // If set, uses alternate stack for signal handling.
  bool use_sigaltstack;
};

inline CommonFlags *common_flags() {
  extern CommonFlags common_flags_dont_use;
  return &common_flags_dont_use;
}

void SetCommonFlagsDefaults(CommonFlags *f);
void ParseCommonFlagsFromString(CommonFlags *f, const char *str);

}  // namespace __sanitizer

#endif  // SANITIZER_FLAGS_H
