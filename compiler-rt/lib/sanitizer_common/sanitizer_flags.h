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
  // If set, use the online symbolizer from common sanitizer runtime.
  bool symbolize;
  // Path to external symbolizer.
  const char *external_symbolizer_path;
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
  // Write logs to "log_path.pid" instead of stderr.
  const char *log_path;
};

extern CommonFlags common_flags_dont_use_directly;

inline CommonFlags *common_flags() {
  return &common_flags_dont_use_directly;
}

void ParseCommonFlagsFromString(const char *str);

}  // namespace __sanitizer

#endif  // SANITIZER_FLAGS_H
