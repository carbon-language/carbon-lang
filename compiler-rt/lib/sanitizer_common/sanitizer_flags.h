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

void ParseFlag(const char *env, bool *flag,
    const char *name, const char *descr);
void ParseFlag(const char *env, int *flag,
    const char *name, const char *descr);
void ParseFlag(const char *env, uptr *flag,
    const char *name, const char *descr);
void ParseFlag(const char *env, const char **flag,
    const char *name, const char *descr);

struct CommonFlags {
  bool symbolize;
  const char *external_symbolizer_path;
  bool allow_addr2line;
  const char *strip_path_prefix;
  bool fast_unwind_on_fatal;
  bool fast_unwind_on_malloc;
  bool handle_ioctl;
  int malloc_context_size;
  const char *log_path;
  int  verbosity;
  bool detect_leaks;
  bool leak_check_at_exit;
  bool allocator_may_return_null;
  bool print_summary;
  bool check_printf;
  bool handle_segv;
  bool allow_user_segv_handler;
  bool use_sigaltstack;
  bool detect_deadlocks;
  uptr clear_shadow_mmap_threshold;
  const char *color;
  bool legacy_pthread_cond;
  bool intercept_tls_get_addr;
  bool help;
  uptr mmap_limit_mb;
};

inline CommonFlags *common_flags() {
  extern CommonFlags common_flags_dont_use;
  return &common_flags_dont_use;
}

void SetCommonFlagsDefaults(CommonFlags *f);
void ParseCommonFlagsFromString(CommonFlags *f, const char *str);
void PrintFlagDescriptions();

}  // namespace __sanitizer

#endif  // SANITIZER_FLAGS_H
