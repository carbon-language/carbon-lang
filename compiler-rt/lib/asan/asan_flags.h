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
  // Flag descriptions are in asan_rtl.cc.
  int  quarantine_size;
  int  redzone;
  int  max_redzone;
  bool debug;
  int  report_globals;
  bool check_initialization_order;
  bool replace_str;
  bool replace_intrin;
  bool mac_ignore_invalid_free;
  bool detect_stack_use_after_return;
  int min_uar_stack_size_log;
  int max_uar_stack_size_log;
  bool uar_noreserve;
  int max_malloc_fill_size, malloc_fill_byte;
  int  exitcode;
  bool allow_user_poisoning;
  int  sleep_before_dying;
  bool check_malloc_usable_size;
  bool unmap_shadow_on_exit;
  bool abort_on_error;
  bool print_stats;
  bool print_legend;
  bool atexit;
  bool coverage;
  bool disable_core;
  bool allow_reexec;
  bool print_full_thread_history;
  bool poison_heap;
  bool poison_partial;
  bool alloc_dealloc_mismatch;
  bool strict_memcmp;
  bool strict_init_order;
  bool start_deactivated;
  int detect_invalid_pointer_pairs;
  bool detect_container_overflow;
  int detect_odr_violation;
};

extern Flags asan_flags_dont_use_directly;
inline Flags *flags() {
  return &asan_flags_dont_use_directly;
}
void InitializeFlags(Flags *f, const char *env);

}  // namespace __asan

#endif  // ASAN_FLAGS_H
