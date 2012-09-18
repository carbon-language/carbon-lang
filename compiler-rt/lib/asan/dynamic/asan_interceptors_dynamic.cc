//===-- asan_interceptors_dynamic.cc --------------------------------------===//
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
// __DATA,__interpose section of the dynamic runtime library for Mac OS.
//===----------------------------------------------------------------------===//

#if defined(__APPLE__)

#include "../asan_interceptors.h"
#include "../asan_intercepted_functions.h"

namespace __asan {

#if !MAC_INTERPOSE_FUNCTIONS
# error \
  Dynamic interposing library should be built with -DMAC_INTERPOSE_FUNCTIONS
#endif

#define INTERPOSE_FUNCTION(function) \
    { reinterpret_cast<const uptr>(WRAP(function)), \
      reinterpret_cast<const uptr>(function) }

#define INTERPOSE_FUNCTION_2(function, wrapper) \
    { reinterpret_cast<const uptr>(wrapper), \
      reinterpret_cast<const uptr>(function) }

struct interpose_substitution {
  const uptr replacement;
  const uptr original;
};

__attribute__((used))
const interpose_substitution substitutions[]
    __attribute__((section("__DATA, __interpose"))) = {
  INTERPOSE_FUNCTION(strlen),
  INTERPOSE_FUNCTION(memcmp),
  INTERPOSE_FUNCTION(memcpy),
  INTERPOSE_FUNCTION(memmove),
  INTERPOSE_FUNCTION(memset),
  INTERPOSE_FUNCTION(strchr),
  INTERPOSE_FUNCTION(strcat),
  INTERPOSE_FUNCTION(strncat),
  INTERPOSE_FUNCTION(strcpy),
  INTERPOSE_FUNCTION(strncpy),
  INTERPOSE_FUNCTION(pthread_create),
  INTERPOSE_FUNCTION(longjmp),
#if ASAN_INTERCEPT__LONGJMP
  INTERPOSE_FUNCTION(_longjmp),
#endif
#if ASAN_INTERCEPT_SIGLONGJMP
  INTERPOSE_FUNCTION(siglongjmp),
#endif
#if ASAN_INTERCEPT_STRDUP
  INTERPOSE_FUNCTION(strdup),
#endif
#if ASAN_INTERCEPT_STRNLEN
  INTERPOSE_FUNCTION(strnlen),
#endif
#if ASAN_INTERCEPT_INDEX
  INTERPOSE_FUNCTION_2(index, WRAP(strchr)),
#endif
  INTERPOSE_FUNCTION(strcmp),
  INTERPOSE_FUNCTION(strncmp),
#if ASAN_INTERCEPT_STRCASECMP_AND_STRNCASECMP
  INTERPOSE_FUNCTION(strcasecmp),
  INTERPOSE_FUNCTION(strncasecmp),
#endif
  INTERPOSE_FUNCTION(atoi),
  INTERPOSE_FUNCTION(atol),
  INTERPOSE_FUNCTION(strtol),
#if ASAN_INTERCEPT_ATOLL_AND_STRTOLL
  INTERPOSE_FUNCTION(atoll),
  INTERPOSE_FUNCTION(strtoll),
#endif
#if ASAN_INTERCEPT_MLOCKX
  INTERPOSE_FUNCTION(mlock),
  INTERPOSE_FUNCTION(munlock),
  INTERPOSE_FUNCTION(mlockall),
  INTERPOSE_FUNCTION(munlockall),
#endif
  INTERPOSE_FUNCTION(dispatch_async_f),
  INTERPOSE_FUNCTION(dispatch_sync_f),
  INTERPOSE_FUNCTION(dispatch_after_f),
  INTERPOSE_FUNCTION(dispatch_barrier_async_f),
  INTERPOSE_FUNCTION(dispatch_group_async_f),

  INTERPOSE_FUNCTION(dispatch_group_async),
  INTERPOSE_FUNCTION(dispatch_async),
  INTERPOSE_FUNCTION(dispatch_after),
  INTERPOSE_FUNCTION(dispatch_source_set_event_handler),
  INTERPOSE_FUNCTION(dispatch_source_set_cancel_handler),

  INTERPOSE_FUNCTION(signal),
  INTERPOSE_FUNCTION(sigaction),

  INTERPOSE_FUNCTION(__CFInitialize),
  INTERPOSE_FUNCTION(CFStringCreateCopy),
  INTERPOSE_FUNCTION(free),
};

}  // namespace __asan

#endif  // __APPLE__
