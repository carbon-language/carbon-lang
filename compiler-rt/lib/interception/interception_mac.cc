//===-- interception_mac.cc -------------------------------------*- C++ -*-===//
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
// Mac-specific interception methods.
//===----------------------------------------------------------------------===//

#ifdef __APPLE__

#define INCLUDED_FROM_INTERCEPTION_LIB
#include "interception_mac.h"
#undef INCLUDED_FROM_INTERCEPTION_LIB
#include "mach_override/mach_override.h"

namespace __interception {
bool OverrideFunction(void *old_func, void *new_func, void **orig_old_func) {
  *orig_old_func = NULL;
  int res = __asan_mach_override_ptr_custom(old_func, new_func,
                                            orig_old_func,
                                            __interception_allocate_island,
                                            __interception_deallocate_island);
  return (res == 0) && (*orig_old_func != NULL);
}
}  // namespace __interception

#endif  // __APPLE__
