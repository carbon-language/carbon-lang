//===-- interception_mac.h --------------------------------------*- C++ -*-===//
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

#if !defined(INCLUDED_FROM_INTERCEPTION_LIB)
# error "interception_mac.h should be included from interception.h only"
#endif

#ifndef INTERCEPTION_MAC_H
#define INTERCEPTION_MAC_H

#include <mach/mach_error.h>
#include <stddef.h>

// Allocate memory for the escape island. This cannot be moved to
// mach_override, because each user of interceptors may specify its
// own memory range for escape islands.
extern "C" {
mach_error_t __interception_allocate_island(void **ptr, size_t unused_size,
                                            void *unused_hint);
mach_error_t __interception_deallocate_island(void *ptr);
}  // extern "C"

namespace __interception {
// returns true if the old function existed.
bool OverrideFunction(uptr old_func, uptr new_func, uptr *orig_old_func);
}  // namespace __interception

# define OVERRIDE_FUNCTION_MAC(old_func, new_func) \
    ::__interception::OverrideFunction( \
          (::__interception::uptr)old_func, \
          (::__interception::uptr)new_func, \
          (::__interception::uptr*)&REAL(old_func))
# define INTERCEPT_FUNCTION_MAC(func) OVERRIDE_FUNCTION_MAC(func, WRAP(func))

#endif  // INTERCEPTION_MAC_H
#endif  // __APPLE__
