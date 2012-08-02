//===-- interception_linux.h ------------------------------------*- C++ -*-===//
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
// Linux-specific interception methods.
//===----------------------------------------------------------------------===//

#ifdef __linux__

#if !defined(INCLUDED_FROM_INTERCEPTION_LIB)
# error "interception_linux.h should be included from interception library only"
#endif

#ifndef INTERCEPTION_LINUX_H
#define INTERCEPTION_LINUX_H

namespace __interception {
// returns true if a function with the given name was found.
bool GetRealFunctionAddress(const char *func_name, uptr *func_addr,
    uptr real, uptr wrapper);
}  // namespace __interception

#define INTERCEPT_FUNCTION_LINUX(func) \
    ::__interception::GetRealFunctionAddress( \
          #func, (::__interception::uptr*)&REAL(func), \
          (::__interception::uptr)&(func), \
          (::__interception::uptr)&WRAP(func))

#endif  // INTERCEPTION_LINUX_H
#endif  // __linux__
