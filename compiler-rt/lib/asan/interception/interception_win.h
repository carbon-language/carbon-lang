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
// Windows-specific interception methods.
//===----------------------------------------------------------------------===//

#ifdef _WIN32

#if !defined(INCLUDED_FROM_INTERCEPTION_LIB)
# error "interception_win.h should be included from interception library only"
#endif

#ifndef INTERCEPTION_WIN_H
#define INTERCEPTION_WIN_H

namespace __interception {
// returns true if a function with the given name was found.
bool GetRealFunctionAddress(const char *func_name, void **func_addr);
}  // namespace __interception

#define INTERCEPT_FUNCTION_WIN(func) \
    ::__interception::GetRealFunctionAddress(#func, (void**)&REAL(func))

#endif  // INTERCEPTION_WIN_H
#endif  // _WIN32
