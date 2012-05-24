//===-- interception_linux.cc -----------------------------------*- C++ -*-===//
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

#include <stddef.h>  // for NULL
#include <dlfcn.h>   // for dlsym

namespace __interception {
bool GetRealFunctionAddress(const char *func_name, void **func_addr,
    void *real, void *wrapper) {
  *func_addr = dlsym(RTLD_NEXT, func_name);
  return real == wrapper;
}
}  // namespace __interception


#endif  // __linux__
