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
// Windows-specific interception methods.
//===----------------------------------------------------------------------===//

#ifdef _WIN32

#include <windows.h>

namespace __interception {

bool GetRealFunctionAddress(const char *func_name, void **func_addr) {
  const char *DLLS[] = {
    "msvcr80.dll",
    "msvcr90.dll",
    "kernel32.dll",
    NULL
  };
  *func_addr = NULL;
  for (size_t i = 0; *func_addr == NULL && DLLS[i]; ++i) {
    *func_addr = GetProcAddress(GetModuleHandleA(DLLS[i]), func_name);
  }
  return (*func_addr != NULL);
}
}  // namespace __interception

#endif  // _WIN32
