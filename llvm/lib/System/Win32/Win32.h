//===- Win32/Win32.h - Common Win32 Include File ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines things specific to Unix implementations.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only generic UNIX code that
//===          is guaranteed to work on all UNIX variants.
//===----------------------------------------------------------------------===//

#include "llvm/Config/config.h"     // Get autoconf configuration settings
#include "windows.h"
#include <cassert>
#include <string>

inline void ThrowError(const std::string& msg) {
  char *buffer = NULL;
  FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER|FORMAT_MESSAGE_FROM_SYSTEM,
      NULL, GetLastError(), 0, (LPSTR)&buffer, 1, NULL);
  std::string s(msg);
  s += buffer;
  LocalFree(buffer);
  throw s;
}

inline void ThrowErrno(const std::string& prefix) {
    ThrowError(prefix + ": " + strerror(errno));
}
