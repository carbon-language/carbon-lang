//===- Win32/Win32.h - Common Win32 Include File ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Jeff Cohen and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines things specific to Win32 implementations.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only generic Win32 code that
//===          is guaranteed to work on *all* Win32 variants.
//===----------------------------------------------------------------------===//

// Require at least Windows 2000 API.
#define _WIN32_WINNT 0x0500

#include "llvm/Config/config.h"     // Get autoconf configuration settings
#include "windows.h"
#include <cassert>
#include <string>

inline bool GetError(const std::string &Prefix, std::string *Dest) {
  if (Dest == 0) return;
  char *buffer = NULL;
  FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER|FORMAT_MESSAGE_FROM_SYSTEM,
      NULL, GetLastError(), 0, (LPSTR)&buffer, 1, NULL);
  *Dest = Prefix + buffer;
  LocalFree(buffer);
  return true;
}

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
