//===- Win32/TimeValue.cpp - Win32 TimeValue Implementation -----*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Jeff Cohen and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file provides the Win32 implementation of the TimeValue class.
//
//===----------------------------------------------------------------------===//

#include "Win32.h"
#include <time.h>

namespace llvm {
using namespace sys;

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only Win32 specific code.
//===----------------------------------------------------------------------===//

TimeValue TimeValue::now() {
  uint64_t ft;
  GetSystemTimeAsFileTime(reinterpret_cast<FILETIME *>(&ft));

  TimeValue t(0, 0);
  t.fromWin32Time(ft);
  return t;
}

std::string TimeValue::toString() const {
  // Alas, asctime is not re-entrant on Windows...

  __time64_t ourTime = this->toEpochTime();
  char* buffer = ::asctime(::_localtime64(&ourTime));

  std::string result(buffer);
  return result.substr(0,24);
}

// vim: sw=2 smartindent smarttab tw=80 autoindent expandtab

}
