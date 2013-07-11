//===- Win32/TimeValue.cpp - Win32 TimeValue Implementation -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides the Win32 implementation of the TimeValue class.
//
//===----------------------------------------------------------------------===//

#include "Windows.h"
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

std::string TimeValue::str() const {
  struct tm LT;
  __time64_t OurTime = this->toEpochTime();
  errno_t Error = ::_localtime64_s(&LT, &OurTime);
  assert(!Error);

  char Buffer[25];
  // FIXME: the windows version of strftime doesn't support %e
  strftime(Buffer, 25, "%b %d %H:%M %Y", &LT);
  return std::string(Buffer);
}


}
