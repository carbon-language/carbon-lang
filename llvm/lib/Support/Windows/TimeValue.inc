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
#include <cctype>
#include <time.h>

using namespace llvm;
using namespace llvm::sys;

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
  struct tm *LT;
#ifdef __MINGW32__
  // Old versions of mingw don't have _localtime64_s. Remove this once we drop support
  // for them.
  time_t OurTime = time_t(this->toEpochTime());
  LT = ::localtime(&OurTime);
  assert(LT);
#else
  struct tm Storage;
  __time64_t OurTime = this->toEpochTime();
  int Error = ::_localtime64_s(&Storage, &OurTime);
  assert(!Error);
  LT = &Storage;
#endif

  char Buffer[25];
  // FIXME: the windows version of strftime doesn't support %e
  strftime(Buffer, 25, "%b %d %H:%M %Y", LT);
  assert((Buffer[3] == ' ' && isdigit(Buffer[5]) && Buffer[6] == ' ') &&
         "Unexpected format in strftime()!");
  // Emulate %e on %d to mute '0'.
  if (Buffer[4] == '0')
    Buffer[4] = ' ';
  return std::string(Buffer);
}
