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

namespace llvm {
using namespace sys;

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only Win32 specific code.
//===----------------------------------------------------------------------===//

TimeValue TimeValue::now() {
  __int64 ft;
  GetSystemTimeAsFileTime(reinterpret_cast<FILETIME *>(&ft));

  return TimeValue(
    static_cast<TimeValue::SecondsType>( ft / 10000000 +
      Win32ZeroTime.seconds_ ),
    static_cast<TimeValue::NanoSecondsType>( (ft % 10000000) * 100) );
}

std::string TimeValue::toString() const {
  return "Don't know how to conver time on Win32";
}

// vim: sw=2 smartindent smarttab tw=80 autoindent expandtab

}
