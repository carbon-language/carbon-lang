//===- Unix/TimeValue.cpp - Unix TimeValue Implementation -------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements the Unix specific portion of the TimeValue class.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only generic UNIX code that
//===          is guaranteed to work on *all* UNIX variants.
//===----------------------------------------------------------------------===//

#include "Unix.h"

#include <time.h>

namespace llvm {
  using namespace sys;


std::string TimeValue::toString() {
  char buffer[32];

  time_t ourTime = time_t(this->toEpochTime());
  ::asctime_r(::localtime(&ourTime), buffer);

  std::string result(buffer);
  return result.substr(0,24);
}

}
// vim: sw=2 smartindent smarttab tw=80 autoindent expandtab
