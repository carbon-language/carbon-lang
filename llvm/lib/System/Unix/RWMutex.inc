//= llvm/System/Unix/RWMutex.inc - Unix Reader/Writer Mutual Exclusion Lock  =//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements the Unix specific (non-pthread) RWMutex class.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only generic UNIX code that
//===          is guaranteed to work on *all* UNIX variants.
//===----------------------------------------------------------------------===//

namespace llvm {

using namespace sys;

RWMutex::RWMutex() { }

RWMutex::~RWMutex() { }

bool RWMutex::reader_acquire() {
  return true;
}

bool RWMutex::reader_release() {
  return true;
}

bool RWMutex::writer_acquire() {
  return true;
}

bool RWMutex::writer_release() {
  return true;
}

}
