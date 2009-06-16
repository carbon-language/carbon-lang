//= llvm/System/Win32/Mutex.inc - Win32 Reader/Writer Mutual Exclusion Lock  =//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements the Win32 specific (non-pthread) RWMutex class.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only generic Win32 code that
//===          is guaranteed to work on *all* Win32 variants.
//===----------------------------------------------------------------------===//

#include "Win32.h"

// FIXME: THIS IS NOT THREAD-SAFE!!
// Windows does not have reader-writer locks pre-Vista.  If you want to have
// thread-safe LLVM on Windows, for now at least, you need to use a pthreads
// replacement library.

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
