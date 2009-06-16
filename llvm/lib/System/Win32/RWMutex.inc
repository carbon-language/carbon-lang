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

namespace llvm {
using namespace sys;

RWMutex::RWMutex() {
  data_ = new PSRWLOCK;
  InitializeSRWLock((PSRWLOCK*)data_);
}

RWMutex::~RWMutex() {
  delete (PSRWLOCK*)data_;
  data_ = 0;
}

bool RWMutex::reader_acquire() {
  AcquireSRWLockShared((PSRWLOCK*)data_);
  return true;
}

bool RWMutex::reader_release() {
  ReleaseSRWLockShared((PSRWLOCK*)data_);
  return true;
}

bool RWMutex::writer_acquire() {
  AcquireSRWLockExclusive((PSRWLOCK*)data_);
  return true;
}

bool RWMutex::writer_release() {
  ReleaseSRWLockExclusive((PSRWLOCK*)data_);
  return true;
}


}
