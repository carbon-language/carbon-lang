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

#include "Windows.h"

// FIXME: Windows does not have reader-writer locks pre-Vista.  If you want
// real reader-writer locks, you a threads implementation for Windows.

namespace llvm {
using namespace sys;

RWMutexImpl::RWMutexImpl() {
  data_ = calloc(1, sizeof(CRITICAL_SECTION));
  InitializeCriticalSection(static_cast<LPCRITICAL_SECTION>(data_));
}

RWMutexImpl::~RWMutexImpl() {
  DeleteCriticalSection(static_cast<LPCRITICAL_SECTION>(data_));
  free(data_);
}

bool RWMutexImpl::reader_acquire() {
  EnterCriticalSection(static_cast<LPCRITICAL_SECTION>(data_));
  return true;
}

bool RWMutexImpl::reader_release() {
  LeaveCriticalSection(static_cast<LPCRITICAL_SECTION>(data_));
  return true;
}

bool RWMutexImpl::writer_acquire() {
  EnterCriticalSection(static_cast<LPCRITICAL_SECTION>(data_));
  return true;
}

bool RWMutexImpl::writer_release() {
  LeaveCriticalSection(static_cast<LPCRITICAL_SECTION>(data_));
  return true;
}


}
