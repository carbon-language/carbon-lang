//===-- sanitizer_deadlock_detector_interface.h -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of Sanitizer runtime.
// Abstract deadlock detector interface.
// FIXME: this is work in progress, nothing really works yet.
//
//===----------------------------------------------------------------------===//

#ifndef SANITIZER_DEADLOCK_DETECTOR_INTERFACE_H
#define SANITIZER_DEADLOCK_DETECTOR_INTERFACE_H

#include "sanitizer_internal_defs.h"

namespace __sanitizer {

// dd - deadlock detector.
// lt - logical (user) thread.
// pt - physical (OS) thread.

struct DDMutex {
  uptr id;
  u32  stk;  // creation (or any other) stack that indentifies the mutex
  u64  ctx;  // user context
};

struct DDReport {
  int n;  // number of entries in loop
  struct {
    u64 thr_ctx;   // user thread context
    u64 mtx_ctx0;  // user mutex context, start of the edge
    u64 mtx_ctx1;  // user mutex context, end of the edge
    u32 stk;       // stack id for the edge
  } loop[16];
};

struct DDPhysicalThread;
struct DDLogicalThread;

struct DDetector {
  static DDetector *Create();

  virtual DDPhysicalThread* CreatePhysicalThread() { return 0; }
  virtual void DestroyPhysicalThread(DDPhysicalThread *pt) {}

  virtual DDLogicalThread* CreateLogicalThread(u64 ctx) { return 0; }
  virtual void DestroyLogicalThread(DDLogicalThread *lt) {}

  virtual void MutexInit(DDMutex *m, u32 stk, u64 ctx) {}
  virtual DDReport *MutexLock(DDPhysicalThread *pt, DDLogicalThread *lt,
      DDMutex *m, bool writelock, bool trylock) { return 0; }
  virtual DDReport *MutexUnlock(DDPhysicalThread *pt, DDLogicalThread *lt,
      DDMutex *m, bool writelock) { return 0; }
  virtual void MutexDestroy(DDPhysicalThread *pt, DDLogicalThread *lt,
      DDMutex *m) {}
};

} // namespace __sanitizer

#endif // SANITIZER_DEADLOCK_DETECTOR_INTERFACE_H
