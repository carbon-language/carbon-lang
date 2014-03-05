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

#ifndef SANITIZER_DEADLOCK_DETECTOR_VERSION
# define SANITIZER_DEADLOCK_DETECTOR_VERSION 1
#endif

#include "sanitizer_internal_defs.h"
#include "sanitizer_atomic.h"

namespace __sanitizer {

// dd - deadlock detector.
// lt - logical (user) thread.
// pt - physical (OS) thread.

struct DDPhysicalThread;
struct DDLogicalThread;

struct DDMutex {
#if SANITIZER_DEADLOCK_DETECTOR_VERSION == 1
  uptr id;
  u32  stk;  // creation stack
  u64  ctx;
#elif SANITIZER_DEADLOCK_DETECTOR_VERSION == 2
  u32              id;
  u32              recursion;
  atomic_uintptr_t owner;
#else
# error "BAD SANITIZER_DEADLOCK_DETECTOR_VERSION"
#endif
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

struct DDCallback {
  DDPhysicalThread *pt;
  DDLogicalThread  *lt;

  virtual u32 Unwind() { return 0; }
};

struct DDetector {
  static DDetector *Create();

  virtual DDPhysicalThread* CreatePhysicalThread() { return 0; }
  virtual void DestroyPhysicalThread(DDPhysicalThread *pt) {}

  virtual DDLogicalThread* CreateLogicalThread(u64 ctx) { return 0; }
  virtual void DestroyLogicalThread(DDLogicalThread *lt) {}

  virtual void MutexInit(DDCallback *cb, DDMutex *m) {}
  virtual void MutexBeforeLock(DDCallback *cb, DDMutex *m, bool wlock) {}
  virtual void MutexAfterLock(DDCallback *cb, DDMutex *m, bool wlock,
      bool trylock) {}
  virtual void MutexBeforeUnlock(DDCallback *cb, DDMutex *m, bool wlock) {}
  virtual void MutexDestroy(DDCallback *cb, DDMutex *m) {}

  virtual DDReport *GetReport(DDCallback *cb) { return 0; }
};

} // namespace __sanitizer

#endif // SANITIZER_DEADLOCK_DETECTOR_INTERFACE_H
