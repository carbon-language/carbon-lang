//===-- tsan_trace.h --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
//===----------------------------------------------------------------------===//
#ifndef TSAN_TRACE_H
#define TSAN_TRACE_H

#include "tsan_defs.h"
#include "tsan_mutex.h"
#include "tsan_sync.h"

namespace __tsan {

const int kTraceParts = 8;
const int kTraceSize = 128*1024;
const int kTracePartSize = kTraceSize / kTraceParts;

// Must fit into 3 bits.
enum EventType {
  EventTypeMop,
  EventTypeFuncEnter,
  EventTypeFuncExit,
  EventTypeLock,
  EventTypeUnlock,
  EventTypeRLock,
  EventTypeRUnlock,
};

// Represents a thread event (from most significant bit):
// u64 typ  : 3;   // EventType.
// u64 addr : 61;  // Associated pc.
typedef u64 Event;

struct TraceHeader {
  StackTrace stack0;  // Start stack for the trace.
  u64        epoch0;  // Start epoch for the trace.
#ifndef TSAN_GO
  uptr       stack0buf[kShadowStackSize];
#endif

  TraceHeader()
#ifndef TSAN_GO
      : stack0(stack0buf, kShadowStackSize)
#else
      : stack0()
#endif
      , epoch0() {
  }
};

struct Trace {
  Event events[kTraceSize];
  TraceHeader headers[kTraceParts];
  Mutex mtx;

  Trace()
    : mtx(MutexTypeTrace, StatMtxTrace) {
  }
};

}  // namespace __tsan

#endif  // TSAN_TRACE_H
