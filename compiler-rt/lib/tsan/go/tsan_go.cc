//===-- tsan_go.cc --------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// ThreadSanitizer runtime for Go language.
//
//===----------------------------------------------------------------------===//

#include "tsan_rtl.h"
#include "tsan_symbolize.h"
#include "sanitizer_common/sanitizer_common.h"
#include <stdlib.h>

namespace __tsan {

struct ThreadStatePlaceholder {
  uptr opaque[sizeof(ThreadState) / sizeof(uptr) + kCacheLineSize];
};

static ThreadStatePlaceholder *threads;

void InitializeInterceptors() {
}

void InitializeDynamicAnnotations() {
}

bool IsExpectedReport(uptr addr, uptr size) {
  return false;
}

void internal_start_thread(void(*func)(void*), void *arg) {
}

ReportStack *SymbolizeCodeAddr2Line(uptr addr) {
  return NewReportStackEntry(addr);
}

ReportStack *SymbolizeDataAddr2Line(uptr addr) {
  return 0;
}

void *internal_alloc(MBlockType typ, uptr sz) {
  return InternalAlloc(sz);
}

void internal_free(void *p) {
  InternalFree(p);
}

extern "C" {

enum Tsan1EventType {
	NOOP,               // Should not appear.
	READ,               // {tid, pc, addr, size}
	WRITE,              // {tid, pc, addr, size}
	READER_LOCK,        // {tid, pc, lock, 0}
	WRITER_LOCK,        // {tid, pc, lock, 0}
	UNLOCK,             // {tid, pc, lock, 0}
	UNLOCK_OR_INIT,     // {tid, pc, lock, 0}
	LOCK_CREATE,        // {tid, pc, lock, 0}
	LOCK_DESTROY,       // {tid, pc, lock, 0}
	THR_CREATE_BEFORE,  // Parent thread's event. {tid, pc, 0, 0}
	THR_CREATE_AFTER,   // Parent thread's event. {tid, 0, 0, child_tid}/* 10 */
	THR_START,          // Child thread's event {tid, CallStack, 0, parent_tid}
	THR_FIRST_INSN,     // Used only by valgrind.
	THR_END,            // {tid, 0, 0, 0}
	THR_JOIN_AFTER,     // {tid, pc, joined_tid}
	THR_STACK_TOP,      // {tid, pc, stack_top, stack_size_if_known}
	RTN_EXIT,           // {tid, 0, 0, 0}
	RTN_CALL,           // {tid, pc, 0, 0}
	SBLOCK_ENTER,       // {tid, pc, 0, 0}
	SIGNAL,             // {tid, pc, obj, 0}
	WAIT,               // {tid, pc, obj, 0} /* 20 */
	CYCLIC_BARRIER_INIT,         // {tid, pc, obj, n}
	CYCLIC_BARRIER_WAIT_BEFORE,  // {tid, pc, obj, 0}
	CYCLIC_BARRIER_WAIT_AFTER,   // {tid, pc, obj, 0}
	PCQ_CREATE,         // {tid, pc, pcq_addr, 0}
	PCQ_DESTROY,        // {tid, pc, pcq_addr, 0}
	PCQ_PUT,            // {tid, pc, pcq_addr, 0}
	PCQ_GET,            // {tid, pc, pcq_addr, 0}
	STACK_MEM_DIE,      // deprecated.
	MALLOC,             // {tid, pc, addr, size}
	FREE,               // {tid, pc, addr, 0} /* 30 */
	MMAP,               // {tid, pc, addr, size}
	MUNMAP,             // {tid, pc, addr, size}
	PUBLISH_RANGE,      // may be deprecated later.
	UNPUBLISH_RANGE,    // deprecated. TODO(kcc): get rid of this.
	HB_LOCK,            // {tid, pc, addr, 0}
	NON_HB_LOCK,        // {tid, pc, addr, 0}
	IGNORE_READS_BEG,   // {tid, pc, 0, 0}
	IGNORE_READS_END,   // {tid, pc, 0, 0}
	IGNORE_WRITES_BEG,  // {tid, pc, 0, 0}
	IGNORE_WRITES_END,  // {tid, pc, 0, 0}
	SET_THREAD_NAME,    // {tid, pc, name_str, 0}
	SET_LOCK_NAME,      // {tid, pc, lock, lock_name_str}
	TRACE_MEM,          // {tid, pc, addr, 0}
	EXPECT_RACE,        // {tid, descr_str, ptr, size}
	BENIGN_RACE,        // {tid, descr_str, ptr, size}
	EXPECT_RACE_BEGIN,  // {tid, pc, 0, 0}
	EXPECT_RACE_END,    // {tid, pc, 0, 0}
	VERBOSITY,          // Used for debugging.
	STACK_TRACE,        // {tid, pc, 0, 0}, for debugging.
	FLUSH_STATE,        // {tid, pc, 0, 0}
	PC_DESCRIPTION,     // {0, pc, descr_str, 0}, for ts_offline.
	PRINT_MESSAGE,      // {tid, pc, message_str, 0}, for ts_offline.
	FLUSH_EXPECTED_RACES,  // {0, 0, 0, 0}
	LAST_EVENT          // Should not appear.
};

void __tsan_init() {
  threads = (ThreadStatePlaceholder*)internal_alloc(MBlockThreadContex,
      kMaxTid * sizeof(ThreadStatePlaceholder));
  //!!! internal_memset(threads, 0, kMaxTid * sizeof(ThreadStatePlaceholder));
  ThreadState *thr = (ThreadState*)&threads[0];
  thr->in_rtl++;
  Initialize(thr);
  thr->in_rtl--;
}

void __tsan_fini() {
  // FIXME: Not necessary thread 0.
  ThreadState *thr = (ThreadState*)&threads[0];
  thr->in_rtl++;
  int res = Finalize(thr);
  thr->in_rtl--;
  exit(res);  
}

void __tsan_event(int typ, int tid, void *pc, void *addr, int info) {
  //if (typ != READ && typ != WRITE && typ != SBLOCK_ENTER)
  //  Printf("typ=%d tid=%d pc=%p addr=%p info=%d\n", typ, tid, pc, addr, info);
  ThreadState *thr = (ThreadState*)&threads[tid];
  switch (typ) {
  case READ:
    MemoryAccess(thr, (uptr)pc, (uptr)addr, 0, false);
    break;
  case WRITE:
    MemoryAccess(thr, (uptr)pc, (uptr)addr, 0, true);
    break;
  case RTN_EXIT:
    FuncExit(thr);
    break;
  case RTN_CALL:
    FuncEntry(thr, (uptr)pc);
    break;
  case SBLOCK_ENTER:
    break;
  case SIGNAL:
    thr->in_rtl++;
    Release(thr, (uptr)pc, (uptr)addr);
    thr->in_rtl--;
    break;
  case WAIT:
    thr->in_rtl++;
    Acquire(thr, (uptr)pc, (uptr)addr);
    thr->in_rtl--;
    break;
  case MALLOC:
    thr->in_rtl++;
    MemoryResetRange(thr, (uptr)pc, (uptr)addr, (uptr)info);
    thr->in_rtl--;
    break;
  case FREE:
    break;
  case THR_START: {
    //Printf("typ=%d tid=%d pc=%p addr=%p info=%d\n", typ, tid, pc, addr, info);
    if (tid == 0)
      return;
    ThreadState *parent = (ThreadState*)&threads[info];
    thr->in_rtl++;
    parent->in_rtl++;
    int tid2 = ThreadCreate(parent, (uptr)pc, 0, true);
    CHECK_EQ(tid2, tid);
    ThreadStart(thr, tid2);
    parent->in_rtl--;
    thr->in_rtl--;
    break;
  }
  default:
    thr->in_rtl++;
    Printf("Event: typ=%d thr=%d\n", typ, tid);
    thr->in_rtl--;
  }
}

}  // extern "C"
}  // namespace __tsan
