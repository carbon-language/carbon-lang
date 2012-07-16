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

static ThreadState *goroutines[kMaxTid];

void InitializeInterceptors() {
}

void InitializeDynamicAnnotations() {
}

bool IsExpectedReport(uptr addr, uptr size) {
  return false;
}

void internal_start_thread(void(*func)(void*), void *arg) {
}

extern "C" int goCallbackCommentPc(uptr pc, char **img, char **rtn,
                                   char **filename, int *lineno);
extern "C" void free(void *p);

ReportStack *SymbolizeCode(uptr addr) {
  ReportStack *s = NewReportStackEntry(addr);
  char *img, *rtn, *filename;
  int lineno;
  if (goCallbackCommentPc(addr, &img, &rtn, &filename, &lineno)) {
    s->module = internal_strdup(img);
    s->offset = addr;
    s->func = internal_strdup(rtn);
    s->file = internal_strdup(filename);
    s->line = lineno;
    s->col = 0;
    free(img);
    free(rtn);
    free(filename);
  }
  return s;
}

ReportStack *SymbolizeData(uptr addr) {
  return 0;
}

ReportStack *NewReportStackEntry(uptr addr) {
  ReportStack *ent = (ReportStack*)internal_alloc(MBlockReportStack,
                                                  sizeof(ReportStack));
  internal_memset(ent, 0, sizeof(*ent));
  ent->pc = addr;
  return ent;
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

static void AllocGoroutine(int tid) {
  goroutines[tid] = (ThreadState*)internal_alloc(MBlockThreadContex,
      sizeof(ThreadState));
  internal_memset(goroutines[tid], 0, sizeof(ThreadState));
}

void __tsan_init() {
  AllocGoroutine(0);
  ThreadState *thr = goroutines[0];
  thr->in_rtl++;
  Initialize(thr);
  thr->in_rtl--;
}

void __tsan_fini() {
  // FIXME: Not necessary thread 0.
  ThreadState *thr = goroutines[0];
  thr->in_rtl++;
  int res = Finalize(thr);
  thr->in_rtl--;
  exit(res);  
}

void __tsan_event(int typ, int tid, void *pc, void *addr, int info) {
  ThreadState *thr = goroutines[tid];
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
    MemoryAccessRange(thr, (uptr)pc, (uptr)addr, (uptr)info, true);
    thr->in_rtl--;
    break;
  case FREE:
    break;
  case THR_START: {
    if (tid == 0)
      return;
    ThreadState *parent = goroutines[info];
    AllocGoroutine(tid);
    thr = goroutines[tid];
    thr->in_rtl++;
    parent->in_rtl++;
    int tid2 = ThreadCreate(parent, (uptr)pc, 0, true);
    ThreadStart(thr, tid2);
    parent->in_rtl--;
    thr->in_rtl--;
    break;
  }
  case THR_END: {
    thr->in_rtl++;
    ThreadFinish(thr);
    thr->in_rtl--;
    break;
  }
  default:
    Printf("Unknown event type %d\n", typ);
    Die();
  }
}

}  // extern "C"
}  // namespace __tsan
