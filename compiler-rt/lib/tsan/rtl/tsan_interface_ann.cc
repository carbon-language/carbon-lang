//===-- tsan_interface_ann.cc ---------------------------------------------===//
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
#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_internal_defs.h"
#include "sanitizer_common/sanitizer_placement_new.h"
#include "sanitizer_common/sanitizer_stacktrace.h"
#include "tsan_interface_ann.h"
#include "tsan_mutex.h"
#include "tsan_report.h"
#include "tsan_rtl.h"
#include "tsan_mman.h"
#include "tsan_flags.h"
#include "tsan_platform.h"
#include "tsan_vector.h"

#define CALLERPC ((uptr)__builtin_return_address(0))

using namespace __tsan;  // NOLINT

namespace __tsan {

class ScopedAnnotation {
 public:
  ScopedAnnotation(ThreadState *thr, const char *aname, const char *f, int l,
                   uptr pc)
      : thr_(thr) {
    FuncEntry(thr_, pc);
    DPrintf("#%d: annotation %s() %s:%d\n", thr_->tid, aname, f, l);
  }

  ~ScopedAnnotation() {
    FuncExit(thr_);
  }
 private:
  ThreadState *const thr_;
};

#define SCOPED_ANNOTATION(typ) \
    if (!flags()->enable_annotations) \
      return; \
    ThreadState *thr = cur_thread(); \
    const uptr caller_pc = (uptr)__builtin_return_address(0); \
    StatInc(thr, StatAnnotation); \
    StatInc(thr, Stat##typ); \
    ScopedAnnotation sa(thr, __func__, f, l, caller_pc); \
    const uptr pc = __sanitizer::StackTrace::GetCurrentPc(); \
    (void)pc; \
/**/

static const int kMaxDescLen = 128;

struct ExpectRace {
  ExpectRace *next;
  ExpectRace *prev;
  int hitcount;
  int addcount;
  uptr addr;
  uptr size;
  char *file;
  int line;
  char desc[kMaxDescLen];
};

struct DynamicAnnContext {
  Mutex mtx;
  ExpectRace expect;
  ExpectRace benign;

  DynamicAnnContext()
    : mtx(MutexTypeAnnotations, StatMtxAnnotations) {
  }
};

static DynamicAnnContext *dyn_ann_ctx;
static char dyn_ann_ctx_placeholder[sizeof(DynamicAnnContext)] ALIGNED(64);

static void AddExpectRace(ExpectRace *list,
    char *f, int l, uptr addr, uptr size, char *desc) {
  ExpectRace *race = list->next;
  for (; race != list; race = race->next) {
    if (race->addr == addr && race->size == size) {
      race->addcount++;
      return;
    }
  }
  race = (ExpectRace*)internal_alloc(MBlockExpectRace, sizeof(ExpectRace));
  race->addr = addr;
  race->size = size;
  race->file = f;
  race->line = l;
  race->desc[0] = 0;
  race->hitcount = 0;
  race->addcount = 1;
  if (desc) {
    int i = 0;
    for (; i < kMaxDescLen - 1 && desc[i]; i++)
      race->desc[i] = desc[i];
    race->desc[i] = 0;
  }
  race->prev = list;
  race->next = list->next;
  race->next->prev = race;
  list->next = race;
}

static ExpectRace *FindRace(ExpectRace *list, uptr addr, uptr size) {
  for (ExpectRace *race = list->next; race != list; race = race->next) {
    uptr maxbegin = max(race->addr, addr);
    uptr minend = min(race->addr + race->size, addr + size);
    if (maxbegin < minend)
      return race;
  }
  return 0;
}

static bool CheckContains(ExpectRace *list, uptr addr, uptr size) {
  ExpectRace *race = FindRace(list, addr, size);
  if (race == 0 && AlternativeAddress(addr))
    race = FindRace(list, AlternativeAddress(addr), size);
  if (race == 0)
    return false;
  DPrintf("Hit expected/benign race: %s addr=%zx:%d %s:%d\n",
      race->desc, race->addr, (int)race->size, race->file, race->line);
  race->hitcount++;
  return true;
}

static void InitList(ExpectRace *list) {
  list->next = list;
  list->prev = list;
}

void InitializeDynamicAnnotations() {
  dyn_ann_ctx = new(dyn_ann_ctx_placeholder) DynamicAnnContext;
  InitList(&dyn_ann_ctx->expect);
  InitList(&dyn_ann_ctx->benign);
}

bool IsExpectedReport(uptr addr, uptr size) {
  Lock lock(&dyn_ann_ctx->mtx);
  if (CheckContains(&dyn_ann_ctx->expect, addr, size))
    return true;
  if (CheckContains(&dyn_ann_ctx->benign, addr, size))
    return true;
  return false;
}

static void CollectMatchedBenignRaces(Vector<ExpectRace> *matched,
    int *unique_count, int *hit_count, int ExpectRace::*counter) {
  ExpectRace *list = &dyn_ann_ctx->benign;
  for (ExpectRace *race = list->next; race != list; race = race->next) {
    (*unique_count)++;
    if (race->*counter == 0)
      continue;
    (*hit_count) += race->*counter;
    uptr i = 0;
    for (; i < matched->Size(); i++) {
      ExpectRace *race0 = &(*matched)[i];
      if (race->line == race0->line
          && internal_strcmp(race->file, race0->file) == 0
          && internal_strcmp(race->desc, race0->desc) == 0) {
        race0->*counter += race->*counter;
        break;
      }
    }
    if (i == matched->Size())
      matched->PushBack(*race);
  }
}

void PrintMatchedBenignRaces() {
  Lock lock(&dyn_ann_ctx->mtx);
  int unique_count = 0;
  int hit_count = 0;
  int add_count = 0;
  Vector<ExpectRace> hit_matched(MBlockScopedBuf);
  CollectMatchedBenignRaces(&hit_matched, &unique_count, &hit_count,
      &ExpectRace::hitcount);
  Vector<ExpectRace> add_matched(MBlockScopedBuf);
  CollectMatchedBenignRaces(&add_matched, &unique_count, &add_count,
      &ExpectRace::addcount);
  if (hit_matched.Size()) {
    Printf("ThreadSanitizer: Matched %d \"benign\" races (pid=%d):\n",
        hit_count, (int)internal_getpid());
    for (uptr i = 0; i < hit_matched.Size(); i++) {
      Printf("%d %s:%d %s\n",
          hit_matched[i].hitcount, hit_matched[i].file,
          hit_matched[i].line, hit_matched[i].desc);
    }
  }
  if (hit_matched.Size()) {
    Printf("ThreadSanitizer: Annotated %d \"benign\" races, %d unique"
           " (pid=%d):\n",
        add_count, unique_count, (int)internal_getpid());
    for (uptr i = 0; i < add_matched.Size(); i++) {
      Printf("%d %s:%d %s\n",
          add_matched[i].addcount, add_matched[i].file,
          add_matched[i].line, add_matched[i].desc);
    }
  }
}

static void ReportMissedExpectedRace(ExpectRace *race) {
  Printf("==================\n");
  Printf("WARNING: ThreadSanitizer: missed expected data race\n");
  Printf("  %s addr=%zx %s:%d\n",
      race->desc, race->addr, race->file, race->line);
  Printf("==================\n");
}
}  // namespace __tsan

using namespace __tsan;  // NOLINT

extern "C" {
void INTERFACE_ATTRIBUTE AnnotateHappensBefore(char *f, int l, uptr addr) {
  SCOPED_ANNOTATION(AnnotateHappensBefore);
  Release(thr, pc, addr);
}

void INTERFACE_ATTRIBUTE AnnotateHappensAfter(char *f, int l, uptr addr) {
  SCOPED_ANNOTATION(AnnotateHappensAfter);
  Acquire(thr, pc, addr);
}

void INTERFACE_ATTRIBUTE AnnotateCondVarSignal(char *f, int l, uptr cv) {
  SCOPED_ANNOTATION(AnnotateCondVarSignal);
}

void INTERFACE_ATTRIBUTE AnnotateCondVarSignalAll(char *f, int l, uptr cv) {
  SCOPED_ANNOTATION(AnnotateCondVarSignalAll);
}

void INTERFACE_ATTRIBUTE AnnotateMutexIsNotPHB(char *f, int l, uptr mu) {
  SCOPED_ANNOTATION(AnnotateMutexIsNotPHB);
}

void INTERFACE_ATTRIBUTE AnnotateCondVarWait(char *f, int l, uptr cv,
                                             uptr lock) {
  SCOPED_ANNOTATION(AnnotateCondVarWait);
}

void INTERFACE_ATTRIBUTE AnnotateRWLockCreate(char *f, int l, uptr m) {
  SCOPED_ANNOTATION(AnnotateRWLockCreate);
  MutexCreate(thr, pc, m, true, true, false);
}

void INTERFACE_ATTRIBUTE AnnotateRWLockCreateStatic(char *f, int l, uptr m) {
  SCOPED_ANNOTATION(AnnotateRWLockCreateStatic);
  MutexCreate(thr, pc, m, true, true, true);
}

void INTERFACE_ATTRIBUTE AnnotateRWLockDestroy(char *f, int l, uptr m) {
  SCOPED_ANNOTATION(AnnotateRWLockDestroy);
  MutexDestroy(thr, pc, m);
}

void INTERFACE_ATTRIBUTE AnnotateRWLockAcquired(char *f, int l, uptr m,
                                                uptr is_w) {
  SCOPED_ANNOTATION(AnnotateRWLockAcquired);
  if (is_w)
    MutexLock(thr, pc, m);
  else
    MutexReadLock(thr, pc, m);
}

void INTERFACE_ATTRIBUTE AnnotateRWLockReleased(char *f, int l, uptr m,
                                                uptr is_w) {
  SCOPED_ANNOTATION(AnnotateRWLockReleased);
  if (is_w)
    MutexUnlock(thr, pc, m);
  else
    MutexReadUnlock(thr, pc, m);
}

void INTERFACE_ATTRIBUTE AnnotateTraceMemory(char *f, int l, uptr mem) {
  SCOPED_ANNOTATION(AnnotateTraceMemory);
}

void INTERFACE_ATTRIBUTE AnnotateFlushState(char *f, int l) {
  SCOPED_ANNOTATION(AnnotateFlushState);
}

void INTERFACE_ATTRIBUTE AnnotateNewMemory(char *f, int l, uptr mem,
                                           uptr size) {
  SCOPED_ANNOTATION(AnnotateNewMemory);
}

void INTERFACE_ATTRIBUTE AnnotateNoOp(char *f, int l, uptr mem) {
  SCOPED_ANNOTATION(AnnotateNoOp);
}

void INTERFACE_ATTRIBUTE AnnotateFlushExpectedRaces(char *f, int l) {
  SCOPED_ANNOTATION(AnnotateFlushExpectedRaces);
  Lock lock(&dyn_ann_ctx->mtx);
  while (dyn_ann_ctx->expect.next != &dyn_ann_ctx->expect) {
    ExpectRace *race = dyn_ann_ctx->expect.next;
    if (race->hitcount == 0) {
      CTX()->nmissed_expected++;
      ReportMissedExpectedRace(race);
    }
    race->prev->next = race->next;
    race->next->prev = race->prev;
    internal_free(race);
  }
}

void INTERFACE_ATTRIBUTE AnnotateEnableRaceDetection(
    char *f, int l, int enable) {
  SCOPED_ANNOTATION(AnnotateEnableRaceDetection);
  // FIXME: Reconsider this functionality later. It may be irrelevant.
}

void INTERFACE_ATTRIBUTE AnnotateMutexIsUsedAsCondVar(
    char *f, int l, uptr mu) {
  SCOPED_ANNOTATION(AnnotateMutexIsUsedAsCondVar);
}

void INTERFACE_ATTRIBUTE AnnotatePCQGet(
    char *f, int l, uptr pcq) {
  SCOPED_ANNOTATION(AnnotatePCQGet);
}

void INTERFACE_ATTRIBUTE AnnotatePCQPut(
    char *f, int l, uptr pcq) {
  SCOPED_ANNOTATION(AnnotatePCQPut);
}

void INTERFACE_ATTRIBUTE AnnotatePCQDestroy(
    char *f, int l, uptr pcq) {
  SCOPED_ANNOTATION(AnnotatePCQDestroy);
}

void INTERFACE_ATTRIBUTE AnnotatePCQCreate(
    char *f, int l, uptr pcq) {
  SCOPED_ANNOTATION(AnnotatePCQCreate);
}

void INTERFACE_ATTRIBUTE AnnotateExpectRace(
    char *f, int l, uptr mem, char *desc) {
  SCOPED_ANNOTATION(AnnotateExpectRace);
  Lock lock(&dyn_ann_ctx->mtx);
  AddExpectRace(&dyn_ann_ctx->expect,
                f, l, mem, 1, desc);
  DPrintf("Add expected race: %s addr=%zx %s:%d\n", desc, mem, f, l);
}

static void BenignRaceImpl(
    char *f, int l, uptr mem, uptr size, char *desc) {
  Lock lock(&dyn_ann_ctx->mtx);
  AddExpectRace(&dyn_ann_ctx->benign,
                f, l, mem, size, desc);
  DPrintf("Add benign race: %s addr=%zx %s:%d\n", desc, mem, f, l);
}

// FIXME: Turn it off later. WTF is benign race?1?? Go talk to Hans Boehm.
void INTERFACE_ATTRIBUTE AnnotateBenignRaceSized(
    char *f, int l, uptr mem, uptr size, char *desc) {
  SCOPED_ANNOTATION(AnnotateBenignRaceSized);
  BenignRaceImpl(f, l, mem, size, desc);
}

void INTERFACE_ATTRIBUTE AnnotateBenignRace(
    char *f, int l, uptr mem, char *desc) {
  SCOPED_ANNOTATION(AnnotateBenignRace);
  BenignRaceImpl(f, l, mem, 1, desc);
}

void INTERFACE_ATTRIBUTE AnnotateIgnoreReadsBegin(char *f, int l) {
  SCOPED_ANNOTATION(AnnotateIgnoreReadsBegin);
  ThreadIgnoreBegin(thr, pc);
}

void INTERFACE_ATTRIBUTE AnnotateIgnoreReadsEnd(char *f, int l) {
  SCOPED_ANNOTATION(AnnotateIgnoreReadsEnd);
  ThreadIgnoreEnd(thr, pc);
}

void INTERFACE_ATTRIBUTE AnnotateIgnoreWritesBegin(char *f, int l) {
  SCOPED_ANNOTATION(AnnotateIgnoreWritesBegin);
  ThreadIgnoreBegin(thr, pc);
}

void INTERFACE_ATTRIBUTE AnnotateIgnoreWritesEnd(char *f, int l) {
  SCOPED_ANNOTATION(AnnotateIgnoreWritesEnd);
  ThreadIgnoreEnd(thr, pc);
}

void INTERFACE_ATTRIBUTE AnnotateIgnoreSyncBegin(char *f, int l) {
  SCOPED_ANNOTATION(AnnotateIgnoreSyncBegin);
  ThreadIgnoreSyncBegin(thr, pc);
}

void INTERFACE_ATTRIBUTE AnnotateIgnoreSyncEnd(char *f, int l) {
  SCOPED_ANNOTATION(AnnotateIgnoreSyncEnd);
  ThreadIgnoreSyncEnd(thr, pc);
}

void INTERFACE_ATTRIBUTE AnnotatePublishMemoryRange(
    char *f, int l, uptr addr, uptr size) {
  SCOPED_ANNOTATION(AnnotatePublishMemoryRange);
}

void INTERFACE_ATTRIBUTE AnnotateUnpublishMemoryRange(
    char *f, int l, uptr addr, uptr size) {
  SCOPED_ANNOTATION(AnnotateUnpublishMemoryRange);
}

void INTERFACE_ATTRIBUTE AnnotateThreadName(
    char *f, int l, char *name) {
  SCOPED_ANNOTATION(AnnotateThreadName);
  ThreadSetName(thr, name);
}

// We deliberately omit the implementation of WTFAnnotateHappensBefore() and
// WTFAnnotateHappensAfter(). Those are being used by Webkit to annotate
// atomic operations, which should be handled by ThreadSanitizer correctly.
void INTERFACE_ATTRIBUTE WTFAnnotateHappensBefore(char *f, int l, uptr addr) {
  SCOPED_ANNOTATION(AnnotateHappensBefore);
}

void INTERFACE_ATTRIBUTE WTFAnnotateHappensAfter(char *f, int l, uptr addr) {
  SCOPED_ANNOTATION(AnnotateHappensAfter);
}

void INTERFACE_ATTRIBUTE WTFAnnotateBenignRaceSized(
    char *f, int l, uptr mem, uptr sz, char *desc) {
  SCOPED_ANNOTATION(AnnotateBenignRaceSized);
  BenignRaceImpl(f, l, mem, sz, desc);
}

int INTERFACE_ATTRIBUTE RunningOnValgrind() {
  return flags()->running_on_valgrind;
}

double __attribute__((weak)) INTERFACE_ATTRIBUTE ValgrindSlowdown(void) {
  return 10.0;
}

const char INTERFACE_ATTRIBUTE* ThreadSanitizerQuery(const char *query) {
  if (internal_strcmp(query, "pure_happens_before") == 0)
    return "1";
  else
    return "0";
}

void INTERFACE_ATTRIBUTE
AnnotateMemoryIsInitialized(char *f, int l, uptr mem, uptr sz) {}
}  // extern "C"
