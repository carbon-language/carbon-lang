//===-- tsan_rtl.cc -------------------------------------------------------===//
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
// Main file (entry points) for the TSan run-time.
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_atomic.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_stackdepot.h"
#include "sanitizer_common/sanitizer_placement_new.h"
#include "sanitizer_common/sanitizer_symbolizer.h"
#include "tsan_defs.h"
#include "tsan_platform.h"
#include "tsan_rtl.h"
#include "tsan_mman.h"
#include "tsan_suppressions.h"

volatile int __tsan_resumed = 0;

extern "C" void __tsan_resume() {
  __tsan_resumed = 1;
}

namespace __tsan {

#ifndef TSAN_GO
THREADLOCAL char cur_thread_placeholder[sizeof(ThreadState)] ALIGNED(64);
#endif
static char ctx_placeholder[sizeof(Context)] ALIGNED(64);

static Context *ctx;
Context *CTX() {
  return ctx;
}

Context::Context()
  : initialized()
  , report_mtx(MutexTypeReport, StatMtxReport)
  , nreported()
  , nmissed_expected()
  , thread_mtx(MutexTypeThreads, StatMtxThreads)
  , racy_stacks(MBlockRacyStacks)
  , racy_addresses(MBlockRacyAddresses)
  , fired_suppressions(MBlockRacyAddresses) {
}

// The objects are allocated in TLS, so one may rely on zero-initialization.
ThreadState::ThreadState(Context *ctx, int tid, int unique_id, u64 epoch,
                         uptr stk_addr, uptr stk_size,
                         uptr tls_addr, uptr tls_size)
  : fast_state(tid, epoch)
  // Do not touch these, rely on zero initialization,
  // they may be accessed before the ctor.
  // , fast_ignore_reads()
  // , fast_ignore_writes()
  // , in_rtl()
  , shadow_stack_pos(&shadow_stack[0])
  , tid(tid)
  , unique_id(unique_id)
  , stk_addr(stk_addr)
  , stk_size(stk_size)
  , tls_addr(tls_addr)
  , tls_size(tls_size) {
}

ThreadContext::ThreadContext(int tid)
  : tid(tid)
  , unique_id()
  , os_id()
  , user_id()
  , thr()
  , status(ThreadStatusInvalid)
  , detached()
  , reuse_count()
  , epoch0()
  , epoch1()
  , dead_info()
  , dead_next() {
}

static void WriteMemoryProfile(char *buf, uptr buf_size, int num) {
  uptr shadow = GetShadowMemoryConsumption();

  int nthread = 0;
  int nlivethread = 0;
  uptr threadmem = 0;
  {
    Lock l(&ctx->thread_mtx);
    for (unsigned i = 0; i < kMaxTid; i++) {
      ThreadContext *tctx = ctx->threads[i];
      if (tctx == 0)
        continue;
      nthread += 1;
      threadmem += sizeof(ThreadContext);
      if (tctx->status != ThreadStatusRunning)
        continue;
      nlivethread += 1;
      threadmem += sizeof(ThreadState);
    }
  }

  uptr nsync = 0;
  uptr syncmem = CTX()->synctab.GetMemoryConsumption(&nsync);

  internal_snprintf(buf, buf_size, "%d: shadow=%zuMB"
                                   " thread=%zuMB(total=%d/live=%d)"
                                   " sync=%zuMB(cnt=%zu)\n",
    num,
    shadow >> 20,
    threadmem >> 20, nthread, nlivethread,
    syncmem >> 20, nsync);
}

static void MemoryProfileThread(void *arg) {
  ScopedInRtl in_rtl;
  fd_t fd = (fd_t)(uptr)arg;
  for (int i = 0; ; i++) {
    InternalScopedBuffer<char> buf(4096);
    WriteMemoryProfile(buf.data(), buf.size(), i);
    internal_write(fd, buf.data(), internal_strlen(buf.data()));
    SleepForSeconds(1);
  }
}

static void InitializeMemoryProfile() {
  if (flags()->profile_memory == 0 || flags()->profile_memory[0] == 0)
    return;
  InternalScopedBuffer<char> filename(4096);
  internal_snprintf(filename.data(), filename.size(), "%s.%d",
      flags()->profile_memory, GetPid());
  fd_t fd = internal_open(filename.data(), true);
  if (fd == kInvalidFd) {
    Printf("Failed to open memory profile file '%s'\n", &filename[0]);
    Die();
  }
  internal_start_thread(&MemoryProfileThread, (void*)(uptr)fd);
}

static void MemoryFlushThread(void *arg) {
  ScopedInRtl in_rtl;
  for (int i = 0; ; i++) {
    SleepForMillis(flags()->flush_memory_ms);
    FlushShadowMemory();
  }
}

static void InitializeMemoryFlush() {
  if (flags()->flush_memory_ms == 0)
    return;
  if (flags()->flush_memory_ms < 100)
    flags()->flush_memory_ms = 100;
  internal_start_thread(&MemoryFlushThread, 0);
}

void MapShadow(uptr addr, uptr size) {
  MmapFixedNoReserve(MemToShadow(addr), size * kShadowMultiplier);
}

void Initialize(ThreadState *thr) {
  // Thread safe because done before all threads exist.
  static bool is_initialized = false;
  if (is_initialized)
    return;
  is_initialized = true;
  // Install tool-specific callbacks in sanitizer_common.
  SetCheckFailedCallback(TsanCheckFailed);

  ScopedInRtl in_rtl;
#ifndef TSAN_GO
  InitializeAllocator();
#endif
  InitializeInterceptors();
  const char *env = InitializePlatform();
  InitializeMutex();
  InitializeDynamicAnnotations();
  ctx = new(ctx_placeholder) Context;
#ifndef TSAN_GO
  InitializeShadowMemory();
#endif
  ctx->dead_list_size = 0;
  ctx->dead_list_head = 0;
  ctx->dead_list_tail = 0;
  InitializeFlags(&ctx->flags, env);
  // Setup correct file descriptor for error reports.
  __sanitizer_set_report_fd(flags()->log_fileno);
  InitializeSuppressions();
#ifndef TSAN_GO
  // Initialize external symbolizer before internal threads are started.
  const char *external_symbolizer = flags()->external_symbolizer_path;
  if (external_symbolizer != 0 && external_symbolizer[0] != '\0') {
    if (!InitializeExternalSymbolizer(external_symbolizer)) {
      Printf("Failed to start external symbolizer: '%s'\n",
             external_symbolizer);
      Die();
    }
  }
#endif
  InitializeMemoryProfile();
  InitializeMemoryFlush();

  if (ctx->flags.verbosity)
    Printf("***** Running under ThreadSanitizer v2 (pid %d) *****\n",
               GetPid());

  // Initialize thread 0.
  ctx->thread_seq = 0;
  int tid = ThreadCreate(thr, 0, 0, true);
  CHECK_EQ(tid, 0);
  ThreadStart(thr, tid, GetPid());
  CHECK_EQ(thr->in_rtl, 1);
  ctx->initialized = true;

  if (flags()->stop_on_start) {
    Printf("ThreadSanitizer is suspended at startup (pid %d)."
           " Call __tsan_resume().\n",
           GetPid());
    while (__tsan_resumed == 0);
  }
}

int Finalize(ThreadState *thr) {
  ScopedInRtl in_rtl;
  Context *ctx = __tsan::ctx;
  bool failed = false;

  if (flags()->atexit_sleep_ms > 0 && ThreadCount(thr) > 1)
    SleepForMillis(flags()->atexit_sleep_ms);

  // Wait for pending reports.
  ctx->report_mtx.Lock();
  ctx->report_mtx.Unlock();

  ThreadFinalize(thr);

  if (ctx->nreported) {
    failed = true;
#ifndef TSAN_GO
    Printf("ThreadSanitizer: reported %d warnings\n", ctx->nreported);
#else
    Printf("Found %d data race(s)\n", ctx->nreported);
#endif
  }

  if (ctx->nmissed_expected) {
    failed = true;
    Printf("ThreadSanitizer: missed %d expected races\n",
        ctx->nmissed_expected);
  }

  StatAggregate(ctx->stat, thr->stat);
  StatOutput(ctx->stat);
  return failed ? flags()->exitcode : 0;
}

#ifndef TSAN_GO
u32 CurrentStackId(ThreadState *thr, uptr pc) {
  if (thr->shadow_stack_pos == 0)  // May happen during bootstrap.
    return 0;
  if (pc) {
    thr->shadow_stack_pos[0] = pc;
    thr->shadow_stack_pos++;
  }
  u32 id = StackDepotPut(thr->shadow_stack,
                         thr->shadow_stack_pos - thr->shadow_stack);
  if (pc)
    thr->shadow_stack_pos--;
  return id;
}
#endif

void TraceSwitch(ThreadState *thr) {
  thr->nomalloc++;
  ScopedInRtl in_rtl;
  Lock l(&thr->trace.mtx);
  unsigned trace = (thr->fast_state.epoch() / kTracePartSize) % kTraceParts;
  TraceHeader *hdr = &thr->trace.headers[trace];
  hdr->epoch0 = thr->fast_state.epoch();
  hdr->stack0.ObtainCurrent(thr, 0);
  thr->nomalloc--;
}

uptr TraceTopPC(ThreadState *thr) {
  Event *events = (Event*)GetThreadTrace(thr->tid);
  uptr pc = events[thr->fast_state.GetTracePos()];
  return pc;
}

uptr TraceSize() {
  return (uptr)(1ull << (kTracePartSizeBits + flags()->history_size + 1));
}

#ifndef TSAN_GO
extern "C" void __tsan_trace_switch() {
  TraceSwitch(cur_thread());
}

extern "C" void __tsan_report_race() {
  ReportRace(cur_thread());
}
#endif

ALWAYS_INLINE
static Shadow LoadShadow(u64 *p) {
  u64 raw = atomic_load((atomic_uint64_t*)p, memory_order_relaxed);
  return Shadow(raw);
}

ALWAYS_INLINE
static void StoreShadow(u64 *sp, u64 s) {
  atomic_store((atomic_uint64_t*)sp, s, memory_order_relaxed);
}

ALWAYS_INLINE
static void StoreIfNotYetStored(u64 *sp, u64 *s) {
  StoreShadow(sp, *s);
  *s = 0;
}

static inline void HandleRace(ThreadState *thr, u64 *shadow_mem,
                              Shadow cur, Shadow old) {
  thr->racy_state[0] = cur.raw();
  thr->racy_state[1] = old.raw();
  thr->racy_shadow_addr = shadow_mem;
#ifndef TSAN_GO
  HACKY_CALL(__tsan_report_race);
#else
  ReportRace(thr);
#endif
}

static inline bool BothReads(Shadow s, int kAccessIsWrite) {
  return !kAccessIsWrite && !s.is_write();
}

static inline bool OldIsRWNotWeaker(Shadow old, int kAccessIsWrite) {
  return old.is_write() || !kAccessIsWrite;
}

static inline bool OldIsRWWeakerOrEqual(Shadow old, int kAccessIsWrite) {
  return !old.is_write() || kAccessIsWrite;
}

static inline bool OldIsInSameSynchEpoch(Shadow old, ThreadState *thr) {
  return old.epoch() >= thr->fast_synch_epoch;
}

static inline bool HappensBefore(Shadow old, ThreadState *thr) {
  return thr->clock.get(old.tid()) >= old.epoch();
}

ALWAYS_INLINE
void MemoryAccessImpl(ThreadState *thr, uptr addr,
    int kAccessSizeLog, bool kAccessIsWrite,
    u64 *shadow_mem, Shadow cur) {
  StatInc(thr, StatMop);
  StatInc(thr, kAccessIsWrite ? StatMopWrite : StatMopRead);
  StatInc(thr, (StatType)(StatMop1 + kAccessSizeLog));

  // This potentially can live in an MMX/SSE scratch register.
  // The required intrinsics are:
  // __m128i _mm_move_epi64(__m128i*);
  // _mm_storel_epi64(u64*, __m128i);
  u64 store_word = cur.raw();

  // scan all the shadow values and dispatch to 4 categories:
  // same, replace, candidate and race (see comments below).
  // we consider only 3 cases regarding access sizes:
  // equal, intersect and not intersect. initially I considered
  // larger and smaller as well, it allowed to replace some
  // 'candidates' with 'same' or 'replace', but I think
  // it's just not worth it (performance- and complexity-wise).

  Shadow old(0);
  if (kShadowCnt == 1) {
    int idx = 0;
#include "tsan_update_shadow_word_inl.h"
  } else if (kShadowCnt == 2) {
    int idx = 0;
#include "tsan_update_shadow_word_inl.h"
    idx = 1;
#include "tsan_update_shadow_word_inl.h"
  } else if (kShadowCnt == 4) {
    int idx = 0;
#include "tsan_update_shadow_word_inl.h"
    idx = 1;
#include "tsan_update_shadow_word_inl.h"
    idx = 2;
#include "tsan_update_shadow_word_inl.h"
    idx = 3;
#include "tsan_update_shadow_word_inl.h"
  } else if (kShadowCnt == 8) {
    int idx = 0;
#include "tsan_update_shadow_word_inl.h"
    idx = 1;
#include "tsan_update_shadow_word_inl.h"
    idx = 2;
#include "tsan_update_shadow_word_inl.h"
    idx = 3;
#include "tsan_update_shadow_word_inl.h"
    idx = 4;
#include "tsan_update_shadow_word_inl.h"
    idx = 5;
#include "tsan_update_shadow_word_inl.h"
    idx = 6;
#include "tsan_update_shadow_word_inl.h"
    idx = 7;
#include "tsan_update_shadow_word_inl.h"
  } else {
    CHECK(false);
  }

  // we did not find any races and had already stored
  // the current access info, so we are done
  if (LIKELY(store_word == 0))
    return;
  // choose a random candidate slot and replace it
  StoreShadow(shadow_mem + (cur.epoch() % kShadowCnt), store_word);
  StatInc(thr, StatShadowReplace);
  return;
 RACE:
  HandleRace(thr, shadow_mem, cur, old);
  return;
}

ALWAYS_INLINE
void MemoryAccess(ThreadState *thr, uptr pc, uptr addr,
    int kAccessSizeLog, bool kAccessIsWrite) {
  u64 *shadow_mem = (u64*)MemToShadow(addr);
  DPrintf2("#%d: tsan::OnMemoryAccess: @%p %p size=%d"
      " is_write=%d shadow_mem=%p {%zx, %zx, %zx, %zx}\n",
      (int)thr->fast_state.tid(), (void*)pc, (void*)addr,
      (int)(1 << kAccessSizeLog), kAccessIsWrite, shadow_mem,
      (uptr)shadow_mem[0], (uptr)shadow_mem[1],
      (uptr)shadow_mem[2], (uptr)shadow_mem[3]);
#if TSAN_DEBUG
  if (!IsAppMem(addr)) {
    Printf("Access to non app mem %zx\n", addr);
    DCHECK(IsAppMem(addr));
  }
  if (!IsShadowMem((uptr)shadow_mem)) {
    Printf("Bad shadow addr %p (%zx)\n", shadow_mem, addr);
    DCHECK(IsShadowMem((uptr)shadow_mem));
  }
#endif

  FastState fast_state = thr->fast_state;
  if (fast_state.GetIgnoreBit())
    return;
  fast_state.IncrementEpoch();
  thr->fast_state = fast_state;
  Shadow cur(fast_state);
  cur.SetAddr0AndSizeLog(addr & 7, kAccessSizeLog);
  cur.SetWrite(kAccessIsWrite);

  // We must not store to the trace if we do not store to the shadow.
  // That is, this call must be moved somewhere below.
  TraceAddEvent(thr, fast_state, EventTypeMop, pc);

  MemoryAccessImpl(thr, addr, kAccessSizeLog, kAccessIsWrite,
      shadow_mem, cur);
}

static void MemoryRangeSet(ThreadState *thr, uptr pc, uptr addr, uptr size,
                           u64 val) {
  if (size == 0)
    return;
  // FIXME: fix me.
  uptr offset = addr % kShadowCell;
  if (offset) {
    offset = kShadowCell - offset;
    if (size <= offset)
      return;
    addr += offset;
    size -= offset;
  }
  DCHECK_EQ(addr % 8, 0);
  // If a user passes some insane arguments (memset(0)),
  // let it just crash as usual.
  if (!IsAppMem(addr) || !IsAppMem(addr + size - 1))
    return;
  (void)thr;
  (void)pc;
  // Some programs mmap like hundreds of GBs but actually used a small part.
  // So, it's better to report a false positive on the memory
  // then to hang here senselessly.
  const uptr kMaxResetSize = 4ull*1024*1024*1024;
  if (size > kMaxResetSize)
    size = kMaxResetSize;
  size = (size + (kShadowCell - 1)) & ~(kShadowCell - 1);
  u64 *p = (u64*)MemToShadow(addr);
  CHECK(IsShadowMem((uptr)p));
  CHECK(IsShadowMem((uptr)(p + size * kShadowCnt / kShadowCell - 1)));
  // FIXME: may overwrite a part outside the region
  for (uptr i = 0; i < size * kShadowCnt / kShadowCell;) {
    p[i++] = val;
    for (uptr j = 1; j < kShadowCnt; j++)
      p[i++] = 0;
  }
}

void MemoryResetRange(ThreadState *thr, uptr pc, uptr addr, uptr size) {
  MemoryRangeSet(thr, pc, addr, size, 0);
}

void MemoryRangeFreed(ThreadState *thr, uptr pc, uptr addr, uptr size) {
  MemoryAccessRange(thr, pc, addr, size, true);
  Shadow s(thr->fast_state);
  s.MarkAsFreed();
  s.SetWrite(true);
  s.SetAddr0AndSizeLog(0, 3);
  MemoryRangeSet(thr, pc, addr, size, s.raw());
}

void MemoryRangeImitateWrite(ThreadState *thr, uptr pc, uptr addr, uptr size) {
  Shadow s(thr->fast_state);
  s.SetWrite(true);
  s.SetAddr0AndSizeLog(0, 3);
  MemoryRangeSet(thr, pc, addr, size, s.raw());
}

ALWAYS_INLINE
void FuncEntry(ThreadState *thr, uptr pc) {
  DCHECK_EQ(thr->in_rtl, 0);
  StatInc(thr, StatFuncEnter);
  DPrintf2("#%d: FuncEntry %p\n", (int)thr->fast_state.tid(), (void*)pc);
  thr->fast_state.IncrementEpoch();
  TraceAddEvent(thr, thr->fast_state, EventTypeFuncEnter, pc);

  // Shadow stack maintenance can be replaced with
  // stack unwinding during trace switch (which presumably must be faster).
  DCHECK_GE(thr->shadow_stack_pos, &thr->shadow_stack[0]);
#ifndef TSAN_GO
  DCHECK_LT(thr->shadow_stack_pos, &thr->shadow_stack[kShadowStackSize]);
#else
  if (thr->shadow_stack_pos == thr->shadow_stack_end) {
    const int sz = thr->shadow_stack_end - thr->shadow_stack;
    const int newsz = 2 * sz;
    uptr *newstack = (uptr*)internal_alloc(MBlockShadowStack,
        newsz * sizeof(uptr));
    internal_memcpy(newstack, thr->shadow_stack, sz * sizeof(uptr));
    internal_free(thr->shadow_stack);
    thr->shadow_stack = newstack;
    thr->shadow_stack_pos = newstack + sz;
    thr->shadow_stack_end = newstack + newsz;
  }
#endif
  thr->shadow_stack_pos[0] = pc;
  thr->shadow_stack_pos++;
}

ALWAYS_INLINE
void FuncExit(ThreadState *thr) {
  DCHECK_EQ(thr->in_rtl, 0);
  StatInc(thr, StatFuncExit);
  DPrintf2("#%d: FuncExit\n", (int)thr->fast_state.tid());
  thr->fast_state.IncrementEpoch();
  TraceAddEvent(thr, thr->fast_state, EventTypeFuncExit, 0);

  DCHECK_GT(thr->shadow_stack_pos, &thr->shadow_stack[0]);
#ifndef TSAN_GO
  DCHECK_LT(thr->shadow_stack_pos, &thr->shadow_stack[kShadowStackSize]);
#endif
  thr->shadow_stack_pos--;
}

void IgnoreCtl(ThreadState *thr, bool write, bool begin) {
  DPrintf("#%d: IgnoreCtl(%d, %d)\n", thr->tid, write, begin);
  thr->ignore_reads_and_writes += begin ? 1 : -1;
  CHECK_GE(thr->ignore_reads_and_writes, 0);
  if (thr->ignore_reads_and_writes)
    thr->fast_state.SetIgnoreBit();
  else
    thr->fast_state.ClearIgnoreBit();
}

bool MD5Hash::operator==(const MD5Hash &other) const {
  return hash[0] == other.hash[0] && hash[1] == other.hash[1];
}

#if TSAN_DEBUG
void build_consistency_debug() {}
#else
void build_consistency_release() {}
#endif

#if TSAN_COLLECT_STATS
void build_consistency_stats() {}
#else
void build_consistency_nostats() {}
#endif

#if TSAN_SHADOW_COUNT == 1
void build_consistency_shadow1() {}
#elif TSAN_SHADOW_COUNT == 2
void build_consistency_shadow2() {}
#elif TSAN_SHADOW_COUNT == 4
void build_consistency_shadow4() {}
#else
void build_consistency_shadow8() {}
#endif

}  // namespace __tsan

#ifndef TSAN_GO
// Must be included in this file to make sure everything is inlined.
#include "tsan_interface_inl.h"
#endif
