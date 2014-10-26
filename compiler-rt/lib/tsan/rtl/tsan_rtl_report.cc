//===-- tsan_rtl_report.cc ------------------------------------------------===//
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
#include "sanitizer_common/sanitizer_placement_new.h"
#include "sanitizer_common/sanitizer_stackdepot.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_stacktrace.h"
#include "tsan_platform.h"
#include "tsan_rtl.h"
#include "tsan_suppressions.h"
#include "tsan_symbolize.h"
#include "tsan_report.h"
#include "tsan_sync.h"
#include "tsan_mman.h"
#include "tsan_flags.h"
#include "tsan_fd.h"

namespace __tsan {

using namespace __sanitizer;  // NOLINT

static ReportStack *SymbolizeStack(const StackTrace& trace);

void TsanCheckFailed(const char *file, int line, const char *cond,
                     u64 v1, u64 v2) {
  // There is high probability that interceptors will check-fail as well,
  // on the other hand there is no sense in processing interceptors
  // since we are going to die soon.
  ScopedIgnoreInterceptors ignore;
  Printf("FATAL: ThreadSanitizer CHECK failed: "
         "%s:%d \"%s\" (0x%zx, 0x%zx)\n",
         file, line, cond, (uptr)v1, (uptr)v2);
  PrintCurrentStackSlow();
  Die();
}

// Can be overriden by an application/test to intercept reports.
#ifdef TSAN_EXTERNAL_HOOKS
bool OnReport(const ReportDesc *rep, bool suppressed);
#else
SANITIZER_INTERFACE_ATTRIBUTE
bool WEAK OnReport(const ReportDesc *rep, bool suppressed) {
  (void)rep;
  return suppressed;
}
#endif

static void StackStripMain(ReportStack *stack) {
  ReportStack *last_frame = 0;
  ReportStack *last_frame2 = 0;
  const char *prefix = "__interceptor_";
  uptr prefix_len = internal_strlen(prefix);
  const char *path_prefix = common_flags()->strip_path_prefix;
  uptr path_prefix_len = internal_strlen(path_prefix);
  char *pos;
  for (ReportStack *ent = stack; ent; ent = ent->next) {
    if (ent->func && 0 == internal_strncmp(ent->func, prefix, prefix_len))
      ent->func += prefix_len;
    if (ent->file && (pos = internal_strstr(ent->file, path_prefix)))
      ent->file = pos + path_prefix_len;
    if (ent->file && ent->file[0] == '.' && ent->file[1] == '/')
      ent->file += 2;
    last_frame2 = last_frame;
    last_frame = ent;
  }

  if (last_frame2 == 0)
    return;
  const char *last = last_frame->func;
#ifndef TSAN_GO
  const char *last2 = last_frame2->func;
  // Strip frame above 'main'
  if (last2 && 0 == internal_strcmp(last2, "main")) {
    last_frame2->next = 0;
  // Strip our internal thread start routine.
  } else if (last && 0 == internal_strcmp(last, "__tsan_thread_start_func")) {
    last_frame2->next = 0;
  // Strip global ctors init.
  } else if (last && 0 == internal_strcmp(last, "__do_global_ctors_aux")) {
    last_frame2->next = 0;
  // If both are 0, then we probably just failed to symbolize.
  } else if (last || last2) {
    // Ensure that we recovered stack completely. Trimmed stack
    // can actually happen if we do not instrument some code,
    // so it's only a debug print. However we must try hard to not miss it
    // due to our fault.
    DPrintf("Bottom stack frame of stack %zx is missed\n", stack->pc);
  }
#else
  // The last frame always point into runtime (gosched0, goexit0, runtime.main).
  last_frame2->next = 0;
  (void)last;
#endif
}

ReportStack *SymbolizeStackId(u32 stack_id) {
  if (stack_id == 0)
    return 0;
  __sanitizer::StackTrace stack = StackDepotGet(stack_id);
  if (stack.trace == nullptr)
    return 0;
  StackTrace trace;
  trace.Init(stack.trace, stack.size);
  return SymbolizeStack(trace);
}

static ReportStack *SymbolizeStack(const StackTrace& trace) {
  if (trace.IsEmpty())
    return 0;
  ReportStack *stack = 0;
  for (uptr si = 0; si < trace.Size(); si++) {
    const uptr pc = trace.Get(si);
#ifndef TSAN_GO
    // We obtain the return address, that is, address of the next instruction,
    // so offset it by 1 byte.
    const uptr pc1 = __sanitizer::StackTrace::GetPreviousInstructionPc(pc);
#else
    // FIXME(dvyukov): Go sometimes uses address of a function as top pc.
    uptr pc1 = pc;
    if (si != trace.Size() - 1)
      pc1 -= 1;
#endif
    ReportStack *ent = SymbolizeCode(pc1);
    CHECK_NE(ent, 0);
    ReportStack *last = ent;
    while (last->next) {
      last->pc = pc;  // restore original pc for report
      last = last->next;
    }
    last->pc = pc;  // restore original pc for report
    last->next = stack;
    stack = ent;
  }
  StackStripMain(stack);
  return stack;
}

ScopedReport::ScopedReport(ReportType typ) {
  ctx->thread_registry->CheckLocked();
  void *mem = internal_alloc(MBlockReport, sizeof(ReportDesc));
  rep_ = new(mem) ReportDesc;
  rep_->typ = typ;
  ctx->report_mtx.Lock();
  CommonSanitizerReportMutex.Lock();
}

ScopedReport::~ScopedReport() {
  CommonSanitizerReportMutex.Unlock();
  ctx->report_mtx.Unlock();
  DestroyAndFree(rep_);
}

void ScopedReport::AddStack(const StackTrace *stack, bool suppressable) {
  ReportStack **rs = rep_->stacks.PushBack();
  *rs = SymbolizeStack(*stack);
  (*rs)->suppressable = suppressable;
}

void ScopedReport::AddMemoryAccess(uptr addr, Shadow s,
    const StackTrace *stack, const MutexSet *mset) {
  void *mem = internal_alloc(MBlockReportMop, sizeof(ReportMop));
  ReportMop *mop = new(mem) ReportMop;
  rep_->mops.PushBack(mop);
  mop->tid = s.tid();
  mop->addr = addr + s.addr0();
  mop->size = s.size();
  mop->write = s.IsWrite();
  mop->atomic = s.IsAtomic();
  mop->stack = SymbolizeStack(*stack);
  if (mop->stack)
    mop->stack->suppressable = true;
  for (uptr i = 0; i < mset->Size(); i++) {
    MutexSet::Desc d = mset->Get(i);
    u64 mid = this->AddMutex(d.id);
    ReportMopMutex mtx = {mid, d.write};
    mop->mset.PushBack(mtx);
  }
}

void ScopedReport::AddUniqueTid(int unique_tid) {
  rep_->unique_tids.PushBack(unique_tid);
}

void ScopedReport::AddThread(const ThreadContext *tctx, bool suppressable) {
  for (uptr i = 0; i < rep_->threads.Size(); i++) {
    if ((u32)rep_->threads[i]->id == tctx->tid)
      return;
  }
  void *mem = internal_alloc(MBlockReportThread, sizeof(ReportThread));
  ReportThread *rt = new(mem) ReportThread();
  rep_->threads.PushBack(rt);
  rt->id = tctx->tid;
  rt->pid = tctx->os_id;
  rt->running = (tctx->status == ThreadStatusRunning);
  rt->name = internal_strdup(tctx->name);
  rt->parent_tid = tctx->parent_tid;
  rt->stack = 0;
  rt->stack = SymbolizeStackId(tctx->creation_stack_id);
  if (rt->stack)
    rt->stack->suppressable = suppressable;
}

#ifndef TSAN_GO
static ThreadContext *FindThreadByUidLocked(int unique_id) {
  ctx->thread_registry->CheckLocked();
  for (unsigned i = 0; i < kMaxTid; i++) {
    ThreadContext *tctx = static_cast<ThreadContext*>(
        ctx->thread_registry->GetThreadLocked(i));
    if (tctx && tctx->unique_id == (u32)unique_id) {
      return tctx;
    }
  }
  return 0;
}

static ThreadContext *FindThreadByTidLocked(int tid) {
  ctx->thread_registry->CheckLocked();
  return static_cast<ThreadContext*>(
      ctx->thread_registry->GetThreadLocked(tid));
}

static bool IsInStackOrTls(ThreadContextBase *tctx_base, void *arg) {
  uptr addr = (uptr)arg;
  ThreadContext *tctx = static_cast<ThreadContext*>(tctx_base);
  if (tctx->status != ThreadStatusRunning)
    return false;
  ThreadState *thr = tctx->thr;
  CHECK(thr);
  return ((addr >= thr->stk_addr && addr < thr->stk_addr + thr->stk_size) ||
          (addr >= thr->tls_addr && addr < thr->tls_addr + thr->tls_size));
}

ThreadContext *IsThreadStackOrTls(uptr addr, bool *is_stack) {
  ctx->thread_registry->CheckLocked();
  ThreadContext *tctx = static_cast<ThreadContext*>(
      ctx->thread_registry->FindThreadContextLocked(IsInStackOrTls,
                                                    (void*)addr));
  if (!tctx)
    return 0;
  ThreadState *thr = tctx->thr;
  CHECK(thr);
  *is_stack = (addr >= thr->stk_addr && addr < thr->stk_addr + thr->stk_size);
  return tctx;
}
#endif

void ScopedReport::AddThread(int unique_tid, bool suppressable) {
#ifndef TSAN_GO
  AddThread(FindThreadByUidLocked(unique_tid), suppressable);
#endif
}

void ScopedReport::AddMutex(const SyncVar *s) {
  for (uptr i = 0; i < rep_->mutexes.Size(); i++) {
    if (rep_->mutexes[i]->id == s->uid)
      return;
  }
  void *mem = internal_alloc(MBlockReportMutex, sizeof(ReportMutex));
  ReportMutex *rm = new(mem) ReportMutex();
  rep_->mutexes.PushBack(rm);
  rm->id = s->uid;
  rm->addr = s->addr;
  rm->destroyed = false;
  rm->stack = SymbolizeStackId(s->creation_stack_id);
}

u64 ScopedReport::AddMutex(u64 id) {
  u64 uid = 0;
  u64 mid = id;
  uptr addr = SyncVar::SplitId(id, &uid);
  SyncVar *s = ctx->metamap.GetIfExistsAndLock(addr);
  // Check that the mutex is still alive.
  // Another mutex can be created at the same address,
  // so check uid as well.
  if (s && s->CheckId(uid)) {
    mid = s->uid;
    AddMutex(s);
  } else {
    AddDeadMutex(id);
  }
  if (s)
    s->mtx.Unlock();
  return mid;
}

void ScopedReport::AddDeadMutex(u64 id) {
  for (uptr i = 0; i < rep_->mutexes.Size(); i++) {
    if (rep_->mutexes[i]->id == id)
      return;
  }
  void *mem = internal_alloc(MBlockReportMutex, sizeof(ReportMutex));
  ReportMutex *rm = new(mem) ReportMutex();
  rep_->mutexes.PushBack(rm);
  rm->id = id;
  rm->addr = 0;
  rm->destroyed = true;
  rm->stack = 0;
}

void ScopedReport::AddLocation(uptr addr, uptr size) {
  if (addr == 0)
    return;
#ifndef TSAN_GO
  int fd = -1;
  int creat_tid = -1;
  u32 creat_stack = 0;
  if (FdLocation(addr, &fd, &creat_tid, &creat_stack)) {
    void *mem = internal_alloc(MBlockReportLoc, sizeof(ReportLocation));
    ReportLocation *loc = new(mem) ReportLocation();
    rep_->locs.PushBack(loc);
    loc->type = ReportLocationFD;
    loc->fd = fd;
    loc->tid = creat_tid;
    loc->stack = SymbolizeStackId(creat_stack);
    ThreadContext *tctx = FindThreadByUidLocked(creat_tid);
    if (tctx)
      AddThread(tctx);
    return;
  }
  MBlock *b = 0;
  Allocator *a = allocator();
  if (a->PointerIsMine((void*)addr)) {
    void *block_begin = a->GetBlockBegin((void*)addr);
    if (block_begin)
      b = ctx->metamap.GetBlock((uptr)block_begin);
  }
  if (b != 0) {
    ThreadContext *tctx = FindThreadByTidLocked(b->tid);
    void *mem = internal_alloc(MBlockReportLoc, sizeof(ReportLocation));
    ReportLocation *loc = new(mem) ReportLocation();
    rep_->locs.PushBack(loc);
    loc->type = ReportLocationHeap;
    loc->addr = (uptr)allocator()->GetBlockBegin((void*)addr);
    loc->size = b->siz;
    loc->tid = tctx ? tctx->tid : b->tid;
    loc->name = 0;
    loc->file = 0;
    loc->line = 0;
    loc->stack = 0;
    loc->stack = SymbolizeStackId(b->stk);
    if (tctx)
      AddThread(tctx);
    return;
  }
  bool is_stack = false;
  if (ThreadContext *tctx = IsThreadStackOrTls(addr, &is_stack)) {
    void *mem = internal_alloc(MBlockReportLoc, sizeof(ReportLocation));
    ReportLocation *loc = new(mem) ReportLocation();
    rep_->locs.PushBack(loc);
    loc->type = is_stack ? ReportLocationStack : ReportLocationTLS;
    loc->tid = tctx->tid;
    AddThread(tctx);
  }
  ReportLocation *loc = SymbolizeData(addr);
  if (loc) {
    loc->suppressable = true;
    rep_->locs.PushBack(loc);
    return;
  }
#endif
}

#ifndef TSAN_GO
void ScopedReport::AddSleep(u32 stack_id) {
  rep_->sleep = SymbolizeStackId(stack_id);
}
#endif

void ScopedReport::SetCount(int count) {
  rep_->count = count;
}

const ReportDesc *ScopedReport::GetReport() const {
  return rep_;
}

void RestoreStack(int tid, const u64 epoch, StackTrace *stk, MutexSet *mset) {
  // This function restores stack trace and mutex set for the thread/epoch.
  // It does so by getting stack trace and mutex set at the beginning of
  // trace part, and then replaying the trace till the given epoch.
  ctx->thread_registry->CheckLocked();
  ThreadContext *tctx = static_cast<ThreadContext*>(
      ctx->thread_registry->GetThreadLocked(tid));
  if (tctx == 0)
    return;
  if (tctx->status != ThreadStatusRunning
      && tctx->status != ThreadStatusFinished
      && tctx->status != ThreadStatusDead)
    return;
  Trace* trace = ThreadTrace(tctx->tid);
  Lock l(&trace->mtx);
  const int partidx = (epoch / kTracePartSize) % TraceParts();
  TraceHeader* hdr = &trace->headers[partidx];
  if (epoch < hdr->epoch0)
    return;
  const u64 epoch0 = RoundDown(epoch, TraceSize());
  const u64 eend = epoch % TraceSize();
  const u64 ebegin = RoundDown(eend, kTracePartSize);
  DPrintf("#%d: RestoreStack epoch=%zu ebegin=%zu eend=%zu partidx=%d\n",
          tid, (uptr)epoch, (uptr)ebegin, (uptr)eend, partidx);
  InternalScopedBuffer<uptr> stack(kShadowStackSize);
  for (uptr i = 0; i < hdr->stack0.Size(); i++) {
    stack[i] = hdr->stack0.Get(i);
    DPrintf2("  #%02lu: pc=%zx\n", i, stack[i]);
  }
  if (mset)
    *mset = hdr->mset0;
  uptr pos = hdr->stack0.Size();
  Event *events = (Event*)GetThreadTrace(tid);
  for (uptr i = ebegin; i <= eend; i++) {
    Event ev = events[i];
    EventType typ = (EventType)(ev >> 61);
    uptr pc = (uptr)(ev & ((1ull << 61) - 1));
    DPrintf2("  %zu typ=%d pc=%zx\n", i, typ, pc);
    if (typ == EventTypeMop) {
      stack[pos] = pc;
    } else if (typ == EventTypeFuncEnter) {
      stack[pos++] = pc;
    } else if (typ == EventTypeFuncExit) {
      if (pos > 0)
        pos--;
    }
    if (mset) {
      if (typ == EventTypeLock) {
        mset->Add(pc, true, epoch0 + i);
      } else if (typ == EventTypeUnlock) {
        mset->Del(pc, true);
      } else if (typ == EventTypeRLock) {
        mset->Add(pc, false, epoch0 + i);
      } else if (typ == EventTypeRUnlock) {
        mset->Del(pc, false);
      }
    }
    for (uptr j = 0; j <= pos; j++)
      DPrintf2("      #%zu: %zx\n", j, stack[j]);
  }
  if (pos == 0 && stack[0] == 0)
    return;
  pos++;
  stk->Init(stack.data(), pos);
}

static bool HandleRacyStacks(ThreadState *thr, const StackTrace (&traces)[2],
    uptr addr_min, uptr addr_max) {
  bool equal_stack = false;
  RacyStacks hash;
  if (flags()->suppress_equal_stacks) {
    hash.hash[0] = md5_hash(traces[0].Begin(), traces[0].Size() * sizeof(uptr));
    hash.hash[1] = md5_hash(traces[1].Begin(), traces[1].Size() * sizeof(uptr));
    for (uptr i = 0; i < ctx->racy_stacks.Size(); i++) {
      if (hash == ctx->racy_stacks[i]) {
        DPrintf("ThreadSanitizer: suppressing report as doubled (stack)\n");
        equal_stack = true;
        break;
      }
    }
  }
  bool equal_address = false;
  RacyAddress ra0 = {addr_min, addr_max};
  if (flags()->suppress_equal_addresses) {
    for (uptr i = 0; i < ctx->racy_addresses.Size(); i++) {
      RacyAddress ra2 = ctx->racy_addresses[i];
      uptr maxbeg = max(ra0.addr_min, ra2.addr_min);
      uptr minend = min(ra0.addr_max, ra2.addr_max);
      if (maxbeg < minend) {
        DPrintf("ThreadSanitizer: suppressing report as doubled (addr)\n");
        equal_address = true;
        break;
      }
    }
  }
  if (equal_stack || equal_address) {
    if (!equal_stack)
      ctx->racy_stacks.PushBack(hash);
    if (!equal_address)
      ctx->racy_addresses.PushBack(ra0);
    return true;
  }
  return false;
}

static void AddRacyStacks(ThreadState *thr, const StackTrace (&traces)[2],
    uptr addr_min, uptr addr_max) {
  if (flags()->suppress_equal_stacks) {
    RacyStacks hash;
    hash.hash[0] = md5_hash(traces[0].Begin(), traces[0].Size() * sizeof(uptr));
    hash.hash[1] = md5_hash(traces[1].Begin(), traces[1].Size() * sizeof(uptr));
    ctx->racy_stacks.PushBack(hash);
  }
  if (flags()->suppress_equal_addresses) {
    RacyAddress ra0 = {addr_min, addr_max};
    ctx->racy_addresses.PushBack(ra0);
  }
}

bool OutputReport(ThreadState *thr, const ScopedReport &srep) {
  atomic_store(&ctx->last_symbolize_time_ns, NanoTime(), memory_order_relaxed);
  const ReportDesc *rep = srep.GetReport();
  Suppression *supp = 0;
  uptr suppress_pc = 0;
  for (uptr i = 0; suppress_pc == 0 && i < rep->mops.Size(); i++)
    suppress_pc = IsSuppressed(rep->typ, rep->mops[i]->stack, &supp);
  for (uptr i = 0; suppress_pc == 0 && i < rep->stacks.Size(); i++)
    suppress_pc = IsSuppressed(rep->typ, rep->stacks[i], &supp);
  for (uptr i = 0; suppress_pc == 0 && i < rep->threads.Size(); i++)
    suppress_pc = IsSuppressed(rep->typ, rep->threads[i]->stack, &supp);
  for (uptr i = 0; suppress_pc == 0 && i < rep->locs.Size(); i++)
    suppress_pc = IsSuppressed(rep->typ, rep->locs[i], &supp);
  if (suppress_pc != 0) {
    FiredSuppression s = {srep.GetReport()->typ, suppress_pc, supp};
    ctx->fired_suppressions.push_back(s);
  }
  {
    bool old_is_freeing = thr->is_freeing;
    thr->is_freeing = false;
    bool suppressed = OnReport(rep, suppress_pc != 0);
    thr->is_freeing = old_is_freeing;
    if (suppressed)
      return false;
  }
  PrintReport(rep);
  ctx->nreported++;
  if (flags()->halt_on_error)
    internal__exit(flags()->exitcode);
  return true;
}

bool IsFiredSuppression(Context *ctx,
                        const ScopedReport &srep,
                        const StackTrace &trace) {
  for (uptr k = 0; k < ctx->fired_suppressions.size(); k++) {
    if (ctx->fired_suppressions[k].type != srep.GetReport()->typ)
      continue;
    for (uptr j = 0; j < trace.Size(); j++) {
      FiredSuppression *s = &ctx->fired_suppressions[k];
      if (trace.Get(j) == s->pc) {
        if (s->supp)
          s->supp->hit_count++;
        return true;
      }
    }
  }
  return false;
}

static bool IsFiredSuppression(Context *ctx,
                               const ScopedReport &srep,
                               uptr addr) {
  for (uptr k = 0; k < ctx->fired_suppressions.size(); k++) {
    if (ctx->fired_suppressions[k].type != srep.GetReport()->typ)
      continue;
    FiredSuppression *s = &ctx->fired_suppressions[k];
    if (addr == s->pc) {
      if (s->supp)
        s->supp->hit_count++;
      return true;
    }
  }
  return false;
}

bool FrameIsInternal(const ReportStack *frame) {
  return frame != 0 && frame->file != 0
      && (internal_strstr(frame->file, "tsan_interceptors.cc") ||
          internal_strstr(frame->file, "sanitizer_common_interceptors.inc") ||
          internal_strstr(frame->file, "tsan_interface_"));
}

static bool RaceBetweenAtomicAndFree(ThreadState *thr) {
  Shadow s0(thr->racy_state[0]);
  Shadow s1(thr->racy_state[1]);
  CHECK(!(s0.IsAtomic() && s1.IsAtomic()));
  if (!s0.IsAtomic() && !s1.IsAtomic())
    return true;
  if (s0.IsAtomic() && s1.IsFreed())
    return true;
  if (s1.IsAtomic() && thr->is_freeing)
    return true;
  return false;
}

void ReportRace(ThreadState *thr) {
  CheckNoLocks(thr);

  // Symbolizer makes lots of intercepted calls. If we try to process them,
  // at best it will cause deadlocks on internal mutexes.
  ScopedIgnoreInterceptors ignore;

  if (!flags()->report_bugs)
    return;
  if (!flags()->report_atomic_races && !RaceBetweenAtomicAndFree(thr))
    return;

  bool freed = false;
  {
    Shadow s(thr->racy_state[1]);
    freed = s.GetFreedAndReset();
    thr->racy_state[1] = s.raw();
  }

  uptr addr = ShadowToMem((uptr)thr->racy_shadow_addr);
  uptr addr_min = 0;
  uptr addr_max = 0;
  {
    uptr a0 = addr + Shadow(thr->racy_state[0]).addr0();
    uptr a1 = addr + Shadow(thr->racy_state[1]).addr0();
    uptr e0 = a0 + Shadow(thr->racy_state[0]).size();
    uptr e1 = a1 + Shadow(thr->racy_state[1]).size();
    addr_min = min(a0, a1);
    addr_max = max(e0, e1);
    if (IsExpectedReport(addr_min, addr_max - addr_min))
      return;
  }

  ThreadRegistryLock l0(ctx->thread_registry);

  ReportType typ = ReportTypeRace;
  if (thr->is_vptr_access && freed)
    typ = ReportTypeVptrUseAfterFree;
  else if (thr->is_vptr_access)
    typ = ReportTypeVptrRace;
  else if (freed)
    typ = ReportTypeUseAfterFree;
  ScopedReport rep(typ);
  if (IsFiredSuppression(ctx, rep, addr))
    return;
  const uptr kMop = 2;
  StackTrace traces[kMop];
  const uptr toppc = TraceTopPC(thr);
  traces[0].ObtainCurrent(thr, toppc);
  if (IsFiredSuppression(ctx, rep, traces[0]))
    return;
  InternalScopedBuffer<MutexSet> mset2(1);
  new(mset2.data()) MutexSet();
  Shadow s2(thr->racy_state[1]);
  RestoreStack(s2.tid(), s2.epoch(), &traces[1], mset2.data());
  if (IsFiredSuppression(ctx, rep, traces[1]))
    return;

  if (HandleRacyStacks(thr, traces, addr_min, addr_max))
    return;

  for (uptr i = 0; i < kMop; i++) {
    Shadow s(thr->racy_state[i]);
    rep.AddMemoryAccess(addr, s, &traces[i],
                        i == 0 ? &thr->mset : mset2.data());
  }

  for (uptr i = 0; i < kMop; i++) {
    FastState s(thr->racy_state[i]);
    ThreadContext *tctx = static_cast<ThreadContext*>(
        ctx->thread_registry->GetThreadLocked(s.tid()));
    if (s.epoch() < tctx->epoch0 || s.epoch() > tctx->epoch1)
      continue;
    rep.AddThread(tctx);
  }

  rep.AddLocation(addr_min, addr_max - addr_min);

#ifndef TSAN_GO
  {  // NOLINT
    Shadow s(thr->racy_state[1]);
    if (s.epoch() <= thr->last_sleep_clock.get(s.tid()))
      rep.AddSleep(thr->last_sleep_stack_id);
  }
#endif

  if (!OutputReport(thr, rep))
    return;

  AddRacyStacks(thr, traces, addr_min, addr_max);
}

void PrintCurrentStack(ThreadState *thr, uptr pc) {
  StackTrace trace;
  trace.ObtainCurrent(thr, pc);
  PrintStack(SymbolizeStack(trace));
}

void PrintCurrentStackSlow() {
#ifndef TSAN_GO
  __sanitizer::BufferedStackTrace *ptrace = new(
      internal_alloc(MBlockStackTrace, sizeof(__sanitizer::BufferedStackTrace)))
      __sanitizer::BufferedStackTrace();
  ptrace->Unwind(kStackTraceMax, __sanitizer::StackTrace::GetCurrentPc(), 0, 0,
                 0, 0, false);
  for (uptr i = 0; i < ptrace->size / 2; i++) {
    uptr tmp = ptrace->trace_buffer[i];
    ptrace->trace_buffer[i] = ptrace->trace_buffer[ptrace->size - i - 1];
    ptrace->trace_buffer[ptrace->size - i - 1] = tmp;
  }
  StackTrace trace;
  trace.Init(ptrace->trace, ptrace->size);
  PrintStack(SymbolizeStack(trace));
#endif
}

}  // namespace __tsan
