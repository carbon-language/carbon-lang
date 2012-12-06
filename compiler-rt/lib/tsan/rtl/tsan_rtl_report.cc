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
#include "tsan_platform.h"
#include "tsan_rtl.h"
#include "tsan_suppressions.h"
#include "tsan_symbolize.h"
#include "tsan_report.h"
#include "tsan_sync.h"
#include "tsan_mman.h"
#include "tsan_flags.h"

namespace __tsan {

using namespace __sanitizer;  // NOLINT

void TsanCheckFailed(const char *file, int line, const char *cond,
                     u64 v1, u64 v2) {
  ScopedInRtl in_rtl;
  Printf("FATAL: ThreadSanitizer CHECK failed: "
         "%s:%d \"%s\" (0x%zx, 0x%zx)\n",
         file, line, cond, (uptr)v1, (uptr)v2);
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
  const char *path_prefix = flags()->strip_path_prefix;
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
  if (last && 0 == internal_strcmp(last, "schedunlock"))
    last_frame2->next = 0;
#endif
}

static ReportStack *SymbolizeStack(const StackTrace& trace) {
  if (trace.IsEmpty())
    return 0;
  ReportStack *stack = 0;
  for (uptr si = 0; si < trace.Size(); si++) {
    // We obtain the return address, that is, address of the next instruction,
    // so offset it by 1 byte.
    bool is_last = (si == trace.Size() - 1);
    ReportStack *ent = SymbolizeCode(trace.Get(si) - !is_last);
    CHECK_NE(ent, 0);
    ReportStack *last = ent;
    while (last->next) {
      last->pc += !is_last;
      last = last->next;
    }
    last->pc += !is_last;
    last->next = stack;
    stack = ent;
  }
  StackStripMain(stack);
  return stack;
}

ScopedReport::ScopedReport(ReportType typ) {
  ctx_ = CTX();
  void *mem = internal_alloc(MBlockReport, sizeof(ReportDesc));
  rep_ = new(mem) ReportDesc;
  rep_->typ = typ;
  ctx_->report_mtx.Lock();
}

ScopedReport::~ScopedReport() {
  ctx_->report_mtx.Unlock();
  DestroyAndFree(rep_);
}

void ScopedReport::AddStack(const StackTrace *stack) {
  ReportStack **rs = rep_->stacks.PushBack();
  *rs = SymbolizeStack(*stack);
}

void ScopedReport::AddMemoryAccess(uptr addr, Shadow s,
    const StackTrace *stack, const MutexSet *mset) {
  void *mem = internal_alloc(MBlockReportMop, sizeof(ReportMop));
  ReportMop *mop = new(mem) ReportMop;
  rep_->mops.PushBack(mop);
  mop->tid = s.tid();
  mop->addr = addr + s.addr0();
  mop->size = s.size();
  mop->write = s.is_write();
  mop->stack = SymbolizeStack(*stack);
  for (uptr i = 0; i < mset->Size(); i++) {
    MutexSet::Desc d = mset->Get(i);
    u64 uid = 0;
    uptr addr = SyncVar::SplitId(d.id, &uid);
    SyncVar *s = ctx_->synctab.GetIfExistsAndLock(addr, false);
    // Check that the mutex is still alive.
    // Another mutex can be created at the same address,
    // so check uid as well.
    if (s && s->CheckId(uid)) {
      ReportMopMutex mtx = {s->uid, d.write};
      mop->mset.PushBack(mtx);
      AddMutex(s);
    } else {
      ReportMopMutex mtx = {d.id, d.write};
      mop->mset.PushBack(mtx);
      AddMutex(d.id);
    }
    if (s)
      s->mtx.ReadUnlock();
  }
}

void ScopedReport::AddThread(const ThreadContext *tctx) {
  for (uptr i = 0; i < rep_->threads.Size(); i++) {
    if (rep_->threads[i]->id == tctx->tid)
      return;
  }
  void *mem = internal_alloc(MBlockReportThread, sizeof(ReportThread));
  ReportThread *rt = new(mem) ReportThread();
  rep_->threads.PushBack(rt);
  rt->id = tctx->tid;
  rt->pid = tctx->os_id;
  rt->running = (tctx->status == ThreadStatusRunning);
  rt->name = tctx->name ? internal_strdup(tctx->name) : 0;
  rt->stack = SymbolizeStack(tctx->creation_stack);
}

#ifndef TSAN_GO
static ThreadContext *FindThread(int unique_id) {
  CTX()->thread_mtx.CheckLocked();
  for (unsigned i = 0; i < kMaxTid; i++) {
    ThreadContext *tctx = CTX()->threads[i];
    if (tctx && tctx->unique_id == unique_id) {
      return tctx;
    }
  }
  return 0;
}
#endif

void ScopedReport::AddMutex(const SyncVar *s) {
  for (uptr i = 0; i < rep_->mutexes.Size(); i++) {
    if (rep_->mutexes[i]->id == s->uid)
      return;
  }
  void *mem = internal_alloc(MBlockReportMutex, sizeof(ReportMutex));
  ReportMutex *rm = new(mem) ReportMutex();
  rep_->mutexes.PushBack(rm);
  rm->id = s->uid;
  rm->destroyed = false;
  rm->stack = SymbolizeStack(s->creation_stack);
}

void ScopedReport::AddMutex(u64 id) {
  for (uptr i = 0; i < rep_->mutexes.Size(); i++) {
    if (rep_->mutexes[i]->id == id)
      return;
  }
  void *mem = internal_alloc(MBlockReportMutex, sizeof(ReportMutex));
  ReportMutex *rm = new(mem) ReportMutex();
  rep_->mutexes.PushBack(rm);
  rm->id = id;
  rm->destroyed = true;
  rm->stack = 0;
}

void ScopedReport::AddLocation(uptr addr, uptr size) {
  if (addr == 0)
    return;
#ifndef TSAN_GO
  if (allocator()->PointerIsMine((void*)addr)) {
    MBlock *b = user_mblock(0, (void*)addr);
    ThreadContext *tctx = FindThread(b->alloc_tid);
    void *mem = internal_alloc(MBlockReportLoc, sizeof(ReportLocation));
    ReportLocation *loc = new(mem) ReportLocation();
    rep_->locs.PushBack(loc);
    loc->type = ReportLocationHeap;
    loc->addr = (uptr)allocator()->GetBlockBegin((void*)addr);
    loc->size = b->size;
    loc->tid = tctx ? tctx->tid : b->alloc_tid;
    loc->name = 0;
    loc->file = 0;
    loc->line = 0;
    loc->stack = 0;
    uptr ssz = 0;
    const uptr *stack = StackDepotGet(b->alloc_stack_id, &ssz);
    if (stack) {
      StackTrace trace;
      trace.Init(stack, ssz);
      loc->stack = SymbolizeStack(trace);
    }
    if (tctx)
      AddThread(tctx);
    return;
  }
#endif
  ReportStack *symb = SymbolizeData(addr);
  if (symb) {
    void *mem = internal_alloc(MBlockReportLoc, sizeof(ReportLocation));
    ReportLocation *loc = new(mem) ReportLocation();
    rep_->locs.PushBack(loc);
    loc->type = ReportLocationGlobal;
    loc->addr = addr;
    loc->size = size;
    loc->module = symb->module ? internal_strdup(symb->module) : 0;
    loc->offset = symb->offset;
    loc->tid = 0;
    loc->name = symb->func ? internal_strdup(symb->func) : 0;
    loc->file = symb->file ? internal_strdup(symb->file) : 0;
    loc->line = symb->line;
    loc->stack = 0;
    internal_free(symb);
    return;
  }
}

#ifndef TSAN_GO
void ScopedReport::AddSleep(u32 stack_id) {
  uptr ssz = 0;
  const uptr *stack = StackDepotGet(stack_id, &ssz);
  if (stack) {
    StackTrace trace;
    trace.Init(stack, ssz);
    rep_->sleep = SymbolizeStack(trace);
  }
}
#endif

const ReportDesc *ScopedReport::GetReport() const {
  return rep_;
}

void RestoreStack(int tid, const u64 epoch, StackTrace *stk, MutexSet *mset) {
  // This function restores stack trace and mutex set for the thread/epoch.
  // It does so by getting stack trace and mutex set at the beginning of
  // trace part, and then replaying the trace till the given epoch.
  ThreadContext *tctx = CTX()->threads[tid];
  if (tctx == 0)
    return;
  Trace* trace = 0;
  if (tctx->status == ThreadStatusRunning) {
    CHECK(tctx->thr);
    trace = &tctx->thr->trace;
  } else if (tctx->status == ThreadStatusFinished
      || tctx->status == ThreadStatusDead) {
    if (tctx->dead_info == 0)
      return;
    trace = &tctx->dead_info->trace;
  } else {
    return;
  }
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
  InternalScopedBuffer<uptr> stack(1024);  // FIXME: de-hardcode 1024
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
  Context *ctx = CTX();
  bool equal_stack = false;
  RacyStacks hash = {};
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
  Context *ctx = CTX();
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

bool OutputReport(Context *ctx,
                  const ScopedReport &srep,
                  const ReportStack *suppress_stack) {
  const ReportDesc *rep = srep.GetReport();
  const uptr suppress_pc = IsSuppressed(rep->typ, suppress_stack);
  if (suppress_pc != 0) {
    FiredSuppression supp = {srep.GetReport()->typ, suppress_pc};
    ctx->fired_suppressions.PushBack(supp);
  }
  if (OnReport(rep, suppress_pc != 0))
    return false;
  PrintReport(rep);
  CTX()->nreported++;
  return true;
}

bool IsFiredSuppression(Context *ctx,
                        const ScopedReport &srep,
                        const StackTrace &trace) {
  for (uptr k = 0; k < ctx->fired_suppressions.Size(); k++) {
    if (ctx->fired_suppressions[k].type != srep.GetReport()->typ)
      continue;
    for (uptr j = 0; j < trace.Size(); j++) {
      if (trace.Get(j) == ctx->fired_suppressions[k].pc)
        return true;
    }
  }
  return false;
}

// On programs that use Java we see weird reports like:
// WARNING: ThreadSanitizer: data race (pid=22512)
//   Read of size 8 at 0x7d2b00084318 by thread 100:
//     #0 memcpy tsan_interceptors.cc:406 (foo+0x00000d8dfae3)
//     #1 <null> <null>:0 (0x7f7ad9b40193)
//   Previous write of size 8 at 0x7d2b00084318 by thread 105:
//     #0 strncpy tsan_interceptors.cc:501 (foo+0x00000d8e0919)
//     #1 <null> <null>:0 (0x7f7ad9b42707)
static bool IsJavaNonsense(const ReportDesc *rep) {
  for (uptr i = 0; i < rep->mops.Size(); i++) {
    ReportMop *mop = rep->mops[i];
    ReportStack *frame = mop->stack;
    if (frame != 0 && frame->func != 0
        && (internal_strcmp(frame->func, "memset") == 0
        || internal_strcmp(frame->func, "memcpy") == 0
        || internal_strcmp(frame->func, "strcmp") == 0
        || internal_strcmp(frame->func, "strncpy") == 0
        || internal_strcmp(frame->func, "pthread_mutex_lock") == 0)) {
      frame = frame->next;
      if (frame == 0
          || (frame->func == 0 && frame->file == 0 && frame->line == 0
            && frame->module == 0)) {
        if (frame) {
          FiredSuppression supp = {rep->typ, frame->pc};
          CTX()->fired_suppressions.PushBack(supp);
        }
        return true;
      }
    }
  }
  return false;
}

void ReportRace(ThreadState *thr) {
  if (!flags()->report_bugs)
    return;
  ScopedInRtl in_rtl;

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

  Context *ctx = CTX();
  Lock l0(&ctx->thread_mtx);

  ScopedReport rep(freed ? ReportTypeUseAfterFree : ReportTypeRace);
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

  if (HandleRacyStacks(thr, traces, addr_min, addr_max))
    return;

  for (uptr i = 0; i < kMop; i++) {
    Shadow s(thr->racy_state[i]);
    rep.AddMemoryAccess(addr, s, &traces[i],
                        i == 0 ? &thr->mset : mset2.data());
  }

  if (flags()->suppress_java && IsJavaNonsense(rep.GetReport()))
    return;

  for (uptr i = 0; i < kMop; i++) {
    FastState s(thr->racy_state[i]);
    ThreadContext *tctx = ctx->threads[s.tid()];
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

  if (!OutputReport(ctx, rep, rep.GetReport()->mops[0]->stack))
    return;

  AddRacyStacks(thr, traces, addr_min, addr_max);
}

void PrintCurrentStack(ThreadState *thr, uptr pc) {
  StackTrace trace;
  trace.ObtainCurrent(thr, pc);
  PrintStack(SymbolizeStack(trace));
}

}  // namespace __tsan
