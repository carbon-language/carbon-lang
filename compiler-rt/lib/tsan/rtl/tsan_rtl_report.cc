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
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_libc.h"
#include "tsan_platform.h"
#include "tsan_rtl.h"
#include "tsan_suppressions.h"
#include "tsan_symbolize.h"
#include "tsan_report.h"
#include "tsan_sync.h"
#include "tsan_mman.h"
#include "tsan_flags.h"
#include "tsan_placement_new.h"

namespace __sanitizer {
using namespace __tsan;

void CheckFailed(const char *file, int line, const char *cond, u64 v1, u64 v2) {
  ScopedInRtl in_rtl;
  TsanPrintf("FATAL: ThreadSanitizer CHECK failed: %s:%d \"%s\" (%zx, %zx)\n",
             file, line, cond, (uptr)v1, (uptr)v2);
  Die();
}

}  // namespace __sanitizer

namespace __tsan {

// Can be overriden by an application/test to intercept reports.
bool WEAK OnReport(const ReportDesc *rep, bool suppressed) {
  (void)rep;
  return suppressed;
}

static void StackStripMain(ReportStack *stack) {
  ReportStack *last_frame = 0;
  ReportStack *last_frame2 = 0;
  const char *prefix = "__interceptor_";
  uptr prefix_len = internal_strlen(prefix);
  const char *path_prefix = flags()->strip_path_prefix;
  uptr path_prefix_len = internal_strlen(path_prefix);
  for (ReportStack *ent = stack; ent; ent = ent->next) {
    if (ent->func && 0 == internal_strncmp(ent->func, prefix, prefix_len))
      ent->func += prefix_len;
    if (ent->file && 0 == internal_strncmp(ent->file, path_prefix,
                                           path_prefix_len))
      ent->file += path_prefix_len;
    if (ent->file && ent->file[0] == '.' && ent->file[1] == '/')
      ent->file += 2;
    last_frame2 = last_frame;
    last_frame = ent;
  }

  if (last_frame2 == 0)
    return;
  const char *last = last_frame->func;
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
    // so it's only a DCHECK. However we must try hard to not miss it
    // due to our fault.
    TsanPrintf("Bottom stack frame of stack %zx is missed\n", stack->pc);
  }
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
  rep_->~ReportDesc();
  internal_free(rep_);
}

void ScopedReport::AddStack(const StackTrace *stack) {
  ReportStack **rs = rep_->stacks.PushBack();
  *rs = SymbolizeStack(*stack);
}

void ScopedReport::AddMemoryAccess(uptr addr, Shadow s,
                                   const StackTrace *stack) {
  void *mem = internal_alloc(MBlockReportMop, sizeof(ReportMop));
  ReportMop *mop = new(mem) ReportMop;
  rep_->mops.PushBack(mop);
  mop->tid = s.tid();
  mop->addr = addr + s.addr0();
  mop->size = s.size();
  mop->write = s.is_write();
  mop->nmutex = 0;
  mop->stack = SymbolizeStack(*stack);
}

void ScopedReport::AddThread(const ThreadContext *tctx) {
  void *mem = internal_alloc(MBlockReportThread, sizeof(ReportThread));
  ReportThread *rt = new(mem) ReportThread();
  rep_->threads.PushBack(rt);
  rt->id = tctx->tid;
  rt->running = (tctx->status == ThreadStatusRunning);
  rt->stack = SymbolizeStack(tctx->creation_stack);
}

void ScopedReport::AddMutex(const SyncVar *s) {
  void *mem = internal_alloc(MBlockReportMutex, sizeof(ReportMutex));
  ReportMutex *rm = new(mem) ReportMutex();
  rep_->mutexes.PushBack(rm);
  rm->id = 42;
  rm->stack = SymbolizeStack(s->creation_stack);
}

void ScopedReport::AddLocation(uptr addr, uptr size) {
  ReportStack *symb = SymbolizeData(addr);
  if (symb) {
    void *mem = internal_alloc(MBlockReportLoc, sizeof(ReportLocation));
    ReportLocation *loc = new(mem) ReportLocation();
    rep_->locs.PushBack(loc);
    loc->type = ReportLocationGlobal;
    loc->addr = addr;
    loc->size = size;
    loc->tid = 0;
    loc->name = symb->func;
    loc->file = symb->file;
    loc->line = symb->line;
    loc->stack = 0;
    internal_free(symb);
  }
}

const ReportDesc *ScopedReport::GetReport() const {
  return rep_;
}

static void RestoreStack(int tid, const u64 epoch, StackTrace *stk) {
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
  const int partidx = (epoch / (kTraceSize / kTraceParts)) % kTraceParts;
  TraceHeader* hdr = &trace->headers[partidx];
  if (epoch < hdr->epoch0)
    return;
  const u64 eend = epoch % kTraceSize;
  const u64 ebegin = eend / kTracePartSize * kTracePartSize;
  DPrintf("#%d: RestoreStack epoch=%zu ebegin=%zu eend=%zu partidx=%d\n",
          tid, (uptr)epoch, (uptr)ebegin, (uptr)eend, partidx);
  InternalScopedBuf<uptr> stack(1024);  // FIXME: de-hardcode 1024
  for (uptr i = 0; i < hdr->stack0.Size(); i++) {
    stack[i] = hdr->stack0.Get(i);
    DPrintf2("  #%02lu: pc=%zx\n", i, stack[i]);
  }
  uptr pos = hdr->stack0.Size();
  for (uptr i = ebegin; i <= eend; i++) {
    Event ev = trace->events[i];
    EventType typ = (EventType)(ev >> 61);
    uptr pc = (uptr)(ev & 0xffffffffffffull);
    DPrintf2("  %zu typ=%d pc=%zx\n", i, typ, pc);
    if (typ == EventTypeMop) {
      stack[pos] = pc;
    } else if (typ == EventTypeFuncEnter) {
      stack[pos++] = pc;
    } else if (typ == EventTypeFuncExit) {
      // Since we have full stacks, this should never happen.
      DCHECK_GT(pos, 0);
      if (pos > 0)
        pos--;
    }
    for (uptr j = 0; j <= pos; j++)
      DPrintf2("      #%zu: %zx\n", j, stack[j]);
  }
  if (pos == 0 && stack[0] == 0)
    return;
  pos++;
  stk->Init(stack, pos);
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

bool OutputReport(const ScopedReport &srep, const ReportStack *suppress_stack) {
  const ReportDesc *rep = srep.GetReport();
  bool suppressed = IsSuppressed(rep->typ, suppress_stack);
  suppressed = OnReport(rep, suppressed);
  if (suppressed)
    return false;
  PrintReport(rep);
  CTX()->nreported++;
  return true;
}

void ReportRace(ThreadState *thr) {
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
  for (uptr i = 0; i < kMop; i++) {
    Shadow s(thr->racy_state[i]);
    RestoreStack(s.tid(), s.epoch(), &traces[i]);
  }

  if (HandleRacyStacks(thr, traces, addr_min, addr_max))
    return;

  for (uptr i = 0; i < kMop; i++) {
    Shadow s(thr->racy_state[i]);
    rep.AddMemoryAccess(addr, s, &traces[i]);
  }

  // Ensure that we have at least something for the current thread.
  CHECK_EQ(traces[0].IsEmpty(), false);

  for (uptr i = 0; i < kMop; i++) {
    FastState s(thr->racy_state[i]);
    ThreadContext *tctx = ctx->threads[s.tid()];
    if (s.epoch() < tctx->epoch0 || s.epoch() > tctx->epoch1)
      continue;
    rep.AddThread(tctx);
  }

  if (!OutputReport(rep, rep.GetReport()->mops[0]->stack))
    return;

  AddRacyStacks(thr, traces, addr_min, addr_max);
}

}  // namespace __tsan
