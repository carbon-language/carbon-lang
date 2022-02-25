//===-- tsan_rtl_report.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

using namespace __sanitizer;

static ReportStack *SymbolizeStack(StackTrace trace);

// Can be overriden by an application/test to intercept reports.
#ifdef TSAN_EXTERNAL_HOOKS
bool OnReport(const ReportDesc *rep, bool suppressed);
#else
SANITIZER_WEAK_CXX_DEFAULT_IMPL
bool OnReport(const ReportDesc *rep, bool suppressed) {
  (void)rep;
  return suppressed;
}
#endif

SANITIZER_WEAK_DEFAULT_IMPL
void __tsan_on_report(const ReportDesc *rep) {
  (void)rep;
}

static void StackStripMain(SymbolizedStack *frames) {
  SymbolizedStack *last_frame = nullptr;
  SymbolizedStack *last_frame2 = nullptr;
  for (SymbolizedStack *cur = frames; cur; cur = cur->next) {
    last_frame2 = last_frame;
    last_frame = cur;
  }

  if (last_frame2 == 0)
    return;
#if !SANITIZER_GO
  const char *last = last_frame->info.function;
  const char *last2 = last_frame2->info.function;
  // Strip frame above 'main'
  if (last2 && 0 == internal_strcmp(last2, "main")) {
    last_frame->ClearAll();
    last_frame2->next = nullptr;
  // Strip our internal thread start routine.
  } else if (last && 0 == internal_strcmp(last, "__tsan_thread_start_func")) {
    last_frame->ClearAll();
    last_frame2->next = nullptr;
    // Strip global ctors init, .preinit_array and main caller.
  } else if (last && (0 == internal_strcmp(last, "__do_global_ctors_aux") ||
                      0 == internal_strcmp(last, "__libc_csu_init") ||
                      0 == internal_strcmp(last, "__libc_start_main"))) {
    last_frame->ClearAll();
    last_frame2->next = nullptr;
  // If both are 0, then we probably just failed to symbolize.
  } else if (last || last2) {
    // Ensure that we recovered stack completely. Trimmed stack
    // can actually happen if we do not instrument some code,
    // so it's only a debug print. However we must try hard to not miss it
    // due to our fault.
    DPrintf("Bottom stack frame is missed\n");
  }
#else
  // The last frame always point into runtime (gosched0, goexit0, runtime.main).
  last_frame->ClearAll();
  last_frame2->next = nullptr;
#endif
}

ReportStack *SymbolizeStackId(u32 stack_id) {
  if (stack_id == 0)
    return 0;
  StackTrace stack = StackDepotGet(stack_id);
  if (stack.trace == nullptr)
    return nullptr;
  return SymbolizeStack(stack);
}

static ReportStack *SymbolizeStack(StackTrace trace) {
  if (trace.size == 0)
    return 0;
  SymbolizedStack *top = nullptr;
  for (uptr si = 0; si < trace.size; si++) {
    const uptr pc = trace.trace[si];
    uptr pc1 = pc;
    // We obtain the return address, but we're interested in the previous
    // instruction.
    if ((pc & kExternalPCBit) == 0)
      pc1 = StackTrace::GetPreviousInstructionPc(pc);
    SymbolizedStack *ent = SymbolizeCode(pc1);
    CHECK_NE(ent, 0);
    SymbolizedStack *last = ent;
    while (last->next) {
      last->info.address = pc;  // restore original pc for report
      last = last->next;
    }
    last->info.address = pc;  // restore original pc for report
    last->next = top;
    top = ent;
  }
  StackStripMain(top);

  auto *stack = New<ReportStack>();
  stack->frames = top;
  return stack;
}

bool ShouldReport(ThreadState *thr, ReportType typ) {
  // We set thr->suppress_reports in the fork context.
  // Taking any locking in the fork context can lead to deadlocks.
  // If any locks are already taken, it's too late to do this check.
  CheckedMutex::CheckNoLocks();
  // For the same reason check we didn't lock thread_registry yet.
  if (SANITIZER_DEBUG)
    ThreadRegistryLock l(&ctx->thread_registry);
  if (!flags()->report_bugs || thr->suppress_reports)
    return false;
  switch (typ) {
    case ReportTypeSignalUnsafe:
      return flags()->report_signal_unsafe;
    case ReportTypeThreadLeak:
#if !SANITIZER_GO
      // It's impossible to join phantom threads
      // in the child after fork.
      if (ctx->after_multithreaded_fork)
        return false;
#endif
      return flags()->report_thread_leaks;
    case ReportTypeMutexDestroyLocked:
      return flags()->report_destroy_locked;
    default:
      return true;
  }
}

ScopedReportBase::ScopedReportBase(ReportType typ, uptr tag) {
  ctx->thread_registry.CheckLocked();
  rep_ = New<ReportDesc>();
  rep_->typ = typ;
  rep_->tag = tag;
  ctx->report_mtx.Lock();
}

ScopedReportBase::~ScopedReportBase() {
  ctx->report_mtx.Unlock();
  DestroyAndFree(rep_);
}

void ScopedReportBase::AddStack(StackTrace stack, bool suppressable) {
  ReportStack **rs = rep_->stacks.PushBack();
  *rs = SymbolizeStack(stack);
  (*rs)->suppressable = suppressable;
}

void ScopedReportBase::AddMemoryAccess(uptr addr, uptr external_tag, Shadow s,
                                       StackTrace stack, const MutexSet *mset) {
  auto *mop = New<ReportMop>();
  rep_->mops.PushBack(mop);
  mop->tid = s.tid();
  mop->addr = addr + s.addr0();
  mop->size = s.size();
  mop->write = s.IsWrite();
  mop->atomic = s.IsAtomic();
  mop->stack = SymbolizeStack(stack);
  mop->external_tag = external_tag;
  if (mop->stack)
    mop->stack->suppressable = true;
  for (uptr i = 0; i < mset->Size(); i++) {
    MutexSet::Desc d = mset->Get(i);
    u64 mid = this->AddMutex(d.id);
    ReportMopMutex mtx = {mid, d.write};
    mop->mset.PushBack(mtx);
  }
}

void ScopedReportBase::AddUniqueTid(Tid unique_tid) {
  rep_->unique_tids.PushBack(unique_tid);
}

void ScopedReportBase::AddThread(const ThreadContext *tctx, bool suppressable) {
  for (uptr i = 0; i < rep_->threads.Size(); i++) {
    if ((u32)rep_->threads[i]->id == tctx->tid)
      return;
  }
  auto *rt = New<ReportThread>();
  rep_->threads.PushBack(rt);
  rt->id = tctx->tid;
  rt->os_id = tctx->os_id;
  rt->running = (tctx->status == ThreadStatusRunning);
  rt->name = internal_strdup(tctx->name);
  rt->parent_tid = tctx->parent_tid;
  rt->thread_type = tctx->thread_type;
  rt->stack = 0;
  rt->stack = SymbolizeStackId(tctx->creation_stack_id);
  if (rt->stack)
    rt->stack->suppressable = suppressable;
}

#if !SANITIZER_GO
static bool FindThreadByUidLockedCallback(ThreadContextBase *tctx, void *arg) {
  int unique_id = *(int *)arg;
  return tctx->unique_id == (u32)unique_id;
}

static ThreadContext *FindThreadByUidLocked(Tid unique_id) {
  ctx->thread_registry.CheckLocked();
  return static_cast<ThreadContext *>(
      ctx->thread_registry.FindThreadContextLocked(
          FindThreadByUidLockedCallback, &unique_id));
}

static ThreadContext *FindThreadByTidLocked(Tid tid) {
  ctx->thread_registry.CheckLocked();
  return static_cast<ThreadContext *>(
      ctx->thread_registry.GetThreadLocked(tid));
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
  ctx->thread_registry.CheckLocked();
  ThreadContext *tctx =
      static_cast<ThreadContext *>(ctx->thread_registry.FindThreadContextLocked(
          IsInStackOrTls, (void *)addr));
  if (!tctx)
    return 0;
  ThreadState *thr = tctx->thr;
  CHECK(thr);
  *is_stack = (addr >= thr->stk_addr && addr < thr->stk_addr + thr->stk_size);
  return tctx;
}
#endif

void ScopedReportBase::AddThread(Tid unique_tid, bool suppressable) {
#if !SANITIZER_GO
  if (const ThreadContext *tctx = FindThreadByUidLocked(unique_tid))
    AddThread(tctx, suppressable);
#endif
}

void ScopedReportBase::AddMutex(const SyncVar *s) {
  for (uptr i = 0; i < rep_->mutexes.Size(); i++) {
    if (rep_->mutexes[i]->id == s->uid)
      return;
  }
  auto *rm = New<ReportMutex>();
  rep_->mutexes.PushBack(rm);
  rm->id = s->uid;
  rm->addr = s->addr;
  rm->destroyed = false;
  rm->stack = SymbolizeStackId(s->creation_stack_id);
}

u64 ScopedReportBase::AddMutex(u64 id) {
  u64 uid = 0;
  u64 mid = id;
  uptr addr = SyncVar::SplitId(id, &uid);
  SyncVar *s = ctx->metamap.GetSyncIfExists(addr);
  // Check that the mutex is still alive.
  // Another mutex can be created at the same address,
  // so check uid as well.
  if (s && s->CheckId(uid)) {
    Lock l(&s->mtx);
    mid = s->uid;
    AddMutex(s);
  } else {
    AddDeadMutex(id);
  }
  return mid;
}

void ScopedReportBase::AddDeadMutex(u64 id) {
  for (uptr i = 0; i < rep_->mutexes.Size(); i++) {
    if (rep_->mutexes[i]->id == id)
      return;
  }
  auto *rm = New<ReportMutex>();
  rep_->mutexes.PushBack(rm);
  rm->id = id;
  rm->addr = 0;
  rm->destroyed = true;
  rm->stack = 0;
}

void ScopedReportBase::AddLocation(uptr addr, uptr size) {
  if (addr == 0)
    return;
#if !SANITIZER_GO
  int fd = -1;
  Tid creat_tid = kInvalidTid;
  StackID creat_stack = 0;
  if (FdLocation(addr, &fd, &creat_tid, &creat_stack)) {
    auto *loc = New<ReportLocation>();
    loc->type = ReportLocationFD;
    loc->fd = fd;
    loc->tid = creat_tid;
    loc->stack = SymbolizeStackId(creat_stack);
    rep_->locs.PushBack(loc);
    ThreadContext *tctx = FindThreadByUidLocked(creat_tid);
    if (tctx)
      AddThread(tctx);
    return;
  }
  MBlock *b = 0;
  uptr block_begin = 0;
  Allocator *a = allocator();
  if (a->PointerIsMine((void*)addr)) {
    block_begin = (uptr)a->GetBlockBegin((void *)addr);
    if (block_begin)
      b = ctx->metamap.GetBlock(block_begin);
  }
  if (!b)
    b = JavaHeapBlock(addr, &block_begin);
  if (b != 0) {
    ThreadContext *tctx = FindThreadByTidLocked(b->tid);
    auto *loc = New<ReportLocation>();
    loc->type = ReportLocationHeap;
    loc->heap_chunk_start = (uptr)allocator()->GetBlockBegin((void *)addr);
    loc->heap_chunk_size = b->siz;
    loc->external_tag = b->tag;
    loc->tid = tctx ? tctx->tid : b->tid;
    loc->stack = SymbolizeStackId(b->stk);
    rep_->locs.PushBack(loc);
    if (tctx)
      AddThread(tctx);
    return;
  }
  bool is_stack = false;
  if (ThreadContext *tctx = IsThreadStackOrTls(addr, &is_stack)) {
    auto *loc = New<ReportLocation>();
    loc->type = is_stack ? ReportLocationStack : ReportLocationTLS;
    loc->tid = tctx->tid;
    rep_->locs.PushBack(loc);
    AddThread(tctx);
  }
#endif
  if (ReportLocation *loc = SymbolizeData(addr)) {
    loc->suppressable = true;
    rep_->locs.PushBack(loc);
    return;
  }
}

#if !SANITIZER_GO
void ScopedReportBase::AddSleep(StackID stack_id) {
  rep_->sleep = SymbolizeStackId(stack_id);
}
#endif

void ScopedReportBase::SetCount(int count) { rep_->count = count; }

const ReportDesc *ScopedReportBase::GetReport() const { return rep_; }

ScopedReport::ScopedReport(ReportType typ, uptr tag)
    : ScopedReportBase(typ, tag) {}

ScopedReport::~ScopedReport() {}

void RestoreStack(Tid tid, const u64 epoch, VarSizeStackTrace *stk,
                  MutexSet *mset, uptr *tag) {
  // This function restores stack trace and mutex set for the thread/epoch.
  // It does so by getting stack trace and mutex set at the beginning of
  // trace part, and then replaying the trace till the given epoch.
  Trace* trace = ThreadTrace(tid);
  ReadLock l(&trace->mtx);
  const int partidx = (epoch / kTracePartSize) % TraceParts();
  TraceHeader* hdr = &trace->headers[partidx];
  if (epoch < hdr->epoch0 || epoch >= hdr->epoch0 + kTracePartSize)
    return;
  CHECK_EQ(RoundDown(epoch, kTracePartSize), hdr->epoch0);
  const u64 epoch0 = RoundDown(epoch, TraceSize());
  const u64 eend = epoch % TraceSize();
  const u64 ebegin = RoundDown(eend, kTracePartSize);
  DPrintf("#%d: RestoreStack epoch=%zu ebegin=%zu eend=%zu partidx=%d\n",
          tid, (uptr)epoch, (uptr)ebegin, (uptr)eend, partidx);
  Vector<uptr> stack;
  stack.Resize(hdr->stack0.size + 64);
  for (uptr i = 0; i < hdr->stack0.size; i++) {
    stack[i] = hdr->stack0.trace[i];
    DPrintf2("  #%02zu: pc=%zx\n", i, stack[i]);
  }
  if (mset)
    *mset = hdr->mset0;
  uptr pos = hdr->stack0.size;
  Event *events = (Event*)GetThreadTrace(tid);
  for (uptr i = ebegin; i <= eend; i++) {
    Event ev = events[i];
    EventType typ = (EventType)(ev >> kEventPCBits);
    uptr pc = (uptr)(ev & ((1ull << kEventPCBits) - 1));
    DPrintf2("  %zu typ=%d pc=%zx\n", i, typ, pc);
    if (typ == EventTypeMop) {
      stack[pos] = pc;
    } else if (typ == EventTypeFuncEnter) {
      if (stack.Size() < pos + 2)
        stack.Resize(pos + 2);
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
  stk->Init(&stack[0], pos);
  ExtractTagFromStack(stk, tag);
}

namespace v3 {

// Replays the trace up to last_pos position in the last part
// or up to the provided epoch/sid (whichever is earlier)
// and calls the provided function f for each event.
template <typename Func>
void TraceReplay(Trace *trace, TracePart *last, Event *last_pos, Sid sid,
                 Epoch epoch, Func f) {
  TracePart *part = trace->parts.Front();
  Sid ev_sid = kFreeSid;
  Epoch ev_epoch = kEpochOver;
  for (;;) {
    DCHECK_EQ(part->trace, trace);
    // Note: an event can't start in the last element.
    // Since an event can take up to 2 elements,
    // we ensure we have at least 2 before adding an event.
    Event *end = &part->events[TracePart::kSize - 1];
    if (part == last)
      end = last_pos;
    for (Event *evp = &part->events[0]; evp < end; evp++) {
      Event *evp0 = evp;
      if (!evp->is_access && !evp->is_func) {
        switch (evp->type) {
          case EventType::kTime: {
            auto *ev = reinterpret_cast<EventTime *>(evp);
            ev_sid = static_cast<Sid>(ev->sid);
            ev_epoch = static_cast<Epoch>(ev->epoch);
            if (ev_sid == sid && ev_epoch > epoch)
              return;
            break;
          }
          case EventType::kAccessExt:
            FALLTHROUGH;
          case EventType::kAccessRange:
            FALLTHROUGH;
          case EventType::kLock:
            FALLTHROUGH;
          case EventType::kRLock:
            // These take 2 Event elements.
            evp++;
            break;
          case EventType::kUnlock:
            // This takes 1 Event element.
            break;
        }
      }
      CHECK_NE(ev_sid, kFreeSid);
      CHECK_NE(ev_epoch, kEpochOver);
      f(ev_sid, ev_epoch, evp0);
    }
    if (part == last)
      return;
    part = trace->parts.Next(part);
    CHECK(part);
  }
  CHECK(0);
}

static void RestoreStackMatch(VarSizeStackTrace *pstk, MutexSet *pmset,
                              Vector<uptr> *stack, MutexSet *mset, uptr pc,
                              bool *found) {
  DPrintf2("    MATCHED\n");
  *pmset = *mset;
  stack->PushBack(pc);
  pstk->Init(&(*stack)[0], stack->Size());
  stack->PopBack();
  *found = true;
}

// Checks if addr1|size1 is fully contained in addr2|size2.
// We check for fully contained instread of just overlapping
// because a memory access is always traced once, but can be
// split into multiple accesses in the shadow.
static constexpr bool IsWithinAccess(uptr addr1, uptr size1, uptr addr2,
                                     uptr size2) {
  return addr1 >= addr2 && addr1 + size1 <= addr2 + size2;
}

// Replays the trace of thread tid up to the target event identified
// by sid/epoch/addr/size/typ and restores and returns stack, mutex set
// and tag for that event. If there are multiple such events, it returns
// the last one. Returns false if the event is not present in the trace.
bool RestoreStack(Tid tid, EventType type, Sid sid, Epoch epoch, uptr addr,
                  uptr size, AccessType typ, VarSizeStackTrace *pstk,
                  MutexSet *pmset, uptr *ptag) {
  // This function restores stack trace and mutex set for the thread/epoch.
  // It does so by getting stack trace and mutex set at the beginning of
  // trace part, and then replaying the trace till the given epoch.
  DPrintf2("RestoreStack: tid=%u sid=%u@%u addr=0x%zx/%zu typ=%x\n", tid, sid,
           epoch, addr, size, typ);
  ctx->slot_mtx.CheckLocked();  // needed to prevent trace part recycling
  ctx->thread_registry.CheckLocked();
  ThreadContext *tctx =
      static_cast<ThreadContext *>(ctx->thread_registry.GetThreadLocked(tid));
  Trace *trace = &tctx->trace;
  // Snapshot first/last parts and the current position in the last part.
  TracePart *first_part;
  TracePart *last_part;
  Event *last_pos;
  {
    Lock lock(&trace->mtx);
    first_part = trace->parts.Front();
    if (!first_part)
      return false;
    last_part = trace->parts.Back();
    last_pos = trace->final_pos;
    if (tctx->thr)
      last_pos = (Event *)atomic_load_relaxed(&tctx->thr->trace_pos);
  }
  // Too large for stack.
  alignas(MutexSet) static char mset_storage[sizeof(MutexSet)];
  MutexSet &mset = *new (mset_storage) MutexSet();
  Vector<uptr> stack;
  uptr prev_pc = 0;
  bool found = false;
  bool is_read = typ & kAccessRead;
  bool is_atomic = typ & kAccessAtomic;
  bool is_free = typ & kAccessFree;
  TraceReplay(
      trace, last_part, last_pos, sid, epoch,
      [&](Sid ev_sid, Epoch ev_epoch, Event *evp) {
        bool match = ev_sid == sid && ev_epoch == epoch;
        if (evp->is_access) {
          if (evp->is_func == 0 && evp->type == EventType::kAccessExt &&
              evp->_ == 0)  // NopEvent
            return;
          auto *ev = reinterpret_cast<EventAccess *>(evp);
          uptr ev_addr = RestoreAddr(ev->addr);
          uptr ev_size = 1 << ev->size_log;
          uptr ev_pc =
              prev_pc + ev->pc_delta - (1 << (EventAccess::kPCBits - 1));
          prev_pc = ev_pc;
          DPrintf2("  Access: pc=0x%zx addr=0x%llx/%llu type=%llu/%llu\n",
                   ev_pc, ev_addr, ev_size, ev->is_read, ev->is_atomic);
          if (match && type == EventType::kAccessExt &&
              IsWithinAccess(addr, size, ev_addr, ev_size) &&
              is_read == ev->is_read && is_atomic == ev->is_atomic && !is_free)
            RestoreStackMatch(pstk, pmset, &stack, &mset, ev_pc, &found);
          return;
        }
        if (evp->is_func) {
          auto *ev = reinterpret_cast<EventFunc *>(evp);
          if (ev->pc) {
            DPrintf2("  FuncEnter: pc=0x%zx\n", ev->pc);
            stack.PushBack(ev->pc);
          } else {
            DPrintf2("  FuncExit\n");
            CHECK(stack.Size());
            stack.PopBack();
          }
          return;
        }
        switch (evp->type) {
          case EventType::kAccessExt: {
            auto *ev = reinterpret_cast<EventAccessExt *>(evp);
            uptr ev_addr = RestoreAddr(ev->addr);
            uptr ev_size = 1 << ev->size_log;
            prev_pc = ev->pc;
            DPrintf2("  AccessExt: pc=0x%zx addr=0x%llx/%llu type=%llu/%llu\n",
                     ev->pc, ev_addr, ev_size, ev->is_read, ev->is_atomic);
            if (match && type == EventType::kAccessExt &&
                IsWithinAccess(addr, size, ev_addr, ev_size) &&
                is_read == ev->is_read && is_atomic == ev->is_atomic &&
                !is_free)
              RestoreStackMatch(pstk, pmset, &stack, &mset, ev->pc, &found);
            break;
          }
          case EventType::kAccessRange: {
            auto *ev = reinterpret_cast<EventAccessRange *>(evp);
            uptr ev_addr = RestoreAddr(ev->addr);
            uptr ev_size =
                (ev->size_hi << EventAccessRange::kSizeLoBits) + ev->size_lo;
            uptr ev_pc = RestoreAddr(ev->pc);
            prev_pc = ev_pc;
            DPrintf2("  Range: pc=0x%zx addr=0x%llx/%llu type=%llu/%llu\n",
                     ev_pc, ev_addr, ev_size, ev->is_read, ev->is_free);
            if (match && type == EventType::kAccessExt &&
                IsWithinAccess(addr, size, ev_addr, ev_size) &&
                is_read == ev->is_read && !is_atomic && is_free == ev->is_free)
              RestoreStackMatch(pstk, pmset, &stack, &mset, ev_pc, &found);
            break;
          }
          case EventType::kLock:
            FALLTHROUGH;
          case EventType::kRLock: {
            auto *ev = reinterpret_cast<EventLock *>(evp);
            bool is_write = ev->type == EventType::kLock;
            uptr ev_addr = RestoreAddr(ev->addr);
            uptr ev_pc = RestoreAddr(ev->pc);
            StackID stack_id =
                (ev->stack_hi << EventLock::kStackIDLoBits) + ev->stack_lo;
            DPrintf2("  Lock: pc=0x%zx addr=0x%llx stack=%u write=%d\n", ev_pc,
                     ev_addr, stack_id, is_write);
            mset.AddAddr(ev_addr, stack_id, is_write);
            // Events with ev_pc == 0 are written to the beginning of trace
            // part as initial mutex set (are not real).
            if (match && type == EventType::kLock && addr == ev_addr && ev_pc)
              RestoreStackMatch(pstk, pmset, &stack, &mset, ev_pc, &found);
            break;
          }
          case EventType::kUnlock: {
            auto *ev = reinterpret_cast<EventUnlock *>(evp);
            uptr ev_addr = RestoreAddr(ev->addr);
            DPrintf2("  Unlock: addr=0x%llx\n", ev_addr);
            mset.DelAddr(ev_addr);
            break;
          }
          case EventType::kTime:
            // TraceReplay already extracted sid/epoch from it,
            // nothing else to do here.
            break;
        }
      });
  ExtractTagFromStack(pstk, ptag);
  return found;
}

}  // namespace v3

static bool FindRacyStacks(const RacyStacks &hash) {
  for (uptr i = 0; i < ctx->racy_stacks.Size(); i++) {
    if (hash == ctx->racy_stacks[i]) {
      VPrintf(2, "ThreadSanitizer: suppressing report as doubled (stack)\n");
      return true;
    }
  }
  return false;
}

static bool HandleRacyStacks(ThreadState *thr, VarSizeStackTrace traces[2]) {
  if (!flags()->suppress_equal_stacks)
    return false;
  RacyStacks hash;
  hash.hash[0] = md5_hash(traces[0].trace, traces[0].size * sizeof(uptr));
  hash.hash[1] = md5_hash(traces[1].trace, traces[1].size * sizeof(uptr));
  {
    ReadLock lock(&ctx->racy_mtx);
    if (FindRacyStacks(hash))
      return true;
  }
  Lock lock(&ctx->racy_mtx);
  if (FindRacyStacks(hash))
    return true;
  ctx->racy_stacks.PushBack(hash);
  return false;
}

static bool FindRacyAddress(const RacyAddress &ra0) {
  for (uptr i = 0; i < ctx->racy_addresses.Size(); i++) {
    RacyAddress ra2 = ctx->racy_addresses[i];
    uptr maxbeg = max(ra0.addr_min, ra2.addr_min);
    uptr minend = min(ra0.addr_max, ra2.addr_max);
    if (maxbeg < minend) {
      VPrintf(2, "ThreadSanitizer: suppressing report as doubled (addr)\n");
      return true;
    }
  }
  return false;
}

static bool HandleRacyAddress(ThreadState *thr, uptr addr_min, uptr addr_max) {
  if (!flags()->suppress_equal_addresses)
    return false;
  RacyAddress ra0 = {addr_min, addr_max};
  {
    ReadLock lock(&ctx->racy_mtx);
    if (FindRacyAddress(ra0))
      return true;
  }
  Lock lock(&ctx->racy_mtx);
  if (FindRacyAddress(ra0))
    return true;
  ctx->racy_addresses.PushBack(ra0);
  return false;
}

bool OutputReport(ThreadState *thr, const ScopedReport &srep) {
  // These should have been checked in ShouldReport.
  // It's too late to check them here, we have already taken locks.
  CHECK(flags()->report_bugs);
  CHECK(!thr->suppress_reports);
  atomic_store_relaxed(&ctx->last_symbolize_time_ns, NanoTime());
  const ReportDesc *rep = srep.GetReport();
  CHECK_EQ(thr->current_report, nullptr);
  thr->current_report = rep;
  Suppression *supp = 0;
  uptr pc_or_addr = 0;
  for (uptr i = 0; pc_or_addr == 0 && i < rep->mops.Size(); i++)
    pc_or_addr = IsSuppressed(rep->typ, rep->mops[i]->stack, &supp);
  for (uptr i = 0; pc_or_addr == 0 && i < rep->stacks.Size(); i++)
    pc_or_addr = IsSuppressed(rep->typ, rep->stacks[i], &supp);
  for (uptr i = 0; pc_or_addr == 0 && i < rep->threads.Size(); i++)
    pc_or_addr = IsSuppressed(rep->typ, rep->threads[i]->stack, &supp);
  for (uptr i = 0; pc_or_addr == 0 && i < rep->locs.Size(); i++)
    pc_or_addr = IsSuppressed(rep->typ, rep->locs[i], &supp);
  if (pc_or_addr != 0) {
    Lock lock(&ctx->fired_suppressions_mtx);
    FiredSuppression s = {srep.GetReport()->typ, pc_or_addr, supp};
    ctx->fired_suppressions.push_back(s);
  }
  {
    bool old_is_freeing = thr->is_freeing;
    thr->is_freeing = false;
    bool suppressed = OnReport(rep, pc_or_addr != 0);
    thr->is_freeing = old_is_freeing;
    if (suppressed) {
      thr->current_report = nullptr;
      return false;
    }
  }
  PrintReport(rep);
  __tsan_on_report(rep);
  ctx->nreported++;
  if (flags()->halt_on_error)
    Die();
  thr->current_report = nullptr;
  return true;
}

bool IsFiredSuppression(Context *ctx, ReportType type, StackTrace trace) {
  ReadLock lock(&ctx->fired_suppressions_mtx);
  for (uptr k = 0; k < ctx->fired_suppressions.size(); k++) {
    if (ctx->fired_suppressions[k].type != type)
      continue;
    for (uptr j = 0; j < trace.size; j++) {
      FiredSuppression *s = &ctx->fired_suppressions[k];
      if (trace.trace[j] == s->pc_or_addr) {
        if (s->supp)
          atomic_fetch_add(&s->supp->hit_count, 1, memory_order_relaxed);
        return true;
      }
    }
  }
  return false;
}

static bool IsFiredSuppression(Context *ctx, ReportType type, uptr addr) {
  ReadLock lock(&ctx->fired_suppressions_mtx);
  for (uptr k = 0; k < ctx->fired_suppressions.size(); k++) {
    if (ctx->fired_suppressions[k].type != type)
      continue;
    FiredSuppression *s = &ctx->fired_suppressions[k];
    if (addr == s->pc_or_addr) {
      if (s->supp)
        atomic_fetch_add(&s->supp->hit_count, 1, memory_order_relaxed);
      return true;
    }
  }
  return false;
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
  CheckedMutex::CheckNoLocks();

  // Symbolizer makes lots of intercepted calls. If we try to process them,
  // at best it will cause deadlocks on internal mutexes.
  ScopedIgnoreInterceptors ignore;

  if (!ShouldReport(thr, ReportTypeRace))
    return;
  if (!flags()->report_atomic_races && !RaceBetweenAtomicAndFree(thr))
    return;

  bool freed = false;
  {
    Shadow s(thr->racy_state[1]);
    freed = s.GetFreedAndReset();
    thr->racy_state[1] = s.raw();
  }

  uptr addr = ShadowToMem(thr->racy_shadow_addr);
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
  if (HandleRacyAddress(thr, addr_min, addr_max))
    return;

  ReportType typ = ReportTypeRace;
  if (thr->is_vptr_access && freed)
    typ = ReportTypeVptrUseAfterFree;
  else if (thr->is_vptr_access)
    typ = ReportTypeVptrRace;
  else if (freed)
    typ = ReportTypeUseAfterFree;

  if (IsFiredSuppression(ctx, typ, addr))
    return;

  const uptr kMop = 2;
  VarSizeStackTrace traces[kMop];
  uptr tags[kMop] = {kExternalTagNone};
  uptr toppc = TraceTopPC(thr);
  if (toppc >> kEventPCBits) {
    // This is a work-around for a known issue.
    // The scenario where this happens is rather elaborate and requires
    // an instrumented __sanitizer_report_error_summary callback and
    // a __tsan_symbolize_external callback and a race during a range memory
    // access larger than 8 bytes. MemoryAccessRange adds the current PC to
    // the trace and starts processing memory accesses. A first memory access
    // triggers a race, we report it and call the instrumented
    // __sanitizer_report_error_summary, which adds more stuff to the trace
    // since it is intrumented. Then a second memory access in MemoryAccessRange
    // also triggers a race and we get here and call TraceTopPC to get the
    // current PC, however now it contains some unrelated events from the
    // callback. Most likely, TraceTopPC will now return a EventTypeFuncExit
    // event. Later we subtract -1 from it (in GetPreviousInstructionPc)
    // and the resulting PC has kExternalPCBit set, so we pass it to
    // __tsan_symbolize_external_ex. __tsan_symbolize_external_ex is within its
    // rights to crash since the PC is completely bogus.
    // test/tsan/double_race.cpp contains a test case for this.
    toppc = 0;
  }
  ObtainCurrentStack(thr, toppc, &traces[0], &tags[0]);
  if (IsFiredSuppression(ctx, typ, traces[0]))
    return;

  // MutexSet is too large to live on stack.
  Vector<u64> mset_buffer;
  mset_buffer.Resize(sizeof(MutexSet) / sizeof(u64) + 1);
  MutexSet *mset2 = new(&mset_buffer[0]) MutexSet();

  Shadow s2(thr->racy_state[1]);
  RestoreStack(s2.tid(), s2.epoch(), &traces[1], mset2, &tags[1]);
  if (IsFiredSuppression(ctx, typ, traces[1]))
    return;

  if (HandleRacyStacks(thr, traces))
    return;

  // If any of the accesses has a tag, treat this as an "external" race.
  uptr tag = kExternalTagNone;
  for (uptr i = 0; i < kMop; i++) {
    if (tags[i] != kExternalTagNone) {
      typ = ReportTypeExternalRace;
      tag = tags[i];
      break;
    }
  }

  ThreadRegistryLock l0(&ctx->thread_registry);
  ScopedReport rep(typ, tag);
  for (uptr i = 0; i < kMop; i++) {
    Shadow s(thr->racy_state[i]);
    rep.AddMemoryAccess(addr, tags[i], s, traces[i],
                        i == 0 ? &thr->mset : mset2);
  }

  for (uptr i = 0; i < kMop; i++) {
    FastState s(thr->racy_state[i]);
    ThreadContext *tctx = static_cast<ThreadContext *>(
        ctx->thread_registry.GetThreadLocked(s.tid()));
    if (s.epoch() < tctx->epoch0 || s.epoch() > tctx->epoch1)
      continue;
    rep.AddThread(tctx);
  }

  rep.AddLocation(addr_min, addr_max - addr_min);

#if !SANITIZER_GO
  {
    Shadow s(thr->racy_state[1]);
    if (s.epoch() <= thr->last_sleep_clock.get(s.tid()))
      rep.AddSleep(thr->last_sleep_stack_id);
  }
#endif

  OutputReport(thr, rep);
}

void PrintCurrentStack(ThreadState *thr, uptr pc) {
  VarSizeStackTrace trace;
  ObtainCurrentStack(thr, pc, &trace);
  PrintStack(SymbolizeStack(trace));
}

// Always inlining PrintCurrentStackSlow, because LocatePcInTrace assumes
// __sanitizer_print_stack_trace exists in the actual unwinded stack, but
// tail-call to PrintCurrentStackSlow breaks this assumption because
// __sanitizer_print_stack_trace disappears after tail-call.
// However, this solution is not reliable enough, please see dvyukov's comment
// http://reviews.llvm.org/D19148#406208
// Also see PR27280 comment 2 and 3 for breaking examples and analysis.
ALWAYS_INLINE USED void PrintCurrentStackSlow(uptr pc) {
#if !SANITIZER_GO
  uptr bp = GET_CURRENT_FRAME();
  auto *ptrace = New<BufferedStackTrace>();
  ptrace->Unwind(pc, bp, nullptr, false);

  for (uptr i = 0; i < ptrace->size / 2; i++) {
    uptr tmp = ptrace->trace_buffer[i];
    ptrace->trace_buffer[i] = ptrace->trace_buffer[ptrace->size - i - 1];
    ptrace->trace_buffer[ptrace->size - i - 1] = tmp;
  }
  PrintStack(SymbolizeStack(*ptrace));
#endif
}

}  // namespace __tsan

using namespace __tsan;

extern "C" {
SANITIZER_INTERFACE_ATTRIBUTE
void __sanitizer_print_stack_trace() {
  PrintCurrentStackSlow(StackTrace::GetCurrentPc());
}
}  // extern "C"
