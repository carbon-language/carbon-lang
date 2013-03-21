//===-- tsan_report.cc ----------------------------------------------------===//
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
#include "tsan_report.h"
#include "tsan_platform.h"
#include "tsan_rtl.h"

namespace __tsan {

ReportDesc::ReportDesc()
    : stacks(MBlockReportStack)
    , mops(MBlockReportMop)
    , locs(MBlockReportLoc)
    , mutexes(MBlockReportMutex)
    , threads(MBlockReportThread)
    , sleep() {
}

ReportMop::ReportMop()
    : mset(MBlockReportMutex) {
}

ReportDesc::~ReportDesc() {
  // FIXME(dvyukov): it must be leaking a lot of memory.
}

#ifndef TSAN_GO

const int kThreadBufSize = 32;
const char *thread_name(char *buf, int tid) {
  if (tid == 0)
    return "main thread";
  internal_snprintf(buf, kThreadBufSize, "thread T%d", tid);
  return buf;
}

static const char *ReportTypeString(ReportType typ) {
  if (typ == ReportTypeRace)
    return "data race";
  if (typ == ReportTypeVptrRace)
    return "data race on vptr (ctor/dtor vs virtual call)";
  if (typ == ReportTypeUseAfterFree)
    return "heap-use-after-free";
  if (typ == ReportTypeThreadLeak)
    return "thread leak";
  if (typ == ReportTypeMutexDestroyLocked)
    return "destroy of a locked mutex";
  if (typ == ReportTypeSignalUnsafe)
    return "signal-unsafe call inside of a signal";
  if (typ == ReportTypeErrnoInSignal)
    return "signal handler spoils errno";
  return "";
}

void PrintStack(const ReportStack *ent) {
  if (ent == 0) {
    Printf("    [failed to restore the stack]\n\n");
    return;
  }
  for (int i = 0; ent; ent = ent->next, i++) {
    Printf("    #%d %s %s:%d", i, ent->func, ent->file, ent->line);
    if (ent->col)
      Printf(":%d", ent->col);
    if (ent->module && ent->offset)
      Printf(" (%s+%p)\n", ent->module, (void*)ent->offset);
    else
      Printf(" (%p)\n", (void*)ent->pc);
  }
  Printf("\n");
}

static void PrintMutexSet(Vector<ReportMopMutex> const& mset) {
  for (uptr i = 0; i < mset.Size(); i++) {
    if (i == 0)
      Printf(" (mutexes:");
    const ReportMopMutex m = mset[i];
    Printf(" %s M%llu", m.write ? "write" : "read", m.id);
    Printf(i == mset.Size() - 1 ? ")" : ",");
  }
}

static const char *MopDesc(bool first, bool write, bool atomic) {
  return atomic ? (first ? (write ? "Atomic write" : "Atomic read")
                : (write ? "Previous atomic write" : "Previous atomic read"))
                : (first ? (write ? "Write" : "Read")
                : (write ? "Previous write" : "Previous read"));
}

static void PrintMop(const ReportMop *mop, bool first) {
  char thrbuf[kThreadBufSize];
  Printf("  %s of size %d at %p by %s",
      MopDesc(first, mop->write, mop->atomic),
      mop->size, (void*)mop->addr,
      thread_name(thrbuf, mop->tid));
  PrintMutexSet(mop->mset);
  Printf(":\n");
  PrintStack(mop->stack);
}

static void PrintLocation(const ReportLocation *loc) {
  char thrbuf[kThreadBufSize];
  if (loc->type == ReportLocationGlobal) {
    Printf("  Location is global '%s' of size %zu at %zx (%s+%p)\n\n",
               loc->name, loc->size, loc->addr, loc->module, loc->offset);
  } else if (loc->type == ReportLocationHeap) {
    char thrbuf[kThreadBufSize];
    Printf("  Location is heap block of size %zu at %p allocated by %s:\n",
        loc->size, loc->addr, thread_name(thrbuf, loc->tid));
    PrintStack(loc->stack);
  } else if (loc->type == ReportLocationStack) {
    Printf("  Location is stack of %s.\n\n", thread_name(thrbuf, loc->tid));
  } else if (loc->type == ReportLocationTLS) {
    Printf("  Location is TLS of %s.\n\n", thread_name(thrbuf, loc->tid));
  } else if (loc->type == ReportLocationFD) {
    Printf("  Location is file descriptor %d created by %s at:\n",
        loc->fd, thread_name(thrbuf, loc->tid));
    PrintStack(loc->stack);
  }
}

static void PrintMutex(const ReportMutex *rm) {
  if (rm->destroyed) {
    Printf("  Mutex M%llu is already destroyed.\n\n", rm->id);
  } else {
    Printf("  Mutex M%llu created at:\n", rm->id);
    PrintStack(rm->stack);
  }
}

static void PrintThread(const ReportThread *rt) {
  if (rt->id == 0)  // Little sense in describing the main thread.
    return;
  Printf("  Thread T%d", rt->id);
  if (rt->name && rt->name[0] != '\0')
    Printf(" '%s'", rt->name);
  char thrbuf[kThreadBufSize];
  Printf(" (tid=%zu, %s) created by %s",
    rt->pid, rt->running ? "running" : "finished",
    thread_name(thrbuf, rt->parent_tid));
  if (rt->stack)
    Printf(" at:");
  Printf("\n");
  PrintStack(rt->stack);
}

static void PrintSleep(const ReportStack *s) {
  Printf("  As if synchronized via sleep:\n");
  PrintStack(s);
}

static ReportStack *ChooseSummaryStack(const ReportDesc *rep) {
  if (rep->mops.Size())
    return rep->mops[0]->stack;
  if (rep->stacks.Size())
    return rep->stacks[0];
  if (rep->mutexes.Size())
    return rep->mutexes[0]->stack;
  if (rep->threads.Size())
    return rep->threads[0]->stack;
  return 0;
}

ReportStack *SkipTsanInternalFrames(ReportStack *ent) {
  while (FrameIsInternal(ent) && ent->next)
    ent = ent->next;
  return ent;
}

void PrintReport(const ReportDesc *rep) {
  Printf("==================\n");
  const char *rep_typ_str = ReportTypeString(rep->typ);
  Printf("WARNING: ThreadSanitizer: %s (pid=%d)\n", rep_typ_str, GetPid());

  for (uptr i = 0; i < rep->stacks.Size(); i++) {
    if (i)
      Printf("  and:\n");
    PrintStack(rep->stacks[i]);
  }

  for (uptr i = 0; i < rep->mops.Size(); i++)
    PrintMop(rep->mops[i], i == 0);

  if (rep->sleep)
    PrintSleep(rep->sleep);

  for (uptr i = 0; i < rep->locs.Size(); i++)
    PrintLocation(rep->locs[i]);

  for (uptr i = 0; i < rep->mutexes.Size(); i++)
    PrintMutex(rep->mutexes[i]);

  for (uptr i = 0; i < rep->threads.Size(); i++)
    PrintThread(rep->threads[i]);

  if (ReportStack *ent = SkipTsanInternalFrames(ChooseSummaryStack(rep)))
    ReportErrorSummary(rep_typ_str, ent->file, ent->line, ent->func);

  Printf("==================\n");
}

#else

void PrintStack(const ReportStack *ent) {
  if (ent == 0) {
    Printf("  [failed to restore the stack]\n\n");
    return;
  }
  for (int i = 0; ent; ent = ent->next, i++) {
    Printf("  %s()\n      %s:%d +0x%zx\n",
        ent->func, ent->file, ent->line, (void*)ent->offset);
  }
  Printf("\n");
}

static void PrintMop(const ReportMop *mop, bool first) {
  Printf("%s by goroutine %d:\n",
      (first ? (mop->write ? "Write" : "Read")
             : (mop->write ? "Previous write" : "Previous read")),
      mop->tid);
  PrintStack(mop->stack);
}

static void PrintThread(const ReportThread *rt) {
  if (rt->id == 0)  // Little sense in describing the main thread.
    return;
  Printf("Goroutine %d (%s) created at:\n",
    rt->id, rt->running ? "running" : "finished");
  PrintStack(rt->stack);
}

void PrintReport(const ReportDesc *rep) {
  Printf("==================\n");
  Printf("WARNING: DATA RACE\n");
  for (uptr i = 0; i < rep->mops.Size(); i++)
    PrintMop(rep->mops[i], i == 0);
  for (uptr i = 0; i < rep->threads.Size(); i++)
    PrintThread(rep->threads[i]);
  Printf("==================\n");
}

#endif

}  // namespace __tsan
