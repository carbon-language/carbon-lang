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

ReportDesc::~ReportDesc() {
}

#ifndef TSAN_GO

static void PrintHeader(ReportType typ) {
  Printf("WARNING: ThreadSanitizer: ");

  if (typ == ReportTypeRace)
    Printf("data race");
  else if (typ == ReportTypeUseAfterFree)
    Printf("heap-use-after-free");
  else if (typ == ReportTypeThreadLeak)
    Printf("thread leak");
  else if (typ == ReportTypeMutexDestroyLocked)
    Printf("destroy of a locked mutex");
  else if (typ == ReportTypeSignalUnsafe)
    Printf("signal-unsafe call inside of a signal");
  else if (typ == ReportTypeErrnoInSignal)
    Printf("signal handler spoils errno");

  Printf(" (pid=%d)\n", GetPid());
}

void PrintStack(const ReportStack *ent) {
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

static void PrintMop(const ReportMop *mop, bool first) {
  Printf("  %s of size %d at %p",
      (first ? (mop->write ? "Write" : "Read")
             : (mop->write ? "Previous write" : "Previous read")),
      mop->size, (void*)mop->addr);
  if (mop->tid == 0)
    Printf(" by main thread:\n");
  else
    Printf(" by thread %d:\n", mop->tid);
  PrintStack(mop->stack);
}

static void PrintLocation(const ReportLocation *loc) {
  if (loc->type == ReportLocationGlobal) {
    Printf("  Location is global '%s' of size %zu at %zx %s:%d\n",
               loc->name, loc->size, loc->addr, loc->file, loc->line);
  } else if (loc->type == ReportLocationHeap) {
    Printf("  Location is heap block of size %zu at %p allocated",
        loc->size, loc->addr);
    if (loc->tid == 0)
      Printf(" by main thread:\n");
    else
      Printf(" by thread %d:\n", loc->tid);
    PrintStack(loc->stack);
  } else if (loc->type == ReportLocationStack) {
    Printf("  Location is stack of thread %d:\n", loc->tid);
  }
}

static void PrintMutex(const ReportMutex *rm) {
  if (rm->stack == 0)
    return;
  Printf("  Mutex %d created at:\n", rm->id);
  PrintStack(rm->stack);
}

static void PrintThread(const ReportThread *rt) {
  if (rt->id == 0)  // Little sense in describing the main thread.
    return;
  Printf("  Thread %d", rt->id);
  if (rt->name)
    Printf(" '%s'", rt->name);
  Printf(" (tid=%zu, %s)", rt->pid, rt->running ? "running" : "finished");
  if (rt->stack)
    Printf(" created at:");
  Printf("\n");
  PrintStack(rt->stack);
}

static void PrintSleep(const ReportStack *s) {
  Printf("  As if synchronized via sleep:\n");
  PrintStack(s);
}

void PrintReport(const ReportDesc *rep) {
  Printf("==================\n");
  PrintHeader(rep->typ);

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

  Printf("==================\n");
}

#else

void PrintStack(const ReportStack *ent) {
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
