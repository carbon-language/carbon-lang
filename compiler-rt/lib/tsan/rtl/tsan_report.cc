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
  TsanPrintf("WARNING: ThreadSanitizer: ");

  if (typ == ReportTypeRace)
    TsanPrintf("data race");
  else if (typ == ReportTypeUseAfterFree)
    TsanPrintf("heap-use-after-free");
  else if (typ == ReportTypeThreadLeak)
    TsanPrintf("thread leak");
  else if (typ == ReportTypeMutexDestroyLocked)
    TsanPrintf("destroy of a locked mutex");
  else if (typ == ReportTypeSignalUnsafe)
    TsanPrintf("signal-unsafe call inside of a signal");
  else if (typ == ReportTypeErrnoInSignal)
    TsanPrintf("signal handler spoils errno");

  TsanPrintf(" (pid=%d)\n", GetPid());
}

void PrintStack(const ReportStack *ent) {
  for (int i = 0; ent; ent = ent->next, i++) {
    TsanPrintf("    #%d %s %s:%d", i, ent->func, ent->file, ent->line);
    if (ent->col)
      TsanPrintf(":%d", ent->col);
    if (ent->module && ent->offset)
      TsanPrintf(" (%s+%p)\n", ent->module, (void*)ent->offset);
    else
      TsanPrintf(" (%p)\n", (void*)ent->pc);
  }
  TsanPrintf("\n");
}

static void PrintMop(const ReportMop *mop, bool first) {
  TsanPrintf("  %s of size %d at %p",
      (first ? (mop->write ? "Write" : "Read")
             : (mop->write ? "Previous write" : "Previous read")),
      mop->size, (void*)mop->addr);
  if (mop->tid == 0)
    TsanPrintf(" by main thread:\n");
  else
    TsanPrintf(" by thread %d:\n", mop->tid);
  PrintStack(mop->stack);
}

static void PrintLocation(const ReportLocation *loc) {
  if (loc->type == ReportLocationGlobal) {
    TsanPrintf("  Location is global '%s' of size %zu at %zx %s:%d\n",
               loc->name, loc->size, loc->addr, loc->file, loc->line);
  } else if (loc->type == ReportLocationHeap) {
    TsanPrintf("  Location is heap block of size %zu at %p allocated",
        loc->size, loc->addr);
    if (loc->tid == 0)
      TsanPrintf(" by main thread:\n");
    else
      TsanPrintf(" by thread %d:\n", loc->tid);
    PrintStack(loc->stack);
  } else if (loc->type == ReportLocationStack) {
    TsanPrintf("  Location is stack of thread %d:\n", loc->tid);
  }
}

static void PrintMutex(const ReportMutex *rm) {
  if (rm->stack == 0)
    return;
  TsanPrintf("  Mutex %d created at:\n", rm->id);
  PrintStack(rm->stack);
}

static void PrintThread(const ReportThread *rt) {
  if (rt->id == 0)  // Little sense in describing the main thread.
    return;
  TsanPrintf("  Thread %d", rt->id);
  if (rt->name)
    TsanPrintf(" '%s'", rt->name);
  TsanPrintf(" (tid=%zu, %s)", rt->pid, rt->running ? "running" : "finished");
  if (rt->stack)
    TsanPrintf(" created at:");
  TsanPrintf("\n");
  PrintStack(rt->stack);
}

static void PrintSleep(const ReportStack *s) {
  TsanPrintf("  As if synchronized via sleep:\n");
  PrintStack(s);
}

void PrintReport(const ReportDesc *rep) {
  TsanPrintf("==================\n");
  PrintHeader(rep->typ);

  for (uptr i = 0; i < rep->stacks.Size(); i++) {
    if (i)
      TsanPrintf("  and:\n");
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

  TsanPrintf("==================\n");
}

#else

void PrintStack(const ReportStack *ent) {
  for (int i = 0; ent; ent = ent->next, i++) {
    TsanPrintf("  %s()\n      %s:%d +0x%zx\n",
        ent->func, ent->file, ent->line, (void*)ent->offset);
  }
  TsanPrintf("\n");
}

static void PrintMop(const ReportMop *mop, bool first) {
  TsanPrintf("%s by goroutine %d:\n",
      (first ? (mop->write ? "Write" : "Read")
             : (mop->write ? "Previous write" : "Previous read")),
      mop->tid);
  PrintStack(mop->stack);
}

static void PrintThread(const ReportThread *rt) {
  if (rt->id == 0)  // Little sense in describing the main thread.
    return;
  TsanPrintf("Goroutine %d (%s) created at:\n",
    rt->id, rt->running ? "running" : "finished");
  PrintStack(rt->stack);
}

void PrintReport(const ReportDesc *rep) {
  TsanPrintf("==================\n");
  TsanPrintf("WARNING: DATA RACE\n");
  for (uptr i = 0; i < rep->mops.Size(); i++)
    PrintMop(rep->mops[i], i == 0);
  for (uptr i = 0; i < rep->threads.Size(); i++)
    PrintThread(rep->threads[i]);
  TsanPrintf("==================\n");
}

#endif

}  // namespace __tsan
