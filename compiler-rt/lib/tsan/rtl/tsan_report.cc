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
#include "sanitizer_common/sanitizer_report_decorator.h"

namespace __tsan {

class Decorator: public __sanitizer::SanitizerCommonDecorator {
 public:
  Decorator() : SanitizerCommonDecorator() { }
  const char *Warning()    { return Red(); }
  const char *EndWarning() { return Default(); }
  const char *Access()     { return Blue(); }
  const char *EndAccess()  { return Default(); }
  const char *ThreadDescription()    { return Cyan(); }
  const char *EndThreadDescription() { return Default(); }
  const char *Location()   { return Green(); }
  const char *EndLocation() { return Default(); }
  const char *Sleep()   { return Yellow(); }
  const char *EndSleep() { return Default(); }
  const char *Mutex()   { return Magenta(); }
  const char *EndMutex() { return Default(); }
};

ReportDesc::ReportDesc()
    : stacks(MBlockReportStack)
    , mops(MBlockReportMop)
    , locs(MBlockReportLoc)
    , mutexes(MBlockReportMutex)
    , threads(MBlockReportThread)
    , unique_tids(MBlockReportThread)
    , sleep()
    , count() {
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
  if (typ == ReportTypeMutexDoubleLock)
    return "double lock of a mutex";
  if (typ == ReportTypeMutexBadUnlock)
    return "unlock of an unlocked mutex (or by a wrong thread)";
  if (typ == ReportTypeMutexBadReadLock)
    return "read lock of a write locked mutex";
  if (typ == ReportTypeMutexBadReadUnlock)
    return "read unlock of a write locked mutex";
  if (typ == ReportTypeSignalUnsafe)
    return "signal-unsafe call inside of a signal";
  if (typ == ReportTypeErrnoInSignal)
    return "signal handler spoils errno";
  if (typ == ReportTypeDeadlock)
    return "lock-order-inversion (potential deadlock)";
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
  Decorator d;
  char thrbuf[kThreadBufSize];
  Printf("%s", d.Access());
  Printf("  %s of size %d at %p by %s",
      MopDesc(first, mop->write, mop->atomic),
      mop->size, (void*)mop->addr,
      thread_name(thrbuf, mop->tid));
  PrintMutexSet(mop->mset);
  Printf(":\n");
  Printf("%s", d.EndAccess());
  PrintStack(mop->stack);
}

static void PrintLocation(const ReportLocation *loc) {
  Decorator d;
  char thrbuf[kThreadBufSize];
  bool print_stack = false;
  Printf("%s", d.Location());
  if (loc->type == ReportLocationGlobal) {
    Printf("  Location is global '%s' of size %zu at %p (%s+%p)\n\n",
               loc->name, loc->size, loc->addr, loc->module, loc->offset);
  } else if (loc->type == ReportLocationHeap) {
    char thrbuf[kThreadBufSize];
    Printf("  Location is heap block of size %zu at %p allocated by %s:\n",
        loc->size, loc->addr, thread_name(thrbuf, loc->tid));
    print_stack = true;
  } else if (loc->type == ReportLocationStack) {
    Printf("  Location is stack of %s.\n\n", thread_name(thrbuf, loc->tid));
  } else if (loc->type == ReportLocationTLS) {
    Printf("  Location is TLS of %s.\n\n", thread_name(thrbuf, loc->tid));
  } else if (loc->type == ReportLocationFD) {
    Printf("  Location is file descriptor %d created by %s at:\n",
        loc->fd, thread_name(thrbuf, loc->tid));
    print_stack = true;
  }
  Printf("%s", d.EndLocation());
  if (print_stack)
    PrintStack(loc->stack);
}

static void PrintMutexShort(const ReportMutex *rm, const char *after) {
  Decorator d;
  Printf("%sM%zd%s%s", d.Mutex(), rm->id, d.EndMutex(), after);
}

static void PrintMutexShortWithAddress(const ReportMutex *rm,
                                       const char *after) {
  Decorator d;
  Printf("%sM%zd (%p)%s%s", d.Mutex(), rm->id, rm->addr, d.EndMutex(), after);
}

static void PrintMutex(const ReportMutex *rm) {
  Decorator d;
  if (rm->destroyed) {
    Printf("%s", d.Mutex());
    Printf("  Mutex M%llu is already destroyed.\n\n", rm->id);
    Printf("%s", d.EndMutex());
  } else {
    Printf("%s", d.Mutex());
    Printf("  Mutex M%llu (%p) created at:\n", rm->id, rm->addr);
    Printf("%s", d.EndMutex());
    PrintStack(rm->stack);
  }
}

static void PrintThread(const ReportThread *rt) {
  Decorator d;
  if (rt->id == 0)  // Little sense in describing the main thread.
    return;
  Printf("%s", d.ThreadDescription());
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
  Printf("%s", d.EndThreadDescription());
  PrintStack(rt->stack);
}

static void PrintSleep(const ReportStack *s) {
  Decorator d;
  Printf("%s", d.Sleep());
  Printf("  As if synchronized via sleep:\n");
  Printf("%s", d.EndSleep());
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
  Decorator d;
  Printf("==================\n");
  const char *rep_typ_str = ReportTypeString(rep->typ);
  Printf("%s", d.Warning());
  Printf("WARNING: ThreadSanitizer: %s (pid=%d)\n", rep_typ_str,
         (int)internal_getpid());
  Printf("%s", d.EndWarning());

  if (rep->typ == ReportTypeDeadlock) {
    char thrbuf[kThreadBufSize];
    Printf("  Cycle in lock order graph: ");
    for (uptr i = 0; i < rep->mutexes.Size(); i++)
      PrintMutexShortWithAddress(rep->mutexes[i], " => ");
    PrintMutexShort(rep->mutexes[0], "\n\n");
    CHECK_GT(rep->mutexes.Size(), 0U);
    CHECK_EQ(rep->mutexes.Size() * (flags()->second_deadlock_stack ? 2 : 1),
             rep->stacks.Size());
    for (uptr i = 0; i < rep->mutexes.Size(); i++) {
      Printf("  Mutex ");
      PrintMutexShort(rep->mutexes[(i + 1) % rep->mutexes.Size()],
                      " acquired here while holding mutex ");
      PrintMutexShort(rep->mutexes[i], " in ");
      Printf("%s", d.ThreadDescription());
      Printf("%s:\n", thread_name(thrbuf, rep->unique_tids[i]));
      Printf("%s", d.EndThreadDescription());
      if (flags()->second_deadlock_stack) {
        PrintStack(rep->stacks[2*i]);
        Printf("  Mutex ");
        PrintMutexShort(rep->mutexes[i],
                        " previously acquired by the same thread here:\n");
        PrintStack(rep->stacks[2*i+1]);
      } else {
        PrintStack(rep->stacks[i]);
        if (i == 0)
          Printf("    Hint: use TSAN_OPTIONS=second_deadlock_stack=1 "
                 "to get more informative warning message\n\n");
      }
    }
  } else {
    for (uptr i = 0; i < rep->stacks.Size(); i++) {
      if (i)
        Printf("  and:\n");
      PrintStack(rep->stacks[i]);
    }
  }

  for (uptr i = 0; i < rep->mops.Size(); i++)
    PrintMop(rep->mops[i], i == 0);

  if (rep->sleep)
    PrintSleep(rep->sleep);

  for (uptr i = 0; i < rep->locs.Size(); i++)
    PrintLocation(rep->locs[i]);

  if (rep->typ != ReportTypeDeadlock) {
    for (uptr i = 0; i < rep->mutexes.Size(); i++)
      PrintMutex(rep->mutexes[i]);
  }

  for (uptr i = 0; i < rep->threads.Size(); i++)
    PrintThread(rep->threads[i]);

  if (rep->typ == ReportTypeThreadLeak && rep->count > 1)
    Printf("  And %d more similar thread leaks.\n\n", rep->count - 1);

  if (ReportStack *ent = SkipTsanInternalFrames(ChooseSummaryStack(rep)))
    ReportErrorSummary(rep_typ_str, ent->file, ent->line, ent->func);

  Printf("==================\n");
}

#else  // #ifndef TSAN_GO

const int kMainThreadId = 1;

void PrintStack(const ReportStack *ent) {
  if (ent == 0) {
    Printf("  [failed to restore the stack]\n");
    return;
  }
  for (int i = 0; ent; ent = ent->next, i++) {
    Printf("  %s()\n      %s:%d +0x%zx\n",
        ent->func, ent->file, ent->line, (void*)ent->offset);
  }
}

static void PrintMop(const ReportMop *mop, bool first) {
  Printf("\n");
  Printf("%s by ",
      (first ? (mop->write ? "Write" : "Read")
             : (mop->write ? "Previous write" : "Previous read")));
  if (mop->tid == kMainThreadId)
    Printf("main goroutine:\n");
  else
    Printf("goroutine %d:\n", mop->tid);
  PrintStack(mop->stack);
}

static void PrintThread(const ReportThread *rt) {
  if (rt->id == kMainThreadId)
    return;
  Printf("\n");
  Printf("Goroutine %d (%s) created at:\n",
    rt->id, rt->running ? "running" : "finished");
  PrintStack(rt->stack);
}

void PrintReport(const ReportDesc *rep) {
  Printf("==================\n");
  if (rep->typ == ReportTypeRace) {
    Printf("WARNING: DATA RACE");
    for (uptr i = 0; i < rep->mops.Size(); i++)
      PrintMop(rep->mops[i], i == 0);
    for (uptr i = 0; i < rep->threads.Size(); i++)
      PrintThread(rep->threads[i]);
  } else if (rep->typ == ReportTypeDeadlock) {
    Printf("WARNING: DEADLOCK\n");
    for (uptr i = 0; i < rep->mutexes.Size(); i++) {
      Printf("Goroutine %d lock mutex %d while holding mutex %d:\n",
          999, rep->mutexes[i]->id,
          rep->mutexes[(i+1) % rep->mutexes.Size()]->id);
      PrintStack(rep->stacks[2*i]);
      Printf("\n");
      Printf("Mutex %d was previously locked here:\n",
          rep->mutexes[(i+1) % rep->mutexes.Size()]->id);
      PrintStack(rep->stacks[2*i + 1]);
      Printf("\n");
    }
  }
  Printf("==================\n");
}

#endif

}  // namespace __tsan
