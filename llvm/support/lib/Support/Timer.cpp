//===-- Timer.cpp - Interval Timing Support -------------------------------===//
//
// Interval Timing implementation.
//
//===----------------------------------------------------------------------===//

#include "Support/Timer.h"
#include "Support/CommandLine.h"
#include <sys/resource.h>
#include <sys/time.h>
#include <sys/unistd.h>
#include <unistd.h>
#include <malloc.h>
#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <functional>

namespace {
  cl::opt<bool>
  TrackSpace("track-memory", cl::desc("Enable -time-passes memory "
                                      "tracking (this may be slow)"),
             cl::Hidden);
}

static TimerGroup *DefaultTimerGroup = 0;
static TimerGroup *getDefaultTimerGroup() {
  if (DefaultTimerGroup) return DefaultTimerGroup;
  return DefaultTimerGroup = new TimerGroup("Miscellaneous Ungrouped Timers");
}

Timer::Timer(const std::string &N)
  : Elapsed(0), UserTime(0), SystemTime(0), MemUsed(0), PeakMem(0), Name(N),
    Started(false), TG(getDefaultTimerGroup()) {
  TG->addTimer();
}

Timer::Timer(const std::string &N, TimerGroup &tg)
  : Elapsed(0), UserTime(0), SystemTime(0), MemUsed(0), PeakMem(0), Name(N),
    Started(false), TG(&tg) {
  TG->addTimer();
}

Timer::Timer(const Timer &T) {
  TG = T.TG;
  if (TG) TG->addTimer();
  operator=(T);
}


// Copy ctor, initialize with no TG member.
Timer::Timer(bool, const Timer &T) {
  TG = T.TG;     // Avoid assertion in operator=
  operator=(T);  // Copy contents
  TG = 0;
}


Timer::~Timer() {
  if (TG) {
    if (Started) {
      Started = false;
      TG->addTimerToPrint(*this);
    }
    TG->removeTimer();
  }
}

static long getMemUsage() {
  if (TrackSpace) {
    struct mallinfo MI = mallinfo();
    return MI.uordblks/*+MI.hblkhd*/;
  } else {
    return 0;
  }
}

struct TimeRecord {
  double Elapsed, UserTime, SystemTime;
  long MemUsed;
};

static TimeRecord getTimeRecord(bool Start) {
  struct rusage RU;
  struct timeval T;
  long MemUsed = 0;
  if (Start) {
    MemUsed = getMemUsage();
    if (getrusage(RUSAGE_SELF, &RU))
      perror("getrusage call failed: -time-passes info incorrect!");
  }
  gettimeofday(&T, 0);

  if (!Start) {
    MemUsed = getMemUsage();
    if (getrusage(RUSAGE_SELF, &RU))
      perror("getrusage call failed: -time-passes info incorrect!");
  }

  TimeRecord Result;
  Result.Elapsed    =           T.tv_sec +           T.tv_usec/1000000.0;
  Result.UserTime   = RU.ru_utime.tv_sec + RU.ru_utime.tv_usec/1000000.0;
  Result.SystemTime = RU.ru_stime.tv_sec + RU.ru_stime.tv_usec/1000000.0;
  Result.MemUsed = MemUsed;

  return Result;
}

static std::vector<Timer*> ActiveTimers;

void Timer::startTimer() {
  Started = true;
  TimeRecord TR = getTimeRecord(true);
  Elapsed    -= TR.Elapsed;
  UserTime   -= TR.UserTime;
  SystemTime -= TR.SystemTime;
  MemUsed    -= TR.MemUsed;
  PeakMemBase = TR.MemUsed;
  ActiveTimers.push_back(this);
}

void Timer::stopTimer() {
  TimeRecord TR = getTimeRecord(false);
  Elapsed    += TR.Elapsed;
  UserTime   += TR.UserTime;
  SystemTime += TR.SystemTime;
  MemUsed    += TR.MemUsed;

  if (ActiveTimers.back() == this) {
    ActiveTimers.pop_back();
  } else {
    std::vector<Timer*>::iterator I =
      std::find(ActiveTimers.begin(), ActiveTimers.end(), this);
    assert(I != ActiveTimers.end() && "stop but no startTimer?");
    ActiveTimers.erase(I);
  }
}

void Timer::sum(const Timer &T) {
  Elapsed    += T.Elapsed;
  UserTime   += T.UserTime;
  SystemTime += T.SystemTime;
  MemUsed    += T.MemUsed;
  PeakMem    += T.PeakMem;
}

/// addPeakMemoryMeasurement - This method should be called whenever memory
/// usage needs to be checked.  It adds a peak memory measurement to the
/// currently active timers, which will be printed when the timer group prints
///
void Timer::addPeakMemoryMeasurement() {
  long MemUsed = getMemUsage();

  for (std::vector<Timer*>::iterator I = ActiveTimers.begin(),
         E = ActiveTimers.end(); I != E; ++I)
    (*I)->PeakMem = std::max((*I)->PeakMem, MemUsed-(*I)->PeakMemBase);
}


//===----------------------------------------------------------------------===//
//   TimerGroup Implementation
//===----------------------------------------------------------------------===//

static void printVal(double Val, double Total) {
  if (Total < 1e-7)   // Avoid dividing by zero...
    fprintf(stderr, "        -----     ");
  else
    fprintf(stderr, "  %7.4f (%5.1f%%)", Val, Val*100/Total);
}

void Timer::print(const Timer &Total) {
  if (Total.UserTime)
    printVal(UserTime, Total.UserTime);
  if (Total.SystemTime)
    printVal(SystemTime, Total.SystemTime);
  if (Total.getProcessTime())
    printVal(getProcessTime(), Total.getProcessTime());
  printVal(Elapsed, Total.Elapsed);
  
  fprintf(stderr, "  ");

  if (Total.MemUsed)
    fprintf(stderr, " %8ld  ", MemUsed);
  if (Total.PeakMem) {
    if (PeakMem)
      fprintf(stderr, " %8ld  ", PeakMem);
    else
      fprintf(stderr, "           ");
  }
  std::cerr << Name << "\n";

  Started = false;  // Once printed, don't print again
}


void TimerGroup::removeTimer() {
  if (--NumTimers == 0 && !TimersToPrint.empty()) { // Print timing report...
    // Sort the timers in descending order by amount of time taken...
    std::sort(TimersToPrint.begin(), TimersToPrint.end(),
              std::greater<Timer>());

    // Figure out how many spaces to indent TimerGroup name...
    unsigned Padding = (80-Name.length())/2;
    if (Padding > 80) Padding = 0;         // Don't allow "negative" numbers

    ++NumTimers;
    {  // Scope to contain Total timer... don't allow total timer to drop us to
       // zero timers...
      Timer Total("TOTAL");
  
      for (unsigned i = 0, e = TimersToPrint.size(); i != e; ++i)
        Total.sum(TimersToPrint[i]);
      
      // Print out timing header...
      std::cerr << "===" << std::string(73, '-') << "===\n"
                << std::string(Padding, ' ') << Name << "\n"
                << "===" << std::string(73, '-')
                << "===\n  Total Execution Time: ";

      // Hack for GCC 2.96... :( it doesn't support manipulators!
      fprintf(stderr, "%.4f seconds (%.4f wall clock)\n\n",
              Total.getProcessTime(), Total.getWallTime());

      if (Total.UserTime)
        std::cerr << "   ---User Time---";
      if (Total.SystemTime)
        std::cerr << "   --System Time--";
      if (Total.getProcessTime())
        std::cerr << "   --User+System--";
      std::cerr << "   ---Wall Time---";
      if (Total.getMemUsed())
        std::cerr << "  ---Mem---";
      if (Total.getPeakMem())
        std::cerr << "  -PeakMem-";
      std::cerr << "  --- Name ---\n";
      
      // Loop through all of the timing data, printing it out...
      for (unsigned i = 0, e = TimersToPrint.size(); i != e; ++i)
        TimersToPrint[i].print(Total);
    
      Total.print(Total);
      std::cerr << std::endl;  // Flush output
    }
    --NumTimers;

    TimersToPrint.clear();
  }

  // Delete default timer group!
  if (NumTimers == 0 && this == DefaultTimerGroup) {
    delete DefaultTimerGroup;
    DefaultTimerGroup = 0;
  }
}
