//===-- Timer.cpp - Interval Timing Support -------------------------------===//
//
// Interval Timing implementation.
//
//===----------------------------------------------------------------------===//

#include "Support/Timer.h"
#include <sys/resource.h>
#include <sys/time.h>
#include <sys/unistd.h>
#include <unistd.h>
#include <malloc.h>
#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <functional>

static TimerGroup *DefaultTimerGroup = 0;
static TimerGroup *getDefaultTimerGroup() {
  if (DefaultTimerGroup) return DefaultTimerGroup;
  return DefaultTimerGroup = new TimerGroup("Miscellaneous Ungrouped Timers");
}

Timer::Timer(const std::string &N)
  : Elapsed(0), UserTime(0), SystemTime(0), MemUsed(0), Name(N),
    Started(false), TG(getDefaultTimerGroup()) {
  TG->addTimer();
}

Timer::Timer(const std::string &N, TimerGroup &tg)
  : Elapsed(0), UserTime(0), SystemTime(0), MemUsed(0), Name(N),
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

struct TimeRecord {
  double Elapsed, UserTime, SystemTime;
  long MemUsed;
};

static TimeRecord getTimeRecord() {
  struct rusage RU;
  struct timeval T;
  gettimeofday(&T, 0);
  if (getrusage(RUSAGE_SELF, &RU)) {
    perror("getrusage call failed: -time-passes info incorrect!");
  }

  TimeRecord Result;
  Result.Elapsed    =           T.tv_sec +           T.tv_usec/1000000.0;
  Result.UserTime   = RU.ru_utime.tv_sec + RU.ru_utime.tv_usec/1000000.0;
  Result.SystemTime = RU.ru_stime.tv_sec + RU.ru_stime.tv_usec/1000000.0;

#ifndef __sparc__
  struct mallinfo MI = mallinfo();
  Result.MemUsed     = MI.uordblks;
#else
  Result.MemUsed     = 0;
#endif

  return Result;
}

void Timer::startTimer() {
  Started = true;
  TimeRecord TR = getTimeRecord();
  Elapsed    -= TR.Elapsed;
  UserTime   -= TR.UserTime;
  SystemTime -= TR.SystemTime;
  MemUsed    -= TR.MemUsed;
}

void Timer::stopTimer() {
  TimeRecord TR = getTimeRecord();
  Elapsed    += TR.Elapsed;
  UserTime   += TR.UserTime;
  SystemTime += TR.SystemTime;
  MemUsed    += TR.MemUsed;
}

void Timer::sum(const Timer &T) {
  Elapsed    += T.Elapsed;
  UserTime   += T.UserTime;
  SystemTime += T.SystemTime;
  MemUsed    += T.MemUsed;
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
                << "===\n  Total Execution Time: " << Total.getProcessTime()
                << " seconds (" << Total.getWallTime()
                << " wall clock)\n\n";

      if (Total.UserTime)
        std::cerr << "   ---User Time---";
      if (Total.SystemTime)
        std::cerr << "   --System Time--";
      if (Total.getProcessTime())
        std::cerr << "   --User+System--";
      std::cerr << "   ---Wall Time---";
      if (Total.getMemUsed())
        std::cerr << "  ---Mem---";
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
