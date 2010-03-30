//===-- Timer.cpp - Interval Timing Support -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Interval Timing implementation.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Timer.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Format.h"
#include "llvm/System/Mutex.h"
#include "llvm/System/Process.h"
#include "llvm/ADT/StringMap.h"
#include <map>
using namespace llvm;

// GetLibSupportInfoOutputFile - Return a file stream to print our output on.
namespace llvm { extern raw_ostream *GetLibSupportInfoOutputFile(); }

// getLibSupportInfoOutputFilename - This ugly hack is brought to you courtesy
// of constructor/destructor ordering being unspecified by C++.  Basically the
// problem is that a Statistic object gets destroyed, which ends up calling
// 'GetLibSupportInfoOutputFile()' (below), which calls this function.
// LibSupportInfoOutputFilename used to be a global variable, but sometimes it
// would get destroyed before the Statistic, causing havoc to ensue.  We "fix"
// this by creating the string the first time it is needed and never destroying
// it.
static ManagedStatic<std::string> LibSupportInfoOutputFilename;
static std::string &getLibSupportInfoOutputFilename() {
  return *LibSupportInfoOutputFilename;
}

static ManagedStatic<sys::SmartMutex<true> > TimerLock;

namespace {
  static cl::opt<bool>
  TrackSpace("track-memory", cl::desc("Enable -time-passes memory "
                                      "tracking (this may be slow)"),
             cl::Hidden);

  static cl::opt<std::string, true>
  InfoOutputFilename("info-output-file", cl::value_desc("filename"),
                     cl::desc("File to append -stats and -timer output to"),
                   cl::Hidden, cl::location(getLibSupportInfoOutputFilename()));
}

// GetLibSupportInfoOutputFile - Return a file stream to print our output on.
raw_ostream *llvm::GetLibSupportInfoOutputFile() {
  std::string &LibSupportInfoOutputFilename = getLibSupportInfoOutputFilename();
  if (LibSupportInfoOutputFilename.empty())
    return &errs();
  if (LibSupportInfoOutputFilename == "-")
    return &outs();
  
  std::string Error;
  raw_ostream *Result = new raw_fd_ostream(LibSupportInfoOutputFilename.c_str(),
                                           Error, raw_fd_ostream::F_Append);
  if (Error.empty())
    return Result;
  
  errs() << "Error opening info-output-file '"
    << LibSupportInfoOutputFilename << " for appending!\n";
  delete Result;
  return &errs();
}


static TimerGroup *DefaultTimerGroup = 0;
static TimerGroup *getDefaultTimerGroup() {
  TimerGroup *tmp = DefaultTimerGroup;
  sys::MemoryFence();
  if (tmp) return tmp;
  
  llvm_acquire_global_lock();
  tmp = DefaultTimerGroup;
  if (!tmp) {
    tmp = new TimerGroup("Miscellaneous Ungrouped Timers");
    sys::MemoryFence();
    DefaultTimerGroup = tmp;
  }
  llvm_release_global_lock();

  return tmp;
}

//===----------------------------------------------------------------------===//
// Timer Implementation
//===----------------------------------------------------------------------===//

void Timer::init(const std::string &N) {
  assert(TG == 0 && "Timer already initialized");
  Name = N;
  Started = false;
  TG = getDefaultTimerGroup();
  TG->addTimer();
}

void Timer::init(const std::string &N, TimerGroup &tg) {
  assert(TG == 0 && "Timer already initialized");
  Name = N;
  Started = false;
  TG = &tg;
  TG->addTimer();
}

Timer::~Timer() {
  if (!TG) return;  // Never initialized.
  
  if (Started) {
    Started = false;
    TG->addTimerToPrint(Time, Name);
  }
  TG->removeTimer();
}

static inline size_t getMemUsage() {
  if (TrackSpace)
    return sys::Process::GetMallocUsage();
  return 0;
}

TimeRecord TimeRecord::getCurrentTime(bool Start) {
  TimeRecord Result;

  sys::TimeValue now(0,0);
  sys::TimeValue user(0,0);
  sys::TimeValue sys(0,0);

  ssize_t MemUsed = 0;
  if (Start) {
    MemUsed = getMemUsage();
    sys::Process::GetTimeUsage(now, user, sys);
  } else {
    sys::Process::GetTimeUsage(now, user, sys);
    MemUsed = getMemUsage();
  }

  Result.WallTime   =  now.seconds() +  now.microseconds() / 1000000.0;
  Result.UserTime   = user.seconds() + user.microseconds() / 1000000.0;
  Result.SystemTime =  sys.seconds() +  sys.microseconds() / 1000000.0;
  Result.MemUsed = MemUsed;
  return Result;
}

static ManagedStatic<std::vector<Timer*> > ActiveTimers;

void Timer::startTimer() {
  Started = true;
  ActiveTimers->push_back(this);
  Time -= TimeRecord::getCurrentTime(true);
}

void Timer::stopTimer() {
  Time += TimeRecord::getCurrentTime(false);

  if (ActiveTimers->back() == this) {
    ActiveTimers->pop_back();
  } else {
    std::vector<Timer*>::iterator I =
      std::find(ActiveTimers->begin(), ActiveTimers->end(), this);
    assert(I != ActiveTimers->end() && "stop but no startTimer?");
    ActiveTimers->erase(I);
  }
}

static void printVal(double Val, double Total, raw_ostream &OS) {
  if (Total < 1e-7)   // Avoid dividing by zero.
    OS << "        -----     ";
  else {
    OS << "  " << format("%7.4f", Val) << " (";
    OS << format("%5.1f", Val*100/Total) << "%)";
  }
}

void TimeRecord::print(const TimeRecord &Total, raw_ostream &OS) const {
  if (Total.getUserTime())
    printVal(getUserTime(), Total.getUserTime(), OS);
  if (Total.getSystemTime())
    printVal(getSystemTime(), Total.getSystemTime(), OS);
  if (Total.getProcessTime())
    printVal(getProcessTime(), Total.getProcessTime(), OS);
  printVal(getWallTime(), Total.getWallTime(), OS);
  
  OS << "  ";
  
  if (Total.getMemUsed())
    OS << format("%9lld", (long long)getMemUsed()) << "  ";
}


//===----------------------------------------------------------------------===//
//   NamedRegionTimer Implementation
//===----------------------------------------------------------------------===//

typedef StringMap<Timer> Name2TimerMap;
typedef StringMap<std::pair<TimerGroup, Name2TimerMap> > Name2PairMap;

static ManagedStatic<Name2TimerMap> NamedTimers;
static ManagedStatic<Name2PairMap> NamedGroupedTimers;

static Timer &getNamedRegionTimer(const std::string &Name) {
  sys::SmartScopedLock<true> L(*TimerLock);
  
  Timer &T = (*NamedTimers)[Name];
  if (!T.isInitialized())
    T.init(Name);
  return T;
}

static Timer &getNamedRegionTimer(const std::string &Name,
                                  const std::string &GroupName) {
  sys::SmartScopedLock<true> L(*TimerLock);

  std::pair<TimerGroup, Name2TimerMap> &GroupEntry =
    (*NamedGroupedTimers)[GroupName];

  if (GroupEntry.second.empty())
    GroupEntry.first.setName(GroupName);

  Timer &T = GroupEntry.second[Name];
  if (!T.isInitialized())
    T.init(Name);
  return T;
}

NamedRegionTimer::NamedRegionTimer(const std::string &Name)
  : TimeRegion(getNamedRegionTimer(Name)) {}

NamedRegionTimer::NamedRegionTimer(const std::string &Name,
                                   const std::string &GroupName)
  : TimeRegion(getNamedRegionTimer(Name, GroupName)) {}

//===----------------------------------------------------------------------===//
//   TimerGroup Implementation
//===----------------------------------------------------------------------===//

void TimerGroup::removeTimer() {
  sys::SmartScopedLock<true> L(*TimerLock);
  if (--NumTimers != 0 || TimersToPrint.empty())
    return; // Don't print timing report.
  
  // Sort the timers in descending order by amount of time taken.
  std::sort(TimersToPrint.begin(), TimersToPrint.end());

  // Figure out how many spaces to indent TimerGroup name.
  unsigned Padding = (80-Name.length())/2;
  if (Padding > 80) Padding = 0;         // Don't allow "negative" numbers

  raw_ostream *OutStream = GetLibSupportInfoOutputFile();

  TimeRecord Total;
  for (unsigned i = 0, e = TimersToPrint.size(); i != e; ++i)
    Total += TimersToPrint[i].first;

  // Print out timing header.
  *OutStream << "===" << std::string(73, '-') << "===\n";
  OutStream->indent(Padding) << Name << '\n';
  *OutStream << "===" << std::string(73, '-') << "===\n";

  // If this is not an collection of ungrouped times, print the total time.
  // Ungrouped timers don't really make sense to add up.  We still print the
  // TOTAL line to make the percentages make sense.
  if (this != DefaultTimerGroup) {
    *OutStream << "  Total Execution Time: ";
    *OutStream << format("%5.4f", Total.getProcessTime()) << " seconds (";
    *OutStream << format("%5.4f", Total.getWallTime()) << " wall clock)\n";
  }
  *OutStream << "\n";

  if (Total.getUserTime())
    *OutStream << "   ---User Time---";
  if (Total.getSystemTime())
    *OutStream << "   --System Time--";
  if (Total.getProcessTime())
    *OutStream << "   --User+System--";
  *OutStream << "   ---Wall Time---";
  if (Total.getMemUsed())
    *OutStream << "  ---Mem---";
  *OutStream << "  --- Name ---\n";

  // Loop through all of the timing data, printing it out.
  for (unsigned i = 0, e = TimersToPrint.size(); i != e; ++i) {
    const std::pair<TimeRecord, std::string> &Entry = TimersToPrint[e-i-1];
    Entry.first.print(Total, *OutStream);
    *OutStream << Entry.second << '\n';
  }

  Total.print(Total, *OutStream);
  *OutStream << "Total\n\n";
  OutStream->flush();

  TimersToPrint.clear();

  if (OutStream != &errs() && OutStream != &outs())
    delete OutStream;   // Close the file.
}

void TimerGroup::addTimer() {
  sys::SmartScopedLock<true> L(*TimerLock);
  ++NumTimers;
}

void TimerGroup::addTimerToPrint(const TimeRecord &T, const std::string &Name) {
  sys::SmartScopedLock<true> L(*TimerLock);
  TimersToPrint.push_back(std::make_pair(T, Name));
}

