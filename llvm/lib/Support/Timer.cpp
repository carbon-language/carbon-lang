//===-- Timer.cpp - Interval Timing Support -------------------------------===//
//
// Interval Timing implementation.
//
//===----------------------------------------------------------------------===//

#include "Support/Timer.h"
#include "Support/CommandLine.h"
#include "Config/sys/resource.h"
#include "Config/sys/time.h"
#include "Config/unistd.h"
#include "Config/malloc.h"
#include <iostream>
#include <algorithm>
#include <functional>
#include <fstream>
#include <map>

// getLibSupportInfoOutputFilename - This ugly hack is brought to you courtesy
// of constructor/destructor ordering being unspecified by C++.  Basically the
// problem is that a Statistic<> object gets destroyed, which ends up calling
// 'GetLibSupportInfoOutputFile()' (below), which calls this function.
// LibSupportInfoOutputFilename used to be a global variable, but sometimes it
// would get destroyed before the Statistic, causing havoc to ensue.  We "fix"
// this by creating the string the first time it is needed and never destroying
// it.
static std::string &getLibSupportInfoOutputFilename() {
  static std::string *LibSupportInfoOutputFilename = new std::string();
  return *LibSupportInfoOutputFilename;
}

namespace {
#ifdef HAVE_MALLINFO
  cl::opt<bool>
  TrackSpace("track-memory", cl::desc("Enable -time-passes memory "
                                      "tracking (this may be slow)"),
             cl::Hidden);
#endif

  cl::opt<std::string, true>
  InfoOutputFilename("info-output-file", cl::value_desc("filename"),
                     cl::desc("File to append -stats and -timer output to"),
                   cl::Hidden, cl::location(getLibSupportInfoOutputFilename()));
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
#ifdef HAVE_MALLINFO
  if (TrackSpace) {
    struct mallinfo MI = mallinfo();
    return MI.uordblks/*+MI.hblkhd*/;
  }
#endif
  return 0;
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
//   NamedRegionTimer Implementation
//===----------------------------------------------------------------------===//

static Timer &getNamedRegionTimer(const std::string &Name) {
  static std::map<std::string, Timer> NamedTimers;

  std::map<std::string, Timer>::iterator I = NamedTimers.lower_bound(Name);
  if (I != NamedTimers.end() && I->first == Name)
    return I->second;

  return NamedTimers.insert(I, std::make_pair(Name, Timer(Name)))->second;
}

NamedRegionTimer::NamedRegionTimer(const std::string &Name)
  : TimeRegion(getNamedRegionTimer(Name)) {}


//===----------------------------------------------------------------------===//
//   TimerGroup Implementation
//===----------------------------------------------------------------------===//

// printAlignedFP - Simulate the printf "%A.Bf" format, where A is the
// TotalWidth size, and B is the AfterDec size.
//
static void printAlignedFP(double Val, unsigned AfterDec, unsigned TotalWidth,
                           std::ostream &OS) {
  assert(TotalWidth >= AfterDec+1 && "Bad FP Format!");
  OS.width(TotalWidth-AfterDec-1);
  char OldFill = OS.fill();
  OS.fill(' ');
  OS << (int)Val;  // Integer part;
  OS << ".";
  OS.width(AfterDec);
  OS.fill('0');
  unsigned ResultFieldSize = 1;
  while (AfterDec--) ResultFieldSize *= 10;
  OS << (int)(Val*ResultFieldSize) % ResultFieldSize;
  OS.fill(OldFill);
}

static void printVal(double Val, double Total, std::ostream &OS) {
  if (Total < 1e-7)   // Avoid dividing by zero...
    OS << "        -----     ";
  else {
    OS << "  ";
    printAlignedFP(Val, 4, 7, OS);
    OS << " (";
    printAlignedFP(Val*100/Total, 1, 5, OS);
    OS << "%)";
  }
}

void Timer::print(const Timer &Total, std::ostream &OS) {
  if (Total.UserTime)
    printVal(UserTime, Total.UserTime, OS);
  if (Total.SystemTime)
    printVal(SystemTime, Total.SystemTime, OS);
  if (Total.getProcessTime())
    printVal(getProcessTime(), Total.getProcessTime(), OS);
  printVal(Elapsed, Total.Elapsed, OS);
  
  OS << "  ";

  if (Total.MemUsed) {
    OS.width(9);
    OS << MemUsed << "  ";
  }
  if (Total.PeakMem) {
    if (PeakMem) {
      OS.width(9);
      OS << PeakMem << "  ";
    } else
      OS << "           ";
  }
  OS << Name << "\n";

  Started = false;  // Once printed, don't print again
}

// GetLibSupportInfoOutputFile - Return a file stream to print our output on...
std::ostream *GetLibSupportInfoOutputFile() {
  std::string &LibSupportInfoOutputFilename = getLibSupportInfoOutputFilename();
  if (LibSupportInfoOutputFilename.empty())
    return &std::cerr;
  if (LibSupportInfoOutputFilename == "-")
    return &std::cout;

  std::ostream *Result = new std::ofstream(LibSupportInfoOutputFilename.c_str(),
                                           std::ios::app);
  if (!Result->good()) {
    std::cerr << "Error opening info-output-file '"
              << LibSupportInfoOutputFilename << " for appending!\n";
    delete Result;
    return &std::cerr;
  }
  return Result;
}


void TimerGroup::removeTimer() {
  if (--NumTimers == 0 && !TimersToPrint.empty()) { // Print timing report...
    // Sort the timers in descending order by amount of time taken...
    std::sort(TimersToPrint.begin(), TimersToPrint.end(),
              std::greater<Timer>());

    // Figure out how many spaces to indent TimerGroup name...
    unsigned Padding = (80-Name.length())/2;
    if (Padding > 80) Padding = 0;         // Don't allow "negative" numbers

    std::ostream *OutStream = GetLibSupportInfoOutputFile();

    ++NumTimers;
    {  // Scope to contain Total timer... don't allow total timer to drop us to
       // zero timers...
      Timer Total("TOTAL");
  
      for (unsigned i = 0, e = TimersToPrint.size(); i != e; ++i)
        Total.sum(TimersToPrint[i]);
      
      // Print out timing header...
      *OutStream << "===" << std::string(73, '-') << "===\n"
                 << std::string(Padding, ' ') << Name << "\n"
                 << "===" << std::string(73, '-')
                 << "===\n  Total Execution Time: ";

      printAlignedFP(Total.getProcessTime(), 4, 5, *OutStream);
      *OutStream << " seconds (";
      printAlignedFP(Total.getWallTime(), 4, 5, *OutStream);
      *OutStream << " wall clock)\n\n";

      if (Total.UserTime)
        *OutStream << "   ---User Time---";
      if (Total.SystemTime)
        *OutStream << "   --System Time--";
      if (Total.getProcessTime())
        *OutStream << "   --User+System--";
      *OutStream << "   ---Wall Time---";
      if (Total.getMemUsed())
        *OutStream << "  ---Mem---";
      if (Total.getPeakMem())
        *OutStream << "  -PeakMem-";
      *OutStream << "  --- Name ---\n";
      
      // Loop through all of the timing data, printing it out...
      for (unsigned i = 0, e = TimersToPrint.size(); i != e; ++i)
        TimersToPrint[i].print(Total, *OutStream);
    
      Total.print(Total, *OutStream);
      *OutStream << std::endl;  // Flush output
    }
    --NumTimers;

    TimersToPrint.clear();

    if (OutStream != &std::cerr && OutStream != &std::cout)
      delete OutStream;   // Close the file...
  }

  // Delete default timer group!
  if (NumTimers == 0 && this == DefaultTimerGroup) {
    delete DefaultTimerGroup;
    DefaultTimerGroup = 0;
  }
}
