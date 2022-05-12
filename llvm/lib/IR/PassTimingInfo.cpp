//===- PassTimingInfo.cpp - LLVM Pass Timing Implementation ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the LLVM Pass Timing infrastructure for both
// new and legacy pass managers.
//
// PassTimingInfo Class - This class is used to calculate information about the
// amount of time each pass takes to execute.  This only happens when
// -time-passes is enabled on the command line.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/PassTimingInfo.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/PassInstrumentation.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/TypeName.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

using namespace llvm;

#define DEBUG_TYPE "time-passes"

namespace llvm {

bool TimePassesIsEnabled = false;
bool TimePassesPerRun = false;

static cl::opt<bool, true> EnableTiming(
    "time-passes", cl::location(TimePassesIsEnabled), cl::Hidden,
    cl::desc("Time each pass, printing elapsed time for each on exit"));

static cl::opt<bool, true> EnableTimingPerRun(
    "time-passes-per-run", cl::location(TimePassesPerRun), cl::Hidden,
    cl::desc("Time each pass run, printing elapsed time for each run on exit"),
    cl::callback([](const bool &) { TimePassesIsEnabled = true; }));

namespace {
namespace legacy {

//===----------------------------------------------------------------------===//
// Legacy pass manager's PassTimingInfo implementation

/// Provides an interface for collecting pass timing information.
///
/// It was intended to be generic but now we decided to split
/// interfaces completely. This is now exclusively for legacy-pass-manager use.
class PassTimingInfo {
public:
  using PassInstanceID = void *;

private:
  StringMap<unsigned> PassIDCountMap; ///< Map that counts instances of passes
  DenseMap<PassInstanceID, std::unique_ptr<Timer>> TimingData; ///< timers for pass instances
  TimerGroup TG;

public:
  /// Default constructor for yet-inactive timeinfo.
  /// Use \p init() to activate it.
  PassTimingInfo();

  /// Print out timing information and release timers.
  ~PassTimingInfo();

  /// Initializes the static \p TheTimeInfo member to a non-null value when
  /// -time-passes is enabled. Leaves it null otherwise.
  ///
  /// This method may be called multiple times.
  static void init();

  /// Prints out timing information and then resets the timers.
  /// By default it uses the stream created by CreateInfoOutputFile().
  void print(raw_ostream *OutStream = nullptr);

  /// Returns the timer for the specified pass if it exists.
  Timer *getPassTimer(Pass *, PassInstanceID);

  static PassTimingInfo *TheTimeInfo;

private:
  Timer *newPassTimer(StringRef PassID, StringRef PassDesc);
};

static ManagedStatic<sys::SmartMutex<true>> TimingInfoMutex;

PassTimingInfo::PassTimingInfo()
    : TG("pass", "... Pass execution timing report ...") {}

PassTimingInfo::~PassTimingInfo() {
  // Deleting the timers accumulates their info into the TG member.
  // Then TG member is (implicitly) deleted, actually printing the report.
  TimingData.clear();
}

void PassTimingInfo::init() {
  if (!TimePassesIsEnabled || TheTimeInfo)
    return;

  // Constructed the first time this is called, iff -time-passes is enabled.
  // This guarantees that the object will be constructed after static globals,
  // thus it will be destroyed before them.
  static ManagedStatic<PassTimingInfo> TTI;
  TheTimeInfo = &*TTI;
}

/// Prints out timing information and then resets the timers.
void PassTimingInfo::print(raw_ostream *OutStream) {
  TG.print(OutStream ? *OutStream : *CreateInfoOutputFile(), true);
}

Timer *PassTimingInfo::newPassTimer(StringRef PassID, StringRef PassDesc) {
  unsigned &num = PassIDCountMap[PassID];
  num++;
  // Appending description with a pass-instance number for all but the first one
  std::string PassDescNumbered =
      num <= 1 ? PassDesc.str() : formatv("{0} #{1}", PassDesc, num).str();
  return new Timer(PassID, PassDescNumbered, TG);
}

Timer *PassTimingInfo::getPassTimer(Pass *P, PassInstanceID Pass) {
  if (P->getAsPMDataManager())
    return nullptr;

  init();
  sys::SmartScopedLock<true> Lock(*TimingInfoMutex);
  std::unique_ptr<Timer> &T = TimingData[Pass];

  if (!T) {
    StringRef PassName = P->getPassName();
    StringRef PassArgument;
    if (const PassInfo *PI = Pass::lookupPassInfo(P->getPassID()))
      PassArgument = PI->getPassArgument();
    T.reset(newPassTimer(PassArgument.empty() ? PassName : PassArgument, PassName));
  }
  return T.get();
}

PassTimingInfo *PassTimingInfo::TheTimeInfo;
} // namespace legacy
} // namespace

Timer *getPassTimer(Pass *P) {
  legacy::PassTimingInfo::init();
  if (legacy::PassTimingInfo::TheTimeInfo)
    return legacy::PassTimingInfo::TheTimeInfo->getPassTimer(P, P);
  return nullptr;
}

/// If timing is enabled, report the times collected up to now and then reset
/// them.
void reportAndResetTimings(raw_ostream *OutStream) {
  if (legacy::PassTimingInfo::TheTimeInfo)
    legacy::PassTimingInfo::TheTimeInfo->print(OutStream);
}

//===----------------------------------------------------------------------===//
// Pass timing handling for the New Pass Manager
//===----------------------------------------------------------------------===//

/// Returns the timer for the specified pass invocation of \p PassID.
/// Each time it creates a new timer.
Timer &TimePassesHandler::getPassTimer(StringRef PassID) {
  if (!PerRun) {
    TimerVector &Timers = TimingData[PassID];
    if (Timers.size() == 0)
      Timers.emplace_back(new Timer(PassID, PassID, TG));
    return *Timers.front();
  }

  // Take a vector of Timers created for this \p PassID and append
  // one more timer to it.
  TimerVector &Timers = TimingData[PassID];
  unsigned Count = Timers.size() + 1;

  std::string FullDesc = formatv("{0} #{1}", PassID, Count).str();

  Timer *T = new Timer(PassID, FullDesc, TG);
  Timers.emplace_back(T);
  assert(Count == Timers.size() && "Timers vector not adjusted correctly.");

  return *T;
}

TimePassesHandler::TimePassesHandler(bool Enabled, bool PerRun)
    : TG("pass", "... Pass execution timing report ..."), Enabled(Enabled),
      PerRun(PerRun) {}

TimePassesHandler::TimePassesHandler()
    : TimePassesHandler(TimePassesIsEnabled, TimePassesPerRun) {}

void TimePassesHandler::setOutStream(raw_ostream &Out) {
  OutStream = &Out;
}

void TimePassesHandler::print() {
  if (!Enabled)
    return;
  TG.print(OutStream ? *OutStream : *CreateInfoOutputFile(), true);
}

LLVM_DUMP_METHOD void TimePassesHandler::dump() const {
  dbgs() << "Dumping timers for " << getTypeName<TimePassesHandler>()
         << ":\n\tRunning:\n";
  for (auto &I : TimingData) {
    StringRef PassID = I.getKey();
    const TimerVector& MyTimers = I.getValue();
    for (unsigned idx = 0; idx < MyTimers.size(); idx++) {
      const Timer* MyTimer = MyTimers[idx].get();
      if (MyTimer && MyTimer->isRunning())
        dbgs() << "\tTimer " << MyTimer << " for pass " << PassID << "(" << idx << ")\n";
    }
  }
  dbgs() << "\tTriggered:\n";
  for (auto &I : TimingData) {
    StringRef PassID = I.getKey();
    const TimerVector& MyTimers = I.getValue();
    for (unsigned idx = 0; idx < MyTimers.size(); idx++) {
      const Timer* MyTimer = MyTimers[idx].get();
      if (MyTimer && MyTimer->hasTriggered() && !MyTimer->isRunning())
        dbgs() << "\tTimer " << MyTimer << " for pass " << PassID << "(" << idx << ")\n";
    }
  }
}

void TimePassesHandler::startTimer(StringRef PassID) {
  Timer &MyTimer = getPassTimer(PassID);
  TimerStack.push_back(&MyTimer);
  if (!MyTimer.isRunning())
    MyTimer.startTimer();
}

void TimePassesHandler::stopTimer(StringRef PassID) {
  assert(TimerStack.size() > 0 && "empty stack in popTimer");
  Timer *MyTimer = TimerStack.pop_back_val();
  assert(MyTimer && "timer should be present");
  if (MyTimer->isRunning())
    MyTimer->stopTimer();
}

void TimePassesHandler::runBeforePass(StringRef PassID) {
  if (isSpecialPass(PassID,
                    {"PassManager", "PassAdaptor", "AnalysisManagerProxy"}))
    return;

  startTimer(PassID);

  LLVM_DEBUG(dbgs() << "after runBeforePass(" << PassID << ")\n");
  LLVM_DEBUG(dump());
}

void TimePassesHandler::runAfterPass(StringRef PassID) {
  if (isSpecialPass(PassID,
                    {"PassManager", "PassAdaptor", "AnalysisManagerProxy"}))
    return;

  stopTimer(PassID);

  LLVM_DEBUG(dbgs() << "after runAfterPass(" << PassID << ")\n");
  LLVM_DEBUG(dump());
}

void TimePassesHandler::registerCallbacks(PassInstrumentationCallbacks &PIC) {
  if (!Enabled)
    return;

  PIC.registerBeforeNonSkippedPassCallback(
      [this](StringRef P, Any) { this->runBeforePass(P); });
  PIC.registerAfterPassCallback(
      [this](StringRef P, Any, const PreservedAnalyses &) {
        this->runAfterPass(P);
      });
  PIC.registerAfterPassInvalidatedCallback(
      [this](StringRef P, const PreservedAnalyses &) {
        this->runAfterPass(P);
      });
  PIC.registerBeforeAnalysisCallback(
      [this](StringRef P, Any) { this->runBeforePass(P); });
  PIC.registerAfterAnalysisCallback(
      [this](StringRef P, Any) { this->runAfterPass(P); });
}

} // namespace llvm
