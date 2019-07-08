//===--- ParallelUtilities.cpp -------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "ParallelUtilities.h"
#include "llvm/Support/Timer.h"
#include <mutex>
#include <shared_mutex>

#define DEBUG_TYPE "par-utils"


namespace opts {
extern cl::OptionCategory BoltCategory;

cl::opt<unsigned>
ThreadCount("thread-count",
  cl::desc("number of threads"),
  cl::init(hardware_concurrency()),
  cl::cat(BoltCategory));

cl::opt<bool>
NoThreads("no-threads",
  cl::desc("disable multithreading"),
  cl::init(false),
  cl::cat(BoltCategory));

cl::opt<unsigned> 
TaskCount("tasks-per-thread",
  cl::desc("number of tasks to be created per thread"),
  cl::init(20),
  cl::cat(BoltCategory));

}

namespace {
/// A single thread pool that is used to run parallel tasks
std::unique_ptr<ThreadPool> ThPoolPtr;
} // namespace

namespace llvm {
namespace bolt {
namespace ParallelUtilities {

ThreadPool &getThreadPool() {
  if (ThPoolPtr.get())
    return *ThPoolPtr;

  ThPoolPtr = std::make_unique<ThreadPool>(opts::ThreadCount);
  return *ThPoolPtr;
}

void runOnEachFunction(BinaryContext &BC, SchedulingPolicy SchedPolicy,
                       WorkFuncTy WorkFunction, PredicateTy SkipPredicate,
                       std::string LogName, unsigned TasksPerThread) {
  auto runBlock = [&](std::map<uint64_t, BinaryFunction>::iterator BlockBegin,
                      std::map<uint64_t, BinaryFunction>::iterator BlockEnd) {
    Timer T(LogName, LogName);
    DEBUG(T.startTimer());

    for (auto It = BlockBegin; It != BlockEnd; ++It) {
      auto &BF = It->second;
      if (SkipPredicate && SkipPredicate(BF))
        continue;

      WorkFunction(BF);
    }
    DEBUG(T.stopTimer());
  };

  if (opts::NoThreads) {
    runBlock(BC.getBinaryFunctions().begin(), BC.getBinaryFunctions().end());
    return;
  }

  // Estimate the overall runtime cost using the scheduling policy
  unsigned TotalCost = 0;
  const unsigned BlocksCount = TasksPerThread * opts::ThreadCount;
  if (SchedPolicy == SchedulingPolicy::SP_TRIVIAL) {
    TotalCost = BC.getBinaryFunctions().size();
  } else {
    for (auto &BFI : BC.getBinaryFunctions()) {
      auto &BF = BFI.second;

      if (SkipPredicate && SkipPredicate(BF))
        continue;

      if (SchedPolicy == SchedulingPolicy::SP_CONSTANT)
        TotalCost++;
      else if (SchedPolicy == SchedulingPolicy::SP_LINEAR)
        TotalCost += BF.size();
      else if (SchedPolicy == SchedulingPolicy::SP_QUADRATIC)
        TotalCost += BF.size() * BF.size();
    }
  }

  // Divide work into blocks of equal cost
  ThreadPool &ThPool = getThreadPool();
  const unsigned BlockCost = TotalCost / BlocksCount;
  auto BlockBegin = BC.getBinaryFunctions().begin();
  unsigned CurrentCost = 0;

  for (auto It = BC.getBinaryFunctions().begin();
       It != BC.getBinaryFunctions().end(); ++It) {
    auto &BF = It->second;

    if (SchedPolicy == SchedulingPolicy::SP_TRIVIAL)
      CurrentCost++;
    else {
      if (SkipPredicate && SkipPredicate(BF))
        continue;

      if (SchedPolicy == SchedulingPolicy::SP_CONSTANT)
        CurrentCost++;
      else if (SchedPolicy == SchedulingPolicy::SP_LINEAR)
        CurrentCost += BF.size();
      else if (SchedPolicy == SchedulingPolicy::SP_QUADRATIC)
        CurrentCost += BF.size() * BF.size();
    }

    if (CurrentCost >= BlockCost) {
      ThPool.async(runBlock, BlockBegin, std::next(It));
      BlockBegin = std::next(It);
      CurrentCost = 0;
    }
  }
  ThPool.async(runBlock, BlockBegin, BC.getBinaryFunctions().end());
  ThPool.wait();
}
} // namespace ParallelUtilities
} // namespace bolt
} // namespace llvm
