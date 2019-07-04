//===-- ParallelUtilities.h - ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This class creates an interface that can be used to run parallel tasks that
// operate on functions. Several scheduling criteria are supported using
// SchedulingPolicy, and are defined by how the runtime cost should be
// estimated.
// If the NoThreads flags is passed, work will execute sequentially.
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_BOLT_PARALLEL_UTILITIES_H
#define LLVM_TOOLS_LLVM_BOLT_PARALLEL_UTILITIES_H

#include "llvm/Support/ThreadPool.h"
#include "BinaryContext.h"
#include "BinaryFunction.h"

using namespace llvm;

namespace opts {
extern cl::opt<unsigned> ThreadCount;
extern cl::opt<bool> NoThreads;
extern cl::opt<unsigned> TaskCount;
}

namespace llvm {
namespace bolt {
namespace ParallelUtilities {

using WorkFuncTy = std::function<void(BinaryFunction &BF)>;
using PredicateTy = std::function<bool(const BinaryFunction &BF)>;

enum SchedulingPolicy {
  SP_TRIVIAL,  /// cost is estimated by the number of functions
  SP_CONSTANT, /// cost is estimated by the number of non-skipped functions
  SP_LINEAR,   /// cost is estimated by the size of non-skipped functions
  SP_QUADRATIC /// cost is estimated by the square of the size of non-skipped
               /// functions
};

/// Return the managed threadpool and initialize it if not intiliazed
ThreadPool &getThreadPool();

// Perform the work on each binary function, except those that are accepted
// by the SkipPredicate, scheduling heuristic is based on SchedPolicy
void runOnEachFunction(BinaryContext &BC, SchedulingPolicy SchedPolicy,
                       WorkFuncTy WorkFunction,
                       PredicateTy SkipPredicate = PredicateTy(),
                       std::string LogName = "",
                       unsigned TasksPerThread = opts::TaskCount);
} // namespace ParallelUtilities
} // namespace bolt
} // namespace llvm
#endif
