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

#include "BinaryContext.h"
#include "BinaryFunction.h"
#include "MCPlusBuilder.h"
#include "llvm/Support/ThreadPool.h"

using namespace llvm;

namespace opts {
extern cl::opt<unsigned> ThreadCount;
extern cl::opt<bool> NoThreads;
extern cl::opt<unsigned> TaskCount;
}

namespace llvm {
namespace bolt {
namespace ParallelUtilities {

using WorkFuncWithAllocTy =
    std::function<void(BinaryFunction &BF, MCPlusBuilder::AllocatorIdTy)>;
using WorkFuncTy = std::function<void(BinaryFunction &BF)>;
using PredicateTy = std::function<bool(const BinaryFunction &BF)>;

enum SchedulingPolicy {
  SP_TRIVIAL,     /// cost is estimated by the number of functions
  SP_CONSTANT,    /// cost is estimated by the number of non-skipped functions
  SP_INST_LINEAR, /// cost is estimated by inst count
  SP_INST_QUADRATIC, /// cost is estimated by the square of the inst count
  SP_BB_LINEAR,      /// cost is estimated by BB count
  SP_BB_QUADRATIC,   /// cost is estimated by the square of the BB count
};

/// Return the managed threadpool and initialize it if not intiliazed
ThreadPool &getThreadPool();

/// Perform the work on each BinaryFunction except those that are accepted
/// by SkipPredicate, scheduling heuristic is based on SchedPolicy.
/// ForceSequential will selectively disable parallel execution and perform the
/// work sequentially.
void runOnEachFunction(BinaryContext &BC, SchedulingPolicy SchedPolicy,
                       WorkFuncTy WorkFunction,
                       PredicateTy SkipPredicate = PredicateTy(),
                       std::string LogName = "", bool ForceSequential = false,
                       unsigned TasksPerThread = opts::TaskCount);

/// Perform the work on each BinaryFunction except those that are rejected
/// by SkipPredicate, and create a unique annotation allocator for each
/// task. This should be used whenever the work function creates annotations to
/// allow thread-safe annotation creation.
/// ForceSequential will selectively disable parallel execution and perform the
/// work sequentially.
void runOnEachFunctionWithUniqueAllocId(
    BinaryContext &BC, SchedulingPolicy SchedPolicy,
    WorkFuncWithAllocTy WorkFunction, PredicateTy SkipPredicate,
    std::string LogName = "", bool ForceSequential = false,
    unsigned TasksPerThread = opts::TaskCount);

} // namespace ParallelUtilities
} // namespace bolt
} // namespace llvm
#endif
