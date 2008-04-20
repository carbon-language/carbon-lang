//===- JumpThreading.cpp - Thread control through conditional blocks ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass performs 'jump threading', which looks at blocks that have multiple
// predecessors and multiple successors.  If one or more of the predecessors of
// the block can be proven to always jump to one of the successors, we forward
// the edge from the predecessor to the successor by duplicating the contents of
// this block.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "jump-threading"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Pass.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
using namespace llvm;

//STATISTIC(NumThreads, "Number of jumps threaded");

namespace {
  cl::opt<unsigned>
  Threshold("jump-threading-threshold", 
            cl::desc("Max block size to duplicate for jump threading"),
            cl::init(6), cl::Hidden);
  class VISIBILITY_HIDDEN JumpThreading : public FunctionPass {
  public:
    static char ID; // Pass identification
    JumpThreading() : FunctionPass((intptr_t)&ID) {}

    bool runOnFunction(Function &F);
  };
  char JumpThreading::ID = 0;
  RegisterPass<JumpThreading> X("jump-threading", "Jump Threading");
}

// Public interface to the Jump Threading pass
FunctionPass *llvm::createJumpThreadingPass() { return new JumpThreading(); }

/// runOnFunction - Top level algorithm.
///
bool JumpThreading::runOnFunction(Function &F) {
  bool Changed = false;
  return Changed;
}
