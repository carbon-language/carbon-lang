//===- JumpThreading.cpp - Thread control through conditional blocks ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Jump Threading pass.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "jump-threading"
#include "llvm/Transforms/Scalar.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Pass.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
using namespace llvm;

//STATISTIC(NumThreads, "Number of jumps threaded");

static cl::opt<unsigned>
Threshold("jump-threading-threshold", 
          cl::desc("Max block size to duplicate for jump threading"),
          cl::init(6), cl::Hidden);

namespace {
  /// This pass performs 'jump threading', which looks at blocks that have
  /// multiple predecessors and multiple successors.  If one or more of the
  /// predecessors of the block can be proven to always jump to one of the
  /// successors, we forward the edge from the predecessor to the successor by
  /// duplicating the contents of this block.
  ///
  /// An example of when this can occur is code like this:
  ///
  ///   if () { ...
  ///     X = 4;
  ///   }
  ///   if (X < 3) {
  ///
  /// In this case, the unconditional branch at the end of the first if can be
  /// revectored to the false side of the second if.
  ///
  class VISIBILITY_HIDDEN JumpThreading : public FunctionPass {
  public:
    static char ID; // Pass identification
    JumpThreading() : FunctionPass((intptr_t)&ID) {}

    bool runOnFunction(Function &F);
    bool ThreadBlock(BasicBlock &BB);
  };
  char JumpThreading::ID = 0;
  RegisterPass<JumpThreading> X("jump-threading", "Jump Threading");
}

// Public interface to the Jump Threading pass
FunctionPass *llvm::createJumpThreadingPass() { return new JumpThreading(); }

/// runOnFunction - Top level algorithm.
///
bool JumpThreading::runOnFunction(Function &F) {
  DOUT << "Jump threading on function '" << F.getNameStart() << "'\n";
  bool Changed = false;
  for (Function::iterator I = F.begin(), E = F.end(); I != E; ++I)
    Changed |= ThreadBlock(*I);
  return Changed;
}

/// getJumpThreadDuplicationCost - Return the cost of duplicating this block to
/// thread across it.
static unsigned getJumpThreadDuplicationCost(const BasicBlock &BB) {
  BasicBlock::const_iterator I = BB.begin();
  /// Ignore PHI nodes, these will be flattened when duplication happens.
  while (isa<PHINode>(*I)) ++I;

  // Sum up the cost of each instruction until we get to the terminator.  Don't
  // include the terminator because the copy won't include it.
  unsigned Size = 0;
  for (; !isa<TerminatorInst>(I); ++I) {
    // Debugger intrinsics don't incur code size.
    if (isa<DbgInfoIntrinsic>(I)) continue;
    
    // If this is a pointer->pointer bitcast, it is free.
    if (isa<BitCastInst>(I) && isa<PointerType>(I->getType()))
      continue;
    
    // All other instructions count for at least one unit.
    ++Size;
    
    // Calls are more expensive.  If they are non-intrinsic calls, we model them
    // as having cost of 4.  If they are a non-vector intrinsic, we model them
    // as having cost of 2 total, and if they are a vector intrinsic, we model
    // them as having cost 1.
    if (const CallInst *CI = dyn_cast<CallInst>(I)) {
      if (!isa<IntrinsicInst>(CI))
        Size += 3;
      else if (isa<VectorType>(CI->getType()))
        Size += 1;
    }
  }
  
  // Threading through a switch statement is particularly profitable.  If this
  // block ends in a switch, decrease its cost to make it more likely to happen.
  if (isa<SwitchInst>(I))
    Size = Size > 6 ? Size-6 : 0;
  
  return Size;
}


/// ThreadBlock - If there are any predecessors whose control can be threaded
/// through to a successor, transform them now.
bool JumpThreading::ThreadBlock(BasicBlock &BB) {
  // If there is only one predecessor or successor, then there is nothing to do.
  if (BB.getTerminator()->getNumSuccessors() == 1 || BB.getSinglePredecessor())
    return false;
  
  // See if this block ends with a branch of switch.  If so, see if the
  // condition is a phi node.  If so, and if an entry of the phi node is a
  // constant, we can thread the block.
  Value *Condition;
  if (BranchInst *BI = dyn_cast<BranchInst>(BB.getTerminator()))
    Condition = BI->getCondition();
  else if (SwitchInst *SI = dyn_cast<SwitchInst>(BB.getTerminator()))
    Condition = SI->getCondition();
  else
    return false; // Must be an invoke.

  // See if this is a phi node in the current block.
  PHINode *PN = dyn_cast<PHINode>(Condition);
  if (!PN || PN->getParent() != &BB) return false;
  
  // See if the phi node has any constant values.  If so, we can determine where
  // the corresponding predecessor will branch.
  unsigned PredNo = ~0U;
  for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
    if (isa<ConstantInt>(PN->getIncomingValue(i))) {
      PredNo = i;
      break;
    }
  }
  
  // If no incoming value has a constant, we don't know the destination of any
  // predecessors.
  if (PredNo == ~0U)
    return false;
  
  // See if the cost of duplicating this block is low enough.
  unsigned JumpThreadCost = getJumpThreadDuplicationCost(BB);
  if (JumpThreadCost > Threshold) {
    DOUT << "  Not threading BB '" << BB.getNameStart()
         << "' - Cost is too high: " << JumpThreadCost << "\n";
    return false;
  }

  DOUT << "  Threading BB '" << BB.getNameStart() << "'.  Cost is: "
       << JumpThreadCost << "\n";
  
  return false;
}
