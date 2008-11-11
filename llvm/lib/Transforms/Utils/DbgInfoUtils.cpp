//===-- DbgInfoUtils.cpp - DbgInfo Utilities -------------------------------==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Utility functions to manipulate debugging information.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/DbgInfoUtils.h"
#include "llvm/IntrinsicInst.h"

using namespace llvm;

/// RemoveDeadDbgIntrinsics - Remove dead dbg intrinsics from this 
/// basic block.
void llvm::RemoveDeadDbgIntrinsics(BasicBlock &BB) {
  BasicBlock::iterator BI = BB.begin(), BE = BB.end();
  if (BI == BE) return;

  Instruction *Prev = BI; ++BI;
  while (BI != BE) {
    Instruction *Next = BI; ++BI;
    DbgInfoIntrinsic *DBI_Prev = dyn_cast<DbgInfoIntrinsic>(Prev);
    if (!DBI_Prev) {
      Prev = Next;
      continue;
    }
    
    // If there are two consecutive llvm.dbg.stoppoint calls then
    // it is likely that the optimizer deleted code in between these
    // two intrinsics. 
    DbgInfoIntrinsic *DBI_Next = dyn_cast<DbgInfoIntrinsic>(Next);
    if (DBI_Next 
        && DBI_Prev->getIntrinsicID() == llvm::Intrinsic::dbg_stoppoint
        && DBI_Next->getIntrinsicID() == llvm::Intrinsic::dbg_stoppoint)
      Prev->eraseFromParent();
    
    // If a llvm.dbg.stoppoint is placed just before an unconditional
    // branch then remove the llvm.dbg.stoppoint intrinsic.
    else if (BranchInst *UC = dyn_cast<BranchInst>(Next)) {
      if (UC->isUnconditional() 
          && DBI_Prev->getIntrinsicID() == llvm::Intrinsic::dbg_stoppoint)
        Prev->eraseFromParent();
    }

    Prev = Next;
  }
}

/// RemoveDeadDbgIntrinsics - Remove dead dbg intrinsics from this function.
void llvm::RemoveDeadDbgIntrinsics(Function &F) {
  for (Function::iterator FI = F.begin(), FE = F.end();
       FI != FE; ++FI) 
    RemoveDeadDbgIntrinsics(*FI);
}
