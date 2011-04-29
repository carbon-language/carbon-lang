//===---------- MayAliasSet.cpp  - May-Alais Set for base pointers --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the MayAliasSet class
//
//===----------------------------------------------------------------------===//

#include "polly/TempScopInfo.h"
#include "polly/MayAliasSet.h"

#include "llvm/LLVMContext.h"
#include "llvm/Analysis/RegionInfo.h"
#include "llvm/Analysis/RegionIterator.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AliasSetTracker.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/Support/Debug.h"

using namespace llvm;
using namespace polly;

void MayAliasSet::print(raw_ostream &OS) const {
  OS << "Must alias {";
  
  for (const_iterator I = mustalias_begin(), E = mustalias_end(); I != E; ++I) {
    WriteAsOperand(OS, *I, false);
    OS << ", ";
  }

  OS << "} May alias {";
  OS << '}';
}

void MayAliasSet::dump() const {
  print(dbgs());
}

void MayAliasSetInfo::buildMayAliasSets(TempScop &Scop, AliasAnalysis &AA) {
  AliasSetTracker AST(AA); 
  Region &MaxR = Scop.getMaxRegion();

  // Find out all base pointers that appeared in Scop and build the Alias set.
  // Note: We may build the alias sets while we are building access functions
  // to obtain better performance.
  for (Region::block_iterator I = MaxR.block_begin(), E = MaxR.block_end();
      I != E; ++I) {
    BasicBlock *BB = I->getNodeAs<BasicBlock>();
    if (const AccFuncSetType *AFS = Scop.getAccessFunctions(BB)) {
      for (AccFuncSetType::const_iterator AI = AFS->begin(), AE = AFS->end();
          AI != AE; ++AI) {
        const SCEVAffFunc &AccFunc = AI->first;
        Instruction *Inst = AI->second;
        Value *BaseAddr = const_cast<Value*>(AccFunc.getBaseAddr());

        AST.add(BaseAddr, AliasAnalysis::UnknownSize,
          Inst->getMetadata(LLVMContext::MD_tbaa));
      }
    }
  }

  // Build the may-alias set with the AliasSetTracker.
  for (AliasSetTracker::iterator I = AST.begin(), E = AST.end(); I != E; ++I) {
    AliasSet &AS = *I;

    // Ignore the dummy alias set.
    if (AS.isForwardingAliasSet()) continue;

    // The most simple case: All pointers in the set must-alias each others.
    if (AS.isMustAlias()) {
      MayAliasSet *MayAS = new (MayASAllocator.Allocate()) MayAliasSet();

      for (AliasSet::iterator PI = AS.begin(), PE = AS.end(); PI != PE; ++PI) {
        Value *Ptr = PI.getPointer();

        MayAS->addMustAliasPtr(Ptr);
        BasePtrMap.insert(std::make_pair(Ptr, MayAS));
      }

      continue;
    }

    assert(0 && "SCoPDetection pass should not allow May-Alias set!");
  }
}
