//===- Reg2Mem.cpp - Convert registers to allocas -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file demotes all registers to memory references.  It is intented to be
// the inverse of PromoteMemoryToRegister.  By converting to loads, the only
// values live accross basic blocks are allocas and loads before phi nodes.
// It is intended that this should make CFG hacking much easier.
// To make later hacking easier, the entry block is split into two, such that
// all introduced allocas and nothing else are in the entry block.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Pass.h"
#include "llvm/Function.h"
#include "llvm/Module.h"
#include "llvm/BasicBlock.h"
#include "llvm/Instructions.h"
#include "llvm/ADT/Statistic.h"

#include <list>

using namespace llvm;

namespace {
  Statistic<> NumDemoted("reg2mem", "Number of registers demoted");
  
  struct RegToMem : public FunctionPass {

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequiredID(BreakCriticalEdgesID);
      AU.addPreservedID(BreakCriticalEdgesID);
    }

   bool valueEscapes(Instruction* i) {
      BasicBlock* bb = i->getParent();
      for(Value::use_iterator ii = i->use_begin(), ie = i->use_end();
          ii != ie; ++ii)
        if (cast<Instruction>(*ii)->getParent() != bb ||
	    isa<PHINode>(*ii))
          return true;
      return false;
    }

    virtual bool runOnFunction(Function &F) {
      if (!F.isExternal()) {
        //give us a clean block
	BasicBlock* bbold = &F.getEntryBlock();
	BasicBlock* bbnew = new BasicBlock("allocablock", &F, &F.getEntryBlock());
	new BranchInst(bbold, bbnew);

        //find the instructions
        std::list<Instruction*> worklist;
        for (Function::iterator ibb = F.begin(), ibe = F.end();
             ibb != ibe; ++ibb)
          for (BasicBlock::iterator iib = ibb->begin(), iie = ibb->end();
               iib != iie; ++iib) {
            if(valueEscapes(iib))
              worklist.push_front(&*iib);
          }
        //demote escaped instructions
        NumDemoted += worklist.size();
        for (std::list<Instruction*>::iterator ilb = worklist.begin(), 
               ile = worklist.end(); ilb != ile; ++ilb)
          DemoteRegToStack(**ilb, false);
        return true;
      }
      return false;
    }
  };
  
  RegisterPass<RegToMem> X("reg2mem", "Demote all values to stack slots");
}

// createDemoteRegisterToMemory - Provide an entry point to create this pass.
//
const PassInfo *llvm::DemoteRegisterToMemoryID = X.getPassInfo();
FunctionPass *llvm::createDemoteRegisterToMemoryPass() {
  return new RegToMem();
}
