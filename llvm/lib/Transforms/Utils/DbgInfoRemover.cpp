//===- DbgInforemover.cpp - Remove Debug Info Intrinsics ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is utility pass removes all debug information intrinsics from a 
// function.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO.h"
#include "llvm/Module.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"
using namespace llvm;

namespace {
  // Remove dbg intrinsics and related globals from this module.
  struct RemoveDbgInfoPass : public ModulePass {
    static char ID; // Pass identification, replacement for typeid
    RemoveDbgInfoPass() : ModulePass(&ID) {}
    
    void getAnalysisUsage(AnalysisUsage &Info) const {
      Info.setPreservesCFG();
    }

    bool runOnModule(Module &M) {
      bool Changed = false;
      DbgGlobals.clear();

      for (Module::iterator I = M.begin(), E = M.end(); I != E; ) {
        Function *F = I++;
        Changed |= cleanupFunction(*F);
        if (F->hasName() && !strncmp(F->getNameStart(), "llvm.dbg", 8))
          M.getFunctionList().erase(F);
      }

      for (Module::global_iterator GVI = M.global_begin(), E = M.global_end();
           GVI != E; ) {
        GlobalVariable *GV = GVI++;
        if (GV->hasName() && !strncmp(GV->getNameStart(), "llvm.dbg", 8)) {
          if (GV->hasInitializer())
            CollectDbgGlobals(GV->getInitializer());
          GV->replaceAllUsesWith(UndefValue::get(GV->getType()));
          GV->removeDeadConstantUsers();
          M.getGlobalList().erase(GV);
        }
      }

      for (SmallPtrSet<GlobalVariable *,8>::iterator CI = DbgGlobals.begin(),
             CE = DbgGlobals.end(); CI != CE; ) {
        GlobalVariable *GV = *CI++;
        GV->removeDeadConstantUsers();
        if (GV->use_empty())
          M.getGlobalList().erase(GV);
      }
      return Changed;
    }

    void CollectDbgGlobals(Constant *C) {
      if (GlobalVariable *GV = dyn_cast<GlobalVariable>(C))
        DbgGlobals.insert(GV);
      for (User::op_iterator I = C->op_begin(), E = C->op_end(); I != E; ++I) 
        CollectDbgGlobals(cast<Constant>(*I));
    }

    bool cleanupFunction(Function &F) {
      SmallVector<Instruction *, 8> WorkList;
      for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB)
        for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I)
          if (isa<DbgInfoIntrinsic>(I))
            WorkList.push_back(I);

      for (SmallVector<Instruction *, 8>::iterator WI = WorkList.begin(),
             WE = WorkList.end(); WI != WE; ++WI)
        (*WI)->eraseFromParent();
     
      return !WorkList.empty();
    }
    
  private:
    SmallPtrSet<GlobalVariable *, 8> DbgGlobals;
  };
  
  char RemoveDbgInfoPass::ID = 0;
  static RegisterPass<RemoveDbgInfoPass> X("remove-dbginfo",
                                           "Remove Debugging Information");
}

ModulePass *llvm::createRemoveDbgInfoPass() {
  return new RemoveDbgInfoPass();
}
