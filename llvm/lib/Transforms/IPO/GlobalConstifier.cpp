//===- GlobalConstifier.cpp - Mark read-only globals constant -------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This pass loops over the non-constant internal global variables in the
// program.  If it can prove that they are never written to, it marks them
// constant.
//
// NOTE: this should eventually use the alias analysis interfaces to do the
// transformation, but for now we just stick with a simple solution. DSA in
// particular could give a much more accurate answer to the mod/ref query, but
// it's not quite ready for this.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO.h"
#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/Statistic.h"
#include <set>
#include <algorithm>
using namespace llvm;

namespace {
  Statistic<> NumMarked ("constify", "Number of globals marked constant");
  Statistic<> NumDeleted("constify", "Number of globals deleted");

  struct Constifier : public ModulePass {
    bool runOnModule(Module &M);
  };

  RegisterOpt<Constifier> X("constify", "Global Constifier");
}

ModulePass *llvm::createGlobalConstifierPass() { return new Constifier(); }

/// A lot of global constants are stored only in trivially dead setter
/// functions.  Because we don't want to cycle between globaldce and this pass,
/// just do a simple check to catch the common case.
static bool ContainingFunctionIsTriviallyDead(Instruction *I) {
  Function *F = I->getParent()->getParent();
  if (!F->hasInternalLinkage()) return false;
  F->removeDeadConstantUsers();
  return F->use_empty();
}


/// isStoredThrough - Return false if the specified pointer is provably never
/// stored through.  If we can't tell, we must conservatively assume it might.
///
static bool isStoredThrough(Value *V, std::set<PHINode*> &PHIUsers) {
  for (Value::use_iterator UI = V->use_begin(), E = V->use_end(); UI != E; ++UI)
    if (ConstantExpr *CE = dyn_cast<ConstantExpr>(*UI)) {
      if (isStoredThrough(CE, PHIUsers))
        return true;
    } else if (Instruction *I = dyn_cast<Instruction>(*UI)) {
      if (!ContainingFunctionIsTriviallyDead(I)) {
        if (I->getOpcode() == Instruction::GetElementPtr ||
            I->getOpcode() == Instruction::Select) {
          if (isStoredThrough(I, PHIUsers)) return true;
        } else if (PHINode *PN = dyn_cast<PHINode>(I)) {
          // PHI nodes we can check just like select or GEP instructions, but we
          // have to be careful about infinite recursion.
          if (PHIUsers.insert(PN).second)  // Not already visited.
            if (isStoredThrough(I, PHIUsers)) return true;
        } else if (StoreInst *SI = dyn_cast<StoreInst>(I)) {
          // If this store is just storing the initializer into a global
          // (i.e. not changing the value), ignore it.  For now we just handle
          // direct stores, no stores to fields of aggregates.
          if (!isa<GlobalVariable>(SI->getOperand(1)))
            return true;
          Constant *GVInit =
            cast<GlobalVariable>(SI->getOperand(1))->getInitializer();
          if (SI->getOperand(0) != GVInit)
            return true;
        } else if (!isa<LoadInst>(I) && !isa<SetCondInst>(I)) {
          return true;  // Any other non-load instruction might store!
        }
      }
    } else {
      // Otherwise must be a global or some other user.
      return true;
    }

  return false;
}

/// CleanupConstantGlobalUsers - We just marked GV constant.  Loop over all
/// users of the global, cleaning up the obvious ones.  This is largely just a
/// quick scan over the use list to clean up the easy and obvious cruft.
static void CleanupConstantGlobalUsers(GlobalVariable *GV) {
  Constant *Init = GV->getInitializer();
  if (!Init->getType()->isFirstClassType())
    return;  // We can't simplify aggregates yet!

  std::vector<User*> Users(GV->use_begin(), GV->use_end());

  std::sort(Users.begin(), Users.end());
  Users.erase(std::unique(Users.begin(), Users.end()), Users.end());
  for (unsigned i = 0, e = Users.size(); i != e; ++i) {
    if (LoadInst *LI = dyn_cast<LoadInst>(Users[i])) {
      // Replace the load with the initializer.
      LI->replaceAllUsesWith(Init);
      LI->getParent()->getInstList().erase(LI);
    } else if (StoreInst *SI = dyn_cast<StoreInst>(Users[i])) {
      // Store must be unreachable or storing Init into the global.
      SI->getParent()->getInstList().erase(SI);
    }
  }
}


bool Constifier::runOnModule(Module &M) {
  bool Changed = false;
  std::set<PHINode*> PHIUsers;
  for (Module::giterator GVI = M.gbegin(), E = M.gend(); GVI != E;) {
    GlobalVariable *GV = GVI++;
    if (!GV->isConstant() && GV->hasInternalLinkage() && GV->hasInitializer()) {
      if (!isStoredThrough(GV, PHIUsers)) {
        DEBUG(std::cerr << "MARKING CONSTANT: " << *GV << "\n");
        GV->setConstant(true);
        
        // Clean up any obviously simplifiable users now.
        CleanupConstantGlobalUsers(GV);

        // If the global is dead now, just nuke it.
        if (GV->use_empty()) {
          M.getGlobalList().erase(GV);
          ++NumDeleted;
        }

        ++NumMarked;
        Changed = true;
      }
      PHIUsers.clear();
    }
  }
  return Changed;
}
