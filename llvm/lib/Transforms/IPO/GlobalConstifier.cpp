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
using namespace llvm;

namespace {
  Statistic<> NumMarked("constify", "Number of globals marked constant");

  struct Constifier : public Pass {
    bool run(Module &M);
  };

  RegisterOpt<Constifier> X("constify", "Global Constifier");
}

Pass *llvm::createGlobalConstifierPass() { return new Constifier(); }

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

bool Constifier::run(Module &M) {
  bool Changed = false;
  std::set<PHINode*> PHIUsers;
  for (Module::giterator GV = M.gbegin(), E = M.gend(); GV != E; ++GV)
    if (!GV->isConstant() && GV->hasInternalLinkage() && GV->hasInitializer()) {
      if (!isStoredThrough(GV, PHIUsers)) {
        DEBUG(std::cerr << "MARKING CONSTANT: " << *GV << "\n");
        GV->setConstant(true);
        ++NumMarked;
        Changed = true;
      }
      PHIUsers.clear();
    }
  return Changed;
}
