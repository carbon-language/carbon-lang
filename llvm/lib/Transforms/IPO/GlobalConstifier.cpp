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
#include "Support/Debug.h"
#include "Support/Statistic.h"
using namespace llvm;

namespace {
  Statistic<> NumMarked("constify", "Number of globals marked constant");

  struct Constifier : public Pass {
    bool run(Module &M);
  };

  RegisterOpt<Constifier> X("constify", "Global Constifier");
}

Pass *llvm::createGlobalConstifierPass() { return new Constifier(); }

/// isStoredThrough - Return false if the specified pointer is provably never
/// stored through.  If we can't tell, we must conservatively assume it might.
///
static bool isStoredThrough(Value *V) {
  for (Value::use_iterator UI = V->use_begin(), E = V->use_end(); UI != E; ++UI)
    if (Constant *C = dyn_cast<Constant>(*UI)) {
      if (ConstantExpr *CE = dyn_cast<ConstantExpr>(C)) {
        if (isStoredThrough(CE))
          return true;
      } else if (ConstantPointerRef *CPR = dyn_cast<ConstantPointerRef>(C)) {
        if (isStoredThrough(CPR)) return true;
      } else {
        // Must be an element of a constant array or something.
        return true;
      }
    } else if (Instruction *I = dyn_cast<Instruction>(*UI)) {
      if (I->getOpcode() == Instruction::GetElementPtr) {
        if (isStoredThrough(I)) return true;
      } else if (!isa<LoadInst>(*UI) && !isa<SetCondInst>(*UI))
        return true;  // Any other non-load instruction might store!
    } else {
      // Otherwise must be a global or some other user.
      return true;
    }

  return false;
}

bool Constifier::run(Module &M) {
  bool Changed = false;
  for (Module::giterator GV = M.gbegin(), E = M.gend(); GV != E; ++GV)
    if (!GV->isConstant() && GV->hasInternalLinkage() && GV->hasInitializer()) {
      if (!isStoredThrough(GV)) {
        DEBUG(std::cerr << "MARKING CONSTANT: " << *GV << "\n");
        GV->setConstant(true);
        ++NumMarked;
        Changed = true;
      }
    }
  return Changed;
}
