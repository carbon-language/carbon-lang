//===-- IPConstantPropagation.cpp - Propagate constants through calls -----===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This pass implements an _extremely_ simple interprocedural constant
// propagation pass.  It could certainly be improved in many different ways,
// like using a worklist.  This pass makes arguments dead, but does not remove
// them.  The existing dead argument elimination pass should be run after this
// to clean up the mess.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Constants.h"
#include "llvm/Support/CallSite.h"
#include "llvm/ADT/Statistic.h"
using namespace llvm;

namespace {
  Statistic<> NumArgumentsProped("ipconstprop",
                                 "Number of args turned into constants");

  /// IPCP - The interprocedural constant propagation pass
  ///
  struct IPCP : public ModulePass {
    bool runOnModule(Module &M);
  private:
    bool processFunction(Function &F);
  };
  RegisterOpt<IPCP> X("ipconstprop", "Interprocedural constant propagation");
}

ModulePass *llvm::createIPConstantPropagationPass() { return new IPCP(); }

bool IPCP::runOnModule(Module &M) {
  bool Changed = false;
  bool LocalChange = true;

  // FIXME: instead of using smart algorithms, we just iterate until we stop
  // making changes.
  while (LocalChange) {
    LocalChange = false;
    for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
      if (!I->isExternal() && I->hasInternalLinkage())
        LocalChange |= processFunction(*I);
    Changed |= LocalChange;
  }
  return Changed;
}

/// processFunction - Look at all uses of the specified function.  If all uses
/// are direct call sites, and all pass a particular constant in for an
/// argument, propagate that constant in as the argument.
///
bool IPCP::processFunction(Function &F) {
  if (F.aempty() || F.use_empty()) return false;  // No arguments?  Early exit.

  // Delete any klingons.
  F.removeDeadConstantUsers();

  std::vector<std::pair<Constant*, bool> > ArgumentConstants;
  ArgumentConstants.resize(F.asize());

  unsigned NumNonconstant = 0;

  for (Value::use_iterator I = F.use_begin(), E = F.use_end(); I != E; ++I)
    if (!isa<Instruction>(*I))
      return false;  // Used by a non-instruction, do not transform
    else {
      CallSite CS = CallSite::get(cast<Instruction>(*I));
      if (CS.getInstruction() == 0 || 
          CS.getCalledFunction() != &F)
        return false;  // Not a direct call site?
      
      // Check out all of the potentially constant arguments
      CallSite::arg_iterator AI = CS.arg_begin();
      Function::aiterator Arg = F.abegin();
      for (unsigned i = 0, e = ArgumentConstants.size(); i != e;
           ++i, ++AI, ++Arg) {
        if (*AI == &F) return false;  // Passes the function into itself

        if (!ArgumentConstants[i].second) {
          if (Constant *C = dyn_cast<Constant>(*AI)) {
            if (!ArgumentConstants[i].first)
              ArgumentConstants[i].first = C;
            else if (ArgumentConstants[i].first != C) {
              // Became non-constant
              ArgumentConstants[i].second = true;
              ++NumNonconstant;
              if (NumNonconstant == ArgumentConstants.size()) return false;
            }
          } else if (*AI != &*Arg) {    // Ignore recursive calls with same arg
            // This is not a constant argument.  Mark the argument as
            // non-constant.
            ArgumentConstants[i].second = true;
            ++NumNonconstant;
            if (NumNonconstant == ArgumentConstants.size()) return false;
          }
        }
      }
    }

  // If we got to this point, there is a constant argument!
  assert(NumNonconstant != ArgumentConstants.size());
  Function::aiterator AI = F.abegin();
  bool MadeChange = false;
  for (unsigned i = 0, e = ArgumentConstants.size(); i != e; ++i, ++AI)
    // Do we have a constant argument!?
    if (!ArgumentConstants[i].second && !AI->use_empty()) {
      assert(ArgumentConstants[i].first && "Unknown constant value!");
      Value *V = ArgumentConstants[i].first;
      AI->replaceAllUsesWith(V);
      ++NumArgumentsProped;
      MadeChange = true;
    }
  return MadeChange;
}

