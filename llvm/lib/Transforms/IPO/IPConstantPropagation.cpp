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
#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/CallSite.h"
#include "llvm/ADT/Statistic.h"
using namespace llvm;

namespace {
  Statistic<> NumArgumentsProped("ipconstprop",
                                 "Number of args turned into constants");
  Statistic<> NumReturnValProped("ipconstprop",
                              "Number of return values turned into constants");

  /// IPCP - The interprocedural constant propagation pass
  ///
  struct IPCP : public ModulePass {
    bool runOnModule(Module &M);
  private:
    bool PropagateConstantsIntoArguments(Function &F);
    bool PropagateConstantReturn(Function &F);
  };
  RegisterPass<IPCP> X("ipconstprop", "Interprocedural constant propagation");
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
      if (!I->isExternal()) {
        // Delete any klingons.
        I->removeDeadConstantUsers();
        if (I->hasInternalLinkage())
          LocalChange |= PropagateConstantsIntoArguments(*I);
        Changed |= PropagateConstantReturn(*I);
      }
    Changed |= LocalChange;
  }
  return Changed;
}

/// PropagateConstantsIntoArguments - Look at all uses of the specified
/// function.  If all uses are direct call sites, and all pass a particular
/// constant in for an argument, propagate that constant in as the argument.
///
bool IPCP::PropagateConstantsIntoArguments(Function &F) {
  if (F.arg_empty() || F.use_empty()) return false; // No arguments? Early exit.

  std::vector<std::pair<Constant*, bool> > ArgumentConstants;
  ArgumentConstants.resize(F.arg_size());

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
      Function::arg_iterator Arg = F.arg_begin();
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
  Function::arg_iterator AI = F.arg_begin();
  bool MadeChange = false;
  for (unsigned i = 0, e = ArgumentConstants.size(); i != e; ++i, ++AI)
    // Do we have a constant argument!?
    if (!ArgumentConstants[i].second && !AI->use_empty()) {
      Value *V = ArgumentConstants[i].first;
      if (V == 0) V = UndefValue::get(AI->getType());
      AI->replaceAllUsesWith(V);
      ++NumArgumentsProped;
      MadeChange = true;
    }
  return MadeChange;
}


// Check to see if this function returns a constant.  If so, replace all callers
// that user the return value with the returned valued.  If we can replace ALL
// callers,
bool IPCP::PropagateConstantReturn(Function &F) {
  if (F.getReturnType() == Type::VoidTy)
    return false; // No return value.

  // Check to see if this function returns a constant.
  Value *RetVal = 0;
  for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB)
    if (ReturnInst *RI = dyn_cast<ReturnInst>(BB->getTerminator()))
      if (isa<UndefValue>(RI->getOperand(0))) {
        // Ignore.
      } else if (Constant *C = dyn_cast<Constant>(RI->getOperand(0))) {
        if (RetVal == 0)
          RetVal = C;
        else if (RetVal != C)
          return false;  // Does not return the same constant.
      } else {
        return false;  // Does not return a constant.
      }

  if (RetVal == 0) RetVal = UndefValue::get(F.getReturnType());

  // If we got here, the function returns a constant value.  Loop over all
  // users, replacing any uses of the return value with the returned constant.
  bool ReplacedAllUsers = true;
  bool MadeChange = false;
  for (Value::use_iterator I = F.use_begin(), E = F.use_end(); I != E; ++I)
    if (!isa<Instruction>(*I))
      ReplacedAllUsers = false;
    else {
      CallSite CS = CallSite::get(cast<Instruction>(*I));
      if (CS.getInstruction() == 0 ||
          CS.getCalledFunction() != &F) {
        ReplacedAllUsers = false;
      } else {
        if (!CS.getInstruction()->use_empty()) {
          CS.getInstruction()->replaceAllUsesWith(RetVal);
          MadeChange = true;
        }
      }
    }

  // If we replace all users with the returned constant, and there can be no
  // other callers of the function, replace the constant being returned in the
  // function with an undef value.
  if (ReplacedAllUsers && F.hasInternalLinkage() && !isa<UndefValue>(RetVal)) {
    Value *RV = UndefValue::get(RetVal->getType());
    for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB)
      if (ReturnInst *RI = dyn_cast<ReturnInst>(BB->getTerminator())) {
        if (RI->getOperand(0) != RV) {
          RI->setOperand(0, RV);
          MadeChange = true;
        }
      }
  }

  if (MadeChange) ++NumReturnValProped;
  return MadeChange;
}
