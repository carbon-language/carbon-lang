//===-- ArgumentPromotion.cpp - Promote 'by reference' arguments ----------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This pass promotes "by reference" arguments to be "by value" arguments.  In
// practice, this means looking for internal functions that have pointer
// arguments.  If we can prove, through the use of alias analysis, that that an
// argument is *only* loaded, then we can pass the value into the function
// instead of the address of the value.  This can cause recursive simplification
// of code, and lead to the elimination of allocas, especially in C++ template
// code like the STL.
//
// Note that this transformation could also be done for arguments that are only
// stored to (returning the value instead), but we do not currently handle that
// case.
//
// Note that we should be able to promote pointers to structures that are only
// loaded from as well.  The danger is creating way to many arguments, so this
// transformation should be limited to 3 element structs or something.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Instructions.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Support/CFG.h"
#include "Support/Debug.h"
#include "Support/DepthFirstIterator.h"
#include "Support/Statistic.h"
#include <set>
using namespace llvm;

namespace {
  Statistic<> NumArgumentsPromoted("argpromotion",
                                   "Number of pointer arguments promoted");
  Statistic<> NumArgumentsDead("argpromotion",
                               "Number of dead pointer args eliminated");

  /// ArgPromotion - The 'by reference' to 'by value' argument promotion pass.
  ///
  class ArgPromotion : public Pass {
    // WorkList - The set of internal functions that we have yet to process.  As
    // we eliminate arguments from a function, we push all callers into this set
    // so that the by reference argument can be bubbled out as far as possible.
    // This set contains only internal functions.
    std::set<Function*> WorkList;
  public:
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<AliasAnalysis>();
      AU.addRequired<TargetData>();
    }

    virtual bool run(Module &M);
  private:
    bool PromoteArguments(Function *F);
    bool isSafeToPromoteArgument(Argument *Arg) const;  
    void DoPromotion(Function *F, std::vector<Argument*> &ArgsToPromote);
  };

  RegisterOpt<ArgPromotion> X("argpromotion",
                              "Promote 'by reference' arguments to scalars");
}

Pass *llvm::createArgumentPromotionPass() {
  return new ArgPromotion();
}

bool ArgPromotion::run(Module &M) {
  bool Changed = false;
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
    if (I->hasInternalLinkage()) {
      WorkList.insert(I);

      // If there are any constant pointer refs pointing to this function,
      // eliminate them now if possible.
      ConstantPointerRef *CPR = 0;
      for (Value::use_iterator UI = I->use_begin(), E = I->use_end(); UI != E;
           ++UI)
        if ((CPR = dyn_cast<ConstantPointerRef>(*UI)))
          break;  // Found one!
      if (CPR) {
        // See if we can transform all users to use the function directly.
        while (!CPR->use_empty()) {
          User *TheUser = CPR->use_back();
          if (!isa<Constant>(TheUser) && !isa<GlobalVariable>(TheUser)) {
            Changed = true;
            TheUser->replaceUsesOfWith(CPR, I);
          } else {
            // We won't be able to eliminate all users.  :(
            WorkList.erase(I);  // Minor efficiency win.
            break;
          }
        }

        // If we nuked all users of the CPR, kill the CPR now!
        if (CPR->use_empty()) {
          CPR->destroyConstant();
          Changed = true;
        }
      }
    }
  
  while (!WorkList.empty()) {
    Function *F = *WorkList.begin();
    WorkList.erase(WorkList.begin());

    if (PromoteArguments(F))    // Attempt to promote an argument.
      Changed = true;           // Remember that we changed something.
  }
  
  return Changed;
}


bool ArgPromotion::PromoteArguments(Function *F) {
  assert(F->hasInternalLinkage() && "We can only process internal functions!");

  // First check: see if there are any pointer arguments!  If not, quick exit.
  std::vector<Argument*> PointerArgs;
  for (Function::aiterator I = F->abegin(), E = F->aend(); I != E; ++I)
    if (isa<PointerType>(I->getType()))
      PointerArgs.push_back(I);
  if (PointerArgs.empty()) return false;

  // Second check: make sure that all callers are direct callers.  We can't
  // transform functions that have indirect callers.
  for (Value::use_iterator UI = F->use_begin(), E = F->use_end();
       UI != E; ++UI)
    // What about CPRs?
    if (!CallSite::get(*UI).getInstruction())
      return false;  // Cannot promote an indirect call!

  // Check to see which arguments are promotable.  If an argument is not
  // promotable, remove it from the PointerArgs vector.
  for (unsigned i = 0; i != PointerArgs.size(); ++i)
    if (!isSafeToPromoteArgument(PointerArgs[i])) {
      std::swap(PointerArgs[i--], PointerArgs.back());
      PointerArgs.pop_back();
    }

  // No promotable pointer arguments.
  if (PointerArgs.empty()) return false;

  // Okay, promote all of the arguments are rewrite the callees!
  DoPromotion(F, PointerArgs);
  return true;
}

bool ArgPromotion::isSafeToPromoteArgument(Argument *Arg) const {
  // We can only promote this argument if all of the uses are loads...
  std::vector<LoadInst*> Loads;
  for (Value::use_iterator UI = Arg->use_begin(), E = Arg->use_end();
       UI != E; ++UI)
    if (LoadInst *LI = dyn_cast<LoadInst>(*UI)) {
      if (LI->isVolatile()) return false;  // Don't hack volatile loads
      Loads.push_back(LI);
    } else
      return false;

  if (Loads.empty()) return true;  // No users, dead argument.

  const Type *LoadTy = cast<PointerType>(Arg->getType())->getElementType();
  unsigned LoadSize = getAnalysis<TargetData>().getTypeSize(LoadTy);

  // Okay, now we know that the argument is only used by load instructions.
  // Check to see if the pointer is guaranteed to not be modified from entry of
  // the function to each of the load instructions.
  Function &F = *Arg->getParent();

  // Because there could be several/many load instructions, remember which
  // blocks we know to be transparent to the load.
  std::set<BasicBlock*> TranspBlocks;

  AliasAnalysis &AA = getAnalysis<AliasAnalysis>();

  for (unsigned i = 0, e = Loads.size(); i != e; ++i) {
    // Check to see if the load is invalidated from the start of the block to
    // the load itself.
    LoadInst *Load = Loads[i];
    BasicBlock *BB = Load->getParent();
    if (AA.canInstructionRangeModify(BB->front(), *Load, Arg, LoadSize))
      return false;  // Pointer is invalidated!

    // Now check every path from the entry block to the load for transparency.
    // To do this, we perform a depth first search on the inverse CFG from the
    // loading block.
    for (pred_iterator PI = pred_begin(BB), E = pred_end(BB); PI != E; ++PI)
      for (idf_ext_iterator<BasicBlock*> I = idf_ext_begin(*PI, TranspBlocks),
             E = idf_ext_end(*PI, TranspBlocks); I != E; ++I)
        if (AA.canBasicBlockModify(**I, Arg, LoadSize))
          return false;
  }

  // If the path from the entry of the function to each load is free of
  // instructions that potentially invalidate the load, we can make the
  // transformation!
  return true;
}


void ArgPromotion::DoPromotion(Function *F, std::vector<Argument*> &Args2Prom) {
  std::set<Argument*> ArgsToPromote(Args2Prom.begin(), Args2Prom.end());
  
  // Start by computing a new prototype for the function, which is the same as
  // the old function, but has modified arguments.
  const FunctionType *FTy = F->getFunctionType();
  std::vector<const Type*> Params;

  for (Function::aiterator I = F->abegin(), E = F->aend(); I != E; ++I)
    if (!ArgsToPromote.count(I)) {
      Params.push_back(I->getType());
    } else if (!I->use_empty()) {
      Params.push_back(cast<PointerType>(I->getType())->getElementType());
      ++NumArgumentsPromoted;
    } else {
      ++NumArgumentsDead;
    }

  const Type *RetTy = FTy->getReturnType();

  // Work around LLVM bug PR56: the CWriter cannot emit varargs functions which
  // have zero fixed arguments.
  bool ExtraArgHack = false;
  if (Params.empty() && FTy->isVarArg()) {
    ExtraArgHack = true;
    Params.push_back(Type::IntTy);
  }
  FunctionType *NFTy = FunctionType::get(RetTy, Params, FTy->isVarArg());
  
   // Create the new function body and insert it into the module...
  Function *NF = new Function(NFTy, F->getLinkage(), F->getName());
  F->getParent()->getFunctionList().insert(F, NF);
  
  // Loop over all of the callers of the function, transforming the call sites
  // to pass in the loaded pointers.
  //
  std::vector<Value*> Args;
  while (!F->use_empty()) {
    CallSite CS = CallSite::get(F->use_back());
    Instruction *Call = CS.getInstruction();

    // Make sure the caller of this function is revisited.
    if (Call->getParent()->getParent()->hasInternalLinkage())
      WorkList.insert(Call->getParent()->getParent());
    
    // Loop over the operands, deleting dead ones...
    CallSite::arg_iterator AI = CS.arg_begin();
    for (Function::aiterator I = F->abegin(), E = F->aend(); I != E; ++I, ++AI)
      if (!ArgsToPromote.count(I))
        Args.push_back(*AI);          // Unmodified argument
      else if (!I->use_empty()) {
        // Non-dead instruction
        Args.push_back(new LoadInst(*AI, (*AI)->getName()+".val", Call));
      }

    if (ExtraArgHack)
      Args.push_back(Constant::getNullValue(Type::IntTy));

    // Push any varargs arguments on the list
    for (; AI != CS.arg_end(); ++AI)
      Args.push_back(*AI);

    Instruction *New;
    if (InvokeInst *II = dyn_cast<InvokeInst>(Call)) {
      New = new InvokeInst(NF, II->getNormalDest(), II->getUnwindDest(),
                           Args, "", Call);
    } else {
      New = new CallInst(NF, Args, "", Call);
    }
    Args.clear();

    if (!Call->use_empty()) {
      Call->replaceAllUsesWith(New);
      std::string Name = Call->getName();
      Call->setName("");
      New->setName(Name);
    }
    
    // Finally, remove the old call from the program, reducing the use-count of
    // F.
    Call->getParent()->getInstList().erase(Call);
  }

  // Since we have now created the new function, splice the body of the old
  // function right into the new function, leaving the old rotting hulk of the
  // function empty.
  NF->getBasicBlockList().splice(NF->begin(), F->getBasicBlockList());

  // Loop over the argument list, transfering uses of the old arguments over to
  // the new arguments, also transfering over the names as well.
  //
  for (Function::aiterator I = F->abegin(), E = F->aend(), I2 = NF->abegin();
       I != E; ++I)
    if (!ArgsToPromote.count(I)) {
      // If this is an unmodified argument, move the name and users over to the
      // new version.
      I->replaceAllUsesWith(I2);
      I2->setName(I->getName());
      ++I2;
    } else if (!I->use_empty()) {
      // Otherwise, if we promoted this argument, then all users are load
      // instructions, and all loads should be using the new argument that we
      // added.
      DEBUG(std::cerr << "*** Promoted argument '" << I->getName()
                      << "' of function '" << F->getName() << "'\n");
      I2->setName(I->getName()+".val");
      while (!I->use_empty()) {
        LoadInst *LI = cast<LoadInst>(I->use_back());
        LI->replaceAllUsesWith(I2);
        LI->getParent()->getInstList().erase(LI);
      }
      ++I2;
    }

  // Now that the old function is dead, delete it.
  F->getParent()->getFunctionList().erase(F);
}
