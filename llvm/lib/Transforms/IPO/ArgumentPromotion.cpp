//===-- ArgumentPromotion.cpp - Promote by-reference arguments ------------===//
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
// of code and lead to the elimination of allocas (especially in C++ template
// code like the STL).
//
// This pass also handles aggregate arguments that are passed into a function,
// scalarizing them if the elements of the aggregate are only loaded.  Note that
// we refuse to scalarize aggregates which would require passing in more than
// three operands to the function, because we don't want to pass thousands of
// operands for a large array or structure!
//
// Note that this transformation could also be done for arguments that are only
// stored to (returning the value instead), but we do not currently handle that
// case.  This case would be best handled when and if we start supporting
// multiple return values from functions.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "argpromotion"
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
#include "Support/StringExtras.h"
#include <set>
using namespace llvm;

namespace {
  Statistic<> NumArgumentsPromoted("argpromotion",
                                   "Number of pointer arguments promoted");
  Statistic<> NumAggregatesPromoted("argpromotion",
                                    "Number of aggregate arguments promoted");
  Statistic<> NumArgumentsDead("argpromotion",
                               "Number of dead pointer args eliminated");

  /// ArgPromotion - The 'by reference' to 'by value' argument promotion pass.
  ///
  class ArgPromotion : public Pass {
    // WorkList - The set of internal functions that we have yet to process.  As
    // we eliminate arguments from a function, we push all callers into this set
    // so that the by-reference argument can be bubbled out as far as possible.
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

/// PromoteArguments - This method checks the specified function to see if there
/// are any promotable arguments and if it is safe to promote the function (for
/// example, all callers are direct).  If safe to promote some arguments, it
/// calls the DoPromotion method.
///
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
       UI != E; ++UI) {
    CallSite CS = CallSite::get(*UI);
    if (!CS.getInstruction())       // "Taking the address" of the function
      return false;

    // Ensure that this call site is CALLING the function, not passing it as
    // an argument.
    for (CallSite::arg_iterator AI = CS.arg_begin(), E = CS.arg_end();
         AI != E; ++AI)
      if (*AI == F) return false;   // Passing the function address in!
  }

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


/// isSafeToPromoteArgument - As you might guess from the name of this method,
/// it checks to see if it is both safe and useful to promote the argument.
/// This method limits promotion of aggregates to only promote up to three
/// elements of the aggregate in order to avoid exploding the number of
/// arguments passed in.
bool ArgPromotion::isSafeToPromoteArgument(Argument *Arg) const {
  // We can only promote this argument if all of the uses are loads, or are GEP
  // instructions (with constant indices) that are subsequently loaded.
  std::vector<LoadInst*> Loads;
  std::vector<std::vector<Constant*> > GEPIndices;
  for (Value::use_iterator UI = Arg->use_begin(), E = Arg->use_end();
       UI != E; ++UI)
    if (LoadInst *LI = dyn_cast<LoadInst>(*UI)) {
      if (LI->isVolatile()) return false;  // Don't hack volatile loads
      Loads.push_back(LI);
    } else if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(*UI)) {
      if (GEP->use_empty()) {
        // Dead GEP's cause trouble later.  Just remove them if we run into
        // them.
        getAnalysis<AliasAnalysis>().deleteValue(GEP);
        GEP->getParent()->getInstList().erase(GEP);
        return isSafeToPromoteArgument(Arg);
      }
      // Ensure that all of the indices are constants.
      std::vector<Constant*> Operands;
      for (unsigned i = 1, e = GEP->getNumOperands(); i != e; ++i)
        if (Constant *C = dyn_cast<Constant>(GEP->getOperand(i)))
          Operands.push_back(C);
        else
          return false;  // Not a constant operand GEP!

      // Ensure that the only users of the GEP are load instructions.
      for (Value::use_iterator UI = GEP->use_begin(), E = GEP->use_end();
           UI != E; ++UI)
        if (LoadInst *LI = dyn_cast<LoadInst>(*UI)) {
          if (LI->isVolatile()) return false;  // Don't hack volatile loads
          Loads.push_back(LI);
        } else {
          return false;
        }

      // See if there is already a GEP with these indices.  If not, check to
      // make sure that we aren't promoting too many elements.  If so, nothing
      // to do.
      if (std::find(GEPIndices.begin(), GEPIndices.end(), Operands) ==
          GEPIndices.end()) {
        if (GEPIndices.size() == 3) {
          DEBUG(std::cerr << "argpromotion disable promoting argument '"
                << Arg->getName() << "' because it would require adding more "
                << "than 3 arguments to the function.\n");
          // We limit aggregate promotion to only promoting up to three elements
          // of the aggregate.
          return false;
        }
        GEPIndices.push_back(Operands);
      }
    } else {
      return false;  // Not a load or a GEP.
    }

  if (Loads.empty()) return true;  // No users, this is a dead argument.

  // Okay, now we know that the argument is only used by load instructions.  Use
  // alias analysis to check to see if the pointer is guaranteed to not be
  // modified from entry of the function to each of the load instructions.
  Function &F = *Arg->getParent();

  // Because there could be several/many load instructions, remember which
  // blocks we know to be transparent to the load.
  std::set<BasicBlock*> TranspBlocks;

  AliasAnalysis &AA = getAnalysis<AliasAnalysis>();
  TargetData &TD = getAnalysis<TargetData>();

  for (unsigned i = 0, e = Loads.size(); i != e; ++i) {
    // Check to see if the load is invalidated from the start of the block to
    // the load itself.
    LoadInst *Load = Loads[i];
    BasicBlock *BB = Load->getParent();

    const PointerType *LoadTy =
      cast<PointerType>(Load->getOperand(0)->getType());
    unsigned LoadSize = TD.getTypeSize(LoadTy->getElementType());

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

/// DoPromotion - This method actually performs the promotion of the specified
/// arguments.  At this point, we know that it's safe to do so.
void ArgPromotion::DoPromotion(Function *F, std::vector<Argument*> &Args2Prom) {
  std::set<Argument*> ArgsToPromote(Args2Prom.begin(), Args2Prom.end());
  
  // Start by computing a new prototype for the function, which is the same as
  // the old function, but has modified arguments.
  const FunctionType *FTy = F->getFunctionType();
  std::vector<const Type*> Params;

  // ScalarizedElements - If we are promoting a pointer that has elements
  // accessed out of it, keep track of which elements are accessed so that we
  // can add one argument for each.
  //
  // Arguments that are directly loaded will have a zero element value here, to
  // handle cases where there are both a direct load and GEP accesses.
  //
  std::map<Argument*, std::set<std::vector<Value*> > > ScalarizedElements;

  // OriginalLoads - Keep track of a representative load instruction from the
  // original function so that we can tell the alias analysis implementation
  // what the new GEP/Load instructions we are inserting look like.
  std::map<std::vector<Value*>, LoadInst*> OriginalLoads;

  for (Function::aiterator I = F->abegin(), E = F->aend(); I != E; ++I)
    if (!ArgsToPromote.count(I)) {
      Params.push_back(I->getType());
    } else if (I->use_empty()) {
      ++NumArgumentsDead;
    } else {
      // Okay, this is being promoted.  Check to see if there are any GEP uses
      // of the argument.
      std::set<std::vector<Value*> > &ArgIndices = ScalarizedElements[I];
      for (Value::use_iterator UI = I->use_begin(), E = I->use_end(); UI != E;
           ++UI) {
        Instruction *User = cast<Instruction>(*UI);
        assert(isa<LoadInst>(User) || isa<GetElementPtrInst>(User));
        std::vector<Value*> Indices(User->op_begin()+1, User->op_end());
        ArgIndices.insert(Indices);
        LoadInst *OrigLoad;
        if (LoadInst *L = dyn_cast<LoadInst>(User))
          OrigLoad = L;
        else
          OrigLoad = cast<LoadInst>(User->use_back());
        OriginalLoads[Indices] = OrigLoad;
      }

      // Add a parameter to the function for each element passed in.
      for (std::set<std::vector<Value*> >::iterator SI = ArgIndices.begin(),
             E = ArgIndices.end(); SI != E; ++SI)
        Params.push_back(GetElementPtrInst::getIndexedType(I->getType(), *SI));

      if (ArgIndices.size() == 1 && ArgIndices.begin()->empty())
        ++NumArgumentsPromoted;
      else
        ++NumAggregatesPromoted;
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

  // Get the alias analysis information that we need to update to reflect our
  // changes.
  AliasAnalysis &AA = getAnalysis<AliasAnalysis>();

  // Loop over all of the callers of the function, transforming the call sites
  // to pass in the loaded pointers.
  //
  std::vector<Value*> Args;
  while (!F->use_empty()) {
    CallSite CS = CallSite::get(F->use_back());
    Instruction *Call = CS.getInstruction();

    // Make sure the caller of this function is revisited now that we promoted
    // arguments in a callee of it.
    if (Call->getParent()->getParent()->hasInternalLinkage())
      WorkList.insert(Call->getParent()->getParent());
    
    // Loop over the operands, inserting GEP and loads in the caller as
    // appropriate.
    CallSite::arg_iterator AI = CS.arg_begin();
    for (Function::aiterator I = F->abegin(), E = F->aend(); I != E; ++I, ++AI)
      if (!ArgsToPromote.count(I))
        Args.push_back(*AI);          // Unmodified argument
      else if (!I->use_empty()) {
        // Non-dead argument: insert GEPs and loads as appropriate.
        std::set<std::vector<Value*> > &ArgIndices = ScalarizedElements[I];
        for (std::set<std::vector<Value*> >::iterator SI = ArgIndices.begin(),
               E = ArgIndices.end(); SI != E; ++SI) {
          Value *V = *AI;
          LoadInst *OrigLoad = OriginalLoads[*SI];
          if (!SI->empty()) {
            V = new GetElementPtrInst(V, *SI, V->getName()+".idx", Call);
            AA.copyValue(OrigLoad->getOperand(0), V);
          }
          Args.push_back(new LoadInst(V, V->getName()+".val", Call));
          AA.copyValue(OrigLoad, Args.back());
        }
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

    // Update the alias analysis implementation to know that we are replacing
    // the old call with a new one.
    AA.replaceWithNewValue(Call, New);

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
      AA.replaceWithNewValue(I, I2);
      ++I2;
    } else if (I->use_empty()) {
      AA.deleteValue(I);
    } else {
      // Otherwise, if we promoted this argument, then all users are load
      // instructions, and all loads should be using the new argument that we
      // added.
      std::set<std::vector<Value*> > &ArgIndices = ScalarizedElements[I];

      while (!I->use_empty()) {
        if (LoadInst *LI = dyn_cast<LoadInst>(I->use_back())) {
          assert(ArgIndices.begin()->empty() &&
                 "Load element should sort to front!");
          I2->setName(I->getName()+".val");
          LI->replaceAllUsesWith(I2);
          AA.replaceWithNewValue(LI, I2);
          LI->getParent()->getInstList().erase(LI);
          DEBUG(std::cerr << "*** Promoted load of argument '" << I->getName()
                          << "' in function '" << F->getName() << "'\n");
        } else {
          GetElementPtrInst *GEP = cast<GetElementPtrInst>(I->use_back());
          std::vector<Value*> Operands(GEP->op_begin()+1, GEP->op_end());

          unsigned ArgNo = 0;
          Function::aiterator TheArg = I2;
          for (std::set<std::vector<Value*> >::iterator It = ArgIndices.begin();
               *It != Operands; ++It, ++TheArg) {
            assert(It != ArgIndices.end() && "GEP not handled??");
          }

          std::string NewName = I->getName();
          for (unsigned i = 0, e = Operands.size(); i != e; ++i)
            if (ConstantInt *CI = dyn_cast<ConstantInt>(Operands[i]))
              NewName += "."+itostr((int64_t)CI->getRawValue());
            else
              NewName += ".x";
          TheArg->setName(NewName+".val");

          DEBUG(std::cerr << "*** Promoted agg argument '" << TheArg->getName()
                          << "' of function '" << F->getName() << "'\n");

          // All of the uses must be load instructions.  Replace them all with
          // the argument specified by ArgNo.
          while (!GEP->use_empty()) {
            LoadInst *L = cast<LoadInst>(GEP->use_back());
            L->replaceAllUsesWith(TheArg);
            AA.replaceWithNewValue(L, TheArg);
            L->getParent()->getInstList().erase(L);
          }
          AA.deleteValue(GEP);
          GEP->getParent()->getInstList().erase(GEP);
        }
      }

      // If we inserted a new pointer type, it's possible that IT could be
      // promoted too.  Also, increment I2 past all of the arguments added for
      // this promoted pointer.
      for (unsigned i = 0, e = ArgIndices.size(); i != e; ++i, ++I2)
        if (isa<PointerType>(I2->getType()))
          WorkList.insert(NF);
    }

  // Notify the alias analysis implementation that we inserted a new argument.
  if (ExtraArgHack)
    AA.copyValue(Constant::getNullValue(Type::IntTy), NF->abegin());


  // Tell the alias analysis that the old function is about to disappear.
  AA.replaceWithNewValue(F, NF);

  // Now that the old function is dead, delete it.
  F->getParent()->getFunctionList().erase(F);
}
