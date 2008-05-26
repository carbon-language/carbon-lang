//===-- DeadArgumentElimination.cpp - Eliminate dead arguments ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass deletes dead arguments from internal functions.  Dead argument
// elimination removes arguments which are directly dead, as well as arguments
// only passed into function calls as dead arguments of other functions.  This
// pass also deletes dead arguments in a similar way.
//
// This pass is often useful as a cleanup pass to run after aggressive
// interprocedural passes, which add possibly-dead arguments.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "deadargelim"
#include "llvm/Transforms/IPO.h"
#include "llvm/CallingConv.h"
#include "llvm/Constant.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Instructions.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Compiler.h"
#include <map>
#include <set>
using namespace llvm;

STATISTIC(NumArgumentsEliminated, "Number of unread args removed");
STATISTIC(NumRetValsEliminated  , "Number of unused return values removed");

namespace {
  /// DAE - The dead argument elimination pass.
  ///
  class VISIBILITY_HIDDEN DAE : public ModulePass {
    /// Liveness enum - During our initial pass over the program, we determine
    /// that things are either definately alive, definately dead, or in need of
    /// interprocedural analysis (MaybeLive).
    ///
    enum Liveness { Live, MaybeLive, Dead };

    /// LiveArguments, MaybeLiveArguments, DeadArguments - These sets contain
    /// all of the arguments in the program.  The Dead set contains arguments
    /// which are completely dead (never used in the function).  The MaybeLive
    /// set contains arguments which are only passed into other function calls,
    /// thus may be live and may be dead.  The Live set contains arguments which
    /// are known to be alive.
    ///
    std::set<Argument*> DeadArguments, MaybeLiveArguments, LiveArguments;

    /// DeadRetVal, MaybeLiveRetVal, LifeRetVal - These sets contain all of the
    /// functions in the program.  The Dead set contains functions whose return
    /// value is known to be dead.  The MaybeLive set contains functions whose
    /// return values are only used by return instructions, and the Live set
    /// contains functions whose return values are used, functions that are
    /// external, and functions that already return void.
    ///
    std::set<Function*> DeadRetVal, MaybeLiveRetVal, LiveRetVal;

    /// InstructionsToInspect - As we mark arguments and return values
    /// MaybeLive, we keep track of which instructions could make the values
    /// live here.  Once the entire program has had the return value and
    /// arguments analyzed, this set is scanned to promote the MaybeLive objects
    /// to be Live if they really are used.
    std::vector<Instruction*> InstructionsToInspect;

    /// CallSites - Keep track of the call sites of functions that have
    /// MaybeLive arguments or return values.
    std::multimap<Function*, CallSite> CallSites;

  public:
    static char ID; // Pass identification, replacement for typeid
    DAE() : ModulePass((intptr_t)&ID) {}
    bool runOnModule(Module &M);

    virtual bool ShouldHackArguments() const { return false; }

  private:
    Liveness getArgumentLiveness(const Argument &A);
    bool isMaybeLiveArgumentNowLive(Argument *Arg);

    bool DeleteDeadVarargs(Function &Fn);
    void SurveyFunction(Function &Fn);

    void MarkArgumentLive(Argument *Arg);
    void MarkRetValLive(Function *F);
    void MarkReturnInstArgumentLive(ReturnInst *RI);

    void RemoveDeadArgumentsFromFunction(Function *F);
  };
}

char DAE::ID = 0;
static RegisterPass<DAE>
X("deadargelim", "Dead Argument Elimination");

namespace {
  /// DAH - DeadArgumentHacking pass - Same as dead argument elimination, but
  /// deletes arguments to functions which are external.  This is only for use
  /// by bugpoint.
  struct DAH : public DAE {
    static char ID;
    virtual bool ShouldHackArguments() const { return true; }
  };
}

char DAH::ID = 0;
static RegisterPass<DAH>
Y("deadarghaX0r", "Dead Argument Hacking (BUGPOINT USE ONLY; DO NOT USE)");

/// createDeadArgEliminationPass - This pass removes arguments from functions
/// which are not used by the body of the function.
///
ModulePass *llvm::createDeadArgEliminationPass() { return new DAE(); }
ModulePass *llvm::createDeadArgHackingPass() { return new DAH(); }

/// DeleteDeadVarargs - If this is an function that takes a ... list, and if
/// llvm.vastart is never called, the varargs list is dead for the function.
bool DAE::DeleteDeadVarargs(Function &Fn) {
  assert(Fn.getFunctionType()->isVarArg() && "Function isn't varargs!");
  if (Fn.isDeclaration() || !Fn.hasInternalLinkage()) return false;

  // Ensure that the function is only directly called.
  for (Value::use_iterator I = Fn.use_begin(), E = Fn.use_end(); I != E; ++I) {
    // If this use is anything other than a call site, give up.
    CallSite CS = CallSite::get(*I);
    Instruction *TheCall = CS.getInstruction();
    if (!TheCall) return false;   // Not a direct call site?

    // The addr of this function is passed to the call.
    if (I.getOperandNo() != 0) return false;
  }

  // Okay, we know we can transform this function if safe.  Scan its body
  // looking for calls to llvm.vastart.
  for (Function::iterator BB = Fn.begin(), E = Fn.end(); BB != E; ++BB) {
    for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I) {
      if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(I)) {
        if (II->getIntrinsicID() == Intrinsic::vastart)
          return false;
      }
    }
  }

  // If we get here, there are no calls to llvm.vastart in the function body,
  // remove the "..." and adjust all the calls.

  // Start by computing a new prototype for the function, which is the same as
  // the old function, but has fewer arguments.
  const FunctionType *FTy = Fn.getFunctionType();
  std::vector<const Type*> Params(FTy->param_begin(), FTy->param_end());
  FunctionType *NFTy = FunctionType::get(FTy->getReturnType(), Params, false);
  unsigned NumArgs = Params.size();

  // Create the new function body and insert it into the module...
  Function *NF = Function::Create(NFTy, Fn.getLinkage());
  NF->copyAttributesFrom(&Fn);
  Fn.getParent()->getFunctionList().insert(&Fn, NF);
  NF->takeName(&Fn);

  // Loop over all of the callers of the function, transforming the call sites
  // to pass in a smaller number of arguments into the new function.
  //
  std::vector<Value*> Args;
  while (!Fn.use_empty()) {
    CallSite CS = CallSite::get(Fn.use_back());
    Instruction *Call = CS.getInstruction();

    // Pass all the same arguments.
    Args.assign(CS.arg_begin(), CS.arg_begin()+NumArgs);

    // Drop any attributes that were on the vararg arguments.
    PAListPtr PAL = CS.getParamAttrs();
    if (!PAL.isEmpty() && PAL.getSlot(PAL.getNumSlots() - 1).Index > NumArgs) {
      SmallVector<ParamAttrsWithIndex, 8> ParamAttrsVec;
      for (unsigned i = 0; PAL.getSlot(i).Index <= NumArgs; ++i)
        ParamAttrsVec.push_back(PAL.getSlot(i));
      PAL = PAListPtr::get(ParamAttrsVec.begin(), ParamAttrsVec.end());
    }

    Instruction *New;
    if (InvokeInst *II = dyn_cast<InvokeInst>(Call)) {
      New = InvokeInst::Create(NF, II->getNormalDest(), II->getUnwindDest(),
                               Args.begin(), Args.end(), "", Call);
      cast<InvokeInst>(New)->setCallingConv(CS.getCallingConv());
      cast<InvokeInst>(New)->setParamAttrs(PAL);
    } else {
      New = CallInst::Create(NF, Args.begin(), Args.end(), "", Call);
      cast<CallInst>(New)->setCallingConv(CS.getCallingConv());
      cast<CallInst>(New)->setParamAttrs(PAL);
      if (cast<CallInst>(Call)->isTailCall())
        cast<CallInst>(New)->setTailCall();
    }
    Args.clear();

    if (!Call->use_empty())
      Call->replaceAllUsesWith(New);

    New->takeName(Call);

    // Finally, remove the old call from the program, reducing the use-count of
    // F.
    Call->eraseFromParent();
  }

  // Since we have now created the new function, splice the body of the old
  // function right into the new function, leaving the old rotting hulk of the
  // function empty.
  NF->getBasicBlockList().splice(NF->begin(), Fn.getBasicBlockList());

  // Loop over the argument list, transfering uses of the old arguments over to
  // the new arguments, also transfering over the names as well.  While we're at
  // it, remove the dead arguments from the DeadArguments list.
  //
  for (Function::arg_iterator I = Fn.arg_begin(), E = Fn.arg_end(),
       I2 = NF->arg_begin(); I != E; ++I, ++I2) {
    // Move the name and users over to the new version.
    I->replaceAllUsesWith(I2);
    I2->takeName(I);
  }

  // Finally, nuke the old function.
  Fn.eraseFromParent();
  return true;
}


static inline bool CallPassesValueThoughVararg(Instruction *Call,
                                               const Value *Arg) {
  CallSite CS = CallSite::get(Call);
  const Type *CalledValueTy = CS.getCalledValue()->getType();
  const Type *FTy = cast<PointerType>(CalledValueTy)->getElementType();
  unsigned NumFixedArgs = cast<FunctionType>(FTy)->getNumParams();
  for (CallSite::arg_iterator AI = CS.arg_begin()+NumFixedArgs;
       AI != CS.arg_end(); ++AI)
    if (AI->get() == Arg)
      return true;
  return false;
}

// getArgumentLiveness - Inspect an argument, determining if is known Live
// (used in a computation), MaybeLive (only passed as an argument to a call), or
// Dead (not used).
DAE::Liveness DAE::getArgumentLiveness(const Argument &A) {
  const Function *F = A.getParent();
  
  // If this is the return value of a struct function, it's not really dead.
  if (F->hasStructRetAttr() && &*(F->arg_begin()) == &A)
    return Live;
  
  if (A.use_empty())  // First check, directly dead?
    return Dead;

  // Scan through all of the uses, looking for non-argument passing uses.
  for (Value::use_const_iterator I = A.use_begin(), E = A.use_end(); I!=E;++I) {
    // Return instructions do not immediately effect liveness.
    if (isa<ReturnInst>(*I))
      continue;

    CallSite CS = CallSite::get(const_cast<User*>(*I));
    if (!CS.getInstruction()) {
      // If its used by something that is not a call or invoke, it's alive!
      return Live;
    }
    // If it's an indirect call, mark it alive...
    Function *Callee = CS.getCalledFunction();
    if (!Callee) return Live;

    // Check to see if it's passed through a va_arg area: if so, we cannot
    // remove it.
    if (CallPassesValueThoughVararg(CS.getInstruction(), &A))
      return Live;   // If passed through va_arg area, we cannot remove it
  }

  return MaybeLive;  // It must be used, but only as argument to a function
}


// SurveyFunction - This performs the initial survey of the specified function,
// checking out whether or not it uses any of its incoming arguments or whether
// any callers use the return value.  This fills in the
// (Dead|MaybeLive|Live)(Arguments|RetVal) sets.
//
// We consider arguments of non-internal functions to be intrinsically alive as
// well as arguments to functions which have their "address taken".
//
void DAE::SurveyFunction(Function &F) {
  bool FunctionIntrinsicallyLive = false;
  Liveness RetValLiveness = F.getReturnType() == Type::VoidTy ? Live : Dead;

  if (!F.hasInternalLinkage() &&
      (!ShouldHackArguments() || F.isIntrinsic()))
    FunctionIntrinsicallyLive = true;
  else
    for (Value::use_iterator I = F.use_begin(), E = F.use_end(); I != E; ++I) {
      // If this use is anything other than a call site, the function is alive.
      CallSite CS = CallSite::get(*I);
      Instruction *TheCall = CS.getInstruction();
      if (!TheCall) {   // Not a direct call site?
        FunctionIntrinsicallyLive = true;
        break;
      }

      // Check to see if the return value is used...
      if (RetValLiveness != Live)
        for (Value::use_iterator I = TheCall->use_begin(),
               E = TheCall->use_end(); I != E; ++I)
          if (isa<ReturnInst>(cast<Instruction>(*I))) {
            RetValLiveness = MaybeLive;
          } else if (isa<CallInst>(cast<Instruction>(*I)) ||
                     isa<InvokeInst>(cast<Instruction>(*I))) {
            if (CallPassesValueThoughVararg(cast<Instruction>(*I), TheCall) ||
                !CallSite::get(cast<Instruction>(*I)).getCalledFunction()) {
              RetValLiveness = Live;
              break;
            } else {
              RetValLiveness = MaybeLive;
            }
          } else {
            RetValLiveness = Live;
            break;
          }

      // If the function is PASSED IN as an argument, its address has been taken
      for (CallSite::arg_iterator AI = CS.arg_begin(), E = CS.arg_end();
           AI != E; ++AI)
        if (AI->get() == &F) {
          FunctionIntrinsicallyLive = true;
          break;
        }
      if (FunctionIntrinsicallyLive) break;
    }

  if (FunctionIntrinsicallyLive) {
    DOUT << "  Intrinsically live fn: " << F.getName() << "\n";
    for (Function::arg_iterator AI = F.arg_begin(), E = F.arg_end();
         AI != E; ++AI)
      LiveArguments.insert(AI);
    LiveRetVal.insert(&F);
    return;
  }

  switch (RetValLiveness) {
  case Live:      LiveRetVal.insert(&F); break;
  case MaybeLive: MaybeLiveRetVal.insert(&F); break;
  case Dead:      DeadRetVal.insert(&F); break;
  }

  DOUT << "  Inspecting args for fn: " << F.getName() << "\n";

  // If it is not intrinsically alive, we know that all users of the
  // function are call sites.  Mark all of the arguments live which are
  // directly used, and keep track of all of the call sites of this function
  // if there are any arguments we assume that are dead.
  //
  bool AnyMaybeLiveArgs = false;
  for (Function::arg_iterator AI = F.arg_begin(), E = F.arg_end();
       AI != E; ++AI)
    switch (getArgumentLiveness(*AI)) {
    case Live:
      DOUT << "    Arg live by use: " << AI->getName() << "\n";
      LiveArguments.insert(AI);
      break;
    case Dead:
      DOUT << "    Arg definitely dead: " << AI->getName() <<"\n";
      DeadArguments.insert(AI);
      break;
    case MaybeLive:
      DOUT << "    Arg only passed to calls: " << AI->getName() << "\n";
      AnyMaybeLiveArgs = true;
      MaybeLiveArguments.insert(AI);
      break;
    }

  // If there are any "MaybeLive" arguments, we need to check callees of
  // this function when/if they become alive.  Record which functions are
  // callees...
  if (AnyMaybeLiveArgs || RetValLiveness == MaybeLive)
    for (Value::use_iterator I = F.use_begin(), E = F.use_end();
         I != E; ++I) {
      if (AnyMaybeLiveArgs)
        CallSites.insert(std::make_pair(&F, CallSite::get(*I)));

      if (RetValLiveness == MaybeLive)
        for (Value::use_iterator UI = I->use_begin(), E = I->use_end();
             UI != E; ++UI)
          InstructionsToInspect.push_back(cast<Instruction>(*UI));
    }
}

// isMaybeLiveArgumentNowLive - Check to see if Arg is alive.  At this point, we
// know that the only uses of Arg are to be passed in as an argument to a
// function call or return.  Check to see if the formal argument passed in is in
// the LiveArguments set.  If so, return true.
//
bool DAE::isMaybeLiveArgumentNowLive(Argument *Arg) {
  for (Value::use_iterator I = Arg->use_begin(), E = Arg->use_end(); I!=E; ++I){
    if (isa<ReturnInst>(*I)) {
      if (LiveRetVal.count(Arg->getParent())) return true;
      continue;
    }

    CallSite CS = CallSite::get(*I);

    // We know that this can only be used for direct calls...
    Function *Callee = CS.getCalledFunction();

    // Loop over all of the arguments (because Arg may be passed into the call
    // multiple times) and check to see if any are now alive...
    CallSite::arg_iterator CSAI = CS.arg_begin();
    for (Function::arg_iterator AI = Callee->arg_begin(), E = Callee->arg_end();
         AI != E; ++AI, ++CSAI)
      // If this is the argument we are looking for, check to see if it's alive
      if (*CSAI == Arg && LiveArguments.count(AI))
        return true;
  }
  return false;
}

/// MarkArgumentLive - The MaybeLive argument 'Arg' is now known to be alive.
/// Mark it live in the specified sets and recursively mark arguments in callers
/// live that are needed to pass in a value.
///
void DAE::MarkArgumentLive(Argument *Arg) {
  std::set<Argument*>::iterator It = MaybeLiveArguments.lower_bound(Arg);
  if (It == MaybeLiveArguments.end() || *It != Arg) return;

  DOUT << "  MaybeLive argument now live: " << Arg->getName() <<"\n";
  MaybeLiveArguments.erase(It);
  LiveArguments.insert(Arg);

  // Loop over all of the call sites of the function, making any arguments
  // passed in to provide a value for this argument live as necessary.
  //
  Function *Fn = Arg->getParent();
  unsigned ArgNo = std::distance(Fn->arg_begin(), Function::arg_iterator(Arg));

  std::multimap<Function*, CallSite>::iterator I = CallSites.lower_bound(Fn);
  for (; I != CallSites.end() && I->first == Fn; ++I) {
    CallSite CS = I->second;
    Value *ArgVal = *(CS.arg_begin()+ArgNo);
    if (Argument *ActualArg = dyn_cast<Argument>(ArgVal)) {
      MarkArgumentLive(ActualArg);
    } else {
      // If the value passed in at this call site is a return value computed by
      // some other call site, make sure to mark the return value at the other
      // call site as being needed.
      CallSite ArgCS = CallSite::get(ArgVal);
      if (ArgCS.getInstruction())
        if (Function *Fn = ArgCS.getCalledFunction())
          MarkRetValLive(Fn);
    }
  }
}

/// MarkArgumentLive - The MaybeLive return value for the specified function is
/// now known to be alive.  Propagate this fact to the return instructions which
/// produce it.
void DAE::MarkRetValLive(Function *F) {
  assert(F && "Shame shame, we can't have null pointers here!");

  // Check to see if we already knew it was live
  std::set<Function*>::iterator I = MaybeLiveRetVal.lower_bound(F);
  if (I == MaybeLiveRetVal.end() || *I != F) return;  // It's already alive!

  DOUT << "  MaybeLive retval now live: " << F->getName() << "\n";

  MaybeLiveRetVal.erase(I);
  LiveRetVal.insert(F);        // It is now known to be live!

  // Loop over all of the functions, noticing that the return value is now live.
  for (Function::iterator BB = F->begin(), E = F->end(); BB != E; ++BB)
    if (ReturnInst *RI = dyn_cast<ReturnInst>(BB->getTerminator()))
      MarkReturnInstArgumentLive(RI);
}

void DAE::MarkReturnInstArgumentLive(ReturnInst *RI) {
  Value *Op = RI->getOperand(0);
  if (Argument *A = dyn_cast<Argument>(Op)) {
    MarkArgumentLive(A);
  } else if (CallInst *CI = dyn_cast<CallInst>(Op)) {
    if (Function *F = CI->getCalledFunction())
      MarkRetValLive(F);
  } else if (InvokeInst *II = dyn_cast<InvokeInst>(Op)) {
    if (Function *F = II->getCalledFunction())
      MarkRetValLive(F);
  }
}

// RemoveDeadArgumentsFromFunction - We know that F has dead arguments, as
// specified by the DeadArguments list.  Transform the function and all of the
// callees of the function to not have these arguments.
//
void DAE::RemoveDeadArgumentsFromFunction(Function *F) {
  // Start by computing a new prototype for the function, which is the same as
  // the old function, but has fewer arguments.
  const FunctionType *FTy = F->getFunctionType();
  std::vector<const Type*> Params;

  // Set up to build a new list of parameter attributes
  SmallVector<ParamAttrsWithIndex, 8> ParamAttrsVec;
  const PAListPtr &PAL = F->getParamAttrs();

  // The existing function return attributes.
  ParameterAttributes RAttrs = PAL.getParamAttrs(0);

  // Make the function return void if the return value is dead.
  const Type *RetTy = FTy->getReturnType();
  if (DeadRetVal.count(F)) {
    RetTy = Type::VoidTy;
    RAttrs &= ~ParamAttr::typeIncompatible(RetTy);
    DeadRetVal.erase(F);
  }

  if (RAttrs)
    ParamAttrsVec.push_back(ParamAttrsWithIndex::get(0, RAttrs));

  // Construct the new parameter list from non-dead arguments. Also construct
  // a new set of parameter attributes to correspond.
  unsigned index = 1;
  for (Function::arg_iterator I = F->arg_begin(), E = F->arg_end(); I != E;
       ++I, ++index)
    if (!DeadArguments.count(I)) {
      Params.push_back(I->getType());
      
      if (ParameterAttributes Attrs = PAL.getParamAttrs(index))
        ParamAttrsVec.push_back(ParamAttrsWithIndex::get(Params.size(), Attrs));
    }

  // Reconstruct the ParamAttrsList based on the vector we constructed.
  PAListPtr NewPAL = PAListPtr::get(ParamAttrsVec.begin(), ParamAttrsVec.end());

  // Work around LLVM bug PR56: the CWriter cannot emit varargs functions which
  // have zero fixed arguments.
  //
  bool ExtraArgHack = false;
  if (Params.empty() && FTy->isVarArg()) {
    ExtraArgHack = true;
    Params.push_back(Type::Int32Ty);
  }

  // Create the new function type based on the recomputed parameters.
  FunctionType *NFTy = FunctionType::get(RetTy, Params, FTy->isVarArg());

  // Create the new function body and insert it into the module...
  Function *NF = Function::Create(NFTy, F->getLinkage());
  NF->copyAttributesFrom(F);
  NF->setParamAttrs(NewPAL);
  F->getParent()->getFunctionList().insert(F, NF);
  NF->takeName(F);

  // Loop over all of the callers of the function, transforming the call sites
  // to pass in a smaller number of arguments into the new function.
  //
  std::vector<Value*> Args;
  while (!F->use_empty()) {
    CallSite CS = CallSite::get(F->use_back());
    Instruction *Call = CS.getInstruction();
    ParamAttrsVec.clear();
    const PAListPtr &CallPAL = CS.getParamAttrs();

    // The call return attributes.
    ParameterAttributes RAttrs = CallPAL.getParamAttrs(0);
    // Adjust in case the function was changed to return void.
    RAttrs &= ~ParamAttr::typeIncompatible(NF->getReturnType());
    if (RAttrs)
      ParamAttrsVec.push_back(ParamAttrsWithIndex::get(0, RAttrs));

    // Loop over the operands, deleting dead ones...
    CallSite::arg_iterator AI = CS.arg_begin();
    index = 1;
    for (Function::arg_iterator I = F->arg_begin(), E = F->arg_end();
         I != E; ++I, ++AI, ++index)
      if (!DeadArguments.count(I)) {    // Remove operands for dead arguments
        Args.push_back(*AI);
        if (ParameterAttributes Attrs = CallPAL.getParamAttrs(index))
          ParamAttrsVec.push_back(ParamAttrsWithIndex::get(Args.size(), Attrs));
      }

    if (ExtraArgHack)
      Args.push_back(UndefValue::get(Type::Int32Ty));

    // Push any varargs arguments on the list. Don't forget their attributes.
    for (; AI != CS.arg_end(); ++AI) {
      Args.push_back(*AI);
      if (ParameterAttributes Attrs = CallPAL.getParamAttrs(index++))
        ParamAttrsVec.push_back(ParamAttrsWithIndex::get(Args.size(), Attrs));
    }

    // Reconstruct the ParamAttrsList based on the vector we constructed.
    PAListPtr NewCallPAL = PAListPtr::get(ParamAttrsVec.begin(),
                                          ParamAttrsVec.end());

    Instruction *New;
    if (InvokeInst *II = dyn_cast<InvokeInst>(Call)) {
      New = InvokeInst::Create(NF, II->getNormalDest(), II->getUnwindDest(),
                               Args.begin(), Args.end(), "", Call);
      cast<InvokeInst>(New)->setCallingConv(CS.getCallingConv());
      cast<InvokeInst>(New)->setParamAttrs(NewCallPAL);
    } else {
      New = CallInst::Create(NF, Args.begin(), Args.end(), "", Call);
      cast<CallInst>(New)->setCallingConv(CS.getCallingConv());
      cast<CallInst>(New)->setParamAttrs(NewCallPAL);
      if (cast<CallInst>(Call)->isTailCall())
        cast<CallInst>(New)->setTailCall();
    }
    Args.clear();

    if (!Call->use_empty()) {
      if (New->getType() == Type::VoidTy)
        Call->replaceAllUsesWith(Constant::getNullValue(Call->getType()));
      else {
        Call->replaceAllUsesWith(New);
        New->takeName(Call);
      }
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
  // the new arguments, also transfering over the names as well.  While we're at
  // it, remove the dead arguments from the DeadArguments list.
  //
  for (Function::arg_iterator I = F->arg_begin(), E = F->arg_end(),
         I2 = NF->arg_begin();
       I != E; ++I)
    if (!DeadArguments.count(I)) {
      // If this is a live argument, move the name and users over to the new
      // version.
      I->replaceAllUsesWith(I2);
      I2->takeName(I);
      ++I2;
    } else {
      // If this argument is dead, replace any uses of it with null constants
      // (these are guaranteed to only be operands to call instructions which
      // will later be simplified).
      I->replaceAllUsesWith(Constant::getNullValue(I->getType()));
      DeadArguments.erase(I);
    }

  // If we change the return value of the function we must rewrite any return
  // instructions.  Check this now.
  if (F->getReturnType() != NF->getReturnType())
    for (Function::iterator BB = NF->begin(), E = NF->end(); BB != E; ++BB)
      if (ReturnInst *RI = dyn_cast<ReturnInst>(BB->getTerminator())) {
        ReturnInst::Create(0, RI);
        BB->getInstList().erase(RI);
      }

  // Now that the old function is dead, delete it.
  F->getParent()->getFunctionList().erase(F);
}

bool DAE::runOnModule(Module &M) {
  bool Changed = false;
  // First pass: Do a simple check to see if any functions can have their "..."
  // removed.  We can do this if they never call va_start.  This loop cannot be
  // fused with the next loop, because deleting a function invalidates
  // information computed while surveying other functions.
  DOUT << "DAE - Deleting dead varargs\n";
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ) {
    Function &F = *I++;
    if (F.getFunctionType()->isVarArg())
      Changed |= DeleteDeadVarargs(F);
  }
  
  // Second phase:loop through the module, determining which arguments are live.
  // We assume all arguments are dead unless proven otherwise (allowing us to
  // determine that dead arguments passed into recursive functions are dead).
  //
  DOUT << "DAE - Determining liveness\n";
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
    SurveyFunction(*I);

  // Loop over the instructions to inspect, propagating liveness among arguments
  // and return values which are MaybeLive.
  while (!InstructionsToInspect.empty()) {
    Instruction *I = InstructionsToInspect.back();
    InstructionsToInspect.pop_back();

    if (ReturnInst *RI = dyn_cast<ReturnInst>(I)) {
      // For return instructions, we just have to check to see if the return
      // value for the current function is known now to be alive.  If so, any
      // arguments used by it are now alive, and any call instruction return
      // value is alive as well.
      if (LiveRetVal.count(RI->getParent()->getParent()))
        MarkReturnInstArgumentLive(RI);

    } else {
      CallSite CS = CallSite::get(I);
      assert(CS.getInstruction() && "Unknown instruction for the I2I list!");

      Function *Callee = CS.getCalledFunction();

      // If we found a call or invoke instruction on this list, that means that
      // an argument of the function is a call instruction.  If the argument is
      // live, then the return value of the called instruction is now live.
      //
      CallSite::arg_iterator AI = CS.arg_begin();  // ActualIterator
      for (Function::arg_iterator FI = Callee->arg_begin(),
             E = Callee->arg_end(); FI != E; ++AI, ++FI) {
        // If this argument is another call...
        CallSite ArgCS = CallSite::get(*AI);
        if (ArgCS.getInstruction() && LiveArguments.count(FI))
          if (Function *Callee = ArgCS.getCalledFunction())
            MarkRetValLive(Callee);
      }
    }
  }

  // Now we loop over all of the MaybeLive arguments, promoting them to be live
  // arguments if one of the calls that uses the arguments to the calls they are
  // passed into requires them to be live.  Of course this could make other
  // arguments live, so process callers recursively.
  //
  // Because elements can be removed from the MaybeLiveArguments set, copy it to
  // a temporary vector.
  //
  std::vector<Argument*> TmpArgList(MaybeLiveArguments.begin(),
                                    MaybeLiveArguments.end());
  for (unsigned i = 0, e = TmpArgList.size(); i != e; ++i) {
    Argument *MLA = TmpArgList[i];
    if (MaybeLiveArguments.count(MLA) &&
        isMaybeLiveArgumentNowLive(MLA))
      MarkArgumentLive(MLA);
  }

  // Recover memory early...
  CallSites.clear();

  // At this point, we know that all arguments in DeadArguments and
  // MaybeLiveArguments are dead.  If the two sets are empty, there is nothing
  // to do.
  if (MaybeLiveArguments.empty() && DeadArguments.empty() &&
      MaybeLiveRetVal.empty() && DeadRetVal.empty())
    return Changed;

  // Otherwise, compact into one set, and start eliminating the arguments from
  // the functions.
  DeadArguments.insert(MaybeLiveArguments.begin(), MaybeLiveArguments.end());
  MaybeLiveArguments.clear();
  DeadRetVal.insert(MaybeLiveRetVal.begin(), MaybeLiveRetVal.end());
  MaybeLiveRetVal.clear();

  LiveArguments.clear();
  LiveRetVal.clear();

  NumArgumentsEliminated += DeadArguments.size();
  NumRetValsEliminated   += DeadRetVal.size();
  while (!DeadArguments.empty())
    RemoveDeadArgumentsFromFunction((*DeadArguments.begin())->getParent());

  while (!DeadRetVal.empty())
    RemoveDeadArgumentsFromFunction(*DeadRetVal.begin());
  return true;
}
