//===-- DeadArgumentElimination.cpp - Eliminate dead arguments ------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This pass deletes dead arguments from internal functions.  Dead argument
// elimination removes arguments which are directly dead, as well as arguments
// only passed into function calls as dead arguments of other functions.
//
// This pass is often useful as a cleanup pass to run after aggressive
// interprocedural passes, which add possibly-dead arguments.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Constant.h"
#include "llvm/iOther.h"
#include "llvm/iTerminators.h"
#include "llvm/Support/CallSite.h"
#include "Support/Debug.h"
#include "Support/Statistic.h"
#include "Support/iterator"
#include <set>

namespace {
  Statistic<> NumArgumentsEliminated("deadargelim", "Number of args removed");

  struct DAE : public Pass {
    DAE(bool DFEF = false) : DeleteFromExternalFunctions(DFEF) {}
    bool run(Module &M);

  private:
    bool DeleteFromExternalFunctions;
    bool FunctionArgumentsIntrinsicallyAlive(const Function &F);
    void RemoveDeadArgumentsFromFunction(Function *F,
                                         std::set<Argument*> &DeadArguments);
  };
  RegisterOpt<DAE> X("deadargelim", "Dead Argument Elimination");
}

/// createDeadArgEliminationPass - This pass removes arguments from functions
/// which are not used by the body of the function.  If
/// DeleteFromExternalFunctions is true, the pass will modify functions that
/// have external linkage, which is not usually safe (this is used by bugpoint
/// to reduce testcases).
///
Pass *createDeadArgEliminationPass(bool DeleteFromExternalFunctions) {
  return new DAE(DeleteFromExternalFunctions);
}


// FunctionArgumentsIntrinsicallyAlive - Return true if the arguments of the
// specified function are intrinsically alive.
//
// We consider arguments of non-internal functions to be intrinsically alive as
// well as arguments to functions which have their "address taken".
//
bool DAE::FunctionArgumentsIntrinsicallyAlive(const Function &F) {
  if (!F.hasInternalLinkage() && !DeleteFromExternalFunctions) return true;

  for (Value::use_const_iterator I = F.use_begin(), E = F.use_end(); I!=E; ++I){
    // If this use is anything other than a call site, the function is alive.
    CallSite CS = CallSite::get(const_cast<User*>(*I));
    if (!CS.getInstruction()) return true;  // Not a valid call site?

    // If the function is PASSED IN as an argument, its address has been taken
    for (CallSite::arg_iterator AI = CS.arg_begin(), E = CS.arg_end(); AI != E;
         ++AI)
      if (AI->get() == &F) return true;
  }
  return false;
}

namespace {
  enum ArgumentLiveness { Alive, MaybeLive, Dead };
}

// getArgumentLiveness - Inspect an argument, determining if is known Alive
// (used in a computation), MaybeLive (only passed as an argument to a call), or
// Dead (not used).
static ArgumentLiveness getArgumentLiveness(const Argument &A) {
  if (A.use_empty()) return Dead;  // First check, directly dead?

  // Scan through all of the uses, looking for non-argument passing uses.
  for (Value::use_const_iterator I = A.use_begin(), E = A.use_end(); I!=E;++I) {
    CallSite CS = CallSite::get(const_cast<User*>(*I));
    if (!CS.getInstruction()) {
      // If its used by something that is not a call or invoke, it's alive!
      return Alive;
    }
    // If it's an indirect call, mark it alive...
    Function *Callee = CS.getCalledFunction();
    if (!Callee) return Alive;

    // Check to see if it's passed through a va_arg area: if so, we cannot
    // remove it.
    unsigned NumFixedArgs = Callee->getFunctionType()->getNumParams();
    for (CallSite::arg_iterator AI = CS.arg_begin()+NumFixedArgs;
         AI != CS.arg_end(); ++AI)
      if (AI->get() == &A) // If passed through va_arg area, we cannot remove it
        return Alive;
  }

  return MaybeLive;  // It must be used, but only as argument to a function
}

// isMaybeLiveArgumentNowAlive - Check to see if Arg is alive.  At this point,
// we know that the only uses of Arg are to be passed in as an argument to a
// function call.  Check to see if the formal argument passed in is in the
// LiveArguments set.  If so, return true.
//
static bool isMaybeLiveArgumentNowAlive(Argument *Arg,
                                     const std::set<Argument*> &LiveArguments) {
  for (Value::use_iterator I = Arg->use_begin(), E = Arg->use_end(); I!=E; ++I){
    CallSite CS = CallSite::get(*I);

    // We know that this can only be used for direct calls...
    Function *Callee = cast<Function>(CS.getCalledValue());

    // Loop over all of the arguments (because Arg may be passed into the call
    // multiple times) and check to see if any are now alive...
    CallSite::arg_iterator CSAI = CS.arg_begin();
    for (Function::aiterator AI = Callee->abegin(), E = Callee->aend();
         AI != E; ++AI, ++CSAI)
      // If this is the argument we are looking for, check to see if it's alive
      if (*CSAI == Arg && LiveArguments.count(AI))
        return true;
  }
  return false;
}

// MarkArgumentLive - The MaybeLive argument 'Arg' is now known to be alive.
// Mark it live in the specified sets and recursively mark arguments in callers
// live that are needed to pass in a value.
//
static void MarkArgumentLive(Argument *Arg,
                             std::set<Argument*> &MaybeLiveArguments,
                             std::set<Argument*> &LiveArguments,
                          const std::multimap<Function*, CallSite> &CallSites) {
  DEBUG(std::cerr << "  MaybeLive argument now live: " << Arg->getName()<<"\n");
  assert(MaybeLiveArguments.count(Arg) && !LiveArguments.count(Arg) &&
         "Arg not MaybeLive?");
  MaybeLiveArguments.erase(Arg);
  LiveArguments.insert(Arg);
  
  // Loop over all of the call sites of the function, making any arguments
  // passed in to provide a value for this argument live as necessary.
  //
  Function *Fn = Arg->getParent();
  unsigned ArgNo = std::distance(Fn->abegin(), Function::aiterator(Arg));

  std::multimap<Function*, CallSite>::const_iterator I =
    CallSites.lower_bound(Fn);
  for (; I != CallSites.end() && I->first == Fn; ++I) {
    const CallSite &CS = I->second;
    if (Argument *ActualArg = dyn_cast<Argument>(*(CS.arg_begin()+ArgNo)))
      if (MaybeLiveArguments.count(ActualArg))
        MarkArgumentLive(ActualArg, MaybeLiveArguments, LiveArguments,
                         CallSites);
  }
}

// RemoveDeadArgumentsFromFunction - We know that F has dead arguments, as
// specified by the DeadArguments list.  Transform the function and all of the
// callees of the function to not have these arguments.
//
void DAE::RemoveDeadArgumentsFromFunction(Function *F,
                                          std::set<Argument*> &DeadArguments){
  // Start by computing a new prototype for the function, which is the same as
  // the old function, but has fewer arguments.
  const FunctionType *FTy = F->getFunctionType();
  std::vector<const Type*> Params;

  for (Function::aiterator I = F->abegin(), E = F->aend(); I != E; ++I)
    if (!DeadArguments.count(I))
      Params.push_back(I->getType());

  FunctionType *NFTy = FunctionType::get(FTy->getReturnType(), Params,
                                         FTy->isVarArg());
  
  // Create the new function body and insert it into the module...
  Function *NF = new Function(NFTy, F->getLinkage(), F->getName());
  F->getParent()->getFunctionList().insert(F, NF);

  // Loop over all of the callers of the function, transforming the call sites
  // to pass in a smaller number of arguments into the new function.
  //
  while (!F->use_empty()) {
    CallSite CS = CallSite::get(F->use_back());
    Instruction *Call = CS.getInstruction();
    CS.setCalledFunction(NF);   // Reduce the uses count of F
    
    // Loop over the operands, deleting dead ones...
    CallSite::arg_iterator AI = CS.arg_begin();
    for (Function::aiterator I = F->abegin(), E = F->aend(); I != E; ++I)
      if (DeadArguments.count(I)) {        // Remove operands for dead arguments
        AI = Call->op_erase(AI);
      }  else {
        ++AI;  // Leave live operands alone...
      }
  }

  // Since we have now created the new function, splice the body of the old
  // function right into the new function, leaving the old rotting hulk of the
  // function empty.
  NF->getBasicBlockList().splice(NF->begin(), F->getBasicBlockList());

  // Loop over the argument list, transfering uses of the old arguments over to
  // the new arguments, also transfering over the names as well.  While we're at
  // it, remove the dead arguments from the DeadArguments list.
  //
  for (Function::aiterator I = F->abegin(), E = F->aend(), I2 = NF->abegin();
       I != E; ++I)
    if (!DeadArguments.count(I)) {
      // If this is a live argument, move the name and users over to the new
      // version.
      I->replaceAllUsesWith(I2);
      I2->setName(I->getName());
      ++I2;
    } else {
      // If this argument is dead, replace any uses of it with null constants
      // (these are guaranteed to only be operands to call instructions which
      // will later be simplified).
      I->replaceAllUsesWith(Constant::getNullValue(I->getType()));
      DeadArguments.erase(I);
    }

  // Now that the old function is dead, delete it.
  F->getParent()->getFunctionList().erase(F);
}

bool DAE::run(Module &M) {
  // First phase: loop through the module, determining which arguments are live.
  // We assume all arguments are dead unless proven otherwise (allowing us to
  // determine that dead arguments passed into recursive functions are dead).
  //
  std::set<Argument*> LiveArguments, MaybeLiveArguments, DeadArguments;
  std::multimap<Function*, CallSite> CallSites;

  DEBUG(std::cerr << "DAE - Determining liveness\n");
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I) {
    Function &Fn = *I;
    // If the function is intrinsically alive, just mark the arguments alive.
    if (FunctionArgumentsIntrinsicallyAlive(Fn)) {
      for (Function::aiterator AI = Fn.abegin(), E = Fn.aend(); AI != E; ++AI)
        LiveArguments.insert(AI);
      DEBUG(std::cerr << "  Args intrinsically live for fn: " << Fn.getName()
                      << "\n");
    } else {
      DEBUG(std::cerr << "  Inspecting args for fn: " << Fn.getName() << "\n");

      // If it is not intrinsically alive, we know that all users of the
      // function are call sites.  Mark all of the arguments live which are
      // directly used, and keep track of all of the call sites of this function
      // if there are any arguments we assume that are dead.
      //
      bool AnyMaybeLiveArgs = false;
      for (Function::aiterator AI = Fn.abegin(), E = Fn.aend(); AI != E; ++AI)
        switch (getArgumentLiveness(*AI)) {
        case Alive:
          DEBUG(std::cerr << "    Arg live by use: " << AI->getName() << "\n");
          LiveArguments.insert(AI);
          break;
        case Dead:
          DEBUG(std::cerr << "    Arg definitely dead: " <<AI->getName()<<"\n");
          DeadArguments.insert(AI);
          break;
        case MaybeLive:
          DEBUG(std::cerr << "    Arg only passed to calls: "
                          << AI->getName() << "\n");
          AnyMaybeLiveArgs = true;
          MaybeLiveArguments.insert(AI);
          break;
        }

      // If there are any "MaybeLive" arguments, we need to check callees of
      // this function when/if they become alive.  Record which functions are
      // callees...
      if (AnyMaybeLiveArgs)
        for (Value::use_iterator I = Fn.use_begin(), E = Fn.use_end();
             I != E; ++I)
          CallSites.insert(std::make_pair(&Fn, CallSite::get(*I)));
    }
  }

  // Now we loop over all of the MaybeLive arguments, promoting them to be live
  // arguments if one of the calls that uses the arguments to the calls they are
  // passed into requires them to be live.  Of course this could make other
  // arguments live, so process callers recursively.
  //
  // Because elements can be removed from the MaybeLiveArguments list, copy it
  // to a temporary vector.
  //
  std::vector<Argument*> TmpArgList(MaybeLiveArguments.begin(),
                                    MaybeLiveArguments.end());
  for (unsigned i = 0, e = TmpArgList.size(); i != e; ++i) {
    Argument *MLA = TmpArgList[i];
    if (MaybeLiveArguments.count(MLA) &&
        isMaybeLiveArgumentNowAlive(MLA, LiveArguments)) {
      MarkArgumentLive(MLA, MaybeLiveArguments, LiveArguments, CallSites);
    }
  }

  // Recover memory early...
  CallSites.clear();

  // At this point, we know that all arguments in DeadArguments and
  // MaybeLiveArguments are dead.  If the two sets are empty, there is nothing
  // to do.
  if (MaybeLiveArguments.empty() && DeadArguments.empty())
    return false;
  
  // Otherwise, compact into one set, and start eliminating the arguments from
  // the functions.
  DeadArguments.insert(MaybeLiveArguments.begin(), MaybeLiveArguments.end());
  MaybeLiveArguments.clear();

  NumArgumentsEliminated += DeadArguments.size();
  while (!DeadArguments.empty())
    RemoveDeadArgumentsFromFunction((*DeadArguments.begin())->getParent(),
                                    DeadArguments);
  return true;
}
