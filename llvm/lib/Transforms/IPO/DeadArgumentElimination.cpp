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
// pass also deletes dead return values in a similar way.
//
// This pass is often useful as a cleanup pass to run after aggressive
// interprocedural passes, which add possibly-dead arguments or return values.
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
  public:

    /// Struct that represent either a (part of a) return value or a function
    /// argument.  Used so that arguments and return values can be used
    /// interchangably.
    struct RetOrArg {
      RetOrArg(const Function* F, unsigned Idx, bool IsArg) : F(F), Idx(Idx), IsArg(IsArg) {}
      const Function *F;
      unsigned Idx;
      bool IsArg;
        
      /// Make RetOrArg comparable, so we can put it into a map
      bool operator<(const RetOrArg &O) const {
        if (F != O.F)
          return F < O.F;
        else if (Idx != O.Idx)
          return Idx < O.Idx;
        else
          return IsArg < O.IsArg;
      }

      /// Make RetOrArg comparable, so we can easily iterate the multimap
      bool operator==(const RetOrArg &O) const {
        return F == O.F && Idx == O.Idx && IsArg == O.IsArg;
      }
    };
    
    /// Liveness enum - During our initial pass over the program, we determine
    /// that things are either definately alive, definately dead, or in need of
    /// interprocedural analysis (MaybeLive).
    ///
    enum Liveness { Live, MaybeLive, Dead };

    /// Convenience wrapper
    RetOrArg CreateRet(const Function *F, unsigned Idx) { return RetOrArg(F, Idx, false); }
    /// Convenience wrapper
    RetOrArg CreateArg(const Function *F, unsigned Idx) { return RetOrArg(F, Idx, true); }

    typedef std::multimap<RetOrArg, RetOrArg> UseMap;
    /// This map maps a return value or argument to all return values or
    /// arguments it uses. 
    /// For example (indices are left out for clarity):
    ///  - Uses[ret F] = ret G
    ///    This means that F calls G, and F returns the value returned by G.
    ///  - Uses[arg F] = ret G
    ///    This means that some function calls G and passes its result as an
    ///    argument to F.
    ///  - Uses[ret F] = arg F
    ///    This means that F returns one of its own arguments.
    ///  - Uses[arg F] = arg G
    ///    This means that G calls F and passes one of its own (G's) arguments
    ///    directly to F.
    UseMap Uses;

    typedef std::set<RetOrArg> LiveSet;

    /// This set contains all values that have been determined to be live
    LiveSet LiveValues;
    
    typedef SmallVector<RetOrArg, 5> UseVector;

  public:
    static char ID; // Pass identification, replacement for typeid
    DAE() : ModulePass((intptr_t)&ID) {}
    bool runOnModule(Module &M);

    virtual bool ShouldHackArguments() const { return false; }

  private:
    Liveness IsMaybeLive(RetOrArg Use, UseVector &MaybeLiveUses);
    Liveness SurveyUse(Value::use_iterator U, UseVector &MaybeLiveUses, unsigned RetValNum = 0);
    Liveness SurveyUses(Value *V, UseVector &MaybeLiveUses);

    void SurveyFunction(Function &F);
    void MarkValue(const RetOrArg &RA, Liveness L, const UseVector &MaybeLiveUses);
    void MarkLive(RetOrArg RA);
    bool RemoveDeadStuffFromFunction(Function *F);
    bool DeleteDeadVarargs(Function &Fn);
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
  // the old function, but doesn't have isVarArg set.
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

/// Convenience function that returns the number of return values. It returns 0
/// for void functions and 1 for functions not returning a struct. It returns
/// the number of struct elements for functions returning a struct.
static unsigned NumRetVals(const Function *F) {
  if (F->getReturnType() == Type::VoidTy)
    return 0;
  else if (const StructType *STy = dyn_cast<StructType>(F->getReturnType()))
    return STy->getNumElements();
  else
    return 1;
}

/// IsMaybeAlive - This checks Use for liveness. If Use is live, returns Live,
/// else returns MaybeLive. Also, adds Use to MaybeLiveUses in the latter case.
DAE::Liveness DAE::IsMaybeLive(RetOrArg Use, UseVector &MaybeLiveUses) {
  // We're live if our use is already marked as live
  if (LiveValues.count(Use))
    return Live;

  // We're maybe live otherwise, but remember that we must become live if
  // Use becomes live.
  MaybeLiveUses.push_back(Use);
  return MaybeLive;
}


/// SurveyUse - This looks at a single use of an argument or return value
/// and determines if it should be alive or not. Adds this use to MaybeLiveUses
/// if it causes the used value to become MaybeAlive.
///
/// RetValNum is the return value number to use when this use is used in a
/// return instruction. This is used in the recursion, you should always leave
/// it at 0.
DAE::Liveness DAE::SurveyUse(Value::use_iterator U, UseVector &MaybeLiveUses, unsigned RetValNum) {
    Value *V = *U;
    if (ReturnInst *RI = dyn_cast<ReturnInst>(V)) {
      // The value is returned from another function. It's only live when the
      // caller's return value is live
      RetOrArg Use = CreateRet(RI->getParent()->getParent(), RetValNum);
      // We might be live, depending on the liveness of Use
      return IsMaybeLive(Use, MaybeLiveUses);
    } 
    if (InsertValueInst *IV = dyn_cast<InsertValueInst>(V)) {
      if (U.getOperandNo() != InsertValueInst::getAggregateOperandIndex() && IV->hasIndices())
        // The use we are examining is inserted into an aggregate. Our liveness
        // depends on all uses of that aggregate, but if it is used as a return
        // value, only index at which we were inserted counts.
        RetValNum = *IV->idx_begin();

      // Note that if we are used as the aggregate operand to the insertvalue,
      // we don't change RetValNum, but do survey all our uses.
      
      Liveness Result = Dead;
      for (Value::use_iterator I = IV->use_begin(),
           E = V->use_end(); I != E; ++I) {
        Result = SurveyUse(I, MaybeLiveUses, RetValNum);
        if (Result == Live)
          break;
      }
      return Result;
    }
    CallSite CS = CallSite::get(V);
    if (CS.getInstruction()) {
      Function *F = CS.getCalledFunction();
      if (F) {
        // Used in a direct call
        
        // Check for vararg. Do - 1 to skip the first operand to call (the
        // function itself).
        if (U.getOperandNo() - 1 >= F->getFunctionType()->getNumParams())
          // The value is passed in through a vararg! Must be live.
          return Live;

        // Value passed to a normal call. It's only live when the corresponding
        // argument (operand number - 1 to skip the function pointer operand) to
        // the called function turns out live
        RetOrArg Use = CreateArg(F, U.getOperandNo() - 1);
        return IsMaybeLive(Use, MaybeLiveUses);
      } else {
        // Used in any other way? Value must be live.
        return Live;
      }
    }
    // Used in any other way? Value must be live.
    return Live;
}

/// SurveyUses - This looks at all the uses of the given return value
/// (possibly a partial return value from a function returning a struct).
/// Returns the Liveness deduced from the uses of this value.
///
/// Adds all uses that cause the result to be MaybeLive to MaybeLiveRetUses.
DAE::Liveness DAE::SurveyUses(Value *V, UseVector &MaybeLiveUses) {
  // Assume it's dead (which will only hold if there are no uses at all..)
  Liveness Result = Dead;
  // Check each use
  for (Value::use_iterator I = V->use_begin(),
       E = V->use_end(); I != E; ++I) {
    Result = SurveyUse(I, MaybeLiveUses);
    if (Result == Live)
      break;
  }
  return Result;
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
  unsigned RetCount = NumRetVals(&F);
  // Assume all return values are dead
  typedef SmallVector<Liveness, 5> RetVals;
  RetVals RetValLiveness(RetCount, Dead);

  // These vectors maps each return value to the uses that make it MaybeLive, so
  // we can add those to the MaybeLiveRetVals list if the return value
  // really turns out to be MaybeLive. Initializes to RetCount empty vectors
  typedef SmallVector<UseVector, 5> RetUses;
  // Intialized to a list of RetCount empty lists
  RetUses MaybeLiveRetUses(RetCount);
  
  for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB)
    if (ReturnInst *RI = dyn_cast<ReturnInst>(BB->getTerminator()))
      if (RI->getNumOperands() != 0 && RI->getOperand(0)->getType() != F.getFunctionType()->getReturnType()) {
        // We don't support old style multiple return values
        FunctionIntrinsicallyLive = true;
        break;
      }

  if (!F.hasInternalLinkage() && (!ShouldHackArguments() || F.isIntrinsic()))
    FunctionIntrinsicallyLive = true;

  if (!FunctionIntrinsicallyLive) {
    DOUT << "DAE - Inspecting callers for fn: " << F.getName() << "\n";
    // Keep track of the number of live retvals, so we can skip checks once all
    // of them turn out to be live.
    unsigned NumLiveRetVals = 0;
    const Type *STy = dyn_cast<StructType>(F.getReturnType());
    // Loop all uses of the function
    for (Value::use_iterator I = F.use_begin(), E = F.use_end(); I != E; ++I) {
      // If the function is PASSED IN as an argument, its address has been taken
      if (I.getOperandNo() != 0) {
        FunctionIntrinsicallyLive = true;
        break;
      }

      // If this use is anything other than a call site, the function is alive.
      CallSite CS = CallSite::get(*I);
      Instruction *TheCall = CS.getInstruction();
      if (!TheCall) {   // Not a direct call site?
        FunctionIntrinsicallyLive = true;
        break;
      }
 
      // If we end up here, we are looking at a direct call to our function.
      
      // Now, check how our return value(s) is/are used in this caller. Don't
      // bother checking return values if all of them are live already
      if (NumLiveRetVals != RetCount) { 
        if (STy) {
          // Check all uses of the return value
          for (Value::use_iterator I = TheCall->use_begin(),
               E = TheCall->use_end(); I != E; ++I) {
            ExtractValueInst *Ext = dyn_cast<ExtractValueInst>(*I);
            if (Ext && Ext->hasIndices()) {
              // This use uses a part of our return value, survey the uses of that
              // part and store the results for this index only.
              unsigned Idx = *Ext->idx_begin();
              if (RetValLiveness[Idx] != Live) {
                RetValLiveness[Idx] = SurveyUses(Ext, MaybeLiveRetUses[Idx]);
                if (RetValLiveness[Idx] == Live)
                  NumLiveRetVals++;
              }
            } else {
              // Used by something else than extractvalue. Mark all
              // return values as live.
              for (unsigned i = 0; i != RetCount; ++i )
                RetValLiveness[i] = Live;
              NumLiveRetVals = RetCount;
              break;
            }
          }
        } else {
          // Single return value
          RetValLiveness[0] = SurveyUses(TheCall, MaybeLiveRetUses[0]);
          if (RetValLiveness[0] == Live)
            NumLiveRetVals = RetCount;
        }
      }
    }
  }
  if (FunctionIntrinsicallyLive) {
    DOUT << "DAE - Intrinsically live fn: " << F.getName() << "\n";
    // Mark all arguments as live
    unsigned i = 0;
    for (Function::arg_iterator AI = F.arg_begin(), E = F.arg_end();
         AI != E; ++AI, ++i)
      MarkLive(CreateArg(&F, i));
    // Mark all return values as live
    i = 0;
    for (unsigned i = 0, e = RetValLiveness.size(); i != e; ++i)
      MarkLive(CreateRet(&F, i));
    return;
  }
 
  // Now we've inspected all callers, record the liveness of our return values.
  for (unsigned i = 0, e = RetValLiveness.size(); i != e; ++i) {
    RetOrArg Ret = CreateRet(&F, i);
    // Mark the result down
    MarkValue(Ret, RetValLiveness[i], MaybeLiveRetUses[i]);
  }
  DOUT << "DAE - Inspecting args for fn: " << F.getName() << "\n";

  // Now, check all of our arguments
  unsigned i = 0;
  UseVector MaybeLiveArgUses;
  for (Function::arg_iterator AI = F.arg_begin(), 
       E = F.arg_end(); AI != E; ++AI, ++i) {
    // See what the effect of this use is (recording any uses that cause
    // MaybeLive in MaybeLiveArgUses)
    Liveness Result = SurveyUses(AI, MaybeLiveArgUses);
    RetOrArg Arg = CreateArg(&F, i);
    // Mark the result down
    MarkValue(Arg, Result, MaybeLiveArgUses);
    // Clear the vector again for the next iteration
    MaybeLiveArgUses.clear();
  }
}

/// MarkValue - This function marks the liveness of RA depending on L. If L is
/// MaybeLive, it also records any uses in MaybeLiveUses such that RA will be
/// marked live if any use in MaybeLiveUses gets marked live later on.
void DAE::MarkValue(const RetOrArg &RA, Liveness L, const UseVector &MaybeLiveUses) {
  switch (L) {
    case Live: MarkLive(RA); break;
    case MaybeLive:
    {
      // Note any uses of this value, so this return value can be
      // marked live whenever one of the uses becomes live.
      UseMap::iterator Where = Uses.begin();
      for (UseVector::const_iterator UI = MaybeLiveUses.begin(), 
           UE = MaybeLiveUses.end(); UI != UE; ++UI)
        Where = Uses.insert(Where, UseMap::value_type(*UI, RA));
      break;
    }
    case Dead: break;
  }
}

/// MarkLive - Mark the given return value or argument as live. Additionally,
/// mark any values that are used by this value (according to Uses) live as
/// well.
void DAE::MarkLive(RetOrArg RA) {
  if (!LiveValues.insert(RA).second)
    return; // We were already marked Live

  if (RA.IsArg)
    DOUT << "DAE - Marking argument " << RA.Idx << " to function " << RA.F->getNameStart() << " live\n";
  else
    DOUT << "DAE - Marking return value " << RA.Idx << " of function " << RA.F->getNameStart() << " live\n";

  // We don't use upper_bound (or equal_range) here, because our recursive call
  // to ourselves is likely to mark the upper_bound (which is the first value
  // not belonging to RA) to become erased and the iterator invalidated.
  UseMap::iterator Begin = Uses.lower_bound(RA);
  UseMap::iterator E = Uses.end();
  UseMap::iterator I;
  for (I = Begin; I != E && I->first == RA; ++I)
    MarkLive(I->second);

  // Erase RA from the Uses map (from the lower bound to wherever we ended up
  // after the loop).
  Uses.erase(Begin, I);
}

// RemoveDeadStuffFromFunction - Remove any arguments and return values from F
// that are not in LiveValues. This function is a noop for any Function created
// by this function before, or any function that was not inspected for liveness.
// specified by the DeadArguments list.  Transform the function and all of the
// callees of the function to not have these arguments.
//
bool DAE::RemoveDeadStuffFromFunction(Function *F) {
  // Quick exit path for external functions
  if (!F->hasInternalLinkage() && (!ShouldHackArguments() || F->isIntrinsic()))
    return false;

  // Start by computing a new prototype for the function, which is the same as
  // the old function, but has fewer arguments and a different return type.
  const FunctionType *FTy = F->getFunctionType();
  std::vector<const Type*> Params;

  // Set up to build a new list of parameter attributes
  SmallVector<ParamAttrsWithIndex, 8> ParamAttrsVec;
  const PAListPtr &PAL = F->getParamAttrs();

  // The existing function return attributes.
  ParameterAttributes RAttrs = PAL.getParamAttrs(0);

  
  // Find out the new return value
 
  const Type *RetTy = FTy->getReturnType();
  const Type *NRetTy;
  unsigned RetCount = NumRetVals(F);
  // -1 means unused, other numbers are the new index
  SmallVector<int, 5> NewRetIdxs(RetCount, -1);
  std::vector<const Type*> RetTypes;
  if (RetTy != Type::VoidTy) {
    const StructType *STy = dyn_cast<StructType>(RetTy);
    if (STy)
      // Look at each of the original return values individually
      for (unsigned i = 0; i != RetCount; ++i) {
        RetOrArg Ret = CreateRet(F, i);
        if (LiveValues.erase(Ret)) {
          RetTypes.push_back(STy->getElementType(i));
          NewRetIdxs[i] = RetTypes.size() - 1;
        } else {
          ++NumRetValsEliminated;
        DOUT << "DAE - Removing return value " << i << " from " << F->getNameStart() << "\n";
        }
      }
    else
      // We used to return a single value
      if (LiveValues.erase(CreateRet(F, 0))) {
        RetTypes.push_back(RetTy);
        NewRetIdxs[0] = 0;
      } else {
        DOUT << "DAE - Removing return value from " << F->getNameStart() << "\n";
        ++NumRetValsEliminated;
      } 
    if (RetTypes.size() == 0)
      // No return types? Make it void
      NRetTy = Type::VoidTy;
    else if (RetTypes.size() == 1)
      // One return type? Just a simple value then
      NRetTy = RetTypes.front();
    else
      // More return types? Return a struct with them
      NRetTy = StructType::get(RetTypes);
  } else {
    NRetTy = Type::VoidTy;
  }
    
  // Remove any incompatible attributes
  RAttrs &= ~ParamAttr::typeIncompatible(NRetTy);
  if (RAttrs)
    ParamAttrsVec.push_back(ParamAttrsWithIndex::get(0, RAttrs));
  
  // Remember which arguments are still alive
  SmallVector<bool, 10> ArgAlive(FTy->getNumParams(), false);
  // Construct the new parameter list from non-dead arguments. Also construct
  // a new set of parameter attributes to correspond. Skip the first parameter
  // attribute, since that belongs to the return value.
  unsigned i = 0;
  for (Function::arg_iterator I = F->arg_begin(), E = F->arg_end();
       I != E; ++I, ++i) {
    RetOrArg Arg = CreateArg(F, i);
    if (LiveValues.erase(Arg)) {
      Params.push_back(I->getType());
      ArgAlive[i] = true;
      
      // Get the original parameter attributes (skipping the first one, that is
      // for the return value
      if (ParameterAttributes Attrs = PAL.getParamAttrs(i + 1))
        ParamAttrsVec.push_back(ParamAttrsWithIndex::get(Params.size(), Attrs));
    } else {
      ++NumArgumentsEliminated;
      DOUT << "DAE - Removing argument " << i << " (" << I->getNameStart() << ") from " << F->getNameStart() << "\n";
    }
  }

  // Reconstruct the ParamAttrsList based on the vector we constructed.
  PAListPtr NewPAL = PAListPtr::get(ParamAttrsVec.begin(), ParamAttrsVec.end());

  // Work around LLVM bug PR56: the CWriter cannot emit varargs functions which
  // have zero fixed arguments.
  //
  // Not that we apply this hack for a vararg fuction that does not have any
  // arguments anymore, but did have them before (so don't bother fixing
  // functions that were already broken wrt CWriter).
  bool ExtraArgHack = false;
  if (Params.empty() && FTy->isVarArg() && FTy->getNumParams() != 0) {
    ExtraArgHack = true;
    Params.push_back(Type::Int32Ty);
  }

  // Create the new function type based on the recomputed parameters.
  FunctionType *NFTy = FunctionType::get(NRetTy, Params, FTy->isVarArg());
  
  // No change?
  if (NFTy == FTy)
    return false;

  // Create the new function body and insert it into the module...
  Function *NF = Function::Create(NFTy, F->getLinkage());
  NF->copyAttributesFrom(F);
  NF->setParamAttrs(NewPAL);
  // Insert the new function before the old function, so we won't be processing
  // it again
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

    // Declare these outside of the loops, so we can reuse them for the second
    // loop, which loops the varargs
    CallSite::arg_iterator I = CS.arg_begin();
    unsigned i = 0; 
    // Loop over those operands, corresponding to the normal arguments to the
    // original function, and add those that are still alive.
    for (unsigned e = FTy->getNumParams(); i != e; ++I, ++i)
      if (ArgAlive[i]) {
        Args.push_back(*I);
        // Get original parameter attributes, but skip return attributes
        if (ParameterAttributes Attrs = CallPAL.getParamAttrs(i + 1))
          ParamAttrsVec.push_back(ParamAttrsWithIndex::get(Args.size(), Attrs));
      }

    if (ExtraArgHack)
      Args.push_back(UndefValue::get(Type::Int32Ty));

    // Push any varargs arguments on the list. Don't forget their attributes.
    for (CallSite::arg_iterator E = CS.arg_end(); I != E; ++I, ++i) {
      Args.push_back(*I);
      if (ParameterAttributes Attrs = CallPAL.getParamAttrs(i + 1))
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
        // Our return value was unused, replace by null for now, uses will get
        // removed later on
        Call->replaceAllUsesWith(Constant::getNullValue(Call->getType()));
      else if (isa<StructType>(RetTy)) {
        // The original return value was a struct, update all uses (which are
        // all extractvalue instructions).
        for (Value::use_iterator I = Call->use_begin(), E = Call->use_end();
             I != E;) {
          assert(isa<ExtractValueInst>(*I) && "Return value not only used by extractvalue?");
          ExtractValueInst *EV = cast<ExtractValueInst>(*I);
          // Increment now, since we're about to throw away this use.
          ++I;
          assert(EV->hasIndices() && "Return value used by extractvalue without indices?");
          unsigned Idx = *EV->idx_begin();
          if (NewRetIdxs[Idx] != -1) {
            if (RetTypes.size() > 1) {
              // We're still returning a struct, create a new extractvalue
              // instruction with the first index updated
              std::vector<unsigned> NewIdxs(EV->idx_begin(), EV->idx_end());
              NewIdxs[0] = NewRetIdxs[Idx];
              Value *NEV = ExtractValueInst::Create(New, NewIdxs.begin(), NewIdxs.end(), "retval", EV);
              EV->replaceAllUsesWith(NEV);
              EV->eraseFromParent();
            } else {
              // We are now only returning a simple value, remove the
              // extractvalue
              EV->replaceAllUsesWith(New);
              EV->eraseFromParent();
            }
          } else {
            // Value unused, replace uses by null for now, they will get removed
            // later on
            EV->replaceAllUsesWith(Constant::getNullValue(EV->getType()));
            EV->eraseFromParent();
          }
        }
        New->takeName(Call);
      } else {
        // The original function had a single return value
        Call->replaceAllUsesWith(New);
        New->takeName(Call);
      }
    }

    // Finally, remove the old call from the program, reducing the use-count of
    // F.
    Call->eraseFromParent();
  }

  // Since we have now created the new function, splice the body of the old
  // function right into the new function, leaving the old rotting hulk of the
  // function empty.
  NF->getBasicBlockList().splice(NF->begin(), F->getBasicBlockList());

  // Loop over the argument list, transfering uses of the old arguments over to
  // the new arguments, also transfering over the names as well.
  i = 0;
  for (Function::arg_iterator I = F->arg_begin(), E = F->arg_end(),
       I2 = NF->arg_begin(); I != E; ++I, ++i)
    if (ArgAlive[i]) {
      // If this is a live argument, move the name and users over to the new
      // version.
      I->replaceAllUsesWith(I2);
      I2->takeName(I);
      ++I2;
    } else {
      // If this argument is dead, replace any uses of it with null constants
      // (these are guaranteed to become unused later on)
      I->replaceAllUsesWith(Constant::getNullValue(I->getType()));
    }

  // If we change the return value of the function we must rewrite any return
  // instructions.  Check this now.
  if (F->getReturnType() != NF->getReturnType())
    for (Function::iterator BB = NF->begin(), E = NF->end(); BB != E; ++BB)
      if (ReturnInst *RI = dyn_cast<ReturnInst>(BB->getTerminator())) {
        Value *RetVal;

        if (NFTy->getReturnType() == Type::VoidTy) {
          RetVal = 0;
        } else {
          assert (isa<StructType>(RetTy));
          // The original return value was a struct, insert
          // extractvalue/insertvalue chains to extract only the values we need
          // to return and insert them into our new result.
          // This does generate messy code, but we'll let it to instcombine to
          // clean that up
          Value *OldRet = RI->getOperand(0);
          // Start out building up our return value from undef
          RetVal = llvm::UndefValue::get(NRetTy);
          for (unsigned i = 0; i != RetCount; ++i)
            if (NewRetIdxs[i] != -1) {
              ExtractValueInst *EV = ExtractValueInst::Create(OldRet, i, "newret", RI);
              if (RetTypes.size() > 1) {
                // We're still returning a struct, so reinsert the value into
                // our new return value at the new index

                RetVal = InsertValueInst::Create(RetVal, EV, NewRetIdxs[i], "oldret");
              } else {
                // We are now only returning a simple value, so just return the
                // extracted value
                RetVal = EV;
              }
            } 
        } 
        // Replace the return instruction with one returning the new return
        // value (possibly 0 if we became void).
        ReturnInst::Create(RetVal, RI);
        BB->getInstList().erase(RI);
      }

  // Now that the old function is dead, delete it.
  F->eraseFromParent();

  return true;
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

  // Now, remove all dead arguments and return values from each function in
  // turn
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ) {
    // Increment now, because the function will probably get removed (ie
    // replaced by a new one)
    Function *F = I++;
    Changed |= RemoveDeadStuffFromFunction(F);
  }

  return Changed;
}
