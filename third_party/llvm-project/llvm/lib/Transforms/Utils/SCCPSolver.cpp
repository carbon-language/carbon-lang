//===- SCCPSolver.cpp - SCCP Utility --------------------------- *- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// \file
// This file implements the Sparse Conditional Constant Propagation (SCCP)
// utility.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/SCCPSolver.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Local.h"
#include <cassert>
#include <utility>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "sccp"

// The maximum number of range extensions allowed for operations requiring
// widening.
static const unsigned MaxNumRangeExtensions = 10;

/// Returns MergeOptions with MaxWidenSteps set to MaxNumRangeExtensions.
static ValueLatticeElement::MergeOptions getMaxWidenStepsOpts() {
  return ValueLatticeElement::MergeOptions().setMaxWidenSteps(
      MaxNumRangeExtensions);
}

namespace {

// Helper to check if \p LV is either a constant or a constant
// range with a single element. This should cover exactly the same cases as the
// old ValueLatticeElement::isConstant() and is intended to be used in the
// transition to ValueLatticeElement.
bool isConstant(const ValueLatticeElement &LV) {
  return LV.isConstant() ||
         (LV.isConstantRange() && LV.getConstantRange().isSingleElement());
}

// Helper to check if \p LV is either overdefined or a constant range with more
// than a single element. This should cover exactly the same cases as the old
// ValueLatticeElement::isOverdefined() and is intended to be used in the
// transition to ValueLatticeElement.
bool isOverdefined(const ValueLatticeElement &LV) {
  return !LV.isUnknownOrUndef() && !isConstant(LV);
}

} // namespace

namespace llvm {

/// Helper class for SCCPSolver. This implements the instruction visitor and
/// holds all the state.
class SCCPInstVisitor : public InstVisitor<SCCPInstVisitor> {
  const DataLayout &DL;
  std::function<const TargetLibraryInfo &(Function &)> GetTLI;
  SmallPtrSet<BasicBlock *, 8> BBExecutable; // The BBs that are executable.
  DenseMap<Value *, ValueLatticeElement>
      ValueState; // The state each value is in.

  /// StructValueState - This maintains ValueState for values that have
  /// StructType, for example for formal arguments, calls, insertelement, etc.
  DenseMap<std::pair<Value *, unsigned>, ValueLatticeElement> StructValueState;

  /// GlobalValue - If we are tracking any values for the contents of a global
  /// variable, we keep a mapping from the constant accessor to the element of
  /// the global, to the currently known value.  If the value becomes
  /// overdefined, it's entry is simply removed from this map.
  DenseMap<GlobalVariable *, ValueLatticeElement> TrackedGlobals;

  /// TrackedRetVals - If we are tracking arguments into and the return
  /// value out of a function, it will have an entry in this map, indicating
  /// what the known return value for the function is.
  MapVector<Function *, ValueLatticeElement> TrackedRetVals;

  /// TrackedMultipleRetVals - Same as TrackedRetVals, but used for functions
  /// that return multiple values.
  MapVector<std::pair<Function *, unsigned>, ValueLatticeElement>
      TrackedMultipleRetVals;

  /// MRVFunctionsTracked - Each function in TrackedMultipleRetVals is
  /// represented here for efficient lookup.
  SmallPtrSet<Function *, 16> MRVFunctionsTracked;

  /// A list of functions whose return cannot be modified.
  SmallPtrSet<Function *, 16> MustPreserveReturnsInFunctions;

  /// TrackingIncomingArguments - This is the set of functions for whose
  /// arguments we make optimistic assumptions about and try to prove as
  /// constants.
  SmallPtrSet<Function *, 16> TrackingIncomingArguments;

  /// The reason for two worklists is that overdefined is the lowest state
  /// on the lattice, and moving things to overdefined as fast as possible
  /// makes SCCP converge much faster.
  ///
  /// By having a separate worklist, we accomplish this because everything
  /// possibly overdefined will become overdefined at the soonest possible
  /// point.
  SmallVector<Value *, 64> OverdefinedInstWorkList;
  SmallVector<Value *, 64> InstWorkList;

  // The BasicBlock work list
  SmallVector<BasicBlock *, 64> BBWorkList;

  /// KnownFeasibleEdges - Entries in this set are edges which have already had
  /// PHI nodes retriggered.
  using Edge = std::pair<BasicBlock *, BasicBlock *>;
  DenseSet<Edge> KnownFeasibleEdges;

  DenseMap<Function *, AnalysisResultsForFn> AnalysisResults;
  DenseMap<Value *, SmallPtrSet<User *, 2>> AdditionalUsers;

  LLVMContext &Ctx;

private:
  ConstantInt *getConstantInt(const ValueLatticeElement &IV) const {
    return dyn_cast_or_null<ConstantInt>(getConstant(IV));
  }

  // pushToWorkList - Helper for markConstant/markOverdefined
  void pushToWorkList(ValueLatticeElement &IV, Value *V);

  // Helper to push \p V to the worklist, after updating it to \p IV. Also
  // prints a debug message with the updated value.
  void pushToWorkListMsg(ValueLatticeElement &IV, Value *V);

  // markConstant - Make a value be marked as "constant".  If the value
  // is not already a constant, add it to the instruction work list so that
  // the users of the instruction are updated later.
  bool markConstant(ValueLatticeElement &IV, Value *V, Constant *C,
                    bool MayIncludeUndef = false);

  bool markConstant(Value *V, Constant *C) {
    assert(!V->getType()->isStructTy() && "structs should use mergeInValue");
    return markConstant(ValueState[V], V, C);
  }

  // markOverdefined - Make a value be marked as "overdefined". If the
  // value is not already overdefined, add it to the overdefined instruction
  // work list so that the users of the instruction are updated later.
  bool markOverdefined(ValueLatticeElement &IV, Value *V);

  /// Merge \p MergeWithV into \p IV and push \p V to the worklist, if \p IV
  /// changes.
  bool mergeInValue(ValueLatticeElement &IV, Value *V,
                    ValueLatticeElement MergeWithV,
                    ValueLatticeElement::MergeOptions Opts = {
                        /*MayIncludeUndef=*/false, /*CheckWiden=*/false});

  bool mergeInValue(Value *V, ValueLatticeElement MergeWithV,
                    ValueLatticeElement::MergeOptions Opts = {
                        /*MayIncludeUndef=*/false, /*CheckWiden=*/false}) {
    assert(!V->getType()->isStructTy() &&
           "non-structs should use markConstant");
    return mergeInValue(ValueState[V], V, MergeWithV, Opts);
  }

  /// getValueState - Return the ValueLatticeElement object that corresponds to
  /// the value.  This function handles the case when the value hasn't been seen
  /// yet by properly seeding constants etc.
  ValueLatticeElement &getValueState(Value *V) {
    assert(!V->getType()->isStructTy() && "Should use getStructValueState");

    auto I = ValueState.insert(std::make_pair(V, ValueLatticeElement()));
    ValueLatticeElement &LV = I.first->second;

    if (!I.second)
      return LV; // Common case, already in the map.

    if (auto *C = dyn_cast<Constant>(V))
      LV.markConstant(C); // Constants are constant

    // All others are unknown by default.
    return LV;
  }

  /// getStructValueState - Return the ValueLatticeElement object that
  /// corresponds to the value/field pair.  This function handles the case when
  /// the value hasn't been seen yet by properly seeding constants etc.
  ValueLatticeElement &getStructValueState(Value *V, unsigned i) {
    assert(V->getType()->isStructTy() && "Should use getValueState");
    assert(i < cast<StructType>(V->getType())->getNumElements() &&
           "Invalid element #");

    auto I = StructValueState.insert(
        std::make_pair(std::make_pair(V, i), ValueLatticeElement()));
    ValueLatticeElement &LV = I.first->second;

    if (!I.second)
      return LV; // Common case, already in the map.

    if (auto *C = dyn_cast<Constant>(V)) {
      Constant *Elt = C->getAggregateElement(i);

      if (!Elt)
        LV.markOverdefined(); // Unknown sort of constant.
      else if (isa<UndefValue>(Elt))
        ; // Undef values remain unknown.
      else
        LV.markConstant(Elt); // Constants are constant.
    }

    // All others are underdefined by default.
    return LV;
  }

  /// markEdgeExecutable - Mark a basic block as executable, adding it to the BB
  /// work list if it is not already executable.
  bool markEdgeExecutable(BasicBlock *Source, BasicBlock *Dest);

  // getFeasibleSuccessors - Return a vector of booleans to indicate which
  // successors are reachable from a given terminator instruction.
  void getFeasibleSuccessors(Instruction &TI, SmallVectorImpl<bool> &Succs);

  // OperandChangedState - This method is invoked on all of the users of an
  // instruction that was just changed state somehow.  Based on this
  // information, we need to update the specified user of this instruction.
  void operandChangedState(Instruction *I) {
    if (BBExecutable.count(I->getParent())) // Inst is executable?
      visit(*I);
  }

  // Add U as additional user of V.
  void addAdditionalUser(Value *V, User *U) {
    auto Iter = AdditionalUsers.insert({V, {}});
    Iter.first->second.insert(U);
  }

  // Mark I's users as changed, including AdditionalUsers.
  void markUsersAsChanged(Value *I) {
    // Functions include their arguments in the use-list. Changed function
    // values mean that the result of the function changed. We only need to
    // update the call sites with the new function result and do not have to
    // propagate the call arguments.
    if (isa<Function>(I)) {
      for (User *U : I->users()) {
        if (auto *CB = dyn_cast<CallBase>(U))
          handleCallResult(*CB);
      }
    } else {
      for (User *U : I->users())
        if (auto *UI = dyn_cast<Instruction>(U))
          operandChangedState(UI);
    }

    auto Iter = AdditionalUsers.find(I);
    if (Iter != AdditionalUsers.end()) {
      // Copy additional users before notifying them of changes, because new
      // users may be added, potentially invalidating the iterator.
      SmallVector<Instruction *, 2> ToNotify;
      for (User *U : Iter->second)
        if (auto *UI = dyn_cast<Instruction>(U))
          ToNotify.push_back(UI);
      for (Instruction *UI : ToNotify)
        operandChangedState(UI);
    }
  }
  void handleCallOverdefined(CallBase &CB);
  void handleCallResult(CallBase &CB);
  void handleCallArguments(CallBase &CB);

private:
  friend class InstVisitor<SCCPInstVisitor>;

  // visit implementations - Something changed in this instruction.  Either an
  // operand made a transition, or the instruction is newly executable.  Change
  // the value type of I to reflect these changes if appropriate.
  void visitPHINode(PHINode &I);

  // Terminators

  void visitReturnInst(ReturnInst &I);
  void visitTerminator(Instruction &TI);

  void visitCastInst(CastInst &I);
  void visitSelectInst(SelectInst &I);
  void visitUnaryOperator(Instruction &I);
  void visitBinaryOperator(Instruction &I);
  void visitCmpInst(CmpInst &I);
  void visitExtractValueInst(ExtractValueInst &EVI);
  void visitInsertValueInst(InsertValueInst &IVI);

  void visitCatchSwitchInst(CatchSwitchInst &CPI) {
    markOverdefined(&CPI);
    visitTerminator(CPI);
  }

  // Instructions that cannot be folded away.

  void visitStoreInst(StoreInst &I);
  void visitLoadInst(LoadInst &I);
  void visitGetElementPtrInst(GetElementPtrInst &I);

  void visitInvokeInst(InvokeInst &II) {
    visitCallBase(II);
    visitTerminator(II);
  }

  void visitCallBrInst(CallBrInst &CBI) {
    visitCallBase(CBI);
    visitTerminator(CBI);
  }

  void visitCallBase(CallBase &CB);
  void visitResumeInst(ResumeInst &I) { /*returns void*/
  }
  void visitUnreachableInst(UnreachableInst &I) { /*returns void*/
  }
  void visitFenceInst(FenceInst &I) { /*returns void*/
  }

  void visitInstruction(Instruction &I);

public:
  void addAnalysis(Function &F, AnalysisResultsForFn A) {
    AnalysisResults.insert({&F, std::move(A)});
  }

  void visitCallInst(CallInst &I) { visitCallBase(I); }

  bool markBlockExecutable(BasicBlock *BB);

  const PredicateBase *getPredicateInfoFor(Instruction *I) {
    auto A = AnalysisResults.find(I->getParent()->getParent());
    if (A == AnalysisResults.end())
      return nullptr;
    return A->second.PredInfo->getPredicateInfoFor(I);
  }

  DomTreeUpdater getDTU(Function &F) {
    auto A = AnalysisResults.find(&F);
    assert(A != AnalysisResults.end() && "Need analysis results for function.");
    return {A->second.DT, A->second.PDT, DomTreeUpdater::UpdateStrategy::Lazy};
  }

  SCCPInstVisitor(const DataLayout &DL,
                  std::function<const TargetLibraryInfo &(Function &)> GetTLI,
                  LLVMContext &Ctx)
      : DL(DL), GetTLI(GetTLI), Ctx(Ctx) {}

  void trackValueOfGlobalVariable(GlobalVariable *GV) {
    // We only track the contents of scalar globals.
    if (GV->getValueType()->isSingleValueType()) {
      ValueLatticeElement &IV = TrackedGlobals[GV];
      if (!isa<UndefValue>(GV->getInitializer()))
        IV.markConstant(GV->getInitializer());
    }
  }

  void addTrackedFunction(Function *F) {
    // Add an entry, F -> undef.
    if (auto *STy = dyn_cast<StructType>(F->getReturnType())) {
      MRVFunctionsTracked.insert(F);
      for (unsigned i = 0, e = STy->getNumElements(); i != e; ++i)
        TrackedMultipleRetVals.insert(
            std::make_pair(std::make_pair(F, i), ValueLatticeElement()));
    } else if (!F->getReturnType()->isVoidTy())
      TrackedRetVals.insert(std::make_pair(F, ValueLatticeElement()));
  }

  void addToMustPreserveReturnsInFunctions(Function *F) {
    MustPreserveReturnsInFunctions.insert(F);
  }

  bool mustPreserveReturn(Function *F) {
    return MustPreserveReturnsInFunctions.count(F);
  }

  void addArgumentTrackedFunction(Function *F) {
    TrackingIncomingArguments.insert(F);
  }

  bool isArgumentTrackedFunction(Function *F) {
    return TrackingIncomingArguments.count(F);
  }

  void solve();

  bool resolvedUndefsIn(Function &F);

  bool isBlockExecutable(BasicBlock *BB) const {
    return BBExecutable.count(BB);
  }

  bool isEdgeFeasible(BasicBlock *From, BasicBlock *To) const;

  std::vector<ValueLatticeElement> getStructLatticeValueFor(Value *V) const {
    std::vector<ValueLatticeElement> StructValues;
    auto *STy = dyn_cast<StructType>(V->getType());
    assert(STy && "getStructLatticeValueFor() can be called only on structs");
    for (unsigned i = 0, e = STy->getNumElements(); i != e; ++i) {
      auto I = StructValueState.find(std::make_pair(V, i));
      assert(I != StructValueState.end() && "Value not in valuemap!");
      StructValues.push_back(I->second);
    }
    return StructValues;
  }

  void removeLatticeValueFor(Value *V) { ValueState.erase(V); }

  const ValueLatticeElement &getLatticeValueFor(Value *V) const {
    assert(!V->getType()->isStructTy() &&
           "Should use getStructLatticeValueFor");
    DenseMap<Value *, ValueLatticeElement>::const_iterator I =
        ValueState.find(V);
    assert(I != ValueState.end() &&
           "V not found in ValueState nor Paramstate map!");
    return I->second;
  }

  const MapVector<Function *, ValueLatticeElement> &getTrackedRetVals() {
    return TrackedRetVals;
  }

  const DenseMap<GlobalVariable *, ValueLatticeElement> &getTrackedGlobals() {
    return TrackedGlobals;
  }

  const SmallPtrSet<Function *, 16> getMRVFunctionsTracked() {
    return MRVFunctionsTracked;
  }

  void markOverdefined(Value *V) {
    if (auto *STy = dyn_cast<StructType>(V->getType()))
      for (unsigned i = 0, e = STy->getNumElements(); i != e; ++i)
        markOverdefined(getStructValueState(V, i), V);
    else
      markOverdefined(ValueState[V], V);
  }

  bool isStructLatticeConstant(Function *F, StructType *STy);

  Constant *getConstant(const ValueLatticeElement &LV) const;

  SmallPtrSetImpl<Function *> &getArgumentTrackedFunctions() {
    return TrackingIncomingArguments;
  }

  void markArgInFuncSpecialization(Function *F, Argument *A, Constant *C);

  void markFunctionUnreachable(Function *F) {
    for (auto &BB : *F)
      BBExecutable.erase(&BB);
  }
};

} // namespace llvm

bool SCCPInstVisitor::markBlockExecutable(BasicBlock *BB) {
  if (!BBExecutable.insert(BB).second)
    return false;
  LLVM_DEBUG(dbgs() << "Marking Block Executable: " << BB->getName() << '\n');
  BBWorkList.push_back(BB); // Add the block to the work list!
  return true;
}

void SCCPInstVisitor::pushToWorkList(ValueLatticeElement &IV, Value *V) {
  if (IV.isOverdefined())
    return OverdefinedInstWorkList.push_back(V);
  InstWorkList.push_back(V);
}

void SCCPInstVisitor::pushToWorkListMsg(ValueLatticeElement &IV, Value *V) {
  LLVM_DEBUG(dbgs() << "updated " << IV << ": " << *V << '\n');
  pushToWorkList(IV, V);
}

bool SCCPInstVisitor::markConstant(ValueLatticeElement &IV, Value *V,
                                   Constant *C, bool MayIncludeUndef) {
  if (!IV.markConstant(C, MayIncludeUndef))
    return false;
  LLVM_DEBUG(dbgs() << "markConstant: " << *C << ": " << *V << '\n');
  pushToWorkList(IV, V);
  return true;
}

bool SCCPInstVisitor::markOverdefined(ValueLatticeElement &IV, Value *V) {
  if (!IV.markOverdefined())
    return false;

  LLVM_DEBUG(dbgs() << "markOverdefined: ";
             if (auto *F = dyn_cast<Function>(V)) dbgs()
             << "Function '" << F->getName() << "'\n";
             else dbgs() << *V << '\n');
  // Only instructions go on the work list
  pushToWorkList(IV, V);
  return true;
}

bool SCCPInstVisitor::isStructLatticeConstant(Function *F, StructType *STy) {
  for (unsigned i = 0, e = STy->getNumElements(); i != e; ++i) {
    const auto &It = TrackedMultipleRetVals.find(std::make_pair(F, i));
    assert(It != TrackedMultipleRetVals.end());
    ValueLatticeElement LV = It->second;
    if (!isConstant(LV))
      return false;
  }
  return true;
}

Constant *SCCPInstVisitor::getConstant(const ValueLatticeElement &LV) const {
  if (LV.isConstant())
    return LV.getConstant();

  if (LV.isConstantRange()) {
    const auto &CR = LV.getConstantRange();
    if (CR.getSingleElement())
      return ConstantInt::get(Ctx, *CR.getSingleElement());
  }
  return nullptr;
}

void SCCPInstVisitor::markArgInFuncSpecialization(Function *F, Argument *A,
                                                  Constant *C) {
  assert(F->arg_size() == A->getParent()->arg_size() &&
         "Functions should have the same number of arguments");

  // Mark the argument constant in the new function.
  markConstant(A, C);

  // For the remaining arguments in the new function, copy the lattice state
  // over from the old function.
  for (auto I = F->arg_begin(), J = A->getParent()->arg_begin(),
            E = F->arg_end();
       I != E; ++I, ++J)
    if (J != A && ValueState.count(I)) {
      // Note: This previously looked like this:
      // ValueState[J] = ValueState[I];
      // This is incorrect because the DenseMap class may resize the underlying
      // memory when inserting `J`, which will invalidate the reference to `I`.
      // Instead, we make sure `J` exists, then set it to `I` afterwards.
      auto &NewValue = ValueState[J];
      NewValue = ValueState[I];
      pushToWorkList(NewValue, J);
    }
}

void SCCPInstVisitor::visitInstruction(Instruction &I) {
  // All the instructions we don't do any special handling for just
  // go to overdefined.
  LLVM_DEBUG(dbgs() << "SCCP: Don't know how to handle: " << I << '\n');
  markOverdefined(&I);
}

bool SCCPInstVisitor::mergeInValue(ValueLatticeElement &IV, Value *V,
                                   ValueLatticeElement MergeWithV,
                                   ValueLatticeElement::MergeOptions Opts) {
  if (IV.mergeIn(MergeWithV, Opts)) {
    pushToWorkList(IV, V);
    LLVM_DEBUG(dbgs() << "Merged " << MergeWithV << " into " << *V << " : "
                      << IV << "\n");
    return true;
  }
  return false;
}

bool SCCPInstVisitor::markEdgeExecutable(BasicBlock *Source, BasicBlock *Dest) {
  if (!KnownFeasibleEdges.insert(Edge(Source, Dest)).second)
    return false; // This edge is already known to be executable!

  if (!markBlockExecutable(Dest)) {
    // If the destination is already executable, we just made an *edge*
    // feasible that wasn't before.  Revisit the PHI nodes in the block
    // because they have potentially new operands.
    LLVM_DEBUG(dbgs() << "Marking Edge Executable: " << Source->getName()
                      << " -> " << Dest->getName() << '\n');

    for (PHINode &PN : Dest->phis())
      visitPHINode(PN);
  }
  return true;
}

// getFeasibleSuccessors - Return a vector of booleans to indicate which
// successors are reachable from a given terminator instruction.
void SCCPInstVisitor::getFeasibleSuccessors(Instruction &TI,
                                            SmallVectorImpl<bool> &Succs) {
  Succs.resize(TI.getNumSuccessors());
  if (auto *BI = dyn_cast<BranchInst>(&TI)) {
    if (BI->isUnconditional()) {
      Succs[0] = true;
      return;
    }

    ValueLatticeElement BCValue = getValueState(BI->getCondition());
    ConstantInt *CI = getConstantInt(BCValue);
    if (!CI) {
      // Overdefined condition variables, and branches on unfoldable constant
      // conditions, mean the branch could go either way.
      if (!BCValue.isUnknownOrUndef())
        Succs[0] = Succs[1] = true;
      return;
    }

    // Constant condition variables mean the branch can only go a single way.
    Succs[CI->isZero()] = true;
    return;
  }

  // Unwinding instructions successors are always executable.
  if (TI.isExceptionalTerminator()) {
    Succs.assign(TI.getNumSuccessors(), true);
    return;
  }

  if (auto *SI = dyn_cast<SwitchInst>(&TI)) {
    if (!SI->getNumCases()) {
      Succs[0] = true;
      return;
    }
    const ValueLatticeElement &SCValue = getValueState(SI->getCondition());
    if (ConstantInt *CI = getConstantInt(SCValue)) {
      Succs[SI->findCaseValue(CI)->getSuccessorIndex()] = true;
      return;
    }

    // TODO: Switch on undef is UB. Stop passing false once the rest of LLVM
    // is ready.
    if (SCValue.isConstantRange(/*UndefAllowed=*/false)) {
      const ConstantRange &Range = SCValue.getConstantRange();
      for (const auto &Case : SI->cases()) {
        const APInt &CaseValue = Case.getCaseValue()->getValue();
        if (Range.contains(CaseValue))
          Succs[Case.getSuccessorIndex()] = true;
      }

      // TODO: Determine whether default case is reachable.
      Succs[SI->case_default()->getSuccessorIndex()] = true;
      return;
    }

    // Overdefined or unknown condition? All destinations are executable!
    if (!SCValue.isUnknownOrUndef())
      Succs.assign(TI.getNumSuccessors(), true);
    return;
  }

  // In case of indirect branch and its address is a blockaddress, we mark
  // the target as executable.
  if (auto *IBR = dyn_cast<IndirectBrInst>(&TI)) {
    // Casts are folded by visitCastInst.
    ValueLatticeElement IBRValue = getValueState(IBR->getAddress());
    BlockAddress *Addr = dyn_cast_or_null<BlockAddress>(getConstant(IBRValue));
    if (!Addr) { // Overdefined or unknown condition?
      // All destinations are executable!
      if (!IBRValue.isUnknownOrUndef())
        Succs.assign(TI.getNumSuccessors(), true);
      return;
    }

    BasicBlock *T = Addr->getBasicBlock();
    assert(Addr->getFunction() == T->getParent() &&
           "Block address of a different function ?");
    for (unsigned i = 0; i < IBR->getNumSuccessors(); ++i) {
      // This is the target.
      if (IBR->getDestination(i) == T) {
        Succs[i] = true;
        return;
      }
    }

    // If we didn't find our destination in the IBR successor list, then we
    // have undefined behavior. Its ok to assume no successor is executable.
    return;
  }

  // In case of callbr, we pessimistically assume that all successors are
  // feasible.
  if (isa<CallBrInst>(&TI)) {
    Succs.assign(TI.getNumSuccessors(), true);
    return;
  }

  LLVM_DEBUG(dbgs() << "Unknown terminator instruction: " << TI << '\n');
  llvm_unreachable("SCCP: Don't know how to handle this terminator!");
}

// isEdgeFeasible - Return true if the control flow edge from the 'From' basic
// block to the 'To' basic block is currently feasible.
bool SCCPInstVisitor::isEdgeFeasible(BasicBlock *From, BasicBlock *To) const {
  // Check if we've called markEdgeExecutable on the edge yet. (We could
  // be more aggressive and try to consider edges which haven't been marked
  // yet, but there isn't any need.)
  return KnownFeasibleEdges.count(Edge(From, To));
}

// visit Implementations - Something changed in this instruction, either an
// operand made a transition, or the instruction is newly executable.  Change
// the value type of I to reflect these changes if appropriate.  This method
// makes sure to do the following actions:
//
// 1. If a phi node merges two constants in, and has conflicting value coming
//    from different branches, or if the PHI node merges in an overdefined
//    value, then the PHI node becomes overdefined.
// 2. If a phi node merges only constants in, and they all agree on value, the
//    PHI node becomes a constant value equal to that.
// 3. If V <- x (op) y && isConstant(x) && isConstant(y) V = Constant
// 4. If V <- x (op) y && (isOverdefined(x) || isOverdefined(y)) V = Overdefined
// 5. If V <- MEM or V <- CALL or V <- (unknown) then V = Overdefined
// 6. If a conditional branch has a value that is constant, make the selected
//    destination executable
// 7. If a conditional branch has a value that is overdefined, make all
//    successors executable.
void SCCPInstVisitor::visitPHINode(PHINode &PN) {
  // If this PN returns a struct, just mark the result overdefined.
  // TODO: We could do a lot better than this if code actually uses this.
  if (PN.getType()->isStructTy())
    return (void)markOverdefined(&PN);

  if (getValueState(&PN).isOverdefined())
    return; // Quick exit

  // Super-extra-high-degree PHI nodes are unlikely to ever be marked constant,
  // and slow us down a lot.  Just mark them overdefined.
  if (PN.getNumIncomingValues() > 64)
    return (void)markOverdefined(&PN);

  unsigned NumActiveIncoming = 0;

  // Look at all of the executable operands of the PHI node.  If any of them
  // are overdefined, the PHI becomes overdefined as well.  If they are all
  // constant, and they agree with each other, the PHI becomes the identical
  // constant.  If they are constant and don't agree, the PHI is a constant
  // range. If there are no executable operands, the PHI remains unknown.
  ValueLatticeElement PhiState = getValueState(&PN);
  for (unsigned i = 0, e = PN.getNumIncomingValues(); i != e; ++i) {
    if (!isEdgeFeasible(PN.getIncomingBlock(i), PN.getParent()))
      continue;

    ValueLatticeElement IV = getValueState(PN.getIncomingValue(i));
    PhiState.mergeIn(IV);
    NumActiveIncoming++;
    if (PhiState.isOverdefined())
      break;
  }

  // We allow up to 1 range extension per active incoming value and one
  // additional extension. Note that we manually adjust the number of range
  // extensions to match the number of active incoming values. This helps to
  // limit multiple extensions caused by the same incoming value, if other
  // incoming values are equal.
  mergeInValue(&PN, PhiState,
               ValueLatticeElement::MergeOptions().setMaxWidenSteps(
                   NumActiveIncoming + 1));
  ValueLatticeElement &PhiStateRef = getValueState(&PN);
  PhiStateRef.setNumRangeExtensions(
      std::max(NumActiveIncoming, PhiStateRef.getNumRangeExtensions()));
}

void SCCPInstVisitor::visitReturnInst(ReturnInst &I) {
  if (I.getNumOperands() == 0)
    return; // ret void

  Function *F = I.getParent()->getParent();
  Value *ResultOp = I.getOperand(0);

  // If we are tracking the return value of this function, merge it in.
  if (!TrackedRetVals.empty() && !ResultOp->getType()->isStructTy()) {
    auto TFRVI = TrackedRetVals.find(F);
    if (TFRVI != TrackedRetVals.end()) {
      mergeInValue(TFRVI->second, F, getValueState(ResultOp));
      return;
    }
  }

  // Handle functions that return multiple values.
  if (!TrackedMultipleRetVals.empty()) {
    if (auto *STy = dyn_cast<StructType>(ResultOp->getType()))
      if (MRVFunctionsTracked.count(F))
        for (unsigned i = 0, e = STy->getNumElements(); i != e; ++i)
          mergeInValue(TrackedMultipleRetVals[std::make_pair(F, i)], F,
                       getStructValueState(ResultOp, i));
  }
}

void SCCPInstVisitor::visitTerminator(Instruction &TI) {
  SmallVector<bool, 16> SuccFeasible;
  getFeasibleSuccessors(TI, SuccFeasible);

  BasicBlock *BB = TI.getParent();

  // Mark all feasible successors executable.
  for (unsigned i = 0, e = SuccFeasible.size(); i != e; ++i)
    if (SuccFeasible[i])
      markEdgeExecutable(BB, TI.getSuccessor(i));
}

void SCCPInstVisitor::visitCastInst(CastInst &I) {
  // ResolvedUndefsIn might mark I as overdefined. Bail out, even if we would
  // discover a concrete value later.
  if (ValueState[&I].isOverdefined())
    return;

  ValueLatticeElement OpSt = getValueState(I.getOperand(0));
  if (OpSt.isUnknownOrUndef())
    return;

  if (Constant *OpC = getConstant(OpSt)) {
    // Fold the constant as we build.
    Constant *C = ConstantFoldCastOperand(I.getOpcode(), OpC, I.getType(), DL);
    if (isa<UndefValue>(C))
      return;
    // Propagate constant value
    markConstant(&I, C);
  } else if (I.getDestTy()->isIntegerTy()) {
    auto &LV = getValueState(&I);
    ConstantRange OpRange =
        OpSt.isConstantRange()
            ? OpSt.getConstantRange()
            : ConstantRange::getFull(
                  I.getOperand(0)->getType()->getScalarSizeInBits());

    Type *DestTy = I.getDestTy();
    // Vectors where all elements have the same known constant range are treated
    // as a single constant range in the lattice. When bitcasting such vectors,
    // there is a mis-match between the width of the lattice value (single
    // constant range) and the original operands (vector). Go to overdefined in
    // that case.
    if (I.getOpcode() == Instruction::BitCast &&
        I.getOperand(0)->getType()->isVectorTy() &&
        OpRange.getBitWidth() < DL.getTypeSizeInBits(DestTy))
      return (void)markOverdefined(&I);

    ConstantRange Res =
        OpRange.castOp(I.getOpcode(), DL.getTypeSizeInBits(DestTy));
    mergeInValue(LV, &I, ValueLatticeElement::getRange(Res));
  } else
    markOverdefined(&I);
}

void SCCPInstVisitor::visitExtractValueInst(ExtractValueInst &EVI) {
  // If this returns a struct, mark all elements over defined, we don't track
  // structs in structs.
  if (EVI.getType()->isStructTy())
    return (void)markOverdefined(&EVI);

  // resolvedUndefsIn might mark I as overdefined. Bail out, even if we would
  // discover a concrete value later.
  if (ValueState[&EVI].isOverdefined())
    return (void)markOverdefined(&EVI);

  // If this is extracting from more than one level of struct, we don't know.
  if (EVI.getNumIndices() != 1)
    return (void)markOverdefined(&EVI);

  Value *AggVal = EVI.getAggregateOperand();
  if (AggVal->getType()->isStructTy()) {
    unsigned i = *EVI.idx_begin();
    ValueLatticeElement EltVal = getStructValueState(AggVal, i);
    mergeInValue(getValueState(&EVI), &EVI, EltVal);
  } else {
    // Otherwise, must be extracting from an array.
    return (void)markOverdefined(&EVI);
  }
}

void SCCPInstVisitor::visitInsertValueInst(InsertValueInst &IVI) {
  auto *STy = dyn_cast<StructType>(IVI.getType());
  if (!STy)
    return (void)markOverdefined(&IVI);

  // resolvedUndefsIn might mark I as overdefined. Bail out, even if we would
  // discover a concrete value later.
  if (isOverdefined(ValueState[&IVI]))
    return (void)markOverdefined(&IVI);

  // If this has more than one index, we can't handle it, drive all results to
  // undef.
  if (IVI.getNumIndices() != 1)
    return (void)markOverdefined(&IVI);

  Value *Aggr = IVI.getAggregateOperand();
  unsigned Idx = *IVI.idx_begin();

  // Compute the result based on what we're inserting.
  for (unsigned i = 0, e = STy->getNumElements(); i != e; ++i) {
    // This passes through all values that aren't the inserted element.
    if (i != Idx) {
      ValueLatticeElement EltVal = getStructValueState(Aggr, i);
      mergeInValue(getStructValueState(&IVI, i), &IVI, EltVal);
      continue;
    }

    Value *Val = IVI.getInsertedValueOperand();
    if (Val->getType()->isStructTy())
      // We don't track structs in structs.
      markOverdefined(getStructValueState(&IVI, i), &IVI);
    else {
      ValueLatticeElement InVal = getValueState(Val);
      mergeInValue(getStructValueState(&IVI, i), &IVI, InVal);
    }
  }
}

void SCCPInstVisitor::visitSelectInst(SelectInst &I) {
  // If this select returns a struct, just mark the result overdefined.
  // TODO: We could do a lot better than this if code actually uses this.
  if (I.getType()->isStructTy())
    return (void)markOverdefined(&I);

  // resolvedUndefsIn might mark I as overdefined. Bail out, even if we would
  // discover a concrete value later.
  if (ValueState[&I].isOverdefined())
    return (void)markOverdefined(&I);

  ValueLatticeElement CondValue = getValueState(I.getCondition());
  if (CondValue.isUnknownOrUndef())
    return;

  if (ConstantInt *CondCB = getConstantInt(CondValue)) {
    Value *OpVal = CondCB->isZero() ? I.getFalseValue() : I.getTrueValue();
    mergeInValue(&I, getValueState(OpVal));
    return;
  }

  // Otherwise, the condition is overdefined or a constant we can't evaluate.
  // See if we can produce something better than overdefined based on the T/F
  // value.
  ValueLatticeElement TVal = getValueState(I.getTrueValue());
  ValueLatticeElement FVal = getValueState(I.getFalseValue());

  bool Changed = ValueState[&I].mergeIn(TVal);
  Changed |= ValueState[&I].mergeIn(FVal);
  if (Changed)
    pushToWorkListMsg(ValueState[&I], &I);
}

// Handle Unary Operators.
void SCCPInstVisitor::visitUnaryOperator(Instruction &I) {
  ValueLatticeElement V0State = getValueState(I.getOperand(0));

  ValueLatticeElement &IV = ValueState[&I];
  // resolvedUndefsIn might mark I as overdefined. Bail out, even if we would
  // discover a concrete value later.
  if (isOverdefined(IV))
    return (void)markOverdefined(&I);

  if (isConstant(V0State)) {
    Constant *C = ConstantExpr::get(I.getOpcode(), getConstant(V0State));

    // op Y -> undef.
    if (isa<UndefValue>(C))
      return;
    return (void)markConstant(IV, &I, C);
  }

  // If something is undef, wait for it to resolve.
  if (!isOverdefined(V0State))
    return;

  markOverdefined(&I);
}

// Handle Binary Operators.
void SCCPInstVisitor::visitBinaryOperator(Instruction &I) {
  ValueLatticeElement V1State = getValueState(I.getOperand(0));
  ValueLatticeElement V2State = getValueState(I.getOperand(1));

  ValueLatticeElement &IV = ValueState[&I];
  if (IV.isOverdefined())
    return;

  // If something is undef, wait for it to resolve.
  if (V1State.isUnknownOrUndef() || V2State.isUnknownOrUndef())
    return;

  if (V1State.isOverdefined() && V2State.isOverdefined())
    return (void)markOverdefined(&I);

  // If either of the operands is a constant, try to fold it to a constant.
  // TODO: Use information from notconstant better.
  if ((V1State.isConstant() || V2State.isConstant())) {
    Value *V1 = isConstant(V1State) ? getConstant(V1State) : I.getOperand(0);
    Value *V2 = isConstant(V2State) ? getConstant(V2State) : I.getOperand(1);
    Value *R = SimplifyBinOp(I.getOpcode(), V1, V2, SimplifyQuery(DL));
    auto *C = dyn_cast_or_null<Constant>(R);
    if (C) {
      // X op Y -> undef.
      if (isa<UndefValue>(C))
        return;
      // Conservatively assume that the result may be based on operands that may
      // be undef. Note that we use mergeInValue to combine the constant with
      // the existing lattice value for I, as different constants might be found
      // after one of the operands go to overdefined, e.g. due to one operand
      // being a special floating value.
      ValueLatticeElement NewV;
      NewV.markConstant(C, /*MayIncludeUndef=*/true);
      return (void)mergeInValue(&I, NewV);
    }
  }

  // Only use ranges for binary operators on integers.
  if (!I.getType()->isIntegerTy())
    return markOverdefined(&I);

  // Try to simplify to a constant range.
  ConstantRange A = ConstantRange::getFull(I.getType()->getScalarSizeInBits());
  ConstantRange B = ConstantRange::getFull(I.getType()->getScalarSizeInBits());
  if (V1State.isConstantRange())
    A = V1State.getConstantRange();
  if (V2State.isConstantRange())
    B = V2State.getConstantRange();

  ConstantRange R = A.binaryOp(cast<BinaryOperator>(&I)->getOpcode(), B);
  mergeInValue(&I, ValueLatticeElement::getRange(R));

  // TODO: Currently we do not exploit special values that produce something
  // better than overdefined with an overdefined operand for vector or floating
  // point types, like and <4 x i32> overdefined, zeroinitializer.
}

// Handle ICmpInst instruction.
void SCCPInstVisitor::visitCmpInst(CmpInst &I) {
  // Do not cache this lookup, getValueState calls later in the function might
  // invalidate the reference.
  if (isOverdefined(ValueState[&I]))
    return (void)markOverdefined(&I);

  Value *Op1 = I.getOperand(0);
  Value *Op2 = I.getOperand(1);

  // For parameters, use ParamState which includes constant range info if
  // available.
  auto V1State = getValueState(Op1);
  auto V2State = getValueState(Op2);

  Constant *C = V1State.getCompare(I.getPredicate(), I.getType(), V2State);
  if (C) {
    if (isa<UndefValue>(C))
      return;
    ValueLatticeElement CV;
    CV.markConstant(C);
    mergeInValue(&I, CV);
    return;
  }

  // If operands are still unknown, wait for it to resolve.
  if ((V1State.isUnknownOrUndef() || V2State.isUnknownOrUndef()) &&
      !isConstant(ValueState[&I]))
    return;

  markOverdefined(&I);
}

// Handle getelementptr instructions.  If all operands are constants then we
// can turn this into a getelementptr ConstantExpr.
void SCCPInstVisitor::visitGetElementPtrInst(GetElementPtrInst &I) {
  if (isOverdefined(ValueState[&I]))
    return (void)markOverdefined(&I);

  SmallVector<Constant *, 8> Operands;
  Operands.reserve(I.getNumOperands());

  for (unsigned i = 0, e = I.getNumOperands(); i != e; ++i) {
    ValueLatticeElement State = getValueState(I.getOperand(i));
    if (State.isUnknownOrUndef())
      return; // Operands are not resolved yet.

    if (isOverdefined(State))
      return (void)markOverdefined(&I);

    if (Constant *C = getConstant(State)) {
      Operands.push_back(C);
      continue;
    }

    return (void)markOverdefined(&I);
  }

  Constant *Ptr = Operands[0];
  auto Indices = makeArrayRef(Operands.begin() + 1, Operands.end());
  Constant *C =
      ConstantExpr::getGetElementPtr(I.getSourceElementType(), Ptr, Indices);
  if (isa<UndefValue>(C))
    return;
  markConstant(&I, C);
}

void SCCPInstVisitor::visitStoreInst(StoreInst &SI) {
  // If this store is of a struct, ignore it.
  if (SI.getOperand(0)->getType()->isStructTy())
    return;

  if (TrackedGlobals.empty() || !isa<GlobalVariable>(SI.getOperand(1)))
    return;

  GlobalVariable *GV = cast<GlobalVariable>(SI.getOperand(1));
  auto I = TrackedGlobals.find(GV);
  if (I == TrackedGlobals.end())
    return;

  // Get the value we are storing into the global, then merge it.
  mergeInValue(I->second, GV, getValueState(SI.getOperand(0)),
               ValueLatticeElement::MergeOptions().setCheckWiden(false));
  if (I->second.isOverdefined())
    TrackedGlobals.erase(I); // No need to keep tracking this!
}

static ValueLatticeElement getValueFromMetadata(const Instruction *I) {
  if (MDNode *Ranges = I->getMetadata(LLVMContext::MD_range))
    if (I->getType()->isIntegerTy())
      return ValueLatticeElement::getRange(
          getConstantRangeFromMetadata(*Ranges));
  if (I->hasMetadata(LLVMContext::MD_nonnull))
    return ValueLatticeElement::getNot(
        ConstantPointerNull::get(cast<PointerType>(I->getType())));
  return ValueLatticeElement::getOverdefined();
}

// Handle load instructions.  If the operand is a constant pointer to a constant
// global, we can replace the load with the loaded constant value!
void SCCPInstVisitor::visitLoadInst(LoadInst &I) {
  // If this load is of a struct or the load is volatile, just mark the result
  // as overdefined.
  if (I.getType()->isStructTy() || I.isVolatile())
    return (void)markOverdefined(&I);

  // resolvedUndefsIn might mark I as overdefined. Bail out, even if we would
  // discover a concrete value later.
  if (ValueState[&I].isOverdefined())
    return (void)markOverdefined(&I);

  ValueLatticeElement PtrVal = getValueState(I.getOperand(0));
  if (PtrVal.isUnknownOrUndef())
    return; // The pointer is not resolved yet!

  ValueLatticeElement &IV = ValueState[&I];

  if (isConstant(PtrVal)) {
    Constant *Ptr = getConstant(PtrVal);

    // load null is undefined.
    if (isa<ConstantPointerNull>(Ptr)) {
      if (NullPointerIsDefined(I.getFunction(), I.getPointerAddressSpace()))
        return (void)markOverdefined(IV, &I);
      else
        return;
    }

    // Transform load (constant global) into the value loaded.
    if (auto *GV = dyn_cast<GlobalVariable>(Ptr)) {
      if (!TrackedGlobals.empty()) {
        // If we are tracking this global, merge in the known value for it.
        auto It = TrackedGlobals.find(GV);
        if (It != TrackedGlobals.end()) {
          mergeInValue(IV, &I, It->second, getMaxWidenStepsOpts());
          return;
        }
      }
    }

    // Transform load from a constant into a constant if possible.
    if (Constant *C = ConstantFoldLoadFromConstPtr(Ptr, I.getType(), DL)) {
      if (isa<UndefValue>(C))
        return;
      return (void)markConstant(IV, &I, C);
    }
  }

  // Fall back to metadata.
  mergeInValue(&I, getValueFromMetadata(&I));
}

void SCCPInstVisitor::visitCallBase(CallBase &CB) {
  handleCallResult(CB);
  handleCallArguments(CB);
}

void SCCPInstVisitor::handleCallOverdefined(CallBase &CB) {
  Function *F = CB.getCalledFunction();

  // Void return and not tracking callee, just bail.
  if (CB.getType()->isVoidTy())
    return;

  // Always mark struct return as overdefined.
  if (CB.getType()->isStructTy())
    return (void)markOverdefined(&CB);

  // Otherwise, if we have a single return value case, and if the function is
  // a declaration, maybe we can constant fold it.
  if (F && F->isDeclaration() && canConstantFoldCallTo(&CB, F)) {
    SmallVector<Constant *, 8> Operands;
    for (const Use &A : CB.args()) {
      if (A.get()->getType()->isStructTy())
        return markOverdefined(&CB); // Can't handle struct args.
      ValueLatticeElement State = getValueState(A);

      if (State.isUnknownOrUndef())
        return; // Operands are not resolved yet.
      if (isOverdefined(State))
        return (void)markOverdefined(&CB);
      assert(isConstant(State) && "Unknown state!");
      Operands.push_back(getConstant(State));
    }

    if (isOverdefined(getValueState(&CB)))
      return (void)markOverdefined(&CB);

    // If we can constant fold this, mark the result of the call as a
    // constant.
    if (Constant *C = ConstantFoldCall(&CB, F, Operands, &GetTLI(*F))) {
      // call -> undef.
      if (isa<UndefValue>(C))
        return;
      return (void)markConstant(&CB, C);
    }
  }

  // Fall back to metadata.
  mergeInValue(&CB, getValueFromMetadata(&CB));
}

void SCCPInstVisitor::handleCallArguments(CallBase &CB) {
  Function *F = CB.getCalledFunction();
  // If this is a local function that doesn't have its address taken, mark its
  // entry block executable and merge in the actual arguments to the call into
  // the formal arguments of the function.
  if (!TrackingIncomingArguments.empty() &&
      TrackingIncomingArguments.count(F)) {
    markBlockExecutable(&F->front());

    // Propagate information from this call site into the callee.
    auto CAI = CB.arg_begin();
    for (Function::arg_iterator AI = F->arg_begin(), E = F->arg_end(); AI != E;
         ++AI, ++CAI) {
      // If this argument is byval, and if the function is not readonly, there
      // will be an implicit copy formed of the input aggregate.
      if (AI->hasByValAttr() && !F->onlyReadsMemory()) {
        markOverdefined(&*AI);
        continue;
      }

      if (auto *STy = dyn_cast<StructType>(AI->getType())) {
        for (unsigned i = 0, e = STy->getNumElements(); i != e; ++i) {
          ValueLatticeElement CallArg = getStructValueState(*CAI, i);
          mergeInValue(getStructValueState(&*AI, i), &*AI, CallArg,
                       getMaxWidenStepsOpts());
        }
      } else
        mergeInValue(&*AI, getValueState(*CAI), getMaxWidenStepsOpts());
    }
  }
}

void SCCPInstVisitor::handleCallResult(CallBase &CB) {
  Function *F = CB.getCalledFunction();

  if (auto *II = dyn_cast<IntrinsicInst>(&CB)) {
    if (II->getIntrinsicID() == Intrinsic::ssa_copy) {
      if (ValueState[&CB].isOverdefined())
        return;

      Value *CopyOf = CB.getOperand(0);
      ValueLatticeElement CopyOfVal = getValueState(CopyOf);
      const auto *PI = getPredicateInfoFor(&CB);
      assert(PI && "Missing predicate info for ssa.copy");

      const Optional<PredicateConstraint> &Constraint = PI->getConstraint();
      if (!Constraint) {
        mergeInValue(ValueState[&CB], &CB, CopyOfVal);
        return;
      }

      CmpInst::Predicate Pred = Constraint->Predicate;
      Value *OtherOp = Constraint->OtherOp;

      // Wait until OtherOp is resolved.
      if (getValueState(OtherOp).isUnknown()) {
        addAdditionalUser(OtherOp, &CB);
        return;
      }

      // TODO: Actually filp MayIncludeUndef for the created range to false,
      // once most places in the optimizer respect the branches on
      // undef/poison are UB rule. The reason why the new range cannot be
      // undef is as follows below:
      // The new range is based on a branch condition. That guarantees that
      // neither of the compare operands can be undef in the branch targets,
      // unless we have conditions that are always true/false (e.g. icmp ule
      // i32, %a, i32_max). For the latter overdefined/empty range will be
      // inferred, but the branch will get folded accordingly anyways.
      bool MayIncludeUndef = !isa<PredicateAssume>(PI);

      ValueLatticeElement CondVal = getValueState(OtherOp);
      ValueLatticeElement &IV = ValueState[&CB];
      if (CondVal.isConstantRange() || CopyOfVal.isConstantRange()) {
        auto ImposedCR =
            ConstantRange::getFull(DL.getTypeSizeInBits(CopyOf->getType()));

        // Get the range imposed by the condition.
        if (CondVal.isConstantRange())
          ImposedCR = ConstantRange::makeAllowedICmpRegion(
              Pred, CondVal.getConstantRange());

        // Combine range info for the original value with the new range from the
        // condition.
        auto CopyOfCR = CopyOfVal.isConstantRange()
                            ? CopyOfVal.getConstantRange()
                            : ConstantRange::getFull(
                                  DL.getTypeSizeInBits(CopyOf->getType()));
        auto NewCR = ImposedCR.intersectWith(CopyOfCR);
        // If the existing information is != x, do not use the information from
        // a chained predicate, as the != x information is more likely to be
        // helpful in practice.
        if (!CopyOfCR.contains(NewCR) && CopyOfCR.getSingleMissingElement())
          NewCR = CopyOfCR;

        addAdditionalUser(OtherOp, &CB);
        mergeInValue(IV, &CB,
                     ValueLatticeElement::getRange(NewCR, MayIncludeUndef));
        return;
      } else if (Pred == CmpInst::ICMP_EQ && CondVal.isConstant()) {
        // For non-integer values or integer constant expressions, only
        // propagate equal constants.
        addAdditionalUser(OtherOp, &CB);
        mergeInValue(IV, &CB, CondVal);
        return;
      } else if (Pred == CmpInst::ICMP_NE && CondVal.isConstant() &&
                 !MayIncludeUndef) {
        // Propagate inequalities.
        addAdditionalUser(OtherOp, &CB);
        mergeInValue(IV, &CB,
                     ValueLatticeElement::getNot(CondVal.getConstant()));
        return;
      }

      return (void)mergeInValue(IV, &CB, CopyOfVal);
    }

    if (ConstantRange::isIntrinsicSupported(II->getIntrinsicID())) {
      // Compute result range for intrinsics supported by ConstantRange.
      // Do this even if we don't know a range for all operands, as we may
      // still know something about the result range, e.g. of abs(x).
      SmallVector<ConstantRange, 2> OpRanges;
      for (Value *Op : II->args()) {
        const ValueLatticeElement &State = getValueState(Op);
        if (State.isConstantRange())
          OpRanges.push_back(State.getConstantRange());
        else
          OpRanges.push_back(
              ConstantRange::getFull(Op->getType()->getScalarSizeInBits()));
      }

      ConstantRange Result =
          ConstantRange::intrinsic(II->getIntrinsicID(), OpRanges);
      return (void)mergeInValue(II, ValueLatticeElement::getRange(Result));
    }
  }

  // The common case is that we aren't tracking the callee, either because we
  // are not doing interprocedural analysis or the callee is indirect, or is
  // external.  Handle these cases first.
  if (!F || F->isDeclaration())
    return handleCallOverdefined(CB);

  // If this is a single/zero retval case, see if we're tracking the function.
  if (auto *STy = dyn_cast<StructType>(F->getReturnType())) {
    if (!MRVFunctionsTracked.count(F))
      return handleCallOverdefined(CB); // Not tracking this callee.

    // If we are tracking this callee, propagate the result of the function
    // into this call site.
    for (unsigned i = 0, e = STy->getNumElements(); i != e; ++i)
      mergeInValue(getStructValueState(&CB, i), &CB,
                   TrackedMultipleRetVals[std::make_pair(F, i)],
                   getMaxWidenStepsOpts());
  } else {
    auto TFRVI = TrackedRetVals.find(F);
    if (TFRVI == TrackedRetVals.end())
      return handleCallOverdefined(CB); // Not tracking this callee.

    // If so, propagate the return value of the callee into this call result.
    mergeInValue(&CB, TFRVI->second, getMaxWidenStepsOpts());
  }
}

void SCCPInstVisitor::solve() {
  // Process the work lists until they are empty!
  while (!BBWorkList.empty() || !InstWorkList.empty() ||
         !OverdefinedInstWorkList.empty()) {
    // Process the overdefined instruction's work list first, which drives other
    // things to overdefined more quickly.
    while (!OverdefinedInstWorkList.empty()) {
      Value *I = OverdefinedInstWorkList.pop_back_val();

      LLVM_DEBUG(dbgs() << "\nPopped off OI-WL: " << *I << '\n');

      // "I" got into the work list because it either made the transition from
      // bottom to constant, or to overdefined.
      //
      // Anything on this worklist that is overdefined need not be visited
      // since all of its users will have already been marked as overdefined
      // Update all of the users of this instruction's value.
      //
      markUsersAsChanged(I);
    }

    // Process the instruction work list.
    while (!InstWorkList.empty()) {
      Value *I = InstWorkList.pop_back_val();

      LLVM_DEBUG(dbgs() << "\nPopped off I-WL: " << *I << '\n');

      // "I" got into the work list because it made the transition from undef to
      // constant.
      //
      // Anything on this worklist that is overdefined need not be visited
      // since all of its users will have already been marked as overdefined.
      // Update all of the users of this instruction's value.
      //
      if (I->getType()->isStructTy() || !getValueState(I).isOverdefined())
        markUsersAsChanged(I);
    }

    // Process the basic block work list.
    while (!BBWorkList.empty()) {
      BasicBlock *BB = BBWorkList.pop_back_val();

      LLVM_DEBUG(dbgs() << "\nPopped off BBWL: " << *BB << '\n');

      // Notify all instructions in this basic block that they are newly
      // executable.
      visit(BB);
    }
  }
}

/// resolvedUndefsIn - While solving the dataflow for a function, we assume
/// that branches on undef values cannot reach any of their successors.
/// However, this is not a safe assumption.  After we solve dataflow, this
/// method should be use to handle this.  If this returns true, the solver
/// should be rerun.
///
/// This method handles this by finding an unresolved branch and marking it one
/// of the edges from the block as being feasible, even though the condition
/// doesn't say it would otherwise be.  This allows SCCP to find the rest of the
/// CFG and only slightly pessimizes the analysis results (by marking one,
/// potentially infeasible, edge feasible).  This cannot usefully modify the
/// constraints on the condition of the branch, as that would impact other users
/// of the value.
///
/// This scan also checks for values that use undefs. It conservatively marks
/// them as overdefined.
bool SCCPInstVisitor::resolvedUndefsIn(Function &F) {
  bool MadeChange = false;
  for (BasicBlock &BB : F) {
    if (!BBExecutable.count(&BB))
      continue;

    for (Instruction &I : BB) {
      // Look for instructions which produce undef values.
      if (I.getType()->isVoidTy())
        continue;

      if (auto *STy = dyn_cast<StructType>(I.getType())) {
        // Only a few things that can be structs matter for undef.

        // Tracked calls must never be marked overdefined in resolvedUndefsIn.
        if (auto *CB = dyn_cast<CallBase>(&I))
          if (Function *F = CB->getCalledFunction())
            if (MRVFunctionsTracked.count(F))
              continue;

        // extractvalue and insertvalue don't need to be marked; they are
        // tracked as precisely as their operands.
        if (isa<ExtractValueInst>(I) || isa<InsertValueInst>(I))
          continue;
        // Send the results of everything else to overdefined.  We could be
        // more precise than this but it isn't worth bothering.
        for (unsigned i = 0, e = STy->getNumElements(); i != e; ++i) {
          ValueLatticeElement &LV = getStructValueState(&I, i);
          if (LV.isUnknownOrUndef()) {
            markOverdefined(LV, &I);
            MadeChange = true;
          }
        }
        continue;
      }

      ValueLatticeElement &LV = getValueState(&I);
      if (!LV.isUnknownOrUndef())
        continue;

      // There are two reasons a call can have an undef result
      // 1. It could be tracked.
      // 2. It could be constant-foldable.
      // Because of the way we solve return values, tracked calls must
      // never be marked overdefined in resolvedUndefsIn.
      if (auto *CB = dyn_cast<CallBase>(&I))
        if (Function *F = CB->getCalledFunction())
          if (TrackedRetVals.count(F))
            continue;

      if (isa<LoadInst>(I)) {
        // A load here means one of two things: a load of undef from a global,
        // a load from an unknown pointer.  Either way, having it return undef
        // is okay.
        continue;
      }

      markOverdefined(&I);
      MadeChange = true;
    }

    // Check to see if we have a branch or switch on an undefined value.  If so
    // we force the branch to go one way or the other to make the successor
    // values live.  It doesn't really matter which way we force it.
    Instruction *TI = BB.getTerminator();
    if (auto *BI = dyn_cast<BranchInst>(TI)) {
      if (!BI->isConditional())
        continue;
      if (!getValueState(BI->getCondition()).isUnknownOrUndef())
        continue;

      // If the input to SCCP is actually branch on undef, fix the undef to
      // false.
      if (isa<UndefValue>(BI->getCondition())) {
        BI->setCondition(ConstantInt::getFalse(BI->getContext()));
        markEdgeExecutable(&BB, TI->getSuccessor(1));
        MadeChange = true;
        continue;
      }

      // Otherwise, it is a branch on a symbolic value which is currently
      // considered to be undef.  Make sure some edge is executable, so a
      // branch on "undef" always flows somewhere.
      // FIXME: Distinguish between dead code and an LLVM "undef" value.
      BasicBlock *DefaultSuccessor = TI->getSuccessor(1);
      if (markEdgeExecutable(&BB, DefaultSuccessor))
        MadeChange = true;

      continue;
    }

    if (auto *IBR = dyn_cast<IndirectBrInst>(TI)) {
      // Indirect branch with no successor ?. Its ok to assume it branches
      // to no target.
      if (IBR->getNumSuccessors() < 1)
        continue;

      if (!getValueState(IBR->getAddress()).isUnknownOrUndef())
        continue;

      // If the input to SCCP is actually branch on undef, fix the undef to
      // the first successor of the indirect branch.
      if (isa<UndefValue>(IBR->getAddress())) {
        IBR->setAddress(BlockAddress::get(IBR->getSuccessor(0)));
        markEdgeExecutable(&BB, IBR->getSuccessor(0));
        MadeChange = true;
        continue;
      }

      // Otherwise, it is a branch on a symbolic value which is currently
      // considered to be undef.  Make sure some edge is executable, so a
      // branch on "undef" always flows somewhere.
      // FIXME: IndirectBr on "undef" doesn't actually need to go anywhere:
      // we can assume the branch has undefined behavior instead.
      BasicBlock *DefaultSuccessor = IBR->getSuccessor(0);
      if (markEdgeExecutable(&BB, DefaultSuccessor))
        MadeChange = true;

      continue;
    }

    if (auto *SI = dyn_cast<SwitchInst>(TI)) {
      if (!SI->getNumCases() ||
          !getValueState(SI->getCondition()).isUnknownOrUndef())
        continue;

      // If the input to SCCP is actually switch on undef, fix the undef to
      // the first constant.
      if (isa<UndefValue>(SI->getCondition())) {
        SI->setCondition(SI->case_begin()->getCaseValue());
        markEdgeExecutable(&BB, SI->case_begin()->getCaseSuccessor());
        MadeChange = true;
        continue;
      }

      // Otherwise, it is a branch on a symbolic value which is currently
      // considered to be undef.  Make sure some edge is executable, so a
      // branch on "undef" always flows somewhere.
      // FIXME: Distinguish between dead code and an LLVM "undef" value.
      BasicBlock *DefaultSuccessor = SI->case_begin()->getCaseSuccessor();
      if (markEdgeExecutable(&BB, DefaultSuccessor))
        MadeChange = true;

      continue;
    }
  }

  return MadeChange;
}

//===----------------------------------------------------------------------===//
//
// SCCPSolver implementations
//
SCCPSolver::SCCPSolver(
    const DataLayout &DL,
    std::function<const TargetLibraryInfo &(Function &)> GetTLI,
    LLVMContext &Ctx)
    : Visitor(new SCCPInstVisitor(DL, std::move(GetTLI), Ctx)) {}

SCCPSolver::~SCCPSolver() = default;

void SCCPSolver::addAnalysis(Function &F, AnalysisResultsForFn A) {
  return Visitor->addAnalysis(F, std::move(A));
}

bool SCCPSolver::markBlockExecutable(BasicBlock *BB) {
  return Visitor->markBlockExecutable(BB);
}

const PredicateBase *SCCPSolver::getPredicateInfoFor(Instruction *I) {
  return Visitor->getPredicateInfoFor(I);
}

DomTreeUpdater SCCPSolver::getDTU(Function &F) { return Visitor->getDTU(F); }

void SCCPSolver::trackValueOfGlobalVariable(GlobalVariable *GV) {
  Visitor->trackValueOfGlobalVariable(GV);
}

void SCCPSolver::addTrackedFunction(Function *F) {
  Visitor->addTrackedFunction(F);
}

void SCCPSolver::addToMustPreserveReturnsInFunctions(Function *F) {
  Visitor->addToMustPreserveReturnsInFunctions(F);
}

bool SCCPSolver::mustPreserveReturn(Function *F) {
  return Visitor->mustPreserveReturn(F);
}

void SCCPSolver::addArgumentTrackedFunction(Function *F) {
  Visitor->addArgumentTrackedFunction(F);
}

bool SCCPSolver::isArgumentTrackedFunction(Function *F) {
  return Visitor->isArgumentTrackedFunction(F);
}

void SCCPSolver::solve() { Visitor->solve(); }

bool SCCPSolver::resolvedUndefsIn(Function &F) {
  return Visitor->resolvedUndefsIn(F);
}

bool SCCPSolver::isBlockExecutable(BasicBlock *BB) const {
  return Visitor->isBlockExecutable(BB);
}

bool SCCPSolver::isEdgeFeasible(BasicBlock *From, BasicBlock *To) const {
  return Visitor->isEdgeFeasible(From, To);
}

std::vector<ValueLatticeElement>
SCCPSolver::getStructLatticeValueFor(Value *V) const {
  return Visitor->getStructLatticeValueFor(V);
}

void SCCPSolver::removeLatticeValueFor(Value *V) {
  return Visitor->removeLatticeValueFor(V);
}

const ValueLatticeElement &SCCPSolver::getLatticeValueFor(Value *V) const {
  return Visitor->getLatticeValueFor(V);
}

const MapVector<Function *, ValueLatticeElement> &
SCCPSolver::getTrackedRetVals() {
  return Visitor->getTrackedRetVals();
}

const DenseMap<GlobalVariable *, ValueLatticeElement> &
SCCPSolver::getTrackedGlobals() {
  return Visitor->getTrackedGlobals();
}

const SmallPtrSet<Function *, 16> SCCPSolver::getMRVFunctionsTracked() {
  return Visitor->getMRVFunctionsTracked();
}

void SCCPSolver::markOverdefined(Value *V) { Visitor->markOverdefined(V); }

bool SCCPSolver::isStructLatticeConstant(Function *F, StructType *STy) {
  return Visitor->isStructLatticeConstant(F, STy);
}

Constant *SCCPSolver::getConstant(const ValueLatticeElement &LV) const {
  return Visitor->getConstant(LV);
}

SmallPtrSetImpl<Function *> &SCCPSolver::getArgumentTrackedFunctions() {
  return Visitor->getArgumentTrackedFunctions();
}

void SCCPSolver::markArgInFuncSpecialization(Function *F, Argument *A,
                                             Constant *C) {
  Visitor->markArgInFuncSpecialization(F, A, C);
}

void SCCPSolver::markFunctionUnreachable(Function *F) {
  Visitor->markFunctionUnreachable(F);
}

void SCCPSolver::visit(Instruction *I) { Visitor->visit(I); }

void SCCPSolver::visitCall(CallInst &I) { Visitor->visitCall(I); }
