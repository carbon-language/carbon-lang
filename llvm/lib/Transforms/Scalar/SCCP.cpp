//===- SCCP.cpp - Sparse Conditional Constant Propagation -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements sparse conditional constant propagation and merging:
//
// Specifically, this:
//   * Assumes values are constant unless proven otherwise
//   * Assumes BasicBlocks are dead unless proven otherwise
//   * Proves values to be constant, and replaces them with constants
//   * Proves conditional branches to be unconditional
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "sccp"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/InstVisitor.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <map>
using namespace llvm;

STATISTIC(NumInstRemoved, "Number of instructions removed");
STATISTIC(NumDeadBlocks , "Number of basic blocks unreachable");

STATISTIC(IPNumInstRemoved, "Number of instructions removed by IPSCCP");
STATISTIC(IPNumArgsElimed ,"Number of arguments constant propagated by IPSCCP");
STATISTIC(IPNumGlobalConst, "Number of globals found to be constant by IPSCCP");

namespace {
/// LatticeVal class - This class represents the different lattice values that
/// an LLVM value may occupy.  It is a simple class with value semantics.
///
class LatticeVal {
  enum LatticeValueTy {
    /// undefined - This LLVM Value has no known value yet.
    undefined,
    
    /// constant - This LLVM Value has a specific constant value.
    constant,

    /// forcedconstant - This LLVM Value was thought to be undef until
    /// ResolvedUndefsIn.  This is treated just like 'constant', but if merged
    /// with another (different) constant, it goes to overdefined, instead of
    /// asserting.
    forcedconstant,
    
    /// overdefined - This instruction is not known to be constant, and we know
    /// it has a value.
    overdefined
  };

  /// Val: This stores the current lattice value along with the Constant* for
  /// the constant if this is a 'constant' or 'forcedconstant' value.
  PointerIntPair<Constant *, 2, LatticeValueTy> Val;
  
  LatticeValueTy getLatticeValue() const {
    return Val.getInt();
  }
  
public:
  LatticeVal() : Val(0, undefined) {}
  
  bool isUndefined() const { return getLatticeValue() == undefined; }
  bool isConstant() const {
    return getLatticeValue() == constant || getLatticeValue() == forcedconstant;
  }
  bool isOverdefined() const { return getLatticeValue() == overdefined; }
  
  Constant *getConstant() const {
    assert(isConstant() && "Cannot get the constant of a non-constant!");
    return Val.getPointer();
  }
  
  /// markOverdefined - Return true if this is a change in status.
  bool markOverdefined() {
    if (isOverdefined())
      return false;
    
    Val.setInt(overdefined);
    return true;
  }

  /// markConstant - Return true if this is a change in status.
  bool markConstant(Constant *V) {
    if (isConstant()) {
      assert(getConstant() == V && "Marking constant with different value");
      return false;
    }
    
    if (isUndefined()) {
      Val.setInt(constant);
      assert(V && "Marking constant with NULL");
      Val.setPointer(V);
    } else {
      assert(getLatticeValue() == forcedconstant && 
             "Cannot move from overdefined to constant!");
      // Stay at forcedconstant if the constant is the same.
      if (V == getConstant()) return false;
      
      // Otherwise, we go to overdefined.  Assumptions made based on the
      // forced value are possibly wrong.  Assuming this is another constant
      // could expose a contradiction.
      Val.setInt(overdefined);
    }
    return true;
  }

  /// getConstantInt - If this is a constant with a ConstantInt value, return it
  /// otherwise return null.
  ConstantInt *getConstantInt() const {
    if (isConstant())
      return dyn_cast<ConstantInt>(getConstant());
    return 0;
  }
  
  void markForcedConstant(Constant *V) {
    assert(isUndefined() && "Can't force a defined value!");
    Val.setInt(forcedconstant);
    Val.setPointer(V);
  }
};
} // end anonymous namespace.


namespace {

//===----------------------------------------------------------------------===//
//
/// SCCPSolver - This class is a general purpose solver for Sparse Conditional
/// Constant Propagation.
///
class SCCPSolver : public InstVisitor<SCCPSolver> {
  const TargetData *TD;
  SmallPtrSet<BasicBlock*, 8> BBExecutable;// The BBs that are executable.
  DenseMap<Value*, LatticeVal> ValueState;  // The state each value is in.

  /// GlobalValue - If we are tracking any values for the contents of a global
  /// variable, we keep a mapping from the constant accessor to the element of
  /// the global, to the currently known value.  If the value becomes
  /// overdefined, it's entry is simply removed from this map.
  DenseMap<GlobalVariable*, LatticeVal> TrackedGlobals;

  /// TrackedRetVals - If we are tracking arguments into and the return
  /// value out of a function, it will have an entry in this map, indicating
  /// what the known return value for the function is.
  DenseMap<Function*, LatticeVal> TrackedRetVals;

  /// TrackedMultipleRetVals - Same as TrackedRetVals, but used for functions
  /// that return multiple values.
  DenseMap<std::pair<Function*, unsigned>, LatticeVal> TrackedMultipleRetVals;

  /// The reason for two worklists is that overdefined is the lowest state
  /// on the lattice, and moving things to overdefined as fast as possible
  /// makes SCCP converge much faster.
  ///
  /// By having a separate worklist, we accomplish this because everything
  /// possibly overdefined will become overdefined at the soonest possible
  /// point.
  SmallVector<Value*, 64> OverdefinedInstWorkList;
  SmallVector<Value*, 64> InstWorkList;


  SmallVector<BasicBlock*, 64>  BBWorkList;  // The BasicBlock work list

  /// UsersOfOverdefinedPHIs - Keep track of any users of PHI nodes that are not
  /// overdefined, despite the fact that the PHI node is overdefined.
  std::multimap<PHINode*, Instruction*> UsersOfOverdefinedPHIs;

  /// KnownFeasibleEdges - Entries in this set are edges which have already had
  /// PHI nodes retriggered.
  typedef std::pair<BasicBlock*, BasicBlock*> Edge;
  DenseSet<Edge> KnownFeasibleEdges;
public:
  SCCPSolver(const TargetData *td) : TD(td) {}

  /// MarkBlockExecutable - This method can be used by clients to mark all of
  /// the blocks that are known to be intrinsically live in the processed unit.
  ///
  /// This returns true if the block was not considered live before.
  bool MarkBlockExecutable(BasicBlock *BB) {
    if (!BBExecutable.insert(BB)) return false;
    DEBUG(errs() << "Marking Block Executable: " << BB->getName() << "\n");
    BBWorkList.push_back(BB);  // Add the block to the work list!
    return true;
  }

  /// TrackValueOfGlobalVariable - Clients can use this method to
  /// inform the SCCPSolver that it should track loads and stores to the
  /// specified global variable if it can.  This is only legal to call if
  /// performing Interprocedural SCCP.
  void TrackValueOfGlobalVariable(GlobalVariable *GV) {
    const Type *ElTy = GV->getType()->getElementType();
    if (ElTy->isFirstClassType()) {
      LatticeVal &IV = TrackedGlobals[GV];
      if (!isa<UndefValue>(GV->getInitializer()))
        IV.markConstant(GV->getInitializer());
    }
  }

  /// AddTrackedFunction - If the SCCP solver is supposed to track calls into
  /// and out of the specified function (which cannot have its address taken),
  /// this method must be called.
  void AddTrackedFunction(Function *F) {
    assert(F->hasLocalLinkage() && "Can only track internal functions!");
    // Add an entry, F -> undef.
    if (const StructType *STy = dyn_cast<StructType>(F->getReturnType())) {
      for (unsigned i = 0, e = STy->getNumElements(); i != e; ++i)
        TrackedMultipleRetVals.insert(std::make_pair(std::make_pair(F, i),
                                                     LatticeVal()));
    } else
      TrackedRetVals.insert(std::make_pair(F, LatticeVal()));
  }

  /// Solve - Solve for constants and executable blocks.
  ///
  void Solve();

  /// ResolvedUndefsIn - While solving the dataflow for a function, we assume
  /// that branches on undef values cannot reach any of their successors.
  /// However, this is not a safe assumption.  After we solve dataflow, this
  /// method should be use to handle this.  If this returns true, the solver
  /// should be rerun.
  bool ResolvedUndefsIn(Function &F);

  bool isBlockExecutable(BasicBlock *BB) const {
    return BBExecutable.count(BB);
  }

  LatticeVal getLatticeValueFor(Value *V) const {
    DenseMap<Value*, LatticeVal>::const_iterator I = ValueState.find(V);
    assert(I != ValueState.end() && "V is not in valuemap!");
    return I->second;
  }

  /// getTrackedRetVals - Get the inferred return value map.
  ///
  const DenseMap<Function*, LatticeVal> &getTrackedRetVals() {
    return TrackedRetVals;
  }

  /// getTrackedGlobals - Get and return the set of inferred initializers for
  /// global variables.
  const DenseMap<GlobalVariable*, LatticeVal> &getTrackedGlobals() {
    return TrackedGlobals;
  }

  void markOverdefined(Value *V) {
    markOverdefined(ValueState[V], V);
  }

private:
  // markConstant - Make a value be marked as "constant".  If the value
  // is not already a constant, add it to the instruction work list so that
  // the users of the instruction are updated later.
  //
  void markConstant(LatticeVal &IV, Value *V, Constant *C) {
    if (!IV.markConstant(C)) return;
    DEBUG(errs() << "markConstant: " << *C << ": " << *V << '\n');
    InstWorkList.push_back(V);
  }
  
  void markConstant(Value *V, Constant *C) {
    markConstant(ValueState[V], V, C);
  }

  void markForcedConstant(Value *V, Constant *C) {
    ValueState[V].markForcedConstant(C);
    DEBUG(errs() << "markForcedConstant: " << *C << ": " << *V << '\n');
    InstWorkList.push_back(V);
  }
  
  
  // markOverdefined - Make a value be marked as "overdefined". If the
  // value is not already overdefined, add it to the overdefined instruction
  // work list so that the users of the instruction are updated later.
  void markOverdefined(LatticeVal &IV, Value *V) {
    if (!IV.markOverdefined()) return;
    
    DEBUG(errs() << "markOverdefined: ";
          if (Function *F = dyn_cast<Function>(V))
            errs() << "Function '" << F->getName() << "'\n";
          else
            errs() << *V << '\n');
    // Only instructions go on the work list
    OverdefinedInstWorkList.push_back(V);
  }

  void mergeInValue(LatticeVal &IV, Value *V, LatticeVal MergeWithV) {
    if (IV.isOverdefined() || MergeWithV.isUndefined())
      return;  // Noop.
    if (MergeWithV.isOverdefined())
      markOverdefined(IV, V);
    else if (IV.isUndefined())
      markConstant(IV, V, MergeWithV.getConstant());
    else if (IV.getConstant() != MergeWithV.getConstant())
      markOverdefined(IV, V);
  }
  
  void mergeInValue(Value *V, LatticeVal MergeWithV) {
    mergeInValue(ValueState[V], V, MergeWithV);
  }


  /// getValueState - Return the LatticeVal object that corresponds to the
  /// value.  This function handles the case when the value hasn't been seen yet
  /// by properly seeding constants etc.
  LatticeVal &getValueState(Value *V) {
    DenseMap<Value*, LatticeVal>::iterator I = ValueState.find(V);
    if (I != ValueState.end()) return I->second;  // Common case, in the map

    LatticeVal &LV = ValueState[V];

    if (Constant *C = dyn_cast<Constant>(V)) {
      // Undef values remain undefined.
      if (!isa<UndefValue>(V))
        LV.markConstant(C);          // Constants are constant
    }
    
    // All others are underdefined by default.
    return LV;
  }

  /// markEdgeExecutable - Mark a basic block as executable, adding it to the BB
  /// work list if it is not already executable.
  void markEdgeExecutable(BasicBlock *Source, BasicBlock *Dest) {
    if (!KnownFeasibleEdges.insert(Edge(Source, Dest)).second)
      return;  // This edge is already known to be executable!

    if (!MarkBlockExecutable(Dest)) {
      // If the destination is already executable, we just made an *edge*
      // feasible that wasn't before.  Revisit the PHI nodes in the block
      // because they have potentially new operands.
      DEBUG(errs() << "Marking Edge Executable: " << Source->getName()
            << " -> " << Dest->getName() << "\n");

      PHINode *PN;
      for (BasicBlock::iterator I = Dest->begin();
           (PN = dyn_cast<PHINode>(I)); ++I)
        visitPHINode(*PN);
    }
  }

  // getFeasibleSuccessors - Return a vector of booleans to indicate which
  // successors are reachable from a given terminator instruction.
  //
  void getFeasibleSuccessors(TerminatorInst &TI, SmallVector<bool, 16> &Succs);

  // isEdgeFeasible - Return true if the control flow edge from the 'From' basic
  // block to the 'To' basic block is currently feasible.
  //
  bool isEdgeFeasible(BasicBlock *From, BasicBlock *To);

  // OperandChangedState - This method is invoked on all of the users of an
  // instruction that was just changed state somehow.  Based on this
  // information, we need to update the specified user of this instruction.
  //
  void OperandChangedState(User *U) {
    // Only instructions use other variable values!
    Instruction &I = cast<Instruction>(*U);
    if (BBExecutable.count(I.getParent()))   // Inst is executable?
      visit(I);
  }
  
  /// RemoveFromOverdefinedPHIs - If I has any entries in the
  /// UsersOfOverdefinedPHIs map for PN, remove them now.
  void RemoveFromOverdefinedPHIs(Instruction *I, PHINode *PN) {
    if (UsersOfOverdefinedPHIs.empty()) return;
    std::multimap<PHINode*, Instruction*>::iterator It, E;
    tie(It, E) = UsersOfOverdefinedPHIs.equal_range(PN);
    while (It != E) {
      if (It->second == I)
        UsersOfOverdefinedPHIs.erase(It++);
      else
        ++It;
    }
  }

private:
  friend class InstVisitor<SCCPSolver>;

  // visit implementations - Something changed in this instruction.  Either an
  // operand made a transition, or the instruction is newly executable.  Change
  // the value type of I to reflect these changes if appropriate.
  void visitPHINode(PHINode &I);

  // Terminators
  void visitReturnInst(ReturnInst &I);
  void visitTerminatorInst(TerminatorInst &TI);

  void visitCastInst(CastInst &I);
  void visitSelectInst(SelectInst &I);
  void visitBinaryOperator(Instruction &I);
  void visitCmpInst(CmpInst &I);
  void visitExtractElementInst(ExtractElementInst &I);
  void visitInsertElementInst(InsertElementInst &I);
  void visitShuffleVectorInst(ShuffleVectorInst &I);
  void visitExtractValueInst(ExtractValueInst &EVI);
  void visitInsertValueInst(InsertValueInst &IVI);

  // Instructions that cannot be folded away.
  void visitStoreInst     (StoreInst &I);
  void visitLoadInst      (LoadInst &I);
  void visitGetElementPtrInst(GetElementPtrInst &I);
  void visitCallInst      (CallInst &I) {
    if (isFreeCall(&I))
      return;
    visitCallSite(CallSite::get(&I));
  }
  void visitInvokeInst    (InvokeInst &II) {
    visitCallSite(CallSite::get(&II));
    visitTerminatorInst(II);
  }
  void visitCallSite      (CallSite CS);
  void visitUnwindInst    (TerminatorInst &I) { /*returns void*/ }
  void visitUnreachableInst(TerminatorInst &I) { /*returns void*/ }
  void visitAllocaInst    (Instruction &I) { markOverdefined(&I); }
  void visitVANextInst    (Instruction &I) { markOverdefined(&I); }
  void visitVAArgInst     (Instruction &I) { markOverdefined(&I); }

  void visitInstruction(Instruction &I) {
    // If a new instruction is added to LLVM that we don't handle.
    errs() << "SCCP: Don't know how to handle: " << I;
    markOverdefined(&I);   // Just in case
  }
};

} // end anonymous namespace


// getFeasibleSuccessors - Return a vector of booleans to indicate which
// successors are reachable from a given terminator instruction.
//
void SCCPSolver::getFeasibleSuccessors(TerminatorInst &TI,
                                       SmallVector<bool, 16> &Succs) {
  Succs.resize(TI.getNumSuccessors());
  if (BranchInst *BI = dyn_cast<BranchInst>(&TI)) {
    if (BI->isUnconditional()) {
      Succs[0] = true;
      return;
    }
    
    LatticeVal BCValue = getValueState(BI->getCondition());
    ConstantInt *CI = BCValue.getConstantInt();
    if (CI == 0) {
      // Overdefined condition variables, and branches on unfoldable constant
      // conditions, mean the branch could go either way.
      if (!BCValue.isUndefined())
        Succs[0] = Succs[1] = true;
      return;
    }
    
    // Constant condition variables mean the branch can only go a single way.
    Succs[CI->isZero()] = true;
    return;
  }
  
  if (isa<InvokeInst>(TI)) {
    // Invoke instructions successors are always executable.
    Succs[0] = Succs[1] = true;
    return;
  }
  
  if (SwitchInst *SI = dyn_cast<SwitchInst>(&TI)) {
    LatticeVal SCValue = getValueState(SI->getCondition());
    ConstantInt *CI = SCValue.getConstantInt();
    
    if (CI == 0) {   // Overdefined or undefined condition?
      // All destinations are executable!
      if (!SCValue.isUndefined())
        Succs.assign(TI.getNumSuccessors(), true);
      return;
    }
      
    Succs[SI->findCaseValue(CI)] = true;
    return;
  }
  
  // TODO: This could be improved if the operand is a [cast of a] BlockAddress.
  if (isa<IndirectBrInst>(&TI)) {
    // Just mark all destinations executable!
    Succs.assign(TI.getNumSuccessors(), true);
    return;
  }
  
#ifndef NDEBUG
  errs() << "Unknown terminator instruction: " << TI << '\n';
#endif
  llvm_unreachable("SCCP: Don't know how to handle this terminator!");
}


// isEdgeFeasible - Return true if the control flow edge from the 'From' basic
// block to the 'To' basic block is currently feasible.
//
bool SCCPSolver::isEdgeFeasible(BasicBlock *From, BasicBlock *To) {
  assert(BBExecutable.count(To) && "Dest should always be alive!");

  // Make sure the source basic block is executable!!
  if (!BBExecutable.count(From)) return false;

  // Check to make sure this edge itself is actually feasible now.
  TerminatorInst *TI = From->getTerminator();
  if (BranchInst *BI = dyn_cast<BranchInst>(TI)) {
    if (BI->isUnconditional())
      return true;
    
    LatticeVal BCValue = getValueState(BI->getCondition());

    // Overdefined condition variables mean the branch could go either way,
    // undef conditions mean that neither edge is feasible yet.
    ConstantInt *CI = BCValue.getConstantInt();
    if (CI == 0)
      return !BCValue.isUndefined();
    
    // Constant condition variables mean the branch can only go a single way.
    return BI->getSuccessor(CI->isZero()) == To;
  }
  
  // Invoke instructions successors are always executable.
  if (isa<InvokeInst>(TI))
    return true;
  
  if (SwitchInst *SI = dyn_cast<SwitchInst>(TI)) {
    LatticeVal SCValue = getValueState(SI->getCondition());
    ConstantInt *CI = SCValue.getConstantInt();
    
    if (CI == 0)
      return !SCValue.isUndefined();

    // Make sure to skip the "default value" which isn't a value
    for (unsigned i = 1, E = SI->getNumSuccessors(); i != E; ++i)
      if (SI->getSuccessorValue(i) == CI) // Found the taken branch.
        return SI->getSuccessor(i) == To;

    // If the constant value is not equal to any of the branches, we must
    // execute default branch.
    return SI->getDefaultDest() == To;
  }
  
  // Just mark all destinations executable!
  // TODO: This could be improved if the operand is a [cast of a] BlockAddress.
  if (isa<IndirectBrInst>(&TI))
    return true;
  
#ifndef NDEBUG
  errs() << "Unknown terminator instruction: " << *TI << '\n';
#endif
  llvm_unreachable(0);
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
//
void SCCPSolver::visitPHINode(PHINode &PN) {
  if (getValueState(&PN).isOverdefined()) {
    // There may be instructions using this PHI node that are not overdefined
    // themselves.  If so, make sure that they know that the PHI node operand
    // changed.
    std::multimap<PHINode*, Instruction*>::iterator I, E;
    tie(I, E) = UsersOfOverdefinedPHIs.equal_range(&PN);
    if (I == E)
      return;
    
    SmallVector<Instruction*, 16> Users;
    for (; I != E; ++I)
      Users.push_back(I->second);
    while (!Users.empty())
      visit(Users.pop_back_val());
    return;  // Quick exit
  }

  // Super-extra-high-degree PHI nodes are unlikely to ever be marked constant,
  // and slow us down a lot.  Just mark them overdefined.
  if (PN.getNumIncomingValues() > 64)
    return markOverdefined(&PN);

  // Look at all of the executable operands of the PHI node.  If any of them
  // are overdefined, the PHI becomes overdefined as well.  If they are all
  // constant, and they agree with each other, the PHI becomes the identical
  // constant.  If they are constant and don't agree, the PHI is overdefined.
  // If there are no executable operands, the PHI remains undefined.
  //
  Constant *OperandVal = 0;
  for (unsigned i = 0, e = PN.getNumIncomingValues(); i != e; ++i) {
    LatticeVal IV = getValueState(PN.getIncomingValue(i));
    if (IV.isUndefined()) continue;  // Doesn't influence PHI node.

    if (!isEdgeFeasible(PN.getIncomingBlock(i), PN.getParent()))
      continue;
    
    if (IV.isOverdefined())    // PHI node becomes overdefined!
      return markOverdefined(&PN);

    if (OperandVal == 0) {   // Grab the first value.
      OperandVal = IV.getConstant();
      continue;
    }
    
    // There is already a reachable operand.  If we conflict with it,
    // then the PHI node becomes overdefined.  If we agree with it, we
    // can continue on.
    
    // Check to see if there are two different constants merging, if so, the PHI
    // node is overdefined.
    if (IV.getConstant() != OperandVal)
      return markOverdefined(&PN);
  }

  // If we exited the loop, this means that the PHI node only has constant
  // arguments that agree with each other(and OperandVal is the constant) or
  // OperandVal is null because there are no defined incoming arguments.  If
  // this is the case, the PHI remains undefined.
  //
  if (OperandVal)
    markConstant(&PN, OperandVal);      // Acquire operand value
}

void SCCPSolver::visitReturnInst(ReturnInst &I) {
  if (I.getNumOperands() == 0) return;  // ret void

  Function *F = I.getParent()->getParent();
  // If we are tracking the return value of this function, merge it in.
  if (!F->hasLocalLinkage())
    return;

  if (!TrackedRetVals.empty()) {
    DenseMap<Function*, LatticeVal>::iterator TFRVI =
      TrackedRetVals.find(F);
    if (TFRVI != TrackedRetVals.end() &&
        !TFRVI->second.isOverdefined()) {
      mergeInValue(TFRVI->second, F, getValueState(I.getOperand(0)));
      return;
    }
  }
  
  // Handle functions that return multiple values.
  if (!TrackedMultipleRetVals.empty() &&
      isa<StructType>(I.getOperand(0)->getType())) {
    for (unsigned i = 0, e = I.getOperand(0)->getType()->getNumContainedTypes();
         i != e; ++i) {
      DenseMap<std::pair<Function*, unsigned>, LatticeVal>::iterator
        It = TrackedMultipleRetVals.find(std::make_pair(F, i));
      if (It == TrackedMultipleRetVals.end()) break;
      if (Value *Val = FindInsertedValue(I.getOperand(0), i, I.getContext()))
        mergeInValue(It->second, F, getValueState(Val));
    }
  }
}

void SCCPSolver::visitTerminatorInst(TerminatorInst &TI) {
  SmallVector<bool, 16> SuccFeasible;
  getFeasibleSuccessors(TI, SuccFeasible);

  BasicBlock *BB = TI.getParent();

  // Mark all feasible successors executable.
  for (unsigned i = 0, e = SuccFeasible.size(); i != e; ++i)
    if (SuccFeasible[i])
      markEdgeExecutable(BB, TI.getSuccessor(i));
}

void SCCPSolver::visitCastInst(CastInst &I) {
  LatticeVal OpSt = getValueState(I.getOperand(0));
  if (OpSt.isOverdefined())          // Inherit overdefinedness of operand
    markOverdefined(&I);
  else if (OpSt.isConstant())        // Propagate constant value
    markConstant(&I, ConstantExpr::getCast(I.getOpcode(), 
                                           OpSt.getConstant(), I.getType()));
}

void SCCPSolver::visitExtractValueInst(ExtractValueInst &EVI) {
  Value *Aggr = EVI.getAggregateOperand();

  // If the operand to the extractvalue is an undef, the result is undef.
  if (isa<UndefValue>(Aggr))
    return;

  // Currently only handle single-index extractvalues.
  if (EVI.getNumIndices() != 1)
    return markOverdefined(&EVI);
  
  Function *F = 0;
  if (CallInst *CI = dyn_cast<CallInst>(Aggr))
    F = CI->getCalledFunction();
  else if (InvokeInst *II = dyn_cast<InvokeInst>(Aggr))
    F = II->getCalledFunction();

  // TODO: If IPSCCP resolves the callee of this function, we could propagate a
  // result back!
  if (F == 0 || TrackedMultipleRetVals.empty())
    return markOverdefined(&EVI);
  
  // See if we are tracking the result of the callee.  If not tracking this
  // function (for example, it is a declaration) just move to overdefined.
  if (!TrackedMultipleRetVals.count(std::make_pair(F, *EVI.idx_begin())))
    return markOverdefined(&EVI);
  
  // Otherwise, the value will be merged in here as a result of CallSite
  // handling.
}

void SCCPSolver::visitInsertValueInst(InsertValueInst &IVI) {
  Value *Aggr = IVI.getAggregateOperand();
  Value *Val = IVI.getInsertedValueOperand();

  // If the operands to the insertvalue are undef, the result is undef.
  if (isa<UndefValue>(Aggr) && isa<UndefValue>(Val))
    return;

  // Currently only handle single-index insertvalues.
  if (IVI.getNumIndices() != 1)
    return markOverdefined(&IVI);

  // Currently only handle insertvalue instructions that are in a single-use
  // chain that builds up a return value.
  for (const InsertValueInst *TmpIVI = &IVI; ; ) {
    if (!TmpIVI->hasOneUse())
      return markOverdefined(&IVI);

    const Value *V = *TmpIVI->use_begin();
    if (isa<ReturnInst>(V))
      break;
    TmpIVI = dyn_cast<InsertValueInst>(V);
    if (!TmpIVI)
      return markOverdefined(&IVI);
  }
  
  // See if we are tracking the result of the callee.
  Function *F = IVI.getParent()->getParent();
  DenseMap<std::pair<Function*, unsigned>, LatticeVal>::iterator
    It = TrackedMultipleRetVals.find(std::make_pair(F, *IVI.idx_begin()));

  // Merge in the inserted member value.
  if (It != TrackedMultipleRetVals.end())
    mergeInValue(It->second, F, getValueState(Val));

  // Mark the aggregate result of the IVI overdefined; any tracking that we do
  // will be done on the individual member values.
  markOverdefined(&IVI);
}

void SCCPSolver::visitSelectInst(SelectInst &I) {
  LatticeVal CondValue = getValueState(I.getCondition());
  if (CondValue.isUndefined())
    return;
  
  if (ConstantInt *CondCB = CondValue.getConstantInt()) {
    Value *OpVal = CondCB->isZero() ? I.getFalseValue() : I.getTrueValue();
    mergeInValue(&I, getValueState(OpVal));
    return;
  }
  
  // Otherwise, the condition is overdefined or a constant we can't evaluate.
  // See if we can produce something better than overdefined based on the T/F
  // value.
  LatticeVal TVal = getValueState(I.getTrueValue());
  LatticeVal FVal = getValueState(I.getFalseValue());
  
  // select ?, C, C -> C.
  if (TVal.isConstant() && FVal.isConstant() && 
      TVal.getConstant() == FVal.getConstant())
    return markConstant(&I, FVal.getConstant());

  if (TVal.isUndefined())   // select ?, undef, X -> X.
    return mergeInValue(&I, FVal);
  if (FVal.isUndefined())   // select ?, X, undef -> X.
    return mergeInValue(&I, TVal);
  markOverdefined(&I);
}

// Handle Binary Operators.
void SCCPSolver::visitBinaryOperator(Instruction &I) {
  LatticeVal V1State = getValueState(I.getOperand(0));
  LatticeVal V2State = getValueState(I.getOperand(1));
  
  LatticeVal &IV = ValueState[&I];
  if (IV.isOverdefined()) return;

  if (V1State.isConstant() && V2State.isConstant())
    return markConstant(IV, &I,
                        ConstantExpr::get(I.getOpcode(), V1State.getConstant(),
                                          V2State.getConstant()));
  
  // If something is undef, wait for it to resolve.
  if (!V1State.isOverdefined() && !V2State.isOverdefined())
    return;
  
  // Otherwise, one of our operands is overdefined.  Try to produce something
  // better than overdefined with some tricks.
  
  // If this is an AND or OR with 0 or -1, it doesn't matter that the other
  // operand is overdefined.
  if (I.getOpcode() == Instruction::And || I.getOpcode() == Instruction::Or) {
    LatticeVal *NonOverdefVal = 0;
    if (!V1State.isOverdefined())
      NonOverdefVal = &V1State;
    else if (!V2State.isOverdefined())
      NonOverdefVal = &V2State;

    if (NonOverdefVal) {
      if (NonOverdefVal->isUndefined()) {
        // Could annihilate value.
        if (I.getOpcode() == Instruction::And)
          markConstant(IV, &I, Constant::getNullValue(I.getType()));
        else if (const VectorType *PT = dyn_cast<VectorType>(I.getType()))
          markConstant(IV, &I, Constant::getAllOnesValue(PT));
        else
          markConstant(IV, &I,
                       Constant::getAllOnesValue(I.getType()));
        return;
      }
      
      if (I.getOpcode() == Instruction::And) {
        // X and 0 = 0
        if (NonOverdefVal->getConstant()->isNullValue())
          return markConstant(IV, &I, NonOverdefVal->getConstant());
      } else {
        if (ConstantInt *CI = NonOverdefVal->getConstantInt())
          if (CI->isAllOnesValue())     // X or -1 = -1
            return markConstant(IV, &I, NonOverdefVal->getConstant());
      }
    }
  }


  // If both operands are PHI nodes, it is possible that this instruction has
  // a constant value, despite the fact that the PHI node doesn't.  Check for
  // this condition now.
  if (PHINode *PN1 = dyn_cast<PHINode>(I.getOperand(0)))
    if (PHINode *PN2 = dyn_cast<PHINode>(I.getOperand(1)))
      if (PN1->getParent() == PN2->getParent()) {
        // Since the two PHI nodes are in the same basic block, they must have
        // entries for the same predecessors.  Walk the predecessor list, and
        // if all of the incoming values are constants, and the result of
        // evaluating this expression with all incoming value pairs is the
        // same, then this expression is a constant even though the PHI node
        // is not a constant!
        LatticeVal Result;
        for (unsigned i = 0, e = PN1->getNumIncomingValues(); i != e; ++i) {
          LatticeVal In1 = getValueState(PN1->getIncomingValue(i));
          BasicBlock *InBlock = PN1->getIncomingBlock(i);
          LatticeVal In2 =getValueState(PN2->getIncomingValueForBlock(InBlock));

          if (In1.isOverdefined() || In2.isOverdefined()) {
            Result.markOverdefined();
            break;  // Cannot fold this operation over the PHI nodes!
          }
          
          if (In1.isConstant() && In2.isConstant()) {
            Constant *V = ConstantExpr::get(I.getOpcode(), In1.getConstant(),
                                            In2.getConstant());
            if (Result.isUndefined())
              Result.markConstant(V);
            else if (Result.isConstant() && Result.getConstant() != V) {
              Result.markOverdefined();
              break;
            }
          }
        }

        // If we found a constant value here, then we know the instruction is
        // constant despite the fact that the PHI nodes are overdefined.
        if (Result.isConstant()) {
          markConstant(IV, &I, Result.getConstant());
          // Remember that this instruction is virtually using the PHI node
          // operands.
          UsersOfOverdefinedPHIs.insert(std::make_pair(PN1, &I));
          UsersOfOverdefinedPHIs.insert(std::make_pair(PN2, &I));
          return;
        }
        
        if (Result.isUndefined())
          return;

        // Okay, this really is overdefined now.  Since we might have
        // speculatively thought that this was not overdefined before, and
        // added ourselves to the UsersOfOverdefinedPHIs list for the PHIs,
        // make sure to clean out any entries that we put there, for
        // efficiency.
        RemoveFromOverdefinedPHIs(&I, PN1);
        RemoveFromOverdefinedPHIs(&I, PN2);
      }

  markOverdefined(&I);
}

// Handle ICmpInst instruction.
void SCCPSolver::visitCmpInst(CmpInst &I) {
  LatticeVal V1State = getValueState(I.getOperand(0));
  LatticeVal V2State = getValueState(I.getOperand(1));

  LatticeVal &IV = ValueState[&I];
  if (IV.isOverdefined()) return;

  if (V1State.isConstant() && V2State.isConstant())
    return markConstant(IV, &I, ConstantExpr::getCompare(I.getPredicate(), 
                                                         V1State.getConstant(), 
                                                        V2State.getConstant()));
  
  // If operands are still undefined, wait for it to resolve.
  if (!V1State.isOverdefined() && !V2State.isOverdefined())
    return;
  
  // If something is overdefined, use some tricks to avoid ending up and over
  // defined if we can.
  
  // If both operands are PHI nodes, it is possible that this instruction has
  // a constant value, despite the fact that the PHI node doesn't.  Check for
  // this condition now.
  if (PHINode *PN1 = dyn_cast<PHINode>(I.getOperand(0)))
    if (PHINode *PN2 = dyn_cast<PHINode>(I.getOperand(1)))
      if (PN1->getParent() == PN2->getParent()) {
        // Since the two PHI nodes are in the same basic block, they must have
        // entries for the same predecessors.  Walk the predecessor list, and
        // if all of the incoming values are constants, and the result of
        // evaluating this expression with all incoming value pairs is the
        // same, then this expression is a constant even though the PHI node
        // is not a constant!
        LatticeVal Result;
        for (unsigned i = 0, e = PN1->getNumIncomingValues(); i != e; ++i) {
          LatticeVal In1 = getValueState(PN1->getIncomingValue(i));
          BasicBlock *InBlock = PN1->getIncomingBlock(i);
          LatticeVal In2 =getValueState(PN2->getIncomingValueForBlock(InBlock));

          if (In1.isOverdefined() || In2.isOverdefined()) {
            Result.markOverdefined();
            break;  // Cannot fold this operation over the PHI nodes!
          }
          
          if (In1.isConstant() && In2.isConstant()) {
            Constant *V = ConstantExpr::getCompare(I.getPredicate(), 
                                                   In1.getConstant(), 
                                                   In2.getConstant());
            if (Result.isUndefined())
              Result.markConstant(V);
            else if (Result.isConstant() && Result.getConstant() != V) {
              Result.markOverdefined();
              break;
            }
          }
        }

        // If we found a constant value here, then we know the instruction is
        // constant despite the fact that the PHI nodes are overdefined.
        if (Result.isConstant()) {
          markConstant(&I, Result.getConstant());
          // Remember that this instruction is virtually using the PHI node
          // operands.
          UsersOfOverdefinedPHIs.insert(std::make_pair(PN1, &I));
          UsersOfOverdefinedPHIs.insert(std::make_pair(PN2, &I));
          return;
        }
        
        if (Result.isUndefined())
          return;

        // Okay, this really is overdefined now.  Since we might have
        // speculatively thought that this was not overdefined before, and
        // added ourselves to the UsersOfOverdefinedPHIs list for the PHIs,
        // make sure to clean out any entries that we put there, for
        // efficiency.
        RemoveFromOverdefinedPHIs(&I, PN1);
        RemoveFromOverdefinedPHIs(&I, PN2);
      }

  markOverdefined(&I);
}

void SCCPSolver::visitExtractElementInst(ExtractElementInst &I) {
  // FIXME : SCCP does not handle vectors properly.
  return markOverdefined(&I);

#if 0
  LatticeVal &ValState = getValueState(I.getOperand(0));
  LatticeVal &IdxState = getValueState(I.getOperand(1));

  if (ValState.isOverdefined() || IdxState.isOverdefined())
    markOverdefined(&I);
  else if(ValState.isConstant() && IdxState.isConstant())
    markConstant(&I, ConstantExpr::getExtractElement(ValState.getConstant(),
                                                     IdxState.getConstant()));
#endif
}

void SCCPSolver::visitInsertElementInst(InsertElementInst &I) {
  // FIXME : SCCP does not handle vectors properly.
  return markOverdefined(&I);
#if 0
  LatticeVal &ValState = getValueState(I.getOperand(0));
  LatticeVal &EltState = getValueState(I.getOperand(1));
  LatticeVal &IdxState = getValueState(I.getOperand(2));

  if (ValState.isOverdefined() || EltState.isOverdefined() ||
      IdxState.isOverdefined())
    markOverdefined(&I);
  else if(ValState.isConstant() && EltState.isConstant() &&
          IdxState.isConstant())
    markConstant(&I, ConstantExpr::getInsertElement(ValState.getConstant(),
                                                    EltState.getConstant(),
                                                    IdxState.getConstant()));
  else if (ValState.isUndefined() && EltState.isConstant() &&
           IdxState.isConstant()) 
    markConstant(&I,ConstantExpr::getInsertElement(UndefValue::get(I.getType()),
                                                   EltState.getConstant(),
                                                   IdxState.getConstant()));
#endif
}

void SCCPSolver::visitShuffleVectorInst(ShuffleVectorInst &I) {
  // FIXME : SCCP does not handle vectors properly.
  return markOverdefined(&I);
#if 0
  LatticeVal &V1State   = getValueState(I.getOperand(0));
  LatticeVal &V2State   = getValueState(I.getOperand(1));
  LatticeVal &MaskState = getValueState(I.getOperand(2));

  if (MaskState.isUndefined() ||
      (V1State.isUndefined() && V2State.isUndefined()))
    return;  // Undefined output if mask or both inputs undefined.
  
  if (V1State.isOverdefined() || V2State.isOverdefined() ||
      MaskState.isOverdefined()) {
    markOverdefined(&I);
  } else {
    // A mix of constant/undef inputs.
    Constant *V1 = V1State.isConstant() ? 
        V1State.getConstant() : UndefValue::get(I.getType());
    Constant *V2 = V2State.isConstant() ? 
        V2State.getConstant() : UndefValue::get(I.getType());
    Constant *Mask = MaskState.isConstant() ? 
      MaskState.getConstant() : UndefValue::get(I.getOperand(2)->getType());
    markConstant(&I, ConstantExpr::getShuffleVector(V1, V2, Mask));
  }
#endif
}

// Handle getelementptr instructions.  If all operands are constants then we
// can turn this into a getelementptr ConstantExpr.
//
void SCCPSolver::visitGetElementPtrInst(GetElementPtrInst &I) {
  LatticeVal &IV = ValueState[&I];
  if (IV.isOverdefined()) return;

  SmallVector<Constant*, 8> Operands;
  Operands.reserve(I.getNumOperands());

  for (unsigned i = 0, e = I.getNumOperands(); i != e; ++i) {
    LatticeVal State = getValueState(I.getOperand(i));
    if (State.isUndefined())
      return;  // Operands are not resolved yet.
    
    if (State.isOverdefined())
      return markOverdefined(IV, &I);

    assert(State.isConstant() && "Unknown state!");
    Operands.push_back(State.getConstant());
  }

  Constant *Ptr = Operands[0];
  markConstant(&I, ConstantExpr::getGetElementPtr(Ptr, &Operands[0]+1,
                                                  Operands.size()-1));
}

void SCCPSolver::visitStoreInst(StoreInst &SI) {
  if (TrackedGlobals.empty() || !isa<GlobalVariable>(SI.getOperand(1)))
    return;
  
  GlobalVariable *GV = cast<GlobalVariable>(SI.getOperand(1));
  DenseMap<GlobalVariable*, LatticeVal>::iterator I = TrackedGlobals.find(GV);
  if (I == TrackedGlobals.end() || I->second.isOverdefined()) return;

  // Get the value we are storing into the global, then merge it.
  mergeInValue(I->second, GV, getValueState(SI.getOperand(0)));
  if (I->second.isOverdefined())
    TrackedGlobals.erase(I);      // No need to keep tracking this!
}


// Handle load instructions.  If the operand is a constant pointer to a constant
// global, we can replace the load with the loaded constant value!
void SCCPSolver::visitLoadInst(LoadInst &I) {
  LatticeVal PtrVal = getValueState(I.getOperand(0));
  if (PtrVal.isUndefined()) return;   // The pointer is not resolved yet!
  
  LatticeVal &IV = ValueState[&I];
  if (IV.isOverdefined()) return;

  if (!PtrVal.isConstant() || I.isVolatile())
    return markOverdefined(IV, &I);
    
  Constant *Ptr = PtrVal.getConstant();

  // load null -> null
  if (isa<ConstantPointerNull>(Ptr) && I.getPointerAddressSpace() == 0)
    return markConstant(IV, &I, Constant::getNullValue(I.getType()));
  
  // Transform load (constant global) into the value loaded.
  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(Ptr)) {
    if (!TrackedGlobals.empty()) {
      // If we are tracking this global, merge in the known value for it.
      DenseMap<GlobalVariable*, LatticeVal>::iterator It =
        TrackedGlobals.find(GV);
      if (It != TrackedGlobals.end()) {
        mergeInValue(IV, &I, It->second);
        return;
      }
    }
  }

  // Transform load from a constant into a constant if possible.
  if (Constant *C = ConstantFoldLoadFromConstPtr(Ptr, TD))
    return markConstant(IV, &I, C);

  // Otherwise we cannot say for certain what value this load will produce.
  // Bail out.
  markOverdefined(IV, &I);
}

void SCCPSolver::visitCallSite(CallSite CS) {
  Function *F = CS.getCalledFunction();
  Instruction *I = CS.getInstruction();
  
  // The common case is that we aren't tracking the callee, either because we
  // are not doing interprocedural analysis or the callee is indirect, or is
  // external.  Handle these cases first.
  if (F == 0 || !F->hasLocalLinkage()) {
CallOverdefined:
    // Void return and not tracking callee, just bail.
    if (I->getType()->isVoidTy()) return;
    
    // Otherwise, if we have a single return value case, and if the function is
    // a declaration, maybe we can constant fold it.
    if (!isa<StructType>(I->getType()) && F && F->isDeclaration() && 
        canConstantFoldCallTo(F)) {
      
      SmallVector<Constant*, 8> Operands;
      for (CallSite::arg_iterator AI = CS.arg_begin(), E = CS.arg_end();
           AI != E; ++AI) {
        LatticeVal State = getValueState(*AI);
        
        if (State.isUndefined())
          return;  // Operands are not resolved yet.
        if (State.isOverdefined())
          return markOverdefined(I);
        assert(State.isConstant() && "Unknown state!");
        Operands.push_back(State.getConstant());
      }
     
      // If we can constant fold this, mark the result of the call as a
      // constant.
      if (Constant *C = ConstantFoldCall(F, Operands.data(), Operands.size()))
        return markConstant(I, C);
    }

    // Otherwise, we don't know anything about this call, mark it overdefined.
    return markOverdefined(I);
  }

  // If this is a single/zero retval case, see if we're tracking the function.
  DenseMap<Function*, LatticeVal>::iterator TFRVI = TrackedRetVals.find(F);
  if (TFRVI != TrackedRetVals.end()) {
    // If so, propagate the return value of the callee into this call result.
    mergeInValue(I, TFRVI->second);
  } else if (isa<StructType>(I->getType())) {
    // Check to see if we're tracking this callee, if not, handle it in the
    // common path above.
    DenseMap<std::pair<Function*, unsigned>, LatticeVal>::iterator
    TMRVI = TrackedMultipleRetVals.find(std::make_pair(F, 0));
    if (TMRVI == TrackedMultipleRetVals.end())
      goto CallOverdefined;

    // Need to mark as overdefined, otherwise it stays undefined which
    // creates extractvalue undef, <idx>
    markOverdefined(I);
    
    // If we are tracking this callee, propagate the return values of the call
    // into this call site.  We do this by walking all the uses. Single-index
    // ExtractValueInst uses can be tracked; anything more complicated is
    // currently handled conservatively.
    for (Value::use_iterator UI = I->use_begin(), E = I->use_end();
         UI != E; ++UI) {
      if (ExtractValueInst *EVI = dyn_cast<ExtractValueInst>(*UI)) {
        if (EVI->getNumIndices() == 1) {
          mergeInValue(EVI, 
                  TrackedMultipleRetVals[std::make_pair(F, *EVI->idx_begin())]);
          continue;
        }
      }
      // The aggregate value is used in a way not handled here. Assume nothing.
      markOverdefined(*UI);
    }
  } else {
    // Otherwise we're not tracking this callee, so handle it in the
    // common path above.
    goto CallOverdefined;
  }
   
  // Finally, if this is the first call to the function hit, mark its entry
  // block executable.
  MarkBlockExecutable(F->begin());
  
  // Propagate information from this call site into the callee.
  CallSite::arg_iterator CAI = CS.arg_begin();
  for (Function::arg_iterator AI = F->arg_begin(), E = F->arg_end();
       AI != E; ++AI, ++CAI) {
    if (AI->hasByValAttr() && !F->onlyReadsMemory()) {
      markOverdefined(AI);
      continue;
    }
    
    mergeInValue(AI, getValueState(*CAI));
  }
}

void SCCPSolver::Solve() {
  // Process the work lists until they are empty!
  while (!BBWorkList.empty() || !InstWorkList.empty() ||
         !OverdefinedInstWorkList.empty()) {
    // Process the overdefined instruction's work list first, which drives other
    // things to overdefined more quickly.
    while (!OverdefinedInstWorkList.empty()) {
      Value *I = OverdefinedInstWorkList.pop_back_val();

      DEBUG(errs() << "\nPopped off OI-WL: " << *I << '\n');

      // "I" got into the work list because it either made the transition from
      // bottom to constant
      //
      // Anything on this worklist that is overdefined need not be visited
      // since all of its users will have already been marked as overdefined
      // Update all of the users of this instruction's value.
      //
      for (Value::use_iterator UI = I->use_begin(), E = I->use_end();
           UI != E; ++UI)
        OperandChangedState(*UI);
    }
    
    // Process the instruction work list.
    while (!InstWorkList.empty()) {
      Value *I = InstWorkList.pop_back_val();

      DEBUG(errs() << "\nPopped off I-WL: " << *I << '\n');

      // "I" got into the work list because it made the transition from undef to
      // constant.
      //
      // Anything on this worklist that is overdefined need not be visited
      // since all of its users will have already been marked as overdefined.
      // Update all of the users of this instruction's value.
      //
      if (!getValueState(I).isOverdefined())
        for (Value::use_iterator UI = I->use_begin(), E = I->use_end();
             UI != E; ++UI)
          OperandChangedState(*UI);
    }

    // Process the basic block work list.
    while (!BBWorkList.empty()) {
      BasicBlock *BB = BBWorkList.back();
      BBWorkList.pop_back();

      DEBUG(errs() << "\nPopped off BBWL: " << *BB << '\n');

      // Notify all instructions in this basic block that they are newly
      // executable.
      visit(BB);
    }
  }
}

/// ResolvedUndefsIn - While solving the dataflow for a function, we assume
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
/// This scan also checks for values that use undefs, whose results are actually
/// defined.  For example, 'zext i8 undef to i32' should produce all zeros
/// conservatively, as "(zext i8 X -> i32) & 0xFF00" must always return zero,
/// even if X isn't defined.
bool SCCPSolver::ResolvedUndefsIn(Function &F) {
  for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB) {
    if (!BBExecutable.count(BB))
      continue;
    
    for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I) {
      // Look for instructions which produce undef values.
      if (I->getType()->isVoidTy()) continue;
      
      LatticeVal &LV = getValueState(I);
      if (!LV.isUndefined()) continue;

      // Get the lattice values of the first two operands for use below.
      LatticeVal Op0LV = getValueState(I->getOperand(0));
      LatticeVal Op1LV;
      if (I->getNumOperands() == 2) {
        // If this is a two-operand instruction, and if both operands are
        // undefs, the result stays undef.
        Op1LV = getValueState(I->getOperand(1));
        if (Op0LV.isUndefined() && Op1LV.isUndefined())
          continue;
      }
      
      // If this is an instructions whose result is defined even if the input is
      // not fully defined, propagate the information.
      const Type *ITy = I->getType();
      switch (I->getOpcode()) {
      default: break;          // Leave the instruction as an undef.
      case Instruction::ZExt:
        // After a zero extend, we know the top part is zero.  SExt doesn't have
        // to be handled here, because we don't know whether the top part is 1's
        // or 0's.
        markForcedConstant(I, Constant::getNullValue(ITy));
        return true;
      case Instruction::Mul:
      case Instruction::And:
        // undef * X -> 0.   X could be zero.
        // undef & X -> 0.   X could be zero.
        markForcedConstant(I, Constant::getNullValue(ITy));
        return true;

      case Instruction::Or:
        // undef | X -> -1.   X could be -1.
        markForcedConstant(I, Constant::getAllOnesValue(ITy));
        return true;

      case Instruction::SDiv:
      case Instruction::UDiv:
      case Instruction::SRem:
      case Instruction::URem:
        // X / undef -> undef.  No change.
        // X % undef -> undef.  No change.
        if (Op1LV.isUndefined()) break;
        
        // undef / X -> 0.   X could be maxint.
        // undef % X -> 0.   X could be 1.
        markForcedConstant(I, Constant::getNullValue(ITy));
        return true;
        
      case Instruction::AShr:
        // undef >>s X -> undef.  No change.
        if (Op0LV.isUndefined()) break;
        
        // X >>s undef -> X.  X could be 0, X could have the high-bit known set.
        if (Op0LV.isConstant())
          markForcedConstant(I, Op0LV.getConstant());
        else
          markOverdefined(I);
        return true;
      case Instruction::LShr:
      case Instruction::Shl:
        // undef >> X -> undef.  No change.
        // undef << X -> undef.  No change.
        if (Op0LV.isUndefined()) break;
        
        // X >> undef -> 0.  X could be 0.
        // X << undef -> 0.  X could be 0.
        markForcedConstant(I, Constant::getNullValue(ITy));
        return true;
      case Instruction::Select:
        // undef ? X : Y  -> X or Y.  There could be commonality between X/Y.
        if (Op0LV.isUndefined()) {
          if (!Op1LV.isConstant())  // Pick the constant one if there is any.
            Op1LV = getValueState(I->getOperand(2));
        } else if (Op1LV.isUndefined()) {
          // c ? undef : undef -> undef.  No change.
          Op1LV = getValueState(I->getOperand(2));
          if (Op1LV.isUndefined())
            break;
          // Otherwise, c ? undef : x -> x.
        } else {
          // Leave Op1LV as Operand(1)'s LatticeValue.
        }
        
        if (Op1LV.isConstant())
          markForcedConstant(I, Op1LV.getConstant());
        else
          markOverdefined(I);
        return true;
      case Instruction::Call:
        // If a call has an undef result, it is because it is constant foldable
        // but one of the inputs was undef.  Just force the result to
        // overdefined.
        markOverdefined(I);
        return true;
      }
    }
  
    TerminatorInst *TI = BB->getTerminator();
    if (BranchInst *BI = dyn_cast<BranchInst>(TI)) {
      if (!BI->isConditional()) continue;
      if (!getValueState(BI->getCondition()).isUndefined())
        continue;
    } else if (SwitchInst *SI = dyn_cast<SwitchInst>(TI)) {
      if (SI->getNumSuccessors() < 2)   // no cases
        continue;
      if (!getValueState(SI->getCondition()).isUndefined())
        continue;
    } else {
      continue;
    }
    
    // If the edge to the second successor isn't thought to be feasible yet,
    // mark it so now.  We pick the second one so that this goes to some
    // enumerated value in a switch instead of going to the default destination.
    if (KnownFeasibleEdges.count(Edge(BB, TI->getSuccessor(1))))
      continue;
    
    // Otherwise, it isn't already thought to be feasible.  Mark it as such now
    // and return.  This will make other blocks reachable, which will allow new
    // values to be discovered and existing ones to be moved in the lattice.
    markEdgeExecutable(BB, TI->getSuccessor(1));
    
    // This must be a conditional branch of switch on undef.  At this point,
    // force the old terminator to branch to the first successor.  This is
    // required because we are now influencing the dataflow of the function with
    // the assumption that this edge is taken.  If we leave the branch condition
    // as undef, then further analysis could think the undef went another way
    // leading to an inconsistent set of conclusions.
    if (BranchInst *BI = dyn_cast<BranchInst>(TI)) {
      BI->setCondition(ConstantInt::getFalse(BI->getContext()));
    } else {
      SwitchInst *SI = cast<SwitchInst>(TI);
      SI->setCondition(SI->getCaseValue(1));
    }
    
    return true;
  }

  return false;
}


namespace {
  //===--------------------------------------------------------------------===//
  //
  /// SCCP Class - This class uses the SCCPSolver to implement a per-function
  /// Sparse Conditional Constant Propagator.
  ///
  struct SCCP : public FunctionPass {
    static char ID; // Pass identification, replacement for typeid
    SCCP() : FunctionPass(&ID) {}

    // runOnFunction - Run the Sparse Conditional Constant Propagation
    // algorithm, and return true if the function was modified.
    //
    bool runOnFunction(Function &F);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
    }
  };
} // end anonymous namespace

char SCCP::ID = 0;
static RegisterPass<SCCP>
X("sccp", "Sparse Conditional Constant Propagation");

// createSCCPPass - This is the public interface to this file.
FunctionPass *llvm::createSCCPPass() {
  return new SCCP();
}

static void DeleteInstructionInBlock(BasicBlock *BB) {
  DEBUG(errs() << "  BasicBlock Dead:" << *BB);
  ++NumDeadBlocks;
  
  // Delete the instructions backwards, as it has a reduced likelihood of
  // having to update as many def-use and use-def chains.
  while (!isa<TerminatorInst>(BB->begin())) {
    Instruction *I = --BasicBlock::iterator(BB->getTerminator());
    
    if (!I->use_empty())
      I->replaceAllUsesWith(UndefValue::get(I->getType()));
    BB->getInstList().erase(I);
    ++NumInstRemoved;
  }
}

// runOnFunction() - Run the Sparse Conditional Constant Propagation algorithm,
// and return true if the function was modified.
//
bool SCCP::runOnFunction(Function &F) {
  DEBUG(errs() << "SCCP on function '" << F.getName() << "'\n");
  SCCPSolver Solver(getAnalysisIfAvailable<TargetData>());

  // Mark the first block of the function as being executable.
  Solver.MarkBlockExecutable(F.begin());

  // Mark all arguments to the function as being overdefined.
  for (Function::arg_iterator AI = F.arg_begin(), E = F.arg_end(); AI != E;++AI)
    Solver.markOverdefined(AI);

  // Solve for constants.
  bool ResolvedUndefs = true;
  while (ResolvedUndefs) {
    Solver.Solve();
    DEBUG(errs() << "RESOLVING UNDEFs\n");
    ResolvedUndefs = Solver.ResolvedUndefsIn(F);
  }

  bool MadeChanges = false;

  // If we decided that there are basic blocks that are dead in this function,
  // delete their contents now.  Note that we cannot actually delete the blocks,
  // as we cannot modify the CFG of the function.

  for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB) {
    if (!Solver.isBlockExecutable(BB)) {
      DeleteInstructionInBlock(BB);
      MadeChanges = true;
      continue;
    }
  
    // Iterate over all of the instructions in a function, replacing them with
    // constants if we have found them to be of constant values.
    //
    for (BasicBlock::iterator BI = BB->begin(), E = BB->end(); BI != E; ) {
      Instruction *Inst = BI++;
      if (Inst->getType()->isVoidTy() || isa<TerminatorInst>(Inst))
        continue;
      
      LatticeVal IV = Solver.getLatticeValueFor(Inst);
      if (IV.isOverdefined())
        continue;
      
      Constant *Const = IV.isConstant()
        ? IV.getConstant() : UndefValue::get(Inst->getType());
      DEBUG(errs() << "  Constant: " << *Const << " = " << *Inst);

      // Replaces all of the uses of a variable with uses of the constant.
      Inst->replaceAllUsesWith(Const);
      
      // Delete the instruction.
      Inst->eraseFromParent();
      
      // Hey, we just changed something!
      MadeChanges = true;
      ++NumInstRemoved;
    }
  }

  return MadeChanges;
}

namespace {
  //===--------------------------------------------------------------------===//
  //
  /// IPSCCP Class - This class implements interprocedural Sparse Conditional
  /// Constant Propagation.
  ///
  struct IPSCCP : public ModulePass {
    static char ID;
    IPSCCP() : ModulePass(&ID) {}
    bool runOnModule(Module &M);
  };
} // end anonymous namespace

char IPSCCP::ID = 0;
static RegisterPass<IPSCCP>
Y("ipsccp", "Interprocedural Sparse Conditional Constant Propagation");

// createIPSCCPPass - This is the public interface to this file.
ModulePass *llvm::createIPSCCPPass() {
  return new IPSCCP();
}


static bool AddressIsTaken(GlobalValue *GV) {
  // Delete any dead constantexpr klingons.
  GV->removeDeadConstantUsers();

  for (Value::use_iterator UI = GV->use_begin(), E = GV->use_end();
       UI != E; ++UI)
    if (StoreInst *SI = dyn_cast<StoreInst>(*UI)) {
      if (SI->getOperand(0) == GV || SI->isVolatile())
        return true;  // Storing addr of GV.
    } else if (isa<InvokeInst>(*UI) || isa<CallInst>(*UI)) {
      // Make sure we are calling the function, not passing the address.
      if (UI.getOperandNo() != 0)
        return true;
    } else if (LoadInst *LI = dyn_cast<LoadInst>(*UI)) {
      if (LI->isVolatile())
        return true;
    } else if (isa<BlockAddress>(*UI)) {
      // blockaddress doesn't take the address of the function, it takes addr
      // of label.
    } else {
      return true;
    }
  return false;
}

bool IPSCCP::runOnModule(Module &M) {
  SCCPSolver Solver(getAnalysisIfAvailable<TargetData>());

  // Loop over all functions, marking arguments to those with their addresses
  // taken or that are external as overdefined.
  //
  for (Module::iterator F = M.begin(), E = M.end(); F != E; ++F) {
    if (F->isDeclaration())
      continue;
    
    if (!F->hasLocalLinkage() || AddressIsTaken(F)) {
      Solver.MarkBlockExecutable(F->begin());
      for (Function::arg_iterator AI = F->arg_begin(), E = F->arg_end();
           AI != E; ++AI)
        Solver.markOverdefined(AI);
    } else {
      Solver.AddTrackedFunction(F);
    }
  }

  // Loop over global variables.  We inform the solver about any internal global
  // variables that do not have their 'addresses taken'.  If they don't have
  // their addresses taken, we can propagate constants through them.
  for (Module::global_iterator G = M.global_begin(), E = M.global_end();
       G != E; ++G)
    if (!G->isConstant() && G->hasLocalLinkage() && !AddressIsTaken(G))
      Solver.TrackValueOfGlobalVariable(G);

  // Solve for constants.
  bool ResolvedUndefs = true;
  while (ResolvedUndefs) {
    Solver.Solve();

    DEBUG(errs() << "RESOLVING UNDEFS\n");
    ResolvedUndefs = false;
    for (Module::iterator F = M.begin(), E = M.end(); F != E; ++F)
      ResolvedUndefs |= Solver.ResolvedUndefsIn(*F);
  }

  bool MadeChanges = false;

  // Iterate over all of the instructions in the module, replacing them with
  // constants if we have found them to be of constant values.
  //
  SmallVector<BasicBlock*, 512> BlocksToErase;

  for (Module::iterator F = M.begin(), E = M.end(); F != E; ++F) {
    if (Solver.isBlockExecutable(F->begin())) {
      for (Function::arg_iterator AI = F->arg_begin(), E = F->arg_end();
           AI != E; ++AI) {
        if (AI->use_empty()) continue;
        
        LatticeVal IV = Solver.getLatticeValueFor(AI);
        if (IV.isOverdefined()) continue;
        
        Constant *CST = IV.isConstant() ?
        IV.getConstant() : UndefValue::get(AI->getType());
        DEBUG(errs() << "***  Arg " << *AI << " = " << *CST <<"\n");
        
        // Replaces all of the uses of a variable with uses of the
        // constant.
        AI->replaceAllUsesWith(CST);
        ++IPNumArgsElimed;
      }
    }

    for (Function::iterator BB = F->begin(), E = F->end(); BB != E; ++BB) {
      if (!Solver.isBlockExecutable(BB)) {
        DeleteInstructionInBlock(BB);
        MadeChanges = true;

        TerminatorInst *TI = BB->getTerminator();
        for (unsigned i = 0, e = TI->getNumSuccessors(); i != e; ++i) {
          BasicBlock *Succ = TI->getSuccessor(i);
          if (!Succ->empty() && isa<PHINode>(Succ->begin()))
            TI->getSuccessor(i)->removePredecessor(BB);
        }
        if (!TI->use_empty())
          TI->replaceAllUsesWith(UndefValue::get(TI->getType()));
        TI->eraseFromParent();

        if (&*BB != &F->front())
          BlocksToErase.push_back(BB);
        else
          new UnreachableInst(M.getContext(), BB);
        continue;
      }
      
      for (BasicBlock::iterator BI = BB->begin(), E = BB->end(); BI != E; ) {
        Instruction *Inst = BI++;
        if (Inst->getType()->isVoidTy())
          continue;
        
        LatticeVal IV = Solver.getLatticeValueFor(Inst);
        if (IV.isOverdefined())
          continue;
        
        Constant *Const = IV.isConstant()
          ? IV.getConstant() : UndefValue::get(Inst->getType());
        DEBUG(errs() << "  Constant: " << *Const << " = " << *Inst);

        // Replaces all of the uses of a variable with uses of the
        // constant.
        Inst->replaceAllUsesWith(Const);
        
        // Delete the instruction.
        if (!isa<CallInst>(Inst) && !isa<TerminatorInst>(Inst))
          Inst->eraseFromParent();

        // Hey, we just changed something!
        MadeChanges = true;
        ++IPNumInstRemoved;
      }
    }

    // Now that all instructions in the function are constant folded, erase dead
    // blocks, because we can now use ConstantFoldTerminator to get rid of
    // in-edges.
    for (unsigned i = 0, e = BlocksToErase.size(); i != e; ++i) {
      // If there are any PHI nodes in this successor, drop entries for BB now.
      BasicBlock *DeadBB = BlocksToErase[i];
      while (!DeadBB->use_empty()) {
        Instruction *I = cast<Instruction>(DeadBB->use_back());
        bool Folded = ConstantFoldTerminator(I->getParent());
        if (!Folded) {
          // The constant folder may not have been able to fold the terminator
          // if this is a branch or switch on undef.  Fold it manually as a
          // branch to the first successor.
#ifndef NDEBUG
          if (BranchInst *BI = dyn_cast<BranchInst>(I)) {
            assert(BI->isConditional() && isa<UndefValue>(BI->getCondition()) &&
                   "Branch should be foldable!");
          } else if (SwitchInst *SI = dyn_cast<SwitchInst>(I)) {
            assert(isa<UndefValue>(SI->getCondition()) && "Switch should fold");
          } else {
            llvm_unreachable("Didn't fold away reference to block!");
          }
#endif
          
          // Make this an uncond branch to the first successor.
          TerminatorInst *TI = I->getParent()->getTerminator();
          BranchInst::Create(TI->getSuccessor(0), TI);
          
          // Remove entries in successor phi nodes to remove edges.
          for (unsigned i = 1, e = TI->getNumSuccessors(); i != e; ++i)
            TI->getSuccessor(i)->removePredecessor(TI->getParent());
          
          // Remove the old terminator.
          TI->eraseFromParent();
        }
      }

      // Finally, delete the basic block.
      F->getBasicBlockList().erase(DeadBB);
    }
    BlocksToErase.clear();
  }

  // If we inferred constant or undef return values for a function, we replaced
  // all call uses with the inferred value.  This means we don't need to bother
  // actually returning anything from the function.  Replace all return
  // instructions with return undef.
  // TODO: Process multiple value ret instructions also.
  const DenseMap<Function*, LatticeVal> &RV = Solver.getTrackedRetVals();
  for (DenseMap<Function*, LatticeVal>::const_iterator I = RV.begin(),
         E = RV.end(); I != E; ++I)
    if (!I->second.isOverdefined() &&
        !I->first->getReturnType()->isVoidTy()) {
      Function *F = I->first;
      for (Function::iterator BB = F->begin(), E = F->end(); BB != E; ++BB)
        if (ReturnInst *RI = dyn_cast<ReturnInst>(BB->getTerminator()))
          if (!isa<UndefValue>(RI->getOperand(0)))
            RI->setOperand(0, UndefValue::get(F->getReturnType()));
    }

  // If we infered constant or undef values for globals variables, we can delete
  // the global and any stores that remain to it.
  const DenseMap<GlobalVariable*, LatticeVal> &TG = Solver.getTrackedGlobals();
  for (DenseMap<GlobalVariable*, LatticeVal>::const_iterator I = TG.begin(),
         E = TG.end(); I != E; ++I) {
    GlobalVariable *GV = I->first;
    assert(!I->second.isOverdefined() &&
           "Overdefined values should have been taken out of the map!");
    DEBUG(errs() << "Found that GV '" << GV->getName() << "' is constant!\n");
    while (!GV->use_empty()) {
      StoreInst *SI = cast<StoreInst>(GV->use_back());
      SI->eraseFromParent();
    }
    M.getGlobalList().erase(GV);
    ++IPNumGlobalConst;
  }

  return MadeChanges;
}
