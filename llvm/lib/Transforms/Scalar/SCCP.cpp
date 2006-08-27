//===- SCCP.cpp - Sparse Conditional Constant Propagation -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
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
// Notice that:
//   * This pass has a habit of making definitions be dead.  It is a good idea
//     to to run a DCE pass sometime after running this pass.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "sccp"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/Support/InstVisitor.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/hash_map"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <iostream>
#include <set>
using namespace llvm;

// LatticeVal class - This class represents the different lattice values that an
// instruction may occupy.  It is a simple class with value semantics.
//
namespace {

class LatticeVal {
  enum {
    undefined,           // This instruction has no known value
    constant,            // This instruction has a constant value
    overdefined          // This instruction has an unknown value
  } LatticeValue;        // The current lattice position
  Constant *ConstantVal; // If Constant value, the current value
public:
  inline LatticeVal() : LatticeValue(undefined), ConstantVal(0) {}

  // markOverdefined - Return true if this is a new status to be in...
  inline bool markOverdefined() {
    if (LatticeValue != overdefined) {
      LatticeValue = overdefined;
      return true;
    }
    return false;
  }

  // markConstant - Return true if this is a new status for us...
  inline bool markConstant(Constant *V) {
    if (LatticeValue != constant) {
      LatticeValue = constant;
      ConstantVal = V;
      return true;
    } else {
      assert(ConstantVal == V && "Marking constant with different value");
    }
    return false;
  }

  inline bool isUndefined()   const { return LatticeValue == undefined; }
  inline bool isConstant()    const { return LatticeValue == constant; }
  inline bool isOverdefined() const { return LatticeValue == overdefined; }

  inline Constant *getConstant() const {
    assert(isConstant() && "Cannot get the constant of a non-constant!");
    return ConstantVal;
  }
};

} // end anonymous namespace


//===----------------------------------------------------------------------===//
//
/// SCCPSolver - This class is a general purpose solver for Sparse Conditional
/// Constant Propagation.
///
class SCCPSolver : public InstVisitor<SCCPSolver> {
  std::set<BasicBlock*>     BBExecutable;// The basic blocks that are executable
  hash_map<Value*, LatticeVal> ValueState;  // The state each value is in...

  /// GlobalValue - If we are tracking any values for the contents of a global
  /// variable, we keep a mapping from the constant accessor to the element of
  /// the global, to the currently known value.  If the value becomes
  /// overdefined, it's entry is simply removed from this map.
  hash_map<GlobalVariable*, LatticeVal> TrackedGlobals;

  /// TrackedFunctionRetVals - If we are tracking arguments into and the return
  /// value out of a function, it will have an entry in this map, indicating
  /// what the known return value for the function is.
  hash_map<Function*, LatticeVal> TrackedFunctionRetVals;

  // The reason for two worklists is that overdefined is the lowest state
  // on the lattice, and moving things to overdefined as fast as possible
  // makes SCCP converge much faster.
  // By having a separate worklist, we accomplish this because everything
  // possibly overdefined will become overdefined at the soonest possible
  // point.
  std::vector<Value*> OverdefinedInstWorkList;
  std::vector<Value*> InstWorkList;


  std::vector<BasicBlock*>  BBWorkList;  // The BasicBlock work list

  /// UsersOfOverdefinedPHIs - Keep track of any users of PHI nodes that are not
  /// overdefined, despite the fact that the PHI node is overdefined.
  std::multimap<PHINode*, Instruction*> UsersOfOverdefinedPHIs;

  /// KnownFeasibleEdges - Entries in this set are edges which have already had
  /// PHI nodes retriggered.
  typedef std::pair<BasicBlock*,BasicBlock*> Edge;
  std::set<Edge> KnownFeasibleEdges;
public:

  /// MarkBlockExecutable - This method can be used by clients to mark all of
  /// the blocks that are known to be intrinsically live in the processed unit.
  void MarkBlockExecutable(BasicBlock *BB) {
    DEBUG(std::cerr << "Marking Block Executable: " << BB->getName() << "\n");
    BBExecutable.insert(BB);   // Basic block is executable!
    BBWorkList.push_back(BB);  // Add the block to the work list!
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
    assert(F->hasInternalLinkage() && "Can only track internal functions!");
    // Add an entry, F -> undef.
    TrackedFunctionRetVals[F];
  }

  /// Solve - Solve for constants and executable blocks.
  ///
  void Solve();

  /// ResolveBranchesIn - While solving the dataflow for a function, we assume
  /// that branches on undef values cannot reach any of their successors.
  /// However, this is not a safe assumption.  After we solve dataflow, this
  /// method should be use to handle this.  If this returns true, the solver
  /// should be rerun.
  bool ResolveBranchesIn(Function &F);

  /// getExecutableBlocks - Once we have solved for constants, return the set of
  /// blocks that is known to be executable.
  std::set<BasicBlock*> &getExecutableBlocks() {
    return BBExecutable;
  }

  /// getValueMapping - Once we have solved for constants, return the mapping of
  /// LLVM values to LatticeVals.
  hash_map<Value*, LatticeVal> &getValueMapping() {
    return ValueState;
  }

  /// getTrackedFunctionRetVals - Get the inferred return value map.
  ///
  const hash_map<Function*, LatticeVal> &getTrackedFunctionRetVals() {
    return TrackedFunctionRetVals;
  }

  /// getTrackedGlobals - Get and return the set of inferred initializers for
  /// global variables.
  const hash_map<GlobalVariable*, LatticeVal> &getTrackedGlobals() {
    return TrackedGlobals;
  }


private:
  // markConstant - Make a value be marked as "constant".  If the value
  // is not already a constant, add it to the instruction work list so that
  // the users of the instruction are updated later.
  //
  inline void markConstant(LatticeVal &IV, Value *V, Constant *C) {
    if (IV.markConstant(C)) {
      DEBUG(std::cerr << "markConstant: " << *C << ": " << *V);
      InstWorkList.push_back(V);
    }
  }
  inline void markConstant(Value *V, Constant *C) {
    markConstant(ValueState[V], V, C);
  }

  // markOverdefined - Make a value be marked as "overdefined". If the
  // value is not already overdefined, add it to the overdefined instruction
  // work list so that the users of the instruction are updated later.

  inline void markOverdefined(LatticeVal &IV, Value *V) {
    if (IV.markOverdefined()) {
      DEBUG(std::cerr << "markOverdefined: ";
            if (Function *F = dyn_cast<Function>(V))
              std::cerr << "Function '" << F->getName() << "'\n";
            else
              std::cerr << *V);
      // Only instructions go on the work list
      OverdefinedInstWorkList.push_back(V);
    }
  }
  inline void markOverdefined(Value *V) {
    markOverdefined(ValueState[V], V);
  }

  inline void mergeInValue(LatticeVal &IV, Value *V, LatticeVal &MergeWithV) {
    if (IV.isOverdefined() || MergeWithV.isUndefined())
      return;  // Noop.
    if (MergeWithV.isOverdefined())
      markOverdefined(IV, V);
    else if (IV.isUndefined())
      markConstant(IV, V, MergeWithV.getConstant());
    else if (IV.getConstant() != MergeWithV.getConstant())
      markOverdefined(IV, V);
  }
  
  inline void mergeInValue(Value *V, LatticeVal &MergeWithV) {
    return mergeInValue(ValueState[V], V, MergeWithV);
  }


  // getValueState - Return the LatticeVal object that corresponds to the value.
  // This function is necessary because not all values should start out in the
  // underdefined state... Argument's should be overdefined, and
  // constants should be marked as constants.  If a value is not known to be an
  // Instruction object, then use this accessor to get its value from the map.
  //
  inline LatticeVal &getValueState(Value *V) {
    hash_map<Value*, LatticeVal>::iterator I = ValueState.find(V);
    if (I != ValueState.end()) return I->second;  // Common case, in the map

    if (Constant *CPV = dyn_cast<Constant>(V)) {
      if (isa<UndefValue>(V)) {
        // Nothing to do, remain undefined.
      } else {
        ValueState[CPV].markConstant(CPV);          // Constants are constant
      }
    }
    // All others are underdefined by default...
    return ValueState[V];
  }

  // markEdgeExecutable - Mark a basic block as executable, adding it to the BB
  // work list if it is not already executable...
  //
  void markEdgeExecutable(BasicBlock *Source, BasicBlock *Dest) {
    if (!KnownFeasibleEdges.insert(Edge(Source, Dest)).second)
      return;  // This edge is already known to be executable!

    if (BBExecutable.count(Dest)) {
      DEBUG(std::cerr << "Marking Edge Executable: " << Source->getName()
                      << " -> " << Dest->getName() << "\n");

      // The destination is already executable, but we just made an edge
      // feasible that wasn't before.  Revisit the PHI nodes in the block
      // because they have potentially new operands.
      for (BasicBlock::iterator I = Dest->begin(); isa<PHINode>(I); ++I)
        visitPHINode(*cast<PHINode>(I));

    } else {
      MarkBlockExecutable(Dest);
    }
  }

  // getFeasibleSuccessors - Return a vector of booleans to indicate which
  // successors are reachable from a given terminator instruction.
  //
  void getFeasibleSuccessors(TerminatorInst &TI, std::vector<bool> &Succs);

  // isEdgeFeasible - Return true if the control flow edge from the 'From' basic
  // block to the 'To' basic block is currently feasible...
  //
  bool isEdgeFeasible(BasicBlock *From, BasicBlock *To);

  // OperandChangedState - This method is invoked on all of the users of an
  // instruction that was just changed state somehow....  Based on this
  // information, we need to update the specified user of this instruction.
  //
  void OperandChangedState(User *U) {
    // Only instructions use other variable values!
    Instruction &I = cast<Instruction>(*U);
    if (BBExecutable.count(I.getParent()))   // Inst is executable?
      visit(I);
  }

private:
  friend class InstVisitor<SCCPSolver>;

  // visit implementations - Something changed in this instruction... Either an
  // operand made a transition, or the instruction is newly executable.  Change
  // the value type of I to reflect these changes if appropriate.
  //
  void visitPHINode(PHINode &I);

  // Terminators
  void visitReturnInst(ReturnInst &I);
  void visitTerminatorInst(TerminatorInst &TI);

  void visitCastInst(CastInst &I);
  void visitSelectInst(SelectInst &I);
  void visitBinaryOperator(Instruction &I);
  void visitShiftInst(ShiftInst &I) { visitBinaryOperator(I); }
  void visitExtractElementInst(ExtractElementInst &I);
  void visitInsertElementInst(InsertElementInst &I);
  void visitShuffleVectorInst(ShuffleVectorInst &I);

  // Instructions that cannot be folded away...
  void visitStoreInst     (Instruction &I);
  void visitLoadInst      (LoadInst &I);
  void visitGetElementPtrInst(GetElementPtrInst &I);
  void visitCallInst      (CallInst &I) { visitCallSite(CallSite::get(&I)); }
  void visitInvokeInst    (InvokeInst &II) {
    visitCallSite(CallSite::get(&II));
    visitTerminatorInst(II);
  }
  void visitCallSite      (CallSite CS);
  void visitUnwindInst    (TerminatorInst &I) { /*returns void*/ }
  void visitUnreachableInst(TerminatorInst &I) { /*returns void*/ }
  void visitAllocationInst(Instruction &I) { markOverdefined(&I); }
  void visitVANextInst    (Instruction &I) { markOverdefined(&I); }
  void visitVAArgInst     (Instruction &I) { markOverdefined(&I); }
  void visitFreeInst      (Instruction &I) { /*returns void*/ }

  void visitInstruction(Instruction &I) {
    // If a new instruction is added to LLVM that we don't handle...
    std::cerr << "SCCP: Don't know how to handle: " << I;
    markOverdefined(&I);   // Just in case
  }
};

// getFeasibleSuccessors - Return a vector of booleans to indicate which
// successors are reachable from a given terminator instruction.
//
void SCCPSolver::getFeasibleSuccessors(TerminatorInst &TI,
                                       std::vector<bool> &Succs) {
  Succs.resize(TI.getNumSuccessors());
  if (BranchInst *BI = dyn_cast<BranchInst>(&TI)) {
    if (BI->isUnconditional()) {
      Succs[0] = true;
    } else {
      LatticeVal &BCValue = getValueState(BI->getCondition());
      if (BCValue.isOverdefined() ||
          (BCValue.isConstant() && !isa<ConstantBool>(BCValue.getConstant()))) {
        // Overdefined condition variables, and branches on unfoldable constant
        // conditions, mean the branch could go either way.
        Succs[0] = Succs[1] = true;
      } else if (BCValue.isConstant()) {
        // Constant condition variables mean the branch can only go a single way
        Succs[BCValue.getConstant() == ConstantBool::False] = true;
      }
    }
  } else if (InvokeInst *II = dyn_cast<InvokeInst>(&TI)) {
    // Invoke instructions successors are always executable.
    Succs[0] = Succs[1] = true;
  } else if (SwitchInst *SI = dyn_cast<SwitchInst>(&TI)) {
    LatticeVal &SCValue = getValueState(SI->getCondition());
    if (SCValue.isOverdefined() ||   // Overdefined condition?
        (SCValue.isConstant() && !isa<ConstantInt>(SCValue.getConstant()))) {
      // All destinations are executable!
      Succs.assign(TI.getNumSuccessors(), true);
    } else if (SCValue.isConstant()) {
      Constant *CPV = SCValue.getConstant();
      // Make sure to skip the "default value" which isn't a value
      for (unsigned i = 1, E = SI->getNumSuccessors(); i != E; ++i) {
        if (SI->getSuccessorValue(i) == CPV) {// Found the right branch...
          Succs[i] = true;
          return;
        }
      }

      // Constant value not equal to any of the branches... must execute
      // default branch then...
      Succs[0] = true;
    }
  } else {
    std::cerr << "SCCP: Don't know how to handle: " << TI;
    Succs.assign(TI.getNumSuccessors(), true);
  }
}


// isEdgeFeasible - Return true if the control flow edge from the 'From' basic
// block to the 'To' basic block is currently feasible...
//
bool SCCPSolver::isEdgeFeasible(BasicBlock *From, BasicBlock *To) {
  assert(BBExecutable.count(To) && "Dest should always be alive!");

  // Make sure the source basic block is executable!!
  if (!BBExecutable.count(From)) return false;

  // Check to make sure this edge itself is actually feasible now...
  TerminatorInst *TI = From->getTerminator();
  if (BranchInst *BI = dyn_cast<BranchInst>(TI)) {
    if (BI->isUnconditional())
      return true;
    else {
      LatticeVal &BCValue = getValueState(BI->getCondition());
      if (BCValue.isOverdefined()) {
        // Overdefined condition variables mean the branch could go either way.
        return true;
      } else if (BCValue.isConstant()) {
        // Not branching on an evaluatable constant?
        if (!isa<ConstantBool>(BCValue.getConstant())) return true;

        // Constant condition variables mean the branch can only go a single way
        return BI->getSuccessor(BCValue.getConstant() ==
                                       ConstantBool::False) == To;
      }
      return false;
    }
  } else if (InvokeInst *II = dyn_cast<InvokeInst>(TI)) {
    // Invoke instructions successors are always executable.
    return true;
  } else if (SwitchInst *SI = dyn_cast<SwitchInst>(TI)) {
    LatticeVal &SCValue = getValueState(SI->getCondition());
    if (SCValue.isOverdefined()) {  // Overdefined condition?
      // All destinations are executable!
      return true;
    } else if (SCValue.isConstant()) {
      Constant *CPV = SCValue.getConstant();
      if (!isa<ConstantInt>(CPV))
        return true;  // not a foldable constant?

      // Make sure to skip the "default value" which isn't a value
      for (unsigned i = 1, E = SI->getNumSuccessors(); i != E; ++i)
        if (SI->getSuccessorValue(i) == CPV) // Found the taken branch...
          return SI->getSuccessor(i) == To;

      // Constant value not equal to any of the branches... must execute
      // default branch then...
      return SI->getDefaultDest() == To;
    }
    return false;
  } else {
    std::cerr << "Unknown terminator instruction: " << *TI;
    abort();
  }
}

// visit Implementations - Something changed in this instruction... Either an
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
  LatticeVal &PNIV = getValueState(&PN);
  if (PNIV.isOverdefined()) {
    // There may be instructions using this PHI node that are not overdefined
    // themselves.  If so, make sure that they know that the PHI node operand
    // changed.
    std::multimap<PHINode*, Instruction*>::iterator I, E;
    tie(I, E) = UsersOfOverdefinedPHIs.equal_range(&PN);
    if (I != E) {
      std::vector<Instruction*> Users;
      Users.reserve(std::distance(I, E));
      for (; I != E; ++I) Users.push_back(I->second);
      while (!Users.empty()) {
        visit(Users.back());
        Users.pop_back();
      }
    }
    return;  // Quick exit
  }

  // Super-extra-high-degree PHI nodes are unlikely to ever be marked constant,
  // and slow us down a lot.  Just mark them overdefined.
  if (PN.getNumIncomingValues() > 64) {
    markOverdefined(PNIV, &PN);
    return;
  }

  // Look at all of the executable operands of the PHI node.  If any of them
  // are overdefined, the PHI becomes overdefined as well.  If they are all
  // constant, and they agree with each other, the PHI becomes the identical
  // constant.  If they are constant and don't agree, the PHI is overdefined.
  // If there are no executable operands, the PHI remains undefined.
  //
  Constant *OperandVal = 0;
  for (unsigned i = 0, e = PN.getNumIncomingValues(); i != e; ++i) {
    LatticeVal &IV = getValueState(PN.getIncomingValue(i));
    if (IV.isUndefined()) continue;  // Doesn't influence PHI node.

    if (isEdgeFeasible(PN.getIncomingBlock(i), PN.getParent())) {
      if (IV.isOverdefined()) {   // PHI node becomes overdefined!
        markOverdefined(PNIV, &PN);
        return;
      }

      if (OperandVal == 0) {   // Grab the first value...
        OperandVal = IV.getConstant();
      } else {                // Another value is being merged in!
        // There is already a reachable operand.  If we conflict with it,
        // then the PHI node becomes overdefined.  If we agree with it, we
        // can continue on.

        // Check to see if there are two different constants merging...
        if (IV.getConstant() != OperandVal) {
          // Yes there is.  This means the PHI node is not constant.
          // You must be overdefined poor PHI.
          //
          markOverdefined(PNIV, &PN);    // The PHI node now becomes overdefined
          return;    // I'm done analyzing you
        }
      }
    }
  }

  // If we exited the loop, this means that the PHI node only has constant
  // arguments that agree with each other(and OperandVal is the constant) or
  // OperandVal is null because there are no defined incoming arguments.  If
  // this is the case, the PHI remains undefined.
  //
  if (OperandVal)
    markConstant(PNIV, &PN, OperandVal);      // Acquire operand value
}

void SCCPSolver::visitReturnInst(ReturnInst &I) {
  if (I.getNumOperands() == 0) return;  // Ret void

  // If we are tracking the return value of this function, merge it in.
  Function *F = I.getParent()->getParent();
  if (F->hasInternalLinkage() && !TrackedFunctionRetVals.empty()) {
    hash_map<Function*, LatticeVal>::iterator TFRVI =
      TrackedFunctionRetVals.find(F);
    if (TFRVI != TrackedFunctionRetVals.end() &&
        !TFRVI->second.isOverdefined()) {
      LatticeVal &IV = getValueState(I.getOperand(0));
      mergeInValue(TFRVI->second, F, IV);
    }
  }
}


void SCCPSolver::visitTerminatorInst(TerminatorInst &TI) {
  std::vector<bool> SuccFeasible;
  getFeasibleSuccessors(TI, SuccFeasible);

  BasicBlock *BB = TI.getParent();

  // Mark all feasible successors executable...
  for (unsigned i = 0, e = SuccFeasible.size(); i != e; ++i)
    if (SuccFeasible[i])
      markEdgeExecutable(BB, TI.getSuccessor(i));
}

void SCCPSolver::visitCastInst(CastInst &I) {
  Value *V = I.getOperand(0);
  LatticeVal &VState = getValueState(V);
  if (VState.isOverdefined())          // Inherit overdefinedness of operand
    markOverdefined(&I);
  else if (VState.isConstant())        // Propagate constant value
    markConstant(&I, ConstantExpr::getCast(VState.getConstant(), I.getType()));
}

void SCCPSolver::visitSelectInst(SelectInst &I) {
  LatticeVal &CondValue = getValueState(I.getCondition());
  if (CondValue.isUndefined())
    return;
  if (CondValue.isConstant()) {
    Value *InVal = 0;
    if (CondValue.getConstant() == ConstantBool::True) {
      mergeInValue(&I, getValueState(I.getTrueValue()));
      return;
    } else if (CondValue.getConstant() == ConstantBool::False) {
      mergeInValue(&I, getValueState(I.getFalseValue()));
      return;
    }
  }
  
  // Otherwise, the condition is overdefined or a constant we can't evaluate.
  // See if we can produce something better than overdefined based on the T/F
  // value.
  LatticeVal &TVal = getValueState(I.getTrueValue());
  LatticeVal &FVal = getValueState(I.getFalseValue());
  
  // select ?, C, C -> C.
  if (TVal.isConstant() && FVal.isConstant() && 
      TVal.getConstant() == FVal.getConstant()) {
    markConstant(&I, FVal.getConstant());
    return;
  }

  if (TVal.isUndefined()) {  // select ?, undef, X -> X.
    mergeInValue(&I, FVal);
  } else if (FVal.isUndefined()) {  // select ?, X, undef -> X.
    mergeInValue(&I, TVal);
  } else {
    markOverdefined(&I);
  }
}

// Handle BinaryOperators and Shift Instructions...
void SCCPSolver::visitBinaryOperator(Instruction &I) {
  LatticeVal &IV = ValueState[&I];
  if (IV.isOverdefined()) return;

  LatticeVal &V1State = getValueState(I.getOperand(0));
  LatticeVal &V2State = getValueState(I.getOperand(1));

  if (V1State.isOverdefined() || V2State.isOverdefined()) {
    // If this is an AND or OR with 0 or -1, it doesn't matter that the other
    // operand is overdefined.
    if (I.getOpcode() == Instruction::And || I.getOpcode() == Instruction::Or) {
      LatticeVal *NonOverdefVal = 0;
      if (!V1State.isOverdefined()) {
        NonOverdefVal = &V1State;
      } else if (!V2State.isOverdefined()) {
        NonOverdefVal = &V2State;
      }

      if (NonOverdefVal) {
        if (NonOverdefVal->isUndefined()) {
          // Could annihilate value.
          if (I.getOpcode() == Instruction::And)
            markConstant(IV, &I, Constant::getNullValue(I.getType()));
          else
            markConstant(IV, &I, ConstantInt::getAllOnesValue(I.getType()));
          return;
        } else {
          if (I.getOpcode() == Instruction::And) {
            if (NonOverdefVal->getConstant()->isNullValue()) {
              markConstant(IV, &I, NonOverdefVal->getConstant());
              return;      // X or 0 = -1
            }
          } else {
            if (ConstantIntegral *CI =
                     dyn_cast<ConstantIntegral>(NonOverdefVal->getConstant()))
              if (CI->isAllOnesValue()) {
                markConstant(IV, &I, NonOverdefVal->getConstant());
                return;    // X or -1 = -1
              }
          }
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
            LatticeVal &In1 = getValueState(PN1->getIncomingValue(i));
            BasicBlock *InBlock = PN1->getIncomingBlock(i);
            LatticeVal &In2 =
              getValueState(PN2->getIncomingValueForBlock(InBlock));

            if (In1.isOverdefined() || In2.isOverdefined()) {
              Result.markOverdefined();
              break;  // Cannot fold this operation over the PHI nodes!
            } else if (In1.isConstant() && In2.isConstant()) {
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
          } else if (Result.isUndefined()) {
            return;
          }

          // Okay, this really is overdefined now.  Since we might have
          // speculatively thought that this was not overdefined before, and
          // added ourselves to the UsersOfOverdefinedPHIs list for the PHIs,
          // make sure to clean out any entries that we put there, for
          // efficiency.
          std::multimap<PHINode*, Instruction*>::iterator It, E;
          tie(It, E) = UsersOfOverdefinedPHIs.equal_range(PN1);
          while (It != E) {
            if (It->second == &I) {
              UsersOfOverdefinedPHIs.erase(It++);
            } else
              ++It;
          }
          tie(It, E) = UsersOfOverdefinedPHIs.equal_range(PN2);
          while (It != E) {
            if (It->second == &I) {
              UsersOfOverdefinedPHIs.erase(It++);
            } else
              ++It;
          }
        }

    markOverdefined(IV, &I);
  } else if (V1State.isConstant() && V2State.isConstant()) {
    markConstant(IV, &I, ConstantExpr::get(I.getOpcode(), V1State.getConstant(),
                                           V2State.getConstant()));
  }
}

void SCCPSolver::visitExtractElementInst(ExtractElementInst &I) {
  LatticeVal &ValState = getValueState(I.getOperand(0));
  LatticeVal &IdxState = getValueState(I.getOperand(1));

  if (ValState.isOverdefined() || IdxState.isOverdefined())
    markOverdefined(&I);
  else if(ValState.isConstant() && IdxState.isConstant())
    markConstant(&I, ConstantExpr::getExtractElement(ValState.getConstant(),
                                                     IdxState.getConstant()));
}

void SCCPSolver::visitInsertElementInst(InsertElementInst &I) {
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
    markConstant(&I, ConstantExpr::getInsertElement(UndefValue::get(I.getType()),
                                                    EltState.getConstant(),
                                                    IdxState.getConstant()));
}

void SCCPSolver::visitShuffleVectorInst(ShuffleVectorInst &I) {
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
}

// Handle getelementptr instructions... if all operands are constants then we
// can turn this into a getelementptr ConstantExpr.
//
void SCCPSolver::visitGetElementPtrInst(GetElementPtrInst &I) {
  LatticeVal &IV = ValueState[&I];
  if (IV.isOverdefined()) return;

  std::vector<Constant*> Operands;
  Operands.reserve(I.getNumOperands());

  for (unsigned i = 0, e = I.getNumOperands(); i != e; ++i) {
    LatticeVal &State = getValueState(I.getOperand(i));
    if (State.isUndefined())
      return;  // Operands are not resolved yet...
    else if (State.isOverdefined()) {
      markOverdefined(IV, &I);
      return;
    }
    assert(State.isConstant() && "Unknown state!");
    Operands.push_back(State.getConstant());
  }

  Constant *Ptr = Operands[0];
  Operands.erase(Operands.begin());  // Erase the pointer from idx list...

  markConstant(IV, &I, ConstantExpr::getGetElementPtr(Ptr, Operands));
}

void SCCPSolver::visitStoreInst(Instruction &SI) {
  if (TrackedGlobals.empty() || !isa<GlobalVariable>(SI.getOperand(1)))
    return;
  GlobalVariable *GV = cast<GlobalVariable>(SI.getOperand(1));
  hash_map<GlobalVariable*, LatticeVal>::iterator I = TrackedGlobals.find(GV);
  if (I == TrackedGlobals.end() || I->second.isOverdefined()) return;

  // Get the value we are storing into the global.
  LatticeVal &PtrVal = getValueState(SI.getOperand(0));

  mergeInValue(I->second, GV, PtrVal);
  if (I->second.isOverdefined())
    TrackedGlobals.erase(I);      // No need to keep tracking this!
}


// Handle load instructions.  If the operand is a constant pointer to a constant
// global, we can replace the load with the loaded constant value!
void SCCPSolver::visitLoadInst(LoadInst &I) {
  LatticeVal &IV = ValueState[&I];
  if (IV.isOverdefined()) return;

  LatticeVal &PtrVal = getValueState(I.getOperand(0));
  if (PtrVal.isUndefined()) return;   // The pointer is not resolved yet!
  if (PtrVal.isConstant() && !I.isVolatile()) {
    Value *Ptr = PtrVal.getConstant();
    if (isa<ConstantPointerNull>(Ptr)) {
      // load null -> null
      markConstant(IV, &I, Constant::getNullValue(I.getType()));
      return;
    }

    // Transform load (constant global) into the value loaded.
    if (GlobalVariable *GV = dyn_cast<GlobalVariable>(Ptr)) {
      if (GV->isConstant()) {
        if (!GV->isExternal()) {
          markConstant(IV, &I, GV->getInitializer());
          return;
        }
      } else if (!TrackedGlobals.empty()) {
        // If we are tracking this global, merge in the known value for it.
        hash_map<GlobalVariable*, LatticeVal>::iterator It =
          TrackedGlobals.find(GV);
        if (It != TrackedGlobals.end()) {
          mergeInValue(IV, &I, It->second);
          return;
        }
      }
    }

    // Transform load (constantexpr_GEP global, 0, ...) into the value loaded.
    if (ConstantExpr *CE = dyn_cast<ConstantExpr>(Ptr))
      if (CE->getOpcode() == Instruction::GetElementPtr)
    if (GlobalVariable *GV = dyn_cast<GlobalVariable>(CE->getOperand(0)))
      if (GV->isConstant() && !GV->isExternal())
        if (Constant *V =
             ConstantFoldLoadThroughGEPConstantExpr(GV->getInitializer(), CE)) {
          markConstant(IV, &I, V);
          return;
        }
  }

  // Otherwise we cannot say for certain what value this load will produce.
  // Bail out.
  markOverdefined(IV, &I);
}

void SCCPSolver::visitCallSite(CallSite CS) {
  Function *F = CS.getCalledFunction();

  // If we are tracking this function, we must make sure to bind arguments as
  // appropriate.
  hash_map<Function*, LatticeVal>::iterator TFRVI =TrackedFunctionRetVals.end();
  if (F && F->hasInternalLinkage())
    TFRVI = TrackedFunctionRetVals.find(F);

  if (TFRVI != TrackedFunctionRetVals.end()) {
    // If this is the first call to the function hit, mark its entry block
    // executable.
    if (!BBExecutable.count(F->begin()))
      MarkBlockExecutable(F->begin());

    CallSite::arg_iterator CAI = CS.arg_begin();
    for (Function::arg_iterator AI = F->arg_begin(), E = F->arg_end();
         AI != E; ++AI, ++CAI) {
      LatticeVal &IV = ValueState[AI];
      if (!IV.isOverdefined())
        mergeInValue(IV, AI, getValueState(*CAI));
    }
  }
  Instruction *I = CS.getInstruction();
  if (I->getType() == Type::VoidTy) return;

  LatticeVal &IV = ValueState[I];
  if (IV.isOverdefined()) return;

  // Propagate the return value of the function to the value of the instruction.
  if (TFRVI != TrackedFunctionRetVals.end()) {
    mergeInValue(IV, I, TFRVI->second);
    return;
  }

  if (F == 0 || !F->isExternal() || !canConstantFoldCallTo(F)) {
    markOverdefined(IV, I);
    return;
  }

  std::vector<Constant*> Operands;
  Operands.reserve(I->getNumOperands()-1);

  for (CallSite::arg_iterator AI = CS.arg_begin(), E = CS.arg_end();
       AI != E; ++AI) {
    LatticeVal &State = getValueState(*AI);
    if (State.isUndefined())
      return;  // Operands are not resolved yet...
    else if (State.isOverdefined()) {
      markOverdefined(IV, I);
      return;
    }
    assert(State.isConstant() && "Unknown state!");
    Operands.push_back(State.getConstant());
  }

  if (Constant *C = ConstantFoldCall(F, Operands))
    markConstant(IV, I, C);
  else
    markOverdefined(IV, I);
}


void SCCPSolver::Solve() {
  // Process the work lists until they are empty!
  while (!BBWorkList.empty() || !InstWorkList.empty() ||
         !OverdefinedInstWorkList.empty()) {
    // Process the instruction work list...
    while (!OverdefinedInstWorkList.empty()) {
      Value *I = OverdefinedInstWorkList.back();
      OverdefinedInstWorkList.pop_back();

      DEBUG(std::cerr << "\nPopped off OI-WL: " << *I);

      // "I" got into the work list because it either made the transition from
      // bottom to constant
      //
      // Anything on this worklist that is overdefined need not be visited
      // since all of its users will have already been marked as overdefined
      // Update all of the users of this instruction's value...
      //
      for (Value::use_iterator UI = I->use_begin(), E = I->use_end();
           UI != E; ++UI)
        OperandChangedState(*UI);
    }
    // Process the instruction work list...
    while (!InstWorkList.empty()) {
      Value *I = InstWorkList.back();
      InstWorkList.pop_back();

      DEBUG(std::cerr << "\nPopped off I-WL: " << *I);

      // "I" got into the work list because it either made the transition from
      // bottom to constant
      //
      // Anything on this worklist that is overdefined need not be visited
      // since all of its users will have already been marked as overdefined.
      // Update all of the users of this instruction's value...
      //
      if (!getValueState(I).isOverdefined())
        for (Value::use_iterator UI = I->use_begin(), E = I->use_end();
             UI != E; ++UI)
          OperandChangedState(*UI);
    }

    // Process the basic block work list...
    while (!BBWorkList.empty()) {
      BasicBlock *BB = BBWorkList.back();
      BBWorkList.pop_back();

      DEBUG(std::cerr << "\nPopped off BBWL: " << *BB);

      // Notify all instructions in this basic block that they are newly
      // executable.
      visit(BB);
    }
  }
}

/// ResolveBranchesIn - While solving the dataflow for a function, we assume
/// that branches on undef values cannot reach any of their successors.
/// However, this is not a safe assumption.  After we solve dataflow, this
/// method should be use to handle this.  If this returns true, the solver
/// should be rerun.
bool SCCPSolver::ResolveBranchesIn(Function &F) {
  bool BranchesResolved = false;
  for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB)
    if (BBExecutable.count(BB)) {
      TerminatorInst *TI = BB->getTerminator();
      if (BranchInst *BI = dyn_cast<BranchInst>(TI)) {
        if (BI->isConditional()) {
          LatticeVal &BCValue = getValueState(BI->getCondition());
          if (BCValue.isUndefined()) {
            BI->setCondition(ConstantBool::True);
            BranchesResolved = true;
            visit(BI);
          }
        }
      } else if (SwitchInst *SI = dyn_cast<SwitchInst>(TI)) {
        LatticeVal &SCValue = getValueState(SI->getCondition());
        if (SCValue.isUndefined()) {
          const Type *CondTy = SI->getCondition()->getType();
          SI->setCondition(Constant::getNullValue(CondTy));
          BranchesResolved = true;
          visit(SI);
        }
      }
    }

  return BranchesResolved;
}


namespace {
  Statistic<> NumInstRemoved("sccp", "Number of instructions removed");
  Statistic<> NumDeadBlocks ("sccp", "Number of basic blocks unreachable");

  //===--------------------------------------------------------------------===//
  //
  /// SCCP Class - This class uses the SCCPSolver to implement a per-function
  /// Sparse Conditional COnstant Propagator.
  ///
  struct SCCP : public FunctionPass {
    // runOnFunction - Run the Sparse Conditional Constant Propagation
    // algorithm, and return true if the function was modified.
    //
    bool runOnFunction(Function &F);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
    }
  };

  RegisterPass<SCCP> X("sccp", "Sparse Conditional Constant Propagation");
} // end anonymous namespace


// createSCCPPass - This is the public interface to this file...
FunctionPass *llvm::createSCCPPass() {
  return new SCCP();
}


// runOnFunction() - Run the Sparse Conditional Constant Propagation algorithm,
// and return true if the function was modified.
//
bool SCCP::runOnFunction(Function &F) {
  DEBUG(std::cerr << "SCCP on function '" << F.getName() << "'\n");
  SCCPSolver Solver;

  // Mark the first block of the function as being executable.
  Solver.MarkBlockExecutable(F.begin());

  // Mark all arguments to the function as being overdefined.
  hash_map<Value*, LatticeVal> &Values = Solver.getValueMapping();
  for (Function::arg_iterator AI = F.arg_begin(), E = F.arg_end(); AI != E; ++AI)
    Values[AI].markOverdefined();

  // Solve for constants.
  bool ResolvedBranches = true;
  while (ResolvedBranches) {
    Solver.Solve();
    DEBUG(std::cerr << "RESOLVING UNDEF BRANCHES\n");
    ResolvedBranches = Solver.ResolveBranchesIn(F);
  }

  bool MadeChanges = false;

  // If we decided that there are basic blocks that are dead in this function,
  // delete their contents now.  Note that we cannot actually delete the blocks,
  // as we cannot modify the CFG of the function.
  //
  std::set<BasicBlock*> &ExecutableBBs = Solver.getExecutableBlocks();
  for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB)
    if (!ExecutableBBs.count(BB)) {
      DEBUG(std::cerr << "  BasicBlock Dead:" << *BB);
      ++NumDeadBlocks;

      // Delete the instructions backwards, as it has a reduced likelihood of
      // having to update as many def-use and use-def chains.
      std::vector<Instruction*> Insts;
      for (BasicBlock::iterator I = BB->begin(), E = BB->getTerminator();
           I != E; ++I)
        Insts.push_back(I);
      while (!Insts.empty()) {
        Instruction *I = Insts.back();
        Insts.pop_back();
        if (!I->use_empty())
          I->replaceAllUsesWith(UndefValue::get(I->getType()));
        BB->getInstList().erase(I);
        MadeChanges = true;
        ++NumInstRemoved;
      }
    } else {
      // Iterate over all of the instructions in a function, replacing them with
      // constants if we have found them to be of constant values.
      //
      for (BasicBlock::iterator BI = BB->begin(), E = BB->end(); BI != E; ) {
        Instruction *Inst = BI++;
        if (Inst->getType() != Type::VoidTy) {
          LatticeVal &IV = Values[Inst];
          if (IV.isConstant() || IV.isUndefined() &&
              !isa<TerminatorInst>(Inst)) {
            Constant *Const = IV.isConstant()
              ? IV.getConstant() : UndefValue::get(Inst->getType());
            DEBUG(std::cerr << "  Constant: " << *Const << " = " << *Inst);

            // Replaces all of the uses of a variable with uses of the constant.
            Inst->replaceAllUsesWith(Const);

            // Delete the instruction.
            BB->getInstList().erase(Inst);

            // Hey, we just changed something!
            MadeChanges = true;
            ++NumInstRemoved;
          }
        }
      }
    }

  return MadeChanges;
}

namespace {
  Statistic<> IPNumInstRemoved("ipsccp", "Number of instructions removed");
  Statistic<> IPNumDeadBlocks ("ipsccp", "Number of basic blocks unreachable");
  Statistic<> IPNumArgsElimed ("ipsccp",
                               "Number of arguments constant propagated");
  Statistic<> IPNumGlobalConst("ipsccp",
                               "Number of globals found to be constant");

  //===--------------------------------------------------------------------===//
  //
  /// IPSCCP Class - This class implements interprocedural Sparse Conditional
  /// Constant Propagation.
  ///
  struct IPSCCP : public ModulePass {
    bool runOnModule(Module &M);
  };

  RegisterPass<IPSCCP>
  Y("ipsccp", "Interprocedural Sparse Conditional Constant Propagation");
} // end anonymous namespace

// createIPSCCPPass - This is the public interface to this file...
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
      CallSite CS = CallSite::get(cast<Instruction>(*UI));
      for (CallSite::arg_iterator AI = CS.arg_begin(),
             E = CS.arg_end(); AI != E; ++AI)
        if (*AI == GV)
          return true;
    } else if (LoadInst *LI = dyn_cast<LoadInst>(*UI)) {
      if (LI->isVolatile())
        return true;
    } else {
      return true;
    }
  return false;
}

bool IPSCCP::runOnModule(Module &M) {
  SCCPSolver Solver;

  // Loop over all functions, marking arguments to those with their addresses
  // taken or that are external as overdefined.
  //
  hash_map<Value*, LatticeVal> &Values = Solver.getValueMapping();
  for (Module::iterator F = M.begin(), E = M.end(); F != E; ++F)
    if (!F->hasInternalLinkage() || AddressIsTaken(F)) {
      if (!F->isExternal())
        Solver.MarkBlockExecutable(F->begin());
      for (Function::arg_iterator AI = F->arg_begin(), E = F->arg_end();
           AI != E; ++AI)
        Values[AI].markOverdefined();
    } else {
      Solver.AddTrackedFunction(F);
    }

  // Loop over global variables.  We inform the solver about any internal global
  // variables that do not have their 'addresses taken'.  If they don't have
  // their addresses taken, we can propagate constants through them.
  for (Module::global_iterator G = M.global_begin(), E = M.global_end();
       G != E; ++G)
    if (!G->isConstant() && G->hasInternalLinkage() && !AddressIsTaken(G))
      Solver.TrackValueOfGlobalVariable(G);

  // Solve for constants.
  bool ResolvedBranches = true;
  while (ResolvedBranches) {
    Solver.Solve();

    DEBUG(std::cerr << "RESOLVING UNDEF BRANCHES\n");
    ResolvedBranches = false;
    for (Module::iterator F = M.begin(), E = M.end(); F != E; ++F)
      ResolvedBranches |= Solver.ResolveBranchesIn(*F);
  }

  bool MadeChanges = false;

  // Iterate over all of the instructions in the module, replacing them with
  // constants if we have found them to be of constant values.
  //
  std::set<BasicBlock*> &ExecutableBBs = Solver.getExecutableBlocks();
  for (Module::iterator F = M.begin(), E = M.end(); F != E; ++F) {
    for (Function::arg_iterator AI = F->arg_begin(), E = F->arg_end();
         AI != E; ++AI)
      if (!AI->use_empty()) {
        LatticeVal &IV = Values[AI];
        if (IV.isConstant() || IV.isUndefined()) {
          Constant *CST = IV.isConstant() ?
            IV.getConstant() : UndefValue::get(AI->getType());
          DEBUG(std::cerr << "***  Arg " << *AI << " = " << *CST <<"\n");

          // Replaces all of the uses of a variable with uses of the
          // constant.
          AI->replaceAllUsesWith(CST);
          ++IPNumArgsElimed;
        }
      }

    std::vector<BasicBlock*> BlocksToErase;
    for (Function::iterator BB = F->begin(), E = F->end(); BB != E; ++BB)
      if (!ExecutableBBs.count(BB)) {
        DEBUG(std::cerr << "  BasicBlock Dead:" << *BB);
        ++IPNumDeadBlocks;

        // Delete the instructions backwards, as it has a reduced likelihood of
        // having to update as many def-use and use-def chains.
        std::vector<Instruction*> Insts;
        TerminatorInst *TI = BB->getTerminator();
        for (BasicBlock::iterator I = BB->begin(), E = TI; I != E; ++I)
          Insts.push_back(I);

        while (!Insts.empty()) {
          Instruction *I = Insts.back();
          Insts.pop_back();
          if (!I->use_empty())
            I->replaceAllUsesWith(UndefValue::get(I->getType()));
          BB->getInstList().erase(I);
          MadeChanges = true;
          ++IPNumInstRemoved;
        }

        for (unsigned i = 0, e = TI->getNumSuccessors(); i != e; ++i) {
          BasicBlock *Succ = TI->getSuccessor(i);
          if (Succ->begin() != Succ->end() && isa<PHINode>(Succ->begin()))
            TI->getSuccessor(i)->removePredecessor(BB);
        }
        if (!TI->use_empty())
          TI->replaceAllUsesWith(UndefValue::get(TI->getType()));
        BB->getInstList().erase(TI);

        if (&*BB != &F->front())
          BlocksToErase.push_back(BB);
        else
          new UnreachableInst(BB);

      } else {
        for (BasicBlock::iterator BI = BB->begin(), E = BB->end(); BI != E; ) {
          Instruction *Inst = BI++;
          if (Inst->getType() != Type::VoidTy) {
            LatticeVal &IV = Values[Inst];
            if (IV.isConstant() || IV.isUndefined() &&
                !isa<TerminatorInst>(Inst)) {
              Constant *Const = IV.isConstant()
                ? IV.getConstant() : UndefValue::get(Inst->getType());
              DEBUG(std::cerr << "  Constant: " << *Const << " = " << *Inst);

              // Replaces all of the uses of a variable with uses of the
              // constant.
              Inst->replaceAllUsesWith(Const);

              // Delete the instruction.
              if (!isa<TerminatorInst>(Inst) && !isa<CallInst>(Inst))
                BB->getInstList().erase(Inst);

              // Hey, we just changed something!
              MadeChanges = true;
              ++IPNumInstRemoved;
            }
          }
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
        assert(Folded && "Didn't fold away reference to block!");
      }

      // Finally, delete the basic block.
      F->getBasicBlockList().erase(DeadBB);
    }
  }

  // If we inferred constant or undef return values for a function, we replaced
  // all call uses with the inferred value.  This means we don't need to bother
  // actually returning anything from the function.  Replace all return
  // instructions with return undef.
  const hash_map<Function*, LatticeVal> &RV =Solver.getTrackedFunctionRetVals();
  for (hash_map<Function*, LatticeVal>::const_iterator I = RV.begin(),
         E = RV.end(); I != E; ++I)
    if (!I->second.isOverdefined() &&
        I->first->getReturnType() != Type::VoidTy) {
      Function *F = I->first;
      for (Function::iterator BB = F->begin(), E = F->end(); BB != E; ++BB)
        if (ReturnInst *RI = dyn_cast<ReturnInst>(BB->getTerminator()))
          if (!isa<UndefValue>(RI->getOperand(0)))
            RI->setOperand(0, UndefValue::get(F->getReturnType()));
    }

  // If we infered constant or undef values for globals variables, we can delete
  // the global and any stores that remain to it.
  const hash_map<GlobalVariable*, LatticeVal> &TG = Solver.getTrackedGlobals();
  for (hash_map<GlobalVariable*, LatticeVal>::const_iterator I = TG.begin(),
         E = TG.end(); I != E; ++I) {
    GlobalVariable *GV = I->first;
    assert(!I->second.isOverdefined() &&
           "Overdefined values should have been taken out of the map!");
    DEBUG(std::cerr << "Found that GV '" << GV->getName()<< "' is constant!\n");
    while (!GV->use_empty()) {
      StoreInst *SI = cast<StoreInst>(GV->use_back());
      SI->eraseFromParent();
    }
    M.getGlobalList().erase(GV);
    ++IPNumGlobalConst;
  }

  return MadeChanges;
}
