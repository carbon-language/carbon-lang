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

#include "llvm/Transforms/Scalar.h"
#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/Type.h"
#include "llvm/Support/InstVisitor.h"
#include "Support/Debug.h"
#include "Support/Statistic.h"
#include "Support/STLExtras.h"
#include <algorithm>
#include <set>
using namespace llvm;

// InstVal class - This class represents the different lattice values that an 
// instruction may occupy.  It is a simple class with value semantics.
//
namespace {
  Statistic<> NumInstRemoved("sccp", "Number of instructions removed");

class InstVal {
  enum { 
    undefined,           // This instruction has no known value
    constant,            // This instruction has a constant value
    overdefined          // This instruction has an unknown value
  } LatticeValue;        // The current lattice position
  Constant *ConstantVal; // If Constant value, the current value
public:
  inline InstVal() : LatticeValue(undefined), ConstantVal(0) {}

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
// SCCP Class
//
// This class does all of the work of Sparse Conditional Constant Propagation.
//
namespace {
class SCCP : public FunctionPass, public InstVisitor<SCCP> {
  std::set<BasicBlock*>     BBExecutable;// The basic blocks that are executable
  std::map<Value*, InstVal> ValueState;  // The state each value is in...

  std::vector<Instruction*> InstWorkList;// The instruction work list
  std::vector<BasicBlock*>  BBWorkList;  // The BasicBlock work list

  /// UsersOfOverdefinedPHIs - Keep track of any users of PHI nodes that are not
  /// overdefined, despite the fact that the PHI node is overdefined.
  std::multimap<PHINode*, Instruction*> UsersOfOverdefinedPHIs;

  /// KnownFeasibleEdges - Entries in this set are edges which have already had
  /// PHI nodes retriggered.
  typedef std::pair<BasicBlock*,BasicBlock*> Edge;
  std::set<Edge> KnownFeasibleEdges;
public:

  // runOnFunction - Run the Sparse Conditional Constant Propagation algorithm,
  // and return true if the function was modified.
  //
  bool runOnFunction(Function &F);

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesCFG();
  }


  //===--------------------------------------------------------------------===//
  // The implementation of this class
  //
private:
  friend class InstVisitor<SCCP>;        // Allow callbacks from visitor

  // markValueOverdefined - Make a value be marked as "constant".  If the value
  // is not already a constant, add it to the instruction work list so that 
  // the users of the instruction are updated later.
  //
  inline void markConstant(InstVal &IV, Instruction *I, Constant *C) {
    if (IV.markConstant(C)) {
      DEBUG(std::cerr << "markConstant: " << *C << ": " << *I);
      InstWorkList.push_back(I);
    }
  }
  inline void markConstant(Instruction *I, Constant *C) {
    markConstant(ValueState[I], I, C);
  }

  // markValueOverdefined - Make a value be marked as "overdefined". If the
  // value is not already overdefined, add it to the instruction work list so
  // that the users of the instruction are updated later.
  //
  inline void markOverdefined(InstVal &IV, Instruction *I) {
    if (IV.markOverdefined()) {
      DEBUG(std::cerr << "markOverdefined: " << *I);
      InstWorkList.push_back(I);  // Only instructions go on the work list
    }
  }
  inline void markOverdefined(Instruction *I) {
    markOverdefined(ValueState[I], I);
  }

  // getValueState - Return the InstVal object that corresponds to the value.
  // This function is necessary because not all values should start out in the
  // underdefined state... Argument's should be overdefined, and
  // constants should be marked as constants.  If a value is not known to be an
  // Instruction object, then use this accessor to get its value from the map.
  //
  inline InstVal &getValueState(Value *V) {
    std::map<Value*, InstVal>::iterator I = ValueState.find(V);
    if (I != ValueState.end()) return I->second;  // Common case, in the map
      
    if (Constant *CPV = dyn_cast<Constant>(V)) {  // Constants are constant
      ValueState[CPV].markConstant(CPV);
    } else if (isa<Argument>(V)) {                // Arguments are overdefined
      ValueState[V].markOverdefined();
    } else if (GlobalValue *GV = dyn_cast<GlobalValue>(V)) {
      // The address of a global is a constant...
      ValueState[V].markConstant(ConstantPointerRef::get(GV));
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
      for (BasicBlock::iterator I = Dest->begin();
           PHINode *PN = dyn_cast<PHINode>(I); ++I)
        visitPHINode(*PN);

    } else {
      DEBUG(std::cerr << "Marking Block Executable: " << Dest->getName()<<"\n");
      BBExecutable.insert(Dest);   // Basic block is executable!
      BBWorkList.push_back(Dest);  // Add the block to the work list!
    }
  }


  // visit implementations - Something changed in this instruction... Either an 
  // operand made a transition, or the instruction is newly executable.  Change
  // the value type of I to reflect these changes if appropriate.
  //
  void visitPHINode(PHINode &I);

  // Terminators
  void visitReturnInst(ReturnInst &I) { /*does not have an effect*/ }
  void visitTerminatorInst(TerminatorInst &TI);

  void visitCastInst(CastInst &I);
  void visitSelectInst(SelectInst &I);
  void visitBinaryOperator(Instruction &I);
  void visitShiftInst(ShiftInst &I) { visitBinaryOperator(I); }

  // Instructions that cannot be folded away...
  void visitStoreInst     (Instruction &I) { /*returns void*/ }
  void visitLoadInst      (LoadInst &I);
  void visitGetElementPtrInst(GetElementPtrInst &I);
  void visitCallInst      (Instruction &I) { markOverdefined(&I); }
  void visitInvokeInst    (TerminatorInst &I) {
    if (I.getType() != Type::VoidTy) markOverdefined(&I);
    visitTerminatorInst(I);
  }
  void visitUnwindInst    (TerminatorInst &I) { /*returns void*/ }
  void visitAllocationInst(Instruction &I) { markOverdefined(&I); }
  void visitVANextInst    (Instruction &I) { markOverdefined(&I); }
  void visitVAArgInst     (Instruction &I) { markOverdefined(&I); }
  void visitFreeInst      (Instruction &I) { /*returns void*/ }

  void visitInstruction(Instruction &I) {
    // If a new instruction is added to LLVM that we don't handle...
    std::cerr << "SCCP: Don't know how to handle: " << I;
    markOverdefined(&I);   // Just in case
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
};

  RegisterOpt<SCCP> X("sccp", "Sparse Conditional Constant Propagation");
} // end anonymous namespace


// createSCCPPass - This is the public interface to this file...
Pass *llvm::createSCCPPass() {
  return new SCCP();
}


//===----------------------------------------------------------------------===//
// SCCP Class Implementation


// runOnFunction() - Run the Sparse Conditional Constant Propagation algorithm,
// and return true if the function was modified.
//
bool SCCP::runOnFunction(Function &F) {
  // Mark the first block of the function as being executable...
  BBExecutable.insert(F.begin());   // Basic block is executable!
  BBWorkList.push_back(F.begin());  // Add the block to the work list!

  // Process the work lists until their are empty!
  while (!BBWorkList.empty() || !InstWorkList.empty()) {
    // Process the instruction work list...
    while (!InstWorkList.empty()) {
      Instruction *I = InstWorkList.back();
      InstWorkList.pop_back();

      DEBUG(std::cerr << "\nPopped off I-WL: " << I);
      
      // "I" got into the work list because it either made the transition from
      // bottom to constant, or to Overdefined.
      //
      // Update all of the users of this instruction's value...
      //
      for_each(I->use_begin(), I->use_end(),
	       bind_obj(this, &SCCP::OperandChangedState));
    }

    // Process the basic block work list...
    while (!BBWorkList.empty()) {
      BasicBlock *BB = BBWorkList.back();
      BBWorkList.pop_back();

      DEBUG(std::cerr << "\nPopped off BBWL: " << BB);

      // Notify all instructions in this basic block that they are newly
      // executable.
      visit(BB);
    }
  }

  if (DebugFlag) {
    for (Function::iterator I = F.begin(), E = F.end(); I != E; ++I)
      if (!BBExecutable.count(I))
        std::cerr << "BasicBlock Dead:" << *I;
  }

  // Iterate over all of the instructions in a function, replacing them with
  // constants if we have found them to be of constant values.
  //
  bool MadeChanges = false;
  for (Function::iterator BB = F.begin(), BBE = F.end(); BB != BBE; ++BB)
    for (BasicBlock::iterator BI = BB->begin(); BI != BB->end();) {
      Instruction &Inst = *BI;
      InstVal &IV = ValueState[&Inst];
      if (IV.isConstant()) {
        Constant *Const = IV.getConstant();
        DEBUG(std::cerr << "Constant: " << Const << " = " << Inst);

        // Replaces all of the uses of a variable with uses of the constant.
        Inst.replaceAllUsesWith(Const);

        // Remove the operator from the list of definitions... and delete it.
        BI = BB->getInstList().erase(BI);

        // Hey, we just changed something!
        MadeChanges = true;
        ++NumInstRemoved;
      } else {
        ++BI;
      }
    }

  // Reset state so that the next invocation will have empty data structures
  BBExecutable.clear();
  ValueState.clear();
  std::vector<Instruction*>().swap(InstWorkList);
  std::vector<BasicBlock*>().swap(BBWorkList);

  return MadeChanges;
}


// getFeasibleSuccessors - Return a vector of booleans to indicate which
// successors are reachable from a given terminator instruction.
//
void SCCP::getFeasibleSuccessors(TerminatorInst &TI, std::vector<bool> &Succs) {
  Succs.resize(TI.getNumSuccessors());
  if (BranchInst *BI = dyn_cast<BranchInst>(&TI)) {
    if (BI->isUnconditional()) {
      Succs[0] = true;
    } else {
      InstVal &BCValue = getValueState(BI->getCondition());
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
    InstVal &SCValue = getValueState(SI->getCondition());
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
bool SCCP::isEdgeFeasible(BasicBlock *From, BasicBlock *To) {
  assert(BBExecutable.count(To) && "Dest should always be alive!");

  // Make sure the source basic block is executable!!
  if (!BBExecutable.count(From)) return false;
  
  // Check to make sure this edge itself is actually feasible now...
  TerminatorInst *TI = From->getTerminator();
  if (BranchInst *BI = dyn_cast<BranchInst>(TI)) {
    if (BI->isUnconditional())
      return true;
    else {
      InstVal &BCValue = getValueState(BI->getCondition());
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
    InstVal &SCValue = getValueState(SI->getCondition());
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
void SCCP::visitPHINode(PHINode &PN) {
  InstVal &PNIV = getValueState(&PN);
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

  // Look at all of the executable operands of the PHI node.  If any of them
  // are overdefined, the PHI becomes overdefined as well.  If they are all
  // constant, and they agree with each other, the PHI becomes the identical
  // constant.  If they are constant and don't agree, the PHI is overdefined.
  // If there are no executable operands, the PHI remains undefined.
  //
  Constant *OperandVal = 0;
  for (unsigned i = 0, e = PN.getNumIncomingValues(); i != e; ++i) {
    InstVal &IV = getValueState(PN.getIncomingValue(i));
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

void SCCP::visitTerminatorInst(TerminatorInst &TI) {
  std::vector<bool> SuccFeasible;
  getFeasibleSuccessors(TI, SuccFeasible);

  BasicBlock *BB = TI.getParent();

  // Mark all feasible successors executable...
  for (unsigned i = 0, e = SuccFeasible.size(); i != e; ++i)
    if (SuccFeasible[i])
      markEdgeExecutable(BB, TI.getSuccessor(i));
}

void SCCP::visitCastInst(CastInst &I) {
  Value *V = I.getOperand(0);
  InstVal &VState = getValueState(V);
  if (VState.isOverdefined())          // Inherit overdefinedness of operand
    markOverdefined(&I);
  else if (VState.isConstant())        // Propagate constant value
    markConstant(&I, ConstantExpr::getCast(VState.getConstant(), I.getType()));
}

void SCCP::visitSelectInst(SelectInst &I) {
  InstVal &CondValue = getValueState(I.getCondition());
  if (CondValue.isOverdefined())
    markOverdefined(&I);
  else if (CondValue.isConstant()) {
    if (CondValue.getConstant() == ConstantBool::True) {
      InstVal &Val = getValueState(I.getTrueValue());
      if (Val.isOverdefined())
        markOverdefined(&I);
      else if (Val.isConstant())
        markConstant(&I, Val.getConstant());
    } else if (CondValue.getConstant() == ConstantBool::False) {
      InstVal &Val = getValueState(I.getFalseValue());
      if (Val.isOverdefined())
        markOverdefined(&I);
      else if (Val.isConstant())
        markConstant(&I, Val.getConstant());
    } else
      markOverdefined(&I);
  }
}

// Handle BinaryOperators and Shift Instructions...
void SCCP::visitBinaryOperator(Instruction &I) {
  InstVal &IV = ValueState[&I];
  if (IV.isOverdefined()) return;

  InstVal &V1State = getValueState(I.getOperand(0));
  InstVal &V2State = getValueState(I.getOperand(1));

  if (V1State.isOverdefined() || V2State.isOverdefined()) {
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
          InstVal Result;
          for (unsigned i = 0, e = PN1->getNumIncomingValues(); i != e; ++i) {
            InstVal &In1 = getValueState(PN1->getIncomingValue(i));
            BasicBlock *InBlock = PN1->getIncomingBlock(i);
            InstVal &In2 =getValueState(PN2->getIncomingValueForBlock(InBlock));

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

// Handle getelementptr instructions... if all operands are constants then we
// can turn this into a getelementptr ConstantExpr.
//
void SCCP::visitGetElementPtrInst(GetElementPtrInst &I) {
  InstVal &IV = ValueState[&I];
  if (IV.isOverdefined()) return;

  std::vector<Constant*> Operands;
  Operands.reserve(I.getNumOperands());

  for (unsigned i = 0, e = I.getNumOperands(); i != e; ++i) {
    InstVal &State = getValueState(I.getOperand(i));
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

/// GetGEPGlobalInitializer - Given a constant and a getelementptr constantexpr,
/// return the constant value being addressed by the constant expression, or
/// null if something is funny.
///
static Constant *GetGEPGlobalInitializer(Constant *C, ConstantExpr *CE) {
  if (CE->getOperand(1) != Constant::getNullValue(Type::LongTy))
    return 0;  // Do not allow stepping over the value!

  // Loop over all of the operands, tracking down which value we are
  // addressing...
  for (unsigned i = 2, e = CE->getNumOperands(); i != e; ++i)
    if (ConstantUInt *CU = dyn_cast<ConstantUInt>(CE->getOperand(i))) {
      ConstantStruct *CS = dyn_cast<ConstantStruct>(C);
      if (CS == 0) return 0;
      if (CU->getValue() >= CS->getValues().size()) return 0;
      C = cast<Constant>(CS->getValues()[CU->getValue()]);
    } else if (ConstantSInt *CS = dyn_cast<ConstantSInt>(CE->getOperand(i))) {
      ConstantArray *CA = dyn_cast<ConstantArray>(C);
      if (CA == 0) return 0;
      if ((uint64_t)CS->getValue() >= CA->getValues().size()) return 0;
      C = cast<Constant>(CA->getValues()[CS->getValue()]);
    } else
      return 0;
  return C;
}

// Handle load instructions.  If the operand is a constant pointer to a constant
// global, we can replace the load with the loaded constant value!
void SCCP::visitLoadInst(LoadInst &I) {
  InstVal &IV = ValueState[&I];
  if (IV.isOverdefined()) return;

  InstVal &PtrVal = getValueState(I.getOperand(0));
  if (PtrVal.isUndefined()) return;   // The pointer is not resolved yet!
  if (PtrVal.isConstant() && !I.isVolatile()) {
    Value *Ptr = PtrVal.getConstant();
    if (isa<ConstantPointerNull>(Ptr)) {
      // load null -> null
      markConstant(IV, &I, Constant::getNullValue(I.getType()));
      return;
    }
      
    if (ConstantPointerRef *CPR = dyn_cast<ConstantPointerRef>(Ptr))
      Ptr = CPR->getValue();

    // Transform load (constant global) into the value loaded.
    if (GlobalVariable *GV = dyn_cast<GlobalVariable>(Ptr))
      if (GV->isConstant() && !GV->isExternal()) {
        markConstant(IV, &I, GV->getInitializer());
        return;
      }

    // Transform load (constantexpr_GEP global, 0, ...) into the value loaded.
    if (ConstantExpr *CE = dyn_cast<ConstantExpr>(Ptr))
      if (CE->getOpcode() == Instruction::GetElementPtr)
        if (ConstantPointerRef *G
            = dyn_cast<ConstantPointerRef>(CE->getOperand(0)))
          if (GlobalVariable *GV = dyn_cast<GlobalVariable>(G->getValue()))
            if (GV->isConstant() && !GV->isExternal())
              if (Constant *V =
                  GetGEPGlobalInitializer(GV->getInitializer(), CE)) {
                markConstant(IV, &I, V);
                return;
              }
  }

  // Otherwise we cannot say for certain what value this load will produce.
  // Bail out.
  markOverdefined(IV, &I);
}
