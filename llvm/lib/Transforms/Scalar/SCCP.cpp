//===- SCCP.cpp - Sparse Conditional Constant Propogation -----------------===//
//
// This file implements sparse conditional constant propogation and merging:
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
#include "llvm/ConstantHandling.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/Support/InstVisitor.h"
#include "Support/STLExtras.h"
#include "Support/Statistic.h"
#include <algorithm>
#include <set>

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

  inline Constant *getConstant() const { return ConstantVal; }
};

} // end anonymous namespace


//===----------------------------------------------------------------------===//
// SCCP Class
//
// This class does all of the work of Sparse Conditional Constant Propogation.
//
namespace {
class SCCP : public FunctionPass, public InstVisitor<SCCP> {
  std::set<BasicBlock*>     BBExecutable;// The basic blocks that are executable
  std::map<Value*, InstVal> ValueState;  // The state each value is in...

  std::vector<Instruction*> InstWorkList;// The instruction work list
  std::vector<BasicBlock*>  BBWorkList;  // The BasicBlock work list
public:

  // runOnFunction - Run the Sparse Conditional Constant Propogation algorithm,
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
  inline bool markConstant(Instruction *I, Constant *V) {
    if (ValueState[I].markConstant(V)) {
      DEBUG(std::cerr << "markConstant: " << V << " = " << I);
      InstWorkList.push_back(I);
      return true;
    }
    return false;
  }

  // markValueOverdefined - Make a value be marked as "overdefined". If the
  // value is not already overdefined, add it to the instruction work list so
  // that the users of the instruction are updated later.
  //
  inline bool markOverdefined(Value *V) {
    if (ValueState[V].markOverdefined()) {
      if (Instruction *I = dyn_cast<Instruction>(V)) {
	DEBUG(std::cerr << "markOverdefined: " << V);
	InstWorkList.push_back(I);  // Only instructions go on the work list
      }
      return true;
    }
    return false;
  }

  // getValueState - Return the InstVal object that corresponds to the value.
  // This function is neccesary because not all values should start out in the
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

  // markExecutable - Mark a basic block as executable, adding it to the BB 
  // work list if it is not already executable...
  // 
  void markExecutable(BasicBlock *BB) {
    if (BBExecutable.count(BB)) {
      // BB is already executable, but we may have just made an edge feasible
      // that wasn't before.  Add the PHI nodes to the work list so that they
      // can be rechecked.
      for (BasicBlock::iterator I = BB->begin();
           PHINode *PN = dyn_cast<PHINode>(I); ++I)
        visitPHINode(*PN);

    } else {
      DEBUG(std::cerr << "Marking BB Executable: " << *BB);
      BBExecutable.insert(BB);   // Basic block is executable!
      BBWorkList.push_back(BB);  // Add the block to the work list!
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
  void visitBinaryOperator(Instruction &I);
  void visitShiftInst(ShiftInst &I) { visitBinaryOperator(I); }

  // Instructions that cannot be folded away...
  void visitStoreInst     (Instruction &I) { /*returns void*/ }
  void visitLoadInst      (Instruction &I) { markOverdefined(&I); }
  void visitGetElementPtrInst(GetElementPtrInst &I);
  void visitCallInst      (Instruction &I) { markOverdefined(&I); }
  void visitInvokeInst    (Instruction &I) { markOverdefined(&I); }
  void visitAllocationInst(Instruction &I) { markOverdefined(&I); }
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
//
Pass *createSCCPPass() {
  return new SCCP();
}


//===----------------------------------------------------------------------===//
// SCCP Class Implementation


// runOnFunction() - Run the Sparse Conditional Constant Propogation algorithm,
// and return true if the function was modified.
//
bool SCCP::runOnFunction(Function &F) {
  // Mark the first block of the function as being executable...
  markExecutable(&F.front());

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
      if (BCValue.isOverdefined()) {
        // Overdefined condition variables mean the branch could go either way.
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
    if (SCValue.isOverdefined()) {  // Overdefined condition?
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
  TerminatorInst *FT = From->getTerminator();
  std::vector<bool> SuccFeasible;
  getFeasibleSuccessors(*FT, SuccFeasible);

  // Check all edges from From to To.  If any are feasible, return true.
  for (unsigned i = 0, e = SuccFeasible.size(); i != e; ++i)
    if (FT->getSuccessor(i) == To && SuccFeasible[i])
      return true;
    
  // Otherwise, none of the edges are actually feasible at this time...
  return false;
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
  if (getValueState(&PN).isOverdefined()) return;  // Quick exit

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
    if (IV.isOverdefined()) {   // PHI node becomes overdefined!
      markOverdefined(&PN);
      return;
    }
    
    if (isEdgeFeasible(PN.getIncomingBlock(i), PN.getParent())) {
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
          markOverdefined(&PN);         // The PHI node now becomes overdefined
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
    markConstant(&PN, OperandVal);      // Aquire operand value
}

void SCCP::visitTerminatorInst(TerminatorInst &TI) {
  std::vector<bool> SuccFeasible;
  getFeasibleSuccessors(TI, SuccFeasible);

  // Mark all feasible successors executable...
  for (unsigned i = 0, e = SuccFeasible.size(); i != e; ++i)
    if (SuccFeasible[i]) {
      BasicBlock *Succ = TI.getSuccessor(i);
      markExecutable(Succ);
    }
}

void SCCP::visitCastInst(CastInst &I) {
  Value *V = I.getOperand(0);
  InstVal &VState = getValueState(V);
  if (VState.isOverdefined()) {        // Inherit overdefinedness of operand
    markOverdefined(&I);
  } else if (VState.isConstant()) {    // Propagate constant value
    Constant *Result =
      ConstantFoldCastInstruction(VState.getConstant(), I.getType());

    if (Result) {
      // This instruction constant folds!
      markConstant(&I, Result);
    } else {
      markOverdefined(&I);   // Don't know how to fold this instruction.  :(
    }
  }
}

// Handle BinaryOperators and Shift Instructions...
void SCCP::visitBinaryOperator(Instruction &I) {
  InstVal &V1State = getValueState(I.getOperand(0));
  InstVal &V2State = getValueState(I.getOperand(1));
  if (V1State.isOverdefined() || V2State.isOverdefined()) {
    markOverdefined(&I);
  } else if (V1State.isConstant() && V2State.isConstant()) {
    Constant *Result = 0;
    if (isa<BinaryOperator>(I))
      Result = ConstantFoldBinaryInstruction(I.getOpcode(),
                                             V1State.getConstant(),
                                             V2State.getConstant());
    else if (isa<ShiftInst>(I))
      Result = ConstantFoldShiftInstruction(I.getOpcode(),
                                            V1State.getConstant(),
                                            V2State.getConstant());
    if (Result)
      markConstant(&I, Result);      // This instruction constant folds!
    else
      markOverdefined(&I);   // Don't know how to fold this instruction.  :(
  }
}

// Handle getelementptr instructions... if all operands are constants then we
// can turn this into a getelementptr ConstantExpr.
//
void SCCP::visitGetElementPtrInst(GetElementPtrInst &I) {
  std::vector<Constant*> Operands;
  Operands.reserve(I.getNumOperands());

  for (unsigned i = 0, e = I.getNumOperands(); i != e; ++i) {
    InstVal &State = getValueState(I.getOperand(i));
    if (State.isUndefined())
      return;  // Operands are not resolved yet...
    else if (State.isOverdefined()) {
      markOverdefined(&I);
      return;
    }
    assert(State.isConstant() && "Unknown state!");
    Operands.push_back(State.getConstant());
  }

  Constant *Ptr = Operands[0];
  Operands.erase(Operands.begin());  // Erase the pointer from idx list...

  markConstant(&I, ConstantExpr::getGetElementPtr(Ptr, Operands));  
}
