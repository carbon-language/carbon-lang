//===- SCCP.cpp - Sparse Conditional Constant Propogation -----------------===//
//
// This file implements sparse conditional constant propogation and merging:
//
// Specifically, this:
//   * Assumes values are constant unless proven otherwise
//   * Assumes BasicBlocks are dead unless proven otherwise
//   * Proves values to be constant, and replaces them with constants
//   . Proves conditional branches constant, and unconditionalizes them
//   * Folds multiple identical constants in the constant pool together
//
// Notice that:
//   * This pass has a habit of making definitions be dead.  It is a good idea
//     to to run a DCE pass sometime after running this pass.
//
//===----------------------------------------------------------------------===//

#include "llvm/Optimizations/ConstantProp.h"
#include "llvm/Optimizations/ConstantHandling.h"
#include "llvm/Method.h"
#include "llvm/BasicBlock.h"
#include "llvm/ConstPoolVals.h"
#include "llvm/InstrTypes.h"
#include "llvm/iOther.h"
#include "llvm/iMemory.h"
#include "llvm/iTerminators.h"
#include "llvm/Support/STLExtras.h"
#include "llvm/Assembly/Writer.h"
#include <algorithm>
#include <map>
#include <set>

// InstVal class - This class represents the different lattice values that an 
// instruction may occupy.  It is a simple class with value semantics.  The
// potential constant value that is pointed to is owned by the constant pool
// for the method being optimized.
//
class InstVal {
  enum { 
    Undefined,           // This instruction has no known value
    Constant,            // This instruction has a constant value
    // Range,            // This instruction is known to fall within a range
    Overdefined          // This instruction has an unknown value
  } LatticeValue;    // The current lattice position
  ConstPoolVal *ConstantVal;     // If Constant value, the current value
public:
  inline InstVal() : LatticeValue(Undefined), ConstantVal(0) {}

  // markOverdefined - Return true if this is a new status to be in...
  inline bool markOverdefined() {
    if (LatticeValue != Overdefined) {
      LatticeValue = Overdefined;
      return true;
    }
    return false;
  }

  // markConstant - Return true if this is a new status for us...
  inline bool markConstant(ConstPoolVal *V) {
    if (LatticeValue != Constant) {
      LatticeValue = Constant;
      ConstantVal = V;
      return true;
    } else {
      assert(ConstantVal == V && "Marking constant with different value");
    }
    return false;
  }

  inline bool isUndefined()   const { return LatticeValue == Undefined; }
  inline bool isConstant()    const { return LatticeValue == Constant; }
  inline bool isOverdefined() const { return LatticeValue == Overdefined; }

  inline ConstPoolVal *getConstant() const { return ConstantVal; }
};



//===----------------------------------------------------------------------===//
// SCCP Class
//
// This class does all of the work of Sparse Conditional Constant Propogation.
// It's public interface consists of a constructor and a doSCCP() method.
//
class SCCP {
  Method *M;                            // The method that we are working on...

  set<BasicBlock*>       BBExecutable;  // The basic blocks that are executable
  map<Value*, InstVal>   ValueState;    // The state each value is in...

  vector<Instruction*>   InstWorkList;  // The instruction work list
  vector<BasicBlock*>    BBWorkList;    // The BasicBlock work list

  //===--------------------------------------------------------------------===//
  // The public interface for this class
  //
public:

  // SCCP Ctor - Save the method to operate on...
  inline SCCP(Method *m) : M(m) {}

  // doSCCP() - Run the Sparse Conditional Constant Propogation algorithm, and 
  // return true if the method was modified.
  bool doSCCP();

  //===--------------------------------------------------------------------===//
  // The implementation of this class
  //
private:

  // markValueOverdefined - Make a value be marked as "constant".  If the value
  // is not already a constant, add it to the instruction work list so that 
  // the users of the instruction are updated later.
  //
  inline bool markConstant(Instruction *I, ConstPoolVal *V) {
    //cerr << "markConstant: " << V << " = " << I;
    if (ValueState[I].markConstant(V)) {
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
	//cerr << "markOverdefined: " << V;
	InstWorkList.push_back(I);  // Only instructions go on the work list
      }
      return true;
    }
    return false;
  }

  // getValueState - Return the InstVal object that corresponds to the value.
  // This function is neccesary because not all values should start out in the
  // underdefined state... MethodArgument's should be overdefined, and constants
  // should be marked as constants.  If a value is not known to be an
  // Instruction object, then use this accessor to get its value from the map.
  //
  inline InstVal &getValueState(Value *V) {
    map<Value*, InstVal>::iterator I = ValueState.find(V);
    if (I != ValueState.end()) return I->second;  // Common case, in the map
      
    if (ConstPoolVal *CPV = dyn_cast<ConstPoolVal>(V)) {//Constants are constant
      ValueState[CPV].markConstant(CPV);
    } else if (isa<MethodArgument>(V)) {          // MethodArgs are overdefined
      ValueState[V].markOverdefined();
    } 
    // All others are underdefined by default...
    return ValueState[V];
  }

  // markExecutable - Mark a basic block as executable, adding it to the BB 
  // work list if it is not already executable...
  // 
  void markExecutable(BasicBlock *BB) {
    if (BBExecutable.count(BB)) return;
    //cerr << "Marking BB Executable: " << BB;
    BBExecutable.insert(BB);   // Basic block is executable!
    BBWorkList.push_back(BB);  // Add the block to the work list!
  }


  // UpdateInstruction - Something changed in this instruction... Either an 
  // operand made a transition, or the instruction is newly executable.  Change
  // the value type of I to reflect these changes if appropriate.
  //
  void UpdateInstruction(Instruction *I);

  // OperandChangedState - This method is invoked on all of the users of an
  // instruction that was just changed state somehow....  Based on this
  // information, we need to update the specified user of this instruction.
  //
  void OperandChangedState(User *U);
};


//===----------------------------------------------------------------------===//
// SCCP Class Implementation


// doSCCP() - Run the Sparse Conditional Constant Propogation algorithm, and 
// return true if the method was modified.
//
bool SCCP::doSCCP() {
  // Mark the first block of the method as being executable...
  markExecutable(M->front());

  // Process the work lists until their are empty!
  while (!BBWorkList.empty() || !InstWorkList.empty()) {
    // Process the instruction work list...
    while (!InstWorkList.empty()) {
      Instruction *I = InstWorkList.back();
      InstWorkList.pop_back();

      //cerr << "\nPopped off I-WL: " << I;

      
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

      //cerr << "\nPopped off BBWL: " << BB;

      // If this block only has a single successor, mark it as executable as
      // well... if not, terminate the do loop.
      //
      if (BB->getTerminator()->getNumSuccessors() == 1)
	markExecutable(BB->getTerminator()->getSuccessor(0));

      // Loop over all of the instructions and notify them that they are newly
      // executable...
      for_each(BB->begin(), BB->end(),
	       bind_obj(this, &SCCP::UpdateInstruction));
    }
  }

#if 0
  for (Method::iterator BBI = M->begin(), BBEnd = M->end(); BBI != BBEnd; ++BBI)
    if (!BBExecutable.count(*BBI))
      cerr << "BasicBlock Dead:" << *BBI;
#endif


  // Iterate over all of the instructions in a method, replacing them with
  // constants if we have found them to be of constant values.
  //
  bool MadeChanges = false;
  for (Method::inst_iterator II = M->inst_begin(); II != M->inst_end(); ) {
    Instruction *Inst = *II;
    InstVal &IV = ValueState[Inst];
    if (IV.isConstant()) {
      ConstPoolVal *Const = IV.getConstant();
      // cerr << "Constant: " << Inst << "  is: " << Const;
      
      // Replaces all of the uses of a variable with uses of the constant.
      Inst->replaceAllUsesWith(Const);

      // Remove the operator from the list of definitions...
      Inst->getParent()->getInstList().remove(II.getInstructionIterator());
      
      // The new constant inherits the old name of the operator...
      if (Inst->hasName() && !Const->hasName())
	Const->setName(Inst->getName(), M->getSymbolTableSure());
  
      // Delete the operator now...
      delete Inst;

      // Incrementing the iterator in an unchecked manner could mess up the
      // internals of 'II'.  To make sure everything is happy, tell it we might
      // have broken it.
      II.resyncInstructionIterator();

      // Hey, we just changed something!
      MadeChanges = true;
      continue;   // Skip the ++II at the end of the loop here...
    } else if (Inst->isTerminator()) {
      MadeChanges |= opt::ConstantFoldTerminator(cast<TerminatorInst>(Inst));
    }

    ++II;
  }

  // Merge identical constants last: this is important because we may have just
  // introduced constants that already exist, and we don't want to pollute later
  // stages with extraneous constants.
  //
  return MadeChanges;
}


// UpdateInstruction - Something changed in this instruction... Either an 
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
void SCCP::UpdateInstruction(Instruction *I) {
  InstVal &IValue = ValueState[I];
  if (IValue.isOverdefined())
    return; // If already overdefined, we aren't going to effect anything

  switch (I->getOpcode()) {
    //===-----------------------------------------------------------------===//
    // Handle PHI nodes...
    //
  case Instruction::PHINode: {
    PHINode *PN = cast<PHINode>(I);
    unsigned NumValues = PN->getNumIncomingValues(), i;
    InstVal *OperandIV = 0;

    // Look at all of the executable operands of the PHI node.  If any of them
    // are overdefined, the PHI becomes overdefined as well.  If they are all
    // constant, and they agree with each other, the PHI becomes the identical
    // constant.  If they are constant and don't agree, the PHI is overdefined.
    // If there are no executable operands, the PHI remains undefined.
    //
    for (i = 0; i < NumValues; ++i) {
      if (BBExecutable.count(PN->getIncomingBlock(i))) {
	InstVal &IV = getValueState(PN->getIncomingValue(i));
	if (IV.isUndefined()) continue;  // Doesn't influence PHI node.
	if (IV.isOverdefined()) {   // PHI node becomes overdefined!
	  markOverdefined(PN);
	  return;
	}

	if (OperandIV == 0) {   // Grab the first value...
	  OperandIV = &IV;
	} else {                // Another value is being merged in!
	  // There is already a reachable operand.  If we conflict with it,
	  // then the PHI node becomes overdefined.  If we agree with it, we
	  // can continue on.
	  
	  // Check to see if there are two different constants merging...
	  if (IV.getConstant() != OperandIV->getConstant()) {
	    // Yes there is.  This means the PHI node is not constant.
	    // You must be overdefined poor PHI.
	    //
	    markOverdefined(I);         // The PHI node now becomes overdefined
	    return;    // I'm done analyzing you
	  }
	}
      }
    }

    // If we exited the loop, this means that the PHI node only has constant
    // arguments that agree with each other(and OperandIV is a pointer to one
    // of their InstVal's) or OperandIV is null because there are no defined
    // incoming arguments.  If this is the case, the PHI remains undefined.
    //
    if (OperandIV) {
      assert(OperandIV->isConstant() && "Should only be here for constants!");
      markConstant(I, OperandIV->getConstant());  // Aquire operand value
    }
    return;
  }

    //===-----------------------------------------------------------------===//
    // Handle instructions that unconditionally provide overdefined values...
    //
  case Instruction::Malloc:
  case Instruction::Free:
  case Instruction::Alloca:
  case Instruction::Load:
  case Instruction::Store:
    // TODO: getfield
  case Instruction::Call:
  case Instruction::Invoke:
    markOverdefined(I);          // Memory and call's are all overdefined
    return;

    //===-----------------------------------------------------------------===//
    // Handle Terminator instructions...
    //
  case Instruction::Ret: return;  // Method return doesn't affect anything
  case Instruction::Br: {        // Handle conditional branches...
    BranchInst *BI = cast<BranchInst>(I);
    if (BI->isUnconditional()) 
      return; // Unconditional branches are already handled!

    InstVal &BCValue = getValueState(BI->getCondition());
    if (BCValue.isOverdefined()) {
      // Overdefined condition variables mean the branch could go either way.
      markExecutable(BI->getSuccessor(0));
      markExecutable(BI->getSuccessor(1));
    } else if (BCValue.isConstant()) {
      // Constant condition variables mean the branch can only go a single way.
      ConstPoolBool *CPB = cast<ConstPoolBool>(BCValue.getConstant());
      if (CPB->getValue())       // If the branch condition is TRUE...
	markExecutable(BI->getSuccessor(0));
      else                       // Else if the br cond is FALSE...
	markExecutable(BI->getSuccessor(1));
    }
    return;
  }

  case Instruction::Switch: {
    SwitchInst *SI = cast<SwitchInst>(I);
    InstVal &SCValue = getValueState(SI->getCondition());
    if (SCValue.isOverdefined()) {  // Overdefined condition?  All dests are exe
      for(unsigned i = 0; BasicBlock *Succ = SI->getSuccessor(i); ++i)
	markExecutable(Succ);
    } else if (SCValue.isConstant()) {
      ConstPoolVal *CPV = SCValue.getConstant();
      // Make sure to skip the "default value" which isn't a value
      for (unsigned i = 1, E = SI->getNumSuccessors(); i != E; ++i) {
	if (SI->getSuccessorValue(i) == CPV) {// Found the right branch...
	  markExecutable(SI->getSuccessor(i));
	  return;
	}
      }
      
      // Constant value not equal to any of the branches... must execute 
      // default branch then...
      markExecutable(SI->getDefaultDest());
    }
    return;
  }

  default: break;  // Handle math operators as groups.
  } // end switch(I->getOpcode())

  
  //===-------------------------------------------------------------------===//
  // Handle Unary instructions...
  //   Also treated as unary here, are cast instructions and getelementptr
  //   instructions on struct* operands.
  //
  if (isa<UnaryOperator>(I) || isa<CastInst>(I) ||
      (isa<GetElementPtrInst>(I) &&
       cast<GetElementPtrInst>(I)->isStructSelector())) {

    Value *V = I->getOperand(0);
    InstVal &VState = getValueState(V);
    if (VState.isOverdefined()) {        // Inherit overdefinedness of operand
      markOverdefined(I);
    } else if (VState.isConstant()) {    // Propogate constant value
      ConstPoolVal *Result = 
	opt::ConstantFoldUnaryInstruction(I->getOpcode(), 
					  VState.getConstant());

      if (Result) {
	// This instruction constant folds!
	markConstant(I, Result);
      } else {
	markOverdefined(I);   // Don't know how to fold this instruction.  :(
      }
    }
    return;
  }

  //===-----------------------------------------------------------------===//
  // Handle Binary instructions...
  //
  if (isa<BinaryOperator>(I) || isa<ShiftInst>(I)) {
    Value *V1 = I->getOperand(0);
    Value *V2 = I->getOperand(1);

    InstVal &V1State = getValueState(V1);
    InstVal &V2State = getValueState(V2);
    if (V1State.isOverdefined() || V2State.isOverdefined()) {
      markOverdefined(I);
    } else if (V1State.isConstant() && V2State.isConstant()) {
      ConstPoolVal *Result = 
	opt::ConstantFoldBinaryInstruction(I->getOpcode(), 
					   V1State.getConstant(),
					   V2State.getConstant());
      if (Result) {
	// This instruction constant folds!
	markConstant(I, Result);
      } else {
	markOverdefined(I);   // Don't know how to fold this instruction.  :(
      }
    }
    return;
  }
  
  // Shouldn't get here... either the switch statement or one of the group
  // handlers should have kicked in...
  //
  cerr << "SCCP: Don't know how to handle: " << I;
  markOverdefined(I);   // Just in case
}



// OperandChangedState - This method is invoked on all of the users of an
// instruction that was just changed state somehow....  Based on this
// information, we need to update the specified user of this instruction.
//
void SCCP::OperandChangedState(User *U) {
  // Only instructions use other variable values!
  Instruction *I = cast<Instruction>(U);
  if (!BBExecutable.count(I->getParent())) return;  // Inst not executable yet!

  UpdateInstruction(I);
}


// DoSparseConditionalConstantProp - Use Sparse Conditional Constant Propogation
// to prove whether a value is constant and whether blocks are used.
//
bool opt::SCCPPass::doSCCP(Method *M) {
  if (M->isExternal()) return false;
  SCCP S(M);
  return S.doSCCP();
}
