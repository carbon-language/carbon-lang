//===-- llvm/iTerminators.h - Termintator instruction nodes ------*- C++ -*--=//
//
// This file contains the declarations for all the subclasses of the
// Instruction class, which is itself defined in the Instruction.h file.  In 
// between these definitions and the Instruction class are classes that expose
// the SSA properties of each instruction, and that form the SSA graph.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ITERMINATORS_H
#define LLVM_ITERMINATORS_H

#include "llvm/InstrTypes.h"
#include "llvm/BasicBlock.h"
#include "llvm/ConstPoolVals.h"

//===----------------------------------------------------------------------===//
//         Classes to represent Basic Block "Terminator" instructions
//===----------------------------------------------------------------------===//


//===---------------------------------------------------------------------------
// ReturnInst - Return a value (possibly void), from a method.  Execution does
//              not continue in this method any longer.
//
class ReturnInst : public TerminatorInst {
  ReturnInst(const ReturnInst &RI) : TerminatorInst(Instruction::Ret) {
    if (RI.Operands.size()) {
      assert(RI.Operands.size() == 1 && "Return insn can only have 1 operand!");
      Operands.reserve(1);
      Operands.push_back(Use(RI.Operands[0], this));
    }
  }
public:
  ReturnInst(Value *RetVal = 0) : TerminatorInst(Instruction::Ret) {
    if (RetVal) {
      Operands.reserve(1);
      Operands.push_back(Use(RetVal, this));
    }
  }
  inline ~ReturnInst() { dropAllReferences(); }

  virtual Instruction *clone() const { return new ReturnInst(*this); }

  virtual string getOpcode() const { return "ret"; }

  inline const Value *getReturnValue() const {
    return Operands.size() ? Operands[0] : 0; 
  }
  inline       Value *getReturnValue()       {
    return Operands.size() ? Operands[0] : 0;
  }

  // Additionally, they must provide a method to get at the successors of this
  // terminator instruction.  If 'idx' is out of range, a null pointer shall be
  // returned.
  //
  virtual const BasicBlock *getSuccessor(unsigned idx) const { return 0; }
  virtual unsigned getNumSuccessors() const { return 0; }
};


//===---------------------------------------------------------------------------
// BranchInst - Conditional or Unconditional Branch instruction.
//
class BranchInst : public TerminatorInst {
  BranchInst(const BranchInst &BI);
public:
  // If cond = null, then is an unconditional br...
  BranchInst(BasicBlock *IfTrue, BasicBlock *IfFalse = 0, Value *cond = 0);
  inline ~BranchInst() { dropAllReferences(); }

  virtual Instruction *clone() const { return new BranchInst(*this); }

  inline bool isUnconditional() const {
    return Operands.size() == 1;
  }

  inline const Value *getCondition() const {
    return isUnconditional() ? 0 : Operands[2];
  }
  inline       Value *getCondition()       {
    return isUnconditional() ? 0 : Operands[2];
  }

  virtual string getOpcode() const { return "br"; }

  // setUnconditionalDest - Change the current branch to an unconditional branch
  // targeting the specified block.
  //
  void setUnconditionalDest(BasicBlock *Dest) {
    if (Operands.size() == 3)
      Operands.erase(Operands.begin()+1, Operands.end());
    Operands[0] = Dest;
  }

  // Additionally, they must provide a method to get at the successors of this
  // terminator instruction.
  //
  virtual const BasicBlock *getSuccessor(unsigned i) const {
    return (i == 0) ? Operands[0]->castBasicBlockAsserting() : 
          ((i == 1 && Operands.size() > 1) 
               ? Operands[1]->castBasicBlockAsserting() : 0);
  }
  inline BasicBlock *getSuccessor(unsigned idx) {
    return (BasicBlock*)((const BranchInst *)this)->getSuccessor(idx);
  }

  virtual unsigned getNumSuccessors() const { return 1+!isUnconditional(); }
};


//===---------------------------------------------------------------------------
// SwitchInst - Multiway switch
//
class SwitchInst : public TerminatorInst {
  // Operand[0] = Value to switch on
  // Operand[1] = Default basic block destination
  SwitchInst(const SwitchInst &RI);
public:
  //typedef vector<dest_value>::iterator       dest_iterator;
  //typedef vector<dest_value>::const_iterator dest_const_iterator;

  SwitchInst(Value *Value, BasicBlock *Default);
  inline ~SwitchInst() { dropAllReferences(); }

  virtual Instruction *clone() const { return new SwitchInst(*this); }

  // Accessor Methods for Switch stmt
  //
  /*
  inline dest_iterator dest_begin() { return Destinations.begin(); }
  inline dest_iterator dest_end  () { return Destinations.end(); }
  inline dest_const_iterator dest_begin() const { return Destinations.begin(); }
  inline dest_const_iterator dest_end  () const { return Destinations.end(); }
  */

  inline const Value *getCondition() const { return Operands[0]; }
  inline       Value *getCondition()       { return Operands[0]; }
  inline const BasicBlock *getDefaultDest() const {
    return Operands[1]->castBasicBlockAsserting();
  }
  inline       BasicBlock *getDefaultDest()       {
    return Operands[1]->castBasicBlockAsserting();
  }

  void dest_push_back(ConstPoolVal *OnVal, BasicBlock *Dest);

  virtual string getOpcode() const { return "switch"; }

  // Additionally, they must provide a method to get at the successors of this
  // terminator instruction.  If 'idx' is out of range, a null pointer shall be
  // returned.
  //
  virtual const BasicBlock *getSuccessor(unsigned idx) const {
    if (idx >= Operands.size()/2) return 0;
    return Operands[idx*2+1]->castBasicBlockAsserting();
  }
  inline BasicBlock *getSuccessor(unsigned idx) {
    if (idx >= Operands.size()/2) return 0;
    return Operands[idx*2+1]->castBasicBlockAsserting();
  }

  // getSuccessorValue - Return the value associated with the specified successor
  // WARNING: This does not gracefully accept idx's out of range!
  inline const ConstPoolVal *getSuccessorValue(unsigned idx) const {
    assert(idx < getNumSuccessors() && "Successor # out of range!");
    return Operands[idx*2]->castConstantAsserting();
  }
  inline ConstPoolVal *getSuccessorValue(unsigned idx) {
    assert(idx < getNumSuccessors() && "Successor # out of range!");
    return Operands[idx*2]->castConstantAsserting();
  }
  virtual unsigned getNumSuccessors() const { return Operands.size()/2; }
};

#endif
