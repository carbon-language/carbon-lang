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
  Use Val;   // Will be null if returning void...
  ReturnInst(const ReturnInst &RI);
public:
  ReturnInst(Value *value = 0);
  inline ~ReturnInst() { dropAllReferences(); }

  virtual Instruction *clone() const { return new ReturnInst(*this); }

  virtual string getOpcode() const { return "ret"; }

  inline const Value *getReturnValue() const { return Val; }
  inline       Value *getReturnValue()       { return Val; }

  virtual void dropAllReferences();
  virtual const Value *getOperand(unsigned i) const {
    return (i == 0) ? Val : 0;
  }
  inline Value *getOperand(unsigned i) { return (i == 0) ? Val : 0;  }
  virtual bool setOperand(unsigned i, Value *Val);
  virtual unsigned getNumOperands() const { return Val != 0; }

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
  BasicBlockUse TrueDest, FalseDest;
  Use Condition;

  BranchInst(const BranchInst &BI);
public:
  // If cond = null, then is an unconditional br...
  BranchInst(BasicBlock *IfTrue, BasicBlock *IfFalse = 0, Value *cond = 0);
  inline ~BranchInst() { dropAllReferences(); }

  virtual Instruction *clone() const { return new BranchInst(*this); }

  virtual void dropAllReferences();
  
  inline bool isUnconditional() const {
    return Condition == 0 || !FalseDest;
  }

  virtual string getOpcode() const { return "br"; }

  inline Value *getOperand(unsigned i) {
    return (Value*)((const BranchInst *)this)->getOperand(i);
  }
  virtual const Value *getOperand(unsigned i) const;
  virtual bool setOperand(unsigned i, Value *Val);
  virtual unsigned getNumOperands() const { return isUnconditional() ? 1 : 3; }

  // Additionally, they must provide a method to get at the successors of this
  // terminator instruction.  If 'idx' is out of range, a null pointer shall be
  // returned.
  //
  virtual const BasicBlock *getSuccessor(unsigned idx) const;
  virtual unsigned getNumSuccessors() const { return 1+!isUnconditional(); }
};


//===---------------------------------------------------------------------------
// SwitchInst - Multiway switch
//
class SwitchInst : public TerminatorInst {
public:
  typedef pair<ConstPoolUse, BasicBlockUse> dest_value;
private:
  BasicBlockUse DefaultDest;
  Use Val;
  vector<dest_value> Destinations;

  SwitchInst(const SwitchInst &RI);
public:
  typedef vector<dest_value>::iterator       dest_iterator;
  typedef vector<dest_value>::const_iterator dest_const_iterator;

  SwitchInst(Value *Value, BasicBlock *Default);
  inline ~SwitchInst() { dropAllReferences(); }

  virtual Instruction *clone() const { return new SwitchInst(*this); }

  void dest_push_back(ConstPoolVal *OnVal, BasicBlock *Dest);

  virtual string getOpcode() const { return "switch"; }
  inline Value *getOperand(unsigned i) {
    return (Value*)((const SwitchInst*)this)->getOperand(i);
  }
  virtual const Value *getOperand(unsigned i) const;
  virtual bool setOperand(unsigned i, Value *Val);
  virtual unsigned getNumOperands() const;
  virtual void dropAllReferences();  

  // Additionally, they must provide a method to get at the successors of this
  // terminator instruction.  If 'idx' is out of range, a null pointer shall be
  // returned.
  //
  virtual const BasicBlock *getSuccessor(unsigned idx) const;
  virtual unsigned getNumSuccessors() const { return 1+Destinations.size(); }
};

#endif
