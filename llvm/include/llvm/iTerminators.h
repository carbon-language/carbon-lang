//===-- llvm/iTerminators.h - Termintator instruction nodes -----*- C++ -*-===//
//
// This file contains the declarations for all the subclasses of the Instruction
// class which represent "terminator" instructions.  Terminator instructions are
// the only instructions allowed and required to terminate a BasicBlock.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ITERMINATORS_H
#define LLVM_ITERMINATORS_H

#include "llvm/InstrTypes.h"

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
  ReturnInst(Value *RetVal = 0, Instruction *InsertBefore = 0)
    : TerminatorInst(Instruction::Ret, InsertBefore) {
    if (RetVal) {
      Operands.reserve(1);
      Operands.push_back(Use(RetVal, this));
    }
  }

  virtual Instruction *clone() const { return new ReturnInst(*this); }

  inline const Value *getReturnValue() const {
    return Operands.size() ? Operands[0].get() : 0; 
  }
  inline       Value *getReturnValue()       {
    return Operands.size() ? Operands[0].get() : 0;
  }

  virtual const BasicBlock *getSuccessor(unsigned idx) const {
    assert(0 && "ReturnInst has no successors!");
    abort();
    return 0;
  }
  virtual void setSuccessor(unsigned idx, BasicBlock *NewSucc) {
    assert(0 && "ReturnInst has no successors!");
  }
  virtual unsigned getNumSuccessors() const { return 0; }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ReturnInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return (I->getOpcode() == Instruction::Ret);
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};


//===---------------------------------------------------------------------------
// BranchInst - Conditional or Unconditional Branch instruction.
//
class BranchInst : public TerminatorInst {
  BranchInst(const BranchInst &BI);
public:
  // If cond = null, then is an unconditional br...
  BranchInst(BasicBlock *IfTrue, BasicBlock *IfFalse, Value *cond = 0,
             Instruction *InsertBefore = 0);
  BranchInst(BasicBlock *IfTrue, Instruction *InsertBefore = 0);

  virtual Instruction *clone() const { return new BranchInst(*this); }

  inline bool isUnconditional() const { return Operands.size() == 1; }
  inline bool isConditional()   const { return Operands.size() == 3; }

  inline const Value *getCondition() const {
    return isUnconditional() ? 0 : Operands[2].get();
  }
  Value *getCondition() {
    return isUnconditional() ? 0 : Operands[2].get();
  }

  // setUnconditionalDest - Change the current branch to an unconditional branch
  // targeting the specified block.
  //
  void setUnconditionalDest(BasicBlock *Dest) {
    if (isConditional()) Operands.erase(Operands.begin()+1, Operands.end());
    Operands[0] = (Value*)Dest;
  }

  virtual const BasicBlock *getSuccessor(unsigned i) const {
    assert(i < getNumSuccessors() && "Successor # out of range for Branch!");
    return (i == 0) ? cast<BasicBlock>(Operands[0].get()) : 
                      cast<BasicBlock>(Operands[1].get());
  }
  inline BasicBlock *getSuccessor(unsigned idx) {
    return (BasicBlock*)((const BranchInst *)this)->getSuccessor(idx);
  }

  virtual void setSuccessor(unsigned idx, BasicBlock *NewSucc) {
    assert(idx < getNumSuccessors() && "Successor # out of range for Branch!");
    Operands[idx] = (Value*)NewSucc;
  }

  virtual unsigned getNumSuccessors() const { return 1+isConditional(); }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const BranchInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return (I->getOpcode() == Instruction::Br);
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};


//===---------------------------------------------------------------------------
// SwitchInst - Multiway switch
//
class SwitchInst : public TerminatorInst {
  // Operand[0]    = Value to switch on
  // Operand[1]    = Default basic block destination
  // Operand[2n  ] = Value to match
  // Operand[2n+1] = BasicBlock to go to on match
  SwitchInst(const SwitchInst &RI);
public:
  SwitchInst(Value *Value, BasicBlock *Default, Instruction *InsertBefore = 0);

  virtual Instruction *clone() const { return new SwitchInst(*this); }

  // Accessor Methods for Switch stmt
  //
  inline const Value *getCondition() const { return Operands[0]; }
  inline       Value *getCondition()       { return Operands[0]; }
  inline const BasicBlock *getDefaultDest() const {
    return cast<BasicBlock>(Operands[1].get());
  }
  inline       BasicBlock *getDefaultDest()       {
    return cast<BasicBlock>(Operands[1].get());
  }

  void dest_push_back(Constant *OnVal, BasicBlock *Dest);

  virtual const BasicBlock *getSuccessor(unsigned idx) const {
    assert(idx < getNumSuccessors() &&"Successor idx out of range for switch!");
    return cast<BasicBlock>(Operands[idx*2+1].get());
  }
  inline BasicBlock *getSuccessor(unsigned idx) {
    assert(idx < getNumSuccessors() &&"Successor idx out of range for switch!");
    return cast<BasicBlock>(Operands[idx*2+1].get());
  }

  virtual void setSuccessor(unsigned idx, BasicBlock *NewSucc) {
    assert(idx < getNumSuccessors() && "Successor # out of range for switch!");
    Operands[idx*2+1] = (Value*)NewSucc;
  }

  // getSuccessorValue - Return the value associated with the specified
  // successor.
  inline const Constant *getSuccessorValue(unsigned idx) const {
    assert(idx < getNumSuccessors() && "Successor # out of range!");
    return cast<Constant>(Operands[idx*2].get());
  }
  inline Constant *getSuccessorValue(unsigned idx) {
    assert(idx < getNumSuccessors() && "Successor # out of range!");
    return cast<Constant>(Operands[idx*2].get());
  }
  virtual unsigned getNumSuccessors() const { return Operands.size()/2; }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const SwitchInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return (I->getOpcode() == Instruction::Switch);
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};


//===---------------------------------------------------------------------------
// InvokeInst - Invoke instruction
//
class InvokeInst : public TerminatorInst {
  InvokeInst(const InvokeInst &BI);
public:
  InvokeInst(Value *Meth, BasicBlock *IfNormal, BasicBlock *IfException,
	     const std::vector<Value*> &Params, const std::string &Name = "",
             Instruction *InsertBefore = 0);

  virtual Instruction *clone() const { return new InvokeInst(*this); }

  bool mayWriteToMemory() const { return true; }

  // getCalledFunction - Return the function called, or null if this is an
  // indirect function invocation...
  //
  inline const Function *getCalledFunction() const {
    return dyn_cast<Function>(Operands[0].get());
  }
  inline Function *getCalledFunction() {
    return dyn_cast<Function>(Operands[0].get());
  }

  // getCalledValue - Get a pointer to a method that is invoked by this inst.
  inline const Value *getCalledValue() const { return Operands[0]; }
  inline       Value *getCalledValue()       { return Operands[0]; }

  // get*Dest - Return the destination basic blocks...
  inline const BasicBlock *getNormalDest() const {
    return cast<BasicBlock>(Operands[1].get());
  }
  inline       BasicBlock *getNormalDest() {
    return cast<BasicBlock>(Operands[1].get());
  }
  inline const BasicBlock *getExceptionalDest() const {
    return cast<BasicBlock>(Operands[2].get());
  }
  inline       BasicBlock *getExceptionalDest() {
    return cast<BasicBlock>(Operands[2].get());
  }

  inline void setNormalDest(BasicBlock *B){
    Operands[1] = (Value*)B;
  }

  inline void setExceptionalDest(BasicBlock *B){
    Operands[2] = (Value*)B;
  }

  virtual const BasicBlock *getSuccessor(unsigned i) const {
    assert(i < 2 && "Successor # out of range for invoke!");
    return i == 0 ? getNormalDest() : getExceptionalDest();
  }
  inline BasicBlock *getSuccessor(unsigned i) {
    assert(i < 2 && "Successor # out of range for invoke!");
    return i == 0 ? getNormalDest() : getExceptionalDest();
  }

  virtual void setSuccessor(unsigned idx, BasicBlock *NewSucc) {
    assert(idx < 2 && "Successor # out of range for invoke!");
    Operands[idx+1] = (Value*)NewSucc;
  }

  virtual unsigned getNumSuccessors() const { return 2; }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const InvokeInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return (I->getOpcode() == Instruction::Invoke);
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

#endif
