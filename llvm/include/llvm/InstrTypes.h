//===-- llvm/InstrTypes.h - Important Instruction subclasses -----*- C++ -*--=//
//
// This file defines various meta classes of instructions that exist in the VM
// representation.  Specific concrete subclasses of these may be found in the 
// i*.h files...
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_INSTRUCTION_TYPES_H
#define LLVM_INSTRUCTION_TYPES_H

#include "llvm/Instruction.h"
#include <list>
#include <vector>

class Method;
class SymTabValue;

//===----------------------------------------------------------------------===//
//                            TerminatorInst Class
//===----------------------------------------------------------------------===//

// TerminatorInst - Subclasses of this class are all able to terminate a basic 
// block.  Thus, these are all the flow control type of operations.
//
class TerminatorInst : public Instruction {
public:
  TerminatorInst(unsigned iType);
  inline ~TerminatorInst() {}

  // Terminators must implement the methods required by Instruction...
  virtual Instruction *clone() const = 0;
  virtual void dropAllReferences() = 0;
  virtual string getOpcode() const = 0;

  virtual bool setOperand(unsigned i, Value *Val) = 0;
  virtual const Value *getOperand(unsigned i) const = 0;
  inline Value *getOperand(unsigned i) {
    return (Value*)((const Instruction *)this)->getOperand(i);
  }

  // Additionally, they must provide a method to get at the successors of this
  // terminator instruction.  If 'idx' is out of range, a null pointer shall be
  // returned.
  //
  virtual const BasicBlock *getSuccessor(unsigned idx) const = 0;
  virtual unsigned getNumSuccessors() const = 0;

  inline BasicBlock *getSuccessor(unsigned idx) {
    return (BasicBlock*)((const TerminatorInst *)this)->getSuccessor(idx);
  }
};


//===----------------------------------------------------------------------===//
//                            UnaryOperator Class
//===----------------------------------------------------------------------===//

class UnaryOperator : public Instruction {
  Use Source;
public:

  // getUnaryOperator() - Construct a unary instruction, given the opcode
  // and its operand.
  //
  static UnaryOperator *getUnaryOperator(unsigned Op, Value *Source);

  UnaryOperator(Value *S, unsigned iType, const string &Name = "")
      : Instruction(S->getType(), iType, Name), Source(S, this) {
  }
  inline ~UnaryOperator() { dropAllReferences(); }

  virtual Instruction *clone() const { 
    return getUnaryOperator(getInstType(), Source);
  }

  virtual void dropAllReferences() {
    Source = 0;
  }

  virtual string getOpcode() const = 0;

  virtual unsigned getNumOperands() const { return 1; }
  inline Value *getOperand(unsigned i) {
    return (i == 0) ? Source : 0;
  }
  virtual const Value *getOperand(unsigned i) const {
    return (i == 0) ? Source : 0;
  }
  virtual bool setOperand(unsigned i, Value *Val) {
    // assert(Val && "operand must not be null!");
    if (i) return false;
    Source = Val;
    return true;
  }
};



//===----------------------------------------------------------------------===//
//                           BinaryOperator Class
//===----------------------------------------------------------------------===//

class BinaryOperator : public Instruction {
  Use Source1, Source2;
public:

  // getBinaryOperator() - Construct a binary instruction, given the opcode
  // and the two operands.
  //
  static BinaryOperator *getBinaryOperator(unsigned Op, Value *S1, Value *S2);

  BinaryOperator(unsigned iType, Value *S1, Value *S2, 
                 const string &Name = "") 
    : Instruction(S1->getType(), iType, Name), Source1(S1, this), 
      Source2(S2, this){
    assert(S1 && S2 && S1->getType() == S2->getType());
  }
  inline ~BinaryOperator() { dropAllReferences(); }

  virtual Instruction *clone() const { 
    return getBinaryOperator(getInstType(), Source1, Source2);
  }

  virtual void dropAllReferences() {
    Source1 = Source2 = 0;
  }

  virtual string getOpcode() const = 0;

  virtual unsigned getNumOperands() const { return 2; }
  virtual const Value *getOperand(unsigned i) const {
    return (i == 0) ? Source1 : ((i == 1) ? Source2 : 0);
  }
  inline Value *getOperand(unsigned i) {
    return (i == 0) ? Source1 : ((i == 1) ? Source2 : 0);
  }

  virtual bool setOperand(unsigned i, Value *Val) {
    // assert(Val && "operand must not be null!");
    if (i == 0) {
      Source1 = Val; //assert(Val->getType() == Source2->getType());
    } else if (i == 1) {
      Source2 = Val; //assert(Val->getType() == Source1->getType());
    } else {
      return false;
    }
    return true;
  }
};

#endif
