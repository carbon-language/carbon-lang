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
  virtual const char *getOpcodeName() const = 0;

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
public:

  // create() - Construct a unary instruction, given the opcode
  // and its operand.
  //
  static UnaryOperator *create(unsigned Op, Value *Source);

  UnaryOperator(Value *S, unsigned iType, const string &Name = "")
      : Instruction(S->getType(), iType, Name) {
    Operands.reserve(1);
    Operands.push_back(Use(S, this));
  }

  virtual Instruction *clone() const { 
    return create(getOpcode(), Operands[0]);
  }

  virtual const char *getOpcodeName() const = 0;
};



//===----------------------------------------------------------------------===//
//                           BinaryOperator Class
//===----------------------------------------------------------------------===//

class BinaryOperator : public Instruction {
public:

  // create() - Construct a binary instruction, given the opcode
  // and the two operands.
  //
  static BinaryOperator *create(unsigned Op, Value *S1, Value *S2,
				const string &Name = "");

  BinaryOperator(unsigned iType, Value *S1, Value *S2, 
                 const string &Name = "") 
    : Instruction(S1->getType(), iType, Name) {
    Operands.reserve(2);
    Operands.push_back(Use(S1, this));
    Operands.push_back(Use(S2, this));
    assert(Operands[0] && Operands[1] && 
	   Operands[0]->getType() == Operands[1]->getType());
  }

  virtual Instruction *clone() const {
    return create(getOpcode(), Operands[0], Operands[1]);
  }

  virtual const char *getOpcodeName() const = 0;
};

#endif
