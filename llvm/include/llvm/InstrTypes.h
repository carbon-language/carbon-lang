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

//===----------------------------------------------------------------------===//
//                            TerminatorInst Class
//===----------------------------------------------------------------------===//

// TerminatorInst - Subclasses of this class are all able to terminate a basic 
// block.  Thus, these are all the flow control type of operations.
//
class TerminatorInst : public Instruction {
protected:
  TerminatorInst(Instruction::TermOps iType);
  TerminatorInst(const Type *Ty, Instruction::TermOps iType,
                 const std::string &Name = "");
public:

  // Terminators must implement the methods required by Instruction...
  virtual Instruction *clone() const = 0;

  // Additionally, they must provide a method to get at the successors of this
  // terminator instruction.  'idx' may not be >= the number of successors
  // returned by getNumSuccessors()!
  //
  virtual const BasicBlock *getSuccessor(unsigned idx) const = 0;
  virtual unsigned getNumSuccessors() const = 0;
  
  // Set a successor at a given index
  virtual void setSuccessor(unsigned idx, BasicBlock *B) = 0;

  inline BasicBlock *getSuccessor(unsigned idx) {
    return (BasicBlock*)((const TerminatorInst *)this)->getSuccessor(idx);
  }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const TerminatorInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() >= FirstTermOp && I->getOpcode() < NumTermOps; 
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};


//===----------------------------------------------------------------------===//
//                           BinaryOperator Class
//===----------------------------------------------------------------------===//

class BinaryOperator : public Instruction {
protected:
  BinaryOperator(BinaryOps iType, Value *S1, Value *S2, 
                 const std::string &Name = "") 
    : Instruction(S1->getType(), iType, Name) {
    Operands.reserve(2);
    Operands.push_back(Use(S1, this));
    Operands.push_back(Use(S2, this));
    assert(Operands[0] && Operands[1] && 
	   Operands[0]->getType() == Operands[1]->getType());
  }

public:

  // create() - Construct a binary instruction, given the opcode
  // and the two operands.
  //
  static BinaryOperator *create(BinaryOps Op, Value *S1, Value *S2,
				const std::string &Name = "");

  // createNeg, createNot - Helper constructors that create the "neg" and "not"
  // instructions out of "sub" and "xor" instructions.
  //
  static BinaryOperator *createNeg(Value *Op, const std::string &Name = "");
  static BinaryOperator *createNot(Value *Op, const std::string &Name = "");

  BinaryOps getOpcode() const { 
    return (BinaryOps)Instruction::getOpcode();
  }

  virtual Instruction *clone() const {
    return create(getOpcode(), Operands[0], Operands[1]);
  }

  // swapOperands - Exchange the two operands to this instruction.
  // This instruction is safe to use on any binary instruction and
  // does not modify the semantics of the instruction.  If the
  // instruction is order dependant (SetLT f.e.) the opcode is
  // changed.  If the instruction cannot be reversed (ie, it's a Div),
  // then return true.
  //
  bool swapOperands();

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const BinaryOperator *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() >= FirstBinaryOp && I->getOpcode() < NumBinaryOps; 
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

#endif
