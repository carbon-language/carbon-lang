//===-- llvm/InstrTypes.h - Important Instruction subclasses ----*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines various meta classes of instructions that exist in the VM
// representation.  Specific concrete subclasses of these may be found in the 
// i*.h files...
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_INSTRUCTION_TYPES_H
#define LLVM_INSTRUCTION_TYPES_H

#include "llvm/Instruction.h"

namespace llvm {

//===----------------------------------------------------------------------===//
//                            TerminatorInst Class
//===----------------------------------------------------------------------===//

/// TerminatorInst - Subclasses of this class are all able to terminate a basic 
/// block.  Thus, these are all the flow control type of operations.
///
class TerminatorInst : public Instruction {
protected:
  TerminatorInst(Instruction::TermOps iType, Instruction *InsertBefore = 0);
  TerminatorInst(const Type *Ty, Instruction::TermOps iType,
                 const std::string &Name = "", Instruction *InsertBefore = 0)
    : Instruction(Ty, iType, Name, InsertBefore) {}

  TerminatorInst(Instruction::TermOps iType, BasicBlock *InsertAtEnd);
  TerminatorInst(const Type *Ty, Instruction::TermOps iType,
                 const std::string &Name, BasicBlock *InsertAtEnd)
    : Instruction(Ty, iType, Name, InsertAtEnd) {}

public:

  /// Terminators must implement the methods required by Instruction...
  virtual Instruction *clone() const = 0;

  /// Additionally, they must provide a method to get at the successors of this
  /// terminator instruction.  'idx' may not be >= the number of successors
  /// returned by getNumSuccessors()!
  ///
  virtual const BasicBlock *getSuccessor(unsigned idx) const = 0;
  virtual unsigned getNumSuccessors() const = 0;
  
  /// Set a successor at a given index
  virtual void setSuccessor(unsigned idx, BasicBlock *B) = 0;

  inline BasicBlock *getSuccessor(unsigned idx) {
    const TerminatorInst *TI = this;
    return const_cast<BasicBlock*>(TI->getSuccessor(idx));
  }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const TerminatorInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() >= TermOpsBegin && I->getOpcode() < TermOpsEnd;
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
  void init(BinaryOps iType, Value *S1, Value *S2);
  BinaryOperator(BinaryOps iType, Value *S1, Value *S2, const Type *Ty,
                 const std::string &Name, Instruction *InsertBefore)
    : Instruction(Ty, iType, Name, InsertBefore) {
    init(iType, S1, S2);
  }
  BinaryOperator(BinaryOps iType, Value *S1, Value *S2, const Type *Ty,
                 const std::string &Name, BasicBlock *InsertAtEnd)
    : Instruction(Ty, iType, Name, InsertAtEnd) {
    init(iType, S1, S2);
  }

public:

  /// create() - Construct a binary instruction, given the opcode and the two
  /// operands.  Optionally (if InstBefore is specified) insert the instruction
  /// into a BasicBlock right before the specified instruction.  The specified
  /// Instruction is allowed to be a dereferenced end iterator.
  ///
  static BinaryOperator *create(BinaryOps Op, Value *S1, Value *S2,
				const std::string &Name = "",
                                Instruction *InsertBefore = 0);
                               
  /// create() - Construct a binary instruction, given the opcode and the two
  /// operands.  Also automatically insert this instruction to the end of the
  /// BasicBlock specified.
  ///
  static BinaryOperator *create(BinaryOps Op, Value *S1, Value *S2,
				const std::string &Name,
                                BasicBlock *InsertAtEnd);

  /// create* - These methods just forward to create, and are useful when you
  /// statically know what type of instruction you're going to create.  These
  /// helpers just save some typing.
#define HANDLE_BINARY_INST(N, OPC, CLASS) \
  static BinaryOperator *create##OPC(Value *V1, Value *V2, \
                                     const std::string &Name = "") {\
    return create(Instruction::OPC, V1, V2, Name);\
  }
#include "llvm/Instruction.def"
#define HANDLE_BINARY_INST(N, OPC, CLASS) \
  static BinaryOperator *create##OPC(Value *V1, Value *V2, \
                                     const std::string &Name, BasicBlock *BB) {\
    return create(Instruction::OPC, V1, V2, Name, BB);\
  }
#include "llvm/Instruction.def"
#define HANDLE_BINARY_INST(N, OPC, CLASS) \
  static BinaryOperator *create##OPC(Value *V1, Value *V2, \
                                     const std::string &Name, Instruction *I) {\
    return create(Instruction::OPC, V1, V2, Name, I);\
  }
#include "llvm/Instruction.def"
                               

  /// Helper functions to construct and inspect unary operations (NEG and NOT)
  /// via binary operators SUB and XOR:
  /// 
  /// createNeg, createNot - Create the NEG and NOT
  ///     instructions out of SUB and XOR instructions.
  ///
  static BinaryOperator *createNeg(Value *Op, const std::string &Name = "",
                                   Instruction *InsertBefore = 0);
  static BinaryOperator *createNeg(Value *Op, const std::string &Name,
                                   BasicBlock *InsertAtEnd);
  static BinaryOperator *createNot(Value *Op, const std::string &Name = "",
                                   Instruction *InsertBefore = 0);
  static BinaryOperator *createNot(Value *Op, const std::string &Name,
                                   BasicBlock *InsertAtEnd);

  /// isNeg, isNot - Check if the given Value is a NEG or NOT instruction.
  ///
  static bool            isNeg(const Value *V);
  static bool            isNot(const Value *V);

  /// getNegArgument, getNotArgument - Helper functions to extract the
  ///     unary argument of a NEG or NOT operation implemented via Sub or Xor.
  /// 
  static const Value*    getNegArgument(const BinaryOperator* Bop);
  static       Value*    getNegArgument(      BinaryOperator* Bop);
  static const Value*    getNotArgument(const BinaryOperator* Bop);
  static       Value*    getNotArgument(      BinaryOperator* Bop);

  BinaryOps getOpcode() const { 
    return static_cast<BinaryOps>(Instruction::getOpcode());
  }

  virtual Instruction *clone() const {
    return create(getOpcode(), Operands[0], Operands[1]);
  }

  /// swapOperands - Exchange the two operands to this instruction.
  /// This instruction is safe to use on any binary instruction and
  /// does not modify the semantics of the instruction.  If the
  /// instruction is order dependent (SetLT f.e.) the opcode is
  /// changed.  If the instruction cannot be reversed (ie, it's a Div),
  /// then return true.
  ///
  bool swapOperands();

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const BinaryOperator *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() >= BinaryOpsBegin && I->getOpcode() < BinaryOpsEnd; 
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

} // End llvm namespace

#endif
