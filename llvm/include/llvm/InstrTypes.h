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
  TerminatorInst(Instruction::TermOps iType, Use *Ops, unsigned NumOps,
                 Instruction *InsertBefore = 0);
  TerminatorInst(const Type *Ty, Instruction::TermOps iType,
                  Use *Ops, unsigned NumOps,
                 const std::string &Name = "", Instruction *InsertBefore = 0)
    : Instruction(Ty, iType, Ops, NumOps, Name, InsertBefore) {}

  TerminatorInst(Instruction::TermOps iType, Use *Ops, unsigned NumOps,
                 BasicBlock *InsertAtEnd);
  TerminatorInst(const Type *Ty, Instruction::TermOps iType,
                  Use *Ops, unsigned NumOps,
                 const std::string &Name, BasicBlock *InsertAtEnd)
    : Instruction(Ty, iType, Ops, NumOps, Name, InsertAtEnd) {}

  // Out of line virtual method, so the vtable, etc has a home.
  ~TerminatorInst();

  /// Virtual methods - Terminators should overload these and provide inline
  /// overrides of non-V methods.
  virtual BasicBlock *getSuccessorV(unsigned idx) const = 0;
  virtual unsigned getNumSuccessorsV() const = 0;
  virtual void setSuccessorV(unsigned idx, BasicBlock *B) = 0;
public:

  virtual Instruction *clone() const = 0;

  /// getNumSuccessors - Return the number of successors that this terminator
  /// has.
  unsigned getNumSuccessors() const {
    return getNumSuccessorsV();
  }

  /// getSuccessor - Return the specified successor.
  ///
  BasicBlock *getSuccessor(unsigned idx) const {
    return getSuccessorV(idx);
  }

  /// setSuccessor - Update the specified successor to point at the provided
  /// block.
  void setSuccessor(unsigned idx, BasicBlock *B) {
    setSuccessorV(idx, B);
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
//                          UnaryInstruction Class
//===----------------------------------------------------------------------===//

class UnaryInstruction : public Instruction {
  Use Op;
protected:
  UnaryInstruction(const Type *Ty, unsigned iType, Value *V,
                   const std::string &Name = "", Instruction *IB = 0)
    : Instruction(Ty, iType, &Op, 1, Name, IB), Op(V, this) {
  }
  UnaryInstruction(const Type *Ty, unsigned iType, Value *V,
                   const std::string &Name, BasicBlock *IAE)
    : Instruction(Ty, iType, &Op, 1, Name, IAE), Op(V, this) {
  }
public:
  // Out of line virtual method, so the vtable, etc has a home.
  ~UnaryInstruction();

  // Transparently provide more efficient getOperand methods.
  Value *getOperand(unsigned i) const {
    assert(i == 0 && "getOperand() out of range!");
    return Op;
  }
  void setOperand(unsigned i, Value *Val) {
    assert(i == 0 && "setOperand() out of range!");
    Op = Val;
  }
  unsigned getNumOperands() const { return 1; }
};

//===----------------------------------------------------------------------===//
//                           BinaryOperator Class
//===----------------------------------------------------------------------===//

class BinaryOperator : public Instruction {
  Use Ops[2];
protected:
  void init(BinaryOps iType);
  BinaryOperator(BinaryOps iType, Value *S1, Value *S2, const Type *Ty,
                 const std::string &Name, Instruction *InsertBefore)
    : Instruction(Ty, iType, Ops, 2, Name, InsertBefore) {
      Ops[0].init(S1, this);
      Ops[1].init(S2, this);
    init(iType);
  }
  BinaryOperator(BinaryOps iType, Value *S1, Value *S2, const Type *Ty,
                 const std::string &Name, BasicBlock *InsertAtEnd)
    : Instruction(Ty, iType, Ops, 2, Name, InsertAtEnd) {
    Ops[0].init(S1, this);
    Ops[1].init(S2, this);
    init(iType);
  }

public:

  /// Transparently provide more efficient getOperand methods.
  Value *getOperand(unsigned i) const {
    assert(i < 2 && "getOperand() out of range!");
    return Ops[i];
  }
  void setOperand(unsigned i, Value *Val) {
    assert(i < 2 && "setOperand() out of range!");
    Ops[i] = Val;
  }
  unsigned getNumOperands() const { return 2; }

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
  static const Value*    getNegArgument(const Value *BinOp);
  static       Value*    getNegArgument(      Value *BinOp);
  static const Value*    getNotArgument(const Value *BinOp);
  static       Value*    getNotArgument(      Value *BinOp);

  BinaryOps getOpcode() const {
    return static_cast<BinaryOps>(Instruction::getOpcode());
  }

  virtual BinaryOperator *clone() const;

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
