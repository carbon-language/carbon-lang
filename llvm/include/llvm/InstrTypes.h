//===-- llvm/InstrTypes.h - Important Instruction subclasses ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
  TerminatorInst(const Type *Ty, Instruction::TermOps iType,
                 Use *Ops, unsigned NumOps,
                 Instruction *InsertBefore = 0)
    : Instruction(Ty, iType, Ops, NumOps, InsertBefore) {}

  TerminatorInst(const Type *Ty, Instruction::TermOps iType,
                 Use *Ops, unsigned NumOps, BasicBlock *InsertAtEnd)
    : Instruction(Ty, iType, Ops, NumOps, InsertAtEnd) {}

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
  
  // avoiding warning: 'this' : used in base member initializer list
  UnaryInstruction* this_() { return this; }
protected:
  UnaryInstruction(const Type *Ty, unsigned iType, Value *V, Instruction *IB =0)
    : Instruction(Ty, iType, &Op, 1, IB), Op(V, this_()) {
  }
  UnaryInstruction(const Type *Ty, unsigned iType, Value *V, BasicBlock *IAE)
    : Instruction(Ty, iType, &Op, 1, IAE), Op(V, this_()) {
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
  
  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const UnaryInstruction *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::Malloc ||
           I->getOpcode() == Instruction::Alloca ||
           I->getOpcode() == Instruction::Free ||
           I->getOpcode() == Instruction::Load ||
           I->getOpcode() == Instruction::VAArg ||
           (I->getOpcode() >= CastOpsBegin && I->getOpcode() < CastOpsEnd);
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

//===----------------------------------------------------------------------===//
//                           BinaryOperator Class
//===----------------------------------------------------------------------===//

class BinaryOperator : public Instruction {
  Use Ops[2];
protected:
  void init(BinaryOps iType);
  BinaryOperator(BinaryOps iType, Value *S1, Value *S2, const Type *Ty,
                 const std::string &Name, Instruction *InsertBefore);
  BinaryOperator(BinaryOps iType, Value *S1, Value *S2, const Type *Ty,
                 const std::string &Name, BasicBlock *InsertAtEnd);
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
  static bool isNeg(const Value *V);
  static bool isNot(const Value *V);

  /// getNegArgument, getNotArgument - Helper functions to extract the
  ///     unary argument of a NEG or NOT operation implemented via Sub or Xor.
  ///
  static const Value *getNegArgument(const Value *BinOp);
  static       Value *getNegArgument(      Value *BinOp);
  static const Value *getNotArgument(const Value *BinOp);
  static       Value *getNotArgument(      Value *BinOp);

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

//===----------------------------------------------------------------------===//
//                               CastInst Class
//===----------------------------------------------------------------------===//

/// CastInst - This is the base class for all instructions that perform data
/// casts. It is simply provided so that instruction category testing
/// can be performed with code like:
///
/// if (isa<CastInst>(Instr)) { ... }
/// @brief Base class of casting instructions.
class CastInst : public UnaryInstruction {
  /// @brief Copy constructor
  CastInst(const CastInst &CI)
    : UnaryInstruction(CI.getType(), CI.getOpcode(), CI.getOperand(0)) {
  }
  /// @brief Do not allow default construction
  CastInst(); 
protected:
  /// @brief Constructor with insert-before-instruction semantics for subclasses
  CastInst(const Type *Ty, unsigned iType, Value *S, 
           const std::string &Name = "", Instruction *InsertBefore = 0)
    : UnaryInstruction(Ty, iType, S, InsertBefore) {
    setName(Name);
  }
  /// @brief Constructor with insert-at-end-of-block semantics for subclasses
  CastInst(const Type *Ty, unsigned iType, Value *S, 
           const std::string &Name, BasicBlock *InsertAtEnd)
    : UnaryInstruction(Ty, iType, S, InsertAtEnd) {
    setName(Name);
  }
public:
  /// Provides a way to construct any of the CastInst subclasses using an 
  /// opcode instead of the subclass's constructor. The opcode must be in the
  /// CastOps category (Instruction::isCast(opcode) returns true). This
  /// constructor has insert-before-instruction semantics to automatically
  /// insert the new CastInst before InsertBefore (if it is non-null).
  /// @brief Construct any of the CastInst subclasses
  static CastInst *create(
    Instruction::CastOps,    ///< The opcode of the cast instruction
    Value *S,                ///< The value to be casted (operand 0)
    const Type *Ty,          ///< The type to which cast should be made
    const std::string &Name = "", ///< Name for the instruction
    Instruction *InsertBefore = 0 ///< Place to insert the instruction
  );
  /// Provides a way to construct any of the CastInst subclasses using an
  /// opcode instead of the subclass's constructor. The opcode must be in the
  /// CastOps category. This constructor has insert-at-end-of-block semantics
  /// to automatically insert the new CastInst at the end of InsertAtEnd (if
  /// its non-null).
  /// @brief Construct any of the CastInst subclasses
  static CastInst *create(
    Instruction::CastOps,    ///< The opcode for the cast instruction
    Value *S,                ///< The value to be casted (operand 0)
    const Type *Ty,          ///< The type to which operand is casted
    const std::string &Name, ///< The name for the instruction
    BasicBlock *InsertAtEnd  ///< The block to insert the instruction into
  );

  /// @brief Create a ZExt or BitCast cast instruction
  static CastInst *createZExtOrBitCast(
    Value *S,                ///< The value to be casted (operand 0)
    const Type *Ty,          ///< The type to which cast should be made
    const std::string &Name = "", ///< Name for the instruction
    Instruction *InsertBefore = 0 ///< Place to insert the instruction
  );

  /// @brief Create a ZExt or BitCast cast instruction
  static CastInst *createZExtOrBitCast(
    Value *S,                ///< The value to be casted (operand 0)
    const Type *Ty,          ///< The type to which operand is casted
    const std::string &Name, ///< The name for the instruction
    BasicBlock *InsertAtEnd  ///< The block to insert the instruction into
  );

  /// @brief Create a SExt or BitCast cast instruction
  static CastInst *createSExtOrBitCast(
    Value *S,                ///< The value to be casted (operand 0)
    const Type *Ty,          ///< The type to which cast should be made
    const std::string &Name = "", ///< Name for the instruction
    Instruction *InsertBefore = 0 ///< Place to insert the instruction
  );

  /// @brief Create a BitCast or a PtrToInt cast instruction
  static CastInst *createPointerCast(
    Value *S,                ///< The pointer value to be casted (operand 0)
    const Type *Ty,          ///< The type to which operand is casted
    const std::string &Name, ///< The name for the instruction
    BasicBlock *InsertAtEnd  ///< The block to insert the instruction into
  );

  /// @brief Create a BitCast or a PtrToInt cast instruction
  static CastInst *createPointerCast(
    Value *S,                ///< The pointer value to be casted (operand 0)
    const Type *Ty,          ///< The type to which cast should be made
    const std::string &Name = "", ///< Name for the instruction
    Instruction *InsertBefore = 0 ///< Place to insert the instruction
  );

  /// @brief Create a ZExt, BitCast, or Trunc for int -> int casts.
  static CastInst *createIntegerCast(
    Value *S,                ///< The pointer value to be casted (operand 0)
    const Type *Ty,          ///< The type to which cast should be made
    bool isSigned,           ///< Whether to regard S as signed or not
    const std::string &Name = "", ///< Name for the instruction
    Instruction *InsertBefore = 0 ///< Place to insert the instruction
  );

  /// @brief Create a ZExt, BitCast, or Trunc for int -> int casts.
  static CastInst *createIntegerCast(
    Value *S,                ///< The integer value to be casted (operand 0)
    const Type *Ty,          ///< The integer type to which operand is casted
    bool isSigned,           ///< Whether to regard S as signed or not
    const std::string &Name, ///< The name for the instruction
    BasicBlock *InsertAtEnd  ///< The block to insert the instruction into
  );

  /// @brief Create an FPExt, BitCast, or FPTrunc for fp -> fp casts
  static CastInst *createFPCast(
    Value *S,                ///< The floating point value to be casted 
    const Type *Ty,          ///< The floating point type to cast to
    const std::string &Name = "", ///< Name for the instruction
    Instruction *InsertBefore = 0 ///< Place to insert the instruction
  );

  /// @brief Create an FPExt, BitCast, or FPTrunc for fp -> fp casts
  static CastInst *createFPCast(
    Value *S,                ///< The floating point value to be casted 
    const Type *Ty,          ///< The floating point type to cast to
    const std::string &Name, ///< The name for the instruction
    BasicBlock *InsertAtEnd  ///< The block to insert the instruction into
  );

  /// @brief Create a SExt or BitCast cast instruction
  static CastInst *createSExtOrBitCast(
    Value *S,                ///< The value to be casted (operand 0)
    const Type *Ty,          ///< The type to which operand is casted
    const std::string &Name, ///< The name for the instruction
    BasicBlock *InsertAtEnd  ///< The block to insert the instruction into
  );

  /// @brief Create a Trunc or BitCast cast instruction
  static CastInst *createTruncOrBitCast(
    Value *S,                ///< The value to be casted (operand 0)
    const Type *Ty,          ///< The type to which cast should be made
    const std::string &Name = "", ///< Name for the instruction
    Instruction *InsertBefore = 0 ///< Place to insert the instruction
  );

  /// @brief Create a Trunc or BitCast cast instruction
  static CastInst *createTruncOrBitCast(
    Value *S,                ///< The value to be casted (operand 0)
    const Type *Ty,          ///< The type to which operand is casted
    const std::string &Name, ///< The name for the instruction
    BasicBlock *InsertAtEnd  ///< The block to insert the instruction into
  );

  /// @brief Check whether it is valid to call getCastOpcode for these types.
  static bool isCastable(
    const Type *SrcTy, ///< The Type from which the value should be cast.
    const Type *DestTy ///< The Type to which the value should be cast.
  );

  /// Returns the opcode necessary to cast Val into Ty using usual casting
  /// rules.
  /// @brief Infer the opcode for cast operand and type
  static Instruction::CastOps getCastOpcode(
    const Value *Val, ///< The value to cast
    bool SrcIsSigned, ///< Whether to treat the source as signed
    const Type *Ty,   ///< The Type to which the value should be casted
    bool DstIsSigned  ///< Whether to treate the dest. as signed
  );

  /// There are several places where we need to know if a cast instruction 
  /// only deals with integer source and destination types. To simplify that
  /// logic, this method is provided.
  /// @returns true iff the cast has only integral typed operand and dest type.
  /// @brief Determine if this is an integer-only cast.
  bool isIntegerCast() const;

  /// A lossless cast is one that does not alter the basic value. It implies
  /// a no-op cast but is more stringent, preventing things like int->float,
  /// long->double, int->ptr, or vector->anything. 
  /// @returns true iff the cast is lossless.
  /// @brief Determine if this is a lossless cast.
  bool isLosslessCast() const;

  /// A no-op cast is one that can be effected without changing any bits. 
  /// It implies that the source and destination types are the same size. The
  /// IntPtrTy argument is used to make accurate determinations for casts 
  /// involving Integer and Pointer types. They are no-op casts if the integer
  /// is the same size as the pointer. However, pointer size varies with 
  /// platform. Generally, the result of TargetData::getIntPtrType() should be
  /// passed in. If that's not available, use Type::Int64Ty, which will make
  /// the isNoopCast call conservative.
  /// @brief Determine if this cast is a no-op cast. 
  bool isNoopCast(
    const Type *IntPtrTy ///< Integer type corresponding to pointer
  ) const;

  /// Determine how a pair of casts can be eliminated, if they can be at all.
  /// This is a helper function for both CastInst and ConstantExpr.
  /// @returns 0 if the CastInst pair can't be eliminated
  /// @returns Instruction::CastOps value for a cast that can replace 
  /// the pair, casting SrcTy to DstTy.
  /// @brief Determine if a cast pair is eliminable
  static unsigned isEliminableCastPair(
    Instruction::CastOps firstOpcode,  ///< Opcode of first cast
    Instruction::CastOps secondOpcode, ///< Opcode of second cast
    const Type *SrcTy, ///< SrcTy of 1st cast
    const Type *MidTy, ///< DstTy of 1st cast & SrcTy of 2nd cast
    const Type *DstTy, ///< DstTy of 2nd cast
    const Type *IntPtrTy ///< Integer type corresponding to Ptr types
  );

  /// @brief Return the opcode of this CastInst
  Instruction::CastOps getOpcode() const { 
    return Instruction::CastOps(Instruction::getOpcode()); 
  }

  /// @brief Return the source type, as a convenience
  const Type* getSrcTy() const { return getOperand(0)->getType(); }
  /// @brief Return the destination type, as a convenience
  const Type* getDestTy() const { return getType(); }

  /// This method can be used to determine if a cast from S to DstTy using
  /// Opcode op is valid or not. 
  /// @returns true iff the proposed cast is valid.
  /// @brief Determine if a cast is valid without creating one.
  static bool castIsValid(Instruction::CastOps op, Value *S, const Type *DstTy);

  /// @brief Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const CastInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() >= CastOpsBegin && I->getOpcode() < CastOpsEnd;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

//===----------------------------------------------------------------------===//
//                               CmpInst Class
//===----------------------------------------------------------------------===//

/// This class is the base class for the comparison instructions. 
/// @brief Abstract base class of comparison instructions.
class CmpInst: public Instruction {
  CmpInst(); // do not implement
protected:
  CmpInst(Instruction::OtherOps op, unsigned short pred, Value *LHS, Value *RHS,
          const std::string &Name = "", Instruction *InsertBefore = 0);
  
  CmpInst(Instruction::OtherOps op, unsigned short pred, Value *LHS, Value *RHS,
          const std::string &Name, BasicBlock *InsertAtEnd);

  Use Ops[2]; // CmpInst instructions always have 2 operands, optimize

public:
  /// Construct a compare instruction, given the opcode, the predicate and 
  /// the two operands.  Optionally (if InstBefore is specified) insert the 
  /// instruction into a BasicBlock right before the specified instruction.  
  /// The specified Instruction is allowed to be a dereferenced end iterator.
  /// @brief Create a CmpInst
  static CmpInst *create(OtherOps Op, unsigned short predicate, Value *S1, 
                         Value *S2, const std::string &Name = "",
                         Instruction *InsertBefore = 0);

  /// Construct a compare instruction, given the opcode, the predicate and the 
  /// two operands.  Also automatically insert this instruction to the end of 
  /// the BasicBlock specified.
  /// @brief Create a CmpInst
  static CmpInst *create(OtherOps Op, unsigned short predicate, Value *S1, 
                         Value *S2, const std::string &Name, 
                         BasicBlock *InsertAtEnd);

  /// @brief Get the opcode casted to the right type
  OtherOps getOpcode() const {
    return static_cast<OtherOps>(Instruction::getOpcode());
  }

  /// The predicate for CmpInst is defined by the subclasses but stored in 
  /// the SubclassData field (see Value.h).  We allow it to be fetched here
  /// as the predicate but there is no enum type for it, just the raw unsigned 
  /// short. This facilitates comparison of CmpInst instances without delving
  /// into the subclasses since predicate values are distinct between the
  /// CmpInst subclasses.
  /// @brief Return the predicate for this instruction.
  unsigned short getPredicate() const {
    return SubclassData;
  }

  /// @brief Provide more efficient getOperand methods.
  Value *getOperand(unsigned i) const {
    assert(i < 2 && "getOperand() out of range!");
    return Ops[i];
  }
  void setOperand(unsigned i, Value *Val) {
    assert(i < 2 && "setOperand() out of range!");
    Ops[i] = Val;
  }

  /// @brief CmpInst instructions always have 2 operands.
  unsigned getNumOperands() const { return 2; }

  /// This is just a convenience that dispatches to the subclasses.
  /// @brief Swap the operands and adjust predicate accordingly to retain
  /// the same comparison.
  void swapOperands();

  /// This is just a convenience that dispatches to the subclasses.
  /// @brief Determine if this CmpInst is commutative.
  bool isCommutative();

  /// This is just a convenience that dispatches to the subclasses.
  /// @brief Determine if this is an equals/not equals predicate.
  bool isEquality();

  /// @returns true if the predicate is unsigned, false otherwise.
  /// @brief Determine if the predicate is an unsigned operation.
  static bool isUnsigned(unsigned short predicate);

  /// @returns true if the predicate is signed, false otherwise.
  /// @brief Determine if the predicate is an signed operation.
  static bool isSigned(unsigned short predicate);

  /// @brief Determine if the predicate is an ordered operation.
  static bool isOrdered(unsigned short predicate);

  /// @brief Determine if the predicate is an unordered operation.
  static bool isUnordered(unsigned short predicate);

  /// @brief Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const CmpInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::ICmp || 
           I->getOpcode() == Instruction::FCmp;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

} // End llvm namespace

#endif
