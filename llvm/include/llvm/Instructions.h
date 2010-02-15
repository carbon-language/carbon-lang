//===-- llvm/Instructions.h - Instruction subclass definitions --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file exposes the class definitions of all of the subclasses of the
// Instruction class.  This is meant to be an easy way to get access to all
// instruction subclasses.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_INSTRUCTIONS_H
#define LLVM_INSTRUCTIONS_H

#include "llvm/InstrTypes.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Attributes.h"
#include "llvm/CallingConv.h"
#include "llvm/ADT/SmallVector.h"
#include <iterator>

namespace llvm {

class ConstantInt;
class ConstantRange;
class APInt;
class LLVMContext;
class DominatorTree;

//===----------------------------------------------------------------------===//
//                                AllocaInst Class
//===----------------------------------------------------------------------===//

/// AllocaInst - an instruction to allocate memory on the stack
///
class AllocaInst : public UnaryInstruction {
protected:
  virtual AllocaInst *clone_impl() const;
public:
  explicit AllocaInst(const Type *Ty, Value *ArraySize = 0,
                      const Twine &Name = "", Instruction *InsertBefore = 0);
  AllocaInst(const Type *Ty, Value *ArraySize, 
             const Twine &Name, BasicBlock *InsertAtEnd);

  AllocaInst(const Type *Ty, const Twine &Name, Instruction *InsertBefore = 0);
  AllocaInst(const Type *Ty, const Twine &Name, BasicBlock *InsertAtEnd);

  AllocaInst(const Type *Ty, Value *ArraySize, unsigned Align,
             const Twine &Name = "", Instruction *InsertBefore = 0);
  AllocaInst(const Type *Ty, Value *ArraySize, unsigned Align,
             const Twine &Name, BasicBlock *InsertAtEnd);

  // Out of line virtual method, so the vtable, etc. has a home.
  virtual ~AllocaInst();

  /// isArrayAllocation - Return true if there is an allocation size parameter
  /// to the allocation instruction that is not 1.
  ///
  bool isArrayAllocation() const;

  /// getArraySize - Get the number of elements allocated. For a simple
  /// allocation of a single element, this will return a constant 1 value.
  ///
  const Value *getArraySize() const { return getOperand(0); }
  Value *getArraySize() { return getOperand(0); }

  /// getType - Overload to return most specific pointer type
  ///
  const PointerType *getType() const {
    return reinterpret_cast<const PointerType*>(Instruction::getType());
  }

  /// getAllocatedType - Return the type that is being allocated by the
  /// instruction.
  ///
  const Type *getAllocatedType() const;

  /// getAlignment - Return the alignment of the memory that is being allocated
  /// by the instruction.
  ///
  unsigned getAlignment() const {
    return (1u << getSubclassDataFromInstruction()) >> 1;
  }
  void setAlignment(unsigned Align);

  /// isStaticAlloca - Return true if this alloca is in the entry block of the
  /// function and is a constant size.  If so, the code generator will fold it
  /// into the prolog/epilog code, so it is basically free.
  bool isStaticAlloca() const;

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const AllocaInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return (I->getOpcode() == Instruction::Alloca);
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
private:
  // Shadow Instruction::setInstructionSubclassData with a private forwarding
  // method so that subclasses cannot accidentally use it.
  void setInstructionSubclassData(unsigned short D) {
    Instruction::setInstructionSubclassData(D);
  }
};


//===----------------------------------------------------------------------===//
//                                LoadInst Class
//===----------------------------------------------------------------------===//

/// LoadInst - an instruction for reading from memory.  This uses the
/// SubclassData field in Value to store whether or not the load is volatile.
///
class LoadInst : public UnaryInstruction {
  void AssertOK();
protected:
  virtual LoadInst *clone_impl() const;
public:
  LoadInst(Value *Ptr, const Twine &NameStr, Instruction *InsertBefore);
  LoadInst(Value *Ptr, const Twine &NameStr, BasicBlock *InsertAtEnd);
  LoadInst(Value *Ptr, const Twine &NameStr, bool isVolatile = false,
           Instruction *InsertBefore = 0);
  LoadInst(Value *Ptr, const Twine &NameStr, bool isVolatile,
           unsigned Align, Instruction *InsertBefore = 0);
  LoadInst(Value *Ptr, const Twine &NameStr, bool isVolatile,
           BasicBlock *InsertAtEnd);
  LoadInst(Value *Ptr, const Twine &NameStr, bool isVolatile,
           unsigned Align, BasicBlock *InsertAtEnd);

  LoadInst(Value *Ptr, const char *NameStr, Instruction *InsertBefore);
  LoadInst(Value *Ptr, const char *NameStr, BasicBlock *InsertAtEnd);
  explicit LoadInst(Value *Ptr, const char *NameStr = 0,
                    bool isVolatile = false,  Instruction *InsertBefore = 0);
  LoadInst(Value *Ptr, const char *NameStr, bool isVolatile,
           BasicBlock *InsertAtEnd);

  /// isVolatile - Return true if this is a load from a volatile memory
  /// location.
  ///
  bool isVolatile() const { return getSubclassDataFromInstruction() & 1; }

  /// setVolatile - Specify whether this is a volatile load or not.
  ///
  void setVolatile(bool V) {
    setInstructionSubclassData((getSubclassDataFromInstruction() & ~1) |
                               (V ? 1 : 0));
  }

  /// getAlignment - Return the alignment of the access that is being performed
  ///
  unsigned getAlignment() const {
    return (1 << (getSubclassDataFromInstruction() >> 1)) >> 1;
  }

  void setAlignment(unsigned Align);

  Value *getPointerOperand() { return getOperand(0); }
  const Value *getPointerOperand() const { return getOperand(0); }
  static unsigned getPointerOperandIndex() { return 0U; }

  unsigned getPointerAddressSpace() const {
    return cast<PointerType>(getPointerOperand()->getType())->getAddressSpace();
  }
  
  
  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const LoadInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::Load;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
private:
  // Shadow Instruction::setInstructionSubclassData with a private forwarding
  // method so that subclasses cannot accidentally use it.
  void setInstructionSubclassData(unsigned short D) {
    Instruction::setInstructionSubclassData(D);
  }
};


//===----------------------------------------------------------------------===//
//                                StoreInst Class
//===----------------------------------------------------------------------===//

/// StoreInst - an instruction for storing to memory
///
class StoreInst : public Instruction {
  void *operator new(size_t, unsigned);  // DO NOT IMPLEMENT
  void AssertOK();
protected:
  virtual StoreInst *clone_impl() const;
public:
  // allocate space for exactly two operands
  void *operator new(size_t s) {
    return User::operator new(s, 2);
  }
  StoreInst(Value *Val, Value *Ptr, Instruction *InsertBefore);
  StoreInst(Value *Val, Value *Ptr, BasicBlock *InsertAtEnd);
  StoreInst(Value *Val, Value *Ptr, bool isVolatile = false,
            Instruction *InsertBefore = 0);
  StoreInst(Value *Val, Value *Ptr, bool isVolatile,
            unsigned Align, Instruction *InsertBefore = 0);
  StoreInst(Value *Val, Value *Ptr, bool isVolatile, BasicBlock *InsertAtEnd);
  StoreInst(Value *Val, Value *Ptr, bool isVolatile,
            unsigned Align, BasicBlock *InsertAtEnd);


  /// isVolatile - Return true if this is a load from a volatile memory
  /// location.
  ///
  bool isVolatile() const { return getSubclassDataFromInstruction() & 1; }

  /// setVolatile - Specify whether this is a volatile load or not.
  ///
  void setVolatile(bool V) {
    setInstructionSubclassData((getSubclassDataFromInstruction() & ~1) |
                               (V ? 1 : 0));
  }

  /// Transparently provide more efficient getOperand methods.
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);

  /// getAlignment - Return the alignment of the access that is being performed
  ///
  unsigned getAlignment() const {
    return (1 << (getSubclassDataFromInstruction() >> 1)) >> 1;
  }

  void setAlignment(unsigned Align);

  Value *getPointerOperand() { return getOperand(1); }
  const Value *getPointerOperand() const { return getOperand(1); }
  static unsigned getPointerOperandIndex() { return 1U; }

  unsigned getPointerAddressSpace() const {
    return cast<PointerType>(getPointerOperand()->getType())->getAddressSpace();
  }
  
  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const StoreInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::Store;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
private:
  // Shadow Instruction::setInstructionSubclassData with a private forwarding
  // method so that subclasses cannot accidentally use it.
  void setInstructionSubclassData(unsigned short D) {
    Instruction::setInstructionSubclassData(D);
  }
};

template <>
struct OperandTraits<StoreInst> : public FixedNumOperandTraits<2> {
};

DEFINE_TRANSPARENT_OPERAND_ACCESSORS(StoreInst, Value)

//===----------------------------------------------------------------------===//
//                             GetElementPtrInst Class
//===----------------------------------------------------------------------===//

// checkType - Simple wrapper function to give a better assertion failure
// message on bad indexes for a gep instruction.
//
static inline const Type *checkType(const Type *Ty) {
  assert(Ty && "Invalid GetElementPtrInst indices for type!");
  return Ty;
}

/// GetElementPtrInst - an instruction for type-safe pointer arithmetic to
/// access elements of arrays and structs
///
class GetElementPtrInst : public Instruction {
  GetElementPtrInst(const GetElementPtrInst &GEPI);
  void init(Value *Ptr, Value* const *Idx, unsigned NumIdx,
            const Twine &NameStr);
  void init(Value *Ptr, Value *Idx, const Twine &NameStr);

  template<typename InputIterator>
  void init(Value *Ptr, InputIterator IdxBegin, InputIterator IdxEnd,
            const Twine &NameStr,
            // This argument ensures that we have an iterator we can
            // do arithmetic on in constant time
            std::random_access_iterator_tag) {
    unsigned NumIdx = static_cast<unsigned>(std::distance(IdxBegin, IdxEnd));

    if (NumIdx > 0) {
      // This requires that the iterator points to contiguous memory.
      init(Ptr, &*IdxBegin, NumIdx, NameStr); // FIXME: for the general case
                                     // we have to build an array here
    }
    else {
      init(Ptr, 0, NumIdx, NameStr);
    }
  }

  /// getIndexedType - Returns the type of the element that would be loaded with
  /// a load instruction with the specified parameters.
  ///
  /// Null is returned if the indices are invalid for the specified
  /// pointer type.
  ///
  template<typename InputIterator>
  static const Type *getIndexedType(const Type *Ptr,
                                    InputIterator IdxBegin,
                                    InputIterator IdxEnd,
                                    // This argument ensures that we
                                    // have an iterator we can do
                                    // arithmetic on in constant time
                                    std::random_access_iterator_tag) {
    unsigned NumIdx = static_cast<unsigned>(std::distance(IdxBegin, IdxEnd));

    if (NumIdx > 0)
      // This requires that the iterator points to contiguous memory.
      return getIndexedType(Ptr, &*IdxBegin, NumIdx);
    else
      return getIndexedType(Ptr, (Value *const*)0, NumIdx);
  }

  /// Constructors - Create a getelementptr instruction with a base pointer an
  /// list of indices.  The first ctor can optionally insert before an existing
  /// instruction, the second appends the new instruction to the specified
  /// BasicBlock.
  template<typename InputIterator>
  inline GetElementPtrInst(Value *Ptr, InputIterator IdxBegin,
                           InputIterator IdxEnd,
                           unsigned Values,
                           const Twine &NameStr,
                           Instruction *InsertBefore);
  template<typename InputIterator>
  inline GetElementPtrInst(Value *Ptr,
                           InputIterator IdxBegin, InputIterator IdxEnd,
                           unsigned Values,
                           const Twine &NameStr, BasicBlock *InsertAtEnd);

  /// Constructors - These two constructors are convenience methods because one
  /// and two index getelementptr instructions are so common.
  GetElementPtrInst(Value *Ptr, Value *Idx, const Twine &NameStr = "",
                    Instruction *InsertBefore = 0);
  GetElementPtrInst(Value *Ptr, Value *Idx,
                    const Twine &NameStr, BasicBlock *InsertAtEnd);
protected:
  virtual GetElementPtrInst *clone_impl() const;
public:
  template<typename InputIterator>
  static GetElementPtrInst *Create(Value *Ptr, InputIterator IdxBegin,
                                   InputIterator IdxEnd,
                                   const Twine &NameStr = "",
                                   Instruction *InsertBefore = 0) {
    typename std::iterator_traits<InputIterator>::difference_type Values =
      1 + std::distance(IdxBegin, IdxEnd);
    return new(Values)
      GetElementPtrInst(Ptr, IdxBegin, IdxEnd, Values, NameStr, InsertBefore);
  }
  template<typename InputIterator>
  static GetElementPtrInst *Create(Value *Ptr,
                                   InputIterator IdxBegin, InputIterator IdxEnd,
                                   const Twine &NameStr,
                                   BasicBlock *InsertAtEnd) {
    typename std::iterator_traits<InputIterator>::difference_type Values =
      1 + std::distance(IdxBegin, IdxEnd);
    return new(Values)
      GetElementPtrInst(Ptr, IdxBegin, IdxEnd, Values, NameStr, InsertAtEnd);
  }

  /// Constructors - These two creators are convenience methods because one
  /// index getelementptr instructions are so common.
  static GetElementPtrInst *Create(Value *Ptr, Value *Idx,
                                   const Twine &NameStr = "",
                                   Instruction *InsertBefore = 0) {
    return new(2) GetElementPtrInst(Ptr, Idx, NameStr, InsertBefore);
  }
  static GetElementPtrInst *Create(Value *Ptr, Value *Idx,
                                   const Twine &NameStr,
                                   BasicBlock *InsertAtEnd) {
    return new(2) GetElementPtrInst(Ptr, Idx, NameStr, InsertAtEnd);
  }

  /// Create an "inbounds" getelementptr. See the documentation for the
  /// "inbounds" flag in LangRef.html for details.
  template<typename InputIterator>
  static GetElementPtrInst *CreateInBounds(Value *Ptr, InputIterator IdxBegin,
                                           InputIterator IdxEnd,
                                           const Twine &NameStr = "",
                                           Instruction *InsertBefore = 0) {
    GetElementPtrInst *GEP = Create(Ptr, IdxBegin, IdxEnd,
                                    NameStr, InsertBefore);
    GEP->setIsInBounds(true);
    return GEP;
  }
  template<typename InputIterator>
  static GetElementPtrInst *CreateInBounds(Value *Ptr,
                                           InputIterator IdxBegin,
                                           InputIterator IdxEnd,
                                           const Twine &NameStr,
                                           BasicBlock *InsertAtEnd) {
    GetElementPtrInst *GEP = Create(Ptr, IdxBegin, IdxEnd,
                                    NameStr, InsertAtEnd);
    GEP->setIsInBounds(true);
    return GEP;
  }
  static GetElementPtrInst *CreateInBounds(Value *Ptr, Value *Idx,
                                           const Twine &NameStr = "",
                                           Instruction *InsertBefore = 0) {
    GetElementPtrInst *GEP = Create(Ptr, Idx, NameStr, InsertBefore);
    GEP->setIsInBounds(true);
    return GEP;
  }
  static GetElementPtrInst *CreateInBounds(Value *Ptr, Value *Idx,
                                           const Twine &NameStr,
                                           BasicBlock *InsertAtEnd) {
    GetElementPtrInst *GEP = Create(Ptr, Idx, NameStr, InsertAtEnd);
    GEP->setIsInBounds(true);
    return GEP;
  }

  /// Transparently provide more efficient getOperand methods.
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);

  // getType - Overload to return most specific pointer type...
  const PointerType *getType() const {
    return reinterpret_cast<const PointerType*>(Instruction::getType());
  }

  /// getIndexedType - Returns the type of the element that would be loaded with
  /// a load instruction with the specified parameters.
  ///
  /// Null is returned if the indices are invalid for the specified
  /// pointer type.
  ///
  template<typename InputIterator>
  static const Type *getIndexedType(const Type *Ptr,
                                    InputIterator IdxBegin,
                                    InputIterator IdxEnd) {
    return getIndexedType(Ptr, IdxBegin, IdxEnd,
                          typename std::iterator_traits<InputIterator>::
                          iterator_category());
  }

  static const Type *getIndexedType(const Type *Ptr,
                                    Value* const *Idx, unsigned NumIdx);

  static const Type *getIndexedType(const Type *Ptr,
                                    uint64_t const *Idx, unsigned NumIdx);

  static const Type *getIndexedType(const Type *Ptr, Value *Idx);

  inline op_iterator       idx_begin()       { return op_begin()+1; }
  inline const_op_iterator idx_begin() const { return op_begin()+1; }
  inline op_iterator       idx_end()         { return op_end(); }
  inline const_op_iterator idx_end()   const { return op_end(); }

  Value *getPointerOperand() {
    return getOperand(0);
  }
  const Value *getPointerOperand() const {
    return getOperand(0);
  }
  static unsigned getPointerOperandIndex() {
    return 0U;                      // get index for modifying correct operand
  }
  
  unsigned getPointerAddressSpace() const {
    return cast<PointerType>(getType())->getAddressSpace();
  }

  /// getPointerOperandType - Method to return the pointer operand as a
  /// PointerType.
  const PointerType *getPointerOperandType() const {
    return reinterpret_cast<const PointerType*>(getPointerOperand()->getType());
  }


  unsigned getNumIndices() const {  // Note: always non-negative
    return getNumOperands() - 1;
  }

  bool hasIndices() const {
    return getNumOperands() > 1;
  }

  /// hasAllZeroIndices - Return true if all of the indices of this GEP are
  /// zeros.  If so, the result pointer and the first operand have the same
  /// value, just potentially different types.
  bool hasAllZeroIndices() const;

  /// hasAllConstantIndices - Return true if all of the indices of this GEP are
  /// constant integers.  If so, the result pointer and the first operand have
  /// a constant offset between them.
  bool hasAllConstantIndices() const;

  /// setIsInBounds - Set or clear the inbounds flag on this GEP instruction.
  /// See LangRef.html for the meaning of inbounds on a getelementptr.
  void setIsInBounds(bool b = true);

  /// isInBounds - Determine whether the GEP has the inbounds flag.
  bool isInBounds() const;

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const GetElementPtrInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return (I->getOpcode() == Instruction::GetElementPtr);
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

template <>
struct OperandTraits<GetElementPtrInst> : public VariadicOperandTraits<1> {
};

template<typename InputIterator>
GetElementPtrInst::GetElementPtrInst(Value *Ptr,
                                     InputIterator IdxBegin,
                                     InputIterator IdxEnd,
                                     unsigned Values,
                                     const Twine &NameStr,
                                     Instruction *InsertBefore)
  : Instruction(PointerType::get(checkType(
                                   getIndexedType(Ptr->getType(),
                                                  IdxBegin, IdxEnd)),
                                 cast<PointerType>(Ptr->getType())
                                   ->getAddressSpace()),
                GetElementPtr,
                OperandTraits<GetElementPtrInst>::op_end(this) - Values,
                Values, InsertBefore) {
  init(Ptr, IdxBegin, IdxEnd, NameStr,
       typename std::iterator_traits<InputIterator>::iterator_category());
}
template<typename InputIterator>
GetElementPtrInst::GetElementPtrInst(Value *Ptr,
                                     InputIterator IdxBegin,
                                     InputIterator IdxEnd,
                                     unsigned Values,
                                     const Twine &NameStr,
                                     BasicBlock *InsertAtEnd)
  : Instruction(PointerType::get(checkType(
                                   getIndexedType(Ptr->getType(),
                                                  IdxBegin, IdxEnd)),
                                 cast<PointerType>(Ptr->getType())
                                   ->getAddressSpace()),
                GetElementPtr,
                OperandTraits<GetElementPtrInst>::op_end(this) - Values,
                Values, InsertAtEnd) {
  init(Ptr, IdxBegin, IdxEnd, NameStr,
       typename std::iterator_traits<InputIterator>::iterator_category());
}


DEFINE_TRANSPARENT_OPERAND_ACCESSORS(GetElementPtrInst, Value)


//===----------------------------------------------------------------------===//
//                               ICmpInst Class
//===----------------------------------------------------------------------===//

/// This instruction compares its operands according to the predicate given
/// to the constructor. It only operates on integers or pointers. The operands
/// must be identical types.
/// @brief Represent an integer comparison operator.
class ICmpInst: public CmpInst {
protected:
  /// @brief Clone an indentical ICmpInst
  virtual ICmpInst *clone_impl() const;  
public:
  /// @brief Constructor with insert-before-instruction semantics.
  ICmpInst(
    Instruction *InsertBefore,  ///< Where to insert
    Predicate pred,  ///< The predicate to use for the comparison
    Value *LHS,      ///< The left-hand-side of the expression
    Value *RHS,      ///< The right-hand-side of the expression
    const Twine &NameStr = ""  ///< Name of the instruction
  ) : CmpInst(makeCmpResultType(LHS->getType()),
              Instruction::ICmp, pred, LHS, RHS, NameStr,
              InsertBefore) {
    assert(pred >= CmpInst::FIRST_ICMP_PREDICATE &&
           pred <= CmpInst::LAST_ICMP_PREDICATE &&
           "Invalid ICmp predicate value");
    assert(getOperand(0)->getType() == getOperand(1)->getType() &&
          "Both operands to ICmp instruction are not of the same type!");
    // Check that the operands are the right type
    assert((getOperand(0)->getType()->isIntOrIntVectorTy() ||
            isa<PointerType>(getOperand(0)->getType())) &&
           "Invalid operand types for ICmp instruction");
  }

  /// @brief Constructor with insert-at-end semantics.
  ICmpInst(
    BasicBlock &InsertAtEnd, ///< Block to insert into.
    Predicate pred,  ///< The predicate to use for the comparison
    Value *LHS,      ///< The left-hand-side of the expression
    Value *RHS,      ///< The right-hand-side of the expression
    const Twine &NameStr = ""  ///< Name of the instruction
  ) : CmpInst(makeCmpResultType(LHS->getType()),
              Instruction::ICmp, pred, LHS, RHS, NameStr,
              &InsertAtEnd) {
    assert(pred >= CmpInst::FIRST_ICMP_PREDICATE &&
          pred <= CmpInst::LAST_ICMP_PREDICATE &&
          "Invalid ICmp predicate value");
    assert(getOperand(0)->getType() == getOperand(1)->getType() &&
          "Both operands to ICmp instruction are not of the same type!");
    // Check that the operands are the right type
    assert((getOperand(0)->getType()->isIntOrIntVectorTy() ||
            isa<PointerType>(getOperand(0)->getType())) &&
           "Invalid operand types for ICmp instruction");
  }

  /// @brief Constructor with no-insertion semantics
  ICmpInst(
    Predicate pred, ///< The predicate to use for the comparison
    Value *LHS,     ///< The left-hand-side of the expression
    Value *RHS,     ///< The right-hand-side of the expression
    const Twine &NameStr = "" ///< Name of the instruction
  ) : CmpInst(makeCmpResultType(LHS->getType()),
              Instruction::ICmp, pred, LHS, RHS, NameStr) {
    assert(pred >= CmpInst::FIRST_ICMP_PREDICATE &&
           pred <= CmpInst::LAST_ICMP_PREDICATE &&
           "Invalid ICmp predicate value");
    assert(getOperand(0)->getType() == getOperand(1)->getType() &&
          "Both operands to ICmp instruction are not of the same type!");
    // Check that the operands are the right type
    assert((getOperand(0)->getType()->isIntOrIntVectorTy() ||
            isa<PointerType>(getOperand(0)->getType())) &&
           "Invalid operand types for ICmp instruction");
  }

  /// For example, EQ->EQ, SLE->SLE, UGT->SGT, etc.
  /// @returns the predicate that would be the result if the operand were
  /// regarded as signed.
  /// @brief Return the signed version of the predicate
  Predicate getSignedPredicate() const {
    return getSignedPredicate(getPredicate());
  }

  /// This is a static version that you can use without an instruction.
  /// @brief Return the signed version of the predicate.
  static Predicate getSignedPredicate(Predicate pred);

  /// For example, EQ->EQ, SLE->ULE, UGT->UGT, etc.
  /// @returns the predicate that would be the result if the operand were
  /// regarded as unsigned.
  /// @brief Return the unsigned version of the predicate
  Predicate getUnsignedPredicate() const {
    return getUnsignedPredicate(getPredicate());
  }

  /// This is a static version that you can use without an instruction.
  /// @brief Return the unsigned version of the predicate.
  static Predicate getUnsignedPredicate(Predicate pred);

  /// isEquality - Return true if this predicate is either EQ or NE.  This also
  /// tests for commutativity.
  static bool isEquality(Predicate P) {
    return P == ICMP_EQ || P == ICMP_NE;
  }

  /// isEquality - Return true if this predicate is either EQ or NE.  This also
  /// tests for commutativity.
  bool isEquality() const {
    return isEquality(getPredicate());
  }

  /// @returns true if the predicate of this ICmpInst is commutative
  /// @brief Determine if this relation is commutative.
  bool isCommutative() const { return isEquality(); }

  /// isRelational - Return true if the predicate is relational (not EQ or NE).
  ///
  bool isRelational() const {
    return !isEquality();
  }

  /// isRelational - Return true if the predicate is relational (not EQ or NE).
  ///
  static bool isRelational(Predicate P) {
    return !isEquality(P);
  }

  /// Initialize a set of values that all satisfy the predicate with C.
  /// @brief Make a ConstantRange for a relation with a constant value.
  static ConstantRange makeConstantRange(Predicate pred, const APInt &C);

  /// Exchange the two operands to this instruction in such a way that it does
  /// not modify the semantics of the instruction. The predicate value may be
  /// changed to retain the same result if the predicate is order dependent
  /// (e.g. ult).
  /// @brief Swap operands and adjust predicate.
  void swapOperands() {
    setPredicate(getSwappedPredicate());
    Op<0>().swap(Op<1>());
  }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ICmpInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::ICmp;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }

};

//===----------------------------------------------------------------------===//
//                               FCmpInst Class
//===----------------------------------------------------------------------===//

/// This instruction compares its operands according to the predicate given
/// to the constructor. It only operates on floating point values or packed
/// vectors of floating point values. The operands must be identical types.
/// @brief Represents a floating point comparison operator.
class FCmpInst: public CmpInst {
protected:
  /// @brief Clone an indentical FCmpInst
  virtual FCmpInst *clone_impl() const;
public:
  /// @brief Constructor with insert-before-instruction semantics.
  FCmpInst(
    Instruction *InsertBefore, ///< Where to insert
    Predicate pred,  ///< The predicate to use for the comparison
    Value *LHS,      ///< The left-hand-side of the expression
    Value *RHS,      ///< The right-hand-side of the expression
    const Twine &NameStr = ""  ///< Name of the instruction
  ) : CmpInst(makeCmpResultType(LHS->getType()),
              Instruction::FCmp, pred, LHS, RHS, NameStr,
              InsertBefore) {
    assert(pred <= FCmpInst::LAST_FCMP_PREDICATE &&
           "Invalid FCmp predicate value");
    assert(getOperand(0)->getType() == getOperand(1)->getType() &&
           "Both operands to FCmp instruction are not of the same type!");
    // Check that the operands are the right type
    assert(getOperand(0)->getType()->isFPOrFPVectorTy() &&
           "Invalid operand types for FCmp instruction");
  }
  
  /// @brief Constructor with insert-at-end semantics.
  FCmpInst(
    BasicBlock &InsertAtEnd, ///< Block to insert into.
    Predicate pred,  ///< The predicate to use for the comparison
    Value *LHS,      ///< The left-hand-side of the expression
    Value *RHS,      ///< The right-hand-side of the expression
    const Twine &NameStr = ""  ///< Name of the instruction
  ) : CmpInst(makeCmpResultType(LHS->getType()),
              Instruction::FCmp, pred, LHS, RHS, NameStr,
              &InsertAtEnd) {
    assert(pred <= FCmpInst::LAST_FCMP_PREDICATE &&
           "Invalid FCmp predicate value");
    assert(getOperand(0)->getType() == getOperand(1)->getType() &&
           "Both operands to FCmp instruction are not of the same type!");
    // Check that the operands are the right type
    assert(getOperand(0)->getType()->isFPOrFPVectorTy() &&
           "Invalid operand types for FCmp instruction");
  }

  /// @brief Constructor with no-insertion semantics
  FCmpInst(
    Predicate pred, ///< The predicate to use for the comparison
    Value *LHS,     ///< The left-hand-side of the expression
    Value *RHS,     ///< The right-hand-side of the expression
    const Twine &NameStr = "" ///< Name of the instruction
  ) : CmpInst(makeCmpResultType(LHS->getType()),
              Instruction::FCmp, pred, LHS, RHS, NameStr) {
    assert(pred <= FCmpInst::LAST_FCMP_PREDICATE &&
           "Invalid FCmp predicate value");
    assert(getOperand(0)->getType() == getOperand(1)->getType() &&
           "Both operands to FCmp instruction are not of the same type!");
    // Check that the operands are the right type
    assert(getOperand(0)->getType()->isFPOrFPVectorTy() &&
           "Invalid operand types for FCmp instruction");
  }

  /// @returns true if the predicate of this instruction is EQ or NE.
  /// @brief Determine if this is an equality predicate.
  bool isEquality() const {
    return getPredicate() == FCMP_OEQ || getPredicate() == FCMP_ONE ||
           getPredicate() == FCMP_UEQ || getPredicate() == FCMP_UNE;
  }

  /// @returns true if the predicate of this instruction is commutative.
  /// @brief Determine if this is a commutative predicate.
  bool isCommutative() const {
    return isEquality() ||
           getPredicate() == FCMP_FALSE ||
           getPredicate() == FCMP_TRUE ||
           getPredicate() == FCMP_ORD ||
           getPredicate() == FCMP_UNO;
  }

  /// @returns true if the predicate is relational (not EQ or NE).
  /// @brief Determine if this a relational predicate.
  bool isRelational() const { return !isEquality(); }

  /// Exchange the two operands to this instruction in such a way that it does
  /// not modify the semantics of the instruction. The predicate value may be
  /// changed to retain the same result if the predicate is order dependent
  /// (e.g. ult).
  /// @brief Swap operands and adjust predicate.
  void swapOperands() {
    setPredicate(getSwappedPredicate());
    Op<0>().swap(Op<1>());
  }

  /// @brief Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const FCmpInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::FCmp;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

//===----------------------------------------------------------------------===//
/// CallInst - This class represents a function call, abstracting a target
/// machine's calling convention.  This class uses low bit of the SubClassData
/// field to indicate whether or not this is a tail call.  The rest of the bits
/// hold the calling convention of the call.
///
class CallInst : public Instruction {
  AttrListPtr AttributeList; ///< parameter attributes for call
  CallInst(const CallInst &CI);
  void init(Value *Func, Value* const *Params, unsigned NumParams);
  void init(Value *Func, Value *Actual1, Value *Actual2);
  void init(Value *Func, Value *Actual);
  void init(Value *Func);

  template<typename InputIterator>
  void init(Value *Func, InputIterator ArgBegin, InputIterator ArgEnd,
            const Twine &NameStr,
            // This argument ensures that we have an iterator we can
            // do arithmetic on in constant time
            std::random_access_iterator_tag) {
    unsigned NumArgs = (unsigned)std::distance(ArgBegin, ArgEnd);

    // This requires that the iterator points to contiguous memory.
    init(Func, NumArgs ? &*ArgBegin : 0, NumArgs);
    setName(NameStr);
  }

  /// Construct a CallInst given a range of arguments.  InputIterator
  /// must be a random-access iterator pointing to contiguous storage
  /// (e.g. a std::vector<>::iterator).  Checks are made for
  /// random-accessness but not for contiguous storage as that would
  /// incur runtime overhead.
  /// @brief Construct a CallInst from a range of arguments
  template<typename InputIterator>
  CallInst(Value *Func, InputIterator ArgBegin, InputIterator ArgEnd,
           const Twine &NameStr, Instruction *InsertBefore);

  /// Construct a CallInst given a range of arguments.  InputIterator
  /// must be a random-access iterator pointing to contiguous storage
  /// (e.g. a std::vector<>::iterator).  Checks are made for
  /// random-accessness but not for contiguous storage as that would
  /// incur runtime overhead.
  /// @brief Construct a CallInst from a range of arguments
  template<typename InputIterator>
  inline CallInst(Value *Func, InputIterator ArgBegin, InputIterator ArgEnd,
                  const Twine &NameStr, BasicBlock *InsertAtEnd);

  CallInst(Value *F, Value *Actual, const Twine &NameStr,
           Instruction *InsertBefore);
  CallInst(Value *F, Value *Actual, const Twine &NameStr,
           BasicBlock *InsertAtEnd);
  explicit CallInst(Value *F, const Twine &NameStr,
                    Instruction *InsertBefore);
  CallInst(Value *F, const Twine &NameStr, BasicBlock *InsertAtEnd);
protected:
  virtual CallInst *clone_impl() const;
public:
  template<typename InputIterator>
  static CallInst *Create(Value *Func,
                          InputIterator ArgBegin, InputIterator ArgEnd,
                          const Twine &NameStr = "",
                          Instruction *InsertBefore = 0) {
    return new((unsigned)(ArgEnd - ArgBegin + 1))
      CallInst(Func, ArgBegin, ArgEnd, NameStr, InsertBefore);
  }
  template<typename InputIterator>
  static CallInst *Create(Value *Func,
                          InputIterator ArgBegin, InputIterator ArgEnd,
                          const Twine &NameStr, BasicBlock *InsertAtEnd) {
    return new((unsigned)(ArgEnd - ArgBegin + 1))
      CallInst(Func, ArgBegin, ArgEnd, NameStr, InsertAtEnd);
  }
  static CallInst *Create(Value *F, Value *Actual,
                          const Twine &NameStr = "",
                          Instruction *InsertBefore = 0) {
    return new(2) CallInst(F, Actual, NameStr, InsertBefore);
  }
  static CallInst *Create(Value *F, Value *Actual, const Twine &NameStr,
                          BasicBlock *InsertAtEnd) {
    return new(2) CallInst(F, Actual, NameStr, InsertAtEnd);
  }
  static CallInst *Create(Value *F, const Twine &NameStr = "",
                          Instruction *InsertBefore = 0) {
    return new(1) CallInst(F, NameStr, InsertBefore);
  }
  static CallInst *Create(Value *F, const Twine &NameStr,
                          BasicBlock *InsertAtEnd) {
    return new(1) CallInst(F, NameStr, InsertAtEnd);
  }
  /// CreateMalloc - Generate the IR for a call to malloc:
  /// 1. Compute the malloc call's argument as the specified type's size,
  ///    possibly multiplied by the array size if the array size is not
  ///    constant 1.
  /// 2. Call malloc with that argument.
  /// 3. Bitcast the result of the malloc call to the specified type.
  static Instruction *CreateMalloc(Instruction *InsertBefore,
                                   const Type *IntPtrTy, const Type *AllocTy,
                                   Value *AllocSize, Value *ArraySize = 0,
                                   const Twine &Name = "");
  static Instruction *CreateMalloc(BasicBlock *InsertAtEnd,
                                   const Type *IntPtrTy, const Type *AllocTy,
                                   Value *AllocSize, Value *ArraySize = 0,
                                   Function* MallocF = 0,
                                   const Twine &Name = "");
  /// CreateFree - Generate the IR for a call to the builtin free function.
  static void CreateFree(Value* Source, Instruction *InsertBefore);
  static Instruction* CreateFree(Value* Source, BasicBlock *InsertAtEnd);

  ~CallInst();

  bool isTailCall() const { return getSubclassDataFromInstruction() & 1; }
  void setTailCall(bool isTC = true) {
    setInstructionSubclassData((getSubclassDataFromInstruction() & ~1) |
                               unsigned(isTC));
  }

  /// Provide fast operand accessors
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);

  /// getCallingConv/setCallingConv - Get or set the calling convention of this
  /// function call.
  CallingConv::ID getCallingConv() const {
    return static_cast<CallingConv::ID>(getSubclassDataFromInstruction() >> 1);
  }
  void setCallingConv(CallingConv::ID CC) {
    setInstructionSubclassData((getSubclassDataFromInstruction() & 1) |
                               (static_cast<unsigned>(CC) << 1));
  }

  /// getAttributes - Return the parameter attributes for this call.
  ///
  const AttrListPtr &getAttributes() const { return AttributeList; }

  /// setAttributes - Set the parameter attributes for this call.
  ///
  void setAttributes(const AttrListPtr &Attrs) { AttributeList = Attrs; }

  /// addAttribute - adds the attribute to the list of attributes.
  void addAttribute(unsigned i, Attributes attr);

  /// removeAttribute - removes the attribute from the list of attributes.
  void removeAttribute(unsigned i, Attributes attr);

  /// @brief Determine whether the call or the callee has the given attribute.
  bool paramHasAttr(unsigned i, Attributes attr) const;

  /// @brief Extract the alignment for a call or parameter (0=unknown).
  unsigned getParamAlignment(unsigned i) const {
    return AttributeList.getParamAlignment(i);
  }

  /// @brief Determine if the call does not access memory.
  bool doesNotAccessMemory() const {
    return paramHasAttr(~0, Attribute::ReadNone);
  }
  void setDoesNotAccessMemory(bool NotAccessMemory = true) {
    if (NotAccessMemory) addAttribute(~0, Attribute::ReadNone);
    else removeAttribute(~0, Attribute::ReadNone);
  }

  /// @brief Determine if the call does not access or only reads memory.
  bool onlyReadsMemory() const {
    return doesNotAccessMemory() || paramHasAttr(~0, Attribute::ReadOnly);
  }
  void setOnlyReadsMemory(bool OnlyReadsMemory = true) {
    if (OnlyReadsMemory) addAttribute(~0, Attribute::ReadOnly);
    else removeAttribute(~0, Attribute::ReadOnly | Attribute::ReadNone);
  }

  /// @brief Determine if the call cannot return.
  bool doesNotReturn() const {
    return paramHasAttr(~0, Attribute::NoReturn);
  }
  void setDoesNotReturn(bool DoesNotReturn = true) {
    if (DoesNotReturn) addAttribute(~0, Attribute::NoReturn);
    else removeAttribute(~0, Attribute::NoReturn);
  }

  /// @brief Determine if the call cannot unwind.
  bool doesNotThrow() const {
    return paramHasAttr(~0, Attribute::NoUnwind);
  }
  void setDoesNotThrow(bool DoesNotThrow = true) {
    if (DoesNotThrow) addAttribute(~0, Attribute::NoUnwind);
    else removeAttribute(~0, Attribute::NoUnwind);
  }

  /// @brief Determine if the call returns a structure through first
  /// pointer argument.
  bool hasStructRetAttr() const {
    // Be friendly and also check the callee.
    return paramHasAttr(1, Attribute::StructRet);
  }

  /// @brief Determine if any call argument is an aggregate passed by value.
  bool hasByValArgument() const {
    return AttributeList.hasAttrSomewhere(Attribute::ByVal);
  }

  /// getCalledFunction - Return the function called, or null if this is an
  /// indirect function invocation.
  ///
  Function *getCalledFunction() const {
    return dyn_cast<Function>(Op<0>());
  }

  /// getCalledValue - Get a pointer to the function that is invoked by this
  /// instruction.
  const Value *getCalledValue() const { return Op<0>(); }
        Value *getCalledValue()       { return Op<0>(); }

  /// setCalledFunction - Set the function called.
  void setCalledFunction(Value* Fn) {
    Op<0>() = Fn;
  }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const CallInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::Call;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
private:
  // Shadow Instruction::setInstructionSubclassData with a private forwarding
  // method so that subclasses cannot accidentally use it.
  void setInstructionSubclassData(unsigned short D) {
    Instruction::setInstructionSubclassData(D);
  }
};

template <>
struct OperandTraits<CallInst> : public VariadicOperandTraits<1> {
};

template<typename InputIterator>
CallInst::CallInst(Value *Func, InputIterator ArgBegin, InputIterator ArgEnd,
                   const Twine &NameStr, BasicBlock *InsertAtEnd)
  : Instruction(cast<FunctionType>(cast<PointerType>(Func->getType())
                                   ->getElementType())->getReturnType(),
                Instruction::Call,
                OperandTraits<CallInst>::op_end(this) - (ArgEnd - ArgBegin + 1),
                (unsigned)(ArgEnd - ArgBegin + 1), InsertAtEnd) {
  init(Func, ArgBegin, ArgEnd, NameStr,
       typename std::iterator_traits<InputIterator>::iterator_category());
}

template<typename InputIterator>
CallInst::CallInst(Value *Func, InputIterator ArgBegin, InputIterator ArgEnd,
                   const Twine &NameStr, Instruction *InsertBefore)
  : Instruction(cast<FunctionType>(cast<PointerType>(Func->getType())
                                   ->getElementType())->getReturnType(),
                Instruction::Call,
                OperandTraits<CallInst>::op_end(this) - (ArgEnd - ArgBegin + 1),
                (unsigned)(ArgEnd - ArgBegin + 1), InsertBefore) {
  init(Func, ArgBegin, ArgEnd, NameStr,
       typename std::iterator_traits<InputIterator>::iterator_category());
}

DEFINE_TRANSPARENT_OPERAND_ACCESSORS(CallInst, Value)

//===----------------------------------------------------------------------===//
//                               SelectInst Class
//===----------------------------------------------------------------------===//

/// SelectInst - This class represents the LLVM 'select' instruction.
///
class SelectInst : public Instruction {
  void init(Value *C, Value *S1, Value *S2) {
    assert(!areInvalidOperands(C, S1, S2) && "Invalid operands for select");
    Op<0>() = C;
    Op<1>() = S1;
    Op<2>() = S2;
  }

  SelectInst(Value *C, Value *S1, Value *S2, const Twine &NameStr,
             Instruction *InsertBefore)
    : Instruction(S1->getType(), Instruction::Select,
                  &Op<0>(), 3, InsertBefore) {
    init(C, S1, S2);
    setName(NameStr);
  }
  SelectInst(Value *C, Value *S1, Value *S2, const Twine &NameStr,
             BasicBlock *InsertAtEnd)
    : Instruction(S1->getType(), Instruction::Select,
                  &Op<0>(), 3, InsertAtEnd) {
    init(C, S1, S2);
    setName(NameStr);
  }
protected:
  virtual SelectInst *clone_impl() const;
public:
  static SelectInst *Create(Value *C, Value *S1, Value *S2,
                            const Twine &NameStr = "",
                            Instruction *InsertBefore = 0) {
    return new(3) SelectInst(C, S1, S2, NameStr, InsertBefore);
  }
  static SelectInst *Create(Value *C, Value *S1, Value *S2,
                            const Twine &NameStr,
                            BasicBlock *InsertAtEnd) {
    return new(3) SelectInst(C, S1, S2, NameStr, InsertAtEnd);
  }

  const Value *getCondition() const { return Op<0>(); }
  const Value *getTrueValue() const { return Op<1>(); }
  const Value *getFalseValue() const { return Op<2>(); }
  Value *getCondition() { return Op<0>(); }
  Value *getTrueValue() { return Op<1>(); }
  Value *getFalseValue() { return Op<2>(); }
  
  /// areInvalidOperands - Return a string if the specified operands are invalid
  /// for a select operation, otherwise return null.
  static const char *areInvalidOperands(Value *Cond, Value *True, Value *False);

  /// Transparently provide more efficient getOperand methods.
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);

  OtherOps getOpcode() const {
    return static_cast<OtherOps>(Instruction::getOpcode());
  }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const SelectInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::Select;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

template <>
struct OperandTraits<SelectInst> : public FixedNumOperandTraits<3> {
};

DEFINE_TRANSPARENT_OPERAND_ACCESSORS(SelectInst, Value)

//===----------------------------------------------------------------------===//
//                                VAArgInst Class
//===----------------------------------------------------------------------===//

/// VAArgInst - This class represents the va_arg llvm instruction, which returns
/// an argument of the specified type given a va_list and increments that list
///
class VAArgInst : public UnaryInstruction {
protected:
  virtual VAArgInst *clone_impl() const;

public:
  VAArgInst(Value *List, const Type *Ty, const Twine &NameStr = "",
             Instruction *InsertBefore = 0)
    : UnaryInstruction(Ty, VAArg, List, InsertBefore) {
    setName(NameStr);
  }
  VAArgInst(Value *List, const Type *Ty, const Twine &NameStr,
            BasicBlock *InsertAtEnd)
    : UnaryInstruction(Ty, VAArg, List, InsertAtEnd) {
    setName(NameStr);
  }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const VAArgInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == VAArg;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

//===----------------------------------------------------------------------===//
//                                ExtractElementInst Class
//===----------------------------------------------------------------------===//

/// ExtractElementInst - This instruction extracts a single (scalar)
/// element from a VectorType value
///
class ExtractElementInst : public Instruction {
  ExtractElementInst(Value *Vec, Value *Idx, const Twine &NameStr = "",
                     Instruction *InsertBefore = 0);
  ExtractElementInst(Value *Vec, Value *Idx, const Twine &NameStr,
                     BasicBlock *InsertAtEnd);
protected:
  virtual ExtractElementInst *clone_impl() const;

public:
  static ExtractElementInst *Create(Value *Vec, Value *Idx,
                                   const Twine &NameStr = "",
                                   Instruction *InsertBefore = 0) {
    return new(2) ExtractElementInst(Vec, Idx, NameStr, InsertBefore);
  }
  static ExtractElementInst *Create(Value *Vec, Value *Idx,
                                   const Twine &NameStr,
                                   BasicBlock *InsertAtEnd) {
    return new(2) ExtractElementInst(Vec, Idx, NameStr, InsertAtEnd);
  }

  /// isValidOperands - Return true if an extractelement instruction can be
  /// formed with the specified operands.
  static bool isValidOperands(const Value *Vec, const Value *Idx);

  Value *getVectorOperand() { return Op<0>(); }
  Value *getIndexOperand() { return Op<1>(); }
  const Value *getVectorOperand() const { return Op<0>(); }
  const Value *getIndexOperand() const { return Op<1>(); }
  
  const VectorType *getVectorOperandType() const {
    return reinterpret_cast<const VectorType*>(getVectorOperand()->getType());
  }
  
  
  /// Transparently provide more efficient getOperand methods.
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ExtractElementInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::ExtractElement;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

template <>
struct OperandTraits<ExtractElementInst> : public FixedNumOperandTraits<2> {
};

DEFINE_TRANSPARENT_OPERAND_ACCESSORS(ExtractElementInst, Value)

//===----------------------------------------------------------------------===//
//                                InsertElementInst Class
//===----------------------------------------------------------------------===//

/// InsertElementInst - This instruction inserts a single (scalar)
/// element into a VectorType value
///
class InsertElementInst : public Instruction {
  InsertElementInst(Value *Vec, Value *NewElt, Value *Idx,
                    const Twine &NameStr = "",
                    Instruction *InsertBefore = 0);
  InsertElementInst(Value *Vec, Value *NewElt, Value *Idx,
                    const Twine &NameStr, BasicBlock *InsertAtEnd);
protected:
  virtual InsertElementInst *clone_impl() const;

public:
  static InsertElementInst *Create(Value *Vec, Value *NewElt, Value *Idx,
                                   const Twine &NameStr = "",
                                   Instruction *InsertBefore = 0) {
    return new(3) InsertElementInst(Vec, NewElt, Idx, NameStr, InsertBefore);
  }
  static InsertElementInst *Create(Value *Vec, Value *NewElt, Value *Idx,
                                   const Twine &NameStr,
                                   BasicBlock *InsertAtEnd) {
    return new(3) InsertElementInst(Vec, NewElt, Idx, NameStr, InsertAtEnd);
  }

  /// isValidOperands - Return true if an insertelement instruction can be
  /// formed with the specified operands.
  static bool isValidOperands(const Value *Vec, const Value *NewElt,
                              const Value *Idx);

  /// getType - Overload to return most specific vector type.
  ///
  const VectorType *getType() const {
    return reinterpret_cast<const VectorType*>(Instruction::getType());
  }

  /// Transparently provide more efficient getOperand methods.
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const InsertElementInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::InsertElement;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

template <>
struct OperandTraits<InsertElementInst> : public FixedNumOperandTraits<3> {
};

DEFINE_TRANSPARENT_OPERAND_ACCESSORS(InsertElementInst, Value)

//===----------------------------------------------------------------------===//
//                           ShuffleVectorInst Class
//===----------------------------------------------------------------------===//

/// ShuffleVectorInst - This instruction constructs a fixed permutation of two
/// input vectors.
///
class ShuffleVectorInst : public Instruction {
protected:
  virtual ShuffleVectorInst *clone_impl() const;

public:
  // allocate space for exactly three operands
  void *operator new(size_t s) {
    return User::operator new(s, 3);
  }
  ShuffleVectorInst(Value *V1, Value *V2, Value *Mask,
                    const Twine &NameStr = "",
                    Instruction *InsertBefor = 0);
  ShuffleVectorInst(Value *V1, Value *V2, Value *Mask,
                    const Twine &NameStr, BasicBlock *InsertAtEnd);

  /// isValidOperands - Return true if a shufflevector instruction can be
  /// formed with the specified operands.
  static bool isValidOperands(const Value *V1, const Value *V2,
                              const Value *Mask);

  /// getType - Overload to return most specific vector type.
  ///
  const VectorType *getType() const {
    return reinterpret_cast<const VectorType*>(Instruction::getType());
  }

  /// Transparently provide more efficient getOperand methods.
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);

  /// getMaskValue - Return the index from the shuffle mask for the specified
  /// output result.  This is either -1 if the element is undef or a number less
  /// than 2*numelements.
  int getMaskValue(unsigned i) const;

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ShuffleVectorInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::ShuffleVector;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

template <>
struct OperandTraits<ShuffleVectorInst> : public FixedNumOperandTraits<3> {
};

DEFINE_TRANSPARENT_OPERAND_ACCESSORS(ShuffleVectorInst, Value)

//===----------------------------------------------------------------------===//
//                                ExtractValueInst Class
//===----------------------------------------------------------------------===//

/// ExtractValueInst - This instruction extracts a struct member or array
/// element value from an aggregate value.
///
class ExtractValueInst : public UnaryInstruction {
  SmallVector<unsigned, 4> Indices;

  ExtractValueInst(const ExtractValueInst &EVI);
  void init(const unsigned *Idx, unsigned NumIdx,
            const Twine &NameStr);
  void init(unsigned Idx, const Twine &NameStr);

  template<typename InputIterator>
  void init(InputIterator IdxBegin, InputIterator IdxEnd,
            const Twine &NameStr,
            // This argument ensures that we have an iterator we can
            // do arithmetic on in constant time
            std::random_access_iterator_tag) {
    unsigned NumIdx = static_cast<unsigned>(std::distance(IdxBegin, IdxEnd));

    // There's no fundamental reason why we require at least one index
    // (other than weirdness with &*IdxBegin being invalid; see
    // getelementptr's init routine for example). But there's no
    // present need to support it.
    assert(NumIdx > 0 && "ExtractValueInst must have at least one index");

    // This requires that the iterator points to contiguous memory.
    init(&*IdxBegin, NumIdx, NameStr); // FIXME: for the general case
                                         // we have to build an array here
  }

  /// getIndexedType - Returns the type of the element that would be extracted
  /// with an extractvalue instruction with the specified parameters.
  ///
  /// Null is returned if the indices are invalid for the specified
  /// pointer type.
  ///
  static const Type *getIndexedType(const Type *Agg,
                                    const unsigned *Idx, unsigned NumIdx);

  template<typename InputIterator>
  static const Type *getIndexedType(const Type *Ptr,
                                    InputIterator IdxBegin,
                                    InputIterator IdxEnd,
                                    // This argument ensures that we
                                    // have an iterator we can do
                                    // arithmetic on in constant time
                                    std::random_access_iterator_tag) {
    unsigned NumIdx = static_cast<unsigned>(std::distance(IdxBegin, IdxEnd));

    if (NumIdx > 0)
      // This requires that the iterator points to contiguous memory.
      return getIndexedType(Ptr, &*IdxBegin, NumIdx);
    else
      return getIndexedType(Ptr, (const unsigned *)0, NumIdx);
  }

  /// Constructors - Create a extractvalue instruction with a base aggregate
  /// value and a list of indices.  The first ctor can optionally insert before
  /// an existing instruction, the second appends the new instruction to the
  /// specified BasicBlock.
  template<typename InputIterator>
  inline ExtractValueInst(Value *Agg, InputIterator IdxBegin,
                          InputIterator IdxEnd,
                          const Twine &NameStr,
                          Instruction *InsertBefore);
  template<typename InputIterator>
  inline ExtractValueInst(Value *Agg,
                          InputIterator IdxBegin, InputIterator IdxEnd,
                          const Twine &NameStr, BasicBlock *InsertAtEnd);

  // allocate space for exactly one operand
  void *operator new(size_t s) {
    return User::operator new(s, 1);
  }
protected:
  virtual ExtractValueInst *clone_impl() const;

public:
  template<typename InputIterator>
  static ExtractValueInst *Create(Value *Agg, InputIterator IdxBegin,
                                  InputIterator IdxEnd,
                                  const Twine &NameStr = "",
                                  Instruction *InsertBefore = 0) {
    return new
      ExtractValueInst(Agg, IdxBegin, IdxEnd, NameStr, InsertBefore);
  }
  template<typename InputIterator>
  static ExtractValueInst *Create(Value *Agg,
                                  InputIterator IdxBegin, InputIterator IdxEnd,
                                  const Twine &NameStr,
                                  BasicBlock *InsertAtEnd) {
    return new ExtractValueInst(Agg, IdxBegin, IdxEnd, NameStr, InsertAtEnd);
  }

  /// Constructors - These two creators are convenience methods because one
  /// index extractvalue instructions are much more common than those with
  /// more than one.
  static ExtractValueInst *Create(Value *Agg, unsigned Idx,
                                  const Twine &NameStr = "",
                                  Instruction *InsertBefore = 0) {
    unsigned Idxs[1] = { Idx };
    return new ExtractValueInst(Agg, Idxs, Idxs + 1, NameStr, InsertBefore);
  }
  static ExtractValueInst *Create(Value *Agg, unsigned Idx,
                                  const Twine &NameStr,
                                  BasicBlock *InsertAtEnd) {
    unsigned Idxs[1] = { Idx };
    return new ExtractValueInst(Agg, Idxs, Idxs + 1, NameStr, InsertAtEnd);
  }

  /// getIndexedType - Returns the type of the element that would be extracted
  /// with an extractvalue instruction with the specified parameters.
  ///
  /// Null is returned if the indices are invalid for the specified
  /// pointer type.
  ///
  template<typename InputIterator>
  static const Type *getIndexedType(const Type *Ptr,
                                    InputIterator IdxBegin,
                                    InputIterator IdxEnd) {
    return getIndexedType(Ptr, IdxBegin, IdxEnd,
                          typename std::iterator_traits<InputIterator>::
                          iterator_category());
  }
  static const Type *getIndexedType(const Type *Ptr, unsigned Idx);

  typedef const unsigned* idx_iterator;
  inline idx_iterator idx_begin() const { return Indices.begin(); }
  inline idx_iterator idx_end()   const { return Indices.end(); }

  Value *getAggregateOperand() {
    return getOperand(0);
  }
  const Value *getAggregateOperand() const {
    return getOperand(0);
  }
  static unsigned getAggregateOperandIndex() {
    return 0U;                      // get index for modifying correct operand
  }

  unsigned getNumIndices() const {  // Note: always non-negative
    return (unsigned)Indices.size();
  }

  bool hasIndices() const {
    return true;
  }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ExtractValueInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::ExtractValue;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

template<typename InputIterator>
ExtractValueInst::ExtractValueInst(Value *Agg,
                                   InputIterator IdxBegin,
                                   InputIterator IdxEnd,
                                   const Twine &NameStr,
                                   Instruction *InsertBefore)
  : UnaryInstruction(checkType(getIndexedType(Agg->getType(),
                                              IdxBegin, IdxEnd)),
                     ExtractValue, Agg, InsertBefore) {
  init(IdxBegin, IdxEnd, NameStr,
       typename std::iterator_traits<InputIterator>::iterator_category());
}
template<typename InputIterator>
ExtractValueInst::ExtractValueInst(Value *Agg,
                                   InputIterator IdxBegin,
                                   InputIterator IdxEnd,
                                   const Twine &NameStr,
                                   BasicBlock *InsertAtEnd)
  : UnaryInstruction(checkType(getIndexedType(Agg->getType(),
                                              IdxBegin, IdxEnd)),
                     ExtractValue, Agg, InsertAtEnd) {
  init(IdxBegin, IdxEnd, NameStr,
       typename std::iterator_traits<InputIterator>::iterator_category());
}


//===----------------------------------------------------------------------===//
//                                InsertValueInst Class
//===----------------------------------------------------------------------===//

/// InsertValueInst - This instruction inserts a struct field of array element
/// value into an aggregate value.
///
class InsertValueInst : public Instruction {
  SmallVector<unsigned, 4> Indices;

  void *operator new(size_t, unsigned); // Do not implement
  InsertValueInst(const InsertValueInst &IVI);
  void init(Value *Agg, Value *Val, const unsigned *Idx, unsigned NumIdx,
            const Twine &NameStr);
  void init(Value *Agg, Value *Val, unsigned Idx, const Twine &NameStr);

  template<typename InputIterator>
  void init(Value *Agg, Value *Val,
            InputIterator IdxBegin, InputIterator IdxEnd,
            const Twine &NameStr,
            // This argument ensures that we have an iterator we can
            // do arithmetic on in constant time
            std::random_access_iterator_tag) {
    unsigned NumIdx = static_cast<unsigned>(std::distance(IdxBegin, IdxEnd));

    // There's no fundamental reason why we require at least one index
    // (other than weirdness with &*IdxBegin being invalid; see
    // getelementptr's init routine for example). But there's no
    // present need to support it.
    assert(NumIdx > 0 && "InsertValueInst must have at least one index");

    // This requires that the iterator points to contiguous memory.
    init(Agg, Val, &*IdxBegin, NumIdx, NameStr); // FIXME: for the general case
                                              // we have to build an array here
  }

  /// Constructors - Create a insertvalue instruction with a base aggregate
  /// value, a value to insert, and a list of indices.  The first ctor can
  /// optionally insert before an existing instruction, the second appends
  /// the new instruction to the specified BasicBlock.
  template<typename InputIterator>
  inline InsertValueInst(Value *Agg, Value *Val, InputIterator IdxBegin,
                         InputIterator IdxEnd,
                         const Twine &NameStr,
                         Instruction *InsertBefore);
  template<typename InputIterator>
  inline InsertValueInst(Value *Agg, Value *Val,
                         InputIterator IdxBegin, InputIterator IdxEnd,
                         const Twine &NameStr, BasicBlock *InsertAtEnd);

  /// Constructors - These two constructors are convenience methods because one
  /// and two index insertvalue instructions are so common.
  InsertValueInst(Value *Agg, Value *Val,
                  unsigned Idx, const Twine &NameStr = "",
                  Instruction *InsertBefore = 0);
  InsertValueInst(Value *Agg, Value *Val, unsigned Idx,
                  const Twine &NameStr, BasicBlock *InsertAtEnd);
protected:
  virtual InsertValueInst *clone_impl() const;
public:
  // allocate space for exactly two operands
  void *operator new(size_t s) {
    return User::operator new(s, 2);
  }

  template<typename InputIterator>
  static InsertValueInst *Create(Value *Agg, Value *Val, InputIterator IdxBegin,
                                 InputIterator IdxEnd,
                                 const Twine &NameStr = "",
                                 Instruction *InsertBefore = 0) {
    return new InsertValueInst(Agg, Val, IdxBegin, IdxEnd,
                               NameStr, InsertBefore);
  }
  template<typename InputIterator>
  static InsertValueInst *Create(Value *Agg, Value *Val,
                                 InputIterator IdxBegin, InputIterator IdxEnd,
                                 const Twine &NameStr,
                                 BasicBlock *InsertAtEnd) {
    return new InsertValueInst(Agg, Val, IdxBegin, IdxEnd,
                               NameStr, InsertAtEnd);
  }

  /// Constructors - These two creators are convenience methods because one
  /// index insertvalue instructions are much more common than those with
  /// more than one.
  static InsertValueInst *Create(Value *Agg, Value *Val, unsigned Idx,
                                 const Twine &NameStr = "",
                                 Instruction *InsertBefore = 0) {
    return new InsertValueInst(Agg, Val, Idx, NameStr, InsertBefore);
  }
  static InsertValueInst *Create(Value *Agg, Value *Val, unsigned Idx,
                                 const Twine &NameStr,
                                 BasicBlock *InsertAtEnd) {
    return new InsertValueInst(Agg, Val, Idx, NameStr, InsertAtEnd);
  }

  /// Transparently provide more efficient getOperand methods.
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);

  typedef const unsigned* idx_iterator;
  inline idx_iterator idx_begin() const { return Indices.begin(); }
  inline idx_iterator idx_end()   const { return Indices.end(); }

  Value *getAggregateOperand() {
    return getOperand(0);
  }
  const Value *getAggregateOperand() const {
    return getOperand(0);
  }
  static unsigned getAggregateOperandIndex() {
    return 0U;                      // get index for modifying correct operand
  }

  Value *getInsertedValueOperand() {
    return getOperand(1);
  }
  const Value *getInsertedValueOperand() const {
    return getOperand(1);
  }
  static unsigned getInsertedValueOperandIndex() {
    return 1U;                      // get index for modifying correct operand
  }

  unsigned getNumIndices() const {  // Note: always non-negative
    return (unsigned)Indices.size();
  }

  bool hasIndices() const {
    return true;
  }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const InsertValueInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::InsertValue;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

template <>
struct OperandTraits<InsertValueInst> : public FixedNumOperandTraits<2> {
};

template<typename InputIterator>
InsertValueInst::InsertValueInst(Value *Agg,
                                 Value *Val,
                                 InputIterator IdxBegin,
                                 InputIterator IdxEnd,
                                 const Twine &NameStr,
                                 Instruction *InsertBefore)
  : Instruction(Agg->getType(), InsertValue,
                OperandTraits<InsertValueInst>::op_begin(this),
                2, InsertBefore) {
  init(Agg, Val, IdxBegin, IdxEnd, NameStr,
       typename std::iterator_traits<InputIterator>::iterator_category());
}
template<typename InputIterator>
InsertValueInst::InsertValueInst(Value *Agg,
                                 Value *Val,
                                 InputIterator IdxBegin,
                                 InputIterator IdxEnd,
                                 const Twine &NameStr,
                                 BasicBlock *InsertAtEnd)
  : Instruction(Agg->getType(), InsertValue,
                OperandTraits<InsertValueInst>::op_begin(this),
                2, InsertAtEnd) {
  init(Agg, Val, IdxBegin, IdxEnd, NameStr,
       typename std::iterator_traits<InputIterator>::iterator_category());
}

DEFINE_TRANSPARENT_OPERAND_ACCESSORS(InsertValueInst, Value)

//===----------------------------------------------------------------------===//
//                               PHINode Class
//===----------------------------------------------------------------------===//

// PHINode - The PHINode class is used to represent the magical mystical PHI
// node, that can not exist in nature, but can be synthesized in a computer
// scientist's overactive imagination.
//
class PHINode : public Instruction {
  void *operator new(size_t, unsigned);  // DO NOT IMPLEMENT
  /// ReservedSpace - The number of operands actually allocated.  NumOperands is
  /// the number actually in use.
  unsigned ReservedSpace;
  PHINode(const PHINode &PN);
  // allocate space for exactly zero operands
  void *operator new(size_t s) {
    return User::operator new(s, 0);
  }
  explicit PHINode(const Type *Ty, const Twine &NameStr = "",
                   Instruction *InsertBefore = 0)
    : Instruction(Ty, Instruction::PHI, 0, 0, InsertBefore),
      ReservedSpace(0) {
    setName(NameStr);
  }

  PHINode(const Type *Ty, const Twine &NameStr, BasicBlock *InsertAtEnd)
    : Instruction(Ty, Instruction::PHI, 0, 0, InsertAtEnd),
      ReservedSpace(0) {
    setName(NameStr);
  }
protected:
  virtual PHINode *clone_impl() const;
public:
  static PHINode *Create(const Type *Ty, const Twine &NameStr = "",
                         Instruction *InsertBefore = 0) {
    return new PHINode(Ty, NameStr, InsertBefore);
  }
  static PHINode *Create(const Type *Ty, const Twine &NameStr,
                         BasicBlock *InsertAtEnd) {
    return new PHINode(Ty, NameStr, InsertAtEnd);
  }
  ~PHINode();

  /// reserveOperandSpace - This method can be used to avoid repeated
  /// reallocation of PHI operand lists by reserving space for the correct
  /// number of operands before adding them.  Unlike normal vector reserves,
  /// this method can also be used to trim the operand space.
  void reserveOperandSpace(unsigned NumValues) {
    resizeOperands(NumValues*2);
  }

  /// Provide fast operand accessors
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);

  /// getNumIncomingValues - Return the number of incoming edges
  ///
  unsigned getNumIncomingValues() const { return getNumOperands()/2; }

  /// getIncomingValue - Return incoming value number x
  ///
  Value *getIncomingValue(unsigned i) const {
    assert(i*2 < getNumOperands() && "Invalid value number!");
    return getOperand(i*2);
  }
  void setIncomingValue(unsigned i, Value *V) {
    assert(i*2 < getNumOperands() && "Invalid value number!");
    setOperand(i*2, V);
  }
  static unsigned getOperandNumForIncomingValue(unsigned i) {
    return i*2;
  }
  static unsigned getIncomingValueNumForOperand(unsigned i) {
    assert(i % 2 == 0 && "Invalid incoming-value operand index!");
    return i/2;
  }

  /// getIncomingBlock - Return incoming basic block #i.
  ///
  BasicBlock *getIncomingBlock(unsigned i) const {
    return cast<BasicBlock>(getOperand(i*2+1));
  }
  
  /// getIncomingBlock - Return incoming basic block corresponding
  /// to an operand of the PHI.
  ///
  BasicBlock *getIncomingBlock(const Use &U) const {
    assert(this == U.getUser() && "Iterator doesn't point to PHI's Uses?");
    return cast<BasicBlock>((&U + 1)->get());
  }
  
  /// getIncomingBlock - Return incoming basic block corresponding
  /// to value use iterator.
  ///
  template <typename U>
  BasicBlock *getIncomingBlock(value_use_iterator<U> I) const {
    return getIncomingBlock(I.getUse());
  }
  
  
  void setIncomingBlock(unsigned i, BasicBlock *BB) {
    setOperand(i*2+1, (Value*)BB);
  }
  static unsigned getOperandNumForIncomingBlock(unsigned i) {
    return i*2+1;
  }
  static unsigned getIncomingBlockNumForOperand(unsigned i) {
    assert(i % 2 == 1 && "Invalid incoming-block operand index!");
    return i/2;
  }

  /// addIncoming - Add an incoming value to the end of the PHI list
  ///
  void addIncoming(Value *V, BasicBlock *BB) {
    assert(V && "PHI node got a null value!");
    assert(BB && "PHI node got a null basic block!");
    assert(getType() == V->getType() &&
           "All operands to PHI node must be the same type as the PHI node!");
    unsigned OpNo = NumOperands;
    if (OpNo+2 > ReservedSpace)
      resizeOperands(0);  // Get more space!
    // Initialize some new operands.
    NumOperands = OpNo+2;
    OperandList[OpNo] = V;
    OperandList[OpNo+1] = (Value*)BB;
  }

  /// removeIncomingValue - Remove an incoming value.  This is useful if a
  /// predecessor basic block is deleted.  The value removed is returned.
  ///
  /// If the last incoming value for a PHI node is removed (and DeletePHIIfEmpty
  /// is true), the PHI node is destroyed and any uses of it are replaced with
  /// dummy values.  The only time there should be zero incoming values to a PHI
  /// node is when the block is dead, so this strategy is sound.
  ///
  Value *removeIncomingValue(unsigned Idx, bool DeletePHIIfEmpty = true);

  Value *removeIncomingValue(const BasicBlock *BB, bool DeletePHIIfEmpty=true) {
    int Idx = getBasicBlockIndex(BB);
    assert(Idx >= 0 && "Invalid basic block argument to remove!");
    return removeIncomingValue(Idx, DeletePHIIfEmpty);
  }

  /// getBasicBlockIndex - Return the first index of the specified basic
  /// block in the value list for this PHI.  Returns -1 if no instance.
  ///
  int getBasicBlockIndex(const BasicBlock *BB) const {
    Use *OL = OperandList;
    for (unsigned i = 0, e = getNumOperands(); i != e; i += 2)
      if (OL[i+1].get() == (const Value*)BB) return i/2;
    return -1;
  }

  Value *getIncomingValueForBlock(const BasicBlock *BB) const {
    return getIncomingValue(getBasicBlockIndex(BB));
  }

  /// hasConstantValue - If the specified PHI node always merges together the
  /// same value, return the value, otherwise return null.
  ///
  /// If the PHI has undef operands, but all the rest of the operands are
  /// some unique value, return that value if it can be proved that the
  /// value dominates the PHI. If DT is null, use a conservative check,
  /// otherwise use DT to test for dominance.
  ///
  Value *hasConstantValue(DominatorTree *DT = 0) const;

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const PHINode *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::PHI;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
 private:
  void resizeOperands(unsigned NumOperands);
};

template <>
struct OperandTraits<PHINode> : public HungoffOperandTraits<2> {
};

DEFINE_TRANSPARENT_OPERAND_ACCESSORS(PHINode, Value)


//===----------------------------------------------------------------------===//
//                               ReturnInst Class
//===----------------------------------------------------------------------===//

//===---------------------------------------------------------------------------
/// ReturnInst - Return a value (possibly void), from a function.  Execution
/// does not continue in this function any longer.
///
class ReturnInst : public TerminatorInst {
  ReturnInst(const ReturnInst &RI);

private:
  // ReturnInst constructors:
  // ReturnInst()                  - 'ret void' instruction
  // ReturnInst(    null)          - 'ret void' instruction
  // ReturnInst(Value* X)          - 'ret X'    instruction
  // ReturnInst(    null, Inst *I) - 'ret void' instruction, insert before I
  // ReturnInst(Value* X, Inst *I) - 'ret X'    instruction, insert before I
  // ReturnInst(    null, BB *B)   - 'ret void' instruction, insert @ end of B
  // ReturnInst(Value* X, BB *B)   - 'ret X'    instruction, insert @ end of B
  //
  // NOTE: If the Value* passed is of type void then the constructor behaves as
  // if it was passed NULL.
  explicit ReturnInst(LLVMContext &C, Value *retVal = 0,
                      Instruction *InsertBefore = 0);
  ReturnInst(LLVMContext &C, Value *retVal, BasicBlock *InsertAtEnd);
  explicit ReturnInst(LLVMContext &C, BasicBlock *InsertAtEnd);
protected:
  virtual ReturnInst *clone_impl() const;
public:
  static ReturnInst* Create(LLVMContext &C, Value *retVal = 0,
                            Instruction *InsertBefore = 0) {
    return new(!!retVal) ReturnInst(C, retVal, InsertBefore);
  }
  static ReturnInst* Create(LLVMContext &C, Value *retVal,
                            BasicBlock *InsertAtEnd) {
    return new(!!retVal) ReturnInst(C, retVal, InsertAtEnd);
  }
  static ReturnInst* Create(LLVMContext &C, BasicBlock *InsertAtEnd) {
    return new(0) ReturnInst(C, InsertAtEnd);
  }
  virtual ~ReturnInst();

  /// Provide fast operand accessors
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);

  /// Convenience accessor
  Value *getReturnValue(unsigned n = 0) const {
    return n < getNumOperands()
      ? getOperand(n)
      : 0;
  }

  unsigned getNumSuccessors() const { return 0; }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ReturnInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return (I->getOpcode() == Instruction::Ret);
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
 private:
  virtual BasicBlock *getSuccessorV(unsigned idx) const;
  virtual unsigned getNumSuccessorsV() const;
  virtual void setSuccessorV(unsigned idx, BasicBlock *B);
};

template <>
struct OperandTraits<ReturnInst> : public OptionalOperandTraits<> {
};

DEFINE_TRANSPARENT_OPERAND_ACCESSORS(ReturnInst, Value)

//===----------------------------------------------------------------------===//
//                               BranchInst Class
//===----------------------------------------------------------------------===//

//===---------------------------------------------------------------------------
/// BranchInst - Conditional or Unconditional Branch instruction.
///
class BranchInst : public TerminatorInst {
  /// Ops list - Branches are strange.  The operands are ordered:
  ///  [Cond, FalseDest,] TrueDest.  This makes some accessors faster because
  /// they don't have to check for cond/uncond branchness. These are mostly
  /// accessed relative from op_end().
  BranchInst(const BranchInst &BI);
  void AssertOK();
  // BranchInst constructors (where {B, T, F} are blocks, and C is a condition):
  // BranchInst(BB *B)                           - 'br B'
  // BranchInst(BB* T, BB *F, Value *C)          - 'br C, T, F'
  // BranchInst(BB* B, Inst *I)                  - 'br B'        insert before I
  // BranchInst(BB* T, BB *F, Value *C, Inst *I) - 'br C, T, F', insert before I
  // BranchInst(BB* B, BB *I)                    - 'br B'        insert at end
  // BranchInst(BB* T, BB *F, Value *C, BB *I)   - 'br C, T, F', insert at end
  explicit BranchInst(BasicBlock *IfTrue, Instruction *InsertBefore = 0);
  BranchInst(BasicBlock *IfTrue, BasicBlock *IfFalse, Value *Cond,
             Instruction *InsertBefore = 0);
  BranchInst(BasicBlock *IfTrue, BasicBlock *InsertAtEnd);
  BranchInst(BasicBlock *IfTrue, BasicBlock *IfFalse, Value *Cond,
             BasicBlock *InsertAtEnd);
protected:
  virtual BranchInst *clone_impl() const;
public:
  static BranchInst *Create(BasicBlock *IfTrue, Instruction *InsertBefore = 0) {
    return new(1, true) BranchInst(IfTrue, InsertBefore);
  }
  static BranchInst *Create(BasicBlock *IfTrue, BasicBlock *IfFalse,
                            Value *Cond, Instruction *InsertBefore = 0) {
    return new(3) BranchInst(IfTrue, IfFalse, Cond, InsertBefore);
  }
  static BranchInst *Create(BasicBlock *IfTrue, BasicBlock *InsertAtEnd) {
    return new(1, true) BranchInst(IfTrue, InsertAtEnd);
  }
  static BranchInst *Create(BasicBlock *IfTrue, BasicBlock *IfFalse,
                            Value *Cond, BasicBlock *InsertAtEnd) {
    return new(3) BranchInst(IfTrue, IfFalse, Cond, InsertAtEnd);
  }

  ~BranchInst();

  /// Transparently provide more efficient getOperand methods.
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);

  bool isUnconditional() const { return getNumOperands() == 1; }
  bool isConditional()   const { return getNumOperands() == 3; }

  Value *getCondition() const {
    assert(isConditional() && "Cannot get condition of an uncond branch!");
    return Op<-3>();
  }

  void setCondition(Value *V) {
    assert(isConditional() && "Cannot set condition of unconditional branch!");
    Op<-3>() = V;
  }

  // setUnconditionalDest - Change the current branch to an unconditional branch
  // targeting the specified block.
  // FIXME: Eliminate this ugly method.
  void setUnconditionalDest(BasicBlock *Dest) {
    Op<-1>() = (Value*)Dest;
    if (isConditional()) {  // Convert this to an uncond branch.
      Op<-2>() = 0;
      Op<-3>() = 0;
      NumOperands = 1;
      OperandList = op_begin();
    }
  }

  unsigned getNumSuccessors() const { return 1+isConditional(); }

  BasicBlock *getSuccessor(unsigned i) const {
    assert(i < getNumSuccessors() && "Successor # out of range for Branch!");
    return cast_or_null<BasicBlock>((&Op<-1>() - i)->get());
  }

  void setSuccessor(unsigned idx, BasicBlock *NewSucc) {
    assert(idx < getNumSuccessors() && "Successor # out of range for Branch!");
    *(&Op<-1>() - idx) = (Value*)NewSucc;
  }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const BranchInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return (I->getOpcode() == Instruction::Br);
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
private:
  virtual BasicBlock *getSuccessorV(unsigned idx) const;
  virtual unsigned getNumSuccessorsV() const;
  virtual void setSuccessorV(unsigned idx, BasicBlock *B);
};

template <>
struct OperandTraits<BranchInst> : public VariadicOperandTraits<1> {};

DEFINE_TRANSPARENT_OPERAND_ACCESSORS(BranchInst, Value)

//===----------------------------------------------------------------------===//
//                               SwitchInst Class
//===----------------------------------------------------------------------===//

//===---------------------------------------------------------------------------
/// SwitchInst - Multiway switch
///
class SwitchInst : public TerminatorInst {
  void *operator new(size_t, unsigned);  // DO NOT IMPLEMENT
  unsigned ReservedSpace;
  // Operand[0]    = Value to switch on
  // Operand[1]    = Default basic block destination
  // Operand[2n  ] = Value to match
  // Operand[2n+1] = BasicBlock to go to on match
  SwitchInst(const SwitchInst &SI);
  void init(Value *Value, BasicBlock *Default, unsigned NumCases);
  void resizeOperands(unsigned No);
  // allocate space for exactly zero operands
  void *operator new(size_t s) {
    return User::operator new(s, 0);
  }
  /// SwitchInst ctor - Create a new switch instruction, specifying a value to
  /// switch on and a default destination.  The number of additional cases can
  /// be specified here to make memory allocation more efficient.  This
  /// constructor can also autoinsert before another instruction.
  SwitchInst(Value *Value, BasicBlock *Default, unsigned NumCases,
             Instruction *InsertBefore);

  /// SwitchInst ctor - Create a new switch instruction, specifying a value to
  /// switch on and a default destination.  The number of additional cases can
  /// be specified here to make memory allocation more efficient.  This
  /// constructor also autoinserts at the end of the specified BasicBlock.
  SwitchInst(Value *Value, BasicBlock *Default, unsigned NumCases,
             BasicBlock *InsertAtEnd);
protected:
  virtual SwitchInst *clone_impl() const;
public:
  static SwitchInst *Create(Value *Value, BasicBlock *Default,
                            unsigned NumCases, Instruction *InsertBefore = 0) {
    return new SwitchInst(Value, Default, NumCases, InsertBefore);
  }
  static SwitchInst *Create(Value *Value, BasicBlock *Default,
                            unsigned NumCases, BasicBlock *InsertAtEnd) {
    return new SwitchInst(Value, Default, NumCases, InsertAtEnd);
  }
  ~SwitchInst();

  /// Provide fast operand accessors
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);

  // Accessor Methods for Switch stmt
  Value *getCondition() const { return getOperand(0); }
  void setCondition(Value *V) { setOperand(0, V); }

  BasicBlock *getDefaultDest() const {
    return cast<BasicBlock>(getOperand(1));
  }

  /// getNumCases - return the number of 'cases' in this switch instruction.
  /// Note that case #0 is always the default case.
  unsigned getNumCases() const {
    return getNumOperands()/2;
  }

  /// getCaseValue - Return the specified case value.  Note that case #0, the
  /// default destination, does not have a case value.
  ConstantInt *getCaseValue(unsigned i) {
    assert(i && i < getNumCases() && "Illegal case value to get!");
    return getSuccessorValue(i);
  }

  /// getCaseValue - Return the specified case value.  Note that case #0, the
  /// default destination, does not have a case value.
  const ConstantInt *getCaseValue(unsigned i) const {
    assert(i && i < getNumCases() && "Illegal case value to get!");
    return getSuccessorValue(i);
  }

  /// findCaseValue - Search all of the case values for the specified constant.
  /// If it is explicitly handled, return the case number of it, otherwise
  /// return 0 to indicate that it is handled by the default handler.
  unsigned findCaseValue(const ConstantInt *C) const {
    for (unsigned i = 1, e = getNumCases(); i != e; ++i)
      if (getCaseValue(i) == C)
        return i;
    return 0;
  }

  /// findCaseDest - Finds the unique case value for a given successor. Returns
  /// null if the successor is not found, not unique, or is the default case.
  ConstantInt *findCaseDest(BasicBlock *BB) {
    if (BB == getDefaultDest()) return NULL;

    ConstantInt *CI = NULL;
    for (unsigned i = 1, e = getNumCases(); i != e; ++i) {
      if (getSuccessor(i) == BB) {
        if (CI) return NULL;   // Multiple cases lead to BB.
        else CI = getCaseValue(i);
      }
    }
    return CI;
  }

  /// addCase - Add an entry to the switch instruction...
  ///
  void addCase(ConstantInt *OnVal, BasicBlock *Dest);

  /// removeCase - This method removes the specified successor from the switch
  /// instruction.  Note that this cannot be used to remove the default
  /// destination (successor #0).
  ///
  void removeCase(unsigned idx);

  unsigned getNumSuccessors() const { return getNumOperands()/2; }
  BasicBlock *getSuccessor(unsigned idx) const {
    assert(idx < getNumSuccessors() &&"Successor idx out of range for switch!");
    return cast<BasicBlock>(getOperand(idx*2+1));
  }
  void setSuccessor(unsigned idx, BasicBlock *NewSucc) {
    assert(idx < getNumSuccessors() && "Successor # out of range for switch!");
    setOperand(idx*2+1, (Value*)NewSucc);
  }

  // getSuccessorValue - Return the value associated with the specified
  // successor.
  ConstantInt *getSuccessorValue(unsigned idx) const {
    assert(idx < getNumSuccessors() && "Successor # out of range!");
    return reinterpret_cast<ConstantInt*>(getOperand(idx*2));
  }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const SwitchInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::Switch;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
private:
  virtual BasicBlock *getSuccessorV(unsigned idx) const;
  virtual unsigned getNumSuccessorsV() const;
  virtual void setSuccessorV(unsigned idx, BasicBlock *B);
};

template <>
struct OperandTraits<SwitchInst> : public HungoffOperandTraits<2> {
};

DEFINE_TRANSPARENT_OPERAND_ACCESSORS(SwitchInst, Value)


//===----------------------------------------------------------------------===//
//                             IndirectBrInst Class
//===----------------------------------------------------------------------===//

//===---------------------------------------------------------------------------
/// IndirectBrInst - Indirect Branch Instruction.
///
class IndirectBrInst : public TerminatorInst {
  void *operator new(size_t, unsigned);  // DO NOT IMPLEMENT
  unsigned ReservedSpace;
  // Operand[0]    = Value to switch on
  // Operand[1]    = Default basic block destination
  // Operand[2n  ] = Value to match
  // Operand[2n+1] = BasicBlock to go to on match
  IndirectBrInst(const IndirectBrInst &IBI);
  void init(Value *Address, unsigned NumDests);
  void resizeOperands(unsigned No);
  // allocate space for exactly zero operands
  void *operator new(size_t s) {
    return User::operator new(s, 0);
  }
  /// IndirectBrInst ctor - Create a new indirectbr instruction, specifying an
  /// Address to jump to.  The number of expected destinations can be specified
  /// here to make memory allocation more efficient.  This constructor can also
  /// autoinsert before another instruction.
  IndirectBrInst(Value *Address, unsigned NumDests, Instruction *InsertBefore);
  
  /// IndirectBrInst ctor - Create a new indirectbr instruction, specifying an
  /// Address to jump to.  The number of expected destinations can be specified
  /// here to make memory allocation more efficient.  This constructor also
  /// autoinserts at the end of the specified BasicBlock.
  IndirectBrInst(Value *Address, unsigned NumDests, BasicBlock *InsertAtEnd);
protected:
  virtual IndirectBrInst *clone_impl() const;
public:
  static IndirectBrInst *Create(Value *Address, unsigned NumDests,
                                Instruction *InsertBefore = 0) {
    return new IndirectBrInst(Address, NumDests, InsertBefore);
  }
  static IndirectBrInst *Create(Value *Address, unsigned NumDests,
                                BasicBlock *InsertAtEnd) {
    return new IndirectBrInst(Address, NumDests, InsertAtEnd);
  }
  ~IndirectBrInst();
  
  /// Provide fast operand accessors.
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);
  
  // Accessor Methods for IndirectBrInst instruction.
  Value *getAddress() { return getOperand(0); }
  const Value *getAddress() const { return getOperand(0); }
  void setAddress(Value *V) { setOperand(0, V); }
  
  
  /// getNumDestinations - return the number of possible destinations in this
  /// indirectbr instruction.
  unsigned getNumDestinations() const { return getNumOperands()-1; }
  
  /// getDestination - Return the specified destination.
  BasicBlock *getDestination(unsigned i) { return getSuccessor(i); }
  const BasicBlock *getDestination(unsigned i) const { return getSuccessor(i); }
  
  /// addDestination - Add a destination.
  ///
  void addDestination(BasicBlock *Dest);
  
  /// removeDestination - This method removes the specified successor from the
  /// indirectbr instruction.
  void removeDestination(unsigned i);
  
  unsigned getNumSuccessors() const { return getNumOperands()-1; }
  BasicBlock *getSuccessor(unsigned i) const {
    return cast<BasicBlock>(getOperand(i+1));
  }
  void setSuccessor(unsigned i, BasicBlock *NewSucc) {
    setOperand(i+1, (Value*)NewSucc);
  }
  
  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const IndirectBrInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::IndirectBr;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
private:
  virtual BasicBlock *getSuccessorV(unsigned idx) const;
  virtual unsigned getNumSuccessorsV() const;
  virtual void setSuccessorV(unsigned idx, BasicBlock *B);
};

template <>
struct OperandTraits<IndirectBrInst> : public HungoffOperandTraits<1> {
};

DEFINE_TRANSPARENT_OPERAND_ACCESSORS(IndirectBrInst, Value)
  
  
//===----------------------------------------------------------------------===//
//                               InvokeInst Class
//===----------------------------------------------------------------------===//

/// InvokeInst - Invoke instruction.  The SubclassData field is used to hold the
/// calling convention of the call.
///
class InvokeInst : public TerminatorInst {
  AttrListPtr AttributeList;
  InvokeInst(const InvokeInst &BI);
  void init(Value *Fn, BasicBlock *IfNormal, BasicBlock *IfException,
            Value* const *Args, unsigned NumArgs);

  template<typename InputIterator>
  void init(Value *Func, BasicBlock *IfNormal, BasicBlock *IfException,
            InputIterator ArgBegin, InputIterator ArgEnd,
            const Twine &NameStr,
            // This argument ensures that we have an iterator we can
            // do arithmetic on in constant time
            std::random_access_iterator_tag) {
    unsigned NumArgs = (unsigned)std::distance(ArgBegin, ArgEnd);

    // This requires that the iterator points to contiguous memory.
    init(Func, IfNormal, IfException, NumArgs ? &*ArgBegin : 0, NumArgs);
    setName(NameStr);
  }

  /// Construct an InvokeInst given a range of arguments.
  /// InputIterator must be a random-access iterator pointing to
  /// contiguous storage (e.g. a std::vector<>::iterator).  Checks are
  /// made for random-accessness but not for contiguous storage as
  /// that would incur runtime overhead.
  ///
  /// @brief Construct an InvokeInst from a range of arguments
  template<typename InputIterator>
  inline InvokeInst(Value *Func, BasicBlock *IfNormal, BasicBlock *IfException,
                    InputIterator ArgBegin, InputIterator ArgEnd,
                    unsigned Values,
                    const Twine &NameStr, Instruction *InsertBefore);

  /// Construct an InvokeInst given a range of arguments.
  /// InputIterator must be a random-access iterator pointing to
  /// contiguous storage (e.g. a std::vector<>::iterator).  Checks are
  /// made for random-accessness but not for contiguous storage as
  /// that would incur runtime overhead.
  ///
  /// @brief Construct an InvokeInst from a range of arguments
  template<typename InputIterator>
  inline InvokeInst(Value *Func, BasicBlock *IfNormal, BasicBlock *IfException,
                    InputIterator ArgBegin, InputIterator ArgEnd,
                    unsigned Values,
                    const Twine &NameStr, BasicBlock *InsertAtEnd);
protected:
  virtual InvokeInst *clone_impl() const;
public:
  template<typename InputIterator>
  static InvokeInst *Create(Value *Func,
                            BasicBlock *IfNormal, BasicBlock *IfException,
                            InputIterator ArgBegin, InputIterator ArgEnd,
                            const Twine &NameStr = "",
                            Instruction *InsertBefore = 0) {
    unsigned Values(ArgEnd - ArgBegin + 3);
    return new(Values) InvokeInst(Func, IfNormal, IfException, ArgBegin, ArgEnd,
                                  Values, NameStr, InsertBefore);
  }
  template<typename InputIterator>
  static InvokeInst *Create(Value *Func,
                            BasicBlock *IfNormal, BasicBlock *IfException,
                            InputIterator ArgBegin, InputIterator ArgEnd,
                            const Twine &NameStr,
                            BasicBlock *InsertAtEnd) {
    unsigned Values(ArgEnd - ArgBegin + 3);
    return new(Values) InvokeInst(Func, IfNormal, IfException, ArgBegin, ArgEnd,
                                  Values, NameStr, InsertAtEnd);
  }

  /// Provide fast operand accessors
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);

  /// getCallingConv/setCallingConv - Get or set the calling convention of this
  /// function call.
  CallingConv::ID getCallingConv() const {
    return static_cast<CallingConv::ID>(getSubclassDataFromInstruction());
  }
  void setCallingConv(CallingConv::ID CC) {
    setInstructionSubclassData(static_cast<unsigned>(CC));
  }

  /// getAttributes - Return the parameter attributes for this invoke.
  ///
  const AttrListPtr &getAttributes() const { return AttributeList; }

  /// setAttributes - Set the parameter attributes for this invoke.
  ///
  void setAttributes(const AttrListPtr &Attrs) { AttributeList = Attrs; }

  /// addAttribute - adds the attribute to the list of attributes.
  void addAttribute(unsigned i, Attributes attr);

  /// removeAttribute - removes the attribute from the list of attributes.
  void removeAttribute(unsigned i, Attributes attr);

  /// @brief Determine whether the call or the callee has the given attribute.
  bool paramHasAttr(unsigned i, Attributes attr) const;

  /// @brief Extract the alignment for a call or parameter (0=unknown).
  unsigned getParamAlignment(unsigned i) const {
    return AttributeList.getParamAlignment(i);
  }

  /// @brief Determine if the call does not access memory.
  bool doesNotAccessMemory() const {
    return paramHasAttr(~0, Attribute::ReadNone);
  }
  void setDoesNotAccessMemory(bool NotAccessMemory = true) {
    if (NotAccessMemory) addAttribute(~0, Attribute::ReadNone);
    else removeAttribute(~0, Attribute::ReadNone);
  }

  /// @brief Determine if the call does not access or only reads memory.
  bool onlyReadsMemory() const {
    return doesNotAccessMemory() || paramHasAttr(~0, Attribute::ReadOnly);
  }
  void setOnlyReadsMemory(bool OnlyReadsMemory = true) {
    if (OnlyReadsMemory) addAttribute(~0, Attribute::ReadOnly);
    else removeAttribute(~0, Attribute::ReadOnly | Attribute::ReadNone);
  }

  /// @brief Determine if the call cannot return.
  bool doesNotReturn() const {
    return paramHasAttr(~0, Attribute::NoReturn);
  }
  void setDoesNotReturn(bool DoesNotReturn = true) {
    if (DoesNotReturn) addAttribute(~0, Attribute::NoReturn);
    else removeAttribute(~0, Attribute::NoReturn);
  }

  /// @brief Determine if the call cannot unwind.
  bool doesNotThrow() const {
    return paramHasAttr(~0, Attribute::NoUnwind);
  }
  void setDoesNotThrow(bool DoesNotThrow = true) {
    if (DoesNotThrow) addAttribute(~0, Attribute::NoUnwind);
    else removeAttribute(~0, Attribute::NoUnwind);
  }

  /// @brief Determine if the call returns a structure through first
  /// pointer argument.
  bool hasStructRetAttr() const {
    // Be friendly and also check the callee.
    return paramHasAttr(1, Attribute::StructRet);
  }

  /// @brief Determine if any call argument is an aggregate passed by value.
  bool hasByValArgument() const {
    return AttributeList.hasAttrSomewhere(Attribute::ByVal);
  }

  /// getCalledFunction - Return the function called, or null if this is an
  /// indirect function invocation.
  ///
  Function *getCalledFunction() const {
    return dyn_cast<Function>(getOperand(0));
  }

  /// getCalledValue - Get a pointer to the function that is invoked by this
  /// instruction
  const Value *getCalledValue() const { return getOperand(0); }
        Value *getCalledValue()       { return getOperand(0); }

  // get*Dest - Return the destination basic blocks...
  BasicBlock *getNormalDest() const {
    return cast<BasicBlock>(getOperand(1));
  }
  BasicBlock *getUnwindDest() const {
    return cast<BasicBlock>(getOperand(2));
  }
  void setNormalDest(BasicBlock *B) {
    setOperand(1, (Value*)B);
  }

  void setUnwindDest(BasicBlock *B) {
    setOperand(2, (Value*)B);
  }

  BasicBlock *getSuccessor(unsigned i) const {
    assert(i < 2 && "Successor # out of range for invoke!");
    return i == 0 ? getNormalDest() : getUnwindDest();
  }

  void setSuccessor(unsigned idx, BasicBlock *NewSucc) {
    assert(idx < 2 && "Successor # out of range for invoke!");
    setOperand(idx+1, (Value*)NewSucc);
  }

  unsigned getNumSuccessors() const { return 2; }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const InvokeInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return (I->getOpcode() == Instruction::Invoke);
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
private:
  virtual BasicBlock *getSuccessorV(unsigned idx) const;
  virtual unsigned getNumSuccessorsV() const;
  virtual void setSuccessorV(unsigned idx, BasicBlock *B);

  // Shadow Instruction::setInstructionSubclassData with a private forwarding
  // method so that subclasses cannot accidentally use it.
  void setInstructionSubclassData(unsigned short D) {
    Instruction::setInstructionSubclassData(D);
  }
};

template <>
struct OperandTraits<InvokeInst> : public VariadicOperandTraits<3> {
};

template<typename InputIterator>
InvokeInst::InvokeInst(Value *Func,
                       BasicBlock *IfNormal, BasicBlock *IfException,
                       InputIterator ArgBegin, InputIterator ArgEnd,
                       unsigned Values,
                       const Twine &NameStr, Instruction *InsertBefore)
  : TerminatorInst(cast<FunctionType>(cast<PointerType>(Func->getType())
                                      ->getElementType())->getReturnType(),
                   Instruction::Invoke,
                   OperandTraits<InvokeInst>::op_end(this) - Values,
                   Values, InsertBefore) {
  init(Func, IfNormal, IfException, ArgBegin, ArgEnd, NameStr,
       typename std::iterator_traits<InputIterator>::iterator_category());
}
template<typename InputIterator>
InvokeInst::InvokeInst(Value *Func,
                       BasicBlock *IfNormal, BasicBlock *IfException,
                       InputIterator ArgBegin, InputIterator ArgEnd,
                       unsigned Values,
                       const Twine &NameStr, BasicBlock *InsertAtEnd)
  : TerminatorInst(cast<FunctionType>(cast<PointerType>(Func->getType())
                                      ->getElementType())->getReturnType(),
                   Instruction::Invoke,
                   OperandTraits<InvokeInst>::op_end(this) - Values,
                   Values, InsertAtEnd) {
  init(Func, IfNormal, IfException, ArgBegin, ArgEnd, NameStr,
       typename std::iterator_traits<InputIterator>::iterator_category());
}

DEFINE_TRANSPARENT_OPERAND_ACCESSORS(InvokeInst, Value)

//===----------------------------------------------------------------------===//
//                              UnwindInst Class
//===----------------------------------------------------------------------===//

//===---------------------------------------------------------------------------
/// UnwindInst - Immediately exit the current function, unwinding the stack
/// until an invoke instruction is found.
///
class UnwindInst : public TerminatorInst {
  void *operator new(size_t, unsigned);  // DO NOT IMPLEMENT
protected:
  virtual UnwindInst *clone_impl() const;
public:
  // allocate space for exactly zero operands
  void *operator new(size_t s) {
    return User::operator new(s, 0);
  }
  explicit UnwindInst(LLVMContext &C, Instruction *InsertBefore = 0);
  explicit UnwindInst(LLVMContext &C, BasicBlock *InsertAtEnd);

  unsigned getNumSuccessors() const { return 0; }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const UnwindInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::Unwind;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
private:
  virtual BasicBlock *getSuccessorV(unsigned idx) const;
  virtual unsigned getNumSuccessorsV() const;
  virtual void setSuccessorV(unsigned idx, BasicBlock *B);
};

//===----------------------------------------------------------------------===//
//                           UnreachableInst Class
//===----------------------------------------------------------------------===//

//===---------------------------------------------------------------------------
/// UnreachableInst - This function has undefined behavior.  In particular, the
/// presence of this instruction indicates some higher level knowledge that the
/// end of the block cannot be reached.
///
class UnreachableInst : public TerminatorInst {
  void *operator new(size_t, unsigned);  // DO NOT IMPLEMENT
protected:
  virtual UnreachableInst *clone_impl() const;

public:
  // allocate space for exactly zero operands
  void *operator new(size_t s) {
    return User::operator new(s, 0);
  }
  explicit UnreachableInst(LLVMContext &C, Instruction *InsertBefore = 0);
  explicit UnreachableInst(LLVMContext &C, BasicBlock *InsertAtEnd);

  unsigned getNumSuccessors() const { return 0; }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const UnreachableInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::Unreachable;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
private:
  virtual BasicBlock *getSuccessorV(unsigned idx) const;
  virtual unsigned getNumSuccessorsV() const;
  virtual void setSuccessorV(unsigned idx, BasicBlock *B);
};

//===----------------------------------------------------------------------===//
//                                 TruncInst Class
//===----------------------------------------------------------------------===//

/// @brief This class represents a truncation of integer types.
class TruncInst : public CastInst {
protected:
  /// @brief Clone an identical TruncInst
  virtual TruncInst *clone_impl() const;

public:
  /// @brief Constructor with insert-before-instruction semantics
  TruncInst(
    Value *S,                     ///< The value to be truncated
    const Type *Ty,               ///< The (smaller) type to truncate to
    const Twine &NameStr = "", ///< A name for the new instruction
    Instruction *InsertBefore = 0 ///< Where to insert the new instruction
  );

  /// @brief Constructor with insert-at-end-of-block semantics
  TruncInst(
    Value *S,                     ///< The value to be truncated
    const Type *Ty,               ///< The (smaller) type to truncate to
    const Twine &NameStr,   ///< A name for the new instruction
    BasicBlock *InsertAtEnd       ///< The block to insert the instruction into
  );

  /// @brief Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const TruncInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Trunc;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

//===----------------------------------------------------------------------===//
//                                 ZExtInst Class
//===----------------------------------------------------------------------===//

/// @brief This class represents zero extension of integer types.
class ZExtInst : public CastInst {
protected:
  /// @brief Clone an identical ZExtInst
  virtual ZExtInst *clone_impl() const;

public:
  /// @brief Constructor with insert-before-instruction semantics
  ZExtInst(
    Value *S,                     ///< The value to be zero extended
    const Type *Ty,               ///< The type to zero extend to
    const Twine &NameStr = "", ///< A name for the new instruction
    Instruction *InsertBefore = 0 ///< Where to insert the new instruction
  );

  /// @brief Constructor with insert-at-end semantics.
  ZExtInst(
    Value *S,                     ///< The value to be zero extended
    const Type *Ty,               ///< The type to zero extend to
    const Twine &NameStr,   ///< A name for the new instruction
    BasicBlock *InsertAtEnd       ///< The block to insert the instruction into
  );

  /// @brief Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ZExtInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == ZExt;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

//===----------------------------------------------------------------------===//
//                                 SExtInst Class
//===----------------------------------------------------------------------===//

/// @brief This class represents a sign extension of integer types.
class SExtInst : public CastInst {
protected:
  /// @brief Clone an identical SExtInst
  virtual SExtInst *clone_impl() const;

public:
  /// @brief Constructor with insert-before-instruction semantics
  SExtInst(
    Value *S,                     ///< The value to be sign extended
    const Type *Ty,               ///< The type to sign extend to
    const Twine &NameStr = "", ///< A name for the new instruction
    Instruction *InsertBefore = 0 ///< Where to insert the new instruction
  );

  /// @brief Constructor with insert-at-end-of-block semantics
  SExtInst(
    Value *S,                     ///< The value to be sign extended
    const Type *Ty,               ///< The type to sign extend to
    const Twine &NameStr,   ///< A name for the new instruction
    BasicBlock *InsertAtEnd       ///< The block to insert the instruction into
  );

  /// @brief Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const SExtInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == SExt;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

//===----------------------------------------------------------------------===//
//                                 FPTruncInst Class
//===----------------------------------------------------------------------===//

/// @brief This class represents a truncation of floating point types.
class FPTruncInst : public CastInst {
protected:
  /// @brief Clone an identical FPTruncInst
  virtual FPTruncInst *clone_impl() const;

public:
  /// @brief Constructor with insert-before-instruction semantics
  FPTruncInst(
    Value *S,                     ///< The value to be truncated
    const Type *Ty,               ///< The type to truncate to
    const Twine &NameStr = "", ///< A name for the new instruction
    Instruction *InsertBefore = 0 ///< Where to insert the new instruction
  );

  /// @brief Constructor with insert-before-instruction semantics
  FPTruncInst(
    Value *S,                     ///< The value to be truncated
    const Type *Ty,               ///< The type to truncate to
    const Twine &NameStr,   ///< A name for the new instruction
    BasicBlock *InsertAtEnd       ///< The block to insert the instruction into
  );

  /// @brief Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const FPTruncInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == FPTrunc;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

//===----------------------------------------------------------------------===//
//                                 FPExtInst Class
//===----------------------------------------------------------------------===//

/// @brief This class represents an extension of floating point types.
class FPExtInst : public CastInst {
protected:
  /// @brief Clone an identical FPExtInst
  virtual FPExtInst *clone_impl() const;

public:
  /// @brief Constructor with insert-before-instruction semantics
  FPExtInst(
    Value *S,                     ///< The value to be extended
    const Type *Ty,               ///< The type to extend to
    const Twine &NameStr = "", ///< A name for the new instruction
    Instruction *InsertBefore = 0 ///< Where to insert the new instruction
  );

  /// @brief Constructor with insert-at-end-of-block semantics
  FPExtInst(
    Value *S,                     ///< The value to be extended
    const Type *Ty,               ///< The type to extend to
    const Twine &NameStr,   ///< A name for the new instruction
    BasicBlock *InsertAtEnd       ///< The block to insert the instruction into
  );

  /// @brief Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const FPExtInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == FPExt;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

//===----------------------------------------------------------------------===//
//                                 UIToFPInst Class
//===----------------------------------------------------------------------===//

/// @brief This class represents a cast unsigned integer to floating point.
class UIToFPInst : public CastInst {
protected:
  /// @brief Clone an identical UIToFPInst
  virtual UIToFPInst *clone_impl() const;

public:
  /// @brief Constructor with insert-before-instruction semantics
  UIToFPInst(
    Value *S,                     ///< The value to be converted
    const Type *Ty,               ///< The type to convert to
    const Twine &NameStr = "", ///< A name for the new instruction
    Instruction *InsertBefore = 0 ///< Where to insert the new instruction
  );

  /// @brief Constructor with insert-at-end-of-block semantics
  UIToFPInst(
    Value *S,                     ///< The value to be converted
    const Type *Ty,               ///< The type to convert to
    const Twine &NameStr,   ///< A name for the new instruction
    BasicBlock *InsertAtEnd       ///< The block to insert the instruction into
  );

  /// @brief Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const UIToFPInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == UIToFP;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

//===----------------------------------------------------------------------===//
//                                 SIToFPInst Class
//===----------------------------------------------------------------------===//

/// @brief This class represents a cast from signed integer to floating point.
class SIToFPInst : public CastInst {
protected:
  /// @brief Clone an identical SIToFPInst
  virtual SIToFPInst *clone_impl() const;

public:
  /// @brief Constructor with insert-before-instruction semantics
  SIToFPInst(
    Value *S,                     ///< The value to be converted
    const Type *Ty,               ///< The type to convert to
    const Twine &NameStr = "", ///< A name for the new instruction
    Instruction *InsertBefore = 0 ///< Where to insert the new instruction
  );

  /// @brief Constructor with insert-at-end-of-block semantics
  SIToFPInst(
    Value *S,                     ///< The value to be converted
    const Type *Ty,               ///< The type to convert to
    const Twine &NameStr,   ///< A name for the new instruction
    BasicBlock *InsertAtEnd       ///< The block to insert the instruction into
  );

  /// @brief Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const SIToFPInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == SIToFP;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

//===----------------------------------------------------------------------===//
//                                 FPToUIInst Class
//===----------------------------------------------------------------------===//

/// @brief This class represents a cast from floating point to unsigned integer
class FPToUIInst  : public CastInst {
protected:
  /// @brief Clone an identical FPToUIInst
  virtual FPToUIInst *clone_impl() const;

public:
  /// @brief Constructor with insert-before-instruction semantics
  FPToUIInst(
    Value *S,                     ///< The value to be converted
    const Type *Ty,               ///< The type to convert to
    const Twine &NameStr = "", ///< A name for the new instruction
    Instruction *InsertBefore = 0 ///< Where to insert the new instruction
  );

  /// @brief Constructor with insert-at-end-of-block semantics
  FPToUIInst(
    Value *S,                     ///< The value to be converted
    const Type *Ty,               ///< The type to convert to
    const Twine &NameStr,   ///< A name for the new instruction
    BasicBlock *InsertAtEnd       ///< Where to insert the new instruction
  );

  /// @brief Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const FPToUIInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == FPToUI;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

//===----------------------------------------------------------------------===//
//                                 FPToSIInst Class
//===----------------------------------------------------------------------===//

/// @brief This class represents a cast from floating point to signed integer.
class FPToSIInst  : public CastInst {
protected:
  /// @brief Clone an identical FPToSIInst
  virtual FPToSIInst *clone_impl() const;

public:
  /// @brief Constructor with insert-before-instruction semantics
  FPToSIInst(
    Value *S,                     ///< The value to be converted
    const Type *Ty,               ///< The type to convert to
    const Twine &NameStr = "", ///< A name for the new instruction
    Instruction *InsertBefore = 0 ///< Where to insert the new instruction
  );

  /// @brief Constructor with insert-at-end-of-block semantics
  FPToSIInst(
    Value *S,                     ///< The value to be converted
    const Type *Ty,               ///< The type to convert to
    const Twine &NameStr,   ///< A name for the new instruction
    BasicBlock *InsertAtEnd       ///< The block to insert the instruction into
  );

  /// @brief Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const FPToSIInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == FPToSI;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

//===----------------------------------------------------------------------===//
//                                 IntToPtrInst Class
//===----------------------------------------------------------------------===//

/// @brief This class represents a cast from an integer to a pointer.
class IntToPtrInst : public CastInst {
public:
  /// @brief Constructor with insert-before-instruction semantics
  IntToPtrInst(
    Value *S,                     ///< The value to be converted
    const Type *Ty,               ///< The type to convert to
    const Twine &NameStr = "", ///< A name for the new instruction
    Instruction *InsertBefore = 0 ///< Where to insert the new instruction
  );

  /// @brief Constructor with insert-at-end-of-block semantics
  IntToPtrInst(
    Value *S,                     ///< The value to be converted
    const Type *Ty,               ///< The type to convert to
    const Twine &NameStr,   ///< A name for the new instruction
    BasicBlock *InsertAtEnd       ///< The block to insert the instruction into
  );

  /// @brief Clone an identical IntToPtrInst
  virtual IntToPtrInst *clone_impl() const;

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const IntToPtrInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == IntToPtr;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

//===----------------------------------------------------------------------===//
//                                 PtrToIntInst Class
//===----------------------------------------------------------------------===//

/// @brief This class represents a cast from a pointer to an integer
class PtrToIntInst : public CastInst {
protected:
  /// @brief Clone an identical PtrToIntInst
  virtual PtrToIntInst *clone_impl() const;

public:
  /// @brief Constructor with insert-before-instruction semantics
  PtrToIntInst(
    Value *S,                     ///< The value to be converted
    const Type *Ty,               ///< The type to convert to
    const Twine &NameStr = "", ///< A name for the new instruction
    Instruction *InsertBefore = 0 ///< Where to insert the new instruction
  );

  /// @brief Constructor with insert-at-end-of-block semantics
  PtrToIntInst(
    Value *S,                     ///< The value to be converted
    const Type *Ty,               ///< The type to convert to
    const Twine &NameStr,   ///< A name for the new instruction
    BasicBlock *InsertAtEnd       ///< The block to insert the instruction into
  );

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const PtrToIntInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == PtrToInt;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

//===----------------------------------------------------------------------===//
//                             BitCastInst Class
//===----------------------------------------------------------------------===//

/// @brief This class represents a no-op cast from one type to another.
class BitCastInst : public CastInst {
protected:
  /// @brief Clone an identical BitCastInst
  virtual BitCastInst *clone_impl() const;

public:
  /// @brief Constructor with insert-before-instruction semantics
  BitCastInst(
    Value *S,                     ///< The value to be casted
    const Type *Ty,               ///< The type to casted to
    const Twine &NameStr = "", ///< A name for the new instruction
    Instruction *InsertBefore = 0 ///< Where to insert the new instruction
  );

  /// @brief Constructor with insert-at-end-of-block semantics
  BitCastInst(
    Value *S,                     ///< The value to be casted
    const Type *Ty,               ///< The type to casted to
    const Twine &NameStr,      ///< A name for the new instruction
    BasicBlock *InsertAtEnd       ///< The block to insert the instruction into
  );

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const BitCastInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == BitCast;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

} // End llvm namespace

#endif
