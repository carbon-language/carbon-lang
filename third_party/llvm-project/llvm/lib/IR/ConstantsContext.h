//===-- ConstantsContext.h - Constants-related Context Interals -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines various helper methods and classes used by
// LLVMContextImpl for creating and managing constants.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_IR_CONSTANTSCONTEXT_H
#define LLVM_LIB_IR_CONSTANTSCONTEXT_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/OperandTraits.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <utility>

#define DEBUG_TYPE "ir"

namespace llvm {

/// UnaryConstantExpr - This class is private to Constants.cpp, and is used
/// behind the scenes to implement unary constant exprs.
class UnaryConstantExpr final : public ConstantExpr {
public:
  UnaryConstantExpr(unsigned Opcode, Constant *C, Type *Ty)
    : ConstantExpr(Ty, Opcode, &Op<0>(), 1) {
    Op<0>() = C;
  }

  // allocate space for exactly one operand
  void *operator new(size_t S) { return User::operator new(S, 1); }
  void operator delete(void *Ptr) { User::operator delete(Ptr); }

  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);

  static bool classof(const ConstantExpr *CE) {
    return Instruction::isCast(CE->getOpcode()) ||
           Instruction::isUnaryOp(CE->getOpcode());
  }
  static bool classof(const Value *V) {
    return isa<ConstantExpr>(V) && classof(cast<ConstantExpr>(V));
  }
};

/// BinaryConstantExpr - This class is private to Constants.cpp, and is used
/// behind the scenes to implement binary constant exprs.
class BinaryConstantExpr final : public ConstantExpr {
public:
  BinaryConstantExpr(unsigned Opcode, Constant *C1, Constant *C2,
                     unsigned Flags)
    : ConstantExpr(C1->getType(), Opcode, &Op<0>(), 2) {
    Op<0>() = C1;
    Op<1>() = C2;
    SubclassOptionalData = Flags;
  }

  // allocate space for exactly two operands
  void *operator new(size_t S) { return User::operator new(S, 2); }
  void operator delete(void *Ptr) { User::operator delete(Ptr); }

  /// Transparently provide more efficient getOperand methods.
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);

  static bool classof(const ConstantExpr *CE) {
    return Instruction::isBinaryOp(CE->getOpcode());
  }
  static bool classof(const Value *V) {
    return isa<ConstantExpr>(V) && classof(cast<ConstantExpr>(V));
  }
};

/// SelectConstantExpr - This class is private to Constants.cpp, and is used
/// behind the scenes to implement select constant exprs.
class SelectConstantExpr final : public ConstantExpr {
public:
  SelectConstantExpr(Constant *C1, Constant *C2, Constant *C3)
    : ConstantExpr(C2->getType(), Instruction::Select, &Op<0>(), 3) {
    Op<0>() = C1;
    Op<1>() = C2;
    Op<2>() = C3;
  }

  // allocate space for exactly three operands
  void *operator new(size_t S) { return User::operator new(S, 3); }
  void operator delete(void *Ptr) { User::operator delete(Ptr); }

  /// Transparently provide more efficient getOperand methods.
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);

  static bool classof(const ConstantExpr *CE) {
    return CE->getOpcode() == Instruction::Select;
  }
  static bool classof(const Value *V) {
    return isa<ConstantExpr>(V) && classof(cast<ConstantExpr>(V));
  }
};

/// ExtractElementConstantExpr - This class is private to
/// Constants.cpp, and is used behind the scenes to implement
/// extractelement constant exprs.
class ExtractElementConstantExpr final : public ConstantExpr {
public:
  ExtractElementConstantExpr(Constant *C1, Constant *C2)
    : ConstantExpr(cast<VectorType>(C1->getType())->getElementType(),
                   Instruction::ExtractElement, &Op<0>(), 2) {
    Op<0>() = C1;
    Op<1>() = C2;
  }

  // allocate space for exactly two operands
  void *operator new(size_t S) { return User::operator new(S, 2); }
  void operator delete(void *Ptr) { User::operator delete(Ptr); }

  /// Transparently provide more efficient getOperand methods.
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);

  static bool classof(const ConstantExpr *CE) {
    return CE->getOpcode() == Instruction::ExtractElement;
  }
  static bool classof(const Value *V) {
    return isa<ConstantExpr>(V) && classof(cast<ConstantExpr>(V));
  }
};

/// InsertElementConstantExpr - This class is private to
/// Constants.cpp, and is used behind the scenes to implement
/// insertelement constant exprs.
class InsertElementConstantExpr final : public ConstantExpr {
public:
  InsertElementConstantExpr(Constant *C1, Constant *C2, Constant *C3)
    : ConstantExpr(C1->getType(), Instruction::InsertElement,
                   &Op<0>(), 3) {
    Op<0>() = C1;
    Op<1>() = C2;
    Op<2>() = C3;
  }

  // allocate space for exactly three operands
  void *operator new(size_t S) { return User::operator new(S, 3); }
  void operator delete(void *Ptr) { User::operator delete(Ptr); }

  /// Transparently provide more efficient getOperand methods.
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);

  static bool classof(const ConstantExpr *CE) {
    return CE->getOpcode() == Instruction::InsertElement;
  }
  static bool classof(const Value *V) {
    return isa<ConstantExpr>(V) && classof(cast<ConstantExpr>(V));
  }
};

/// ShuffleVectorConstantExpr - This class is private to
/// Constants.cpp, and is used behind the scenes to implement
/// shufflevector constant exprs.
class ShuffleVectorConstantExpr final : public ConstantExpr {
public:
  ShuffleVectorConstantExpr(Constant *C1, Constant *C2, ArrayRef<int> Mask)
      : ConstantExpr(VectorType::get(
                         cast<VectorType>(C1->getType())->getElementType(),
                         Mask.size(), isa<ScalableVectorType>(C1->getType())),
                     Instruction::ShuffleVector, &Op<0>(), 2) {
    assert(ShuffleVectorInst::isValidOperands(C1, C2, Mask) &&
           "Invalid shuffle vector instruction operands!");
    Op<0>() = C1;
    Op<1>() = C2;
    ShuffleMask.assign(Mask.begin(), Mask.end());
    ShuffleMaskForBitcode =
        ShuffleVectorInst::convertShuffleMaskForBitcode(Mask, getType());
  }

  SmallVector<int, 4> ShuffleMask;
  Constant *ShuffleMaskForBitcode;

  void *operator new(size_t S) { return User::operator new(S, 2); }
  void operator delete(void *Ptr) { return User::operator delete(Ptr); }

  /// Transparently provide more efficient getOperand methods.
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);

  static bool classof(const ConstantExpr *CE) {
    return CE->getOpcode() == Instruction::ShuffleVector;
  }
  static bool classof(const Value *V) {
    return isa<ConstantExpr>(V) && classof(cast<ConstantExpr>(V));
  }
};

/// ExtractValueConstantExpr - This class is private to
/// Constants.cpp, and is used behind the scenes to implement
/// extractvalue constant exprs.
class ExtractValueConstantExpr final : public ConstantExpr {
public:
  ExtractValueConstantExpr(Constant *Agg, ArrayRef<unsigned> IdxList,
                           Type *DestTy)
      : ConstantExpr(DestTy, Instruction::ExtractValue, &Op<0>(), 1),
        Indices(IdxList.begin(), IdxList.end()) {
    Op<0>() = Agg;
  }

  // allocate space for exactly one operand
  void *operator new(size_t S) { return User::operator new(S, 1); }
  void operator delete(void *Ptr) { User::operator delete(Ptr); }

  /// Indices - These identify which value to extract.
  const SmallVector<unsigned, 4> Indices;

  /// Transparently provide more efficient getOperand methods.
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);

  static bool classof(const ConstantExpr *CE) {
    return CE->getOpcode() == Instruction::ExtractValue;
  }
  static bool classof(const Value *V) {
    return isa<ConstantExpr>(V) && classof(cast<ConstantExpr>(V));
  }
};

/// InsertValueConstantExpr - This class is private to
/// Constants.cpp, and is used behind the scenes to implement
/// insertvalue constant exprs.
class InsertValueConstantExpr final : public ConstantExpr {
public:
  InsertValueConstantExpr(Constant *Agg, Constant *Val,
                          ArrayRef<unsigned> IdxList, Type *DestTy)
      : ConstantExpr(DestTy, Instruction::InsertValue, &Op<0>(), 2),
        Indices(IdxList.begin(), IdxList.end()) {
    Op<0>() = Agg;
    Op<1>() = Val;
  }

  // allocate space for exactly one operand
  void *operator new(size_t S) { return User::operator new(S, 2); }
  void operator delete(void *Ptr) { User::operator delete(Ptr); }

  /// Indices - These identify the position for the insertion.
  const SmallVector<unsigned, 4> Indices;

  /// Transparently provide more efficient getOperand methods.
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);

  static bool classof(const ConstantExpr *CE) {
    return CE->getOpcode() == Instruction::InsertValue;
  }
  static bool classof(const Value *V) {
    return isa<ConstantExpr>(V) && classof(cast<ConstantExpr>(V));
  }
};

/// GetElementPtrConstantExpr - This class is private to Constants.cpp, and is
/// used behind the scenes to implement getelementpr constant exprs.
class GetElementPtrConstantExpr final : public ConstantExpr {
  Type *SrcElementTy;
  Type *ResElementTy;

  GetElementPtrConstantExpr(Type *SrcElementTy, Constant *C,
                            ArrayRef<Constant *> IdxList, Type *DestTy);

public:
  static GetElementPtrConstantExpr *Create(Type *SrcElementTy, Constant *C,
                                           ArrayRef<Constant *> IdxList,
                                           Type *DestTy, unsigned Flags) {
    GetElementPtrConstantExpr *Result = new (IdxList.size() + 1)
        GetElementPtrConstantExpr(SrcElementTy, C, IdxList, DestTy);
    Result->SubclassOptionalData = Flags;
    return Result;
  }

  Type *getSourceElementType() const;
  Type *getResultElementType() const;

  /// Transparently provide more efficient getOperand methods.
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);

  static bool classof(const ConstantExpr *CE) {
    return CE->getOpcode() == Instruction::GetElementPtr;
  }
  static bool classof(const Value *V) {
    return isa<ConstantExpr>(V) && classof(cast<ConstantExpr>(V));
  }
};

// CompareConstantExpr - This class is private to Constants.cpp, and is used
// behind the scenes to implement ICmp and FCmp constant expressions. This is
// needed in order to store the predicate value for these instructions.
class CompareConstantExpr final : public ConstantExpr {
public:
  unsigned short predicate;
  CompareConstantExpr(Type *ty, Instruction::OtherOps opc,
                      unsigned short pred,  Constant* LHS, Constant* RHS)
    : ConstantExpr(ty, opc, &Op<0>(), 2), predicate(pred) {
    Op<0>() = LHS;
    Op<1>() = RHS;
  }

  // allocate space for exactly two operands
  void *operator new(size_t S) { return User::operator new(S, 2); }
  void operator delete(void *Ptr) { return User::operator delete(Ptr); }

  /// Transparently provide more efficient getOperand methods.
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);

  static bool classof(const ConstantExpr *CE) {
    return CE->getOpcode() == Instruction::ICmp ||
           CE->getOpcode() == Instruction::FCmp;
  }
  static bool classof(const Value *V) {
    return isa<ConstantExpr>(V) && classof(cast<ConstantExpr>(V));
  }
};

template <>
struct OperandTraits<UnaryConstantExpr>
    : public FixedNumOperandTraits<UnaryConstantExpr, 1> {};
DEFINE_TRANSPARENT_OPERAND_ACCESSORS(UnaryConstantExpr, Value)

template <>
struct OperandTraits<BinaryConstantExpr>
    : public FixedNumOperandTraits<BinaryConstantExpr, 2> {};
DEFINE_TRANSPARENT_OPERAND_ACCESSORS(BinaryConstantExpr, Value)

template <>
struct OperandTraits<SelectConstantExpr>
    : public FixedNumOperandTraits<SelectConstantExpr, 3> {};
DEFINE_TRANSPARENT_OPERAND_ACCESSORS(SelectConstantExpr, Value)

template <>
struct OperandTraits<ExtractElementConstantExpr>
    : public FixedNumOperandTraits<ExtractElementConstantExpr, 2> {};
DEFINE_TRANSPARENT_OPERAND_ACCESSORS(ExtractElementConstantExpr, Value)

template <>
struct OperandTraits<InsertElementConstantExpr>
    : public FixedNumOperandTraits<InsertElementConstantExpr, 3> {};
DEFINE_TRANSPARENT_OPERAND_ACCESSORS(InsertElementConstantExpr, Value)

template <>
struct OperandTraits<ShuffleVectorConstantExpr>
    : public FixedNumOperandTraits<ShuffleVectorConstantExpr, 2> {};
DEFINE_TRANSPARENT_OPERAND_ACCESSORS(ShuffleVectorConstantExpr, Value)

template <>
struct OperandTraits<ExtractValueConstantExpr>
    : public FixedNumOperandTraits<ExtractValueConstantExpr, 1> {};
DEFINE_TRANSPARENT_OPERAND_ACCESSORS(ExtractValueConstantExpr, Value)

template <>
struct OperandTraits<InsertValueConstantExpr>
    : public FixedNumOperandTraits<InsertValueConstantExpr, 2> {};
DEFINE_TRANSPARENT_OPERAND_ACCESSORS(InsertValueConstantExpr, Value)

template <>
struct OperandTraits<GetElementPtrConstantExpr>
    : public VariadicOperandTraits<GetElementPtrConstantExpr, 1> {};

DEFINE_TRANSPARENT_OPERAND_ACCESSORS(GetElementPtrConstantExpr, Value)

template <>
struct OperandTraits<CompareConstantExpr>
    : public FixedNumOperandTraits<CompareConstantExpr, 2> {};
DEFINE_TRANSPARENT_OPERAND_ACCESSORS(CompareConstantExpr, Value)

template <class ConstantClass> struct ConstantAggrKeyType;
struct InlineAsmKeyType;
struct ConstantExprKeyType;

template <class ConstantClass> struct ConstantInfo;
template <> struct ConstantInfo<ConstantExpr> {
  using ValType = ConstantExprKeyType;
  using TypeClass = Type;
};
template <> struct ConstantInfo<InlineAsm> {
  using ValType = InlineAsmKeyType;
  using TypeClass = PointerType;
};
template <> struct ConstantInfo<ConstantArray> {
  using ValType = ConstantAggrKeyType<ConstantArray>;
  using TypeClass = ArrayType;
};
template <> struct ConstantInfo<ConstantStruct> {
  using ValType = ConstantAggrKeyType<ConstantStruct>;
  using TypeClass = StructType;
};
template <> struct ConstantInfo<ConstantVector> {
  using ValType = ConstantAggrKeyType<ConstantVector>;
  using TypeClass = VectorType;
};

template <class ConstantClass> struct ConstantAggrKeyType {
  ArrayRef<Constant *> Operands;

  ConstantAggrKeyType(ArrayRef<Constant *> Operands) : Operands(Operands) {}

  ConstantAggrKeyType(ArrayRef<Constant *> Operands, const ConstantClass *)
      : Operands(Operands) {}

  ConstantAggrKeyType(const ConstantClass *C,
                      SmallVectorImpl<Constant *> &Storage) {
    assert(Storage.empty() && "Expected empty storage");
    for (unsigned I = 0, E = C->getNumOperands(); I != E; ++I)
      Storage.push_back(C->getOperand(I));
    Operands = Storage;
  }

  bool operator==(const ConstantAggrKeyType &X) const {
    return Operands == X.Operands;
  }

  bool operator==(const ConstantClass *C) const {
    if (Operands.size() != C->getNumOperands())
      return false;
    for (unsigned I = 0, E = Operands.size(); I != E; ++I)
      if (Operands[I] != C->getOperand(I))
        return false;
    return true;
  }

  unsigned getHash() const {
    return hash_combine_range(Operands.begin(), Operands.end());
  }

  using TypeClass = typename ConstantInfo<ConstantClass>::TypeClass;

  ConstantClass *create(TypeClass *Ty) const {
    return new (Operands.size()) ConstantClass(Ty, Operands);
  }
};

struct InlineAsmKeyType {
  StringRef AsmString;
  StringRef Constraints;
  FunctionType *FTy;
  bool HasSideEffects;
  bool IsAlignStack;
  InlineAsm::AsmDialect AsmDialect;
  bool CanThrow;

  InlineAsmKeyType(StringRef AsmString, StringRef Constraints,
                   FunctionType *FTy, bool HasSideEffects, bool IsAlignStack,
                   InlineAsm::AsmDialect AsmDialect, bool canThrow)
      : AsmString(AsmString), Constraints(Constraints), FTy(FTy),
        HasSideEffects(HasSideEffects), IsAlignStack(IsAlignStack),
        AsmDialect(AsmDialect), CanThrow(canThrow) {}

  InlineAsmKeyType(const InlineAsm *Asm, SmallVectorImpl<Constant *> &)
      : AsmString(Asm->getAsmString()), Constraints(Asm->getConstraintString()),
        FTy(Asm->getFunctionType()), HasSideEffects(Asm->hasSideEffects()),
        IsAlignStack(Asm->isAlignStack()), AsmDialect(Asm->getDialect()),
        CanThrow(Asm->canThrow()) {}

  bool operator==(const InlineAsmKeyType &X) const {
    return HasSideEffects == X.HasSideEffects &&
           IsAlignStack == X.IsAlignStack && AsmDialect == X.AsmDialect &&
           AsmString == X.AsmString && Constraints == X.Constraints &&
           FTy == X.FTy && CanThrow == X.CanThrow;
  }

  bool operator==(const InlineAsm *Asm) const {
    return HasSideEffects == Asm->hasSideEffects() &&
           IsAlignStack == Asm->isAlignStack() &&
           AsmDialect == Asm->getDialect() &&
           AsmString == Asm->getAsmString() &&
           Constraints == Asm->getConstraintString() &&
           FTy == Asm->getFunctionType() && CanThrow == Asm->canThrow();
  }

  unsigned getHash() const {
    return hash_combine(AsmString, Constraints, HasSideEffects, IsAlignStack,
                        AsmDialect, FTy, CanThrow);
  }

  using TypeClass = ConstantInfo<InlineAsm>::TypeClass;

  InlineAsm *create(TypeClass *Ty) const {
    assert(PointerType::getUnqual(FTy) == Ty);
    return new InlineAsm(FTy, std::string(AsmString), std::string(Constraints),
                         HasSideEffects, IsAlignStack, AsmDialect, CanThrow);
  }
};

struct ConstantExprKeyType {
private:
  uint8_t Opcode;
  uint8_t SubclassOptionalData;
  uint16_t SubclassData;
  ArrayRef<Constant *> Ops;
  ArrayRef<unsigned> Indexes;
  ArrayRef<int> ShuffleMask;
  Type *ExplicitTy;

  static ArrayRef<int> getShuffleMaskIfValid(const ConstantExpr *CE) {
    if (CE->getOpcode() == Instruction::ShuffleVector)
      return CE->getShuffleMask();
    return None;
  }

  static ArrayRef<unsigned> getIndicesIfValid(const ConstantExpr *CE) {
    if (CE->hasIndices())
      return CE->getIndices();
    return None;
  }

  static Type *getSourceElementTypeIfValid(const ConstantExpr *CE) {
    if (auto *GEPCE = dyn_cast<GetElementPtrConstantExpr>(CE))
      return GEPCE->getSourceElementType();
    return nullptr;
  }

public:
  ConstantExprKeyType(unsigned Opcode, ArrayRef<Constant *> Ops,
                      unsigned short SubclassData = 0,
                      unsigned short SubclassOptionalData = 0,
                      ArrayRef<unsigned> Indexes = None,
                      ArrayRef<int> ShuffleMask = None,
                      Type *ExplicitTy = nullptr)
      : Opcode(Opcode), SubclassOptionalData(SubclassOptionalData),
        SubclassData(SubclassData), Ops(Ops), Indexes(Indexes),
        ShuffleMask(ShuffleMask), ExplicitTy(ExplicitTy) {}

  ConstantExprKeyType(ArrayRef<Constant *> Operands, const ConstantExpr *CE)
      : Opcode(CE->getOpcode()),
        SubclassOptionalData(CE->getRawSubclassOptionalData()),
        SubclassData(CE->isCompare() ? CE->getPredicate() : 0), Ops(Operands),
        Indexes(getIndicesIfValid(CE)), ShuffleMask(getShuffleMaskIfValid(CE)),
        ExplicitTy(getSourceElementTypeIfValid(CE)) {}

  ConstantExprKeyType(const ConstantExpr *CE,
                      SmallVectorImpl<Constant *> &Storage)
      : Opcode(CE->getOpcode()),
        SubclassOptionalData(CE->getRawSubclassOptionalData()),
        SubclassData(CE->isCompare() ? CE->getPredicate() : 0),
        Indexes(getIndicesIfValid(CE)), ShuffleMask(getShuffleMaskIfValid(CE)),
        ExplicitTy(getSourceElementTypeIfValid(CE)) {
    assert(Storage.empty() && "Expected empty storage");
    for (unsigned I = 0, E = CE->getNumOperands(); I != E; ++I)
      Storage.push_back(CE->getOperand(I));
    Ops = Storage;
  }

  bool operator==(const ConstantExprKeyType &X) const {
    return Opcode == X.Opcode && SubclassData == X.SubclassData &&
           SubclassOptionalData == X.SubclassOptionalData && Ops == X.Ops &&
           Indexes == X.Indexes && ShuffleMask == X.ShuffleMask &&
           ExplicitTy == X.ExplicitTy;
  }

  bool operator==(const ConstantExpr *CE) const {
    if (Opcode != CE->getOpcode())
      return false;
    if (SubclassOptionalData != CE->getRawSubclassOptionalData())
      return false;
    if (Ops.size() != CE->getNumOperands())
      return false;
    if (SubclassData != (CE->isCompare() ? CE->getPredicate() : 0))
      return false;
    for (unsigned I = 0, E = Ops.size(); I != E; ++I)
      if (Ops[I] != CE->getOperand(I))
        return false;
    if (Indexes != getIndicesIfValid(CE))
      return false;
    if (ShuffleMask != getShuffleMaskIfValid(CE))
      return false;
    if (ExplicitTy != getSourceElementTypeIfValid(CE))
      return false;
    return true;
  }

  unsigned getHash() const {
    return hash_combine(
        Opcode, SubclassOptionalData, SubclassData,
        hash_combine_range(Ops.begin(), Ops.end()),
        hash_combine_range(Indexes.begin(), Indexes.end()),
        hash_combine_range(ShuffleMask.begin(), ShuffleMask.end()), ExplicitTy);
  }

  using TypeClass = ConstantInfo<ConstantExpr>::TypeClass;

  ConstantExpr *create(TypeClass *Ty) const {
    switch (Opcode) {
    default:
      if (Instruction::isCast(Opcode) ||
          (Opcode >= Instruction::UnaryOpsBegin &&
           Opcode < Instruction::UnaryOpsEnd))
        return new UnaryConstantExpr(Opcode, Ops[0], Ty);
      if ((Opcode >= Instruction::BinaryOpsBegin &&
           Opcode < Instruction::BinaryOpsEnd))
        return new BinaryConstantExpr(Opcode, Ops[0], Ops[1],
                                      SubclassOptionalData);
      llvm_unreachable("Invalid ConstantExpr!");
    case Instruction::Select:
      return new SelectConstantExpr(Ops[0], Ops[1], Ops[2]);
    case Instruction::ExtractElement:
      return new ExtractElementConstantExpr(Ops[0], Ops[1]);
    case Instruction::InsertElement:
      return new InsertElementConstantExpr(Ops[0], Ops[1], Ops[2]);
    case Instruction::ShuffleVector:
      return new ShuffleVectorConstantExpr(Ops[0], Ops[1], ShuffleMask);
    case Instruction::InsertValue:
      return new InsertValueConstantExpr(Ops[0], Ops[1], Indexes, Ty);
    case Instruction::ExtractValue:
      return new ExtractValueConstantExpr(Ops[0], Indexes, Ty);
    case Instruction::GetElementPtr:
      return GetElementPtrConstantExpr::Create(ExplicitTy, Ops[0], Ops.slice(1),
                                               Ty, SubclassOptionalData);
    case Instruction::ICmp:
      return new CompareConstantExpr(Ty, Instruction::ICmp, SubclassData,
                                     Ops[0], Ops[1]);
    case Instruction::FCmp:
      return new CompareConstantExpr(Ty, Instruction::FCmp, SubclassData,
                                     Ops[0], Ops[1]);
    }
  }
};

// Free memory for a given constant.  Assumes the constant has already been
// removed from all relevant maps.
void deleteConstant(Constant *C);

template <class ConstantClass> class ConstantUniqueMap {
public:
  using ValType = typename ConstantInfo<ConstantClass>::ValType;
  using TypeClass = typename ConstantInfo<ConstantClass>::TypeClass;
  using LookupKey = std::pair<TypeClass *, ValType>;

  /// Key and hash together, so that we compute the hash only once and reuse it.
  using LookupKeyHashed = std::pair<unsigned, LookupKey>;

private:
  struct MapInfo {
    using ConstantClassInfo = DenseMapInfo<ConstantClass *>;

    static inline ConstantClass *getEmptyKey() {
      return ConstantClassInfo::getEmptyKey();
    }

    static inline ConstantClass *getTombstoneKey() {
      return ConstantClassInfo::getTombstoneKey();
    }

    static unsigned getHashValue(const ConstantClass *CP) {
      SmallVector<Constant *, 32> Storage;
      return getHashValue(LookupKey(CP->getType(), ValType(CP, Storage)));
    }

    static bool isEqual(const ConstantClass *LHS, const ConstantClass *RHS) {
      return LHS == RHS;
    }

    static unsigned getHashValue(const LookupKey &Val) {
      return hash_combine(Val.first, Val.second.getHash());
    }

    static unsigned getHashValue(const LookupKeyHashed &Val) {
      return Val.first;
    }

    static bool isEqual(const LookupKey &LHS, const ConstantClass *RHS) {
      if (RHS == getEmptyKey() || RHS == getTombstoneKey())
        return false;
      if (LHS.first != RHS->getType())
        return false;
      return LHS.second == RHS;
    }

    static bool isEqual(const LookupKeyHashed &LHS, const ConstantClass *RHS) {
      return isEqual(LHS.second, RHS);
    }
  };

public:
  using MapTy = DenseSet<ConstantClass *, MapInfo>;

private:
  MapTy Map;

public:
  typename MapTy::iterator begin() { return Map.begin(); }
  typename MapTy::iterator end() { return Map.end(); }

  void freeConstants() {
    for (auto &I : Map)
      deleteConstant(I);
  }

private:
  ConstantClass *create(TypeClass *Ty, ValType V, LookupKeyHashed &HashKey) {
    ConstantClass *Result = V.create(Ty);

    assert(Result->getType() == Ty && "Type specified is not correct!");
    Map.insert_as(Result, HashKey);

    return Result;
  }

public:
  /// Return the specified constant from the map, creating it if necessary.
  ConstantClass *getOrCreate(TypeClass *Ty, ValType V) {
    LookupKey Key(Ty, V);
    /// Hash once, and reuse it for the lookup and the insertion if needed.
    LookupKeyHashed Lookup(MapInfo::getHashValue(Key), Key);

    ConstantClass *Result = nullptr;

    auto I = Map.find_as(Lookup);
    if (I == Map.end())
      Result = create(Ty, V, Lookup);
    else
      Result = *I;
    assert(Result && "Unexpected nullptr");

    return Result;
  }

  /// Remove this constant from the map
  void remove(ConstantClass *CP) {
    typename MapTy::iterator I = Map.find(CP);
    assert(I != Map.end() && "Constant not found in constant table!");
    assert(*I == CP && "Didn't find correct element?");
    Map.erase(I);
  }

  ConstantClass *replaceOperandsInPlace(ArrayRef<Constant *> Operands,
                                        ConstantClass *CP, Value *From,
                                        Constant *To, unsigned NumUpdated = 0,
                                        unsigned OperandNo = ~0u) {
    LookupKey Key(CP->getType(), ValType(Operands, CP));
    /// Hash once, and reuse it for the lookup and the insertion if needed.
    LookupKeyHashed Lookup(MapInfo::getHashValue(Key), Key);

    auto ItMap = Map.find_as(Lookup);
    if (ItMap != Map.end())
      return *ItMap;

    // Update to the new value.  Optimize for the case when we have a single
    // operand that we're changing, but handle bulk updates efficiently.
    remove(CP);
    if (NumUpdated == 1) {
      assert(OperandNo < CP->getNumOperands() && "Invalid index");
      assert(CP->getOperand(OperandNo) != To && "I didn't contain From!");
      CP->setOperand(OperandNo, To);
    } else {
      for (unsigned I = 0, E = CP->getNumOperands(); I != E; ++I)
        if (CP->getOperand(I) == From)
          CP->setOperand(I, To);
    }
    Map.insert_as(CP, Lookup);
    return nullptr;
  }

  void dump() const {
    LLVM_DEBUG(dbgs() << "Constant.cpp: ConstantUniqueMap\n");
  }
};

template <> inline void ConstantUniqueMap<InlineAsm>::freeConstants() {
  for (auto &I : Map)
    delete I;
}

} // end namespace llvm

#endif // LLVM_LIB_IR_CONSTANTSCONTEXT_H
