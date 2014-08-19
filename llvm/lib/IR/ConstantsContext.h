//===-- ConstantsContext.h - Constants-related Context Interals -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines various helper methods and classes used by
// LLVMContextImpl for creating and managing constants.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_IR_CONSTANTSCONTEXT_H
#define LLVM_LIB_IR_CONSTANTSCONTEXT_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <tuple>

#define DEBUG_TYPE "ir"

namespace llvm {
template<class ValType>
struct ConstantTraits;

/// UnaryConstantExpr - This class is private to Constants.cpp, and is used
/// behind the scenes to implement unary constant exprs.
class UnaryConstantExpr : public ConstantExpr {
  void anchor() override;
  void *operator new(size_t, unsigned) LLVM_DELETED_FUNCTION;
public:
  // allocate space for exactly one operand
  void *operator new(size_t s) {
    return User::operator new(s, 1);
  }
  UnaryConstantExpr(unsigned Opcode, Constant *C, Type *Ty)
    : ConstantExpr(Ty, Opcode, &Op<0>(), 1) {
    Op<0>() = C;
  }
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);
};

/// BinaryConstantExpr - This class is private to Constants.cpp, and is used
/// behind the scenes to implement binary constant exprs.
class BinaryConstantExpr : public ConstantExpr {
  void anchor() override;
  void *operator new(size_t, unsigned) LLVM_DELETED_FUNCTION;
public:
  // allocate space for exactly two operands
  void *operator new(size_t s) {
    return User::operator new(s, 2);
  }
  BinaryConstantExpr(unsigned Opcode, Constant *C1, Constant *C2,
                     unsigned Flags)
    : ConstantExpr(C1->getType(), Opcode, &Op<0>(), 2) {
    Op<0>() = C1;
    Op<1>() = C2;
    SubclassOptionalData = Flags;
  }
  /// Transparently provide more efficient getOperand methods.
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);
};

/// SelectConstantExpr - This class is private to Constants.cpp, and is used
/// behind the scenes to implement select constant exprs.
class SelectConstantExpr : public ConstantExpr {
  void anchor() override;
  void *operator new(size_t, unsigned) LLVM_DELETED_FUNCTION;
public:
  // allocate space for exactly three operands
  void *operator new(size_t s) {
    return User::operator new(s, 3);
  }
  SelectConstantExpr(Constant *C1, Constant *C2, Constant *C3)
    : ConstantExpr(C2->getType(), Instruction::Select, &Op<0>(), 3) {
    Op<0>() = C1;
    Op<1>() = C2;
    Op<2>() = C3;
  }
  /// Transparently provide more efficient getOperand methods.
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);
};

/// ExtractElementConstantExpr - This class is private to
/// Constants.cpp, and is used behind the scenes to implement
/// extractelement constant exprs.
class ExtractElementConstantExpr : public ConstantExpr {
  void anchor() override;
  void *operator new(size_t, unsigned) LLVM_DELETED_FUNCTION;
public:
  // allocate space for exactly two operands
  void *operator new(size_t s) {
    return User::operator new(s, 2);
  }
  ExtractElementConstantExpr(Constant *C1, Constant *C2)
    : ConstantExpr(cast<VectorType>(C1->getType())->getElementType(), 
                   Instruction::ExtractElement, &Op<0>(), 2) {
    Op<0>() = C1;
    Op<1>() = C2;
  }
  /// Transparently provide more efficient getOperand methods.
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);
};

/// InsertElementConstantExpr - This class is private to
/// Constants.cpp, and is used behind the scenes to implement
/// insertelement constant exprs.
class InsertElementConstantExpr : public ConstantExpr {
  void anchor() override;
  void *operator new(size_t, unsigned) LLVM_DELETED_FUNCTION;
public:
  // allocate space for exactly three operands
  void *operator new(size_t s) {
    return User::operator new(s, 3);
  }
  InsertElementConstantExpr(Constant *C1, Constant *C2, Constant *C3)
    : ConstantExpr(C1->getType(), Instruction::InsertElement, 
                   &Op<0>(), 3) {
    Op<0>() = C1;
    Op<1>() = C2;
    Op<2>() = C3;
  }
  /// Transparently provide more efficient getOperand methods.
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);
};

/// ShuffleVectorConstantExpr - This class is private to
/// Constants.cpp, and is used behind the scenes to implement
/// shufflevector constant exprs.
class ShuffleVectorConstantExpr : public ConstantExpr {
  void anchor() override;
  void *operator new(size_t, unsigned) LLVM_DELETED_FUNCTION;
public:
  // allocate space for exactly three operands
  void *operator new(size_t s) {
    return User::operator new(s, 3);
  }
  ShuffleVectorConstantExpr(Constant *C1, Constant *C2, Constant *C3)
  : ConstantExpr(VectorType::get(
                   cast<VectorType>(C1->getType())->getElementType(),
                   cast<VectorType>(C3->getType())->getNumElements()),
                 Instruction::ShuffleVector, 
                 &Op<0>(), 3) {
    Op<0>() = C1;
    Op<1>() = C2;
    Op<2>() = C3;
  }
  /// Transparently provide more efficient getOperand methods.
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);
};

/// ExtractValueConstantExpr - This class is private to
/// Constants.cpp, and is used behind the scenes to implement
/// extractvalue constant exprs.
class ExtractValueConstantExpr : public ConstantExpr {
  void anchor() override;
  void *operator new(size_t, unsigned) LLVM_DELETED_FUNCTION;
public:
  // allocate space for exactly one operand
  void *operator new(size_t s) {
    return User::operator new(s, 1);
  }
  ExtractValueConstantExpr(Constant *Agg, ArrayRef<unsigned> IdxList,
                           Type *DestTy)
      : ConstantExpr(DestTy, Instruction::ExtractValue, &Op<0>(), 1),
        Indices(IdxList.begin(), IdxList.end()) {
    Op<0>() = Agg;
  }

  /// Indices - These identify which value to extract.
  const SmallVector<unsigned, 4> Indices;

  /// Transparently provide more efficient getOperand methods.
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);
};

/// InsertValueConstantExpr - This class is private to
/// Constants.cpp, and is used behind the scenes to implement
/// insertvalue constant exprs.
class InsertValueConstantExpr : public ConstantExpr {
  void anchor() override;
  void *operator new(size_t, unsigned) LLVM_DELETED_FUNCTION;
public:
  // allocate space for exactly one operand
  void *operator new(size_t s) {
    return User::operator new(s, 2);
  }
  InsertValueConstantExpr(Constant *Agg, Constant *Val,
                          ArrayRef<unsigned> IdxList, Type *DestTy)
      : ConstantExpr(DestTy, Instruction::InsertValue, &Op<0>(), 2),
        Indices(IdxList.begin(), IdxList.end()) {
    Op<0>() = Agg;
    Op<1>() = Val;
  }

  /// Indices - These identify the position for the insertion.
  const SmallVector<unsigned, 4> Indices;

  /// Transparently provide more efficient getOperand methods.
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);
};


/// GetElementPtrConstantExpr - This class is private to Constants.cpp, and is
/// used behind the scenes to implement getelementpr constant exprs.
class GetElementPtrConstantExpr : public ConstantExpr {
  void anchor() override;
  GetElementPtrConstantExpr(Constant *C, ArrayRef<Constant*> IdxList,
                            Type *DestTy);
public:
  static GetElementPtrConstantExpr *Create(Constant *C,
                                           ArrayRef<Constant*> IdxList,
                                           Type *DestTy,
                                           unsigned Flags) {
    GetElementPtrConstantExpr *Result =
      new(IdxList.size() + 1) GetElementPtrConstantExpr(C, IdxList, DestTy);
    Result->SubclassOptionalData = Flags;
    return Result;
  }
  /// Transparently provide more efficient getOperand methods.
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);
};

// CompareConstantExpr - This class is private to Constants.cpp, and is used
// behind the scenes to implement ICmp and FCmp constant expressions. This is
// needed in order to store the predicate value for these instructions.
class CompareConstantExpr : public ConstantExpr {
  void anchor() override;
  void *operator new(size_t, unsigned) LLVM_DELETED_FUNCTION;
public:
  // allocate space for exactly two operands
  void *operator new(size_t s) {
    return User::operator new(s, 2);
  }
  unsigned short predicate;
  CompareConstantExpr(Type *ty, Instruction::OtherOps opc,
                      unsigned short pred,  Constant* LHS, Constant* RHS)
    : ConstantExpr(ty, opc, &Op<0>(), 2), predicate(pred) {
    Op<0>() = LHS;
    Op<1>() = RHS;
  }
  /// Transparently provide more efficient getOperand methods.
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);
};

template <>
struct OperandTraits<UnaryConstantExpr> :
  public FixedNumOperandTraits<UnaryConstantExpr, 1> {
};
DEFINE_TRANSPARENT_OPERAND_ACCESSORS(UnaryConstantExpr, Value)

template <>
struct OperandTraits<BinaryConstantExpr> :
  public FixedNumOperandTraits<BinaryConstantExpr, 2> {
};
DEFINE_TRANSPARENT_OPERAND_ACCESSORS(BinaryConstantExpr, Value)

template <>
struct OperandTraits<SelectConstantExpr> :
  public FixedNumOperandTraits<SelectConstantExpr, 3> {
};
DEFINE_TRANSPARENT_OPERAND_ACCESSORS(SelectConstantExpr, Value)

template <>
struct OperandTraits<ExtractElementConstantExpr> :
  public FixedNumOperandTraits<ExtractElementConstantExpr, 2> {
};
DEFINE_TRANSPARENT_OPERAND_ACCESSORS(ExtractElementConstantExpr, Value)

template <>
struct OperandTraits<InsertElementConstantExpr> :
  public FixedNumOperandTraits<InsertElementConstantExpr, 3> {
};
DEFINE_TRANSPARENT_OPERAND_ACCESSORS(InsertElementConstantExpr, Value)

template <>
struct OperandTraits<ShuffleVectorConstantExpr> :
    public FixedNumOperandTraits<ShuffleVectorConstantExpr, 3> {
};
DEFINE_TRANSPARENT_OPERAND_ACCESSORS(ShuffleVectorConstantExpr, Value)

template <>
struct OperandTraits<ExtractValueConstantExpr> :
  public FixedNumOperandTraits<ExtractValueConstantExpr, 1> {
};
DEFINE_TRANSPARENT_OPERAND_ACCESSORS(ExtractValueConstantExpr, Value)

template <>
struct OperandTraits<InsertValueConstantExpr> :
  public FixedNumOperandTraits<InsertValueConstantExpr, 2> {
};
DEFINE_TRANSPARENT_OPERAND_ACCESSORS(InsertValueConstantExpr, Value)

template <>
struct OperandTraits<GetElementPtrConstantExpr> :
  public VariadicOperandTraits<GetElementPtrConstantExpr, 1> {
};

DEFINE_TRANSPARENT_OPERAND_ACCESSORS(GetElementPtrConstantExpr, Value)


template <>
struct OperandTraits<CompareConstantExpr> :
  public FixedNumOperandTraits<CompareConstantExpr, 2> {
};
DEFINE_TRANSPARENT_OPERAND_ACCESSORS(CompareConstantExpr, Value)

struct InlineAsmKeyType;
struct ConstantExprKeyType;

template <class ConstantClass> struct ConstantInfo;
template <> struct ConstantInfo<ConstantExpr> {
  typedef ConstantExprKeyType ValType;
  typedef Type TypeClass;
};
template <> struct ConstantInfo<InlineAsm> {
  typedef InlineAsmKeyType ValType;
  typedef PointerType TypeClass;
};

struct InlineAsmKeyType {
  StringRef AsmString;
  StringRef Constraints;
  bool HasSideEffects;
  bool IsAlignStack;
  InlineAsm::AsmDialect AsmDialect;

  InlineAsmKeyType(StringRef AsmString, StringRef Constraints,
                   bool HasSideEffects, bool IsAlignStack,
                   InlineAsm::AsmDialect AsmDialect)
      : AsmString(AsmString), Constraints(Constraints),
        HasSideEffects(HasSideEffects), IsAlignStack(IsAlignStack),
        AsmDialect(AsmDialect) {}
  InlineAsmKeyType(const InlineAsm *Asm, SmallVectorImpl<Constant *> &)
      : AsmString(Asm->getAsmString()), Constraints(Asm->getConstraintString()),
        HasSideEffects(Asm->hasSideEffects()),
        IsAlignStack(Asm->isAlignStack()), AsmDialect(Asm->getDialect()) {}

  bool operator==(const InlineAsmKeyType &X) const {
    return HasSideEffects == X.HasSideEffects &&
           IsAlignStack == X.IsAlignStack && AsmDialect == X.AsmDialect &&
           AsmString == X.AsmString && Constraints == X.Constraints;
  }
  bool operator==(const InlineAsm *Asm) const {
    return HasSideEffects == Asm->hasSideEffects() &&
           IsAlignStack == Asm->isAlignStack() &&
           AsmDialect == Asm->getDialect() &&
           AsmString == Asm->getAsmString() &&
           Constraints == Asm->getConstraintString();
  }
  unsigned getHash() const {
    return hash_combine(AsmString, Constraints, HasSideEffects, IsAlignStack,
                        AsmDialect);
  }

  typedef typename ConstantInfo<InlineAsm>::TypeClass TypeClass;
  InlineAsm *create(TypeClass *Ty) const {
    return new InlineAsm(Ty, AsmString, Constraints, HasSideEffects,
                         IsAlignStack, AsmDialect);
  }
};

struct ConstantExprKeyType {
  uint8_t Opcode;
  uint8_t SubclassOptionalData;
  uint16_t SubclassData;
  ArrayRef<Constant *> Ops;
  ArrayRef<unsigned> Indexes;

  ConstantExprKeyType(unsigned Opcode, ArrayRef<Constant *> Ops,
                      unsigned short SubclassData = 0,
                      unsigned short SubclassOptionalData = 0,
                      ArrayRef<unsigned> Indexes = None)
      : Opcode(Opcode), SubclassOptionalData(SubclassOptionalData),
        SubclassData(SubclassData), Ops(Ops), Indexes(Indexes) {}
  ConstantExprKeyType(const ConstantExpr *CE,
                      SmallVectorImpl<Constant *> &Storage)
      : Opcode(CE->getOpcode()),
        SubclassOptionalData(CE->getRawSubclassOptionalData()),
        SubclassData(CE->isCompare() ? CE->getPredicate() : 0),
        Indexes(CE->hasIndices() ? CE->getIndices() : ArrayRef<unsigned>()) {
    assert(Storage.empty() && "Expected empty storage");
    for (unsigned I = 0, E = CE->getNumOperands(); I != E; ++I)
      Storage.push_back(CE->getOperand(I));
    Ops = Storage;
  }

  bool operator==(const ConstantExprKeyType &X) const {
    return Opcode == X.Opcode && SubclassData == X.SubclassData &&
           SubclassOptionalData == X.SubclassOptionalData && Ops == X.Ops &&
           Indexes == X.Indexes;
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
    if (Indexes != (CE->hasIndices() ? CE->getIndices() : ArrayRef<unsigned>()))
      return false;
    return true;
  }

  unsigned getHash() const {
    return hash_combine(Opcode, SubclassOptionalData, SubclassData,
                        hash_combine_range(Ops.begin(), Ops.end()),
                        hash_combine_range(Indexes.begin(), Indexes.end()));
  }

  typedef typename ConstantInfo<ConstantExpr>::TypeClass TypeClass;
  ConstantExpr *create(TypeClass *Ty) const {
    switch (Opcode) {
    default:
      if (Instruction::isCast(Opcode))
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
      return new ShuffleVectorConstantExpr(Ops[0], Ops[1], Ops[2]);
    case Instruction::InsertValue:
      return new InsertValueConstantExpr(Ops[0], Ops[1], Indexes, Ty);
    case Instruction::ExtractValue:
      return new ExtractValueConstantExpr(Ops[0], Indexes, Ty);
    case Instruction::GetElementPtr:
      return GetElementPtrConstantExpr::Create(Ops[0], Ops.slice(1), Ty,
                                               SubclassOptionalData);
    case Instruction::ICmp:
      return new CompareConstantExpr(Ty, Instruction::ICmp, SubclassData,
                                     Ops[0], Ops[1]);
    case Instruction::FCmp:
      return new CompareConstantExpr(Ty, Instruction::FCmp, SubclassData,
                                     Ops[0], Ops[1]);
    }
  }
};

// The number of operands for each ConstantCreator::create method is
// determined by the ConstantTraits template.
// ConstantCreator - A class that is used to create constants by
// ConstantUniqueMap*.  This class should be partially specialized if there is
// something strange that needs to be done to interface to the ctor for the
// constant.
//
template<typename T, typename Alloc>
struct ConstantTraits< std::vector<T, Alloc> > {
  static unsigned uses(const std::vector<T, Alloc>& v) {
    return v.size();
  }
};

template<>
struct ConstantTraits<Constant *> {
  static unsigned uses(Constant * const & v) {
    return 1;
  }
};

template<class ConstantClass, class TypeClass, class ValType>
struct ConstantCreator {
  static ConstantClass *create(TypeClass *Ty, const ValType &V) {
    return new(ConstantTraits<ValType>::uses(V)) ConstantClass(Ty, V);
  }
};

template<class ConstantClass, class TypeClass>
struct ConstantArrayCreator {
  static ConstantClass *create(TypeClass *Ty, ArrayRef<Constant*> V) {
    return new(V.size()) ConstantClass(Ty, V);
  }
};

template <class ConstantClass> class ConstantUniqueMap {
public:
  typedef typename ConstantInfo<ConstantClass>::ValType ValType;
  typedef typename ConstantInfo<ConstantClass>::TypeClass TypeClass;
  typedef std::pair<TypeClass *, ValType> LookupKey;

private:
  struct MapInfo {
    typedef DenseMapInfo<ConstantClass *> ConstantClassInfo;
    static inline ConstantClass *getEmptyKey() {
      return ConstantClassInfo::getEmptyKey();
    }
    static inline ConstantClass *getTombstoneKey() {
      return ConstantClassInfo::getTombstoneKey();
    }
    static unsigned getHashValue(const ConstantClass *CP) {
      SmallVector<Constant *, 8> Storage;
      return getHashValue(LookupKey(CP->getType(), ValType(CP, Storage)));
    }
    static bool isEqual(const ConstantClass *LHS, const ConstantClass *RHS) {
      return LHS == RHS;
    }
    static unsigned getHashValue(const LookupKey &Val) {
      return hash_combine(Val.first, Val.second.getHash());
    }
    static bool isEqual(const LookupKey &LHS, const ConstantClass *RHS) {
      if (RHS == getEmptyKey() || RHS == getTombstoneKey())
        return false;
      if (LHS.first != RHS->getType())
        return false;
      return LHS.second == RHS;
    }
  };

public:
  typedef DenseMap<ConstantClass *, char, MapInfo> MapTy;

private:
  MapTy Map;

public:
  typename MapTy::iterator map_begin() { return Map.begin(); }
  typename MapTy::iterator map_end() { return Map.end(); }

  void freeConstants() {
    for (auto &I : Map)
      // Asserts that use_empty().
      delete I.first;
  }

private:
  ConstantClass *create(TypeClass *Ty, ValType V) {
    ConstantClass *Result = V.create(Ty);

    assert(Result->getType() == Ty && "Type specified is not correct!");
    insert(Result);

    return Result;
  }

public:
  /// Return the specified constant from the map, creating it if necessary.
  ConstantClass *getOrCreate(TypeClass *Ty, ValType V) {
    LookupKey Lookup(Ty, V);
    ConstantClass *Result = nullptr;

    auto I = find(Lookup);
    if (I == Map.end())
      Result = create(Ty, V);
    else
      Result = I->first;
    assert(Result && "Unexpected nullptr");

    return Result;
  }

  /// Find the constant by lookup key.
  typename MapTy::iterator find(LookupKey Lookup) {
    return Map.find_as(Lookup);
  }

  /// Insert the constant into its proper slot.
  void insert(ConstantClass *CP) { Map[CP] = '\0'; }

  /// Remove this constant from the map
  void remove(ConstantClass *CP) {
    typename MapTy::iterator I = Map.find(CP);
    assert(I != Map.end() && "Constant not found in constant table!");
    assert(I->first == CP && "Didn't find correct element?");
    Map.erase(I);
  }

  void dump() const { DEBUG(dbgs() << "Constant.cpp: ConstantUniqueMap\n"); }
};

// Unique map for aggregate constants
template<class TypeClass, class ConstantClass>
class ConstantAggrUniqueMap {
public:
  typedef ArrayRef<Constant*> Operands;
  typedef std::pair<TypeClass*, Operands> LookupKey;
private:
  struct MapInfo {
    typedef DenseMapInfo<ConstantClass*> ConstantClassInfo;
    typedef DenseMapInfo<Constant*> ConstantInfo;
    typedef DenseMapInfo<TypeClass*> TypeClassInfo;
    static inline ConstantClass* getEmptyKey() {
      return ConstantClassInfo::getEmptyKey();
    }
    static inline ConstantClass* getTombstoneKey() {
      return ConstantClassInfo::getTombstoneKey();
    }
    static unsigned getHashValue(const ConstantClass *CP) {
      SmallVector<Constant*, 8> CPOperands;
      CPOperands.reserve(CP->getNumOperands());
      for (unsigned I = 0, E = CP->getNumOperands(); I < E; ++I)
        CPOperands.push_back(CP->getOperand(I));
      return getHashValue(LookupKey(CP->getType(), CPOperands));
    }
    static bool isEqual(const ConstantClass *LHS, const ConstantClass *RHS) {
      return LHS == RHS;
    }
    static unsigned getHashValue(const LookupKey &Val) {
      return hash_combine(Val.first, hash_combine_range(Val.second.begin(),
                                                        Val.second.end()));
    }
    static bool isEqual(const LookupKey &LHS, const ConstantClass *RHS) {
      if (RHS == getEmptyKey() || RHS == getTombstoneKey())
        return false;
      if (LHS.first != RHS->getType()
          || LHS.second.size() != RHS->getNumOperands())
        return false;
      for (unsigned I = 0, E = RHS->getNumOperands(); I < E; ++I) {
        if (LHS.second[I] != RHS->getOperand(I))
          return false;
      }
      return true;
    }
  };
public:
  typedef DenseMap<ConstantClass *, char, MapInfo> MapTy;

private:
  /// Map - This is the main map from the element descriptor to the Constants.
  /// This is the primary way we avoid creating two of the same shape
  /// constant.
  MapTy Map;

public:
  typename MapTy::iterator map_begin() { return Map.begin(); }
  typename MapTy::iterator map_end() { return Map.end(); }

  void freeConstants() {
    for (typename MapTy::iterator I=Map.begin(), E=Map.end();
         I != E; ++I) {
      // Asserts that use_empty().
      delete I->first;
    }
  }

private:
  typename MapTy::iterator findExistingElement(ConstantClass *CP) {
    return Map.find(CP);
  }

  ConstantClass *Create(TypeClass *Ty, Operands V, typename MapTy::iterator I) {
    ConstantClass* Result =
      ConstantArrayCreator<ConstantClass,TypeClass>::create(Ty, V);

    assert(Result->getType() == Ty && "Type specified is not correct!");
    Map[Result] = '\0';

    return Result;
  }
public:

  /// getOrCreate - Return the specified constant from the map, creating it if
  /// necessary.
  ConstantClass *getOrCreate(TypeClass *Ty, Operands V) {
    LookupKey Lookup(Ty, V);
    ConstantClass* Result = nullptr;

    typename MapTy::iterator I = Map.find_as(Lookup);
    // Is it in the map?
    if (I != Map.end())
      Result = I->first;

    if (!Result) {
      // If no preexisting value, create one now...
      Result = Create(Ty, V, I);
    }

    return Result;
  }

  /// Find the constant by lookup key.
  typename MapTy::iterator find(LookupKey Lookup) {
    return Map.find_as(Lookup);
  }

  /// Insert the constant into its proper slot.
  void insert(ConstantClass *CP) {
    Map[CP] = '\0';
  }

  /// Remove this constant from the map
  void remove(ConstantClass *CP) {
    typename MapTy::iterator I = findExistingElement(CP);
    assert(I != Map.end() && "Constant not found in constant table!");
    assert(I->first == CP && "Didn't find correct element?");
    Map.erase(I);
  }

  void dump() const {
    DEBUG(dbgs() << "Constant.cpp: ConstantUniqueMap\n");
  }
};

} // end namespace llvm

#endif
