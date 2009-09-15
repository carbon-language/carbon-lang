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

#ifndef LLVM_CONSTANTSCONTEXT_H
#define LLVM_CONSTANTSCONTEXT_H

#include "llvm/Instructions.h"
#include "llvm/Operator.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/System/Mutex.h"
#include "llvm/System/RWMutex.h"
#include <map>

namespace llvm {
template<class ValType>
struct ConstantTraits;

/// UnaryConstantExpr - This class is private to Constants.cpp, and is used
/// behind the scenes to implement unary constant exprs.
class UnaryConstantExpr : public ConstantExpr {
  void *operator new(size_t, unsigned);  // DO NOT IMPLEMENT
public:
  // allocate space for exactly one operand
  void *operator new(size_t s) {
    return User::operator new(s, 1);
  }
  UnaryConstantExpr(unsigned Opcode, Constant *C, const Type *Ty)
    : ConstantExpr(Ty, Opcode, &Op<0>(), 1) {
    Op<0>() = C;
  }
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);
};

/// BinaryConstantExpr - This class is private to Constants.cpp, and is used
/// behind the scenes to implement binary constant exprs.
class BinaryConstantExpr : public ConstantExpr {
  void *operator new(size_t, unsigned);  // DO NOT IMPLEMENT
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
  void *operator new(size_t, unsigned);  // DO NOT IMPLEMENT
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
  void *operator new(size_t, unsigned);  // DO NOT IMPLEMENT
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
  void *operator new(size_t, unsigned);  // DO NOT IMPLEMENT
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
  void *operator new(size_t, unsigned);  // DO NOT IMPLEMENT
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
  void *operator new(size_t, unsigned);  // DO NOT IMPLEMENT
public:
  // allocate space for exactly one operand
  void *operator new(size_t s) {
    return User::operator new(s, 1);
  }
  ExtractValueConstantExpr(Constant *Agg,
                           const SmallVector<unsigned, 4> &IdxList,
                           const Type *DestTy)
    : ConstantExpr(DestTy, Instruction::ExtractValue, &Op<0>(), 1),
      Indices(IdxList) {
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
  void *operator new(size_t, unsigned);  // DO NOT IMPLEMENT
public:
  // allocate space for exactly one operand
  void *operator new(size_t s) {
    return User::operator new(s, 2);
  }
  InsertValueConstantExpr(Constant *Agg, Constant *Val,
                          const SmallVector<unsigned, 4> &IdxList,
                          const Type *DestTy)
    : ConstantExpr(DestTy, Instruction::InsertValue, &Op<0>(), 2),
      Indices(IdxList) {
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
  GetElementPtrConstantExpr(Constant *C, const std::vector<Constant*> &IdxList,
                            const Type *DestTy);
public:
  static GetElementPtrConstantExpr *Create(Constant *C,
                                           const std::vector<Constant*>&IdxList,
                                           const Type *DestTy,
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
struct CompareConstantExpr : public ConstantExpr {
  void *operator new(size_t, unsigned);  // DO NOT IMPLEMENT
  // allocate space for exactly two operands
  void *operator new(size_t s) {
    return User::operator new(s, 2);
  }
  unsigned short predicate;
  CompareConstantExpr(const Type *ty, Instruction::OtherOps opc,
                      unsigned short pred,  Constant* LHS, Constant* RHS)
    : ConstantExpr(ty, opc, &Op<0>(), 2), predicate(pred) {
    Op<0>() = LHS;
    Op<1>() = RHS;
  }
  /// Transparently provide more efficient getOperand methods.
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);
};

template <>
struct OperandTraits<UnaryConstantExpr> : public FixedNumOperandTraits<1> {
};
DEFINE_TRANSPARENT_OPERAND_ACCESSORS(UnaryConstantExpr, Value)

template <>
struct OperandTraits<BinaryConstantExpr> : public FixedNumOperandTraits<2> {
};
DEFINE_TRANSPARENT_OPERAND_ACCESSORS(BinaryConstantExpr, Value)

template <>
struct OperandTraits<SelectConstantExpr> : public FixedNumOperandTraits<3> {
};
DEFINE_TRANSPARENT_OPERAND_ACCESSORS(SelectConstantExpr, Value)

template <>
struct OperandTraits<ExtractElementConstantExpr> : public FixedNumOperandTraits<2> {
};
DEFINE_TRANSPARENT_OPERAND_ACCESSORS(ExtractElementConstantExpr, Value)

template <>
struct OperandTraits<InsertElementConstantExpr> : public FixedNumOperandTraits<3> {
};
DEFINE_TRANSPARENT_OPERAND_ACCESSORS(InsertElementConstantExpr, Value)

template <>
struct OperandTraits<ShuffleVectorConstantExpr> : public FixedNumOperandTraits<3> {
};
DEFINE_TRANSPARENT_OPERAND_ACCESSORS(ShuffleVectorConstantExpr, Value)

template <>
struct OperandTraits<ExtractValueConstantExpr> : public FixedNumOperandTraits<1> {
};
DEFINE_TRANSPARENT_OPERAND_ACCESSORS(ExtractValueConstantExpr, Value)

template <>
struct OperandTraits<InsertValueConstantExpr> : public FixedNumOperandTraits<2> {
};
DEFINE_TRANSPARENT_OPERAND_ACCESSORS(InsertValueConstantExpr, Value)

template <>
struct OperandTraits<GetElementPtrConstantExpr> : public VariadicOperandTraits<1> {
};

DEFINE_TRANSPARENT_OPERAND_ACCESSORS(GetElementPtrConstantExpr, Value)


template <>
struct OperandTraits<CompareConstantExpr> : public FixedNumOperandTraits<2> {
};
DEFINE_TRANSPARENT_OPERAND_ACCESSORS(CompareConstantExpr, Value)

struct ExprMapKeyType {
  typedef SmallVector<unsigned, 4> IndexList;

  ExprMapKeyType(unsigned opc,
      const std::vector<Constant*> &ops,
      unsigned short flags = 0,
      unsigned short optionalflags = 0,
      const IndexList &inds = IndexList())
        : opcode(opc), subclassoptionaldata(optionalflags), subclassdata(flags),
        operands(ops), indices(inds) {}
  uint8_t opcode;
  uint8_t subclassoptionaldata;
  uint16_t subclassdata;
  std::vector<Constant*> operands;
  IndexList indices;
  bool operator==(const ExprMapKeyType& that) const {
    return this->opcode == that.opcode &&
           this->subclassdata == that.subclassdata &&
           this->subclassoptionaldata == that.subclassoptionaldata &&
           this->operands == that.operands &&
           this->indices == that.indices;
  }
  bool operator<(const ExprMapKeyType & that) const {
    if (this->opcode != that.opcode) return this->opcode < that.opcode;
    if (this->operands != that.operands) return this->operands < that.operands;
    if (this->subclassdata != that.subclassdata)
      return this->subclassdata < that.subclassdata;
    if (this->subclassoptionaldata != that.subclassoptionaldata)
      return this->subclassoptionaldata < that.subclassoptionaldata;
    if (this->indices != that.indices) return this->indices < that.indices;
    return false;
  }

  bool operator!=(const ExprMapKeyType& that) const {
    return !(*this == that);
  }
};

// The number of operands for each ConstantCreator::create method is
// determined by the ConstantTraits template.
// ConstantCreator - A class that is used to create constants by
// ValueMap*.  This class should be partially specialized if there is
// something strange that needs to be done to interface to the ctor for the
// constant.
//
template<typename T, typename Alloc>
struct ConstantTraits< std::vector<T, Alloc> > {
  static unsigned uses(const std::vector<T, Alloc>& v) {
    return v.size();
  }
};

template<class ConstantClass, class TypeClass, class ValType>
struct ConstantCreator {
  static ConstantClass *create(const TypeClass *Ty, const ValType &V) {
    return new(ConstantTraits<ValType>::uses(V)) ConstantClass(Ty, V);
  }
};

template<class ConstantClass>
struct ConstantKeyData {
  typedef void ValType;
  static ValType getValType(ConstantClass *C) {
    llvm_unreachable("Unknown Constant type!");
  }
};

template<>
struct ConstantCreator<ConstantExpr, Type, ExprMapKeyType> {
  static ConstantExpr *create(const Type *Ty, const ExprMapKeyType &V,
      unsigned short pred = 0) {
    if (Instruction::isCast(V.opcode))
      return new UnaryConstantExpr(V.opcode, V.operands[0], Ty);
    if ((V.opcode >= Instruction::BinaryOpsBegin &&
         V.opcode < Instruction::BinaryOpsEnd))
      return new BinaryConstantExpr(V.opcode, V.operands[0], V.operands[1],
                                    V.subclassoptionaldata);
    if (V.opcode == Instruction::Select)
      return new SelectConstantExpr(V.operands[0], V.operands[1], 
                                    V.operands[2]);
    if (V.opcode == Instruction::ExtractElement)
      return new ExtractElementConstantExpr(V.operands[0], V.operands[1]);
    if (V.opcode == Instruction::InsertElement)
      return new InsertElementConstantExpr(V.operands[0], V.operands[1],
                                           V.operands[2]);
    if (V.opcode == Instruction::ShuffleVector)
      return new ShuffleVectorConstantExpr(V.operands[0], V.operands[1],
                                           V.operands[2]);
    if (V.opcode == Instruction::InsertValue)
      return new InsertValueConstantExpr(V.operands[0], V.operands[1],
                                         V.indices, Ty);
    if (V.opcode == Instruction::ExtractValue)
      return new ExtractValueConstantExpr(V.operands[0], V.indices, Ty);
    if (V.opcode == Instruction::GetElementPtr) {
      std::vector<Constant*> IdxList(V.operands.begin()+1, V.operands.end());
      return GetElementPtrConstantExpr::Create(V.operands[0], IdxList, Ty,
                                               V.subclassoptionaldata);
    }

    // The compare instructions are weird. We have to encode the predicate
    // value and it is combined with the instruction opcode by multiplying
    // the opcode by one hundred. We must decode this to get the predicate.
    if (V.opcode == Instruction::ICmp)
      return new CompareConstantExpr(Ty, Instruction::ICmp, V.subclassdata,
                                     V.operands[0], V.operands[1]);
    if (V.opcode == Instruction::FCmp) 
      return new CompareConstantExpr(Ty, Instruction::FCmp, V.subclassdata,
                                     V.operands[0], V.operands[1]);
    llvm_unreachable("Invalid ConstantExpr!");
    return 0;
  }
};

template<>
struct ConstantKeyData<ConstantExpr> {
  typedef ExprMapKeyType ValType;
  static ValType getValType(ConstantExpr *CE) {
    std::vector<Constant*> Operands;
    Operands.reserve(CE->getNumOperands());
    for (unsigned i = 0, e = CE->getNumOperands(); i != e; ++i)
      Operands.push_back(cast<Constant>(CE->getOperand(i)));
    return ExprMapKeyType(CE->getOpcode(), Operands,
        CE->isCompare() ? CE->getPredicate() : 0,
        CE->getRawSubclassOptionalData(),
        CE->hasIndices() ?
          CE->getIndices() : SmallVector<unsigned, 4>());
  }
};

// ConstantAggregateZero does not take extra "value" argument...
template<class ValType>
struct ConstantCreator<ConstantAggregateZero, Type, ValType> {
  static ConstantAggregateZero *create(const Type *Ty, const ValType &V){
    return new ConstantAggregateZero(Ty);
  }
};

template<>
struct ConstantKeyData<ConstantVector> {
  typedef std::vector<Constant*> ValType;
  static ValType getValType(ConstantVector *CP) {
    std::vector<Constant*> Elements;
    Elements.reserve(CP->getNumOperands());
    for (unsigned i = 0, e = CP->getNumOperands(); i != e; ++i)
      Elements.push_back(CP->getOperand(i));
    return Elements;
  }
};

template<>
struct ConstantKeyData<ConstantAggregateZero> {
  typedef char ValType;
  static ValType getValType(ConstantAggregateZero *C) {
    return 0;
  }
};

template<>
struct ConstantKeyData<ConstantArray> {
  typedef std::vector<Constant*> ValType;
  static ValType getValType(ConstantArray *CA) {
    std::vector<Constant*> Elements;
    Elements.reserve(CA->getNumOperands());
    for (unsigned i = 0, e = CA->getNumOperands(); i != e; ++i)
      Elements.push_back(cast<Constant>(CA->getOperand(i)));
    return Elements;
  }
};

template<>
struct ConstantKeyData<ConstantStruct> {
  typedef std::vector<Constant*> ValType;
  static ValType getValType(ConstantStruct *CS) {
    std::vector<Constant*> Elements;
    Elements.reserve(CS->getNumOperands());
    for (unsigned i = 0, e = CS->getNumOperands(); i != e; ++i)
      Elements.push_back(cast<Constant>(CS->getOperand(i)));
    return Elements;
  }
};

// ConstantPointerNull does not take extra "value" argument...
template<class ValType>
struct ConstantCreator<ConstantPointerNull, PointerType, ValType> {
  static ConstantPointerNull *create(const PointerType *Ty, const ValType &V){
    return new ConstantPointerNull(Ty);
  }
};

template<>
struct ConstantKeyData<ConstantPointerNull> {
  typedef char ValType;
  static ValType getValType(ConstantPointerNull *C) {
    return 0;
  }
};

// UndefValue does not take extra "value" argument...
template<class ValType>
struct ConstantCreator<UndefValue, Type, ValType> {
  static UndefValue *create(const Type *Ty, const ValType &V) {
    return new UndefValue(Ty);
  }
};

template<>
struct ConstantKeyData<UndefValue> {
  typedef char ValType;
  static ValType getValType(UndefValue *C) {
    return 0;
  }
};

template<class ValType, class TypeClass, class ConstantClass,
         bool HasLargeKey = false /*true for arrays and structs*/ >
class ValueMap : public AbstractTypeUser {
public:
  typedef std::pair<const TypeClass*, ValType> MapKey;
  typedef std::map<MapKey, ConstantClass *> MapTy;
  typedef std::map<ConstantClass *, typename MapTy::iterator> InverseMapTy;
  typedef std::map<const DerivedType*, typename MapTy::iterator>
    AbstractTypeMapTy;
private:
  /// Map - This is the main map from the element descriptor to the Constants.
  /// This is the primary way we avoid creating two of the same shape
  /// constant.
  MapTy Map;
    
  /// InverseMap - If "HasLargeKey" is true, this contains an inverse mapping
  /// from the constants to their element in Map.  This is important for
  /// removal of constants from the array, which would otherwise have to scan
  /// through the map with very large keys.
  InverseMapTy InverseMap;

  /// AbstractTypeMap - Map for abstract type constants.
  ///
  AbstractTypeMapTy AbstractTypeMap;
    
  /// ValueMapLock - Mutex for this map.
  sys::SmartMutex<true> ValueMapLock;

public:
  // NOTE: This function is not locked.  It is the caller's responsibility
  // to enforce proper synchronization.
  typename MapTy::iterator map_begin() { return Map.begin(); }
  typename MapTy::iterator map_end() { return Map.end(); }

  void freeConstants() {
    for (typename MapTy::iterator I=Map.begin(), E=Map.end();
         I != E; ++I) {
      if (I->second->use_empty())
        delete I->second;
    }
  }
    
  /// InsertOrGetItem - Return an iterator for the specified element.
  /// If the element exists in the map, the returned iterator points to the
  /// entry and Exists=true.  If not, the iterator points to the newly
  /// inserted entry and returns Exists=false.  Newly inserted entries have
  /// I->second == 0, and should be filled in.
  /// NOTE: This function is not locked.  It is the caller's responsibility
  // to enforce proper synchronization.
  typename MapTy::iterator InsertOrGetItem(std::pair<MapKey, ConstantClass *>
                                 &InsertVal,
                                 bool &Exists) {
    std::pair<typename MapTy::iterator, bool> IP = Map.insert(InsertVal);
    Exists = !IP.second;
    return IP.first;
  }
    
private:
  typename MapTy::iterator FindExistingElement(ConstantClass *CP) {
    if (HasLargeKey) {
      typename InverseMapTy::iterator IMI = InverseMap.find(CP);
      assert(IMI != InverseMap.end() && IMI->second != Map.end() &&
             IMI->second->second == CP &&
             "InverseMap corrupt!");
      return IMI->second;
    }
      
    typename MapTy::iterator I =
      Map.find(MapKey(static_cast<const TypeClass*>(CP->getRawType()),
                      ConstantKeyData<ConstantClass>::getValType(CP)));
    if (I == Map.end() || I->second != CP) {
      // FIXME: This should not use a linear scan.  If this gets to be a
      // performance problem, someone should look at this.
      for (I = Map.begin(); I != Map.end() && I->second != CP; ++I)
        /* empty */;
    }
    return I;
  }
    
  void AddAbstractTypeUser(const Type *Ty, typename MapTy::iterator I) {
    // If the type of the constant is abstract, make sure that an entry
    // exists for it in the AbstractTypeMap.
    if (Ty->isAbstract()) {
      const DerivedType *DTy = static_cast<const DerivedType *>(Ty);
      typename AbstractTypeMapTy::iterator TI = AbstractTypeMap.find(DTy);

      if (TI == AbstractTypeMap.end()) {
        // Add ourselves to the ATU list of the type.
        cast<DerivedType>(DTy)->addAbstractTypeUser(this);

        AbstractTypeMap.insert(TI, std::make_pair(DTy, I));
      }
    }
  }

  ConstantClass* Create(const TypeClass *Ty, const ValType &V,
                        typename MapTy::iterator I) {
    ConstantClass* Result =
      ConstantCreator<ConstantClass,TypeClass,ValType>::create(Ty, V);

    assert(Result->getType() == Ty && "Type specified is not correct!");
    I = Map.insert(I, std::make_pair(MapKey(Ty, V), Result));

    if (HasLargeKey)  // Remember the reverse mapping if needed.
      InverseMap.insert(std::make_pair(Result, I));

    AddAbstractTypeUser(Ty, I);
      
    return Result;
  }
public:
    
  /// getOrCreate - Return the specified constant from the map, creating it if
  /// necessary.
  ConstantClass *getOrCreate(const TypeClass *Ty, const ValType &V) {
    sys::SmartScopedLock<true> Lock(ValueMapLock);
    MapKey Lookup(Ty, V);
    ConstantClass* Result = 0;
    
    typename MapTy::iterator I = Map.find(Lookup);
    // Is it in the map?  
    if (I != Map.end())
      Result = I->second;
        
    if (!Result) {
      // If no preexisting value, create one now...
      Result = Create(Ty, V, I);
    }
        
    return Result;
  }

  void UpdateAbstractTypeMap(const DerivedType *Ty,
                             typename MapTy::iterator I) {
    assert(AbstractTypeMap.count(Ty) &&
           "Abstract type not in AbstractTypeMap?");
    typename MapTy::iterator &ATMEntryIt = AbstractTypeMap[Ty];
    if (ATMEntryIt == I) {
      // Yes, we are removing the representative entry for this type.
      // See if there are any other entries of the same type.
      typename MapTy::iterator TmpIt = ATMEntryIt;

      // First check the entry before this one...
      if (TmpIt != Map.begin()) {
        --TmpIt;
        if (TmpIt->first.first != Ty) // Not the same type, move back...
          ++TmpIt;
      }

      // If we didn't find the same type, try to move forward...
      if (TmpIt == ATMEntryIt) {
        ++TmpIt;
        if (TmpIt == Map.end() || TmpIt->first.first != Ty)
          --TmpIt;   // No entry afterwards with the same type
      }

      // If there is another entry in the map of the same abstract type,
      // update the AbstractTypeMap entry now.
      if (TmpIt != ATMEntryIt) {
        ATMEntryIt = TmpIt;
      } else {
        // Otherwise, we are removing the last instance of this type
        // from the table.  Remove from the ATM, and from user list.
        cast<DerivedType>(Ty)->removeAbstractTypeUser(this);
        AbstractTypeMap.erase(Ty);
      }
    }
  }

  void remove(ConstantClass *CP) {
    sys::SmartScopedLock<true> Lock(ValueMapLock);
    typename MapTy::iterator I = FindExistingElement(CP);
    assert(I != Map.end() && "Constant not found in constant table!");
    assert(I->second == CP && "Didn't find correct element?");

    if (HasLargeKey)  // Remember the reverse mapping if needed.
      InverseMap.erase(CP);
      
    // Now that we found the entry, make sure this isn't the entry that
    // the AbstractTypeMap points to.
    const TypeClass *Ty = I->first.first;
    if (Ty->isAbstract())
      UpdateAbstractTypeMap(static_cast<const DerivedType *>(Ty), I);

    Map.erase(I);
  }

  /// MoveConstantToNewSlot - If we are about to change C to be the element
  /// specified by I, update our internal data structures to reflect this
  /// fact.
  /// NOTE: This function is not locked. It is the responsibility of the
  /// caller to enforce proper synchronization if using this method.
  void MoveConstantToNewSlot(ConstantClass *C, typename MapTy::iterator I) {
    // First, remove the old location of the specified constant in the map.
    typename MapTy::iterator OldI = FindExistingElement(C);
    assert(OldI != Map.end() && "Constant not found in constant table!");
    assert(OldI->second == C && "Didn't find correct element?");
      
    // If this constant is the representative element for its abstract type,
    // update the AbstractTypeMap so that the representative element is I.
    if (C->getType()->isAbstract()) {
      typename AbstractTypeMapTy::iterator ATI =
          AbstractTypeMap.find(C->getType());
      assert(ATI != AbstractTypeMap.end() &&
             "Abstract type not in AbstractTypeMap?");
      if (ATI->second == OldI)
        ATI->second = I;
    }
      
    // Remove the old entry from the map.
    Map.erase(OldI);
    
    // Update the inverse map so that we know that this constant is now
    // located at descriptor I.
    if (HasLargeKey) {
      assert(I->second == C && "Bad inversemap entry!");
      InverseMap[C] = I;
    }
  }
    
  void refineAbstractType(const DerivedType *OldTy, const Type *NewTy) {
    sys::SmartScopedLock<true> Lock(ValueMapLock);
    typename AbstractTypeMapTy::iterator I = AbstractTypeMap.find(OldTy);

    assert(I != AbstractTypeMap.end() &&
           "Abstract type not in AbstractTypeMap?");

    // Convert a constant at a time until the last one is gone.  The last one
    // leaving will remove() itself, causing the AbstractTypeMapEntry to be
    // eliminated eventually.
    do {
      ConstantClass *C = I->second->second;
      MapKey Key(cast<TypeClass>(NewTy),
                 ConstantKeyData<ConstantClass>::getValType(C));

      std::pair<typename MapTy::iterator, bool> IP =
        Map.insert(std::make_pair(Key, C));
      if (IP.second) {
        // The map didn't previously have an appropriate constant in the
        // new type.
        
        // Remove the old entry.
        typename MapTy::iterator OldI =
          Map.find(MapKey(cast<TypeClass>(OldTy), IP.first->first.second));
        assert(OldI != Map.end() && "Constant not in map!");
        UpdateAbstractTypeMap(OldTy, OldI);
        Map.erase(OldI);

        // Set the constant's type. This is done in place!
        setType(C, NewTy);

        // Update the inverse map so that we know that this constant is now
        // located at descriptor I.
        if (HasLargeKey)
          InverseMap[C] = IP.first;

        AddAbstractTypeUser(NewTy, IP.first);
      } else {
        // The map already had an appropriate constant in the new type, so
        // there's no longer a need for the old constant.
        C->uncheckedReplaceAllUsesWith(IP.first->second);
        C->destroyConstant();    // This constant is now dead, destroy it.
      }
      I = AbstractTypeMap.find(OldTy);
    } while (I != AbstractTypeMap.end());
  }

  // If the type became concrete without being refined to any other existing
  // type, we just remove ourselves from the ATU list.
  void typeBecameConcrete(const DerivedType *AbsTy) {
    AbsTy->removeAbstractTypeUser(this);
  }

  void dump() const {
    DEBUG(errs() << "Constant.cpp: ValueMap\n");
  }
};

}

#endif
