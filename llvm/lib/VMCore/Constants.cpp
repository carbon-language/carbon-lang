//===-- Constants.cpp - Implement Constant nodes --------------------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements the Constant* classes...
//
//===----------------------------------------------------------------------===//

#include "llvm/Constants.h"
#include "ConstantFolding.h"
#include "llvm/DerivedTypes.h"
#include "llvm/iMemory.h"
#include "llvm/SymbolTable.h"
#include "llvm/Module.h"
#include "Support/StringExtras.h"
#include <algorithm>
using namespace llvm;

ConstantBool *ConstantBool::True  = new ConstantBool(true);
ConstantBool *ConstantBool::False = new ConstantBool(false);


//===----------------------------------------------------------------------===//
//                              Constant Class
//===----------------------------------------------------------------------===//

// Specialize setName to take care of symbol table majik
void Constant::setName(const std::string &Name, SymbolTable *ST) {
  assert(ST && "Type::setName - Must provide symbol table argument!");

  if (Name.size()) ST->insert(Name, this);
}

void Constant::destroyConstantImpl() {
  // When a Constant is destroyed, there may be lingering
  // references to the constant by other constants in the constant pool.  These
  // constants are implicitly dependent on the module that is being deleted,
  // but they don't know that.  Because we only find out when the CPV is
  // deleted, we must now notify all of our users (that should only be
  // Constants) that they are, in fact, invalid now and should be deleted.
  //
  while (!use_empty()) {
    Value *V = use_back();
#ifndef NDEBUG      // Only in -g mode...
    if (!isa<Constant>(V))
      std::cerr << "While deleting: " << *this
                << "\n\nUse still stuck around after Def is destroyed: "
                << *V << "\n\n";
#endif
    assert(isa<Constant>(V) && "References remain to Constant being destroyed");
    Constant *CPV = cast<Constant>(V);
    CPV->destroyConstant();

    // The constant should remove itself from our use list...
    assert((use_empty() || use_back() != V) && "Constant not removed!");
  }

  // Value has no outstanding references it is safe to delete it now...
  delete this;
}

static std::map<const Type *, Constant*> NullValues;

// Static constructor to create a '0' constant of arbitrary type...
Constant *Constant::getNullValue(const Type *Ty) {
  switch (Ty->getPrimitiveID()) {
  case Type::BoolTyID: {
    static Constant *NullBool = ConstantBool::get(false);
    return NullBool;
  }
  case Type::SByteTyID: {
    static Constant *NullSByte = ConstantSInt::get(Type::SByteTy, 0);
    return NullSByte;
  }
  case Type::UByteTyID: {
    static Constant *NullUByte = ConstantUInt::get(Type::UByteTy, 0);
    return NullUByte;
  }
  case Type::ShortTyID: {
    static Constant *NullShort = ConstantSInt::get(Type::ShortTy, 0);
    return NullShort;
  }
  case Type::UShortTyID: {
    static Constant *NullUShort = ConstantUInt::get(Type::UShortTy, 0);
    return NullUShort;
  }
  case Type::IntTyID: {
    static Constant *NullInt = ConstantSInt::get(Type::IntTy, 0);
    return NullInt;
  }
  case Type::UIntTyID: {
    static Constant *NullUInt = ConstantUInt::get(Type::UIntTy, 0);
    return NullUInt;
  }
  case Type::LongTyID: {
    static Constant *NullLong = ConstantSInt::get(Type::LongTy, 0);
    return NullLong;
  }
  case Type::ULongTyID: {
    static Constant *NullULong = ConstantUInt::get(Type::ULongTy, 0);
    return NullULong;
  }

  case Type::FloatTyID: {
    static Constant *NullFloat = ConstantFP::get(Type::FloatTy, 0);
    return NullFloat;
  }
  case Type::DoubleTyID: {
    static Constant *NullDouble = ConstantFP::get(Type::DoubleTy, 0);
    return NullDouble;
  }

  case Type::PointerTyID: 
    return ConstantPointerNull::get(cast<PointerType>(Ty));

  case Type::StructTyID: {
    if (!Ty->isAbstract())
      if (Constant *V = NullValues[Ty])
        return V;

    const StructType *ST = cast<StructType>(Ty);
    std::vector<Constant*> Elements;
    Elements.resize(ST->getNumElements());
    for (unsigned i = 0, e = ST->getNumElements(); i != e; ++i)
      Elements[i] = Constant::getNullValue(ST->getElementType(i));
    Constant *Ret = ConstantStruct::get(ST, Elements);
    if (!Ty->isAbstract())
      NullValues[Ty] = Ret;
    return Ret;
  }
  case Type::ArrayTyID: {
    if (!Ty->isAbstract())
      if (Constant *V = NullValues[Ty])
        return V;

    const ArrayType *AT = cast<ArrayType>(Ty);
    Constant *El = Constant::getNullValue(AT->getElementType());
    unsigned NumElements = AT->getNumElements();
    Constant *Ret = ConstantArray::get(AT,
                                       std::vector<Constant*>(NumElements, El));
    if (!Ty->isAbstract())
      NullValues[Ty] = Ret;
    return Ret;
  }
  default:
    // Function, Type, Label, or Opaque type?
    assert(0 && "Cannot create a null constant of that type!");
    return 0;
  }
}

// Static constructor to create the maximum constant of an integral type...
ConstantIntegral *ConstantIntegral::getMaxValue(const Type *Ty) {
  switch (Ty->getPrimitiveID()) {
  case Type::BoolTyID:   return ConstantBool::True;
  case Type::SByteTyID:
  case Type::ShortTyID:
  case Type::IntTyID:
  case Type::LongTyID: {
    // Calculate 011111111111111... 
    unsigned TypeBits = Ty->getPrimitiveSize()*8;
    int64_t Val = INT64_MAX;             // All ones
    Val >>= 64-TypeBits;                 // Shift out unwanted 1 bits...
    return ConstantSInt::get(Ty, Val);
  }

  case Type::UByteTyID:
  case Type::UShortTyID:
  case Type::UIntTyID:
  case Type::ULongTyID:  return getAllOnesValue(Ty);

  default: return 0;
  }
}

// Static constructor to create the minimum constant for an integral type...
ConstantIntegral *ConstantIntegral::getMinValue(const Type *Ty) {
  switch (Ty->getPrimitiveID()) {
  case Type::BoolTyID:   return ConstantBool::False;
  case Type::SByteTyID:
  case Type::ShortTyID:
  case Type::IntTyID:
  case Type::LongTyID: {
     // Calculate 1111111111000000000000 
     unsigned TypeBits = Ty->getPrimitiveSize()*8;
     int64_t Val = -1;                    // All ones
     Val <<= TypeBits-1;                  // Shift over to the right spot
     return ConstantSInt::get(Ty, Val);
  }

  case Type::UByteTyID:
  case Type::UShortTyID:
  case Type::UIntTyID:
  case Type::ULongTyID:  return ConstantUInt::get(Ty, 0);

  default: return 0;
  }
}

// Static constructor to create an integral constant with all bits set
ConstantIntegral *ConstantIntegral::getAllOnesValue(const Type *Ty) {
  switch (Ty->getPrimitiveID()) {
  case Type::BoolTyID:   return ConstantBool::True;
  case Type::SByteTyID:
  case Type::ShortTyID:
  case Type::IntTyID:
  case Type::LongTyID:   return ConstantSInt::get(Ty, -1);

  case Type::UByteTyID:
  case Type::UShortTyID:
  case Type::UIntTyID:
  case Type::ULongTyID: {
    // Calculate ~0 of the right type...
    unsigned TypeBits = Ty->getPrimitiveSize()*8;
    uint64_t Val = ~0ULL;                // All ones
    Val >>= 64-TypeBits;                 // Shift out unwanted 1 bits...
    return ConstantUInt::get(Ty, Val);
  }
  default: return 0;
  }
}

bool ConstantUInt::isAllOnesValue() const {
  unsigned TypeBits = getType()->getPrimitiveSize()*8;
  uint64_t Val = ~0ULL;                // All ones
  Val >>= 64-TypeBits;                 // Shift out inappropriate bits
  return getValue() == Val;
}


//===----------------------------------------------------------------------===//
//                            ConstantXXX Classes
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//                             Normal Constructors

ConstantBool::ConstantBool(bool V) : ConstantIntegral(Type::BoolTy) {
  Val = V;
}

ConstantInt::ConstantInt(const Type *Ty, uint64_t V) : ConstantIntegral(Ty) {
  Val.Unsigned = V;
}

ConstantSInt::ConstantSInt(const Type *Ty, int64_t V) : ConstantInt(Ty, V) {
  assert(Ty->isInteger() && Ty->isSigned() &&
         "Illegal type for unsigned integer constant!");
  assert(isValueValidForType(Ty, V) && "Value too large for type!");
}

ConstantUInt::ConstantUInt(const Type *Ty, uint64_t V) : ConstantInt(Ty, V) {
  assert(Ty->isInteger() && Ty->isUnsigned() &&
         "Illegal type for unsigned integer constant!");
  assert(isValueValidForType(Ty, V) && "Value too large for type!");
}

ConstantFP::ConstantFP(const Type *Ty, double V) : Constant(Ty) {
  assert(isValueValidForType(Ty, V) && "Value too large for type!");
  Val = V;
}

ConstantArray::ConstantArray(const ArrayType *T,
                             const std::vector<Constant*> &V) : Constant(T) {
  Operands.reserve(V.size());
  for (unsigned i = 0, e = V.size(); i != e; ++i) {
    assert(V[i]->getType() == T->getElementType() ||
           (T->isAbstract() &&
            V[i]->getType()->getPrimitiveID() ==
            T->getElementType()->getPrimitiveID()));
    Operands.push_back(Use(V[i], this));
  }
}

ConstantStruct::ConstantStruct(const StructType *T,
                               const std::vector<Constant*> &V) : Constant(T) {
  assert(V.size() == T->getNumElements() &&
         "Invalid initializer vector for constant structure");
  Operands.reserve(V.size());
  for (unsigned i = 0, e = V.size(); i != e; ++i) {
    assert((V[i]->getType() == T->getElementType(i) ||
            ((T->getElementType(i)->isAbstract() ||
              V[i]->getType()->isAbstract()) &&
             T->getElementType(i)->getPrimitiveID() == 
                      V[i]->getType()->getPrimitiveID())) &&
           "Initializer for struct element doesn't match struct element type!");
    Operands.push_back(Use(V[i], this));
  }
}

ConstantPointerRef::ConstantPointerRef(GlobalValue *GV)
  : Constant(GV->getType()) {
  Operands.push_back(Use(GV, this));
}

ConstantExpr::ConstantExpr(unsigned Opcode, Constant *C, const Type *Ty)
  : Constant(Ty), iType(Opcode) {
  Operands.push_back(Use(C, this));
}

static bool isSetCC(unsigned Opcode) {
  return Opcode == Instruction::SetEQ || Opcode == Instruction::SetNE ||
         Opcode == Instruction::SetLT || Opcode == Instruction::SetGT ||
         Opcode == Instruction::SetLE || Opcode == Instruction::SetGE;
}

ConstantExpr::ConstantExpr(unsigned Opcode, Constant *C1, Constant *C2)
  : Constant(isSetCC(Opcode) ? Type::BoolTy : C1->getType()), iType(Opcode) {
  Operands.push_back(Use(C1, this));
  Operands.push_back(Use(C2, this));
}

ConstantExpr::ConstantExpr(Constant *C, const std::vector<Constant*> &IdxList,
                           const Type *DestTy)
  : Constant(DestTy), iType(Instruction::GetElementPtr) {
  Operands.reserve(1+IdxList.size());
  Operands.push_back(Use(C, this));
  for (unsigned i = 0, E = IdxList.size(); i != E; ++i)
    Operands.push_back(Use(IdxList[i], this));
}



//===----------------------------------------------------------------------===//
//                           classof implementations

bool ConstantIntegral::classof(const Constant *CPV) {
  return CPV->getType()->isIntegral() && !isa<ConstantExpr>(CPV);
}

bool ConstantInt::classof(const Constant *CPV) {
  return CPV->getType()->isInteger() && !isa<ConstantExpr>(CPV);
}
bool ConstantSInt::classof(const Constant *CPV) {
  return CPV->getType()->isSigned() && !isa<ConstantExpr>(CPV);
}
bool ConstantUInt::classof(const Constant *CPV) {
  return CPV->getType()->isUnsigned() && !isa<ConstantExpr>(CPV);
}
bool ConstantFP::classof(const Constant *CPV) {
  const Type *Ty = CPV->getType();
  return ((Ty == Type::FloatTy || Ty == Type::DoubleTy) &&
          !isa<ConstantExpr>(CPV));
}
bool ConstantArray::classof(const Constant *CPV) {
  return isa<ArrayType>(CPV->getType()) && !isa<ConstantExpr>(CPV);
}
bool ConstantStruct::classof(const Constant *CPV) {
  return isa<StructType>(CPV->getType()) && !isa<ConstantExpr>(CPV);
}

bool ConstantPointerNull::classof(const Constant *CPV) {
  return isa<PointerType>(CPV->getType()) && !isa<ConstantExpr>(CPV) &&
         CPV->getNumOperands() == 0;
}

bool ConstantPointerRef::classof(const Constant *CPV) {
  return isa<PointerType>(CPV->getType()) && !isa<ConstantExpr>(CPV) &&
         CPV->getNumOperands() == 1;
}



//===----------------------------------------------------------------------===//
//                      isValueValidForType implementations

bool ConstantSInt::isValueValidForType(const Type *Ty, int64_t Val) {
  switch (Ty->getPrimitiveID()) {
  default:
    return false;         // These can't be represented as integers!!!

    // Signed types...
  case Type::SByteTyID:
    return (Val <= INT8_MAX && Val >= INT8_MIN);
  case Type::ShortTyID:
    return (Val <= INT16_MAX && Val >= INT16_MIN);
  case Type::IntTyID:
    return (Val <= INT32_MAX && Val >= INT32_MIN);
  case Type::LongTyID:
    return true;          // This is the largest type...
  }
  assert(0 && "WTF?");
  return false;
}

bool ConstantUInt::isValueValidForType(const Type *Ty, uint64_t Val) {
  switch (Ty->getPrimitiveID()) {
  default:
    return false;         // These can't be represented as integers!!!

    // Unsigned types...
  case Type::UByteTyID:
    return (Val <= UINT8_MAX);
  case Type::UShortTyID:
    return (Val <= UINT16_MAX);
  case Type::UIntTyID:
    return (Val <= UINT32_MAX);
  case Type::ULongTyID:
    return true;          // This is the largest type...
  }
  assert(0 && "WTF?");
  return false;
}

bool ConstantFP::isValueValidForType(const Type *Ty, double Val) {
  switch (Ty->getPrimitiveID()) {
  default:
    return false;         // These can't be represented as floating point!

    // TODO: Figure out how to test if a double can be cast to a float!
  case Type::FloatTyID:
  case Type::DoubleTyID:
    return true;          // This is the largest type...
  }
};

//===----------------------------------------------------------------------===//
//                replaceUsesOfWithOnConstant implementations

void ConstantArray::replaceUsesOfWithOnConstant(Value *From, Value *To,
                                                bool DisableChecking) {
  assert(isa<Constant>(To) && "Cannot make Constant refer to non-constant!");

  std::vector<Constant*> Values;
  Values.reserve(getValues().size());  // Build replacement array...
  for (unsigned i = 0, e = getValues().size(); i != e; ++i) {
    Constant *Val = cast<Constant>(getValues()[i]);
    if (Val == From) Val = cast<Constant>(To);
    Values.push_back(Val);
  }
  
  Constant *Replacement = ConstantArray::get(getType(), Values);
  assert(Replacement != this && "I didn't contain From!");

  // Everyone using this now uses the replacement...
  if (DisableChecking)
    uncheckedReplaceAllUsesWith(Replacement);
  else
    replaceAllUsesWith(Replacement);
  
  // Delete the old constant!
  destroyConstant();  
}

void ConstantStruct::replaceUsesOfWithOnConstant(Value *From, Value *To,
                                                 bool DisableChecking) {
  assert(isa<Constant>(To) && "Cannot make Constant refer to non-constant!");

  std::vector<Constant*> Values;
  Values.reserve(getValues().size());
  for (unsigned i = 0, e = getValues().size(); i != e; ++i) {
    Constant *Val = cast<Constant>(getValues()[i]);
    if (Val == From) Val = cast<Constant>(To);
    Values.push_back(Val);
  }
  
  Constant *Replacement = ConstantStruct::get(getType(), Values);
  assert(Replacement != this && "I didn't contain From!");

  // Everyone using this now uses the replacement...
  if (DisableChecking)
    uncheckedReplaceAllUsesWith(Replacement);
  else
    replaceAllUsesWith(Replacement);
  
  // Delete the old constant!
  destroyConstant();
}

void ConstantPointerRef::replaceUsesOfWithOnConstant(Value *From, Value *To,
                                                     bool DisableChecking) {
  if (isa<GlobalValue>(To)) {
    assert(From == getOperand(0) && "Doesn't contain from!");
    ConstantPointerRef *Replacement =
      ConstantPointerRef::get(cast<GlobalValue>(To));
    
    // Everyone using this now uses the replacement...
    if (DisableChecking)
      uncheckedReplaceAllUsesWith(Replacement);
    else
      replaceAllUsesWith(Replacement);
    
  } else {
    // Just replace ourselves with the To value specified.
    if (DisableChecking)
      uncheckedReplaceAllUsesWith(To);
    else
      replaceAllUsesWith(To);
  }

  // Delete the old constant!
  destroyConstant();
}

void ConstantExpr::replaceUsesOfWithOnConstant(Value *From, Value *ToV,
                                               bool DisableChecking) {
  assert(isa<Constant>(ToV) && "Cannot make Constant refer to non-constant!");
  Constant *To = cast<Constant>(ToV);

  Constant *Replacement = 0;
  if (getOpcode() == Instruction::GetElementPtr) {
    std::vector<Constant*> Indices;
    Constant *Pointer = getOperand(0);
    Indices.reserve(getNumOperands()-1);
    if (Pointer == From) Pointer = To;
    
    for (unsigned i = 1, e = getNumOperands(); i != e; ++i) {
      Constant *Val = getOperand(i);
      if (Val == From) Val = To;
      Indices.push_back(Val);
    }
    Replacement = ConstantExpr::getGetElementPtr(Pointer, Indices);
  } else if (getOpcode() == Instruction::Cast) {
    assert(getOperand(0) == From && "Cast only has one use!");
    Replacement = ConstantExpr::getCast(To, getType());
  } else if (getNumOperands() == 2) {
    Constant *C1 = getOperand(0);
    Constant *C2 = getOperand(1);
    if (C1 == From) C1 = To;
    if (C2 == From) C2 = To;
    Replacement = ConstantExpr::get(getOpcode(), C1, C2);
  } else {
    assert(0 && "Unknown ConstantExpr type!");
    return;
  }
  
  assert(Replacement != this && "I didn't contain From!");

  // Everyone using this now uses the replacement...
  if (DisableChecking)
    uncheckedReplaceAllUsesWith(Replacement);
  else
    replaceAllUsesWith(Replacement);
  
  // Delete the old constant!
  destroyConstant();
}

//===----------------------------------------------------------------------===//
//                      Factory Function Implementation

// ConstantCreator - A class that is used to create constants by
// ValueMap*.  This class should be partially specialized if there is
// something strange that needs to be done to interface to the ctor for the
// constant.
//
namespace llvm {
  template<class ConstantClass, class TypeClass, class ValType>
  struct ConstantCreator {
    static ConstantClass *create(const TypeClass *Ty, const ValType &V) {
      return new ConstantClass(Ty, V);
    }
  };
  
  template<class ConstantClass, class TypeClass>
  struct ConvertConstantType {
    static void convert(ConstantClass *OldC, const TypeClass *NewTy) {
      assert(0 && "This type cannot be converted!\n");
      abort();
    }
  };
}

namespace {
  template<class ValType, class TypeClass, class ConstantClass>
  class ValueMap : public AbstractTypeUser {
    typedef std::pair<const TypeClass*, ValType> MapKey;
    typedef std::map<MapKey, ConstantClass *> MapTy;
    typedef typename MapTy::iterator MapIterator;
    MapTy Map;

    typedef std::map<const TypeClass*, MapIterator> AbstractTypeMapTy;
    AbstractTypeMapTy AbstractTypeMap;
  public:
    // getOrCreate - Return the specified constant from the map, creating it if
    // necessary.
    ConstantClass *getOrCreate(const TypeClass *Ty, const ValType &V) {
      MapKey Lookup(Ty, V);
      MapIterator I = Map.lower_bound(Lookup);
      if (I != Map.end() && I->first == Lookup)
        return I->second;  // Is it in the map?

      // If no preexisting value, create one now...
      ConstantClass *Result =
        ConstantCreator<ConstantClass,TypeClass,ValType>::create(Ty, V);


      /// FIXME: why does this assert fail when loading 176.gcc?
      //assert(Result->getType() == Ty && "Type specified is not correct!");
      I = Map.insert(I, std::make_pair(MapKey(Ty, V), Result));

      // If the type of the constant is abstract, make sure that an entry exists
      // for it in the AbstractTypeMap.
      if (Ty->isAbstract()) {
        typename AbstractTypeMapTy::iterator TI =
          AbstractTypeMap.lower_bound(Ty);

        if (TI == AbstractTypeMap.end() || TI->first != Ty) {
          // Add ourselves to the ATU list of the type.
          cast<DerivedType>(Ty)->addAbstractTypeUser(this);

          AbstractTypeMap.insert(TI, std::make_pair(Ty, I));
        }
      }
      return Result;
    }
    
    void remove(ConstantClass *CP) {
      // FIXME: This should not use a linear scan.  If this gets to be a
      // performance problem, someone should look at this.
      MapIterator I = Map.begin();
      for (MapIterator E = Map.end(); I != E && I->second != CP; ++I)
        /* empty */;
      
      assert(I != Map.end() && "Constant not found in constant table!");

      // Now that we found the entry, make sure this isn't the entry that
      // the AbstractTypeMap points to.
      const TypeClass *Ty = I->first.first;
      if (Ty->isAbstract()) {
        assert(AbstractTypeMap.count(Ty) &&
               "Abstract type not in AbstractTypeMap?");
        MapIterator &ATMEntryIt = AbstractTypeMap[Ty];
        if (ATMEntryIt == I) {
          // Yes, we are removing the representative entry for this type.
          // See if there are any other entries of the same type.
          MapIterator TmpIt = ATMEntryIt;
          
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
      
      Map.erase(I);
    }

    void refineAbstractType(const DerivedType *OldTy, const Type *NewTy) {
      typename AbstractTypeMapTy::iterator I = 
        AbstractTypeMap.find(cast<TypeClass>(OldTy));

      assert(I != AbstractTypeMap.end() &&
             "Abstract type not in AbstractTypeMap?");

      // Convert a constant at a time until the last one is gone.  The last one
      // leaving will remove() itself, causing the AbstractTypeMapEntry to be
      // eliminated eventually.
      do {
        ConvertConstantType<ConstantClass,
                            TypeClass>::convert(I->second->second,
                                                cast<TypeClass>(NewTy));

        I = AbstractTypeMap.find(cast<TypeClass>(OldTy));
      } while (I != AbstractTypeMap.end());
    }

    // If the type became concrete without being refined to any other existing
    // type, we just remove ourselves from the ATU list.
    void typeBecameConcrete(const DerivedType *AbsTy) {
      AbsTy->removeAbstractTypeUser(this);
    }

    void dump() const {
      std::cerr << "Constant.cpp: ValueMap\n";
    }
  };
}



//---- ConstantUInt::get() and ConstantSInt::get() implementations...
//
static ValueMap< int64_t, Type, ConstantSInt> SIntConstants;
static ValueMap<uint64_t, Type, ConstantUInt> UIntConstants;

ConstantSInt *ConstantSInt::get(const Type *Ty, int64_t V) {
  return SIntConstants.getOrCreate(Ty, V);
}

ConstantUInt *ConstantUInt::get(const Type *Ty, uint64_t V) {
  return UIntConstants.getOrCreate(Ty, V);
}

ConstantInt *ConstantInt::get(const Type *Ty, unsigned char V) {
  assert(V <= 127 && "Can only be used with very small positive constants!");
  if (Ty->isSigned()) return ConstantSInt::get(Ty, V);
  return ConstantUInt::get(Ty, V);
}

//---- ConstantFP::get() implementation...
//
namespace llvm {
  template<>
  struct ConstantCreator<ConstantFP, Type, uint64_t> {
    static ConstantFP *create(const Type *Ty, uint64_t V) {
      assert(Ty == Type::DoubleTy);
      union {
        double F;
        uint64_t I;
      } T;
      T.I = V;
      return new ConstantFP(Ty, T.F);
    }
  };
  template<>
  struct ConstantCreator<ConstantFP, Type, uint32_t> {
    static ConstantFP *create(const Type *Ty, uint32_t V) {
      assert(Ty == Type::FloatTy);
      union {
        float F;
        uint32_t I;
      } T;
      T.I = V;
      return new ConstantFP(Ty, T.F);
    }
  };
}

static ValueMap<uint64_t, Type, ConstantFP> DoubleConstants;
static ValueMap<uint32_t, Type, ConstantFP> FloatConstants;

ConstantFP *ConstantFP::get(const Type *Ty, double V) {
  if (Ty == Type::FloatTy) {
    // Force the value through memory to normalize it.
    union {
      float F;
      uint32_t I;
    } T;
    T.F = (float)V;
    return FloatConstants.getOrCreate(Ty, T.I);
  } else {
    assert(Ty == Type::DoubleTy);
    union {
      double F;
      uint64_t I;
    } T;
    T.F = V;
    return DoubleConstants.getOrCreate(Ty, T.I);
  }
}

//---- ConstantArray::get() implementation...
//
namespace llvm {
  template<>
  struct ConvertConstantType<ConstantArray, ArrayType> {
    static void convert(ConstantArray *OldC, const ArrayType *NewTy) {
      // Make everyone now use a constant of the new type...
      std::vector<Constant*> C;
      for (unsigned i = 0, e = OldC->getNumOperands(); i != e; ++i)
        C.push_back(cast<Constant>(OldC->getOperand(i)));
      Constant *New = ConstantArray::get(NewTy, C);
      assert(New != OldC && "Didn't replace constant??");
      OldC->uncheckedReplaceAllUsesWith(New);
      OldC->destroyConstant();    // This constant is now dead, destroy it.
    }
  };
}

static ValueMap<std::vector<Constant*>, ArrayType,
                ConstantArray> ArrayConstants;

Constant *ConstantArray::get(const ArrayType *Ty,
                                  const std::vector<Constant*> &V) {
  return ArrayConstants.getOrCreate(Ty, V);
}

// destroyConstant - Remove the constant from the constant table...
//
void ConstantArray::destroyConstant() {
  ArrayConstants.remove(this);
  destroyConstantImpl();
}

// ConstantArray::get(const string&) - Return an array that is initialized to
// contain the specified string.  A null terminator is added to the specified
// string so that it may be used in a natural way...
//
Constant *ConstantArray::get(const std::string &Str) {
  std::vector<Constant*> ElementVals;

  for (unsigned i = 0; i < Str.length(); ++i)
    ElementVals.push_back(ConstantSInt::get(Type::SByteTy, Str[i]));

  // Add a null terminator to the string...
  ElementVals.push_back(ConstantSInt::get(Type::SByteTy, 0));

  ArrayType *ATy = ArrayType::get(Type::SByteTy, Str.length()+1);
  return ConstantArray::get(ATy, ElementVals);
}

/// isString - This method returns true if the array is an array of sbyte or
/// ubyte, and if the elements of the array are all ConstantInt's.
bool ConstantArray::isString() const {
  // Check the element type for sbyte or ubyte...
  if (getType()->getElementType() != Type::UByteTy &&
      getType()->getElementType() != Type::SByteTy)
    return false;
  // Check the elements to make sure they are all integers, not constant
  // expressions.
  for (unsigned i = 0, e = getNumOperands(); i != e; ++i)
    if (!isa<ConstantInt>(getOperand(i)))
      return false;
  return true;
}

// getAsString - If the sub-element type of this array is either sbyte or ubyte,
// then this method converts the array to an std::string and returns it.
// Otherwise, it asserts out.
//
std::string ConstantArray::getAsString() const {
  assert(isString() && "Not a string!");
  std::string Result;
  for (unsigned i = 0, e = getNumOperands(); i != e; ++i)
    Result += (char)cast<ConstantInt>(getOperand(i))->getRawValue();
  return Result;
}


//---- ConstantStruct::get() implementation...
//

namespace llvm {
  template<>
  struct ConvertConstantType<ConstantStruct, StructType> {
    static void convert(ConstantStruct *OldC, const StructType *NewTy) {
      // Make everyone now use a constant of the new type...
      std::vector<Constant*> C;
      for (unsigned i = 0, e = OldC->getNumOperands(); i != e; ++i)
        C.push_back(cast<Constant>(OldC->getOperand(i)));
      Constant *New = ConstantStruct::get(NewTy, C);
      assert(New != OldC && "Didn't replace constant??");
      
      OldC->uncheckedReplaceAllUsesWith(New);
      OldC->destroyConstant();    // This constant is now dead, destroy it.
    }
  };
}

static ValueMap<std::vector<Constant*>, StructType, 
                ConstantStruct> StructConstants;

Constant *ConstantStruct::get(const StructType *Ty,
                              const std::vector<Constant*> &V) {
  return StructConstants.getOrCreate(Ty, V);
}

// destroyConstant - Remove the constant from the constant table...
//
void ConstantStruct::destroyConstant() {
  StructConstants.remove(this);
  destroyConstantImpl();
}

//---- ConstantPointerNull::get() implementation...
//

namespace llvm {
  // ConstantPointerNull does not take extra "value" argument...
  template<class ValType>
  struct ConstantCreator<ConstantPointerNull, PointerType, ValType> {
    static ConstantPointerNull *create(const PointerType *Ty, const ValType &V){
      return new ConstantPointerNull(Ty);
    }
  };

  template<>
  struct ConvertConstantType<ConstantPointerNull, PointerType> {
    static void convert(ConstantPointerNull *OldC, const PointerType *NewTy) {
      // Make everyone now use a constant of the new type...
      Constant *New = ConstantPointerNull::get(NewTy);
      assert(New != OldC && "Didn't replace constant??");
      OldC->uncheckedReplaceAllUsesWith(New);
      OldC->destroyConstant();     // This constant is now dead, destroy it.
    }
  };
}

static ValueMap<char, PointerType, ConstantPointerNull> NullPtrConstants;

ConstantPointerNull *ConstantPointerNull::get(const PointerType *Ty) {
  return NullPtrConstants.getOrCreate(Ty, 0);
}

// destroyConstant - Remove the constant from the constant table...
//
void ConstantPointerNull::destroyConstant() {
  NullPtrConstants.remove(this);
  destroyConstantImpl();
}


//---- ConstantPointerRef::get() implementation...
//
ConstantPointerRef *ConstantPointerRef::get(GlobalValue *GV) {
  assert(GV->getParent() && "Global Value must be attached to a module!");
  
  // The Module handles the pointer reference sharing...
  return GV->getParent()->getConstantPointerRef(GV);
}

// destroyConstant - Remove the constant from the constant table...
//
void ConstantPointerRef::destroyConstant() {
  getValue()->getParent()->destroyConstantPointerRef(this);
  destroyConstantImpl();
}


//---- ConstantExpr::get() implementations...
//
typedef std::pair<unsigned, std::vector<Constant*> > ExprMapKeyType;

namespace llvm {
  template<>
  struct ConstantCreator<ConstantExpr, Type, ExprMapKeyType> {
    static ConstantExpr *create(const Type *Ty, const ExprMapKeyType &V) {
      if (V.first == Instruction::Cast)
        return new ConstantExpr(Instruction::Cast, V.second[0], Ty);
      if ((V.first >= Instruction::BinaryOpsBegin &&
           V.first < Instruction::BinaryOpsEnd) ||
          V.first == Instruction::Shl || V.first == Instruction::Shr)
        return new ConstantExpr(V.first, V.second[0], V.second[1]);
      
      assert(V.first == Instruction::GetElementPtr && "Invalid ConstantExpr!");
      
      std::vector<Constant*> IdxList(V.second.begin()+1, V.second.end());
      return new ConstantExpr(V.second[0], IdxList, Ty);
    }
  };

  template<>
  struct ConvertConstantType<ConstantExpr, Type> {
    static void convert(ConstantExpr *OldC, const Type *NewTy) {
      Constant *New;
      switch (OldC->getOpcode()) {
      case Instruction::Cast:
        New = ConstantExpr::getCast(OldC->getOperand(0), NewTy);
        break;
      case Instruction::Shl:
      case Instruction::Shr:
        New = ConstantExpr::getShiftTy(NewTy, OldC->getOpcode(),
                                     OldC->getOperand(0), OldC->getOperand(1));
        break;
      default:
        assert(OldC->getOpcode() >= Instruction::BinaryOpsBegin &&
               OldC->getOpcode() < Instruction::BinaryOpsEnd);
        New = ConstantExpr::getTy(NewTy, OldC->getOpcode(), OldC->getOperand(0),
                                  OldC->getOperand(1));
        break;
      case Instruction::GetElementPtr:
        // Make everyone now use a constant of the new type... 
        std::vector<Constant*> C;
        for (unsigned i = 1, e = OldC->getNumOperands(); i != e; ++i)
          C.push_back(cast<Constant>(OldC->getOperand(i)));
        New = ConstantExpr::getGetElementPtrTy(NewTy, OldC->getOperand(0), C);
        break;
      }
      
      assert(New != OldC && "Didn't replace constant??");
      OldC->uncheckedReplaceAllUsesWith(New);
      OldC->destroyConstant();    // This constant is now dead, destroy it.
    }
  };
} // end namespace llvm


static ValueMap<ExprMapKeyType, Type, ConstantExpr> ExprConstants;

Constant *ConstantExpr::getCast(Constant *C, const Type *Ty) {
  assert(Ty->isFirstClassType() && "Cannot cast to an aggregate type!");

  if (Constant *FC = ConstantFoldCastInstruction(C, Ty))
    return FC;          // Fold a few common cases...

  // Look up the constant in the table first to ensure uniqueness
  std::vector<Constant*> argVec(1, C);
  ExprMapKeyType Key = std::make_pair(Instruction::Cast, argVec);
  return ExprConstants.getOrCreate(Ty, Key);
}

Constant *ConstantExpr::getTy(const Type *ReqTy, unsigned Opcode,
                              Constant *C1, Constant *C2) {
  if (Opcode == Instruction::Shl || Opcode == Instruction::Shr)
    return getShiftTy(ReqTy, Opcode, C1, C2);
  // Check the operands for consistency first
  assert((Opcode >= Instruction::BinaryOpsBegin &&
          Opcode < Instruction::BinaryOpsEnd) &&
         "Invalid opcode in binary constant expression");
  assert(C1->getType() == C2->getType() &&
         "Operand types in binary constant expression should match");

  if (ReqTy == C1->getType())
    if (Constant *FC = ConstantFoldBinaryInstruction(Opcode, C1, C2))
      return FC;          // Fold a few common cases...

  std::vector<Constant*> argVec(1, C1); argVec.push_back(C2);
  ExprMapKeyType Key = std::make_pair(Opcode, argVec);
  return ExprConstants.getOrCreate(ReqTy, Key);
}

/// getShiftTy - Return a shift left or shift right constant expr
Constant *ConstantExpr::getShiftTy(const Type *ReqTy, unsigned Opcode,
                                   Constant *C1, Constant *C2) {
  // Check the operands for consistency first
  assert((Opcode == Instruction::Shl ||
          Opcode == Instruction::Shr) &&
         "Invalid opcode in binary constant expression");
  assert(C1->getType()->isIntegral() && C2->getType() == Type::UByteTy &&
         "Invalid operand types for Shift constant expr!");

  if (Constant *FC = ConstantFoldBinaryInstruction(Opcode, C1, C2))
    return FC;          // Fold a few common cases...

  // Look up the constant in the table first to ensure uniqueness
  std::vector<Constant*> argVec(1, C1); argVec.push_back(C2);
  ExprMapKeyType Key = std::make_pair(Opcode, argVec);
  return ExprConstants.getOrCreate(ReqTy, Key);
}


Constant *ConstantExpr::getGetElementPtrTy(const Type *ReqTy, Constant *C,
                                        const std::vector<Constant*> &IdxList) {
  if (Constant *FC = ConstantFoldGetElementPtr(C, IdxList))
    return FC;          // Fold a few common cases...
  assert(isa<PointerType>(C->getType()) &&
         "Non-pointer type for constant GetElementPtr expression");

  // Look up the constant in the table first to ensure uniqueness
  std::vector<Constant*> argVec(1, C);
  argVec.insert(argVec.end(), IdxList.begin(), IdxList.end());
  const ExprMapKeyType &Key = std::make_pair(Instruction::GetElementPtr,argVec);
  return ExprConstants.getOrCreate(ReqTy, Key);
}

Constant *ConstantExpr::getGetElementPtr(Constant *C,
                                         const std::vector<Constant*> &IdxList){
  // Get the result type of the getelementptr!
  std::vector<Value*> VIdxList(IdxList.begin(), IdxList.end());

  const Type *Ty = GetElementPtrInst::getIndexedType(C->getType(), VIdxList,
                                                     true);
  assert(Ty && "GEP indices invalid!");

  if (C->isNullValue()) {
    bool isNull = true;
    for (unsigned i = 0, e = IdxList.size(); i != e; ++i)
      if (!IdxList[i]->isNullValue()) {
        isNull = false;
        break;
      }
    if (isNull) return ConstantPointerNull::get(PointerType::get(Ty));
  }

  return getGetElementPtrTy(PointerType::get(Ty), C, IdxList);
}


// destroyConstant - Remove the constant from the constant table...
//
void ConstantExpr::destroyConstant() {
  ExprConstants.remove(this);
  destroyConstantImpl();
}

const char *ConstantExpr::getOpcodeName() const {
  return Instruction::getOpcodeName(getOpcode());
}

unsigned Constant::mutateReferences(Value *OldV, Value *NewV) {
  // Uses of constant pointer refs are global values, not constants!
  if (ConstantPointerRef *CPR = dyn_cast<ConstantPointerRef>(this)) {
    GlobalValue *NewGV = cast<GlobalValue>(NewV);
    GlobalValue *OldGV = CPR->getValue();

    assert(OldGV == OldV && "Cannot mutate old value if I'm not using it!");
    Operands[0] = NewGV;
    OldGV->getParent()->mutateConstantPointerRef(OldGV, NewGV);
    return 1;
  } else {
    Constant *NewC = cast<Constant>(NewV);
    unsigned NumReplaced = 0;
    for (unsigned i = 0, N = getNumOperands(); i != N; ++i)
      if (Operands[i] == OldV) {
        ++NumReplaced;
        Operands[i] = NewC;
      }
    return NumReplaced;
  }
}

