//===-- iConstPool.cpp - Implement ConstPool instructions --------*- C++ -*--=//
//
// This file implements the ConstPool* classes...
//
//===----------------------------------------------------------------------===//

#define __STDC_LIMIT_MACROS           // Get defs for INT64_MAX and friends...
#include "llvm/ConstPoolVals.h"
#include "llvm/Support/StringExtras.h"  // itostr
#include "llvm/DerivedTypes.h"
#include "llvm/SymbolTable.h"
#include <algorithm>
#include <assert.h>

ConstPoolBool *ConstPoolBool::True  = new ConstPoolBool(true);
ConstPoolBool *ConstPoolBool::False = new ConstPoolBool(false);


//===----------------------------------------------------------------------===//
//                              ConstPoolVal Class
//===----------------------------------------------------------------------===//

// Specialize setName to take care of symbol table majik
void ConstPoolVal::setName(const string &Name, SymbolTable *ST) {
  assert(ST && "Type::setName - Must provide symbol table argument!");

  if (Name.size()) ST->insert(Name, this);
}

// Static constructor to create a '0' constant of arbitrary type...
ConstPoolVal *ConstPoolVal::getNullConstant(const Type *Ty) {
  switch (Ty->getPrimitiveID()) {
  case Type::BoolTyID:   return ConstPoolBool::get(false);
  case Type::SByteTyID:
  case Type::ShortTyID:
  case Type::IntTyID:
  case Type::LongTyID:   return ConstPoolSInt::get(Ty, 0);

  case Type::UByteTyID:
  case Type::UShortTyID:
  case Type::UIntTyID:
  case Type::ULongTyID:  return ConstPoolUInt::get(Ty, 0);

  case Type::FloatTyID:
  case Type::DoubleTyID: return ConstPoolFP::get(Ty, 0);

  case Type::PointerTyID: 
    return ConstPoolPointer::getNullPointer(Ty->castPointerType());
  default:
    return 0;
  }
}

bool ConstPoolInt::isa(const ConstPoolVal *CPV) {
  return CPV->getType()->isIntegral();
}

//===----------------------------------------------------------------------===//
//                            ConstPoolXXX Classes
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//                             Normal Constructors

ConstPoolBool::ConstPoolBool(bool V) : ConstPoolVal(Type::BoolTy) {
  Val = V;
}

ConstPoolInt::ConstPoolInt(const Type *Ty, uint64_t V) : ConstPoolVal(Ty) {
  Val.Unsigned = V;
}

ConstPoolSInt::ConstPoolSInt(const Type *Ty, int64_t V) : ConstPoolInt(Ty, V) {
  assert(isValueValidForType(Ty, V) && "Value too large for type!");
}

ConstPoolUInt::ConstPoolUInt(const Type *Ty, uint64_t V) : ConstPoolInt(Ty, V) {
  assert(isValueValidForType(Ty, V) && "Value too large for type!");
}

ConstPoolFP::ConstPoolFP(const Type *Ty, double V) : ConstPoolVal(Ty) {
  assert(isValueValidForType(Ty, V) && "Value too large for type!");
  Val = V;
}

ConstPoolArray::ConstPoolArray(const ArrayType *T,
			       const vector<ConstPoolVal*> &V)
  : ConstPoolVal(T) {
  for (unsigned i = 0; i < V.size(); i++) {
    assert(V[i]->getType() == T->getElementType());
    Operands.push_back(Use(V[i], this));
  }
}

ConstPoolStruct::ConstPoolStruct(const StructType *T,
				 const vector<ConstPoolVal*> &V)
  : ConstPoolVal(T) {
  const StructType::ElementTypes &ETypes = T->getElementTypes();
  
  for (unsigned i = 0; i < V.size(); i++) {
    assert(V[i]->getType() == ETypes[i]);
    Operands.push_back(Use(V[i], this));
  }
}

ConstPoolPointer::ConstPoolPointer(const PointerType *T) : ConstPoolVal(T) {}


//===----------------------------------------------------------------------===//
//                          getStrValue implementations

string ConstPoolBool::getStrValue() const {
  return Val ? "true" : "false";
}

string ConstPoolSInt::getStrValue() const {
  return itostr(Val.Signed);
}

string ConstPoolUInt::getStrValue() const {
  return utostr(Val.Unsigned);
}

string ConstPoolFP::getStrValue() const {
  return ftostr(Val);
}

string ConstPoolArray::getStrValue() const {
  string Result = "[";
  if (Operands.size()) {
    Result += " " + Operands[0]->getType()->getDescription() + 
	      " " + cast<ConstPoolVal>(Operands[0])->getStrValue();
    for (unsigned i = 1; i < Operands.size(); i++)
      Result += ", " + Operands[i]->getType()->getDescription() + 
	         " " + cast<ConstPoolVal>(Operands[i])->getStrValue();
  }

  return Result + " ]";
}

string ConstPoolStruct::getStrValue() const {
  string Result = "{";
  if (Operands.size()) {
    Result += " " + Operands[0]->getType()->getDescription() + 
	      " " + cast<ConstPoolVal>(Operands[0])->getStrValue();
    for (unsigned i = 1; i < Operands.size(); i++)
      Result += ", " + Operands[i]->getType()->getDescription() + 
	         " " + cast<ConstPoolVal>(Operands[i])->getStrValue();
  }

  return Result + " }";
}

string ConstPoolPointer::getStrValue() const {
  return "null";
}

//===----------------------------------------------------------------------===//
//                      isValueValidForType implementations

bool ConstPoolSInt::isValueValidForType(const Type *Ty, int64_t Val) {
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

bool ConstPoolUInt::isValueValidForType(const Type *Ty, uint64_t Val) {
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

bool ConstPoolFP::isValueValidForType(const Type *Ty, double Val) {
  switch (Ty->getPrimitiveID()) {
  default:
    return false;         // These can't be represented as floating point!

    // TODO: Figure out how to test if a double can be cast to a float!
  case Type::FloatTyID:
    /*
    return (Val <= UINT8_MAX);
    */
  case Type::DoubleTyID:
    return true;          // This is the largest type...
  }
};

//===----------------------------------------------------------------------===//
//                      Hash Function Implementations
#if 0
unsigned ConstPoolSInt::hash(const Type *Ty, int64_t V) {
  return unsigned(Ty->getPrimitiveID() ^ V);
}

unsigned ConstPoolUInt::hash(const Type *Ty, uint64_t V) {
  return unsigned(Ty->getPrimitiveID() ^ V);
}

unsigned ConstPoolFP::hash(const Type *Ty, double V) {
  return Ty->getPrimitiveID() ^ unsigned(V);
}

unsigned ConstPoolArray::hash(const ArrayType *Ty,
			      const vector<ConstPoolVal*> &V) {
  unsigned Result = (Ty->getUniqueID() << 5) ^ (Ty->getUniqueID() * 7);
  for (unsigned i = 0; i < V.size(); ++i)
    Result ^= V[i]->getHash() << (i & 7);
  return Result;
}

unsigned ConstPoolStruct::hash(const StructType *Ty,
			       const vector<ConstPoolVal*> &V) {
  unsigned Result = (Ty->getUniqueID() << 5) ^ (Ty->getUniqueID() * 7);
  for (unsigned i = 0; i < V.size(); ++i)
    Result ^= V[i]->getHash() << (i & 7);
  return Result;
}
#endif

//===----------------------------------------------------------------------===//
//                      Factory Function Implementation

template<class ValType, class ConstPoolClass>
struct ValueMap {
  typedef pair<const Type*, ValType> ConstHashKey;
  map<ConstHashKey, ConstPoolClass *> Map;

  inline ConstPoolClass *get(const Type *Ty, ValType V) {
    map<ConstHashKey,ConstPoolClass *>::iterator I =
      Map.find(ConstHashKey(Ty, V));
    return (I != Map.end()) ? I->second : 0;
  }

  inline void add(const Type *Ty, ValType V, ConstPoolClass *CP) {
    Map.insert(make_pair(ConstHashKey(Ty, V), CP));
  }
};

//---- ConstPoolUInt::get() and ConstPoolSInt::get() implementations...
//
static ValueMap<uint64_t, ConstPoolInt> IntConstants;

ConstPoolSInt *ConstPoolSInt::get(const Type *Ty, int64_t V) {
  ConstPoolSInt *Result = (ConstPoolSInt*)IntConstants.get(Ty, (uint64_t)V);
  if (!Result)   // If no preexisting value, create one now...
    IntConstants.add(Ty, V, Result = new ConstPoolSInt(Ty, V));
  return Result;
}

ConstPoolUInt *ConstPoolUInt::get(const Type *Ty, uint64_t V) {
  ConstPoolUInt *Result = (ConstPoolUInt*)IntConstants.get(Ty, V);
  if (!Result)   // If no preexisting value, create one now...
    IntConstants.add(Ty, V, Result = new ConstPoolUInt(Ty, V));
  return Result;
}

ConstPoolInt *ConstPoolInt::get(const Type *Ty, unsigned char V) {
  assert(V <= 127 && "Can only be used with very small positive constants!");
  if (Ty->isSigned()) return ConstPoolSInt::get(Ty, V);
  return ConstPoolUInt::get(Ty, V);
}

//---- ConstPoolFP::get() implementation...
//
static ValueMap<double, ConstPoolFP> FPConstants;

ConstPoolFP *ConstPoolFP::get(const Type *Ty, double V) {
  ConstPoolFP *Result = FPConstants.get(Ty, V);
  if (!Result)   // If no preexisting value, create one now...
    FPConstants.add(Ty, V, Result = new ConstPoolFP(Ty, V));
  return Result;
}

//---- ConstPoolArray::get() implementation...
//
static ValueMap<vector<ConstPoolVal*>, ConstPoolArray> ArrayConstants;

ConstPoolArray *ConstPoolArray::get(const ArrayType *Ty,
				    const vector<ConstPoolVal*> &V) {
  ConstPoolArray *Result = ArrayConstants.get(Ty, V);
  if (!Result)   // If no preexisting value, create one now...
    ArrayConstants.add(Ty, V, Result = new ConstPoolArray(Ty, V));
  return Result;
}

//---- ConstPoolStruct::get() implementation...
//
static ValueMap<vector<ConstPoolVal*>, ConstPoolStruct> StructConstants;

ConstPoolStruct *ConstPoolStruct::get(const StructType *Ty,
				      const vector<ConstPoolVal*> &V) {
  ConstPoolStruct *Result = StructConstants.get(Ty, V);
  if (!Result)   // If no preexisting value, create one now...
    StructConstants.add(Ty, V, Result = new ConstPoolStruct(Ty, V));
  return Result;
}
