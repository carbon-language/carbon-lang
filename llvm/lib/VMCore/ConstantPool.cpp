//===-- iConstPool.cpp - Implement ConstPool instructions --------*- C++ -*--=//
//
// This file implements the ConstPool* classes...
//
//===----------------------------------------------------------------------===//

#define __STDC_LIMIT_MACROS           // Get defs for INT64_MAX and friends...
#include "llvm/ConstPoolVals.h"
#include "llvm/ConstantPool.h"
#include "llvm/Tools/StringExtras.h"  // itostr
#include "llvm/DerivedTypes.h"
#include "llvm/SymbolTable.h"
#include <algorithm>
#include <assert.h>

//===----------------------------------------------------------------------===//
//                             ConstantPool Class
//===----------------------------------------------------------------------===//

void ConstantPool::setParent(SymTabValue *STV) {
  Parent = STV;
  for (unsigned i = 0; i < Planes.size(); i++)
    Planes[i]->setParent(Parent);  
}

// Constant getPlane - Returns true if the type plane does not exist, otherwise
// updates the pointer to point to the correct plane.
//
bool ConstantPool::getPlane(const Type *T, const PlaneType *&Plane) const {
  unsigned Ty = T->getUniqueID();
  if (Ty >= Planes.size()) return true;
  Plane = Planes[Ty];
  return false;
}

// Constant getPlane - Returns true if the type plane does not exist, otherwise
// updates the pointer to point to the correct plane.
//
bool ConstantPool::getPlane(const Type *T, PlaneType *&Plane) {
  unsigned Ty = T->getUniqueID();
  if (Ty >= Planes.size()) return true;
  Plane = Planes[Ty];
  return false;
}

void ConstantPool::resize(unsigned size) {
  unsigned oldSize = Planes.size();
  Planes.resize(size, 0);
  while (oldSize < size)
    Planes[oldSize++] = new PlaneType(Parent, Parent);
}

ConstantPool::PlaneType &ConstantPool::getPlane(const Type *T) {
  unsigned Ty = T->getUniqueID();
  if (Ty >= Planes.size()) resize(Ty+1);
  return *Planes[Ty];
}

// insert - Add constant into the symbol table...
void ConstantPool::insert(ConstPoolVal *N) {
  unsigned Ty = N->getType()->getUniqueID();
  if (Ty >= Planes.size()) resize(Ty+1);
  Planes[Ty]->push_back(N);
}

bool ConstantPool::remove(ConstPoolVal *N) {
  unsigned Ty = N->getType()->getUniqueID();
  if (Ty >= Planes.size()) return true;     // Doesn't contain any of that type

  PlaneType::iterator I = ::find(Planes[Ty]->begin(), Planes[Ty]->end(), N);
  if (I == Planes[Ty]->end()) return true;
  Planes[Ty]->remove(I);
  return false;
}

void ConstantPool::delete_all() {
  dropAllReferences();
  for (unsigned i = 0; i < Planes.size(); i++) {
    Planes[i]->delete_all();
    Planes[i]->setParent(0);
    delete Planes[i];
  }
  Planes.clear();
}

void ConstantPool::dropAllReferences() {
  for (unsigned i = 0; i < Planes.size(); i++)
    for_each(Planes[i]->begin(), Planes[i]->end(),
	     mem_fun(&ConstPoolVal::dropAllReferences));
}

struct EqualsConstant {
  const ConstPoolVal *v;
  inline EqualsConstant(const ConstPoolVal *V) { v = V; }
  inline bool operator()(const ConstPoolVal *V) const {
    return v->equals(V);
  }
};


ConstPoolVal *ConstantPool::find(const ConstPoolVal *V) {
  const PlaneType *P;
  if (getPlane(V->getType(), P)) return 0;
  PlaneType::const_iterator PI = find_if(P->begin(), P->end(), 
					 EqualsConstant(V));
  if (PI == P->end()) return 0;
  return *PI;
}

const ConstPoolVal *ConstantPool::find(const ConstPoolVal *V) const {
  const PlaneType *P;
  if (getPlane(V->getType(), P)) return 0;
  PlaneType::const_iterator PI = find_if(P->begin(), P->end(), 
					 EqualsConstant(V));
  if (PI == P->end()) return 0;
  return *PI;
}

ConstPoolVal *ConstantPool::find(const Type *Ty) {
  const PlaneType *P;
  if (getPlane(Type::TypeTy, P)) return 0;

  // TODO: This is kinda silly
  ConstPoolType V(Ty);

  PlaneType::const_iterator PI = 
    find_if(P->begin(), P->end(), EqualsConstant(&V));
  if (PI == P->end()) return 0;
  return *PI;
}

const ConstPoolVal *ConstantPool::find(const Type *Ty) const {
  const PlaneType *P;
  if (getPlane(Type::TypeTy, P)) return 0;

  // TODO: This is kinda silly
  ConstPoolType V(Ty);

  PlaneType::const_iterator PI = 
    find_if(P->begin(), P->end(), EqualsConstant(&V));
  if (PI == P->end()) return 0;
  return *PI;
}

//===----------------------------------------------------------------------===//
//                              ConstPoolVal Class
//===----------------------------------------------------------------------===//

// Specialize setName to take care of symbol table majik
void ConstPoolVal::setName(const string &name) {
  SymTabValue *P;
  if ((P = getParent()) && hasName()) P->getSymbolTable()->remove(this);
  Value::setName(name);
  if (P && hasName()) P->getSymbolTable()->insert(this);
}

// Static constructor to create a '0' constant of arbitrary type...
ConstPoolVal *ConstPoolVal::getNullConstant(const Type *Ty) {
  switch (Ty->getPrimitiveID()) {
  case Type::BoolTyID:   return new ConstPoolBool(false);
  case Type::SByteTyID:
  case Type::ShortTyID:
  case Type::IntTyID:
  case Type::LongTyID:   return new ConstPoolSInt(Ty, 0);

  case Type::UByteTyID:
  case Type::UShortTyID:
  case Type::UIntTyID:
  case Type::ULongTyID:  return new ConstPoolUInt(Ty, 0);

  case Type::FloatTyID:
  case Type::DoubleTyID: return new ConstPoolFP(Ty, 0);
  default:
    return 0;
  }
}



//===----------------------------------------------------------------------===//
//                            ConstPoolXXX Classes
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//                             Normal Constructors

ConstPoolBool::ConstPoolBool(bool V, const string &Name = "") 
  : ConstPoolVal(Type::BoolTy, Name) {
  Val = V;
}

ConstPoolSInt::ConstPoolSInt(const Type *Ty, int64_t V, const string &Name)
  : ConstPoolVal(Ty, Name) {
  //cerr << "value = " << (int)V << ": " << Ty->getName() << endl;
  assert(isValueValidForType(Ty, V) && "Value to large for type!");
  Val = V;
}

ConstPoolUInt::ConstPoolUInt(const Type *Ty, uint64_t V, const string &Name)
  : ConstPoolVal(Ty, Name) {
  //cerr << "Uvalue = " << (int)V << ": " << Ty->getName() << endl;
  assert(isValueValidForType(Ty, V) && "Value to large for type!");
  Val = V;
}

ConstPoolFP::ConstPoolFP(const Type *Ty, double V, const string &Name)
  : ConstPoolVal(Ty, Name) {
  assert(isValueValidForType(Ty, V) && "Value to large for type!");
  Val = V;
}

ConstPoolType::ConstPoolType(const Type *V, const string &Name) 
  : ConstPoolVal(Type::TypeTy, Name), Val(V) {
}

ConstPoolArray::ConstPoolArray(const ArrayType *T, 
			       vector<ConstPoolVal*> &V, 
			       const string &Name)
  : ConstPoolVal(T, Name) {
  for (unsigned i = 0; i < V.size(); i++) {
    assert(V[i]->getType() == T->getElementType());
    Val.push_back(ConstPoolUse(V[i], this));
  }
}

ConstPoolStruct::ConstPoolStruct(const StructType *T, 
				 vector<ConstPoolVal*> &V, 
				 const string &Name)
  : ConstPoolVal(T, Name) {
  const StructType::ElementTypes &ETypes = T->getElementTypes();

  for (unsigned i = 0; i < V.size(); i++) {
    assert(V[i]->getType() == ETypes[i]);
    Val.push_back(ConstPoolUse(V[i], this));
  }
}


//===----------------------------------------------------------------------===//
//                               Copy Constructors

ConstPoolBool::ConstPoolBool(const ConstPoolBool &CPB)
  : ConstPoolVal(Type::BoolTy) {
  Val = CPB.Val;
}

ConstPoolSInt::ConstPoolSInt(const ConstPoolSInt &CPSI)
  : ConstPoolVal(CPSI.getType()) {
  Val = CPSI.Val;
}

ConstPoolUInt::ConstPoolUInt(const ConstPoolUInt &CPUI)
  : ConstPoolVal(CPUI.getType()) {
  Val = CPUI.Val;
}

ConstPoolFP::ConstPoolFP(const ConstPoolFP &CPFP)
  : ConstPoolVal(CPFP.getType()) {
  Val = CPFP.Val;
}

ConstPoolType::ConstPoolType(const ConstPoolType &CPT) 
  : ConstPoolVal(Type::TypeTy), Val(CPT.Val) {
}

ConstPoolArray::ConstPoolArray(const ConstPoolArray &CPA)
  : ConstPoolVal(CPA.getType()) {
  for (unsigned i = 0; i < CPA.Val.size(); i++)
    Val.push_back(ConstPoolUse((ConstPoolVal*)CPA.Val[i], this));
}

ConstPoolStruct::ConstPoolStruct(const ConstPoolStruct &CPS)
  : ConstPoolVal(CPS.getType()) {
  for (unsigned i = 0; i < CPS.Val.size(); i++)
    Val.push_back(ConstPoolUse((ConstPoolVal*)CPS.Val[i], this));
}

//===----------------------------------------------------------------------===//
//                          getStrValue implementations

string ConstPoolBool::getStrValue() const {
  return Val ? "true" : "false";
}

string ConstPoolSInt::getStrValue() const {
  return itostr(Val);
}

string ConstPoolUInt::getStrValue() const {
  return utostr(Val);
}

string ConstPoolFP::getStrValue() const {
  assert(0 && "FP Constants Not implemented yet!!!!!!!!!!!");
  return "% FP Constants NI!" /* + dtostr(Val)*/;
}

string ConstPoolType::getStrValue() const {
  return Val->getName();
}

string ConstPoolArray::getStrValue() const {
  string Result = "[";
  if (Val.size()) {
    Result += " " + Val[0]->getType()->getName() + 
	      " " + Val[0]->getStrValue();
    for (unsigned i = 1; i < Val.size(); i++)
      Result += ", " + Val[i]->getType()->getName() + 
	         " " + Val[i]->getStrValue();
  }

  return Result + " ]";
}

string ConstPoolStruct::getStrValue() const {
  string Result = "{";
  if (Val.size()) {
    Result += " " + Val[0]->getType()->getName() + 
	      " " + Val[0]->getStrValue();
    for (unsigned i = 1; i < Val.size(); i++)
      Result += ", " + Val[i]->getType()->getName() + 
	         " " + Val[i]->getStrValue();
  }

  return Result + " }";
}

//===----------------------------------------------------------------------===//
//                             equals implementations

bool ConstPoolBool::equals(const ConstPoolVal *V) const {
  assert(getType() == V->getType());
  return ((ConstPoolBool*)V)->getValue() == Val;
}

bool ConstPoolSInt::equals(const ConstPoolVal *V) const {
  assert(getType() == V->getType());
  return ((ConstPoolSInt*)V)->getValue() == Val;
}

bool ConstPoolUInt::equals(const ConstPoolVal *V) const {
  assert(getType() == V->getType());
  return ((ConstPoolUInt*)V)->getValue() == Val;
}

bool ConstPoolFP::equals(const ConstPoolVal *V) const {
  assert(getType() == V->getType());
  return ((ConstPoolFP*)V)->getValue() == Val;
}

bool ConstPoolType::equals(const ConstPoolVal *V) const {
  assert(getType() == V->getType());
  return ((ConstPoolType*)V)->getValue() == Val;
}

bool ConstPoolArray::equals(const ConstPoolVal *V) const {
  assert(getType() == V->getType());
  ConstPoolArray *AV = (ConstPoolArray*)V;
  if (Val.size() != AV->Val.size()) return false;
  for (unsigned i = 0; i < Val.size(); i++)
    if (!Val[i]->equals(AV->Val[i])) return false;

  return true;
}

bool ConstPoolStruct::equals(const ConstPoolVal *V) const {
  assert(getType() == V->getType());
  ConstPoolStruct *SV = (ConstPoolStruct*)V;
  if (Val.size() != SV->Val.size()) return false;
  for (unsigned i = 0; i < Val.size(); i++)
    if (!Val[i]->equals(SV->Val[i])) return false;

  return true;
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
    /*
  case Type::FloatTyID:
    return (Val <= UINT8_MAX);
    */
  case Type::DoubleTyID:
    return true;          // This is the largest type...
  }
};
