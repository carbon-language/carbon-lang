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
#include "llvm/GlobalValue.h"
#include "llvm/Module.h"
#include "llvm/Analysis/SlotCalculator.h"
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
    return ConstPoolPointerNull::get(cast<PointerType>(Ty));
  default:
    return 0;
  }
}

#ifndef NDEBUG
#include "llvm/Assembly/Writer.h"
#endif

void ConstPoolVal::destroyConstantImpl() {
  // When a ConstPoolVal is destroyed, there may be lingering
  // references to the constant by other constants in the constant pool.  These
  // constants are implicitly dependant on the module that is being deleted,
  // but they don't know that.  Because we only find out when the CPV is
  // deleted, we must now notify all of our users (that should only be
  // ConstPoolVals) that they are, in fact, invalid now and should be deleted.
  //
  while (!use_empty()) {
    Value *V = use_back();
#ifndef NDEBUG      // Only in -g mode...
    if (!isa<ConstPoolVal>(V)) {
      cerr << "While deleting: " << this << endl;
      cerr << "Use still stuck around after Def is destroyed: " << V << endl;
    }
#endif
    assert(isa<ConstPoolVal>(V) && "References remain to ConstPoolPointerRef!");
    ConstPoolVal *CPV = cast<ConstPoolVal>(V);
    CPV->destroyConstant();

    // The constant should remove itself from our use list...
    assert((use_empty() || use_back() == V) && "Constant not removed!");
  }

  // Value has no outstanding references it is safe to delete it now...
  delete this;
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

ConstPoolPointerRef::ConstPoolPointerRef(GlobalValue *GV)
  : ConstPoolPointer(GV->getType()) {
  Operands.push_back(Use(GV, this));
}



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
  string Result;
  
  // As a special case, print the array as a string if it is an array of
  // ubytes or an array of sbytes with positive values.
  // 
  const Type *ETy = cast<ArrayType>(getType())->getElementType();
  bool isString = (ETy == Type::SByteTy || ETy == Type::UByteTy);
  for (unsigned i = 0; i < Operands.size(); ++i)
    if (ETy == Type::SByteTy &&
        cast<ConstPoolSInt>(Operands[i])->getValue() < 0) {
      isString = false;
      break;
    }

  if (isString) {
    Result = "c\"";
    for (unsigned i = 0; i < Operands.size(); ++i) {
      unsigned char C = (ETy == Type::SByteTy) ?
        (unsigned char)cast<ConstPoolSInt>(Operands[i])->getValue() :
        (unsigned char)cast<ConstPoolUInt>(Operands[i])->getValue();

      if (isprint(C)) {
        Result += C;
      } else {
        Result += '\\';
        Result += ( C/16  < 10) ? ( C/16 +'0') : ( C/16 -10+'A');
        Result += ((C&15) < 10) ? ((C&15)+'0') : ((C&15)-10+'A');
      }
    }
    Result += "\"";

  } else {
    Result = "[";
    if (Operands.size()) {
      Result += " " + Operands[0]->getType()->getDescription() + 
	        " " + cast<ConstPoolVal>(Operands[0])->getStrValue();
      for (unsigned i = 1; i < Operands.size(); i++)
        Result += ", " + Operands[i]->getType()->getDescription() + 
	           " " + cast<ConstPoolVal>(Operands[i])->getStrValue();
    }
    Result += " ]";
  }
  
  return Result;
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

string ConstPoolPointerNull::getStrValue() const {
  return "null";
}

string ConstPoolPointerRef::getStrValue() const {
  const GlobalValue *V = getValue();
  if (V->hasName()) return "%" + V->getName();

  SlotCalculator *Table = new SlotCalculator(V->getParent(), true);
  int Slot = Table->getValSlot(V);
  delete Table;

  if (Slot >= 0) return string(" %") + itostr(Slot);
  else return "<pointer reference badref>";
}


//===----------------------------------------------------------------------===//
//                           classof implementations

bool ConstPoolInt::classof(const ConstPoolVal *CPV) {
  return CPV->getType()->isIntegral();
}
bool ConstPoolSInt::classof(const ConstPoolVal *CPV) {
  return CPV->getType()->isSigned();
}
bool ConstPoolUInt::classof(const ConstPoolVal *CPV) {
  return CPV->getType()->isUnsigned();
}
bool ConstPoolFP::classof(const ConstPoolVal *CPV) {
  const Type *Ty = CPV->getType();
  return Ty == Type::FloatTy || Ty == Type::DoubleTy;
}
bool ConstPoolArray::classof(const ConstPoolVal *CPV) {
  return isa<ArrayType>(CPV->getType());
}
bool ConstPoolStruct::classof(const ConstPoolVal *CPV) {
  return isa<StructType>(CPV->getType());
}
bool ConstPoolPointer::classof(const ConstPoolVal *CPV) {
  return isa<PointerType>(CPV->getType());
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

  inline void remove(ConstPoolClass *CP) {
    for (map<ConstHashKey,ConstPoolClass *>::iterator I = Map.begin(),
                                                      E = Map.end(); I != E;++I)
      if (I->second == CP) {
	Map.erase(I);
	return;
      }
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

// ConstPoolArray::get(const string&) - Return an array that is initialized to
// contain the specified string.  A null terminator is added to the specified
// string so that it may be used in a natural way...
//
ConstPoolArray *ConstPoolArray::get(const string &Str) {
  vector<ConstPoolVal*> ElementVals;

  for (unsigned i = 0; i < Str.length(); ++i)
    ElementVals.push_back(ConstPoolUInt::get(Type::UByteTy, Str[i]));

  // Add a null terminator to the string...
  ElementVals.push_back(ConstPoolUInt::get(Type::UByteTy, 0));

  ArrayType *ATy = ArrayType::get(Type::UByteTy/*,stringConstant.length()*/);
  return ConstPoolArray::get(ATy, ElementVals);
}


// destroyConstant - Remove the constant from the constant table...
//
void ConstPoolArray::destroyConstant() {
  ArrayConstants.remove(this);
  destroyConstantImpl();
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

// destroyConstant - Remove the constant from the constant table...
//
void ConstPoolStruct::destroyConstant() {
  StructConstants.remove(this);
  destroyConstantImpl();
}

//---- ConstPoolPointerNull::get() implementation...
//
static ValueMap<char, ConstPoolPointerNull> NullPtrConstants;

ConstPoolPointerNull *ConstPoolPointerNull::get(const PointerType *Ty) {
  ConstPoolPointerNull *Result = NullPtrConstants.get(Ty, 0);
  if (!Result)   // If no preexisting value, create one now...
    NullPtrConstants.add(Ty, 0, Result = new ConstPoolPointerNull(Ty));
  return Result;
}

//---- ConstPoolPointerRef::get() implementation...
//
ConstPoolPointerRef *ConstPoolPointerRef::get(GlobalValue *GV) {
  assert(GV->getParent() && "Global Value must be attached to a module!");

  // The Module handles the pointer reference sharing...
  return GV->getParent()->getConstPoolPointerRef(GV);
}


void ConstPoolPointerRef::mutateReference(GlobalValue *NewGV) {
  getValue()->getParent()->mutateConstPoolPointerRef(getValue(), NewGV);
  Operands[0] = NewGV;
}
