//===-- Constants.cpp - Implement Constant nodes -----------------*- C++ -*--=//
//
// This file implements the Constant* classes...
//
//===----------------------------------------------------------------------===//

#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/iMemory.h"
#include "llvm/SymbolTable.h"
#include "llvm/Module.h"
#include "Support/StringExtras.h"
#include <algorithm>

using std::map;
using std::pair;
using std::make_pair;
using std::vector;

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
  // constants are implicitly dependant on the module that is being deleted,
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

// Static constructor to create a '0' constant of arbitrary type...
Constant *Constant::getNullValue(const Type *Ty) {
  switch (Ty->getPrimitiveID()) {
  case Type::BoolTyID:   return ConstantBool::get(false);
  case Type::SByteTyID:
  case Type::ShortTyID:
  case Type::IntTyID:
  case Type::LongTyID:   return ConstantSInt::get(Ty, 0);

  case Type::UByteTyID:
  case Type::UShortTyID:
  case Type::UIntTyID:
  case Type::ULongTyID:  return ConstantUInt::get(Ty, 0);

  case Type::FloatTyID:
  case Type::DoubleTyID: return ConstantFP::get(Ty, 0);

  case Type::PointerTyID: 
    return ConstantPointerNull::get(cast<PointerType>(Ty));
  case Type::StructTyID: {
    const StructType *ST = cast<StructType>(Ty);

    const StructType::ElementTypes &ETs = ST->getElementTypes();
    std::vector<Constant*> Elements;
    Elements.resize(ETs.size());
    for (unsigned i = 0, e = ETs.size(); i != e; ++i)
      Elements[i] = Constant::getNullValue(ETs[i]);
    return ConstantStruct::get(ST, Elements);
  }
  case Type::ArrayTyID: {
    const ArrayType *AT = cast<ArrayType>(Ty);
    Constant *El = Constant::getNullValue(AT->getElementType());
    unsigned NumElements = AT->getNumElements();
    return ConstantArray::get(AT, std::vector<Constant*>(NumElements, El));
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
    assert(V[i]->getType() == T->getElementType());
    Operands.push_back(Use(V[i], this));
  }
}

ConstantStruct::ConstantStruct(const StructType *T,
                               const std::vector<Constant*> &V) : Constant(T) {
  const StructType::ElementTypes &ETypes = T->getElementTypes();
  assert(V.size() == ETypes.size() &&
         "Invalid initializer vector for constant structure");
  Operands.reserve(V.size());
  for (unsigned i = 0, e = V.size(); i != e; ++i) {
    assert(V[i]->getType() == ETypes[i]);
    Operands.push_back(Use(V[i], this));
  }
}

ConstantPointerRef::ConstantPointerRef(GlobalValue *GV)
  : ConstantPointer(GV->getType()) {
  Operands.push_back(Use(GV, this));
}

ConstantExpr::ConstantExpr(unsigned Opcode, Constant *C, const Type *Ty)
  : Constant(Ty), iType(Opcode) {
  Operands.push_back(Use(C, this));
}

ConstantExpr::ConstantExpr(unsigned Opcode, Constant *C1, Constant *C2)
  : Constant(C1->getType()), iType(Opcode) {
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
bool ConstantPointer::classof(const Constant *CPV) {
  return (isa<PointerType>(CPV->getType()) && !isa<ConstantExpr>(CPV));
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
    /*
    return (Val <= UINT8_MAX);
    */
  case Type::DoubleTyID:
    return true;          // This is the largest type...
  }
};

//===----------------------------------------------------------------------===//
//                replaceUsesOfWithOnConstant implementations

void ConstantArray::replaceUsesOfWithOnConstant(Value *From, Value *To) {
  assert(isa<Constant>(To) && "Cannot make Constant refer to non-constant!");

  std::vector<Constant*> Values;
  Values.reserve(getValues().size());  // Build replacement array...
  for (unsigned i = 0, e = getValues().size(); i != e; ++i) {
    Constant *Val = cast<Constant>(getValues()[i]);
    if (Val == From) Val = cast<Constant>(To);
    Values.push_back(Val);
  }
  
  ConstantArray *Replacement = ConstantArray::get(getType(), Values);
  assert(Replacement != this && "I didn't contain From!");

  // Everyone using this now uses the replacement...
  replaceAllUsesWith(Replacement);
  
  // Delete the old constant!
  destroyConstant();  
}

void ConstantStruct::replaceUsesOfWithOnConstant(Value *From, Value *To) {
  assert(isa<Constant>(To) && "Cannot make Constant refer to non-constant!");

  std::vector<Constant*> Values;
  Values.reserve(getValues().size());
  for (unsigned i = 0, e = getValues().size(); i != e; ++i) {
    Constant *Val = cast<Constant>(getValues()[i]);
    if (Val == From) Val = cast<Constant>(To);
    Values.push_back(Val);
  }
  
  ConstantStruct *Replacement = ConstantStruct::get(getType(), Values);
  assert(Replacement != this && "I didn't contain From!");

  // Everyone using this now uses the replacement...
  replaceAllUsesWith(Replacement);
  
  // Delete the old constant!
  destroyConstant();
}

void ConstantPointerRef::replaceUsesOfWithOnConstant(Value *From, Value *To) {
  if (isa<GlobalValue>(To)) {
    assert(From == getOperand(0) && "Doesn't contain from!");
    ConstantPointerRef *Replacement =
      ConstantPointerRef::get(cast<GlobalValue>(To));
    
    // Everyone using this now uses the replacement...
    replaceAllUsesWith(Replacement);
    
    // Delete the old constant!
    destroyConstant();
  } else {
    // Just replace ourselves with the To value specified.
    replaceAllUsesWith(To);
  
    // Delete the old constant!
    destroyConstant();
  }
}

void ConstantExpr::replaceUsesOfWithOnConstant(Value *From, Value *To) {
  assert(isa<Constant>(To) && "Cannot make Constant refer to non-constant!");

  ConstantExpr *Replacement = 0;
  if (getOpcode() == Instruction::GetElementPtr) {
    std::vector<Constant*> Indices;
    Constant *Pointer = cast<Constant>(getOperand(0));
    Indices.reserve(getNumOperands()-1);
    if (Pointer == From) Pointer = cast<Constant>(To);
    
    for (unsigned i = 1, e = getNumOperands(); i != e; ++i) {
      Constant *Val = cast<Constant>(getOperand(i));
      if (Val == From) Val = cast<Constant>(To);
      Indices.push_back(Val);
    }
    Replacement = ConstantExpr::getGetElementPtr(Pointer, Indices);
  } else if (getOpcode() == Instruction::Cast) {
    assert(getOperand(0) == From && "Cast only has one use!");
    Replacement = ConstantExpr::getCast(cast<Constant>(To), getType());
  } else if (getNumOperands() == 2) {
    Constant *C1 = cast<Constant>(getOperand(0));
    Constant *C2 = cast<Constant>(getOperand(1));
    if (C1 == From) C1 = cast<Constant>(To);
    if (C2 == From) C2 = cast<Constant>(To);
    Replacement = ConstantExpr::get(getOpcode(), C1, C2);
  } else {
    assert(0 && "Unknown ConstantExpr type!");
    return;
  }
  
  assert(Replacement != this && "I didn't contain From!");

  // Everyone using this now uses the replacement...
  replaceAllUsesWith(Replacement);
  
  // Delete the old constant!
  destroyConstant();
}



//===----------------------------------------------------------------------===//
//                      Factory Function Implementation

template<class ValType, class ConstantClass>
struct ValueMap {
  typedef pair<const Type*, ValType> ConstHashKey;
  map<ConstHashKey, ConstantClass *> Map;

  inline ConstantClass *get(const Type *Ty, ValType V) {
    typename map<ConstHashKey,ConstantClass *>::iterator I =
      Map.find(ConstHashKey(Ty, V));
    return (I != Map.end()) ? I->second : 0;
  }

  inline void add(const Type *Ty, ValType V, ConstantClass *CP) {
    Map.insert(make_pair(ConstHashKey(Ty, V), CP));
  }

  inline void remove(ConstantClass *CP) {
    for (typename map<ConstHashKey,ConstantClass *>::iterator I = Map.begin(),
                                                      E = Map.end(); I != E;++I)
      if (I->second == CP) {
	Map.erase(I);
	return;
      }
  }
};

//---- ConstantUInt::get() and ConstantSInt::get() implementations...
//
static ValueMap<uint64_t, ConstantInt> IntConstants;

ConstantSInt *ConstantSInt::get(const Type *Ty, int64_t V) {
  ConstantSInt *Result = (ConstantSInt*)IntConstants.get(Ty, (uint64_t)V);
  if (!Result)   // If no preexisting value, create one now...
    IntConstants.add(Ty, V, Result = new ConstantSInt(Ty, V));
  return Result;
}

ConstantUInt *ConstantUInt::get(const Type *Ty, uint64_t V) {
  ConstantUInt *Result = (ConstantUInt*)IntConstants.get(Ty, V);
  if (!Result)   // If no preexisting value, create one now...
    IntConstants.add(Ty, V, Result = new ConstantUInt(Ty, V));
  return Result;
}

ConstantInt *ConstantInt::get(const Type *Ty, unsigned char V) {
  assert(V <= 127 && "Can only be used with very small positive constants!");
  if (Ty->isSigned()) return ConstantSInt::get(Ty, V);
  return ConstantUInt::get(Ty, V);
}

//---- ConstantFP::get() implementation...
//
static ValueMap<double, ConstantFP> FPConstants;

ConstantFP *ConstantFP::get(const Type *Ty, double V) {
  ConstantFP *Result = FPConstants.get(Ty, V);
  if (!Result)   // If no preexisting value, create one now...
    FPConstants.add(Ty, V, Result = new ConstantFP(Ty, V));
  return Result;
}

//---- ConstantArray::get() implementation...
//
static ValueMap<std::vector<Constant*>, ConstantArray> ArrayConstants;

ConstantArray *ConstantArray::get(const ArrayType *Ty,
                                  const std::vector<Constant*> &V) {
  ConstantArray *Result = ArrayConstants.get(Ty, V);
  if (!Result)   // If no preexisting value, create one now...
    ArrayConstants.add(Ty, V, Result = new ConstantArray(Ty, V));
  return Result;
}

// ConstantArray::get(const string&) - Return an array that is initialized to
// contain the specified string.  A null terminator is added to the specified
// string so that it may be used in a natural way...
//
ConstantArray *ConstantArray::get(const std::string &Str) {
  std::vector<Constant*> ElementVals;

  for (unsigned i = 0; i < Str.length(); ++i)
    ElementVals.push_back(ConstantSInt::get(Type::SByteTy, Str[i]));

  // Add a null terminator to the string...
  ElementVals.push_back(ConstantSInt::get(Type::SByteTy, 0));

  ArrayType *ATy = ArrayType::get(Type::SByteTy, Str.length()+1);
  return ConstantArray::get(ATy, ElementVals);
}


// destroyConstant - Remove the constant from the constant table...
//
void ConstantArray::destroyConstant() {
  ArrayConstants.remove(this);
  destroyConstantImpl();
}

// getAsString - If the sub-element type of this array is either sbyte or ubyte,
// then this method converts the array to an std::string and returns it.
// Otherwise, it asserts out.
//
std::string ConstantArray::getAsString() const {
  std::string Result;
  if (getType()->getElementType() == Type::SByteTy)
    for (unsigned i = 0, e = getNumOperands(); i != e; ++i)
      Result += (char)cast<ConstantSInt>(getOperand(i))->getValue();
  else {
    assert(getType()->getElementType() == Type::UByteTy && "Not a string!");
    for (unsigned i = 0, e = getNumOperands(); i != e; ++i)
      Result += (char)cast<ConstantUInt>(getOperand(i))->getValue();
  }
  return Result;
}


//---- ConstantStruct::get() implementation...
//
static ValueMap<std::vector<Constant*>, ConstantStruct> StructConstants;

ConstantStruct *ConstantStruct::get(const StructType *Ty,
                                    const std::vector<Constant*> &V) {
  ConstantStruct *Result = StructConstants.get(Ty, V);
  if (!Result)   // If no preexisting value, create one now...
    StructConstants.add(Ty, V, Result = new ConstantStruct(Ty, V));
  return Result;
}

// destroyConstant - Remove the constant from the constant table...
//
void ConstantStruct::destroyConstant() {
  StructConstants.remove(this);
  destroyConstantImpl();
}


//---- ConstantPointerNull::get() implementation...
//
static ValueMap<char, ConstantPointerNull> NullPtrConstants;

ConstantPointerNull *ConstantPointerNull::get(const PointerType *Ty) {
  ConstantPointerNull *Result = NullPtrConstants.get(Ty, 0);
  if (!Result)   // If no preexisting value, create one now...
    NullPtrConstants.add(Ty, 0, Result = new ConstantPointerNull(Ty));
  return Result;
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
typedef pair<unsigned, vector<Constant*> > ExprMapKeyType;
static ValueMap<const ExprMapKeyType, ConstantExpr> ExprConstants;

ConstantExpr *ConstantExpr::getCast(Constant *C, const Type *Ty) {

  // Look up the constant in the table first to ensure uniqueness
  vector<Constant*> argVec(1, C);
  const ExprMapKeyType &Key = make_pair(Instruction::Cast, argVec);
  ConstantExpr *Result = ExprConstants.get(Ty, Key);
  if (Result) return Result;
  
  // Its not in the table so create a new one and put it in the table.
  Result = new ConstantExpr(Instruction::Cast, C, Ty);
  ExprConstants.add(Ty, Key, Result);
  return Result;
}

ConstantExpr *ConstantExpr::get(unsigned Opcode, Constant *C1, Constant *C2) {
  // Look up the constant in the table first to ensure uniqueness
  vector<Constant*> argVec(1, C1); argVec.push_back(C2);
  const ExprMapKeyType &Key = make_pair(Opcode, argVec);
  ConstantExpr *Result = ExprConstants.get(C1->getType(), Key);
  if (Result) return Result;
  
  // Its not in the table so create a new one and put it in the table.
  // Check the operands for consistency first
  assert((Opcode >= Instruction::BinaryOpsBegin &&
          Opcode < Instruction::BinaryOpsEnd) &&
         "Invalid opcode in binary constant expression");

  assert(C1->getType() == C2->getType() &&
         "Operand types in binary constant expression should match");
  
  Result = new ConstantExpr(Opcode, C1, C2);
  ExprConstants.add(C1->getType(), Key, Result);
  return Result;
}

ConstantExpr *ConstantExpr::getGetElementPtr(Constant *C,
                                        const std::vector<Constant*> &IdxList) {
  const Type *Ty = C->getType();

  // Look up the constant in the table first to ensure uniqueness
  vector<Constant*> argVec(1, C);
  argVec.insert(argVec.end(), IdxList.begin(), IdxList.end());
  
  const ExprMapKeyType &Key = make_pair(Instruction::GetElementPtr, argVec);
  ConstantExpr *Result = ExprConstants.get(Ty, Key);
  if (Result) return Result;

  // Its not in the table so create a new one and put it in the table.
  // Check the operands for consistency first
  // 
  assert(isa<PointerType>(Ty) &&
         "Non-pointer type for constant GelElementPtr expression");

  // Check that the indices list is valid...
  std::vector<Value*> ValIdxList(IdxList.begin(), IdxList.end());
  const Type *DestTy = GetElementPtrInst::getIndexedType(Ty, ValIdxList, true);
  assert(DestTy && "Invalid index list for constant GelElementPtr expression");
  
  Result = new ConstantExpr(C, IdxList, PointerType::get(DestTy));
  ExprConstants.add(Ty, Key, Result);
  return Result;
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

    OldGV->getParent()->mutateConstantPointerRef(OldGV, NewGV);
    Operands[0] = NewGV;
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
