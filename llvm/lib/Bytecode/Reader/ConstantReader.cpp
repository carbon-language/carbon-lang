//===- ReadConst.cpp - Code to constants and constant pools -----------------===
//
// This file implements functionality to deserialize constants and entire 
// constant pools.
// 
// Note that this library should be as fast as possible, reentrant, and 
// threadsafe!!
//
//===------------------------------------------------------------------------===

#include "llvm/Module.h"
#include "llvm/BasicBlock.h"
#include "llvm/ConstPoolVals.h"
#include "llvm/DerivedTypes.h"
#include "ReaderInternals.h"
#include <algorithm>



const Type *BytecodeParser::parseTypeConstant(const uchar *&Buf,
					      const uchar *EndBuf) {
  unsigned PrimType;
  if (read_vbr(Buf, EndBuf, PrimType)) return failure<const Type*>(0);

  const Type *Val = 0;
  if ((Val = Type::getPrimitiveType((Type::PrimitiveID)PrimType)))
    return Val;
  
  switch (PrimType) {
  case Type::MethodTyID: {
    unsigned Typ;
    if (read_vbr(Buf, EndBuf, Typ)) return failure(Val);
    const Type *RetType = getType(Typ);
    if (RetType == 0) return failure(Val);

    unsigned NumParams;
    if (read_vbr(Buf, EndBuf, NumParams)) return failure(Val);

    vector<const Type*> Params;
    while (NumParams--) {
      if (read_vbr(Buf, EndBuf, Typ)) return failure(Val);
      const Type *Ty = getType(Typ);
      if (Ty == 0) return failure(Val);
      Params.push_back(Ty);
    }

    Val = MethodType::get(RetType, Params);
    break;
  }
  case Type::ArrayTyID: {
    unsigned ElTyp;
    if (read_vbr(Buf, EndBuf, ElTyp)) return failure(Val);
    const Type *ElementType = getType(ElTyp);
    if (ElementType == 0) return failure(Val);

    int NumElements;
    if (read_vbr(Buf, EndBuf, NumElements)) return failure(Val);
    Val = ArrayType::get(ElementType, NumElements);
    break;
  }
  case Type::StructTyID: {
    unsigned Typ;
    vector<const Type*> Elements;

    if (read_vbr(Buf, EndBuf, Typ)) return failure(Val);
    while (Typ) {         // List is terminated by void/0 typeid
      const Type *Ty = getType(Typ);
      if (Ty == 0) return failure(Val);
      Elements.push_back(Ty);
      
      if (read_vbr(Buf, EndBuf, Typ)) return failure(Val);
    }

    Val = StructType::get(Elements);
    break;
  }
  case Type::PointerTyID: {
    unsigned ElTyp;
    if (read_vbr(Buf, EndBuf, ElTyp)) return failure(Val);
    const Type *ElementType = getType(ElTyp);
    if (ElementType == 0) return failure(Val);
    Val = PointerType::get(ElementType);
    break;
  }

  default:
    cerr << __FILE__ << ":" << __LINE__ << ": Don't know how to deserialize"
	 << " primitive Type " << PrimType << "\n";
    return failure(Val);
  }

  return Val;
}

// refineAbstractType - The callback method is invoked when one of the
// elements of TypeValues becomes more concrete...
//
void BytecodeParser::refineAbstractType(const DerivedType *OldType, 
					const Type *NewType) {
  TypeValuesListTy::iterator I = find(MethodTypeValues.begin(), 
				      MethodTypeValues.end(), OldType);
  if (I == MethodTypeValues.end()) {
    I = find(ModuleTypeValues.begin(), ModuleTypeValues.end(), OldType);
    assert(I != ModuleTypeValues.end() && 
	   "Can't refine a type I don't know about!");
  }

  *I = NewType;  // Update to point to new, more refined type.
}



// parseTypeConstants - We have to use this wierd code to handle recursive
// types.  We know that recursive types will only reference the current slab of
// values in the type plane, but they can forward reference types before they
// have been read.  For example, Type #0 might be '{ Ty#1 }' and Type #1 might
// be 'Ty#0*'.  When reading Type #0, type number one doesn't exist.  To fix
// this ugly problem, we pesimistically insert an opaque type for each type we
// are about to read.  This means that forward references will resolve to
// something and when we reread the type later, we can replace the opaque type
// with a new resolved concrete type.
//
bool BytecodeParser::parseTypeConstants(const uchar *&Buf, const uchar *EndBuf,
					TypeValuesListTy &Tab,
					unsigned NumEntries) {
  assert(Tab.size() == 0 && "I think table should always be empty here!"
	 "This should simplify later code");

  // Record the base, starting level that we will begin with.
  unsigned BaseLevel = Tab.size();

  // Insert a bunch of opaque types to be resolved later...
  for (unsigned i = 0; i < NumEntries; i++)
    Tab.push_back(PATypeHandle<Type>(OpaqueType::get(), this));

  // Loop through reading all of the types.  Forward types will make use of the
  // opaque types just inserted.
  //
  for (unsigned i = 0; i < NumEntries; i++) {
    const Type *NewTy = parseTypeConstant(Buf, EndBuf);
    if (NewTy == 0) return failure(true);
    BCR_TRACE(4, "Read Type Constant: '" << NewTy << "'\n");

    // Don't insertValue the new type... instead we want to replace the opaque
    // type with the new concrete value...
    //

    // Refine the abstract type to the new type.  This causes all uses of the
    // abstract type to use the newty.  This also will cause the opaque type
    // to be deleted...
    //
    cast<DerivedType>(Tab[i+BaseLevel].get())->refineAbstractTypeTo(NewTy);

    // This should have replace the old opaque type with the new type in the
    // value table...
    assert(Tab[i+BaseLevel] == NewTy && "refineAbstractType didn't work!");
  }

  BCR_TRACE(5, "Resulting types:\n");
  for (unsigned i = 0; i < NumEntries; i++) {
    BCR_TRACE(5, cast<const Type>(Tab[i+BaseLevel]) << "\n");
  }
  return false;
}


bool BytecodeParser::parseConstPoolValue(const uchar *&Buf, 
					 const uchar *EndBuf,
					 const Type *Ty, ConstPoolVal *&V) {
  switch (Ty->getPrimitiveID()) {
  case Type::BoolTyID: {
    unsigned Val;
    if (read_vbr(Buf, EndBuf, Val)) return failure(true);
    if (Val != 0 && Val != 1) return failure(true);
    V = ConstPoolBool::get(Val == 1);
    break;
  }

  case Type::UByteTyID:   // Unsigned integer types...
  case Type::UShortTyID:
  case Type::UIntTyID: {
    unsigned Val;
    if (read_vbr(Buf, EndBuf, Val)) return failure(true);
    if (!ConstPoolUInt::isValueValidForType(Ty, Val)) return failure(true);
    V = ConstPoolUInt::get(Ty, Val);
    break;
  }

  case Type::ULongTyID: {
    uint64_t Val;
    if (read_vbr(Buf, EndBuf, Val)) return failure(true);
    V = ConstPoolUInt::get(Ty, Val);
    break;
  }

  case Type::SByteTyID:   // Unsigned integer types...
  case Type::ShortTyID:
  case Type::IntTyID: {
    int Val;
    if (read_vbr(Buf, EndBuf, Val)) return failure(true);
    if (!ConstPoolSInt::isValueValidForType(Ty, Val)) return failure(true);
    V = ConstPoolSInt::get(Ty, Val);
    break;
  }

  case Type::LongTyID: {
    int64_t Val;
    if (read_vbr(Buf, EndBuf, Val)) return failure(true);
    V = ConstPoolSInt::get(Ty, Val);
    break;
  }

  case Type::FloatTyID: {
    float F;
    if (input_data(Buf, EndBuf, &F, &F+1)) return failure(true);
    V = ConstPoolFP::get(Ty, F);
    break;
  }

  case Type::DoubleTyID: {
    double Val;
    if (input_data(Buf, EndBuf, &Val, &Val+1)) return failure(true);
    V = ConstPoolFP::get(Ty, Val);
    break;
  }

  case Type::TypeTyID:
    assert(0 && "Type constants should be handled seperately!!!");
    abort();

  case Type::ArrayTyID: {
    const ArrayType *AT = cast<const ArrayType>(Ty);
    unsigned NumElements;
    if (AT->isSized())          // Sized array, # elements stored in type!
      NumElements = (unsigned)AT->getNumElements();
    else                        // Unsized array, # elements stored in stream!
      if (read_vbr(Buf, EndBuf, NumElements)) return failure(true);

    vector<ConstPoolVal *> Elements;
    while (NumElements--) {   // Read all of the elements of the constant.
      unsigned Slot;
      if (read_vbr(Buf, EndBuf, Slot)) return failure(true);
      Value *V = getValue(AT->getElementType(), Slot, false);
      if (!V || !isa<ConstPoolVal>(V)) return failure(true);
      Elements.push_back(cast<ConstPoolVal>(V));
    }
    V = ConstPoolArray::get(AT, Elements);
    break;
  }

  case Type::StructTyID: {
    const StructType *ST = cast<StructType>(Ty);
    const StructType::ElementTypes &ET = ST->getElementTypes();

    vector<ConstPoolVal *> Elements;
    for (unsigned i = 0; i < ET.size(); ++i) {
      unsigned Slot;
      if (read_vbr(Buf, EndBuf, Slot)) return failure(true);
      Value *V = getValue(ET[i], Slot, false);
      if (!V || !isa<ConstPoolVal>(V))
	return failure(true);
      Elements.push_back(cast<ConstPoolVal>(V));      
    }

    V = ConstPoolStruct::get(ST, Elements);
    break;
  }    

  case Type::PointerTyID: {
    const PointerType *PT = cast<const PointerType>(Ty);
    unsigned SubClass;
    if (read_vbr(Buf, EndBuf, SubClass)) return failure(true);
    if (SubClass != 0) return failure(true);


    V = ConstPoolPointer::getNullPointer(PT);
    break;
  }

  default:
    cerr << __FILE__ << ":" << __LINE__ 
	 << ": Don't know how to deserialize constant value of type '"
	 << Ty->getName() << "'\n";
    return failure(true);
  }

  return false;
}

bool BytecodeParser::ParseConstantPool(const uchar *&Buf, const uchar *EndBuf,
				       ValueTable &Tab, 
				       TypeValuesListTy &TypeTab) {
  while (Buf < EndBuf) {
    unsigned NumEntries, Typ;

    if (read_vbr(Buf, EndBuf, NumEntries) ||
        read_vbr(Buf, EndBuf, Typ)) return failure(true);
    const Type *Ty = getType(Typ);
    if (Ty == 0) return failure(true);
    BCR_TRACE(3, "Type: '" << Ty << "'  NumEntries: " << NumEntries << "\n");

    if (Typ == Type::TypeTyID) {
      if (parseTypeConstants(Buf, EndBuf, TypeTab, NumEntries)) return true;
    } else {
      for (unsigned i = 0; i < NumEntries; i++) {
	ConstPoolVal *I;
	if (parseConstPoolValue(Buf, EndBuf, Ty, I)) return failure(true);
	BCR_TRACE(4, "Read Constant: '" << I << "'\n");
	insertValue(I, Tab);
      }
    }
  }
  
  if (Buf > EndBuf) return failure(true);
  return false;
}
