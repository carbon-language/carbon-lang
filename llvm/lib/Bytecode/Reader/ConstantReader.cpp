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

bool BytecodeParser::parseTypeConstant(const uchar *&Buf, const uchar *EndBuf,
				       ConstPoolVal *&V) {
  const Type *Val = 0;

  unsigned PrimType;
  if (read_vbr(Buf, EndBuf, PrimType)) return true;

  if ((Val = Type::getPrimitiveType((Type::PrimitiveID)PrimType))) {
    V = new ConstPoolType(Val);    // It's just a primitive ID.
    return false;
  }
  
  switch (PrimType) {
  case Type::MethodTyID: {
    unsigned Typ;
    if (read_vbr(Buf, EndBuf, Typ)) return true;
    const Type *RetType = getType(Typ);
    if (RetType == 0) return true;

    MethodType::ParamTypes Params;

    if (read_vbr(Buf, EndBuf, Typ)) return true;
    while (Typ) {
      const Type *Ty = getType(Typ);
      if (Ty == 0) return true;
      Params.push_back(Ty);
      
      if (read_vbr(Buf, EndBuf, Typ)) return true;
    }

    Val = MethodType::getMethodType(RetType, Params);
    break;
  }
  case Type::ArrayTyID: {
    unsigned ElTyp;
    if (read_vbr(Buf, EndBuf, ElTyp)) return true;
    const Type *ElementType = getType(ElTyp);
    if (ElementType == 0) return true;

    int NumElements;
    if (read_vbr(Buf, EndBuf, NumElements)) return true;
    Val = ArrayType::getArrayType(ElementType, NumElements);
    break;
  }
  case Type::StructTyID: {
    unsigned Typ;
    StructType::ElementTypes Elements;

    if (read_vbr(Buf, EndBuf, Typ)) return true;
    while (Typ) {         // List is terminated by void/0 typeid
      const Type *Ty = getType(Typ);
      if (Ty == 0) return true;
      Elements.push_back(Ty);
      
      if (read_vbr(Buf, EndBuf, Typ)) return true;
    }

    Val = StructType::getStructType(Elements);
    break;
  }
  case Type::PointerTyID: {
    unsigned ElTyp;
    if (read_vbr(Buf, EndBuf, ElTyp)) return true;
    const Type *ElementType = getType(ElTyp);
    if (ElementType == 0) return true;
    Val = PointerType::getPointerType(ElementType);
    break;
  }

  default:
    cerr << __FILE__ << ":" << __LINE__ << ": Don't know how to deserialize"
	 << " primitive Type " << PrimType << "\n";
    return true;
  }

  V = new ConstPoolType(Val);
  return false;
}

bool BytecodeParser::parseConstPoolValue(const uchar *&Buf, 
					 const uchar *EndBuf,
					 const Type *Ty, ConstPoolVal *&V) {
  switch (Ty->getPrimitiveID()) {
  case Type::BoolTyID: {
    unsigned Val;
    if (read_vbr(Buf, EndBuf, Val)) return true;
    if (Val != 0 && Val != 1) return true;
    V = new ConstPoolBool(Val == 1);
    break;
  }

  case Type::UByteTyID:   // Unsigned integer types...
  case Type::UShortTyID:
  case Type::UIntTyID: {
    unsigned Val;
    if (read_vbr(Buf, EndBuf, Val)) return true;
    if (!ConstPoolUInt::isValueValidForType(Ty, Val)) return true;
    V = new ConstPoolUInt(Ty, Val);
    break;
  }

  case Type::ULongTyID: {
    uint64_t Val;
    if (read_vbr(Buf, EndBuf, Val)) return true;
    V = new ConstPoolUInt(Ty, Val);
    break;
  }

  case Type::SByteTyID:   // Unsigned integer types...
  case Type::ShortTyID:
  case Type::IntTyID: {
    int Val;
    if (read_vbr(Buf, EndBuf, Val)) return true;
    if (!ConstPoolSInt::isValueValidForType(Ty, Val)) return 0;
    V = new ConstPoolSInt(Ty, Val);
    break;
  }

  case Type::LongTyID: {
    int64_t Val;
    if (read_vbr(Buf, EndBuf, Val)) return true;
    V = new ConstPoolSInt(Ty, Val);
    break;
  }

  case Type::TypeTyID:
    if (parseTypeConstant(Buf, EndBuf, V)) return true;
    break;

  case Type::ArrayTyID: {
    const ArrayType *AT = (const ArrayType*)Ty;
    unsigned NumElements;
    if (AT->isSized())          // Sized array, # elements stored in type!
      NumElements = (unsigned)AT->getNumElements();
    else                        // Unsized array, # elements stored in stream!
      if (read_vbr(Buf, EndBuf, NumElements)) return true;

    vector<ConstPoolVal *> Elements;
    while (NumElements--) {   // Read all of the elements of the constant.
      unsigned Slot;
      if (read_vbr(Buf, EndBuf, Slot)) return true;
      Value *V = getValue(AT->getElementType(), Slot, false);
      if (!V || !V->isConstant())
	return true;
      Elements.push_back((ConstPoolVal*)V);
    }
    V = new ConstPoolArray(AT, Elements);
    break;
  }

  case Type::StructTyID: {
    const StructType *ST = (const StructType*)Ty;
    const StructType::ElementTypes &ET = ST->getElementTypes();

    vector<ConstPoolVal *> Elements;
    for (unsigned i = 0; i < ET.size(); ++i) {
      unsigned Slot;
      if (read_vbr(Buf, EndBuf, Slot)) return true;
      Value *V = getValue(ET[i], Slot, false);
      if (!V || !V->isConstant())
	return true;
      Elements.push_back((ConstPoolVal*)V);      
    }

    V = new ConstPoolStruct(ST, Elements);
    break;
  }    

  default:
    cerr << __FILE__ << ":" << __LINE__ 
	 << ": Don't know how to deserialize constant value of type '"
	 << Ty->getName() << "'\n";
    return true;
  }
  return false;
}

bool BytecodeParser::ParseConstantPool(const uchar *&Buf, const uchar *EndBuf,
				       SymTabValue::ConstantPoolType &CP, 
				       ValueTable &Tab) {
  while (Buf < EndBuf) {
    unsigned NumEntries, Typ;

    if (read_vbr(Buf, EndBuf, NumEntries) ||
        read_vbr(Buf, EndBuf, Typ)) return true;
    const Type *Ty = getType(Typ);
    if (Ty == 0) return true;

    for (unsigned i = 0; i < NumEntries; i++) {
      ConstPoolVal *I;
      if (parseConstPoolValue(Buf, EndBuf, Ty, I)) return true;
#if 0
      cerr << "  Read const value: <" << I->getType()->getName() 
	   << ">: " << I->getStrValue() << endl;
#endif
      insertValue(I, Tab);
      CP.insert(I);
    }
  }
  
  return Buf > EndBuf;
}
