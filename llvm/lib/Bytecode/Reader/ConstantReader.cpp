//===- ConstantReader.cpp - Code to constants and types ====---------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements functionality to deserialize constants and types from
// bytecode files.
//
//===----------------------------------------------------------------------===//

#include "ReaderInternals.h"
#include "llvm/Module.h"
#include "llvm/Constants.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
#include <algorithm>
using namespace llvm;

const Type *BytecodeParser::parseTypeConstant(const unsigned char *&Buf,
					      const unsigned char *EndBuf) {
  unsigned PrimType = read_vbr_uint(Buf, EndBuf);

  const Type *Val = 0;
  if ((Val = Type::getPrimitiveType((Type::TypeID)PrimType)))
    return Val;
  
  switch (PrimType) {
  case Type::FunctionTyID: {
    const Type *RetType = getType(read_vbr_uint(Buf, EndBuf));

    unsigned NumParams = read_vbr_uint(Buf, EndBuf);

    std::vector<const Type*> Params;
    while (NumParams--)
      Params.push_back(getType(read_vbr_uint(Buf, EndBuf)));

    bool isVarArg = Params.size() && Params.back() == Type::VoidTy;
    if (isVarArg) Params.pop_back();

    return FunctionType::get(RetType, Params, isVarArg);
  }
  case Type::ArrayTyID: {
    unsigned ElTyp = read_vbr_uint(Buf, EndBuf);
    const Type *ElementType = getType(ElTyp);

    unsigned NumElements = read_vbr_uint(Buf, EndBuf);

    BCR_TRACE(5, "Array Type Constant #" << ElTyp << " size=" 
              << NumElements << "\n");
    return ArrayType::get(ElementType, NumElements);
  }
  case Type::StructTyID: {
    std::vector<const Type*> Elements;
    unsigned Typ = read_vbr_uint(Buf, EndBuf);
    while (Typ) {         // List is terminated by void/0 typeid
      Elements.push_back(getType(Typ));
      Typ = read_vbr_uint(Buf, EndBuf);
    }

    return StructType::get(Elements);
  }
  case Type::PointerTyID: {
    unsigned ElTyp = read_vbr_uint(Buf, EndBuf);
    BCR_TRACE(5, "Pointer Type Constant #" << ElTyp << "\n");
    return PointerType::get(getType(ElTyp));
  }

  case Type::OpaqueTyID: {
    return OpaqueType::get();
  }

  default:
    std::cerr << __FILE__ << ":" << __LINE__
              << ": Don't know how to deserialize"
              << " primitive Type " << PrimType << "\n";
    return Val;
  }
}

// parseTypeConstants - We have to use this weird code to handle recursive
// types.  We know that recursive types will only reference the current slab of
// values in the type plane, but they can forward reference types before they
// have been read.  For example, Type #0 might be '{ Ty#1 }' and Type #1 might
// be 'Ty#0*'.  When reading Type #0, type number one doesn't exist.  To fix
// this ugly problem, we pessimistically insert an opaque type for each type we
// are about to read.  This means that forward references will resolve to
// something and when we reread the type later, we can replace the opaque type
// with a new resolved concrete type.
//
void BytecodeParser::parseTypeConstants(const unsigned char *&Buf,
                                        const unsigned char *EndBuf,
					TypeValuesListTy &Tab,
					unsigned NumEntries) {
  assert(Tab.size() == 0 && "should not have read type constants in before!");

  // Insert a bunch of opaque types to be resolved later...
  Tab.reserve(NumEntries);
  for (unsigned i = 0; i != NumEntries; ++i)
    Tab.push_back(OpaqueType::get());

  // Loop through reading all of the types.  Forward types will make use of the
  // opaque types just inserted.
  //
  for (unsigned i = 0; i != NumEntries; ++i) {
    const Type *NewTy = parseTypeConstant(Buf, EndBuf), *OldTy = Tab[i].get();
    if (NewTy == 0) throw std::string("Couldn't parse type!");
    BCR_TRACE(4, "#" << i << ": Read Type Constant: '" << NewTy <<
              "' Replacing: " << OldTy << "\n");

    // Don't insertValue the new type... instead we want to replace the opaque
    // type with the new concrete value...
    //

    // Refine the abstract type to the new type.  This causes all uses of the
    // abstract type to use NewTy.  This also will cause the opaque type to be
    // deleted...
    //
    cast<DerivedType>(const_cast<Type*>(OldTy))->refineAbstractTypeTo(NewTy);

    // This should have replace the old opaque type with the new type in the
    // value table... or with a preexisting type that was already in the system
    assert(Tab[i] != OldTy && "refineAbstractType didn't work!");
  }

  BCR_TRACE(5, "Resulting types:\n");
  for (unsigned i = 0; i < NumEntries; ++i) {
    BCR_TRACE(5, (void*)Tab[i].get() << " - " << Tab[i].get() << "\n");
  }
}


Constant *BytecodeParser::parseConstantValue(const unsigned char *&Buf,
                                             const unsigned char *EndBuf,
                                             unsigned TypeID) {

  // We must check for a ConstantExpr before switching by type because
  // a ConstantExpr can be of any type, and has no explicit value.
  // 
  // 0 if not expr; numArgs if is expr
  unsigned isExprNumArgs = read_vbr_uint(Buf, EndBuf);
  
  if (isExprNumArgs) {
    // FIXME: Encoding of constant exprs could be much more compact!
    std::vector<Constant*> ArgVec;
    ArgVec.reserve(isExprNumArgs);
    unsigned Opcode = read_vbr_uint(Buf, EndBuf);
    
    // Read the slot number and types of each of the arguments
    for (unsigned i = 0; i != isExprNumArgs; ++i) {
      unsigned ArgValSlot = read_vbr_uint(Buf, EndBuf);
      unsigned ArgTypeSlot = read_vbr_uint(Buf, EndBuf);
      BCR_TRACE(4, "CE Arg " << i << ": Type: '" << *getType(ArgTypeSlot)
                << "'  slot: " << ArgValSlot << "\n");
      
      // Get the arg value from its slot if it exists, otherwise a placeholder
      ArgVec.push_back(getConstantValue(ArgTypeSlot, ArgValSlot));
    }
    
    // Construct a ConstantExpr of the appropriate kind
    if (isExprNumArgs == 1) {           // All one-operand expressions
      assert(Opcode == Instruction::Cast);
      return ConstantExpr::getCast(ArgVec[0], getType(TypeID));
    } else if (Opcode == Instruction::GetElementPtr) { // GetElementPtr
      std::vector<Constant*> IdxList(ArgVec.begin()+1, ArgVec.end());

      if (hasRestrictedGEPTypes) {
        const Type *BaseTy = ArgVec[0]->getType();
        generic_gep_type_iterator<std::vector<Constant*>::iterator>
          GTI = gep_type_begin(BaseTy, IdxList.begin(), IdxList.end()),
          E = gep_type_end(BaseTy, IdxList.begin(), IdxList.end());
        for (unsigned i = 0; GTI != E; ++GTI, ++i)
          if (isa<StructType>(*GTI)) {
            if (IdxList[i]->getType() != Type::UByteTy)
              throw std::string("Invalid index for getelementptr!");
            IdxList[i] = ConstantExpr::getCast(IdxList[i], Type::UIntTy);
          }
      }

      return ConstantExpr::getGetElementPtr(ArgVec[0], IdxList);
    } else if (Opcode == Instruction::Select) {
      assert(ArgVec.size() == 3);
      return ConstantExpr::getSelect(ArgVec[0], ArgVec[1], ArgVec[2]);
    } else {                            // All other 2-operand expressions
      return ConstantExpr::get(Opcode, ArgVec[0], ArgVec[1]);
    }
  }
  
  // Ok, not an ConstantExpr.  We now know how to read the given type...
  const Type *Ty = getType(TypeID);
  switch (Ty->getTypeID()) {
  case Type::BoolTyID: {
    unsigned Val = read_vbr_uint(Buf, EndBuf);
    if (Val != 0 && Val != 1) throw std::string("Invalid boolean value read.");
    return ConstantBool::get(Val == 1);
  }

  case Type::UByteTyID:   // Unsigned integer types...
  case Type::UShortTyID:
  case Type::UIntTyID: {
    unsigned Val = read_vbr_uint(Buf, EndBuf);
    if (!ConstantUInt::isValueValidForType(Ty, Val)) 
      throw std::string("Invalid unsigned byte/short/int read.");
    return ConstantUInt::get(Ty, Val);
  }

  case Type::ULongTyID: {
    return ConstantUInt::get(Ty, read_vbr_uint64(Buf, EndBuf));
  }

  case Type::SByteTyID:   // Signed integer types...
  case Type::ShortTyID:
  case Type::IntTyID: {
  case Type::LongTyID:
    int64_t Val = read_vbr_int64(Buf, EndBuf);
    if (!ConstantSInt::isValueValidForType(Ty, Val)) 
      throw std::string("Invalid signed byte/short/int/long read.");
    return ConstantSInt::get(Ty, Val);
  }

  case Type::FloatTyID: {
    float F;
    input_data(Buf, EndBuf, &F, &F+1);
    return ConstantFP::get(Ty, F);
  }

  case Type::DoubleTyID: {
    double Val;
    input_data(Buf, EndBuf, &Val, &Val+1);
    return ConstantFP::get(Ty, Val);
  }

  case Type::TypeTyID:
    throw std::string("Type constants shouldn't live in constant table!");

  case Type::ArrayTyID: {
    const ArrayType *AT = cast<ArrayType>(Ty);
    unsigned NumElements = AT->getNumElements();
    unsigned TypeSlot = getTypeSlot(AT->getElementType());
    std::vector<Constant*> Elements;
    Elements.reserve(NumElements);
    while (NumElements--)     // Read all of the elements of the constant.
      Elements.push_back(getConstantValue(TypeSlot,
                                          read_vbr_uint(Buf, EndBuf)));
    return ConstantArray::get(AT, Elements);
  }

  case Type::StructTyID: {
    const StructType *ST = cast<StructType>(Ty);

    std::vector<Constant *> Elements;
    Elements.reserve(ST->getNumElements());
    for (unsigned i = 0; i != ST->getNumElements(); ++i)
      Elements.push_back(getConstantValue(ST->getElementType(i),
                                          read_vbr_uint(Buf, EndBuf)));

    return ConstantStruct::get(ST, Elements);
  }    

  case Type::PointerTyID: {  // ConstantPointerRef value...
    const PointerType *PT = cast<PointerType>(Ty);
    unsigned Slot = read_vbr_uint(Buf, EndBuf);
    BCR_TRACE(4, "CPR: Type: '" << Ty << "'  slot: " << Slot << "\n");
    
    // Check to see if we have already read this global variable...
    Value *Val = getValue(TypeID, Slot, false);
    GlobalValue *GV;
    if (Val) {
      if (!(GV = dyn_cast<GlobalValue>(Val))) 
        throw std::string("Value of ConstantPointerRef not in ValueTable!");
      BCR_TRACE(5, "Value Found in ValueTable!\n");
    } else {
      throw std::string("Forward references are not allowed here.");
    }
    
    return ConstantPointerRef::get(GV);
  }

  default:
    throw std::string("Don't know how to deserialize constant value of type '"+
                      Ty->getDescription());
  }
}

void BytecodeParser::ParseGlobalTypes(const unsigned char *&Buf,
                                      const unsigned char *EndBuf) {
  ValueTable T;
  ParseConstantPool(Buf, EndBuf, T, ModuleTypeValues);
}

void BytecodeParser::parseStringConstants(const unsigned char *&Buf,
                                          const unsigned char *EndBuf,
                                          unsigned NumEntries, ValueTable &Tab){
  for (; NumEntries; --NumEntries) {
    unsigned Typ = read_vbr_uint(Buf, EndBuf);
    const Type *Ty = getType(Typ);
    if (!isa<ArrayType>(Ty))
      throw std::string("String constant data invalid!");
    
    const ArrayType *ATy = cast<ArrayType>(Ty);
    if (ATy->getElementType() != Type::SByteTy &&
        ATy->getElementType() != Type::UByteTy)
      throw std::string("String constant data invalid!");
    
    // Read character data.  The type tells us how long the string is.
    char Data[ATy->getNumElements()];
    input_data(Buf, EndBuf, Data, Data+ATy->getNumElements());

    std::vector<Constant*> Elements(ATy->getNumElements());
    if (ATy->getElementType() == Type::SByteTy)
      for (unsigned i = 0, e = ATy->getNumElements(); i != e; ++i)
        Elements[i] = ConstantSInt::get(Type::SByteTy, (signed char)Data[i]);
    else
      for (unsigned i = 0, e = ATy->getNumElements(); i != e; ++i)
        Elements[i] = ConstantUInt::get(Type::UByteTy, (unsigned char)Data[i]);

    // Create the constant, inserting it as needed.
    Constant *C = ConstantArray::get(ATy, Elements);
    unsigned Slot = insertValue(C, Typ, Tab);
    ResolveReferencesToConstant(C, Slot);
  }
}


void BytecodeParser::ParseConstantPool(const unsigned char *&Buf,
                                       const unsigned char *EndBuf,
                                       ValueTable &Tab, 
                                       TypeValuesListTy &TypeTab) {
  while (Buf < EndBuf) {
    unsigned NumEntries = read_vbr_uint(Buf, EndBuf);
    unsigned Typ = read_vbr_uint(Buf, EndBuf);
    if (Typ == Type::TypeTyID) {
      BCR_TRACE(3, "Type: 'type'  NumEntries: " << NumEntries << "\n");
      parseTypeConstants(Buf, EndBuf, TypeTab, NumEntries);
    } else if (Typ == Type::VoidTyID) {
      assert(&Tab == &ModuleValues && "Cannot read strings in functions!");
      parseStringConstants(Buf, EndBuf, NumEntries, Tab);
    } else {
      BCR_TRACE(3, "Type: '" << *getType(Typ) << "'  NumEntries: "
                << NumEntries << "\n");

      for (unsigned i = 0; i < NumEntries; ++i) {
        Constant *C = parseConstantValue(Buf, EndBuf, Typ);
        assert(C && "parseConstantValue returned NULL!");
        BCR_TRACE(4, "Read Constant: '" << *C << "'\n");
        unsigned Slot = insertValue(C, Typ, Tab);

        // If we are reading a function constant table, make sure that we adjust
        // the slot number to be the real global constant number.
        //
        if (&Tab != &ModuleValues && Typ < ModuleValues.size() &&
            ModuleValues[Typ])
          Slot += ModuleValues[Typ]->size();
        ResolveReferencesToConstant(C, Slot);
      }
    }
  }
  
  if (Buf > EndBuf) throw std::string("Read past end of buffer.");
}
