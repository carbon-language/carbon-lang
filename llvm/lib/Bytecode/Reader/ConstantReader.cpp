//===- ReadConst.cpp - Code to constants and constant pools ---------------===//
//
// This file implements functionality to deserialize constants and entire 
// constant pools.
// 
// Note that this library should be as fast as possible, reentrant, and 
// thread-safe!!
//
//===----------------------------------------------------------------------===//

#include "ReaderInternals.h"
#include "llvm/Module.h"
#include "llvm/Constants.h"
#include <algorithm>

const Type *BytecodeParser::parseTypeConstant(const unsigned char *&Buf,
					      const unsigned char *EndBuf) {
  unsigned PrimType;
  if (read_vbr(Buf, EndBuf, PrimType)) throw Error_readvbr;

  const Type *Val = 0;
  if ((Val = Type::getPrimitiveType((Type::PrimitiveID)PrimType)))
    return Val;
  
  switch (PrimType) {
  case Type::FunctionTyID: {
    unsigned Typ;
    if (read_vbr(Buf, EndBuf, Typ)) return Val;
    const Type *RetType = getType(Typ);

    unsigned NumParams;
    if (read_vbr(Buf, EndBuf, NumParams)) return Val;

    std::vector<const Type*> Params;
    while (NumParams--) {
      if (read_vbr(Buf, EndBuf, Typ)) return Val;
      Params.push_back(getType(Typ));
    }

    bool isVarArg = Params.size() && Params.back() == Type::VoidTy;
    if (isVarArg) Params.pop_back();

    return FunctionType::get(RetType, Params, isVarArg);
  }
  case Type::ArrayTyID: {
    unsigned ElTyp;
    if (read_vbr(Buf, EndBuf, ElTyp)) return Val;
    const Type *ElementType = getType(ElTyp);

    unsigned NumElements;
    if (read_vbr(Buf, EndBuf, NumElements)) return Val;

    BCR_TRACE(5, "Array Type Constant #" << ElTyp << " size=" 
              << NumElements << "\n");
    return ArrayType::get(ElementType, NumElements);
  }
  case Type::StructTyID: {
    unsigned Typ;
    std::vector<const Type*> Elements;

    if (read_vbr(Buf, EndBuf, Typ)) return Val;
    while (Typ) {         // List is terminated by void/0 typeid
      Elements.push_back(getType(Typ));
      if (read_vbr(Buf, EndBuf, Typ)) return Val;
    }

    return StructType::get(Elements);
  }
  case Type::PointerTyID: {
    unsigned ElTyp;
    if (read_vbr(Buf, EndBuf, ElTyp)) return Val;
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
void debug_type_tables();
void BytecodeParser::parseTypeConstants(const unsigned char *&Buf,
                                        const unsigned char *EndBuf,
					TypeValuesListTy &Tab,
					unsigned NumEntries) {
  assert(Tab.size() == 0 && "should not have read type constants in before!");

  // Insert a bunch of opaque types to be resolved later...
  for (unsigned i = 0; i < NumEntries; ++i)
    Tab.push_back(OpaqueType::get());

  // Loop through reading all of the types.  Forward types will make use of the
  // opaque types just inserted.
  //
  for (unsigned i = 0; i < NumEntries; ++i) {
    const Type *NewTy = parseTypeConstant(Buf, EndBuf), *OldTy = Tab[i].get();
    if (NewTy == 0) throw std::string("Parsed invalid type.");
    BCR_TRACE(4, "#" << i << ": Read Type Constant: '" << NewTy <<
              "' Replacing: " << OldTy << "\n");

    // Don't insertValue the new type... instead we want to replace the opaque
    // type with the new concrete value...
    //

    // Refine the abstract type to the new type.  This causes all uses of the
    // abstract type to use the newty.  This also will cause the opaque type
    // to be deleted...
    //
    ((DerivedType*)Tab[i].get())->refineAbstractTypeTo(NewTy);

    // This should have replace the old opaque type with the new type in the
    // value table... or with a preexisting type that was already in the system
    assert(Tab[i] != OldTy && "refineAbstractType didn't work!");
  }

  BCR_TRACE(5, "Resulting types:\n");
  for (unsigned i = 0; i < NumEntries; ++i) {
    BCR_TRACE(5, (void*)Tab[i].get() << " - " << Tab[i].get() << "\n");
  }
  debug_type_tables();
}


Constant *BytecodeParser::parseConstantValue(const unsigned char *&Buf,
                                             const unsigned char *EndBuf,
                                             const Type *Ty) {

  // We must check for a ConstantExpr before switching by type because
  // a ConstantExpr can be of any type, and has no explicit value.
  // 
  unsigned isExprNumArgs;               // 0 if not expr; numArgs if is expr
  if (read_vbr(Buf, EndBuf, isExprNumArgs)) throw Error_readvbr;
  if (isExprNumArgs) {
    // FIXME: Encoding of constant exprs could be much more compact!
    unsigned Opcode;
    std::vector<Constant*> ArgVec;
    ArgVec.reserve(isExprNumArgs);
    if (read_vbr(Buf, EndBuf, Opcode)) throw Error_readvbr;

    // Read the slot number and types of each of the arguments
    for (unsigned i = 0; i != isExprNumArgs; ++i) {
      unsigned ArgValSlot, ArgTypeSlot;
      if (read_vbr(Buf, EndBuf, ArgValSlot)) throw Error_readvbr;
      if (read_vbr(Buf, EndBuf, ArgTypeSlot)) throw Error_readvbr;
      const Type *ArgTy = getType(ArgTypeSlot);
      
      BCR_TRACE(4, "CE Arg " << i << ": Type: '" << *ArgTy << "'  slot: "
                << ArgValSlot << "\n");
      
      // Get the arg value from its slot if it exists, otherwise a placeholder
      ArgVec.push_back(getConstantValue(ArgTy, ArgValSlot));
    }
    
    // Construct a ConstantExpr of the appropriate kind
    if (isExprNumArgs == 1) {           // All one-operand expressions
      assert(Opcode == Instruction::Cast);
      return ConstantExpr::getCast(ArgVec[0], Ty);
    } else if (Opcode == Instruction::GetElementPtr) { // GetElementPtr
      std::vector<Constant*> IdxList(ArgVec.begin()+1, ArgVec.end());
      return ConstantExpr::getGetElementPtr(ArgVec[0], IdxList);
    } else {                            // All other 2-operand expressions
      return ConstantExpr::get(Opcode, ArgVec[0], ArgVec[1]);
    }
  }
  
  // Ok, not an ConstantExpr.  We now know how to read the given type...
  switch (Ty->getPrimitiveID()) {
  case Type::BoolTyID: {
    unsigned Val;
    if (read_vbr(Buf, EndBuf, Val)) throw Error_readvbr;
    if (Val != 0 && Val != 1) throw std::string("Invalid boolean value read.");
    return ConstantBool::get(Val == 1);
  }

  case Type::UByteTyID:   // Unsigned integer types...
  case Type::UShortTyID:
  case Type::UIntTyID: {
    unsigned Val;
    if (read_vbr(Buf, EndBuf, Val)) throw Error_readvbr;
    if (!ConstantUInt::isValueValidForType(Ty, Val)) 
      throw std::string("Invalid unsigned byte/short/int read.");
    return ConstantUInt::get(Ty, Val);
  }

  case Type::ULongTyID: {
    uint64_t Val;
    if (read_vbr(Buf, EndBuf, Val)) throw Error_readvbr;
    return ConstantUInt::get(Ty, Val);
  }

  case Type::SByteTyID:   // Signed integer types...
  case Type::ShortTyID:
  case Type::IntTyID: {
  case Type::LongTyID:
    int64_t Val;
    if (read_vbr(Buf, EndBuf, Val)) throw Error_readvbr;
    if (!ConstantSInt::isValueValidForType(Ty, Val)) 
      throw std::string("Invalid signed byte/short/int/long read.");
    return ConstantSInt::get(Ty, Val);
  }

  case Type::FloatTyID: {
    float F;
    if (input_data(Buf, EndBuf, &F, &F+1)) throw Error_inputdata;
    return ConstantFP::get(Ty, F);
  }

  case Type::DoubleTyID: {
    double Val;
    if (input_data(Buf, EndBuf, &Val, &Val+1)) throw Error_inputdata;
    return ConstantFP::get(Ty, Val);
  }

  case Type::TypeTyID:
    throw std::string("Type constants shouldn't live in constant table!");

  case Type::ArrayTyID: {
    const ArrayType *AT = cast<ArrayType>(Ty);
    unsigned NumElements = AT->getNumElements();

    std::vector<Constant*> Elements;
    while (NumElements--) {   // Read all of the elements of the constant.
      unsigned Slot;
      if (read_vbr(Buf, EndBuf, Slot)) throw Error_readvbr;
      Elements.push_back(getConstantValue(AT->getElementType(), Slot));
    }
    return ConstantArray::get(AT, Elements);
  }

  case Type::StructTyID: {
    const StructType *ST = cast<StructType>(Ty);
    const StructType::ElementTypes &ET = ST->getElementTypes();

    std::vector<Constant *> Elements;
    for (unsigned i = 0; i < ET.size(); ++i) {
      unsigned Slot;
      if (read_vbr(Buf, EndBuf, Slot)) throw Error_readvbr;
      Elements.push_back(getConstantValue(ET[i], Slot));
    }

    return ConstantStruct::get(ST, Elements);
  }    

  case Type::PointerTyID: {
    const PointerType *PT = cast<PointerType>(Ty);
    unsigned SubClass;
    if (HasImplicitZeroInitializer)
      SubClass = 1;
    else
      if (read_vbr(Buf, EndBuf, SubClass)) throw Error_readvbr;

    switch (SubClass) {
    case 0:    // ConstantPointerNull value...
      return ConstantPointerNull::get(PT);

    case 1: {  // ConstantPointerRef value...
      unsigned Slot;
      if (read_vbr(Buf, EndBuf, Slot)) throw Error_readvbr;
      BCR_TRACE(4, "CPR: Type: '" << Ty << "'  slot: " << Slot << "\n");

      // Check to see if we have already read this global variable...
      Value *Val = getValue(PT, Slot, false);
      GlobalValue *GV;
      if (Val) {
        if (!(GV = dyn_cast<GlobalValue>(Val))) 
          throw std::string("Value of ConstantPointerRef not in ValueTable!");
        BCR_TRACE(5, "Value Found in ValueTable!\n");
      } else if (RevisionNum > 0) {
        // Revision #0 could have forward references to globals that were weird.
        // We got rid of this in subsequent revs.
        throw std::string("Forward references to globals not allowed.");
      } else {         // Nope... find or create a forward ref. for it
        GlobalRefsType::iterator I = GlobalRefs.find(std::make_pair(PT, Slot));

        if (I != GlobalRefs.end()) {
          BCR_TRACE(5, "Previous forward ref found!\n");
          GV = cast<GlobalValue>(I->second);
        } else {
          BCR_TRACE(5, "Creating new forward ref to a global variable!\n");
          
          // Create a placeholder for the global variable reference...
          GlobalVariable *GVar =
            new GlobalVariable(PT->getElementType(), false,
                               GlobalValue::InternalLinkage);
          
          // Keep track of the fact that we have a forward ref to recycle it
          GlobalRefs.insert(std::make_pair(std::make_pair(PT, Slot), GVar));
          
          // Must temporarily push this value into the module table...
          TheModule->getGlobalList().push_back(GVar);
          GV = GVar;
        }
      }

      return ConstantPointerRef::get(GV);
    }
    
    default:
      BCR_TRACE(5, "UNKNOWN Pointer Constant Type!\n");
      throw std::string("Unknown pointer constant type.");
    }
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

void BytecodeParser::ParseConstantPool(const unsigned char *&Buf,
                                       const unsigned char *EndBuf,
                                       ValueTable &Tab, 
                                       TypeValuesListTy &TypeTab) {
  while (Buf < EndBuf) {
    unsigned NumEntries, Typ;

    if (read_vbr(Buf, EndBuf, NumEntries) ||
        read_vbr(Buf, EndBuf, Typ)) throw Error_readvbr;
    if (Typ == Type::TypeTyID) {
      BCR_TRACE(3, "Type: 'type'  NumEntries: " << NumEntries << "\n");
      parseTypeConstants(Buf, EndBuf, TypeTab, NumEntries);
    } else {
      const Type *Ty = getType(Typ);
      BCR_TRACE(3, "Type: '" << *Ty << "'  NumEntries: " << NumEntries << "\n");

      for (unsigned i = 0; i < NumEntries; ++i) {
        Constant *C = parseConstantValue(Buf, EndBuf, Ty);
        assert(C && "parseConstantValue returned NULL!");
        BCR_TRACE(4, "Read Constant: '" << *C << "'\n");
        unsigned Slot = insertValue(C, Typ, Tab);

        // If we are reading a function constant table, make sure that we adjust
        // the slot number to be the real global constant number.
        //
        if (&Tab != &ModuleValues && Typ < ModuleValues.size())
          Slot += ModuleValues[Typ]->size();
        ResolveReferencesToValue(C, Slot);
      }
    }
  }
  
  if (Buf > EndBuf) throw std::string("Read past end of buffer.");
}
