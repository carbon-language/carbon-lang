//===- ReadConst.cpp - Code to constants and constant pools ---------------===//
//
// This file implements functionality to deserialize constants and entire 
// constant pools.
// 
// Note that this library should be as fast as possible, reentrant, and 
// threadsafe!!
//
//===----------------------------------------------------------------------===//

#include "ReaderInternals.h"
#include "llvm/Module.h"
#include "llvm/Constants.h"
#include <algorithm>

const Type *BytecodeParser::parseTypeConstant(const unsigned char *&Buf,
					      const unsigned char *EndBuf) {
  unsigned PrimType;
  if (read_vbr(Buf, EndBuf, PrimType)) return 0;

  const Type *Val = 0;
  if ((Val = Type::getPrimitiveType((Type::PrimitiveID)PrimType)))
    return Val;
  
  switch (PrimType) {
  case Type::FunctionTyID: {
    unsigned Typ;
    if (read_vbr(Buf, EndBuf, Typ)) return Val;
    const Type *RetType = getType(Typ);
    if (RetType == 0) return Val;

    unsigned NumParams;
    if (read_vbr(Buf, EndBuf, NumParams)) return Val;

    std::vector<const Type*> Params;
    while (NumParams--) {
      if (read_vbr(Buf, EndBuf, Typ)) return Val;
      const Type *Ty = getType(Typ);
      if (Ty == 0) return Val;
      Params.push_back(Ty);
    }

    bool isVarArg = Params.size() && Params.back() == Type::VoidTy;
    if (isVarArg) Params.pop_back();

    return FunctionType::get(RetType, Params, isVarArg);
  }
  case Type::ArrayTyID: {
    unsigned ElTyp;
    if (read_vbr(Buf, EndBuf, ElTyp)) return Val;
    const Type *ElementType = getType(ElTyp);
    if (ElementType == 0) return Val;

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
      const Type *Ty = getType(Typ);
      if (Ty == 0) return Val;
      Elements.push_back(Ty);
      
      if (read_vbr(Buf, EndBuf, Typ)) return Val;
    }

    return StructType::get(Elements);
  }
  case Type::PointerTyID: {
    unsigned ElTyp;
    if (read_vbr(Buf, EndBuf, ElTyp)) return Val;
    BCR_TRACE(5, "Pointer Type Constant #" << (ElTyp-14) << "\n");
    const Type *ElementType = getType(ElTyp);
    if (ElementType == 0) return Val;
    return PointerType::get(ElementType);
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

// refineAbstractType - The callback method is invoked when one of the
// elements of TypeValues becomes more concrete...
//
void BytecodeParser::refineAbstractType(const DerivedType *OldType, 
					const Type *NewType) {
  if (OldType == NewType &&
      OldType->isAbstract()) return;  // Type is modified, but same

  TypeValuesListTy::iterator I = find(FunctionTypeValues.begin(), 
				      FunctionTypeValues.end(), OldType);
  if (I == FunctionTypeValues.end()) {
    I = find(ModuleTypeValues.begin(), ModuleTypeValues.end(), OldType);
    assert(I != ModuleTypeValues.end() && 
	   "Can't refine a type I don't know about!");
  }

  if (OldType == NewType) {
    assert(!OldType->isAbstract());
    I->removeUserFromConcrete();
  } else {
    *I = NewType;  // Update to point to new, more refined type.
  }
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
void debug_type_tables();
bool BytecodeParser::parseTypeConstants(const unsigned char *&Buf,
                                        const unsigned char *EndBuf,
					TypeValuesListTy &Tab,
					unsigned NumEntries) {
  assert(Tab.size() == 0 && "should not have read type constants in before!");

  // Insert a bunch of opaque types to be resolved later...
  for (unsigned i = 0; i < NumEntries; ++i)
    Tab.push_back(PATypeHandle(OpaqueType::get(), this));

  // Loop through reading all of the types.  Forward types will make use of the
  // opaque types just inserted.
  //
  for (unsigned i = 0; i < NumEntries; ++i) {
    const Type *NewTy = parseTypeConstant(Buf, EndBuf), *OldTy = Tab[i].get();
    if (NewTy == 0) return true;
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
  return false;
}


bool BytecodeParser::parseConstantValue(const unsigned char *&Buf,
                                        const unsigned char *EndBuf,
                                        const Type *Ty, Constant *&V) {

  // We must check for a ConstantExpr before switching by type because
  // a ConstantExpr can be of any type, and has no explicit value.
  // 
  unsigned isExprNumArgs;               // 0 if not expr; numArgs if is expr
  if (read_vbr(Buf, EndBuf, isExprNumArgs)) return true;
  if (isExprNumArgs) {
    // FIXME: Encoding of constant exprs could be much more compact!
    unsigned Opcode;
    std::vector<Constant*> ArgVec;
    ArgVec.reserve(isExprNumArgs);
    if (read_vbr(Buf, EndBuf, Opcode)) return true;    

    // Read the slot number and types of each of the arguments
    for (unsigned i = 0; i != isExprNumArgs; ++i) {
      unsigned ArgValSlot, ArgTypeSlot;
      if (read_vbr(Buf, EndBuf, ArgValSlot)) return true;
      if (read_vbr(Buf, EndBuf, ArgTypeSlot)) return true;
      const Type *ArgTy = getType(ArgTypeSlot);
      if (ArgTy == 0) return true;
      
      BCR_TRACE(4, "CE Arg " << i << ": Type: '" << ArgTy << "'  slot: "
                << ArgValSlot << "\n");
      
      // Get the arg value from its slot if it exists, otherwise a placeholder
      Constant *C = getConstantValue(ArgTy, ArgValSlot);
      if (C == 0) return true;
      ArgVec.push_back(C);
    }
    
    // Construct a ConstantExpr of the appropriate kind
    if (isExprNumArgs == 1) {           // All one-operand expressions
      assert(Opcode == Instruction::Cast);
      V = ConstantExpr::getCast(ArgVec[0], Ty);
    } else if (Opcode == Instruction::GetElementPtr) { // GetElementPtr
      std::vector<Constant*> IdxList(ArgVec.begin()+1, ArgVec.end());
      V = ConstantExpr::getGetElementPtr(ArgVec[0], IdxList);
    } else {                            // All other 2-operand expressions
      V = ConstantExpr::get(Opcode, ArgVec[0], ArgVec[1]);
    }
    return false;
  }
  
  // Ok, not an ConstantExpr.  We now know how to read the given type...
  switch (Ty->getPrimitiveID()) {
  case Type::BoolTyID: {
    unsigned Val;
    if (read_vbr(Buf, EndBuf, Val)) return true;
    if (Val != 0 && Val != 1) return true;
    V = ConstantBool::get(Val == 1);
    break;
  }

  case Type::UByteTyID:   // Unsigned integer types...
  case Type::UShortTyID:
  case Type::UIntTyID: {
    unsigned Val;
    if (read_vbr(Buf, EndBuf, Val)) return true;
    if (!ConstantUInt::isValueValidForType(Ty, Val)) return true;
    V = ConstantUInt::get(Ty, Val);
    break;
  }

  case Type::ULongTyID: {
    uint64_t Val;
    if (read_vbr(Buf, EndBuf, Val)) return true;
    V = ConstantUInt::get(Ty, Val);
    break;
  }

  case Type::SByteTyID:   // Signed integer types...
  case Type::ShortTyID:
  case Type::IntTyID: {
  case Type::LongTyID:
    int64_t Val;
    if (read_vbr(Buf, EndBuf, Val)) return true;
    if (!ConstantSInt::isValueValidForType(Ty, Val)) return true;
    V = ConstantSInt::get(Ty, Val);
    break;
  }

  case Type::FloatTyID: {
    float F;
    if (input_data(Buf, EndBuf, &F, &F+1)) return true;
    V = ConstantFP::get(Ty, F);
    break;
  }

  case Type::DoubleTyID: {
    double Val;
    if (input_data(Buf, EndBuf, &Val, &Val+1)) return true;
    V = ConstantFP::get(Ty, Val);
    break;
  }

  case Type::TypeTyID:
    assert(0 && "Type constants should be handled separately!!!");
    abort();

  case Type::ArrayTyID: {
    const ArrayType *AT = cast<const ArrayType>(Ty);
    unsigned NumElements = AT->getNumElements();

    std::vector<Constant*> Elements;
    while (NumElements--) {   // Read all of the elements of the constant.
      unsigned Slot;
      if (read_vbr(Buf, EndBuf, Slot)) return true;
      Constant *C = getConstantValue(AT->getElementType(), Slot);
      if (!C) return true;
      Elements.push_back(C);
    }
    V = ConstantArray::get(AT, Elements);
    break;
  }

  case Type::StructTyID: {
    const StructType *ST = cast<StructType>(Ty);
    const StructType::ElementTypes &ET = ST->getElementTypes();

    std::vector<Constant *> Elements;
    for (unsigned i = 0; i < ET.size(); ++i) {
      unsigned Slot;
      if (read_vbr(Buf, EndBuf, Slot)) return true;
      Constant *C = getConstantValue(ET[i], Slot);
      if (!C) return true;
      Elements.push_back(C);
    }

    V = ConstantStruct::get(ST, Elements);
    break;
  }    

  case Type::PointerTyID: {
    const PointerType *PT = cast<const PointerType>(Ty);
    unsigned SubClass;
    if (HasImplicitZeroInitializer)
      SubClass = 1;
    else
      if (read_vbr(Buf, EndBuf, SubClass)) return true;

    switch (SubClass) {
    case 0:    // ConstantPointerNull value...
      V = ConstantPointerNull::get(PT);
      break;

    case 1: {  // ConstantPointerRef value...
      unsigned Slot;
      if (read_vbr(Buf, EndBuf, Slot)) return true;
      BCR_TRACE(4, "CPR: Type: '" << Ty << "'  slot: " << Slot << "\n");

      // Check to see if we have already read this global variable...
      Value *Val = getValue(PT, Slot, false);
      GlobalValue *GV;
      if (Val) {
        if (!(GV = dyn_cast<GlobalValue>(Val))) return true;
        BCR_TRACE(5, "Value Found in ValueTable!\n");
      } else if (RevisionNum > 0) {
        // Revision #0 could have forward references to globals that were wierd.
        // We got rid of this in subsequent revs.
        return true;
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

      V = ConstantPointerRef::get(GV);
      break;
    }
    
    default:
      BCR_TRACE(5, "UNKNOWN Pointer Constant Type!\n");
      return true;
    }
    break;
  }

  default:
    std::cerr << __FILE__ << ":" << __LINE__ 
              << ": Don't know how to deserialize constant value of type '"
              << Ty->getName() << "'\n";
    return true;
  }

  return false;
}

bool BytecodeParser::ParseGlobalTypes(const unsigned char *&Buf,
                                      const unsigned char *EndBuf) {
  ValueTable T;
  return ParseConstantPool(Buf, EndBuf, T, ModuleTypeValues);
}

bool BytecodeParser::ParseConstantPool(const unsigned char *&Buf,
                                       const unsigned char *EndBuf,
				       ValueTable &Tab, 
				       TypeValuesListTy &TypeTab) {
  while (Buf < EndBuf) {
    unsigned NumEntries, Typ;

    if (read_vbr(Buf, EndBuf, NumEntries) ||
        read_vbr(Buf, EndBuf, Typ)) return true;
    const Type *Ty = getType(Typ);
    if (Ty == 0) return true;
    BCR_TRACE(3, "Type: '" << Ty << "'  NumEntries: " << NumEntries << "\n");

    if (Typ == Type::TypeTyID) {
      if (parseTypeConstants(Buf, EndBuf, TypeTab, NumEntries)) return true;
    } else {
      for (unsigned i = 0; i < NumEntries; ++i) {
	Constant *C;
        int Slot;
	if (parseConstantValue(Buf, EndBuf, Ty, C)) return true;
        assert(C && "parseConstantValue returned NULL!");
	BCR_TRACE(4, "Read Constant: '" << *C << "'\n");
	if ((Slot = insertValue(C, Tab)) == -1) return true;

        // If we are reading a function constant table, make sure that we adjust
        // the slot number to be the real global constant number.
        //
        if (&Tab != &ModuleValues && Typ < ModuleValues.size())
          Slot += ModuleValues[Typ]->size();
        ResolveReferencesToValue(C, (unsigned)Slot);
      }
    }
  }
  
  if (Buf > EndBuf) return true;
  return false;
}
