//===-- ExecutionEngine.cpp - Common Implementation shared by EE's --------===//
// 
// This file defines the common interface used by the various execution engine
// subclasses.
//
//===----------------------------------------------------------------------===//

#include "ExecutionEngine.h"
#include "GenericValue.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Constants.h"
#include "llvm/Module.h"
#include "llvm/Target/TargetData.h"
#include "Support/Statistic.h"
#include <dlfcn.h>

Statistic<> NumInitBytes("lli", "Number of bytes of global vars initialized");

// getPointerToGlobal - This returns the address of the specified global
// value.  This may involve code generation if it's a function.
//
void *ExecutionEngine::getPointerToGlobal(const GlobalValue *GV) {
  if (const Function *F = dyn_cast<Function>(GV))
    return getPointerToFunction(F);

  assert(GlobalAddress[GV] && "Global hasn't had an address allocated yet?");
  return GlobalAddress[GV];
}


GenericValue ExecutionEngine::getConstantValue(const Constant *C) {
  GenericValue Result;

  if (ConstantExpr *CE = (ConstantExpr*)dyn_cast<ConstantExpr>(C))
    switch (CE->getOpcode()) {
    case Instruction::GetElementPtr: {
      Result = getConstantValue(cast<Constant>(CE->getOperand(0)));
      std::vector<Value*> Indexes(CE->op_begin()+1, CE->op_end());
      uint64_t Offset =
        TD->getIndexedOffset(CE->getOperand(0)->getType(), Indexes);
                             
      Result.LongVal += Offset;
      return Result;
    }

    default:
      std::cerr << "ConstantExpr not handled as global var init: " << *CE
                << "\n";
      abort();
    }

  switch (C->getType()->getPrimitiveID()) {
#define GET_CONST_VAL(TY, CLASS) \
  case Type::TY##TyID: Result.TY##Val = cast<CLASS>(C)->getValue(); break
    GET_CONST_VAL(Bool   , ConstantBool);
    GET_CONST_VAL(UByte  , ConstantUInt);
    GET_CONST_VAL(SByte  , ConstantSInt);
    GET_CONST_VAL(UShort , ConstantUInt);
    GET_CONST_VAL(Short  , ConstantSInt);
    GET_CONST_VAL(UInt   , ConstantUInt);
    GET_CONST_VAL(Int    , ConstantSInt);
    GET_CONST_VAL(ULong  , ConstantUInt);
    GET_CONST_VAL(Long   , ConstantSInt);
    GET_CONST_VAL(Float  , ConstantFP);
    GET_CONST_VAL(Double , ConstantFP);
#undef GET_CONST_VAL
  case Type::PointerTyID:
    if (isa<ConstantPointerNull>(C)) {
      Result.PointerVal = 0;
    } else if (const ConstantPointerRef *CPR = dyn_cast<ConstantPointerRef>(C)){
      Result = PTOGV(getPointerToGlobal(CPR->getValue()));

    } else {
      assert(0 && "Unknown constant pointer type!");
    }
    break;
  default:
    std::cout << "ERROR: Constant unimp for type: " << C->getType() << "\n";
    abort();
  }
  return Result;
}

void ExecutionEngine::StoreValueToMemory(GenericValue Val, GenericValue *Ptr,
				     const Type *Ty) {
  if (getTargetData().isLittleEndian()) {
    switch (Ty->getPrimitiveID()) {
    case Type::BoolTyID:
    case Type::UByteTyID:
    case Type::SByteTyID:   Ptr->Untyped[0] = Val.UByteVal; break;
    case Type::UShortTyID:
    case Type::ShortTyID:   Ptr->Untyped[0] = Val.UShortVal & 255;
                            Ptr->Untyped[1] = (Val.UShortVal >> 8) & 255;
                            break;
    case Type::FloatTyID:
    case Type::UIntTyID:
    case Type::IntTyID:     Ptr->Untyped[0] =  Val.UIntVal        & 255;
                            Ptr->Untyped[1] = (Val.UIntVal >>  8) & 255;
                            Ptr->Untyped[2] = (Val.UIntVal >> 16) & 255;
                            Ptr->Untyped[3] = (Val.UIntVal >> 24) & 255;
                            break;
    case Type::DoubleTyID:
    case Type::ULongTyID:
    case Type::LongTyID:    
    case Type::PointerTyID: Ptr->Untyped[0] =  Val.ULongVal        & 255;
                            Ptr->Untyped[1] = (Val.ULongVal >>  8) & 255;
                            Ptr->Untyped[2] = (Val.ULongVal >> 16) & 255;
                            Ptr->Untyped[3] = (Val.ULongVal >> 24) & 255;
                            Ptr->Untyped[4] = (Val.ULongVal >> 32) & 255;
                            Ptr->Untyped[5] = (Val.ULongVal >> 40) & 255;
                            Ptr->Untyped[6] = (Val.ULongVal >> 48) & 255;
                            Ptr->Untyped[7] = (Val.ULongVal >> 56) & 255;
                            break;
    default:
      std::cout << "Cannot store value of type " << Ty << "!\n";
    }
  } else {
    switch (Ty->getPrimitiveID()) {
    case Type::BoolTyID:
    case Type::UByteTyID:
    case Type::SByteTyID:   Ptr->Untyped[0] = Val.UByteVal; break;
    case Type::UShortTyID:
    case Type::ShortTyID:   Ptr->Untyped[1] = Val.UShortVal & 255;
                            Ptr->Untyped[0] = (Val.UShortVal >> 8) & 255;
                            break;
    case Type::FloatTyID:
    case Type::UIntTyID:
    case Type::IntTyID:     Ptr->Untyped[3] =  Val.UIntVal        & 255;
                            Ptr->Untyped[2] = (Val.UIntVal >>  8) & 255;
                            Ptr->Untyped[1] = (Val.UIntVal >> 16) & 255;
                            Ptr->Untyped[0] = (Val.UIntVal >> 24) & 255;
                            break;
    case Type::DoubleTyID:
    case Type::ULongTyID:
    case Type::LongTyID:    
    case Type::PointerTyID: Ptr->Untyped[7] =  Val.ULongVal        & 255;
                            Ptr->Untyped[6] = (Val.ULongVal >>  8) & 255;
                            Ptr->Untyped[5] = (Val.ULongVal >> 16) & 255;
                            Ptr->Untyped[4] = (Val.ULongVal >> 24) & 255;
                            Ptr->Untyped[3] = (Val.ULongVal >> 32) & 255;
                            Ptr->Untyped[2] = (Val.ULongVal >> 40) & 255;
                            Ptr->Untyped[1] = (Val.ULongVal >> 48) & 255;
                            Ptr->Untyped[0] = (Val.ULongVal >> 56) & 255;
                            break;
    default:
      std::cout << "Cannot store value of type " << Ty << "!\n";
    }
  }
}

// InitializeMemory - Recursive function to apply a Constant value into the
// specified memory location...
//
void ExecutionEngine::InitializeMemory(const Constant *Init, void *Addr) {
  if (Init->getType()->isFirstClassType()) {
    GenericValue Val = getConstantValue(Init);
    StoreValueToMemory(Val, (GenericValue*)Addr, Init->getType());
    return;
  }

  switch (Init->getType()->getPrimitiveID()) {
  case Type::ArrayTyID: {
    const ConstantArray *CPA = cast<ConstantArray>(Init);
    const std::vector<Use> &Val = CPA->getValues();
    unsigned ElementSize = 
      getTargetData().getTypeSize(cast<ArrayType>(CPA->getType())->getElementType());
    for (unsigned i = 0; i < Val.size(); ++i)
      InitializeMemory(cast<Constant>(Val[i].get()), (char*)Addr+i*ElementSize);
    return;
  }

  case Type::StructTyID: {
    const ConstantStruct *CPS = cast<ConstantStruct>(Init);
    const StructLayout *SL =
      getTargetData().getStructLayout(cast<StructType>(CPS->getType()));
    const std::vector<Use> &Val = CPS->getValues();
    for (unsigned i = 0; i < Val.size(); ++i)
      InitializeMemory(cast<Constant>(Val[i].get()),
                       (char*)Addr+SL->MemberOffsets[i]);
    return;
  }

  default:
    std::cerr << "Bad Type: " << Init->getType() << "\n";
    assert(0 && "Unknown constant type to initialize memory with!");
  }
}



void *ExecutionEngine::CreateArgv(const std::vector<std::string> &InputArgv) {
  // Pointers are 64 bits...
  // FIXME: Assumes 64 bit target
  PointerTy *Result = new PointerTy[InputArgv.size()+1];
  DEBUG(std::cerr << "ARGV = " << (void*)Result << "\n");

  for (unsigned i = 0; i < InputArgv.size(); ++i) {
    unsigned Size = InputArgv[i].size()+1;
    char *Dest = new char[Size];
    DEBUG(std::cerr << "ARGV[" << i << "] = " << (void*)Dest << "\n");

    copy(InputArgv[i].begin(), InputArgv[i].end(), Dest);
    Dest[Size-1] = 0;

    // Endian safe: Result[i] = (PointerTy)Dest;
    StoreValueToMemory(PTOGV(Dest), (GenericValue*)(Result+i),
                       Type::LongTy);  // 64 bit assumption
  }

  Result[InputArgv.size()] = 0;
  return Result;
}

/// EmitGlobals - Emit all of the global variables to memory, storing their
/// addresses into GlobalAddress.  This must make sure to copy the contents of
/// their initializers into the memory.
///
void ExecutionEngine::emitGlobals() {
  const TargetData &TD = getTargetData();
  
  // Loop over all of the global variables in the program, allocating the memory
  // to hold them.
  for (Module::giterator I = getModule().gbegin(), E = getModule().gend();
       I != E; ++I)
    if (!I->isExternal()) {
      // Get the type of the global...
      const Type *Ty = I->getType()->getElementType();
      
      // Allocate some memory for it!
      unsigned Size = TD.getTypeSize(Ty);
      GlobalAddress[I] = new char[Size];
      NumInitBytes += Size;

      DEBUG(std::cerr << "Global '" << I->getName() << "' -> "
	              << (void*)GlobalAddress[I] << "\n");
    } else {
      // External variable reference, try to use dlsym to get a pointer to it in
      // the LLI image.
      if (void *SymAddr = dlsym(0, I->getName().c_str()))
        GlobalAddress[I] = SymAddr;
      else {
        std::cerr << "Could not resolve external global address: "
                  << I->getName() << "\n";
        abort();
      }
    }
  
  // Now that all of the globals are set up in memory, loop through them all and
  // initialize their contents.
  for (Module::giterator I = getModule().gbegin(), E = getModule().gend();
       I != E; ++I)
    if (!I->isExternal())
      InitializeMemory(I->getInitializer(), GlobalAddress[I]);
}

