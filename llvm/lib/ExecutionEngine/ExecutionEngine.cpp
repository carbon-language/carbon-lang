//===-- ExecutionEngine.cpp - Common Implementation shared by EEs ---------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
// 
// This file defines the common interface used by the various execution engine
// subclasses.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "jit"
#include "Interpreter/Interpreter.h"
#include "JIT/VM.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/ModuleProvider.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/GenericValue.h"
#include "llvm/Target/TargetData.h"
#include "Support/Debug.h"
#include "Support/Statistic.h"
#include "Support/DynamicLinker.h"
#include "Config/dlfcn.h"
using namespace llvm;

namespace {
  Statistic<> NumInitBytes("lli", "Number of bytes of global vars initialized");
}

ExecutionEngine::ExecutionEngine(ModuleProvider *P) : 
  CurMod(*P->getModule()), MP(P) {
  assert(P && "ModuleProvider is null?");
}

ExecutionEngine::ExecutionEngine(Module *M) : CurMod(*M), MP(0) {
  assert(M && "Module is null?");
}

ExecutionEngine::~ExecutionEngine() {
  delete MP;
}

/// If possible, create a JIT, unless the caller specifically requests an
/// Interpreter or there's an error. If even an Interpreter cannot be created,
/// NULL is returned. 
///
ExecutionEngine *ExecutionEngine::create(ModuleProvider *MP, 
                                         bool ForceInterpreter) {
  ExecutionEngine *EE = 0;

  // Unless the interpreter was explicitly selected, make a JIT.
  if (!ForceInterpreter)
    EE = VM::create(MP);

  // If we can't make a JIT, make an interpreter instead.
  try {
    if (EE == 0)
      EE = Interpreter::create(MP->materializeModule());
  } catch (...) {
    EE = 0;
  }
  return EE;
}

/// getPointerToGlobal - This returns the address of the specified global
/// value.  This may involve code generation if it's a function.
///
void *ExecutionEngine::getPointerToGlobal(const GlobalValue *GV) {
  if (Function *F = const_cast<Function*>(dyn_cast<Function>(GV)))
    return getPointerToFunction(F);

  assert(GlobalAddress[GV] && "Global hasn't had an address allocated yet?");
  return GlobalAddress[GV];
}

/// FIXME: document
/// 
GenericValue ExecutionEngine::getConstantValue(const Constant *C) {
  GenericValue Result;

  if (ConstantExpr *CE = const_cast<ConstantExpr*>(dyn_cast<ConstantExpr>(C))) {
    switch (CE->getOpcode()) {
    case Instruction::GetElementPtr: {
      Result = getConstantValue(CE->getOperand(0));
      std::vector<Value*> Indexes(CE->op_begin()+1, CE->op_end());
      uint64_t Offset =
        TD->getIndexedOffset(CE->getOperand(0)->getType(), Indexes);
                             
      Result.LongVal += Offset;
      return Result;
    }
    case Instruction::Cast: {
      // We only need to handle a few cases here.  Almost all casts will
      // automatically fold, just the ones involving pointers won't.
      //
      Constant *Op = CE->getOperand(0);

      // Handle cast of pointer to pointer...
      if (Op->getType()->getPrimitiveID() == C->getType()->getPrimitiveID())
        return getConstantValue(Op);

      // Handle a cast of pointer to any integral type...
      if (isa<PointerType>(Op->getType()) && C->getType()->isIntegral())
        return getConstantValue(Op);
        
      // Handle cast of long to pointer...
      if (isa<PointerType>(C->getType()) && (Op->getType() == Type::LongTy ||
                                             Op->getType() == Type::ULongTy))
        return getConstantValue(Op);
      break;
    }

    case Instruction::Add:
      if (CE->getOperand(0)->getType() == Type::LongTy ||
          CE->getOperand(0)->getType() == Type::ULongTy)
        Result.LongVal = getConstantValue(CE->getOperand(0)).LongVal +
                         getConstantValue(CE->getOperand(1)).LongVal;
      else
        break;
      return Result;

    default:
      break;
    }
    std::cerr << "ConstantExpr not handled as global var init: " << *CE << "\n";
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

/// FIXME: document
///
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
    Store4BytesLittleEndian:
    case Type::FloatTyID:
    case Type::UIntTyID:
    case Type::IntTyID:     Ptr->Untyped[0] =  Val.UIntVal        & 255;
                            Ptr->Untyped[1] = (Val.UIntVal >>  8) & 255;
                            Ptr->Untyped[2] = (Val.UIntVal >> 16) & 255;
                            Ptr->Untyped[3] = (Val.UIntVal >> 24) & 255;
                            break;
    case Type::PointerTyID: if (getTargetData().getPointerSize() == 4)
                              goto Store4BytesLittleEndian;
    case Type::DoubleTyID:
    case Type::ULongTyID:
    case Type::LongTyID:    Ptr->Untyped[0] =  Val.ULongVal        & 255;
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
    Store4BytesBigEndian:
    case Type::FloatTyID:
    case Type::UIntTyID:
    case Type::IntTyID:     Ptr->Untyped[3] =  Val.UIntVal        & 255;
                            Ptr->Untyped[2] = (Val.UIntVal >>  8) & 255;
                            Ptr->Untyped[1] = (Val.UIntVal >> 16) & 255;
                            Ptr->Untyped[0] = (Val.UIntVal >> 24) & 255;
                            break;
    case Type::PointerTyID: if (getTargetData().getPointerSize() == 4)
                              goto Store4BytesBigEndian;
    case Type::DoubleTyID:
    case Type::ULongTyID:
    case Type::LongTyID:    Ptr->Untyped[7] =  Val.ULongVal        & 255;
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

/// FIXME: document
///
GenericValue ExecutionEngine::LoadValueFromMemory(GenericValue *Ptr,
                                                  const Type *Ty) {
  GenericValue Result;
  if (getTargetData().isLittleEndian()) {
    switch (Ty->getPrimitiveID()) {
    case Type::BoolTyID:
    case Type::UByteTyID:
    case Type::SByteTyID:   Result.UByteVal = Ptr->Untyped[0]; break;
    case Type::UShortTyID:
    case Type::ShortTyID:   Result.UShortVal = (unsigned)Ptr->Untyped[0] |
                                              ((unsigned)Ptr->Untyped[1] << 8);
                            break;
    Load4BytesLittleEndian:                            
    case Type::FloatTyID:
    case Type::UIntTyID:
    case Type::IntTyID:     Result.UIntVal = (unsigned)Ptr->Untyped[0] |
                                            ((unsigned)Ptr->Untyped[1] <<  8) |
                                            ((unsigned)Ptr->Untyped[2] << 16) |
                                            ((unsigned)Ptr->Untyped[3] << 24);
                            break;
    case Type::PointerTyID: if (getTargetData().getPointerSize() == 4)
                              goto Load4BytesLittleEndian;
    case Type::DoubleTyID:
    case Type::ULongTyID:
    case Type::LongTyID:    Result.ULongVal = (uint64_t)Ptr->Untyped[0] |
                                             ((uint64_t)Ptr->Untyped[1] <<  8) |
                                             ((uint64_t)Ptr->Untyped[2] << 16) |
                                             ((uint64_t)Ptr->Untyped[3] << 24) |
                                             ((uint64_t)Ptr->Untyped[4] << 32) |
                                             ((uint64_t)Ptr->Untyped[5] << 40) |
                                             ((uint64_t)Ptr->Untyped[6] << 48) |
                                             ((uint64_t)Ptr->Untyped[7] << 56);
                            break;
    default:
      std::cout << "Cannot load value of type " << *Ty << "!\n";
      abort();
    }
  } else {
    switch (Ty->getPrimitiveID()) {
    case Type::BoolTyID:
    case Type::UByteTyID:
    case Type::SByteTyID:   Result.UByteVal = Ptr->Untyped[0]; break;
    case Type::UShortTyID:
    case Type::ShortTyID:   Result.UShortVal = (unsigned)Ptr->Untyped[1] |
                                              ((unsigned)Ptr->Untyped[0] << 8);
                            break;
    Load4BytesBigEndian:
    case Type::FloatTyID:
    case Type::UIntTyID:
    case Type::IntTyID:     Result.UIntVal = (unsigned)Ptr->Untyped[3] |
                                            ((unsigned)Ptr->Untyped[2] <<  8) |
                                            ((unsigned)Ptr->Untyped[1] << 16) |
                                            ((unsigned)Ptr->Untyped[0] << 24);
                            break;
    case Type::PointerTyID: if (getTargetData().getPointerSize() == 4)
                              goto Load4BytesBigEndian;
    case Type::DoubleTyID:
    case Type::ULongTyID:
    case Type::LongTyID:    Result.ULongVal = (uint64_t)Ptr->Untyped[7] |
                                             ((uint64_t)Ptr->Untyped[6] <<  8) |
                                             ((uint64_t)Ptr->Untyped[5] << 16) |
                                             ((uint64_t)Ptr->Untyped[4] << 24) |
                                             ((uint64_t)Ptr->Untyped[3] << 32) |
                                             ((uint64_t)Ptr->Untyped[2] << 40) |
                                             ((uint64_t)Ptr->Untyped[1] << 48) |
                                             ((uint64_t)Ptr->Untyped[0] << 56);
                            break;
    default:
      std::cout << "Cannot load value of type " << *Ty << "!\n";
      abort();
    }
  }
  return Result;
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
      // External variable reference. Try to use the dynamic loader to
      // get a pointer to it.
      if (void *SymAddr = GetAddressOfSymbol(I->getName().c_str()))
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
