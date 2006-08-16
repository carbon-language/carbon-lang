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
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/ModuleProvider.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/GenericValue.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MutexGuard.h"
#include "llvm/System/DynamicLibrary.h"
#include "llvm/Target/TargetData.h"
#include <iostream>
using namespace llvm;

namespace {
  Statistic<> NumInitBytes("lli", "Number of bytes of global vars initialized");
  Statistic<> NumGlobals  ("lli", "Number of global vars initialized");
}

ExecutionEngine::EECtorFn ExecutionEngine::JITCtor = 0;
ExecutionEngine::EECtorFn ExecutionEngine::InterpCtor = 0;

ExecutionEngine::ExecutionEngine(ModuleProvider *P) {
  Modules.push_back(P);
  assert(P && "ModuleProvider is null?");
}

ExecutionEngine::ExecutionEngine(Module *M) {
  assert(M && "Module is null?");
  Modules.push_back(new ExistingModuleProvider(M));
}

ExecutionEngine::~ExecutionEngine() {
  for (unsigned i = 0, e = Modules.size(); i != e; ++i)
    delete Modules[i];
}

/// FindFunctionNamed - Search all of the active modules to find the one that
/// defines FnName.  This is very slow operation and shouldn't be used for
/// general code.
Function *ExecutionEngine::FindFunctionNamed(const char *FnName) {
  for (unsigned i = 0, e = Modules.size(); i != e; ++i) {
    if (Function *F = Modules[i]->getModule()->getNamedFunction(FnName))
      return F;
  }
  return 0;
}


/// addGlobalMapping - Tell the execution engine that the specified global is
/// at the specified location.  This is used internally as functions are JIT'd
/// and as global variables are laid out in memory.  It can and should also be
/// used by clients of the EE that want to have an LLVM global overlay
/// existing data in memory.
void ExecutionEngine::addGlobalMapping(const GlobalValue *GV, void *Addr) {
  MutexGuard locked(lock);
  
  void *&CurVal = state.getGlobalAddressMap(locked)[GV];
  assert((CurVal == 0 || Addr == 0) && "GlobalMapping already established!");
  CurVal = Addr;
  
  // If we are using the reverse mapping, add it too
  if (!state.getGlobalAddressReverseMap(locked).empty()) {
    const GlobalValue *&V = state.getGlobalAddressReverseMap(locked)[Addr];
    assert((V == 0 || GV == 0) && "GlobalMapping already established!");
    V = GV;
  }
}

/// clearAllGlobalMappings - Clear all global mappings and start over again
/// use in dynamic compilation scenarios when you want to move globals
void ExecutionEngine::clearAllGlobalMappings() {
  MutexGuard locked(lock);
  
  state.getGlobalAddressMap(locked).clear();
  state.getGlobalAddressReverseMap(locked).clear();
}

/// updateGlobalMapping - Replace an existing mapping for GV with a new
/// address.  This updates both maps as required.  If "Addr" is null, the
/// entry for the global is removed from the mappings.
void ExecutionEngine::updateGlobalMapping(const GlobalValue *GV, void *Addr) {
  MutexGuard locked(lock);
  
  // Deleting from the mapping?
  if (Addr == 0) {
    state.getGlobalAddressMap(locked).erase(GV);
    if (!state.getGlobalAddressReverseMap(locked).empty())
      state.getGlobalAddressReverseMap(locked).erase(Addr);
    return;
  }
  
  void *&CurVal = state.getGlobalAddressMap(locked)[GV];
  if (CurVal && !state.getGlobalAddressReverseMap(locked).empty())
    state.getGlobalAddressReverseMap(locked).erase(CurVal);
  CurVal = Addr;
  
  // If we are using the reverse mapping, add it too
  if (!state.getGlobalAddressReverseMap(locked).empty()) {
    const GlobalValue *&V = state.getGlobalAddressReverseMap(locked)[Addr];
    assert((V == 0 || GV == 0) && "GlobalMapping already established!");
    V = GV;
  }
}

/// getPointerToGlobalIfAvailable - This returns the address of the specified
/// global value if it is has already been codegen'd, otherwise it returns null.
///
void *ExecutionEngine::getPointerToGlobalIfAvailable(const GlobalValue *GV) {
  MutexGuard locked(lock);
  
  std::map<const GlobalValue*, void*>::iterator I =
  state.getGlobalAddressMap(locked).find(GV);
  return I != state.getGlobalAddressMap(locked).end() ? I->second : 0;
}

/// getGlobalValueAtAddress - Return the LLVM global value object that starts
/// at the specified address.
///
const GlobalValue *ExecutionEngine::getGlobalValueAtAddress(void *Addr) {
  MutexGuard locked(lock);

  // If we haven't computed the reverse mapping yet, do so first.
  if (state.getGlobalAddressReverseMap(locked).empty()) {
    for (std::map<const GlobalValue*, void *>::iterator
         I = state.getGlobalAddressMap(locked).begin(),
         E = state.getGlobalAddressMap(locked).end(); I != E; ++I)
      state.getGlobalAddressReverseMap(locked).insert(std::make_pair(I->second,
                                                                     I->first));
  }

  std::map<void *, const GlobalValue*>::iterator I =
    state.getGlobalAddressReverseMap(locked).find(Addr);
  return I != state.getGlobalAddressReverseMap(locked).end() ? I->second : 0;
}

// CreateArgv - Turn a vector of strings into a nice argv style array of
// pointers to null terminated strings.
//
static void *CreateArgv(ExecutionEngine *EE,
                        const std::vector<std::string> &InputArgv) {
  unsigned PtrSize = EE->getTargetData()->getPointerSize();
  char *Result = new char[(InputArgv.size()+1)*PtrSize];

  DEBUG(std::cerr << "ARGV = " << (void*)Result << "\n");
  const Type *SBytePtr = PointerType::get(Type::SByteTy);

  for (unsigned i = 0; i != InputArgv.size(); ++i) {
    unsigned Size = InputArgv[i].size()+1;
    char *Dest = new char[Size];
    DEBUG(std::cerr << "ARGV[" << i << "] = " << (void*)Dest << "\n");

    std::copy(InputArgv[i].begin(), InputArgv[i].end(), Dest);
    Dest[Size-1] = 0;

    // Endian safe: Result[i] = (PointerTy)Dest;
    EE->StoreValueToMemory(PTOGV(Dest), (GenericValue*)(Result+i*PtrSize),
                           SBytePtr);
  }

  // Null terminate it
  EE->StoreValueToMemory(PTOGV(0),
                         (GenericValue*)(Result+InputArgv.size()*PtrSize),
                         SBytePtr);
  return Result;
}


/// runStaticConstructorsDestructors - This method is used to execute all of
/// the static constructors or destructors for a program, depending on the
/// value of isDtors.
void ExecutionEngine::runStaticConstructorsDestructors(bool isDtors) {
  const char *Name = isDtors ? "llvm.global_dtors" : "llvm.global_ctors";
  
  // Execute global ctors/dtors for each module in the program.
  for (unsigned m = 0, e = Modules.size(); m != e; ++m) {
    GlobalVariable *GV = Modules[m]->getModule()->getNamedGlobal(Name);

    // If this global has internal linkage, or if it has a use, then it must be
    // an old-style (llvmgcc3) static ctor with __main linked in and in use.  If
    // this is the case, don't execute any of the global ctors, __main will do
    // it.
    if (!GV || GV->isExternal() || GV->hasInternalLinkage()) continue;
  
    // Should be an array of '{ int, void ()* }' structs.  The first value is
    // the init priority, which we ignore.
    ConstantArray *InitList = dyn_cast<ConstantArray>(GV->getInitializer());
    if (!InitList) continue;
    for (unsigned i = 0, e = InitList->getNumOperands(); i != e; ++i)
      if (ConstantStruct *CS = 
          dyn_cast<ConstantStruct>(InitList->getOperand(i))) {
        if (CS->getNumOperands() != 2) break; // Not array of 2-element structs.
      
        Constant *FP = CS->getOperand(1);
        if (FP->isNullValue())
          break;  // Found a null terminator, exit.
      
        if (ConstantExpr *CE = dyn_cast<ConstantExpr>(FP))
          if (CE->getOpcode() == Instruction::Cast)
            FP = CE->getOperand(0);
        if (Function *F = dyn_cast<Function>(FP)) {
          // Execute the ctor/dtor function!
          runFunction(F, std::vector<GenericValue>());
        }
      }
  }
}

/// runFunctionAsMain - This is a helper function which wraps runFunction to
/// handle the common task of starting up main with the specified argc, argv,
/// and envp parameters.
int ExecutionEngine::runFunctionAsMain(Function *Fn,
                                       const std::vector<std::string> &argv,
                                       const char * const * envp) {
  std::vector<GenericValue> GVArgs;
  GenericValue GVArgc;
  GVArgc.IntVal = argv.size();
  unsigned NumArgs = Fn->getFunctionType()->getNumParams();
  if (NumArgs) {
    GVArgs.push_back(GVArgc); // Arg #0 = argc.
    if (NumArgs > 1) {
      GVArgs.push_back(PTOGV(CreateArgv(this, argv))); // Arg #1 = argv.
      assert(((char **)GVTOP(GVArgs[1]))[0] &&
             "argv[0] was null after CreateArgv");
      if (NumArgs > 2) {
        std::vector<std::string> EnvVars;
        for (unsigned i = 0; envp[i]; ++i)
          EnvVars.push_back(envp[i]);
        GVArgs.push_back(PTOGV(CreateArgv(this, EnvVars))); // Arg #2 = envp.
      }
    }
  }
  return runFunction(Fn, GVArgs).IntVal;
}

/// If possible, create a JIT, unless the caller specifically requests an
/// Interpreter or there's an error. If even an Interpreter cannot be created,
/// NULL is returned.
///
ExecutionEngine *ExecutionEngine::create(ModuleProvider *MP,
                                         bool ForceInterpreter) {
  ExecutionEngine *EE = 0;

  // Unless the interpreter was explicitly selected, try making a JIT.
  if (!ForceInterpreter && JITCtor)
    EE = JITCtor(MP);

  // If we can't make a JIT, make an interpreter instead.
  if (EE == 0 && InterpCtor)
    EE = InterpCtor(MP);

  if (EE) {
    // Make sure we can resolve symbols in the program as well. The zero arg
    // to the function tells DynamicLibrary to load the program, not a library.
    try {
      sys::DynamicLibrary::LoadLibraryPermanently(0);
    } catch (...) {
    }
  }

  return EE;
}

/// getPointerToGlobal - This returns the address of the specified global
/// value.  This may involve code generation if it's a function.
///
void *ExecutionEngine::getPointerToGlobal(const GlobalValue *GV) {
  if (Function *F = const_cast<Function*>(dyn_cast<Function>(GV)))
    return getPointerToFunction(F);

  MutexGuard locked(lock);
  void *p = state.getGlobalAddressMap(locked)[GV];
  if (p)
    return p;

  // Global variable might have been added since interpreter started.
  if (GlobalVariable *GVar =
          const_cast<GlobalVariable *>(dyn_cast<GlobalVariable>(GV)))
    EmitGlobalVariable(GVar);
  else
    assert("Global hasn't had an address allocated yet!");
  return state.getGlobalAddressMap(locked)[GV];
}

/// FIXME: document
///
GenericValue ExecutionEngine::getConstantValue(const Constant *C) {
  GenericValue Result;
  if (isa<UndefValue>(C)) return Result;

  if (ConstantExpr *CE = const_cast<ConstantExpr*>(dyn_cast<ConstantExpr>(C))) {
    switch (CE->getOpcode()) {
    case Instruction::GetElementPtr: {
      Result = getConstantValue(CE->getOperand(0));
      std::vector<Value*> Indexes(CE->op_begin()+1, CE->op_end());
      uint64_t Offset =
        TD->getIndexedOffset(CE->getOperand(0)->getType(), Indexes);

      if (getTargetData()->getPointerSize() == 4)
        Result.IntVal += Offset;
      else
        Result.LongVal += Offset;
      return Result;
    }
    case Instruction::Cast: {
      // We only need to handle a few cases here.  Almost all casts will
      // automatically fold, just the ones involving pointers won't.
      //
      Constant *Op = CE->getOperand(0);
      GenericValue GV = getConstantValue(Op);

      // Handle cast of pointer to pointer...
      if (Op->getType()->getTypeID() == C->getType()->getTypeID())
        return GV;

      // Handle a cast of pointer to any integral type...
      if (isa<PointerType>(Op->getType()) && C->getType()->isIntegral())
        return GV;

      // Handle cast of integer to a pointer...
      if (isa<PointerType>(C->getType()) && Op->getType()->isIntegral())
        switch (Op->getType()->getTypeID()) {
        case Type::BoolTyID:    return PTOGV((void*)(uintptr_t)GV.BoolVal);
        case Type::SByteTyID:   return PTOGV((void*)( intptr_t)GV.SByteVal);
        case Type::UByteTyID:   return PTOGV((void*)(uintptr_t)GV.UByteVal);
        case Type::ShortTyID:   return PTOGV((void*)( intptr_t)GV.ShortVal);
        case Type::UShortTyID:  return PTOGV((void*)(uintptr_t)GV.UShortVal);
        case Type::IntTyID:     return PTOGV((void*)( intptr_t)GV.IntVal);
        case Type::UIntTyID:    return PTOGV((void*)(uintptr_t)GV.UIntVal);
        case Type::LongTyID:    return PTOGV((void*)( intptr_t)GV.LongVal);
        case Type::ULongTyID:   return PTOGV((void*)(uintptr_t)GV.ULongVal);
        default: assert(0 && "Unknown integral type!");
        }
      break;
    }

    case Instruction::Add:
      switch (CE->getOperand(0)->getType()->getTypeID()) {
      default: assert(0 && "Bad add type!"); abort();
      case Type::LongTyID:
      case Type::ULongTyID:
        Result.LongVal = getConstantValue(CE->getOperand(0)).LongVal +
                         getConstantValue(CE->getOperand(1)).LongVal;
        break;
      case Type::IntTyID:
      case Type::UIntTyID:
        Result.IntVal = getConstantValue(CE->getOperand(0)).IntVal +
                        getConstantValue(CE->getOperand(1)).IntVal;
        break;
      case Type::ShortTyID:
      case Type::UShortTyID:
        Result.ShortVal = getConstantValue(CE->getOperand(0)).ShortVal +
                          getConstantValue(CE->getOperand(1)).ShortVal;
        break;
      case Type::SByteTyID:
      case Type::UByteTyID:
        Result.SByteVal = getConstantValue(CE->getOperand(0)).SByteVal +
                          getConstantValue(CE->getOperand(1)).SByteVal;
        break;
      case Type::FloatTyID:
        Result.FloatVal = getConstantValue(CE->getOperand(0)).FloatVal +
                          getConstantValue(CE->getOperand(1)).FloatVal;
        break;
      case Type::DoubleTyID:
        Result.DoubleVal = getConstantValue(CE->getOperand(0)).DoubleVal +
                           getConstantValue(CE->getOperand(1)).DoubleVal;
        break;
      }
      return Result;
    default:
      break;
    }
    std::cerr << "ConstantExpr not handled as global var init: " << *CE << "\n";
    abort();
  }

  switch (C->getType()->getTypeID()) {
#define GET_CONST_VAL(TY, CTY, CLASS) \
  case Type::TY##TyID: Result.TY##Val = (CTY)cast<CLASS>(C)->getValue(); break
    GET_CONST_VAL(Bool   , bool          , ConstantBool);
    GET_CONST_VAL(UByte  , unsigned char , ConstantUInt);
    GET_CONST_VAL(SByte  , signed char   , ConstantSInt);
    GET_CONST_VAL(UShort , unsigned short, ConstantUInt);
    GET_CONST_VAL(Short  , signed short  , ConstantSInt);
    GET_CONST_VAL(UInt   , unsigned int  , ConstantUInt);
    GET_CONST_VAL(Int    , signed int    , ConstantSInt);
    GET_CONST_VAL(ULong  , uint64_t      , ConstantUInt);
    GET_CONST_VAL(Long   , int64_t       , ConstantSInt);
    GET_CONST_VAL(Float  , float         , ConstantFP);
    GET_CONST_VAL(Double , double        , ConstantFP);
#undef GET_CONST_VAL
  case Type::PointerTyID:
    if (isa<ConstantPointerNull>(C))
      Result.PointerVal = 0;
    else if (const Function *F = dyn_cast<Function>(C))
      Result = PTOGV(getPointerToFunctionOrStub(const_cast<Function*>(F)));
    else if (const GlobalVariable* GV = dyn_cast<GlobalVariable>(C))
      Result = PTOGV(getOrEmitGlobalVariable(const_cast<GlobalVariable*>(GV)));
    else
      assert(0 && "Unknown constant pointer type!");
    break;
  default:
    std::cout << "ERROR: Constant unimp for type: " << *C->getType() << "\n";
    abort();
  }
  return Result;
}

/// StoreValueToMemory - Stores the data in Val of type Ty at address Ptr.  Ptr
/// is the address of the memory at which to store Val, cast to GenericValue *.
/// It is not a pointer to a GenericValue containing the address at which to
/// store Val.
///
void ExecutionEngine::StoreValueToMemory(GenericValue Val, GenericValue *Ptr,
                                         const Type *Ty) {
  if (getTargetData()->isLittleEndian()) {
    switch (Ty->getTypeID()) {
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
    case Type::PointerTyID: if (getTargetData()->getPointerSize() == 4)
                              goto Store4BytesLittleEndian;
    case Type::DoubleTyID:
    case Type::ULongTyID:
    case Type::LongTyID:
      Ptr->Untyped[0] = (unsigned char)(Val.ULongVal      );
      Ptr->Untyped[1] = (unsigned char)(Val.ULongVal >>  8);
      Ptr->Untyped[2] = (unsigned char)(Val.ULongVal >> 16);
      Ptr->Untyped[3] = (unsigned char)(Val.ULongVal >> 24);
      Ptr->Untyped[4] = (unsigned char)(Val.ULongVal >> 32);
      Ptr->Untyped[5] = (unsigned char)(Val.ULongVal >> 40);
      Ptr->Untyped[6] = (unsigned char)(Val.ULongVal >> 48);
      Ptr->Untyped[7] = (unsigned char)(Val.ULongVal >> 56);
      break;
    default:
      std::cout << "Cannot store value of type " << *Ty << "!\n";
    }
  } else {
    switch (Ty->getTypeID()) {
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
    case Type::PointerTyID: if (getTargetData()->getPointerSize() == 4)
                              goto Store4BytesBigEndian;
    case Type::DoubleTyID:
    case Type::ULongTyID:
    case Type::LongTyID:
      Ptr->Untyped[7] = (unsigned char)(Val.ULongVal      );
      Ptr->Untyped[6] = (unsigned char)(Val.ULongVal >>  8);
      Ptr->Untyped[5] = (unsigned char)(Val.ULongVal >> 16);
      Ptr->Untyped[4] = (unsigned char)(Val.ULongVal >> 24);
      Ptr->Untyped[3] = (unsigned char)(Val.ULongVal >> 32);
      Ptr->Untyped[2] = (unsigned char)(Val.ULongVal >> 40);
      Ptr->Untyped[1] = (unsigned char)(Val.ULongVal >> 48);
      Ptr->Untyped[0] = (unsigned char)(Val.ULongVal >> 56);
      break;
    default:
      std::cout << "Cannot store value of type " << *Ty << "!\n";
    }
  }
}

/// FIXME: document
///
GenericValue ExecutionEngine::LoadValueFromMemory(GenericValue *Ptr,
                                                  const Type *Ty) {
  GenericValue Result;
  if (getTargetData()->isLittleEndian()) {
    switch (Ty->getTypeID()) {
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
    case Type::PointerTyID: if (getTargetData()->getPointerSize() == 4)
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
    switch (Ty->getTypeID()) {
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
    case Type::PointerTyID: if (getTargetData()->getPointerSize() == 4)
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
  if (isa<UndefValue>(Init)) {
    return;
  } else if (const ConstantPacked *CP = dyn_cast<ConstantPacked>(Init)) {
    unsigned ElementSize =
      getTargetData()->getTypeSize(CP->getType()->getElementType());
    for (unsigned i = 0, e = CP->getNumOperands(); i != e; ++i)
      InitializeMemory(CP->getOperand(i), (char*)Addr+i*ElementSize);
    return;
  } else if (Init->getType()->isFirstClassType()) {
    GenericValue Val = getConstantValue(Init);
    StoreValueToMemory(Val, (GenericValue*)Addr, Init->getType());
    return;
  } else if (isa<ConstantAggregateZero>(Init)) {
    memset(Addr, 0, (size_t)getTargetData()->getTypeSize(Init->getType()));
    return;
  }

  switch (Init->getType()->getTypeID()) {
  case Type::ArrayTyID: {
    const ConstantArray *CPA = cast<ConstantArray>(Init);
    unsigned ElementSize =
      getTargetData()->getTypeSize(CPA->getType()->getElementType());
    for (unsigned i = 0, e = CPA->getNumOperands(); i != e; ++i)
      InitializeMemory(CPA->getOperand(i), (char*)Addr+i*ElementSize);
    return;
  }

  case Type::StructTyID: {
    const ConstantStruct *CPS = cast<ConstantStruct>(Init);
    const StructLayout *SL =
      getTargetData()->getStructLayout(cast<StructType>(CPS->getType()));
    for (unsigned i = 0, e = CPS->getNumOperands(); i != e; ++i)
      InitializeMemory(CPS->getOperand(i), (char*)Addr+SL->MemberOffsets[i]);
    return;
  }

  default:
    std::cerr << "Bad Type: " << *Init->getType() << "\n";
    assert(0 && "Unknown constant type to initialize memory with!");
  }
}

/// EmitGlobals - Emit all of the global variables to memory, storing their
/// addresses into GlobalAddress.  This must make sure to copy the contents of
/// their initializers into the memory.
///
void ExecutionEngine::emitGlobals() {
  const TargetData *TD = getTargetData();

  // Loop over all of the global variables in the program, allocating the memory
  // to hold them.  If there is more than one module, do a prepass over globals
  // to figure out how the different modules should link together.
  //
  std::map<std::pair<std::string, const Type*>,
           const GlobalValue*> LinkedGlobalsMap;

  if (Modules.size() != 1) {
    for (unsigned m = 0, e = Modules.size(); m != e; ++m) {
      Module &M = *Modules[m]->getModule();
      for (Module::const_global_iterator I = M.global_begin(),
           E = M.global_end(); I != E; ++I) {
        const GlobalValue *GV = I;
        if (GV->hasInternalLinkage() || GV->isExternal() ||
            GV->hasAppendingLinkage() || !GV->hasName())
          continue;// Ignore external globals and globals with internal linkage.
          
        const GlobalValue *&GVEntry = 
          LinkedGlobalsMap[std::make_pair(GV->getName(), GV->getType())];

        // If this is the first time we've seen this global, it is the canonical
        // version.
        if (!GVEntry) {
          GVEntry = GV;
          continue;
        }
        
        // If the existing global is strong, never replace it.
        if (GVEntry->hasExternalLinkage())
          continue;
        
        // Otherwise, we know it's linkonce/weak, replace it if this is a strong
        // symbol.
        if (GV->hasExternalLinkage())
          GVEntry = GV;
      }
    }
  }
  
  std::vector<const GlobalValue*> NonCanonicalGlobals;
  for (unsigned m = 0, e = Modules.size(); m != e; ++m) {
    Module &M = *Modules[m]->getModule();
    for (Module::const_global_iterator I = M.global_begin(), E = M.global_end();
         I != E; ++I) {
      // In the multi-module case, see what this global maps to.
      if (!LinkedGlobalsMap.empty()) {
        if (const GlobalValue *GVEntry = 
              LinkedGlobalsMap[std::make_pair(I->getName(), I->getType())]) {
          // If something else is the canonical global, ignore this one.
          if (GVEntry != &*I) {
            NonCanonicalGlobals.push_back(I);
            continue;
          }
        }
      }
      
      if (!I->isExternal()) {
        // Get the type of the global.
        const Type *Ty = I->getType()->getElementType();

        // Allocate some memory for it!
        unsigned Size = TD->getTypeSize(Ty);
        addGlobalMapping(I, new char[Size]);
      } else {
        // External variable reference. Try to use the dynamic loader to
        // get a pointer to it.
        if (void *SymAddr =
            sys::DynamicLibrary::SearchForAddressOfSymbol(I->getName().c_str()))
          addGlobalMapping(I, SymAddr);
        else {
          std::cerr << "Could not resolve external global address: "
                    << I->getName() << "\n";
          abort();
        }
      }
    }
    
    // If there are multiple modules, map the non-canonical globals to their
    // canonical location.
    if (!NonCanonicalGlobals.empty()) {
      for (unsigned i = 0, e = NonCanonicalGlobals.size(); i != e; ++i) {
        const GlobalValue *GV = NonCanonicalGlobals[i];
        const GlobalValue *CGV =
          LinkedGlobalsMap[std::make_pair(GV->getName(), GV->getType())];
        void *Ptr = getPointerToGlobalIfAvailable(CGV);
        assert(Ptr && "Canonical global wasn't codegen'd!");
        addGlobalMapping(GV, getPointerToGlobalIfAvailable(CGV));
      }
    }
    
    // Now that all of the globals are set up in memory, loop through them all and
    // initialize their contents.
    for (Module::const_global_iterator I = M.global_begin(), E = M.global_end();
         I != E; ++I) {
      if (!I->isExternal()) {
        if (!LinkedGlobalsMap.empty()) {
          if (const GlobalValue *GVEntry = 
                LinkedGlobalsMap[std::make_pair(I->getName(), I->getType())])
            if (GVEntry != &*I)  // Not the canonical variable.
              continue;
        }
        EmitGlobalVariable(I);
      }
    }
  }
}

// EmitGlobalVariable - This method emits the specified global variable to the
// address specified in GlobalAddresses, or allocates new memory if it's not
// already in the map.
void ExecutionEngine::EmitGlobalVariable(const GlobalVariable *GV) {
  void *GA = getPointerToGlobalIfAvailable(GV);
  DEBUG(std::cerr << "Global '" << GV->getName() << "' -> " << GA << "\n");

  const Type *ElTy = GV->getType()->getElementType();
  size_t GVSize = (size_t)getTargetData()->getTypeSize(ElTy);
  if (GA == 0) {
    // If it's not already specified, allocate memory for the global.
    GA = new char[GVSize];
    addGlobalMapping(GV, GA);
  }

  InitializeMemory(GV->getInitializer(), GA);
  NumInitBytes += (unsigned)GVSize;
  ++NumGlobals;
}
