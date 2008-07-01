//===-- JIT.cpp - LLVM Just in Time Compiler ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This tool implements a just-in-time compiler for LLVM, allowing direct
// execution of LLVM bitcode in an efficient manner.
//
//===----------------------------------------------------------------------===//

#include "JIT.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Instructions.h"
#include "llvm/ModuleProvider.h"
#include "llvm/CodeGen/MachineCodeEmitter.h"
#include "llvm/ExecutionEngine/GenericValue.h"
#include "llvm/Support/MutexGuard.h"
#include "llvm/System/DynamicLibrary.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetJITInfo.h"

#include "llvm/Config/config.h"

using namespace llvm;

#ifdef __APPLE__ 
// Apple gcc defaults to -fuse-cxa-atexit (i.e. calls __cxa_atexit instead
// of atexit). It passes the address of linker generated symbol __dso_handle
// to the function.
// This configuration change happened at version 5330.
# include <AvailabilityMacros.h>
# if defined(MAC_OS_X_VERSION_10_4) && \
     ((MAC_OS_X_VERSION_MIN_REQUIRED > MAC_OS_X_VERSION_10_4) || \
      (MAC_OS_X_VERSION_MIN_REQUIRED == MAC_OS_X_VERSION_10_4 && \
       __APPLE_CC__ >= 5330))
#  ifndef HAVE___DSO_HANDLE
#   define HAVE___DSO_HANDLE 1
#  endif
# endif
#endif

#if HAVE___DSO_HANDLE
extern void *__dso_handle __attribute__ ((__visibility__ ("hidden")));
#endif

namespace {

static struct RegisterJIT {
  RegisterJIT() { JIT::Register(); }
} JITRegistrator;

}

namespace llvm {
  void LinkInJIT() {
  }
}

#if defined (__GNUC__)
extern "C" void __register_frame(void*);
#endif

/// createJIT - This is the factory method for creating a JIT for the current
/// machine, it does not fall back to the interpreter.  This takes ownership
/// of the module provider.
ExecutionEngine *ExecutionEngine::createJIT(ModuleProvider *MP,
                                            std::string *ErrorStr,
                                            JITMemoryManager *JMM) {
  ExecutionEngine *EE = JIT::createJIT(MP, ErrorStr, JMM);
  if (!EE) return 0;
  
  // Register routine for informing unwinding runtime about new EH frames
#if defined(__GNUC__)
  EE->InstallExceptionTableRegister(__register_frame);
#endif

  // Make sure we can resolve symbols in the program as well. The zero arg
  // to the function tells DynamicLibrary to load the program, not a library.
  sys::DynamicLibrary::LoadLibraryPermanently(0, ErrorStr);
  return EE;
}

JIT::JIT(ModuleProvider *MP, TargetMachine &tm, TargetJITInfo &tji,
         JITMemoryManager *JMM)
  : ExecutionEngine(MP), TM(tm), TJI(tji) {
  setTargetData(TM.getTargetData());

  jitstate = new JITState(MP);

  // Initialize MCE
  MCE = createEmitter(*this, JMM);

  // Add target data
  MutexGuard locked(lock);
  FunctionPassManager &PM = jitstate->getPM(locked);
  PM.add(new TargetData(*TM.getTargetData()));

  // Turn the machine code intermediate representation into bytes in memory that
  // may be executed.
  if (TM.addPassesToEmitMachineCode(PM, *MCE, false /*fast*/)) {
    cerr << "Target does not support machine code emission!\n";
    abort();
  }
  
  // Initialize passes.
  PM.doInitialization();
}

JIT::~JIT() {
  delete jitstate;
  delete MCE;
  delete &TM;
}

/// addModuleProvider - Add a new ModuleProvider to the JIT.  If we previously
/// removed the last ModuleProvider, we need re-initialize jitstate with a valid
/// ModuleProvider.
void JIT::addModuleProvider(ModuleProvider *MP) {
  MutexGuard locked(lock);

  if (Modules.empty()) {
    assert(!jitstate && "jitstate should be NULL if Modules vector is empty!");

    jitstate = new JITState(MP);

    FunctionPassManager &PM = jitstate->getPM(locked);
    PM.add(new TargetData(*TM.getTargetData()));

    // Turn the machine code intermediate representation into bytes in memory
    // that may be executed.
    if (TM.addPassesToEmitMachineCode(PM, *MCE, false /*fast*/)) {
      cerr << "Target does not support machine code emission!\n";
      abort();
    }
    
    // Initialize passes.
    PM.doInitialization();
  }
  
  ExecutionEngine::addModuleProvider(MP);
}

/// removeModuleProvider - If we are removing the last ModuleProvider, 
/// invalidate the jitstate since the PassManager it contains references a
/// released ModuleProvider.
Module *JIT::removeModuleProvider(ModuleProvider *MP, std::string *E) {
  Module *result = ExecutionEngine::removeModuleProvider(MP, E);
  
  MutexGuard locked(lock);
  if (Modules.empty()) {
    delete jitstate;
    jitstate = 0;
  }
  
  return result;
}

/// run - Start execution with the specified function and arguments.
///
GenericValue JIT::runFunction(Function *F,
                              const std::vector<GenericValue> &ArgValues) {
  assert(F && "Function *F was null at entry to run()");

  void *FPtr = getPointerToFunction(F);
  assert(FPtr && "Pointer to fn's code was null after getPointerToFunction");
  const FunctionType *FTy = F->getFunctionType();
  const Type *RetTy = FTy->getReturnType();

  assert((FTy->getNumParams() <= ArgValues.size() || FTy->isVarArg()) &&
         "Too many arguments passed into function!");
  assert(FTy->getNumParams() == ArgValues.size() &&
         "This doesn't support passing arguments through varargs (yet)!");

  // Handle some common cases first.  These cases correspond to common `main'
  // prototypes.
  if (RetTy == Type::Int32Ty || RetTy == Type::VoidTy) {
    switch (ArgValues.size()) {
    case 3:
      if (FTy->getParamType(0) == Type::Int32Ty &&
          isa<PointerType>(FTy->getParamType(1)) &&
          isa<PointerType>(FTy->getParamType(2))) {
        int (*PF)(int, char **, const char **) =
          (int(*)(int, char **, const char **))(intptr_t)FPtr;

        // Call the function.
        GenericValue rv;
        rv.IntVal = APInt(32, PF(ArgValues[0].IntVal.getZExtValue(), 
                                 (char **)GVTOP(ArgValues[1]),
                                 (const char **)GVTOP(ArgValues[2])));
        return rv;
      }
      break;
    case 2:
      if (FTy->getParamType(0) == Type::Int32Ty &&
          isa<PointerType>(FTy->getParamType(1))) {
        int (*PF)(int, char **) = (int(*)(int, char **))(intptr_t)FPtr;

        // Call the function.
        GenericValue rv;
        rv.IntVal = APInt(32, PF(ArgValues[0].IntVal.getZExtValue(), 
                                 (char **)GVTOP(ArgValues[1])));
        return rv;
      }
      break;
    case 1:
      if (FTy->getNumParams() == 1 &&
          FTy->getParamType(0) == Type::Int32Ty) {
        GenericValue rv;
        int (*PF)(int) = (int(*)(int))(intptr_t)FPtr;
        rv.IntVal = APInt(32, PF(ArgValues[0].IntVal.getZExtValue()));
        return rv;
      }
      break;
    }
  }

  // Handle cases where no arguments are passed first.
  if (ArgValues.empty()) {
    GenericValue rv;
    switch (RetTy->getTypeID()) {
    default: assert(0 && "Unknown return type for function call!");
    case Type::IntegerTyID: {
      unsigned BitWidth = cast<IntegerType>(RetTy)->getBitWidth();
      if (BitWidth == 1)
        rv.IntVal = APInt(BitWidth, ((bool(*)())(intptr_t)FPtr)());
      else if (BitWidth <= 8)
        rv.IntVal = APInt(BitWidth, ((char(*)())(intptr_t)FPtr)());
      else if (BitWidth <= 16)
        rv.IntVal = APInt(BitWidth, ((short(*)())(intptr_t)FPtr)());
      else if (BitWidth <= 32)
        rv.IntVal = APInt(BitWidth, ((int(*)())(intptr_t)FPtr)());
      else if (BitWidth <= 64)
        rv.IntVal = APInt(BitWidth, ((int64_t(*)())(intptr_t)FPtr)());
      else 
        assert(0 && "Integer types > 64 bits not supported");
      return rv;
    }
    case Type::VoidTyID:
      rv.IntVal = APInt(32, ((int(*)())(intptr_t)FPtr)());
      return rv;
    case Type::FloatTyID:
      rv.FloatVal = ((float(*)())(intptr_t)FPtr)();
      return rv;
    case Type::DoubleTyID:
      rv.DoubleVal = ((double(*)())(intptr_t)FPtr)();
      return rv;
    case Type::X86_FP80TyID:
    case Type::FP128TyID:
    case Type::PPC_FP128TyID:
      assert(0 && "long double not supported yet");
      return rv;
    case Type::PointerTyID:
      return PTOGV(((void*(*)())(intptr_t)FPtr)());
    }
  }

  // Okay, this is not one of our quick and easy cases.  Because we don't have a
  // full FFI, we have to codegen a nullary stub function that just calls the
  // function we are interested in, passing in constants for all of the
  // arguments.  Make this function and return.

  // First, create the function.
  FunctionType *STy=FunctionType::get(RetTy, std::vector<const Type*>(), false);
  Function *Stub = Function::Create(STy, Function::InternalLinkage, "",
                                    F->getParent());

  // Insert a basic block.
  BasicBlock *StubBB = BasicBlock::Create("", Stub);

  // Convert all of the GenericValue arguments over to constants.  Note that we
  // currently don't support varargs.
  SmallVector<Value*, 8> Args;
  for (unsigned i = 0, e = ArgValues.size(); i != e; ++i) {
    Constant *C = 0;
    const Type *ArgTy = FTy->getParamType(i);
    const GenericValue &AV = ArgValues[i];
    switch (ArgTy->getTypeID()) {
    default: assert(0 && "Unknown argument type for function call!");
    case Type::IntegerTyID:
        C = ConstantInt::get(AV.IntVal);
        break;
    case Type::FloatTyID:
        C = ConstantFP::get(APFloat(AV.FloatVal));
        break;
    case Type::DoubleTyID:
        C = ConstantFP::get(APFloat(AV.DoubleVal));
        break;
    case Type::PPC_FP128TyID:
    case Type::X86_FP80TyID:
    case Type::FP128TyID:
        C = ConstantFP::get(APFloat(AV.IntVal));
        break;
    case Type::PointerTyID:
      void *ArgPtr = GVTOP(AV);
      if (sizeof(void*) == 4)
        C = ConstantInt::get(Type::Int32Ty, (int)(intptr_t)ArgPtr);
      else
        C = ConstantInt::get(Type::Int64Ty, (intptr_t)ArgPtr);
      C = ConstantExpr::getIntToPtr(C, ArgTy);  // Cast the integer to pointer
      break;
    }
    Args.push_back(C);
  }

  CallInst *TheCall = CallInst::Create(F, Args.begin(), Args.end(),
                                       "", StubBB);
  TheCall->setTailCall();
  if (TheCall->getType() != Type::VoidTy)
    ReturnInst::Create(TheCall, StubBB);    // Return result of the call.
  else
    ReturnInst::Create(StubBB);             // Just return void.

  // Finally, return the value returned by our nullary stub function.
  return runFunction(Stub, std::vector<GenericValue>());
}

/// runJITOnFunction - Run the FunctionPassManager full of
/// just-in-time compilation passes on F, hopefully filling in
/// GlobalAddress[F] with the address of F's machine code.
///
void JIT::runJITOnFunction(Function *F) {
  static bool isAlreadyCodeGenerating = false;

  MutexGuard locked(lock);
  assert(!isAlreadyCodeGenerating && "Error: Recursive compilation detected!");

  // JIT the function
  isAlreadyCodeGenerating = true;
  jitstate->getPM(locked).run(*F);
  isAlreadyCodeGenerating = false;

  // If the function referred to a global variable that had not yet been
  // emitted, it allocates memory for the global, but doesn't emit it yet.  Emit
  // all of these globals now.
  while (!jitstate->getPendingGlobals(locked).empty()) {
    const GlobalVariable *GV = jitstate->getPendingGlobals(locked).back();
    jitstate->getPendingGlobals(locked).pop_back();
    EmitGlobalVariable(GV);
  }
}

/// getPointerToFunction - This method is used to get the address of the
/// specified function, compiling it if neccesary.
///
void *JIT::getPointerToFunction(Function *F) {

  if (void *Addr = getPointerToGlobalIfAvailable(F))
    return Addr;   // Check if function already code gen'd

  // Make sure we read in the function if it exists in this Module.
  if (F->hasNotBeenReadFromBitcode()) {
    // Determine the module provider this function is provided by.
    Module *M = F->getParent();
    ModuleProvider *MP = 0;
    for (unsigned i = 0, e = Modules.size(); i != e; ++i) {
      if (Modules[i]->getModule() == M) {
        MP = Modules[i];
        break;
      }
    }
    assert(MP && "Function isn't in a module we know about!");
    
    std::string ErrorMsg;
    if (MP->materializeFunction(F, &ErrorMsg)) {
      cerr << "Error reading function '" << F->getName()
           << "' from bitcode file: " << ErrorMsg << "\n";
      abort();
    }
  }
  
  if (void *Addr = getPointerToGlobalIfAvailable(F)) {
    return Addr;
  }

  MutexGuard locked(lock);
  
  if (F->isDeclaration()) {
    void *Addr = getPointerToNamedFunction(F->getName());
    addGlobalMapping(F, Addr);
    return Addr;
  }

  runJITOnFunction(F);

  void *Addr = getPointerToGlobalIfAvailable(F);
  assert(Addr && "Code generation didn't add function to GlobalAddress table!");
  return Addr;
}

/// getOrEmitGlobalVariable - Return the address of the specified global
/// variable, possibly emitting it to memory if needed.  This is used by the
/// Emitter.
void *JIT::getOrEmitGlobalVariable(const GlobalVariable *GV) {
  MutexGuard locked(lock);

  void *Ptr = getPointerToGlobalIfAvailable(GV);
  if (Ptr) return Ptr;

  // If the global is external, just remember the address.
  if (GV->isDeclaration()) {
#if HAVE___DSO_HANDLE
    if (GV->getName() == "__dso_handle")
      return (void*)&__dso_handle;
#endif
    Ptr = sys::DynamicLibrary::SearchForAddressOfSymbol(GV->getName().c_str());
    if (Ptr == 0) {
      cerr << "Could not resolve external global address: "
           << GV->getName() << "\n";
      abort();
    }
  } else {
    // If the global hasn't been emitted to memory yet, allocate space.  We will
    // actually initialize the global after current function has finished
    // compilation.
    const Type *GlobalType = GV->getType()->getElementType();
    size_t S = getTargetData()->getABITypeSize(GlobalType);
    size_t A = getTargetData()->getPreferredAlignment(GV);
    if (A <= 8) {
      Ptr = malloc(S);
    } else {
      // Allocate S+A bytes of memory, then use an aligned pointer within that
      // space.
      Ptr = malloc(S+A);
      unsigned MisAligned = ((intptr_t)Ptr & (A-1));
      Ptr = (char*)Ptr + (MisAligned ? (A-MisAligned) : 0);
    }
    jitstate->getPendingGlobals(locked).push_back(GV);
  }
  addGlobalMapping(GV, Ptr);
  return Ptr;
}


/// recompileAndRelinkFunction - This method is used to force a function
/// which has already been compiled, to be compiled again, possibly
/// after it has been modified. Then the entry to the old copy is overwritten
/// with a branch to the new copy. If there was no old copy, this acts
/// just like JIT::getPointerToFunction().
///
void *JIT::recompileAndRelinkFunction(Function *F) {
  void *OldAddr = getPointerToGlobalIfAvailable(F);

  // If it's not already compiled there is no reason to patch it up.
  if (OldAddr == 0) { return getPointerToFunction(F); }

  // Delete the old function mapping.
  addGlobalMapping(F, 0);

  // Recodegen the function
  runJITOnFunction(F);

  // Update state, forward the old function to the new function.
  void *Addr = getPointerToGlobalIfAvailable(F);
  assert(Addr && "Code generation didn't add function to GlobalAddress table!");
  TJI.replaceMachineCodeForFunction(OldAddr, Addr);
  return Addr;
}

