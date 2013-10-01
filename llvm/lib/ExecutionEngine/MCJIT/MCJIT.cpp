//===-- MCJIT.cpp - MC-based Just-in-Time Compiler ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MCJIT.h"
#include "llvm/ExecutionEngine/GenericValue.h"
#include "llvm/ExecutionEngine/JITEventListener.h"
#include "llvm/ExecutionEngine/JITMemoryManager.h"
#include "llvm/ExecutionEngine/MCJIT.h"
#include "llvm/ExecutionEngine/ObjectBuffer.h"
#include "llvm/ExecutionEngine/ObjectImage.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/MutexGuard.h"

using namespace llvm;

namespace {

static struct RegisterJIT {
  RegisterJIT() { MCJIT::Register(); }
} JITRegistrator;

}

extern "C" void LLVMLinkInMCJIT() {
}

ExecutionEngine *MCJIT::createJIT(Module *M,
                                  std::string *ErrorStr,
                                  RTDyldMemoryManager *MemMgr,
                                  bool GVsWithCode,
                                  TargetMachine *TM) {
  // Try to register the program as a source of symbols to resolve against.
  //
  // FIXME: Don't do this here.
  sys::DynamicLibrary::LoadLibraryPermanently(0, NULL);

  return new MCJIT(M, TM, MemMgr ? MemMgr : new SectionMemoryManager(),
                   GVsWithCode);
}

MCJIT::MCJIT(Module *m, TargetMachine *tm, RTDyldMemoryManager *MM,
             bool AllocateGVsWithCode)
  : ExecutionEngine(m), TM(tm), Ctx(0), MemMgr(this, MM), Dyld(&MemMgr),
    ObjCache(0) {

  ModuleStates[m] = ModuleAdded;
  setDataLayout(TM->getDataLayout());
}

MCJIT::~MCJIT() {
  LoadedObjectMap::iterator it, end = LoadedObjects.end();
  for (it = LoadedObjects.begin(); it != end; ++it) {
    ObjectImage *Obj = it->second;
    if (Obj) {
      NotifyFreeingObject(*Obj);
      delete Obj;
    }
  }
  LoadedObjects.clear();
  delete TM;
}

void MCJIT::addModule(Module *M) {
  Modules.push_back(M);
  ModuleStates[M] = MCJITModuleState();
}

void MCJIT::setObjectCache(ObjectCache* NewCache) {
  ObjCache = NewCache;
}

ObjectBufferStream* MCJIT::emitObject(Module *M) {
  // This must be a module which has already been added to this MCJIT instance.
  assert(std::find(Modules.begin(), Modules.end(), M) != Modules.end());
  assert(ModuleStates.find(M) != ModuleStates.end());

  // Get a thread lock to make sure we aren't trying to compile multiple times
  MutexGuard locked(lock);

  // Re-compilation is not supported
  assert(!ModuleStates[M].hasBeenEmitted());

  PassManager PM;

  PM.add(new DataLayout(*TM->getDataLayout()));

  // The RuntimeDyld will take ownership of this shortly
  OwningPtr<ObjectBufferStream> CompiledObject(new ObjectBufferStream());

  // Turn the machine code intermediate representation into bytes in memory
  // that may be executed.
  if (TM->addPassesToEmitMC(PM, Ctx, CompiledObject->getOStream(), false)) {
    report_fatal_error("Target does not support MC emission!");
  }

  // Initialize passes.
  PM.run(*M);
  // Flush the output buffer to get the generated code into memory
  CompiledObject->flush();

  // If we have an object cache, tell it about the new object.
  // Note that we're using the compiled image, not the loaded image (as below).
  if (ObjCache) {
    // MemoryBuffer is a thin wrapper around the actual memory, so it's OK
    // to create a temporary object here and delete it after the call.
    OwningPtr<MemoryBuffer> MB(CompiledObject->getMemBuffer());
    ObjCache->notifyObjectCompiled(M, MB.get());
  }

  return CompiledObject.take();
}

void MCJIT::generateCodeForModule(Module *M) {
  // This must be a module which has already been added to this MCJIT instance.
  assert(std::find(Modules.begin(), Modules.end(), M) != Modules.end());
  assert(ModuleStates.find(M) != ModuleStates.end());

  // Get a thread lock to make sure we aren't trying to load multiple times
  MutexGuard locked(lock);

  // Re-compilation is not supported
  if (ModuleStates[M].hasBeenLoaded())
    return;

  OwningPtr<ObjectBuffer> ObjectToLoad;
  // Try to load the pre-compiled object from cache if possible
  if (0 != ObjCache) {
    OwningPtr<MemoryBuffer> PreCompiledObject(ObjCache->getObject(M));
    if (0 != PreCompiledObject.get())
      ObjectToLoad.reset(new ObjectBuffer(PreCompiledObject.take()));
  }

  // If the cache did not contain a suitable object, compile the object
  if (!ObjectToLoad) {
    ObjectToLoad.reset(emitObject(M));
    assert(ObjectToLoad.get() && "Compilation did not produce an object.");
  }

  // Load the object into the dynamic linker.
  // MCJIT now owns the ObjectImage pointer (via its LoadedObjects map).
  ObjectImage *LoadedObject = Dyld.loadObject(ObjectToLoad.take());
  LoadedObjects[M] = LoadedObject;
  if (!LoadedObject)
    report_fatal_error(Dyld.getErrorString());

  // FIXME: Make this optional, maybe even move it to a JIT event listener
  LoadedObject->registerWithDebugger();

  NotifyObjectEmitted(*LoadedObject);

  ModuleStates[M] = ModuleLoaded;
}

void MCJIT::finalizeLoadedModules() {
  // Resolve any outstanding relocations.
  Dyld.resolveRelocations();

  // Register EH frame data for any module we own which has been loaded
  SmallVector<Module *, 1>::iterator end = Modules.end();
  SmallVector<Module *, 1>::iterator it;
  for (it = Modules.begin(); it != end; ++it) {
    Module *M = *it;
    assert(ModuleStates.find(M) != ModuleStates.end());

    if (ModuleStates[M].hasBeenLoaded() &&
        !ModuleStates[M].hasBeenFinalized()) {
      // FIXME: This should be module specific!
      StringRef EHData = Dyld.getEHFrameSection();
      if (!EHData.empty())
        MemMgr.registerEHFrames(EHData);
      ModuleStates[M] = ModuleFinalized;
    }
  }

  // Set page permissions.
  MemMgr.finalizeMemory();
}

// FIXME: Rename this.
void MCJIT::finalizeObject() {
  // FIXME: This is a temporary hack to get around problems with calling
  // finalize multiple times.
  bool finalizeNeeded = false;
  SmallVector<Module *, 1>::iterator end = Modules.end();
  SmallVector<Module *, 1>::iterator it;
  for (it = Modules.begin(); it != end; ++it) {
    Module *M = *it;
    assert(ModuleStates.find(M) != ModuleStates.end());
    if (!ModuleStates[M].hasBeenFinalized())
      finalizeNeeded = true;

    // I don't really like this, but the C API depends on this behavior.
    // I suppose it's OK for a deprecated function.
    if (!ModuleStates[M].hasBeenLoaded())
      generateCodeForModule(M);
  }
  if (!finalizeNeeded)
    return;

  // Resolve any outstanding relocations.
  Dyld.resolveRelocations();

  // Register EH frame data for any module we own which has been loaded
  for (it = Modules.begin(); it != end; ++it) {
    Module *M = *it;
    assert(ModuleStates.find(M) != ModuleStates.end());

    if (ModuleStates[M].hasBeenLoaded() &&
        !ModuleStates[M].hasBeenFinalized()) {
      // FIXME: This should be module specific!
      StringRef EHData = Dyld.getEHFrameSection();
      if (!EHData.empty())
        MemMgr.registerEHFrames(EHData);
      ModuleStates[M] = ModuleFinalized;
    }
  }

  // Set page permissions.
  MemMgr.finalizeMemory();
}

void MCJIT::finalizeModule(Module *M) {
  // This must be a module which has already been added to this MCJIT instance.
  assert(std::find(Modules.begin(), Modules.end(), M) != Modules.end());
  assert(ModuleStates.find(M) != ModuleStates.end());

  if (ModuleStates[M].hasBeenFinalized())
    return;

  // If the module hasn't been compiled, just do that.
  if (!ModuleStates[M].hasBeenLoaded())
    generateCodeForModule(M);

  // Resolve any outstanding relocations.
  Dyld.resolveRelocations();

  // FIXME: Should this be module specific?
  StringRef EHData = Dyld.getEHFrameSection();
  if (!EHData.empty())
    MemMgr.registerEHFrames(EHData);

  // Set page permissions.
  MemMgr.finalizeMemory();

  ModuleStates[M] = ModuleFinalized;
}

void *MCJIT::getPointerToBasicBlock(BasicBlock *BB) {
  report_fatal_error("not yet implemented");
}

uint64_t MCJIT::getExistingSymbolAddress(const std::string &Name) {
  // Check with the RuntimeDyld to see if we already have this symbol.
  if (Name[0] == '\1')
    return Dyld.getSymbolLoadAddress(Name.substr(1));
  return Dyld.getSymbolLoadAddress((TM->getMCAsmInfo()->getGlobalPrefix()
                                       + Name));
}

Module *MCJIT::findModuleForSymbol(const std::string &Name,
                                   bool CheckFunctionsOnly) {
  // If it hasn't already been generated, see if it's in one of our modules.
  SmallVector<Module *, 1>::iterator end = Modules.end();
  SmallVector<Module *, 1>::iterator it;
  for (it = Modules.begin(); it != end; ++it) {
    Module *M = *it;
    Function *F = M->getFunction(Name);
    if (F && !F->empty())
      return M;
    if (!CheckFunctionsOnly) {
      GlobalVariable *G = M->getGlobalVariable(Name);
      if (G)
        return M;
      // FIXME: Do we need to worry about global aliases?
    }
  }
  // We didn't find the symbol in any of our modules.
  return NULL;
}

uint64_t MCJIT::getSymbolAddress(const std::string &Name,
                                 bool CheckFunctionsOnly)
{
  // First, check to see if we already have this symbol.
  uint64_t Addr = getExistingSymbolAddress(Name);
  if (Addr)
    return Addr;

  // If it hasn't already been generated, see if it's in one of our modules.
  Module *M = findModuleForSymbol(Name, CheckFunctionsOnly);
  if (!M)
    return 0;

  // If this is in one of our modules, generate code for that module.
  assert(ModuleStates.find(M) != ModuleStates.end());
  // If the module code has already been generated, we won't find the symbol.
  if (ModuleStates[M].hasBeenLoaded())
    return 0;

  // FIXME: We probably need to make sure we aren't in the process of
  //        loading or finalizing this module.
  generateCodeForModule(M);

  // Check the RuntimeDyld table again, it should be there now.
  return getExistingSymbolAddress(Name);
}

uint64_t MCJIT::getGlobalValueAddress(const std::string &Name) {
  uint64_t Result = getSymbolAddress(Name, false);
  if (Result != 0)
    finalizeLoadedModules();
  return Result;
}

uint64_t MCJIT::getFunctionAddress(const std::string &Name) {
  uint64_t Result = getSymbolAddress(Name, true);
  if (Result != 0)
    finalizeLoadedModules();
  return Result;
}

// Deprecated.  Use getFunctionAddress instead.
void *MCJIT::getPointerToFunction(Function *F) {

  if (F->isDeclaration() || F->hasAvailableExternallyLinkage()) {
    bool AbortOnFailure = !F->hasExternalWeakLinkage();
    void *Addr = getPointerToNamedFunction(F->getName(), AbortOnFailure);
    addGlobalMapping(F, Addr);
    return Addr;
  }

  // If this function doesn't belong to one of our modules, we're done.
  Module *M = F->getParent();
  if (std::find(Modules.begin(), Modules.end(), M) == Modules.end())
    return NULL;

  assert(ModuleStates.find(M) != ModuleStates.end());

  // Make sure the relevant module has been compiled and loaded.
  if (!ModuleStates[M].hasBeenLoaded())
    generateCodeForModule(M);

  // FIXME: Should the Dyld be retaining module information? Probably not.
  // FIXME: Should we be using the mangler for this? Probably.
  //
  // This is the accessor for the target address, so make sure to check the
  // load address of the symbol, not the local address.
  StringRef BaseName = F->getName();
  if (BaseName[0] == '\1')
    return (void*)Dyld.getSymbolLoadAddress(BaseName.substr(1));
  return (void*)Dyld.getSymbolLoadAddress((TM->getMCAsmInfo()->getGlobalPrefix()
                                       + BaseName).str());
}

void *MCJIT::recompileAndRelinkFunction(Function *F) {
  report_fatal_error("not yet implemented");
}

void MCJIT::freeMachineCodeForFunction(Function *F) {
  report_fatal_error("not yet implemented");
}

GenericValue MCJIT::runFunction(Function *F,
                                const std::vector<GenericValue> &ArgValues) {
  assert(F && "Function *F was null at entry to run()");

  void *FPtr = getPointerToFunction(F);
  assert(FPtr && "Pointer to fn's code was null after getPointerToFunction");
  FunctionType *FTy = F->getFunctionType();
  Type *RetTy = FTy->getReturnType();

  assert((FTy->getNumParams() == ArgValues.size() ||
          (FTy->isVarArg() && FTy->getNumParams() <= ArgValues.size())) &&
         "Wrong number of arguments passed into function!");
  assert(FTy->getNumParams() == ArgValues.size() &&
         "This doesn't support passing arguments through varargs (yet)!");

  // Handle some common cases first.  These cases correspond to common `main'
  // prototypes.
  if (RetTy->isIntegerTy(32) || RetTy->isVoidTy()) {
    switch (ArgValues.size()) {
    case 3:
      if (FTy->getParamType(0)->isIntegerTy(32) &&
          FTy->getParamType(1)->isPointerTy() &&
          FTy->getParamType(2)->isPointerTy()) {
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
      if (FTy->getParamType(0)->isIntegerTy(32) &&
          FTy->getParamType(1)->isPointerTy()) {
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
          FTy->getParamType(0)->isIntegerTy(32)) {
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
    default: llvm_unreachable("Unknown return type for function call!");
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
        llvm_unreachable("Integer types > 64 bits not supported");
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
      llvm_unreachable("long double not supported yet");
    case Type::PointerTyID:
      return PTOGV(((void*(*)())(intptr_t)FPtr)());
    }
  }

  llvm_unreachable("Full-featured argument passing not supported yet!");
}

void *MCJIT::getPointerToNamedFunction(const std::string &Name,
                                       bool AbortOnFailure) {
  if (!isSymbolSearchingDisabled()) {
    void *ptr = MemMgr.getPointerToNamedFunction(Name, false);
    if (ptr)
      return ptr;
  }

  /// If a LazyFunctionCreator is installed, use it to get/create the function.
  if (LazyFunctionCreator)
    if (void *RP = LazyFunctionCreator(Name))
      return RP;

  if (AbortOnFailure) {
    report_fatal_error("Program used external function '"+Name+
                       "' which could not be resolved!");
  }
  return 0;
}

void MCJIT::RegisterJITEventListener(JITEventListener *L) {
  if (L == NULL)
    return;
  MutexGuard locked(lock);
  EventListeners.push_back(L);
}
void MCJIT::UnregisterJITEventListener(JITEventListener *L) {
  if (L == NULL)
    return;
  MutexGuard locked(lock);
  SmallVector<JITEventListener*, 2>::reverse_iterator I=
      std::find(EventListeners.rbegin(), EventListeners.rend(), L);
  if (I != EventListeners.rend()) {
    std::swap(*I, EventListeners.back());
    EventListeners.pop_back();
  }
}
void MCJIT::NotifyObjectEmitted(const ObjectImage& Obj) {
  MutexGuard locked(lock);
  for (unsigned I = 0, S = EventListeners.size(); I < S; ++I) {
    EventListeners[I]->NotifyObjectEmitted(Obj);
  }
}
void MCJIT::NotifyFreeingObject(const ObjectImage& Obj) {
  MutexGuard locked(lock);
  for (unsigned I = 0, S = EventListeners.size(); I < S; ++I) {
    EventListeners[I]->NotifyFreeingObject(Obj);
  }
}

uint64_t LinkingMemoryManager::getSymbolAddress(const std::string &Name) {
  uint64_t Result = ParentEngine->getSymbolAddress(Name, false);
  // If the symbols wasn't found and it begins with an underscore, try again
  // without the underscore.
  if (!Result && Name[0] == '_')
    Result = ParentEngine->getSymbolAddress(Name.substr(1), false);
  if (Result)
    return Result;
  return ClientMM->getSymbolAddress(Name);
}
