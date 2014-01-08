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
#include "llvm/IR/Mangler.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/PassManager.h"
#include "llvm/Object/Archive.h"
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

  OwnedModules.addModule(m);
  setDataLayout(TM->getDataLayout());
}

MCJIT::~MCJIT() {
  MutexGuard locked(lock);
  // FIXME: We are managing our modules, so we do not want the base class
  // ExecutionEngine to manage them as well. To avoid double destruction
  // of the first (and only) module added in ExecutionEngine constructor
  // we remove it from EE and will destruct it ourselves.
  //
  // It may make sense to move our module manager (based on SmallStPtr) back
  // into EE if the JIT and Interpreter can live with it.
  // If so, additional functions: addModule, removeModule, FindFunctionNamed,
  // runStaticConstructorsDestructors could be moved back to EE as well.
  //
  Modules.clear();
  Dyld.deregisterEHFrames();

  LoadedObjectList::iterator it, end;
  for (it = LoadedObjects.begin(), end = LoadedObjects.end(); it != end; ++it) {
    ObjectImage *Obj = *it;
    if (Obj) {
      NotifyFreeingObject(*Obj);
      delete Obj;
    }
  }
  LoadedObjects.clear();


  SmallVector<object::Archive *, 2>::iterator ArIt, ArEnd;
  for (ArIt = Archives.begin(), ArEnd = Archives.end(); ArIt != ArEnd; ++ArIt) {
    object::Archive *A = *ArIt;
    delete A;
  }
  Archives.clear();

  delete TM;
}

void MCJIT::addModule(Module *M) {
  MutexGuard locked(lock);
  OwnedModules.addModule(M);
}

bool MCJIT::removeModule(Module *M) {
  MutexGuard locked(lock);
  return OwnedModules.removeModule(M);
}



void MCJIT::addObjectFile(object::ObjectFile *Obj) {
  ObjectImage *LoadedObject = Dyld.loadObject(Obj);
  if (!LoadedObject)
    report_fatal_error(Dyld.getErrorString());

  LoadedObjects.push_back(LoadedObject);

  NotifyObjectEmitted(*LoadedObject);
}

void MCJIT::addArchive(object::Archive *A) {
  Archives.push_back(A);
}


void MCJIT::setObjectCache(ObjectCache* NewCache) {
  MutexGuard locked(lock);
  ObjCache = NewCache;
}

ObjectBufferStream* MCJIT::emitObject(Module *M) {
  MutexGuard locked(lock);

  // This must be a module which has already been added but not loaded to this
  // MCJIT instance, since these conditions are tested by our caller,
  // generateCodeForModule.

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
  // Get a thread lock to make sure we aren't trying to load multiple times
  MutexGuard locked(lock);

  // This must be a module which has already been added to this MCJIT instance.
  assert(OwnedModules.ownsModule(M) &&
         "MCJIT::generateCodeForModule: Unknown module.");

  // Re-compilation is not supported
  if (OwnedModules.hasModuleBeenLoaded(M))
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
  // MCJIT now owns the ObjectImage pointer (via its LoadedObjects list).
  ObjectImage *LoadedObject = Dyld.loadObject(ObjectToLoad.take());
  LoadedObjects.push_back(LoadedObject);
  if (!LoadedObject)
    report_fatal_error(Dyld.getErrorString());

  // FIXME: Make this optional, maybe even move it to a JIT event listener
  LoadedObject->registerWithDebugger();

  NotifyObjectEmitted(*LoadedObject);

  OwnedModules.markModuleAsLoaded(M);
}

void MCJIT::finalizeLoadedModules() {
  MutexGuard locked(lock);

  // Resolve any outstanding relocations.
  Dyld.resolveRelocations();

  OwnedModules.markAllLoadedModulesAsFinalized();

  // Register EH frame data for any module we own which has been loaded
  Dyld.registerEHFrames();

  // Set page permissions.
  MemMgr.finalizeMemory();
}

// FIXME: Rename this.
void MCJIT::finalizeObject() {
  MutexGuard locked(lock);

  for (ModulePtrSet::iterator I = OwnedModules.begin_added(),
                              E = OwnedModules.end_added();
       I != E; ++I) {
    Module *M = *I;
    generateCodeForModule(M);
  }

  finalizeLoadedModules();
}

void MCJIT::finalizeModule(Module *M) {
  MutexGuard locked(lock);

  // This must be a module which has already been added to this MCJIT instance.
  assert(OwnedModules.ownsModule(M) && "MCJIT::finalizeModule: Unknown module.");

  // If the module hasn't been compiled, just do that.
  if (!OwnedModules.hasModuleBeenLoaded(M))
    generateCodeForModule(M);

  finalizeLoadedModules();
}

void *MCJIT::getPointerToBasicBlock(BasicBlock *BB) {
  report_fatal_error("not yet implemented");
}

uint64_t MCJIT::getExistingSymbolAddress(const std::string &Name) {
  Mangler Mang(TM->getDataLayout());
  SmallString<128> FullName;
  Mang.getNameWithPrefix(FullName, Name);
  return Dyld.getSymbolLoadAddress(FullName);
}

Module *MCJIT::findModuleForSymbol(const std::string &Name,
                                   bool CheckFunctionsOnly) {
  MutexGuard locked(lock);

  // If it hasn't already been generated, see if it's in one of our modules.
  for (ModulePtrSet::iterator I = OwnedModules.begin_added(),
                              E = OwnedModules.end_added();
       I != E; ++I) {
    Module *M = *I;
    Function *F = M->getFunction(Name);
    if (F && !F->isDeclaration())
      return M;
    if (!CheckFunctionsOnly) {
      GlobalVariable *G = M->getGlobalVariable(Name);
      if (G && !G->isDeclaration())
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
  MutexGuard locked(lock);

  // First, check to see if we already have this symbol.
  uint64_t Addr = getExistingSymbolAddress(Name);
  if (Addr)
    return Addr;

  SmallVector<object::Archive*, 2>::iterator I, E;
  for (I = Archives.begin(), E = Archives.end(); I != E; ++I) {
    object::Archive *A = *I;
    // Look for our symbols in each Archive
    object::Archive::child_iterator ChildIt = A->findSym(Name);
    if (ChildIt != A->end_children()) {
      OwningPtr<object::Binary> ChildBin;
      // FIXME: Support nested archives?
      if (!ChildIt->getAsBinary(ChildBin) && ChildBin->isObject()) {
        object::ObjectFile *OF = reinterpret_cast<object::ObjectFile *>(
                                                            ChildBin.take());
        // This causes the object file to be loaded.
        addObjectFile(OF);
        // The address should be here now.
        Addr = getExistingSymbolAddress(Name);
        if (Addr)
          return Addr;
      }
    }
  }

  // If it hasn't already been generated, see if it's in one of our modules.
  Module *M = findModuleForSymbol(Name, CheckFunctionsOnly);
  if (!M)
    return 0;

  generateCodeForModule(M);

  // Check the RuntimeDyld table again, it should be there now.
  return getExistingSymbolAddress(Name);
}

uint64_t MCJIT::getGlobalValueAddress(const std::string &Name) {
  MutexGuard locked(lock);
  uint64_t Result = getSymbolAddress(Name, false);
  if (Result != 0)
    finalizeLoadedModules();
  return Result;
}

uint64_t MCJIT::getFunctionAddress(const std::string &Name) {
  MutexGuard locked(lock);
  uint64_t Result = getSymbolAddress(Name, true);
  if (Result != 0)
    finalizeLoadedModules();
  return Result;
}

// Deprecated.  Use getFunctionAddress instead.
void *MCJIT::getPointerToFunction(Function *F) {
  MutexGuard locked(lock);

  if (F->isDeclaration() || F->hasAvailableExternallyLinkage()) {
    bool AbortOnFailure = !F->hasExternalWeakLinkage();
    void *Addr = getPointerToNamedFunction(F->getName(), AbortOnFailure);
    addGlobalMapping(F, Addr);
    return Addr;
  }

  Module *M = F->getParent();
  bool HasBeenAddedButNotLoaded = OwnedModules.hasModuleBeenAddedButNotLoaded(M);

  // Make sure the relevant module has been compiled and loaded.
  if (HasBeenAddedButNotLoaded)
    generateCodeForModule(M);
  else if (!OwnedModules.hasModuleBeenLoaded(M))
    // If this function doesn't belong to one of our modules, we're done.
    return NULL;

  // FIXME: Should the Dyld be retaining module information? Probably not.
  //
  // This is the accessor for the target address, so make sure to check the
  // load address of the symbol, not the local address.
  Mangler Mang(TM->getDataLayout());
  SmallString<128> Name;
  Mang.getNameWithPrefix(Name, F);
  return (void*)Dyld.getSymbolLoadAddress(Name);
}

void *MCJIT::recompileAndRelinkFunction(Function *F) {
  report_fatal_error("not yet implemented");
}

void MCJIT::freeMachineCodeForFunction(Function *F) {
  report_fatal_error("not yet implemented");
}

void MCJIT::runStaticConstructorsDestructorsInModulePtrSet(
    bool isDtors, ModulePtrSet::iterator I, ModulePtrSet::iterator E) {
  for (; I != E; ++I) {
    ExecutionEngine::runStaticConstructorsDestructors(*I, isDtors);
  }
}

void MCJIT::runStaticConstructorsDestructors(bool isDtors) {
  // Execute global ctors/dtors for each module in the program.
  runStaticConstructorsDestructorsInModulePtrSet(
      isDtors, OwnedModules.begin_added(), OwnedModules.end_added());
  runStaticConstructorsDestructorsInModulePtrSet(
      isDtors, OwnedModules.begin_loaded(), OwnedModules.end_loaded());
  runStaticConstructorsDestructorsInModulePtrSet(
      isDtors, OwnedModules.begin_finalized(), OwnedModules.end_finalized());
}

Function *MCJIT::FindFunctionNamedInModulePtrSet(const char *FnName,
                                                 ModulePtrSet::iterator I,
                                                 ModulePtrSet::iterator E) {
  for (; I != E; ++I) {
    if (Function *F = (*I)->getFunction(FnName))
      return F;
  }
  return 0;
}

Function *MCJIT::FindFunctionNamed(const char *FnName) {
  Function *F = FindFunctionNamedInModulePtrSet(
      FnName, OwnedModules.begin_added(), OwnedModules.end_added());
  if (!F)
    F = FindFunctionNamedInModulePtrSet(FnName, OwnedModules.begin_loaded(),
                                        OwnedModules.end_loaded());
  if (!F)
    F = FindFunctionNamedInModulePtrSet(FnName, OwnedModules.begin_finalized(),
                                        OwnedModules.end_finalized());
  return F;
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
  MemMgr.notifyObjectLoaded(this, &Obj);
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
