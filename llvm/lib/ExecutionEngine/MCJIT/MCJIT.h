//===-- MCJIT.h - Class definition for the MCJIT ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_EXECUTIONENGINE_MCJIT_H
#define LLVM_LIB_EXECUTIONENGINE_MCJIT_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/ObjectCache.h"
#include "llvm/ExecutionEngine/ObjectImage.h"
#include "llvm/ExecutionEngine/RuntimeDyld.h"
#include "llvm/PassManager.h"

namespace llvm {

class MCJIT;

// This is a helper class that the MCJIT execution engine uses for linking
// functions across modules that it owns.  It aggregates the memory manager
// that is passed in to the MCJIT constructor and defers most functionality
// to that object.
class LinkingMemoryManager : public RTDyldMemoryManager {
public:
  LinkingMemoryManager(MCJIT *Parent, RTDyldMemoryManager *MM)
    : ParentEngine(Parent), ClientMM(MM) {}

  virtual uint64_t getSymbolAddress(const std::string &Name);

  // Functions deferred to client memory manager
  virtual uint8_t *allocateCodeSection(uintptr_t Size, unsigned Alignment,
                                       unsigned SectionID, StringRef SectionName) {
    return ClientMM->allocateCodeSection(Size, Alignment, SectionID, SectionName);
  }

  virtual uint8_t *allocateDataSection(uintptr_t Size, unsigned Alignment,
                                       unsigned SectionID, StringRef SectionName,
                                       bool IsReadOnly) {
    return ClientMM->allocateDataSection(Size, Alignment,
                                         SectionID, SectionName, IsReadOnly);
  }

  virtual void registerEHFrames(StringRef SectionData) {
    ClientMM->registerEHFrames(SectionData);
  }

  virtual bool finalizeMemory(std::string *ErrMsg = 0) {
    return ClientMM->finalizeMemory(ErrMsg);
  }

private:
  MCJIT *ParentEngine;
  OwningPtr<RTDyldMemoryManager> ClientMM;
};

// FIXME: This makes all kinds of horrible assumptions for the time being,
// like only having one module, not needing to worry about multi-threading,
// blah blah. Purely in get-it-up-and-limping mode for now.

class MCJIT : public ExecutionEngine {
  MCJIT(Module *M, TargetMachine *tm, RTDyldMemoryManager *MemMgr,
        bool AllocateGVsWithCode);

  enum ModuleState {
    ModuleAdded,
    ModuleEmitted,
    ModuleLoading,
    ModuleLoaded,
    ModuleFinalizing,
    ModuleFinalized
  };

  class MCJITModuleState {
  public:
    MCJITModuleState() : State(ModuleAdded) {}

    MCJITModuleState & operator=(ModuleState s) { State = s; return *this; }
    bool hasBeenEmitted() { return State != ModuleAdded; }
    bool hasBeenLoaded() { return State != ModuleAdded &&
                                  State != ModuleEmitted; }
    bool hasBeenFinalized() { return State == ModuleFinalized; }

  private:
    ModuleState State;
  };

  TargetMachine *TM;
  MCContext *Ctx;
  LinkingMemoryManager MemMgr;
  RuntimeDyld Dyld;
  SmallVector<JITEventListener*, 2> EventListeners;

  typedef DenseMap<Module *, MCJITModuleState> ModuleStateMap;
  ModuleStateMap  ModuleStates;

  typedef DenseMap<Module *, ObjectImage *> LoadedObjectMap;
  LoadedObjectMap  LoadedObjects;

  // An optional ObjectCache to be notified of compiled objects and used to
  // perform lookup of pre-compiled code to avoid re-compilation.
  ObjectCache *ObjCache;

public:
  ~MCJIT();

  /// @name ExecutionEngine interface implementation
  /// @{
  virtual void addModule(Module *M);

  /// Sets the object manager that MCJIT should use to avoid compilation.
  virtual void setObjectCache(ObjectCache *manager);

  virtual void generateCodeForModule(Module *M);

  /// finalizeObject - ensure the module is fully processed and is usable.
  ///
  /// It is the user-level function for completing the process of making the
  /// object usable for execution. It should be called after sections within an
  /// object have been relocated using mapSectionAddress.  When this method is
  /// called the MCJIT execution engine will reapply relocations for a loaded
  /// object.
  /// FIXME: Do we really need both of these?
  virtual void finalizeObject();
  virtual void finalizeModule(Module *);
  void finalizeLoadedModules();

  virtual void *getPointerToBasicBlock(BasicBlock *BB);

  virtual void *getPointerToFunction(Function *F);

  virtual void *recompileAndRelinkFunction(Function *F);

  virtual void freeMachineCodeForFunction(Function *F);

  virtual GenericValue runFunction(Function *F,
                                   const std::vector<GenericValue> &ArgValues);

  /// getPointerToNamedFunction - This method returns the address of the
  /// specified function by using the dlsym function call.  As such it is only
  /// useful for resolving library symbols, not code generated symbols.
  ///
  /// If AbortOnFailure is false and no function with the given name is
  /// found, this function silently returns a null pointer. Otherwise,
  /// it prints a message to stderr and aborts.
  ///
  virtual void *getPointerToNamedFunction(const std::string &Name,
                                          bool AbortOnFailure = true);

  /// mapSectionAddress - map a section to its target address space value.
  /// Map the address of a JIT section as returned from the memory manager
  /// to the address in the target process as the running code will see it.
  /// This is the address which will be used for relocation resolution.
  virtual void mapSectionAddress(const void *LocalAddress,
                                 uint64_t TargetAddress) {
    Dyld.mapSectionAddress(LocalAddress, TargetAddress);
  }
  virtual void RegisterJITEventListener(JITEventListener *L);
  virtual void UnregisterJITEventListener(JITEventListener *L);

  // If successful, these function will implicitly finalize all loaded objects.
  // To get a function address within MCJIT without causing a finalize, use
  // getSymbolAddress.
  virtual uint64_t getGlobalValueAddress(const std::string &Name);
  virtual uint64_t getFunctionAddress(const std::string &Name);

  /// @}
  /// @name (Private) Registration Interfaces
  /// @{

  static void Register() {
    MCJITCtor = createJIT;
  }

  static ExecutionEngine *createJIT(Module *M,
                                    std::string *ErrorStr,
                                    RTDyldMemoryManager *MemMgr,
                                    bool GVsWithCode,
                                    TargetMachine *TM);

  // @}

  // This is not directly exposed via the ExecutionEngine API, but it is
  // used by the LinkingMemoryManager.
  uint64_t getSymbolAddress(const std::string &Name,
                          bool CheckFunctionsOnly);

protected:
  /// emitObject -- Generate a JITed object in memory from the specified module
  /// Currently, MCJIT only supports a single module and the module passed to
  /// this function call is expected to be the contained module.  The module
  /// is passed as a parameter here to prepare for multiple module support in 
  /// the future.
  ObjectBufferStream* emitObject(Module *M);

  void NotifyObjectEmitted(const ObjectImage& Obj);
  void NotifyFreeingObject(const ObjectImage& Obj);

  uint64_t getExistingSymbolAddress(const std::string &Name);
  Module *findModuleForSymbol(const std::string &Name,
                              bool CheckFunctionsOnly);
};

} // End llvm namespace

#endif
