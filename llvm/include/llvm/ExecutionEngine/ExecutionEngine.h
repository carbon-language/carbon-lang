//===- ExecutionEngine.h - Abstract Execution Engine Interface --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the abstract interface that implements execution support
// for LLVM.
//
//===----------------------------------------------------------------------===//

#ifndef EXECUTION_ENGINE_H
#define EXECUTION_ENGINE_H

#include <vector>
#include <map>
#include <cassert>
#include <string>
#include "llvm/Support/MutexGuard.h"

namespace llvm {

union GenericValue;
class Constant;
class Function;
class GlobalVariable;
class GlobalValue;
class Module;
class ModuleProvider;
class TargetData;
class Type;
class IntrinsicLowering;


class ExecutionEngineState {
private:
  /// GlobalAddressMap - A mapping between LLVM global values and their
  /// actualized version...
  std::map<const GlobalValue*, void *> GlobalAddressMap;

  /// GlobalAddressReverseMap - This is the reverse mapping of GlobalAddressMap,
  /// used to convert raw addresses into the LLVM global value that is emitted
  /// at the address.  This map is not computed unless getGlobalValueAtAddress
  /// is called at some point.
  std::map<void *, const GlobalValue*> GlobalAddressReverseMap;

public:
  std::map<const GlobalValue*, void *>& getGlobalAddressMap(const MutexGuard& locked) {
    return GlobalAddressMap;
  }

  std::map<void *, const GlobalValue*>& getGlobalAddressReverseMap(const MutexGuard& locked) {
    return GlobalAddressReverseMap;
  }
};


class ExecutionEngine {
  Module &CurMod;
  const TargetData *TD;

  ExecutionEngineState state;

protected:
  ModuleProvider *MP;

  void setTargetData(const TargetData &td) {
    TD = &td;
  }

public:
  /// lock - This lock is protects the ExecutionEngine, JIT, JITResolver and JITEmitter classes.
  /// It must be held while changing the internal state of any of those classes.
  sys::Mutex lock; // Used to make this class and subclasses thread-safe

  ExecutionEngine(ModuleProvider *P);
  ExecutionEngine(Module *M);
  virtual ~ExecutionEngine();

  Module &getModule() const { return CurMod; }
  const TargetData &getTargetData() const { return *TD; }

  /// create - This is the factory method for creating an execution engine which
  /// is appropriate for the current machine.  If specified, the
  /// IntrinsicLowering implementation should be allocated on the heap.
  static ExecutionEngine *create(ModuleProvider *MP, bool ForceInterpreter,
                                 IntrinsicLowering *IL = 0);

  /// runFunction - Execute the specified function with the specified arguments,
  /// and return the result.
  ///
  virtual GenericValue runFunction(Function *F,
                                const std::vector<GenericValue> &ArgValues) = 0;

  /// runStaticConstructorsDestructors - This method is used to execute all of
  /// the static constructors or destructors for a module, depending on the
  /// value of isDtors.
  void runStaticConstructorsDestructors(bool isDtors);
  
  
  /// runFunctionAsMain - This is a helper function which wraps runFunction to
  /// handle the common task of starting up main with the specified argc, argv,
  /// and envp parameters.
  int runFunctionAsMain(Function *Fn, const std::vector<std::string> &argv,
                        const char * const * envp);


  void addGlobalMapping(const GlobalValue *GV, void *Addr) {
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
  void clearAllGlobalMappings() {
    MutexGuard locked(lock);

    state.getGlobalAddressMap(locked).clear();
    state.getGlobalAddressReverseMap(locked).clear();
  }

  /// updateGlobalMapping - Replace an existing mapping for GV with a new
  /// address.  This updates both maps as required.
  void updateGlobalMapping(const GlobalValue *GV, void *Addr) {
    MutexGuard locked(lock);

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
  /// global value if it is available, otherwise it returns null.
  ///
  void *getPointerToGlobalIfAvailable(const GlobalValue *GV) {
    MutexGuard locked(lock);

    std::map<const GlobalValue*, void*>::iterator I = state.getGlobalAddressMap(locked).find(GV);
    return I != state.getGlobalAddressMap(locked).end() ? I->second : 0;
  }

  /// getPointerToGlobal - This returns the address of the specified global
  /// value.  This may involve code generation if it's a function.
  ///
  void *getPointerToGlobal(const GlobalValue *GV);

  /// getPointerToFunction - The different EE's represent function bodies in
  /// different ways.  They should each implement this to say what a function
  /// pointer should look like.
  ///
  virtual void *getPointerToFunction(Function *F) = 0;

  /// getPointerToFunctionOrStub - If the specified function has been
  /// code-gen'd, return a pointer to the function.  If not, compile it, or use
  /// a stub to implement lazy compilation if available.
  ///
  virtual void *getPointerToFunctionOrStub(Function *F) {
    // Default implementation, just codegen the function.
    return getPointerToFunction(F);
  }

  /// getGlobalValueAtAddress - Return the LLVM global value object that starts
  /// at the specified address.
  ///
  const GlobalValue *getGlobalValueAtAddress(void *Addr);


  void StoreValueToMemory(GenericValue Val, GenericValue *Ptr, const Type *Ty);
  void InitializeMemory(const Constant *Init, void *Addr);

  /// recompileAndRelinkFunction - This method is used to force a function
  /// which has already been compiled to be compiled again, possibly
  /// after it has been modified. Then the entry to the old copy is overwritten
  /// with a branch to the new copy. If there was no old copy, this acts
  /// just like VM::getPointerToFunction().
  ///
  virtual void *recompileAndRelinkFunction(Function *F) = 0;

  /// freeMachineCodeForFunction - Release memory in the ExecutionEngine
  /// corresponding to the machine code emitted to execute this function, useful
  /// for garbage-collecting generated code.
  ///
  virtual void freeMachineCodeForFunction(Function *F) = 0;

  /// getOrEmitGlobalVariable - Return the address of the specified global
  /// variable, possibly emitting it to memory if needed.  This is used by the
  /// Emitter.
  virtual void *getOrEmitGlobalVariable(const GlobalVariable *GV) {
    return getPointerToGlobal((GlobalValue*)GV);
  }

protected:
  void emitGlobals();

  // EmitGlobalVariable - This method emits the specified global variable to the
  // address specified in GlobalAddresses, or allocates new memory if it's not
  // already in the map.
  void EmitGlobalVariable(const GlobalVariable *GV);

  GenericValue getConstantValue(const Constant *C);
  GenericValue LoadValueFromMemory(GenericValue *Ptr, const Type *Ty);
};

} // End llvm namespace

#endif
