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

class ExecutionEngine {
  Module &CurMod;
  const TargetData *TD;

  // GlobalAddress - A mapping between LLVM global values and their actualized
  // version...
  std::map<const GlobalValue*, void *> GlobalAddress;
protected:
  ModuleProvider *MP;

  void setTargetData(const TargetData &td) {
    TD = &td;
  }

public:
  ExecutionEngine(ModuleProvider *P);
  ExecutionEngine(Module *M);
  virtual ~ExecutionEngine();
  
  Module &getModule() const { return CurMod; }
  const TargetData &getTargetData() const { return *TD; }

  /// create - This is the factory method for creating an execution engine which
  /// is appropriate for the current machine.
  static ExecutionEngine *create(ModuleProvider *MP, bool ForceInterpreter);

  /// runFunction - Execute the specified function with the specified arguments,
  /// and return the result.
  ///
  virtual GenericValue runFunction(Function *F,
                                const std::vector<GenericValue> &ArgValues) = 0;

  /// runFunctionAsMain - This is a helper function which wraps runFunction to
  /// handle the common task of starting up main with the specified argc, argv,
  /// and envp parameters.
  int runFunctionAsMain(Function *Fn, const std::vector<std::string> &argv,
                        const char * const * envp);


  void addGlobalMapping(const GlobalValue *GV, void *Addr) {
    void *&CurVal = GlobalAddress[GV];
    assert((CurVal == 0 || Addr == 0) && "GlobalMapping already established!");
    CurVal = Addr;
  }

  /// getPointerToGlobalIfAvailable - This returns the address of the specified
  /// global value if it is available, otherwise it returns null.
  ///
  void *getPointerToGlobalIfAvailable(const GlobalValue *GV) {
    std::map<const GlobalValue*, void*>::iterator I = GlobalAddress.find(GV);
    return I != GlobalAddress.end() ? I->second : 0;
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

  void StoreValueToMemory(GenericValue Val, GenericValue *Ptr, const Type *Ty);
  void InitializeMemory(const Constant *Init, void *Addr);

  /// recompileAndRelinkFunction - This method is used to force a function
  /// which has already been compiled to be compiled again, possibly
  /// after it has been modified. Then the entry to the old copy is overwritten
  /// with a branch to the new copy. If there was no old copy, this acts
  /// just like VM::getPointerToFunction().
  ///
  virtual void *recompileAndRelinkFunction(Function *F) = 0;

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
