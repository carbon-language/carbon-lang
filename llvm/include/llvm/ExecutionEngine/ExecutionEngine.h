//===- ExecutionEngine.h - Abstract Execution Engine Interface --*- C++ -*-===//
//
// This file defines the abstract interface that implements execution support
// for LLVM.
//
//===----------------------------------------------------------------------===//

#ifndef EXECUTION_ENGINE_H
#define EXECUTION_ENGINE_H

#include <vector>
#include <string>
#include <map>
#include <cassert>
class Constant;
class Type;
class GlobalValue;
class Function;
class Module;
class TargetData;
union GenericValue;

class ExecutionEngine {
  Module &CurMod;
  const TargetData *TD;

protected:
  // GlobalAddress - A mapping between LLVM global values and their actualized
  // version...
  std::map<const GlobalValue*, void *> GlobalAddress;

  void setTargetData(const TargetData &td) {
    TD = &td;
  }
public:
  ExecutionEngine(Module *M) : CurMod(*M) {
    assert(M && "Module is null?");
  }
  virtual ~ExecutionEngine();
  
  Module &getModule() const { return CurMod; }
  const TargetData &getTargetData() const { return *TD; }

  /// run - Start execution with the specified function, arguments, and
  ///       environment.
  ///
  virtual int run(const std::string &FnName,
                  const std::vector<std::string> &Args,
                  const char ** envp) = 0;

  /// createJIT - Create an return a new JIT compiler if there is one available
  /// for the current target.  Otherwise it returns null.
  ///
  static ExecutionEngine *createJIT(Module *M);

  /// createInterpreter - Create a new interpreter object.  This can never fail.
  ///
  static ExecutionEngine *createInterpreter(Module *M, bool DebugMode,
                                            bool TraceMode);

  void addGlobalMapping(const Function *F, void *Addr) {
    void *&CurVal = GlobalAddress[(const GlobalValue*)F];
    assert(CurVal == 0 && "GlobalMapping already established!");
    CurVal = Addr;
  }

  // getPointerToGlobalIfAvailable - This returns the address of the specified
  // global value if it is available, otherwise it returns null.
  //
  void *getPointerToGlobalIfAvailable(const GlobalValue *GV) {
    std::map<const GlobalValue*, void*>::iterator I = GlobalAddress.find(GV);
    return I != GlobalAddress.end() ? I->second : 0;
  }

  // getPointerToGlobal - This returns the address of the specified global
  // value.  This may involve code generation if it's a function.
  //
  void *getPointerToGlobal(const GlobalValue *GV);

  // getPointerToFunction - The different EE's represent function bodies in
  // different ways.  They should each implement this to say what a function
  // pointer should look like.
  //
  virtual void *getPointerToFunction(Function *F) = 0;

protected:
  void emitGlobals();

public:   // FIXME: protected:   // API shared among subclasses
  GenericValue getConstantValue(const Constant *C);
  void StoreValueToMemory(GenericValue Val, GenericValue *Ptr, const Type *Ty);
  GenericValue LoadValueFromMemory(GenericValue *Ptr, const Type *Ty);
  void *CreateArgv(const std::vector<std::string> &InputArgv);
  void InitializeMemory(const Constant *Init, void *Addr);
};

#endif
