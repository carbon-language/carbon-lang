//===-- VM.h - Definitions for Virtual Machine ------------------*- C++ -*-===//
//
// This file defines the top-level Virtual Machine data structure.
//
//===----------------------------------------------------------------------===//

#ifndef VM_H
#define VM_H

#include "../ExecutionEngine.h"
#include "llvm/PassManager.h"
#include <map>

class Function;
class GlobalValue;
class Constant;
class TargetMachine;
class MachineCodeEmitter;

class VM : public ExecutionEngine {
  TargetMachine &TM;       // The current target we are compiling to
  PassManager PM;          // Passes to compile a function
  MachineCodeEmitter *MCE; // MCE object

  // FunctionRefs - A mapping between addresses that refer to unresolved
  // functions and the LLVM function object itself.  This is used by the fault
  // handler to lazily patch up references...
  //
  std::map<void*, Function*> FunctionRefs;
public:
  VM(Module *M, TargetMachine *tm);
  ~VM();

  /// run - Start execution with the specified function and arguments.
  ///
  virtual int run(const std::string &FnName,
		  const std::vector<std::string> &Args);

  void addFunctionRef(void *Ref, Function *F) {
    FunctionRefs[Ref] = F;
  }

  const std::string &getFunctionReferencedName(void *RefAddr);

  void *resolveFunctionReference(void *RefAddr);

  /// getPointerToNamedFunction - This method returns the address of the
  /// specified function by using the dlsym function call.  As such it is only
  /// useful for resolving library symbols, not code generated symbols.
  ///
  void *getPointerToNamedFunction(const std::string &Name);

private:
  static MachineCodeEmitter *createEmitter(VM &V);
  void setupPassManager();
  void *getPointerToFunction(const Function *F);
  void registerCallback();
};

#endif
