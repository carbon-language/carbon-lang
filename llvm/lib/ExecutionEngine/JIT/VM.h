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

public:
  VM(Module *M, TargetMachine *tm);
  ~VM();

  /// run - Start execution with the specified function and arguments.
  ///
  virtual int run(const std::string &FnName,
		  const std::vector<std::string> &Args);

  /// getPointerToNamedFunction - This method returns the address of the
  /// specified function by using the dlsym function call.  As such it is only
  /// useful for resolving library symbols, not code generated symbols.
  ///
  void *getPointerToNamedFunction(const std::string &Name);

  // CompilationCallback - Invoked the first time that a call site is found,
  // which causes lazy compilation of the target function.
  // 
  static void CompilationCallback();

  /// runAtExitHandlers - Before exiting the program, at_exit functions must be
  /// called.  This method calls them.
  ///
  static void runAtExitHandlers();

  /// getPointerToFunction - This returns the address of the specified function,
  /// compiling it if necessary.
  void *getPointerToFunction(const Function *F);

private:
  static MachineCodeEmitter *createEmitter(VM &V);
  void setupPassManager();
};

#endif
