//===-- VM.h - Definitions for Virtual Machine ------------------*- C++ -*-===//
//
// This file defines the top-level Virtual Machine data structure.
//
//===----------------------------------------------------------------------===//

#ifndef VM_H
#define VM_H

#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/PassManager.h"
#include <map>

class Function;
class GlobalValue;
class Constant;
class TargetMachine;
class MachineCodeEmitter;

class VM : public ExecutionEngine {
  TargetMachine &TM;       // The current target we are compiling to
  FunctionPassManager PM;  // Passes to compile a function
  MachineCodeEmitter *MCE; // MCE object

public:
  VM(ModuleProvider *MP, TargetMachine *tm);
  ~VM();

  /// create - Create an return a new JIT compiler if there is one available
  /// for the current target.  Otherwise, return null.
  ///
  static ExecutionEngine *create(ModuleProvider *MP);

  /// run - Start execution with the specified function and arguments.
  ///
  virtual GenericValue run(Function *F,
			   const std::vector<GenericValue> &ArgValues);

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
  void *getPointerToFunction(Function *F);

private:
  static MachineCodeEmitter *createEmitter(VM &V);
  void setupPassManager();
};

#endif
