//===-- JIT.h - Class definition for the JIT --------------------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines the top-level JIT data structure.
//
//===----------------------------------------------------------------------===//

#ifndef JIT_H
#define JIT_H

#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/PassManager.h"
#include <map>

namespace llvm {

class Function;
class GlobalValue;
class Constant;
class TargetMachine;
class TargetJITInfo;
class MachineCodeEmitter;

class JIT : public ExecutionEngine {
  TargetMachine &TM;       // The current target we are compiling to
  TargetJITInfo &TJI;      // The JITInfo for the target we are compiling to
  
  FunctionPassManager PM;  // Passes to compile a function
  MachineCodeEmitter *MCE; // MCE object

  /// PendingGlobals - Global variables which have had memory allocated for them
  /// while a function was code generated, but which have not been initialized
  /// yet.
  std::vector<const GlobalVariable*> PendingGlobals;

  JIT(ModuleProvider *MP, TargetMachine &tm, TargetJITInfo &tji);
public:
  ~JIT();

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
  ///
  void *getPointerToFunction(Function *F);

  /// getOrEmitGlobalVariable - Return the address of the specified global
  /// variable, possibly emitting it to memory if needed.  This is used by the
  /// Emitter.
  void *getOrEmitGlobalVariable(const GlobalVariable *GV);

  /// getPointerToFunctionOrStub - If the specified function has been
  /// code-gen'd, return a pointer to the function.  If not, compile it, or use
  /// a stub to implement lazy compilation if available.
  ///
  void *getPointerToFunctionOrStub(Function *F);

  /// recompileAndRelinkFunction - This method is used to force a function
  /// which has already been compiled, to be compiled again, possibly
  /// after it has been modified. Then the entry to the old copy is overwritten
  /// with a branch to the new copy. If there was no old copy, this acts
  /// just like JIT::getPointerToFunction().
  ///
  void *recompileAndRelinkFunction(Function *F);

private:
  static MachineCodeEmitter *createEmitter(JIT &J);
  void runJITOnFunction (Function *F);
};

} // End llvm namespace

#endif
