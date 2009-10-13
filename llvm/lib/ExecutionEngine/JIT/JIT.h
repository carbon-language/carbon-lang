//===-- JIT.h - Class definition for the JIT --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "llvm/Support/ValueHandle.h"

namespace llvm {

class Function;
struct JITEvent_EmittedFunctionDetails;
class MachineCodeEmitter;
class MachineCodeInfo;
class TargetJITInfo;
class TargetMachine;

class JITState {
private:
  FunctionPassManager PM;  // Passes to compile a function
  ModuleProvider *MP;      // ModuleProvider used to create the PM

  /// PendingFunctions - Functions which have not been code generated yet, but
  /// were called from a function being code generated.
  std::vector<AssertingVH<Function> > PendingFunctions;

public:
  explicit JITState(ModuleProvider *MP) : PM(MP), MP(MP) {}

  FunctionPassManager &getPM(const MutexGuard &L) {
    return PM;
  }
  
  ModuleProvider *getMP() const { return MP; }
  std::vector<AssertingVH<Function> > &getPendingFunctions(const MutexGuard &L){
    return PendingFunctions;
  }
};


class JIT : public ExecutionEngine {
  TargetMachine &TM;       // The current target we are compiling to
  TargetJITInfo &TJI;      // The JITInfo for the target we are compiling to
  JITCodeEmitter *JCE;     // JCE object
  std::vector<JITEventListener*> EventListeners;

  /// AllocateGVsWithCode - Some applications require that global variables and
  /// code be allocated into the same region of memory, in which case this flag
  /// should be set to true.  Doing so breaks freeMachineCodeForFunction.
  bool AllocateGVsWithCode;

  JITState *jitstate;

  JIT(ModuleProvider *MP, TargetMachine &tm, TargetJITInfo &tji,
      JITMemoryManager *JMM, CodeGenOpt::Level OptLevel,
      bool AllocateGVsWithCode);
public:
  ~JIT();

  static void Register() {
    JITCtor = create;
  }
  
  /// getJITInfo - Return the target JIT information structure.
  ///
  TargetJITInfo &getJITInfo() const { return TJI; }

  /// create - Create an return a new JIT compiler if there is one available
  /// for the current target.  Otherwise, return null.
  ///
  static ExecutionEngine *create(ModuleProvider *MP,
                                 std::string *Err,
                                 JITMemoryManager *JMM,
                                 CodeGenOpt::Level OptLevel =
                                   CodeGenOpt::Default,
                                 bool GVsWithCode = true) {
    return ExecutionEngine::createJIT(MP, Err, JMM, OptLevel, GVsWithCode);
  }

  virtual void addModuleProvider(ModuleProvider *MP);
  
  /// removeModuleProvider - Remove a ModuleProvider from the list of modules.
  /// Relases the Module from the ModuleProvider, materializing it in the
  /// process, and returns the materialized Module.
  virtual Module *removeModuleProvider(ModuleProvider *MP,
                                       std::string *ErrInfo = 0);

  /// deleteModuleProvider - Remove a ModuleProvider from the list of modules,
  /// and deletes the ModuleProvider and owned Module.  Avoids materializing 
  /// the underlying module.
  virtual void deleteModuleProvider(ModuleProvider *P,std::string *ErrInfo = 0);

  /// runFunction - Start execution with the specified function and arguments.
  ///
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
  void *getPointerToNamedFunction(const std::string &Name,
                                  bool AbortOnFailure = true);

  // CompilationCallback - Invoked the first time that a call site is found,
  // which causes lazy compilation of the target function.
  //
  static void CompilationCallback();

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

  /// freeMachineCodeForFunction - deallocate memory used to code-generate this
  /// Function.
  ///
  void freeMachineCodeForFunction(Function *F);

  /// addPendingFunction - while jitting non-lazily, a called but non-codegen'd
  /// function was encountered.  Add it to a pending list to be processed after 
  /// the current function.
  ///
  void addPendingFunction(Function *F);

  /// getCodeEmitter - Return the code emitter this JIT is emitting into.
  ///
  JITCodeEmitter *getCodeEmitter() const { return JCE; }

  /// selectTarget - Pick a target either via -march or by guessing the native
  /// arch.  Add any CPU features specified via -mcpu or -mattr.
  static TargetMachine *selectTarget(ModuleProvider *MP, std::string *Err);

  static ExecutionEngine *createJIT(ModuleProvider *MP,
                                    std::string *ErrorStr,
                                    JITMemoryManager *JMM,
                                    CodeGenOpt::Level OptLevel,
                                    bool GVsWithCode);

  // Run the JIT on F and return information about the generated code
  void runJITOnFunction(Function *F, MachineCodeInfo *MCI = 0);

  virtual void RegisterJITEventListener(JITEventListener *L);
  virtual void UnregisterJITEventListener(JITEventListener *L);
  /// These functions correspond to the methods on JITEventListener.  They
  /// iterate over the registered listeners and call the corresponding method on
  /// each.
  void NotifyFunctionEmitted(
      const Function &F, void *Code, size_t Size,
      const JITEvent_EmittedFunctionDetails &Details);
  void NotifyFreeingMachineCode(const Function &F, void *OldPtr);

private:
  static JITCodeEmitter *createEmitter(JIT &J, JITMemoryManager *JMM,
                                       TargetMachine &tm);
  void runJITOnFunctionUnlocked(Function *F, const MutexGuard &locked);
  void updateFunctionStub(Function *F);
  void updateDlsymStubTable();

protected:

  /// getMemoryforGV - Allocate memory for a global variable.
  virtual char* getMemoryForGV(const GlobalVariable* GV);

};

} // End llvm namespace

#endif
