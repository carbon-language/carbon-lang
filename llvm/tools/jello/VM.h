//===-- VM.h - Definitions for Virtual Machine ------------------*- C++ -*-===//
//
// This file defines the top level Virtual Machine data structure.
//
//===----------------------------------------------------------------------===//

#ifndef VM_H
#define VM_H

#include "llvm/PassManager.h"
#include <string>
#include <map>
#include <vector>

class TargetMachine;
class Function;
class GlobalValue;
class MachineCodeEmitter;

class VM {
  std::string ExeName;
  Module &M;               // The LLVM program we are running
  TargetMachine &TM;       // The current target we are compiling to
  PassManager PM;          // Passes to compile a function
  MachineCodeEmitter *MCE; // MCE object

  // GlobalAddress - A mapping between LLVM values and their native code
  // generated versions...
  std::map<const GlobalValue*, void *> GlobalAddress;

  // FunctionRefs - A mapping between addresses that refer to unresolved
  // functions and the LLVM function object itself.  This is used by the fault
  // handler to lazily patch up references...
  //
  std::map<void*, Function*> FunctionRefs;
public:
  VM(const std::string &name, Module &m, TargetMachine &tm)
    : ExeName(name), M(m), TM(tm) {
    MCE = createEmitter(*this);  // Initialize MCE
    setupPassManager();
    registerCallback();
  }

  ~VM();

  int run(Function *F);

  void addGlobalMapping(const Function *F, void *Addr) {
    void *&CurVal = GlobalAddress[(const GlobalValue*)F];
    assert(CurVal == 0 && "GlobalMapping already established!");
    CurVal = Addr;
  }

  void addFunctionRef(void *Ref, Function *F) {
    FunctionRefs[Ref] = F;
  }

  const std::string &getFunctionReferencedName(void *RefAddr);

  void *resolveFunctionReference(void *RefAddr);

private:
  static MachineCodeEmitter *createEmitter(VM &V);
  void setupPassManager();
  void *getPointerToFunction(Function *F);
  void registerCallback();
};

#endif
