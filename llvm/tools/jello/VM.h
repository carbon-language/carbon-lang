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

  std::map<const GlobalValue*, void *> GlobalAddress;
public:
  VM(const std::string &name, Module &m, TargetMachine &tm)
    : ExeName(name), M(m), TM(tm) {
    MCE = createEmitter(*this);  // Initialize MCE
    setupPassManager();
  }

  ~VM();

  int run(Function *F);

  void addGlobalMapping(const Function *F, void *Addr) {
    void *&CurVal = GlobalAddress[(const GlobalValue*)F];
    assert(CurVal == 0 && "GlobalMapping already established!");
    CurVal = Addr;
  }

private:
  static MachineCodeEmitter *createEmitter(VM &V);
  void setupPassManager();
  void *getPointerToFunction(Function *F);
};

#endif
