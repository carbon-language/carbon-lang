//===-- jello.cpp - LLVM Just in Time Compiler ----------------------------===//
//
// This tool implements a just-in-time compiler for LLVM, allowing direct
// execution of LLVM bytecode in an efficient manner.
//
//===----------------------------------------------------------------------===//

#include "VM.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/CodeGen/MachineCodeEmitter.h"
#include "llvm/Function.h"
#include <iostream>


VM::~VM() {
  delete MCE;
}

/// setupPassManager - Initialize the VM PassManager object with all of the
/// passes needed for the target to generate code.
///
void VM::setupPassManager() {
  // Compile LLVM Code down to machine code in the intermediate representation
  if (TM.addPassesToJITCompile(PM)) {
    std::cerr << ExeName << ": target '" << TM.getName()
              << "' doesn't support JIT compilation!\n";
    abort();
  }

  // Turn the machine code intermediate representation into bytes in memory that
  // may be executed.
  //
  if (TM.addPassesToEmitMachineCode(PM, *MCE)) {
    std::cerr << ExeName << ": target '" << TM.getName()
              << "' doesn't support machine code emission!\n";
    abort();
  }
}

int VM::run(Function *F) {
  int(*PF)() = (int(*)())getPointerToFunction(F);
  assert(PF != 0 && "Null pointer to function?");
  return PF();
}

void *VM::resolveFunctionReference(void *RefAddr) {
  Function *F = FunctionRefs[RefAddr];
  assert(F && "Reference address not known!");

  void *Addr = getPointerToFunction(F);
  assert(Addr && "Pointer to function unknown!");

  FunctionRefs.erase(RefAddr);
  return Addr;
}

const std::string &VM::getFunctionReferencedName(void *RefAddr) {
  return FunctionRefs[RefAddr]->getName();
}


/// getPointerToFunction - This method is used to get the address of the
/// specified function, compiling it if neccesary.
///
void *VM::getPointerToFunction(Function *F) {
  void *&Addr = GlobalAddress[F];   // Function already code gen'd
  if (Addr) return Addr;

  if (F->isExternal()) {
    assert(0 && "VM::getPointerToFunction: Doesn't handle external fn's yet!");
  }

  // JIT all of the functions in the module.  Eventually this will JIT functions
  // on demand.  This has the effect of populating all of the non-external
  // functions into the GlobalAddress table.
  PM.run(M);

  assert(Addr && "Code generation didn't add function to GlobalAddress table!");
  return Addr;
}
