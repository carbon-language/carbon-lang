//===-- VM.cpp - LLVM Just in Time Compiler -------------------------------===//
//
// This tool implements a just-in-time compiler for LLVM, allowing direct
// execution of LLVM bytecode in an efficient manner.
//
//===----------------------------------------------------------------------===//

#include "VM.h"
#include "llvm/Function.h"
#include "llvm/ModuleProvider.h"
#include "llvm/CodeGen/MachineCodeEmitter.h"
#include "llvm/Target/TargetMachine.h"

VM::~VM() {
  delete MCE;
  delete &TM;
}

/// setupPassManager - Initialize the VM PassManager object with all of the
/// passes needed for the target to generate code.
///
void VM::setupPassManager() {
  // Compile LLVM Code down to machine code in the intermediate representation
  if (TM.addPassesToJITCompile(PM)) {
    std::cerr << "lli: target '" << TM.getName()
              << "' doesn't support JIT compilation!\n";
    abort();
  }

  // Turn the machine code intermediate representation into bytes in memory that
  // may be executed.
  //
  if (TM.addPassesToEmitMachineCode(PM, *MCE)) {
    std::cerr << "lli: target '" << TM.getName()
              << "' doesn't support machine code emission!\n";
    abort();
  }
}

/// getPointerToFunction - This method is used to get the address of the
/// specified function, compiling it if neccesary.
///
void *VM::getPointerToFunction(Function *F) {
  void *&Addr = GlobalAddress[F];   // Function already code gen'd
  if (Addr) return Addr;

  // Make sure we read in the function if it exists in this Module
  MP->materializeFunction(F);

  if (F->isExternal())
    return Addr = getPointerToNamedFunction(F->getName());

  static bool isAlreadyCodeGenerating = false;
  assert(!isAlreadyCodeGenerating && "ERROR: RECURSIVE COMPILATION DETECTED!");

  // JIT the function
  isAlreadyCodeGenerating = true;
  PM.run(*F);
  isAlreadyCodeGenerating = false;

  assert(Addr && "Code generation didn't add function to GlobalAddress table!");
  return Addr;
}
