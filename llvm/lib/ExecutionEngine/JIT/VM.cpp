//===-- VM.cpp - LLVM Just in Time Compiler -------------------------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This tool implements a just-in-time compiler for LLVM, allowing direct
// execution of LLVM bytecode in an efficient manner.
//
//===----------------------------------------------------------------------===//

#include "VM.h"
#include "llvm/Function.h"
#include "llvm/ModuleProvider.h"
#include "llvm/CodeGen/MachineCodeEmitter.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetJITInfo.h"
using namespace llvm;

VM::~VM() {
  delete MCE;
  delete &TM;
}

/// setupPassManager - Initialize the VM PassManager object with all of the
/// passes needed for the target to generate code.
///
void VM::setupPassManager() {
  // Compile LLVM Code down to machine code in the intermediate representation
  TJI.addPassesToJITCompile(PM);

  // Turn the machine code intermediate representation into bytes in memory that
  // may be executed.
  if (TM.addPassesToEmitMachineCode(PM, *MCE)) {
    std::cerr << "lli: target '" << TM.getName()
              << "' doesn't support machine code emission!\n";
    abort();
  }
}

/// runJITOnFunction - Run the FunctionPassManager full of
/// just-in-time compilation passes on F, hopefully filling in
/// GlobalAddress[F] with the address of F's machine code.
///
void VM::runJITOnFunction(Function *F) {
  static bool isAlreadyCodeGenerating = false;
  assert(!isAlreadyCodeGenerating && "Error: Recursive compilation detected!");

  // JIT the function
  isAlreadyCodeGenerating = true;
  PM.run(*F);
  isAlreadyCodeGenerating = false;
}

/// getPointerToFunction - This method is used to get the address of the
/// specified function, compiling it if neccesary.
///
void *VM::getPointerToFunction(Function *F) {
  void *&Addr = GlobalAddress[F];   // Check if function already code gen'd
  if (Addr) return Addr;

  // Make sure we read in the function if it exists in this Module
  MP->materializeFunction(F);

  if (F->isExternal())
    return Addr = getPointerToNamedFunction(F->getName());

  runJITOnFunction(F);
  assert(Addr && "Code generation didn't add function to GlobalAddress table!");
  return Addr;
}

// getPointerToFunctionOrStub - If the specified function has been
// code-gen'd, return a pointer to the function.  If not, compile it, or use
// a stub to implement lazy compilation if available.
//
void *VM::getPointerToFunctionOrStub(Function *F) {
  // If we have already code generated the function, just return the address.
  std::map<const GlobalValue*, void *>::iterator I = GlobalAddress.find(F);
  if (I != GlobalAddress.end()) return I->second;

  // If the target supports "stubs" for functions, get a stub now.
  if (void *Ptr = TJI.getJITStubForFunction(F, *MCE))
    return Ptr;

  // Otherwise, if the target doesn't support it, just codegen the function.
  return getPointerToFunction(F);
}

/// recompileAndRelinkFunction - This method is used to force a function
/// which has already been compiled, to be compiled again, possibly
/// after it has been modified. Then the entry to the old copy is overwritten
/// with a branch to the new copy. If there was no old copy, this acts
/// just like VM::getPointerToFunction().
///
void *VM::recompileAndRelinkFunction(Function *F) {
  void *&Addr = GlobalAddress[F];   // Check if function already code gen'd

  // If it's not already compiled (this is kind of weird) there is no
  // reason to patch it up.
  if (!Addr) { return getPointerToFunction (F); }

  void *OldAddr = Addr;
  Addr = 0;
  MachineFunction::destruct(F);
  runJITOnFunction(F);
  assert(Addr && "Code generation didn't add function to GlobalAddress table!");
  TJI.replaceMachineCodeForFunction(OldAddr, Addr);
  return Addr;
}
