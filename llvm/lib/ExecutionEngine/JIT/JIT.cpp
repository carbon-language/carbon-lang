//===-- JIT.cpp - LLVM Just in Time Compiler ------------------------------===//
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

#include "JIT.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/GlobalVariable.h"
#include "llvm/ModuleProvider.h"
#include "llvm/CodeGen/MachineCodeEmitter.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/ExecutionEngine/GenericValue.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetJITInfo.h"
#include "Support/DynamicLinker.h"
using namespace llvm;

JIT::JIT(ModuleProvider *MP, TargetMachine &tm, TargetJITInfo &tji)
  : ExecutionEngine(MP), TM(tm), TJI(tji), PM(MP) {
  setTargetData(TM.getTargetData());

  // Initialize MCE
  MCE = createEmitter(*this);
  
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

JIT::~JIT() {
  delete MCE;
  delete &TM;
}

/// run - Start execution with the specified function and arguments.
///
GenericValue JIT::run(Function *F, const std::vector<GenericValue> &ArgValues) {
  assert (F && "Function *F was null at entry to run()");

  int (*PF)(int, char **, const char **) =
    (int(*)(int, char **, const char **))getPointerToFunction(F);
  assert(PF != 0 && "Pointer to fn's code was null after getPointerToFunction");

  // Call the function.
  int ExitCode = PF(ArgValues[0].IntVal, (char **) GVTOP (ArgValues[1]),
		    (const char **) GVTOP (ArgValues[2]));

  // Run any atexit handlers now!
  runAtExitHandlers();

  GenericValue rv;
  rv.IntVal = ExitCode;
  return rv;
}

/// runJITOnFunction - Run the FunctionPassManager full of
/// just-in-time compilation passes on F, hopefully filling in
/// GlobalAddress[F] with the address of F's machine code.
///
void JIT::runJITOnFunction(Function *F) {
  static bool isAlreadyCodeGenerating = false;
  assert(!isAlreadyCodeGenerating && "Error: Recursive compilation detected!");

  // JIT the function
  isAlreadyCodeGenerating = true;
  PM.run(*F);
  isAlreadyCodeGenerating = false;

  // If the function referred to a global variable that had not yet been
  // emitted, it allocates memory for the global, but doesn't emit it yet.  Emit
  // all of these globals now.
  while (!PendingGlobals.empty()) {
    const GlobalVariable *GV = PendingGlobals.back();
    PendingGlobals.pop_back();
    EmitGlobalVariable(GV);
  }
}

/// getPointerToFunction - This method is used to get the address of the
/// specified function, compiling it if neccesary.
///
void *JIT::getPointerToFunction(Function *F) {
  if (void *Addr = getPointerToGlobalIfAvailable(F))
    return Addr;   // Check if function already code gen'd

  // Make sure we read in the function if it exists in this Module
  MP->materializeFunction(F);

  if (F->isExternal()) {
    void *Addr = getPointerToNamedFunction(F->getName());
    addGlobalMapping(F, Addr);
    return Addr;
  }

  runJITOnFunction(F);

  void *Addr = getPointerToGlobalIfAvailable(F);
  assert(Addr && "Code generation didn't add function to GlobalAddress table!");
  return Addr;
}

// getPointerToFunctionOrStub - If the specified function has been
// code-gen'd, return a pointer to the function.  If not, compile it, or use
// a stub to implement lazy compilation if available.
//
void *JIT::getPointerToFunctionOrStub(Function *F) {
  // If we have already code generated the function, just return the address.
  if (void *Addr = getPointerToGlobalIfAvailable(F))
    return Addr;

  // If the target supports "stubs" for functions, get a stub now.
  if (void *Ptr = TJI.getJITStubForFunction(F, *MCE))
    return Ptr;

  // Otherwise, if the target doesn't support it, just codegen the function.
  return getPointerToFunction(F);
}

/// getOrEmitGlobalVariable - Return the address of the specified global
/// variable, possibly emitting it to memory if needed.  This is used by the
/// Emitter.
void *JIT::getOrEmitGlobalVariable(const GlobalVariable *GV) {
  void *Ptr = getPointerToGlobalIfAvailable(GV);
  if (Ptr) return Ptr;

  // If the global is external, just remember the address.
  if (GV->isExternal()) {
    Ptr = GetAddressOfSymbol(GV->getName().c_str());
    if (Ptr == 0) {
      std::cerr << "Could not resolve external global address: "
                << GV->getName() << "\n";
      abort();
    }
  } else {
    // If the global hasn't been emitted to memory yet, allocate space.  We will
    // actually initialize the global after current function has finished
    // compilation.
    Ptr =new char[getTargetData().getTypeSize(GV->getType()->getElementType())];
    PendingGlobals.push_back(GV);
  }
  addGlobalMapping(GV, Ptr);
  return Ptr;
}


/// recompileAndRelinkFunction - This method is used to force a function
/// which has already been compiled, to be compiled again, possibly
/// after it has been modified. Then the entry to the old copy is overwritten
/// with a branch to the new copy. If there was no old copy, this acts
/// just like JIT::getPointerToFunction().
///
void *JIT::recompileAndRelinkFunction(Function *F) {
  void *OldAddr = getPointerToGlobalIfAvailable(F);

  // If it's not already compiled there is no reason to patch it up.
  if (OldAddr == 0) { return getPointerToFunction(F); }

  // Delete the old function mapping.
  addGlobalMapping(F, 0);

  // Recodegen the function
  runJITOnFunction(F);

  // Update state, forward the old function to the new function.
  void *Addr = getPointerToGlobalIfAvailable(F);
  assert(Addr && "Code generation didn't add function to GlobalAddress table!");
  TJI.replaceMachineCodeForFunction(OldAddr, Addr);
  return Addr;
}
