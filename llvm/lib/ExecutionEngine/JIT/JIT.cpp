//===-- JIT.cpp - LLVM Just in Time Compiler ------------------------------===//
//
// This file implements the top-level support for creating a Just-In-Time
// compiler for the current architecture.
//
//===----------------------------------------------------------------------===//

#include "VM.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetMachineImpls.h"
#include "llvm/Module.h"


/// createJIT - Create an return a new JIT compiler if there is one available
/// for the current target.  Otherwise it returns null.
///
ExecutionEngine *ExecutionEngine::createJIT(Module *M, unsigned Config) {
  // FIXME: This should be controlled by which subdirectory gets linked in!
#if !defined(i386) && !defined(__i386__) && !defined(__x86__)
  return 0;
#endif
  // Allocate a target... in the future this will be controllable on the
  // command line.
  TargetMachine *Target = allocateX86TargetMachine(Config);
  assert(Target && "Could not allocate X86 target machine!");

  // Create the virtual machine object...
  return new VM(M, Target);
}

VM::VM(Module *M, TargetMachine *tm) : ExecutionEngine(M), TM(*tm) {
  setTargetData(TM.getTargetData());
  MCE = createEmitter(*this);  // Initialize MCE
  setupPassManager();
  registerCallback();
  emitGlobals();
}

int VM::run(const std::string &FnName, const std::vector<std::string> &Args) {
  Function *F = getModule().getNamedFunction(FnName);
  if (F == 0) {
    std::cerr << "Could not find function '" << FnName <<"' in module!\n";
    return 1;
  }

  int(*PF)(int, char**) = (int(*)(int, char**))getPointerToFunction(F);
  assert(PF != 0 && "Null pointer to function?");

  // Build an argv vector...
  char **Argv = (char**)CreateArgv(Args);

  // Call the main function...
  int Result = PF(Args.size(), Argv);

  // Run any atexit handlers now!
  runAtExitHandlers();
  return Result;
}
