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
#include "Support/CommandLine.h"

// FIXME: REMOVE THIS
#include "llvm/PassManager.h"

#if !defined(ENABLE_X86_JIT) && !defined(ENABLE_SPARC_JIT)
#define NO_JITS_ENABLED
#endif

namespace {
  enum ArchName { x86, Sparc };

#ifndef NO_JITS_ENABLED
  cl::opt<ArchName>
  Arch("march", cl::desc("Architecture to JIT to:"), cl::Prefix,
       cl::values(
#ifdef ENABLE_X86_JIT
                  clEnumVal(x86, "  IA-32 (Pentium and above)"),
#endif
#ifdef ENABLE_SPARC_JIT
                  clEnumValN(Sparc, "sparc", "  Sparc-V9"),
#endif
                  0),
#if defined(ENABLE_X86_JIT)
  cl::init(x86)
#elif defined(ENABLE_SPARC_JIT)
  cl::init(Sparc)
#endif
       );
#endif /* NO_JITS_ENABLED */
}

/// createJIT - Create an return a new JIT compiler if there is one available
/// for the current target.  Otherwise it returns null.
///
ExecutionEngine *ExecutionEngine::createJIT(Module *M, unsigned Config) {
  
  TargetMachine* (*TargetMachineAllocator)(unsigned) = 0;

  // Allow a command-line switch to override what *should* be the default target
  // machine for this platform. This allows for debugging a Sparc JIT on X86 --
  // our X86 machines are much faster at recompiling LLVM and linking LLI.
#ifdef NO_JITS_ENABLED
  return 0;
#endif

  switch (Arch) {
#ifdef ENABLE_X86_JIT
  case x86:
    TargetMachineAllocator = allocateX86TargetMachine;
    break;
#endif
#ifdef ENABLE_SPARC_JIT
  case Sparc:
    TargetMachineAllocator = allocateSparcTargetMachine;
    break;
#endif
  default:
    assert(0 && "-march flag not supported on this host!");
  }

  // Allocate a target...
  TargetMachine *Target = (*TargetMachineAllocator)(Config);
  assert(Target && "Could not allocate target machine!");
  
  // Create the virtual machine object...
  return new VM(M, Target);
}

VM::VM(Module *M, TargetMachine *tm) : ExecutionEngine(M), TM(*tm) {
  setTargetData(TM.getTargetData());

  // Initialize MCE
  MCE = createEmitter(*this);

  setupPassManager();

#ifdef ENABLE_SPARC_JIT
  // THIS GOES BEYOND UGLY HACKS
  if (TM.getName() == "UltraSparc-Native") {
    extern Pass *createPreSelectionPass(TargetMachine &TM);
    PassManager PM;
    // Specialize LLVM code for this target machine and then
    // run basic dataflow optimizations on LLVM code.
    PM.add(createPreSelectionPass(TM));
    PM.run(*M);
  }
#endif

  emitGlobals();
}

/// VM::run - This method begins the execution of a program beginning at the
/// specified function name.  The function is called with the specified
/// arguments and array of environment variables (a la main()).
///
/// Inputs:
///	FnName - The name of the function as a C++ string.
///	Args   - A vector of C++ strings containing the arguments.
///	envp   - An array of C strings containing the environment.
///
/// Return value:
///	1 - An error occurred.
///	Otherwise, the return value from the specified function is returned.
///
int VM::run(const std::string &FnName, const std::vector<std::string> &Args,
            const char **envp) {
  Function *F = getModule().getNamedFunction(FnName);
  if (F == 0) {
    std::cerr << "Could not find function '" << FnName << "' in module!\n";
    return 1;
  }

  int (*PF)(int, char**, const char**) =
    (int(*)(int, char**, const char**))getPointerToFunction(F);
  assert(PF != 0 && "Null pointer to function?");

  // Build an argv vector...
  char **Argv = (char**)CreateArgv(Args);

  // Call the main function...
  int Result = PF(Args.size(), Argv, envp);

  // Run any atexit handlers now!
  runAtExitHandlers();
  return Result;
}
