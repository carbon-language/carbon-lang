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

namespace {
  enum ArchName { nojit, x86, Sparc };

  cl::opt<ArchName>
  Arch("march", cl::desc("Architecture to JIT to:"), cl::Prefix,
       cl::values(clEnumVal(x86, "  IA-32 (pentium and above)"),
#if defined(sparc) || defined(__sparc__) || defined(__sparcv9)
                  clEnumValN(Sparc, "sparc", "  Sparc-V9"),
#endif
                  0),
#if defined(i386) || defined(__i386__) || defined(__x86__)
  cl::init(x86)
#elif defined(sparc) || defined(__sparc__) || defined(__sparcv9)
  cl::init(Sparc)
#else
  cl::init(nojit)
#endif
       );
}

/// createJIT - Create an return a new JIT compiler if there is one available
/// for the current target.  Otherwise it returns null.
///
ExecutionEngine *ExecutionEngine::createJIT(Module *M, unsigned Config) {
  
  TargetMachine* (*TargetMachineAllocator)(unsigned) = 0;

  // Allow a command-line switch to override what *should* be the default target
  // machine for this platform. This allows for debugging a Sparc JIT on X86 --
  // our X86 machines are much faster at recompiling LLVM and linking lli.
  switch (Arch) {
  case x86:
    TargetMachineAllocator = allocateX86TargetMachine;
    break;
#if defined(sparc) || defined(__sparc__) || defined(__sparcv9)
  case Sparc:
    TargetMachineAllocator = allocateSparcTargetMachine;
    break;
#endif
  default:
    assert(0 && "-march flag not supported on this host!");
  case nojit:
    return 0;
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

#if defined(sparc) || defined(__sparc__) || defined(__sparcv9)
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
