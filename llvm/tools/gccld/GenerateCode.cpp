//===- GenerateCode.cpp - Functions for generating executable files  ------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file contains functions for generating executable files once linking
// has finished.  This includes generating a shell script to run the JIT or
// a native executable derived from the bytecode.
//
//===----------------------------------------------------------------------===//

#include "gccld.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Analysis/LoadValueNumbering.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Bytecode/WriteBytecodePass.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Linker.h"
#include "Support/SystemUtils.h"
#include "Support/CommandLine.h"
using namespace llvm;

namespace {
  cl::opt<bool>
  DisableInline("disable-inlining", cl::desc("Do not run the inliner pass"));

  cl::opt<bool>
  Verify("verify", cl::desc("Verify intermediate results of all passes"));

  cl::opt<bool>
  DisableOptimizations("disable-opt",
                       cl::desc("Do not run any optimization passes"));
}

static inline void addPass(PassManager &PM, Pass *P) {
  // Add the pass to the pass manager...
  PM.add(P);
  
  // If we are verifying all of the intermediate steps, add the verifier...
  if (Verify) PM.add(createVerifierPass());
}

/// GenerateBytecode - generates a bytecode file from the specified module.
///
/// Inputs:
///  M           - The module for which bytecode should be generated.
///  Strip       - Flags whether symbols should be stripped from the output.
///  Internalize - Flags whether all symbols should be marked internal.
///  Out         - Pointer to file stream to which to write the output.
///
/// Returns non-zero value on error.
///
int llvm::GenerateBytecode(Module *M, bool Strip, bool Internalize,
                           std::ostream *Out) {
  // In addition to just linking the input from GCC, we also want to spiff it up
  // a little bit.  Do this now.
  PassManager Passes;

  if (Verify) Passes.add(createVerifierPass());

  // Add an appropriate TargetData instance for this module...
  addPass(Passes, new TargetData("gccld", M));

  // Often if the programmer does not specify proper prototypes for the
  // functions they are calling, they end up calling a vararg version of the
  // function that does not get a body filled in (the real function has typed
  // arguments).  This pass merges the two functions.
  addPass(Passes, createFunctionResolvingPass());

  if (!DisableOptimizations) {
    if (Internalize) {
      // Now that composite has been compiled, scan through the module, looking
      // for a main function.  If main is defined, mark all other functions
      // internal.
      addPass(Passes, createInternalizePass());
    }

    // Now that we internalized some globals, see if we can mark any globals as
    // being constant!
    addPass(Passes, createGlobalConstifierPass());

    // Linking modules together can lead to duplicated global constants, only
    // keep one copy of each constant...
    addPass(Passes, createConstantMergePass());

    // If the -s command line option was specified, strip the symbols out of the
    // resulting program to make it smaller.  -s is a GCC option that we are
    // supporting.
    if (Strip)
      addPass(Passes, createSymbolStrippingPass());

    // Propagate constants at call sites into the functions they call.
    addPass(Passes, createIPConstantPropagationPass());

    // Remove unused arguments from functions...
    addPass(Passes, createDeadArgEliminationPass());

    if (!DisableInline)
      addPass(Passes, createFunctionInliningPass()); // Inline small functions

    // If we didn't decide to inline a function, check to see if we can
    // transform it to pass arguments by value instead of by reference.
    addPass(Passes, createArgumentPromotionPass());

    // The IPO passes may leave cruft around.  Clean up after them.
    addPass(Passes, createInstructionCombiningPass());

    addPass(Passes, createScalarReplAggregatesPass()); // Break up allocas

    // Run a few AA driven optimizations here and now, to cleanup the code.
    // Eventually we should put an IP AA in place here.

    addPass(Passes, createLICMPass());               // Hoist loop invariants
    addPass(Passes, createLoadValueNumberingPass()); // GVN for load instrs
    addPass(Passes, createGCSEPass());               // Remove common subexprs

    // Cleanup and simplify the code after the scalar optimizations.
    addPass(Passes, createInstructionCombiningPass());

    // Delete basic blocks, which optimization passes may have killed...
    addPass(Passes, createCFGSimplificationPass());

    // Now that we have optimized the program, discard unreachable functions...
    addPass(Passes, createGlobalDCEPass());
  }

  // Make sure everything is still good.
  Passes.add(createVerifierPass());

  // Add the pass that writes bytecode to the output file...
  addPass(Passes, new WriteBytecodePass(Out));

  // Run our queue of passes all at once now, efficiently.
  Passes.run(*M);

  return 0;
}

/// GenerateAssembly - generates a native assembly language source file from the
/// specified bytecode file.
///
/// Inputs:
///  InputFilename  - The name of the output bytecode file.
///  OutputFilename - The name of the file to generate.
///  llc            - The pathname to use for LLC.
///  envp           - The environment to use when running LLC.
///
/// Return non-zero value on error.
///
int llvm::GenerateAssembly(const std::string &OutputFilename,
                           const std::string &InputFilename,
                           const std::string &llc,
                           char ** const envp) {
  // Run LLC to convert the bytecode file into assembly code.
  const char *cmd[6];
  cmd[0] = llc.c_str();
  cmd[1] = "-f";
  cmd[2] = "-o";
  cmd[3] = OutputFilename.c_str();
  cmd[4] = InputFilename.c_str();
  cmd[5] = 0;

  return ExecWait(cmd, envp);
}

/// GenerateAssembly - generates a native assembly language source file from the
/// specified bytecode file.
int llvm::GenerateCFile(const std::string &OutputFile,
                        const std::string &InputFile,
                        const std::string &llc, char ** const envp) {
  // Run LLC to convert the bytecode file into C.
  const char *cmd[7];

  cmd[0] = llc.c_str();
  cmd[1] = "-march=c";
  cmd[2] = "-f";
  cmd[3] = "-o";
  cmd[4] = OutputFile.c_str();
  cmd[5] = InputFile.c_str();
  cmd[6] = 0;
  return ExecWait(cmd, envp);
}

/// GenerateNative - generates a native assembly language source file from the
/// specified assembly source file.
///
/// Inputs:
///  InputFilename  - The name of the output bytecode file.
///  OutputFilename - The name of the file to generate.
///  Libraries      - The list of libraries with which to link.
///  LibPaths       - The list of directories in which to find libraries.
///  gcc            - The pathname to use for GGC.
///  envp           - A copy of the process's current environment.
///
/// Outputs:
///  None.
///
/// Returns non-zero value on error.
///
int llvm::GenerateNative(const std::string &OutputFilename,
                         const std::string &InputFilename,
                         const std::vector<std::string> &Libraries,
                         const std::vector<std::string> &LibPaths,
                         const std::string &gcc, char ** const envp) {
  // Remove these environment variables from the environment of the
  // programs that we will execute.  It appears that GCC sets these
  // environment variables so that the programs it uses can configure
  // themselves identically.
  //
  // However, when we invoke GCC below, we want it to use its normal
  // configuration.  Hence, we must sanitize its environment.
  char ** clean_env = CopyEnv(envp);
  if (clean_env == NULL)
    return 1;
  RemoveEnv("LIBRARY_PATH", clean_env);
  RemoveEnv("COLLECT_GCC_OPTIONS", clean_env);
  RemoveEnv("GCC_EXEC_PREFIX", clean_env);
  RemoveEnv("COMPILER_PATH", clean_env);
  RemoveEnv("COLLECT_GCC", clean_env);

  std::vector<const char *> cmd;

  // Run GCC to assemble and link the program into native code.
  //
  // Note:
  //  We can't just assemble and link the file with the system assembler
  //  and linker because we don't know where to put the _start symbol.
  //  GCC mysteriously knows how to do it.
  cmd.push_back(gcc.c_str());
  cmd.push_back("-fno-strict-aliasing");
  cmd.push_back("-O3");
  cmd.push_back("-o");
  cmd.push_back(OutputFilename.c_str());
  cmd.push_back(InputFilename.c_str());

  // Adding the library paths creates a problem for native generation.  If we
  // include the search paths from llvmgcc, then we'll be telling normal gcc
  // to look inside of llvmgcc's library directories for libraries.  This is
  // bad because those libraries hold only bytecode files (not native object
  // files).  In the end, we attempt to link the bytecode libgcc into a native
  // program.
#if 0
  // Add in the library path options.
  for (unsigned index=0; index < LibPaths.size(); index++) {
    cmd.push_back("-L");
    cmd.push_back(LibPaths[index].c_str());
  }
#endif

  // Add in the libraries to link.
  std::vector<std::string> Libs(Libraries);
  for (unsigned index = 0; index < Libs.size(); index++) {
    if (Libs[index] != "crtend") {
      Libs[index] = "-l" + Libs[index];
      cmd.push_back(Libs[index].c_str());
    }
  }
  cmd.push_back(NULL);

  // Run the compiler to assembly and link together the program.
  return ExecWait(&(cmd[0]), clean_env);
}
