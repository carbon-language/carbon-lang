//===- genexec.cpp - Functions for generating executable files  ------------===//
//
// This file contains functions for generating executable files once linking
// has finished.  This includes generating a shell script to run the JIT or
// a native executable derived from the bytecode.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/Linker.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Bytecode/WriteBytecodePass.h"
#include "Support/SystemUtils.h"
#include "util.h"

#include <fstream>
#include <string>
#include <vector>

//
// Function: GenerateBytecode ()
//
// Description:
//  This function generates a bytecode file from the specified module.
//
// Inputs:
//  M           - The module for which bytecode should be generated.
//  Strip       - Flags whether symbols should be stripped from the output.
//  Internalize - Flags whether all symbols should be marked internal.
//  Out         - Pointer to file stream to which to write the output.
//
// Outputs:
//  None.
//
// Return value:
//  0 - No error.
//  1 - Error.
//
int
GenerateBytecode (Module * M,
                  bool Strip,
                  bool Internalize,
                  std::ofstream * Out)
{
  // In addition to just linking the input from GCC, we also want to spiff it up
  // a little bit.  Do this now.
  PassManager Passes;

  // Add an appropriate TargetData instance for this module...
  Passes.add(new TargetData("gccld", M));

  // Linking modules together can lead to duplicated global constants, only keep
  // one copy of each constant...
  //
  Passes.add(createConstantMergePass());

  // If the -s command line option was specified, strip the symbols out of the
  // resulting program to make it smaller.  -s is a GCC option that we are
  // supporting.
  //
  if (Strip)
    Passes.add(createSymbolStrippingPass());

  // Often if the programmer does not specify proper prototypes for the
  // functions they are calling, they end up calling a vararg version of the
  // function that does not get a body filled in (the real function has typed
  // arguments).  This pass merges the two functions.
  //
  Passes.add(createFunctionResolvingPass());

  if (Internalize) {
    // Now that composite has been compiled, scan through the module, looking
    // for a main function.  If main is defined, mark all other functions
    // internal.
    //
    Passes.add(createInternalizePass());
  }

  // Remove unused arguments from functions...
  //
  Passes.add(createDeadArgEliminationPass());

  // The FuncResolve pass may leave cruft around if functions were prototyped
  // differently than they were defined.  Remove this cruft.
  //
  Passes.add(createInstructionCombiningPass());

  // Delete basic blocks, which optimization passes may have killed...
  //
  Passes.add(createCFGSimplificationPass());

  // Now that we have optimized the program, discard unreachable functions...
  //
  Passes.add(createGlobalDCEPass());

  // Add the pass that writes bytecode to the output file...
  Passes.add(new WriteBytecodePass(Out));

  // Run our queue of passes all at once now, efficiently.
  Passes.run(*M);

  return 0;
}

//
// Function: generate_assembly ()
//
// Description:
//  This function generates a native assembly language source file from the
//  specified bytecode file.
//
// Inputs:
//  InputFilename  - The name of the output bytecode file.
//  OutputFilename - The name of the file to generate.
//  llc            - The pathname to use for LLC.
//  envp           - The environment to use when running LLC.
//
// Outputs:
//  None.
//
// Return value:
//  0 - Success
//  1 - Failure
//
int
generate_assembly (std::string OutputFilename,
                   std::string InputFilename,
                   std::string llc,
                   char ** const envp)
{
  //
  // Run LLC to convert the bytecode file into assembly code.
  //
  const char * cmd[8];

  cmd[0] =  llc.c_str();
  cmd[1] =  "-f";
  cmd[2] =  "-o";
  cmd[3] =  OutputFilename.c_str();
  cmd[4] =  InputFilename.c_str();
  cmd[5] =  NULL;
  if ((ExecWait (cmd, envp)) == -1)
  {
    return 1;
  }

  return 0;
}

//
// Function: generate_native ()
//
// Description:
//  This function generates a native assembly language source file from the
//  specified assembly source file.
//
// Inputs:
//  InputFilename  - The name of the output bytecode file.
//  OutputFilename - The name of the file to generate.
//  Libraries      - The list of libraries with which to link.
//  gcc            - The pathname to use for GGC.
//  envp           - A copy of the process's current environment.
//
// Outputs:
//  None.
//
// Return value:
//  0 - Success
//  1 - Failure
//
int
generate_native (std::string OutputFilename,
                 std::string InputFilename,
                 std::vector<std::string> Libraries,
                 std::string gcc,
                 char ** const envp)
{
  //
  // Remove these environment variables from the environment of the
  // programs that we will execute.  It appears that GCC sets these
  // environment variables so that the programs it uses can configure
  // themselves identically.
  //
  // However, when we invoke GCC below, we want it to use its  normal
  // configuration.  Hence, we must sanitize it's environment.
  //
  char ** clean_env = copy_env (envp);
  if (clean_env == NULL)
  {
    return 1;
  }
  remove_env ("LIBRARY_PATH", clean_env);
  remove_env ("COLLECT_GCC_OPTIONS", clean_env);
  remove_env ("GCC_EXEC_PREFIX", clean_env);
  remove_env ("COMPILER_PATH", clean_env);
  remove_env ("COLLECT_GCC", clean_env);

  const char * cmd[8 + Libraries.size()];

  //
  // Run GCC to assemble and link the program into native code.
  //
  // Note:
  //  We can't just assemble and link the file with the system assembler
  //  and linker because we don't know where to put the _start symbol.
  //  GCC mysteriously knows how to do it.
  //
  unsigned int index=0;
  cmd[index++] =  gcc.c_str();
  cmd[index++] =  "-o";
  cmd[index++] =  OutputFilename.c_str();
  cmd[index++] =  InputFilename.c_str();
  for (; (index - 4) < Libraries.size(); index++)
  {
    Libraries[index - 4] = "-l" + Libraries[index - 4];
    cmd[index] = Libraries[index-4].c_str();
  }
  cmd[index++] =  NULL;
  if ((ExecWait (cmd, clean_env)) == -1)
  {
    return 1;
  }

  return 0;
}
