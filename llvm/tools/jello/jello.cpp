//===-- jello.cpp - LLVM Just in Time Compiler ----------------------------===//
//
// This tool implements a just-in-time compiler for LLVM, allowing direct
// execution of LLVM bytecode in an efficient manner.
//
// FIXME: This code will get more object oriented as we get the call back
// intercept stuff implemented.
//
//===----------------------------------------------------------------------===//

#include "llvm/Module.h"
#include "llvm/Bytecode/Reader.h"
#include "llvm/CodeGen/MFunction.h"
#include "../lib/Target/X86/X86.h"   // FIXME: become generic eventually
#include "../lib/Target/X86/X86InstructionInfo.h"
#include "Support/CommandLine.h"
#include "Support/Statistic.h"

namespace {
  cl::opt<std::string>
  InputFile(cl::desc("<input bytecode>"), cl::Positional, cl::init("-"));

  cl::opt<std::string>
  MainFunction("f", cl::desc("Function to execute"), cl::init("main"),
               cl::value_desc("function name"));
}


/// ExecuteFunction - Compile the specified function to machine code, and
/// execute it.
///
static void ExecuteFunction(Function &F) {
  X86InstructionInfo II;

  // Perform instruction selection to turn the function into an x86 SSA form
  MFunction *MF = X86SimpleInstructionSelection(F);

  // TODO: optional optimizations go here

  // If -debug is specified, output selected code to stderr
  /*DEBUG*/(MF->print(std::cerr, II));

  // Perform register allocation to convert to a concrete x86 representation
  X86SimpleRegisterAllocation(MF);
  
  // If -debug is specified, output compiled code to stderr
  /*DEBUG*/(X86PrintCode(MF, std::cerr));

  // Emit register allocated X86 code now...
  void *PFun = X86EmitCodeToMemory(MF);

  // We don't need the machine specific representation for this function anymore
  delete MF;
}


//===----------------------------------------------------------------------===//
// main Driver function
//
int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, " llvm just in time compiler\n");

  std::string ErrorMsg;
  if (Module *M = ParseBytecodeFile(InputFile, &ErrorMsg)) {
    for (Module::iterator I = M->begin(), E = M->end(); I != E; ++I)
      if (I->getName() == MainFunction)
        ExecuteFunction(*I);

    delete M;
    return 0;
  }
  
  std::cerr << "Error parsing '" << InputFile << "': " << ErrorMsg << "\n";
  return 1;
}

