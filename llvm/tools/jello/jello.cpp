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
#include "llvm/PassManager.h"
#include "llvm/Bytecode/Reader.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetMachineImpls.h"
#include "Support/CommandLine.h"
#include "Support/Statistic.h"

namespace {
  cl::opt<std::string>
  InputFile(cl::desc("<input bytecode>"), cl::Positional, cl::init("-"));

  cl::opt<std::string>
  MainFunction("f", cl::desc("Function to execute"), cl::init("main"),
               cl::value_desc("function name"));
}

//===----------------------------------------------------------------------===//
// main Driver function
//
int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, " llvm just in time compiler\n");

  // Allocate a target... in the future this will be controllable on the
  // command line.
  std::auto_ptr<TargetMachine> target(allocateX86TargetMachine());
  assert(target.get() && "Could not allocate target machine!");

  TargetMachine &Target = *target.get();

  // Parse the input bytecode file...
  std::string ErrorMsg;
  std::auto_ptr<Module> M(ParseBytecodeFile(InputFile, &ErrorMsg));
  if (M.get() == 0) {
    std::cerr << argv[0] << ": bytecode '" << InputFile
              << "' didn't read correctly: << " << ErrorMsg << "\n";
    return 1;
  }

  PassManager Passes;
  if (Target.addPassesToJITCompile(Passes)) {
    std::cerr << argv[0] << ": target '" << Target.TargetName
              << "' doesn't support JIT compilation!\n";
    return 1;
  }

  // JIT all of the methods in the module.  Eventually this will JIT functions
  // on demand.
  Passes.run(*M.get());
  
  return 1;
}

