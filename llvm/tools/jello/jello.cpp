//===-- jello.cpp - LLVM Just in Time Compiler ----------------------------===//
//
// This tool implements a just-in-time compiler for LLVM, allowing direct
// execution of LLVM bytecode in an efficient manner.
//
//===----------------------------------------------------------------------===//

#include "llvm/Module.h"
#include "llvm/Bytecode/Reader.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetMachineImpls.h"
#include "Support/CommandLine.h"
#include "VM.h"

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
  std::auto_ptr<TargetMachine> Target(allocateX86TargetMachine());
  assert(Target.get() && "Could not allocate target machine!");

  // Parse the input bytecode file...
  std::string ErrorMsg;
  std::auto_ptr<Module> M(ParseBytecodeFile(InputFile, &ErrorMsg));
  if (M.get() == 0) {
    std::cerr << argv[0] << ": bytecode '" << InputFile
              << "' didn't read correctly: << " << ErrorMsg << "\n";
    return 1;
  }

  // Create the virtual machine object...
  VM TheVM(argv[0], *M.get(), *Target.get());

  Function *F = M.get()->getNamedFunction(MainFunction);
  if (F == 0) {
    std::cerr << "Could not find function '" << MainFunction <<"' in module!\n";
    return 1;
  }

  // Run the virtual machine...
  return TheVM.run(F);
}
