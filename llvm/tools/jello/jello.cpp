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


#include "llvm/CodeGen/MachineCodeEmitter.h"
#include "llvm/CodeGen/MachineFunction.h"
struct JelloMachineCodeEmitter : public MachineCodeEmitter {
  void startFunction(MachineFunction &F) {
    std::cout << "\n**** Writing machine code for function: "
              << F.getFunction()->getName() << "\n";
  }
  void finishFunction(MachineFunction &F) {
    std::cout << "\n";
  }
  void startBasicBlock(MachineBasicBlock &BB) {
    std::cout << "\n--- Basic Block: " << BB.getBasicBlock()->getName() << "\n";
  }

  void emitByte(unsigned char B) {
    std::cout << "0x" << std::hex << (unsigned int)B << std::dec << " ";
  }
  void emitPCRelativeDisp(Value *V) {
    std::cout << "<" << V->getName() << ": 0x00 0x00 0x00 0x00> ";
  }
};


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

  // Compile LLVM Code down to machine code in the intermediate representation
  if (Target.addPassesToJITCompile(Passes)) {
    std::cerr << argv[0] << ": target '" << Target.getName()
              << "' doesn't support JIT compilation!\n";
    return 1;
  }

  // Turn the machine code intermediate representation into bytes in memory that
  // may be executed.
  //
  JelloMachineCodeEmitter MCE;
  if (Target.addPassesToEmitMachineCode(Passes, MCE)) {
    std::cerr << argv[0] << ": target '" << Target.getName()
              << "' doesn't support machine code emission!\n";
    return 1;
  }

  // JIT all of the methods in the module.  Eventually this will JIT functions
  // on demand.
  Passes.run(*M.get());
  
  return 0;
}

