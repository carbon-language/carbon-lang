//===-- llc.cpp - Implement the LLVM Compiler ----------------------------===//
//
// This is the llc compiler driver.
//
//===---------------------------------------------------------------------===//

#include "llvm/Bytecode/Reader.h"
#include "llvm/Optimizations/Normalize.h"
#include "llvm/Target/Sparc.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Module.h"
#include "llvm/Method.h"
#include <memory>

cl::String InputFilename ("", "Input filename", cl::NoFlags, "-");
cl::String OutputFilename("o", "Output filename", cl::NoFlags, "");


//-------------------------- Internal Functions -----------------------------//

static void NormalizeMethod(Method* method) {
  NormalizePhiConstantArgs(method);
}


//===---------------------------------------------------------------------===//
// Function main()
// 
// Entry point for the llc compiler.
//===---------------------------------------------------------------------===//

int main(int argc, char **argv) {
  // Parse command line options...
  cl::ParseCommandLineOptions(argc, argv, " llvm system compiler\n");

  // Allocate a target... in the future this will be controllable on the
  // command line.
  auto_ptr<TargetMachine> Target(allocateSparcTargetMachine());

  // Load the module to be compiled...
  auto_ptr<Module> M(ParseBytecodeFile(InputFilename));
  if (M.get() == 0) {
    cerr << "bytecode didn't read correctly.\n";
    return 1;
  }

  // Loop over all of the methods in the module, compiling them.
  for (Module::const_iterator MI = M->begin(), ME = M->end(); MI != ME; ++MI) {
    Method *Meth = *MI;
    
    NormalizeMethod(Meth);
    
    if (Target.get()->compileMethod(Meth)) {
      cerr << "Error compiling " << InputFilename << "!\n";
      return 1;
    }
  }
  
  Target->emitAssembly(M.get(), cout);
  return 0;
}


