//===------------------------------------------------------------------------===
// LLVM 'LLC' UTILITY 
//
// This is the llc compiler driver.
//
//===------------------------------------------------------------------------===

#include "llvm/Bytecode/Reader.h"
#include "llvm/Optimizations/Normalize.h"
#include "llvm/CodeGen/Sparc.h"
#include "llvm/CodeGen/TargetMachine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Module.h"
#include "llvm/Method.h"

cl::String InputFilename ("", "Input filename", cl::NoFlags, "-");
cl::String OutputFilename("o", "Output filename", cl::NoFlags, "");

static void NormalizeMethod(Method* method) {
  NormalizePhiConstantArgs(method);
}


static bool CompileModule(Module *M, TargetMachine &Target) {
  for (Module::const_iterator MI = M->begin(), ME = M->end(); MI != ME; ++MI) {
    Method *Meth = *MI;
      
    NormalizeMethod(Meth);
      
    if (Target.compileMethod(Meth)) return true;
  }
  
  return false;
}



//---------------------------------------------------------------------------
// Function main()
// 
// Entry point for the llc compiler.
//---------------------------------------------------------------------------

int main(int argc, char** argv) {
  cl::ParseCommandLineOptions(argc, argv, " llvm system compiler\n");
  TargetMachine *Target = allocateSparcTargetMachine();
  
  Module *module = ParseBytecodeFile(InputFilename);
  if (module == 0) {
    cerr << "bytecode didn't read correctly.\n";
    return 1;
  }

  if (CompileModule(module, *Target)) {
    cerr << "Error compiling " << InputFilename << "!\n";
    delete module;
    return 1;
  }
  
  // Clean up and exit
  delete module;
  delete Target;
  return 0;
}

