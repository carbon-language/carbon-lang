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

cl::String InputFilename ("", "Input filename", cl::NoFlags, "-");
cl::String OutputFilename("o", "Output filename", cl::NoFlags, "");


//-------------------------- Internal Functions -----------------------------//

static void
NormalizeMethod(Method* method)
{
  NormalizePhiConstantArgs(method);
}


//===---------------------------------------------------------------------===//
// Function main()
// 
// Entry point for the llc compiler.
//===---------------------------------------------------------------------===//

int
main(int argc, char** argv)
{
  cl::ParseCommandLineOptions(argc, argv, " llvm system compiler\n");
  TargetMachine *Target = allocateSparcTargetMachine();
  
  Module *M = ParseBytecodeFile(InputFilename);
  if (M == 0)
    {
      cerr << "bytecode didn't read correctly.\n";
      delete Target;
      return 1;
    }

  bool Failed = false;
  for (Module::const_iterator MI = M->begin(), ME = M->end(); MI != ME; ++MI)
    {
      Method *Meth = *MI;
    
      NormalizeMethod(Meth);
    
      if (Target->compileMethod(Meth))
	{
	  cerr << "Error compiling " << InputFilename << "!\n";
	  Failed = true;
	  break;
	}
    }
  
  // Clean up and exit
  delete M;
  delete Target;
  return Failed;
}


