// $Id$
//***************************************************************************
// File:
//	llc.cpp
// 
// Purpose:
//	Driver for llc compiler.
// 
// History:
//	7/15/01	 -  Vikram Adve  -  Created
// 
//**************************************************************************/

#include "llvm/Module.h"
#include "llvm/Method.h"
#include "llvm/Bytecode/Reader.h"
#include "llvm/Bytecode/Writer.h"
#include "llvm/CodeGen/InstrSelection.h"
#include "llvm/CodeGen/Sparc.h"
#include "llvm/Support/CommandLine.h"

cl::String InputFilename ("", "Input filename", cl::NoFlags, "");
cl::String OutputFilename("o", "Output filename", cl::NoFlags, "");

static bool CompileModule(Module *module, TargetMachine &Target) {
  bool failed = false;
  
  for (Module::MethodListType::const_iterator
	 methodIter = module->getMethodList().begin();
       methodIter != module->getMethodList().end();
       ++methodIter)
    {
      Method* method = *methodIter;
      
      if (SelectInstructionsForMethod(method, Target))
	{
	  failed = true;
	  cerr << "Instruction selection failed for method "
	       << method->getName() << "\n\n";
	}
    }
  
  return failed;
}


//---------------------------------------------------------------------------
// Function main()
// 
// Entry point for the driver.
//---------------------------------------------------------------------------

int main(int argc, char** argv) {
  cl::ParseCommandLineOptions(argc, argv, " llvm system compiler\n");
  UltraSparc Target;

  Module *module = ParseBytecodeFile(InputFilename.getValue());
  if (module == 0) {
    cerr << "bytecode didn't read correctly.\n";
    return 1;
  }
  
  bool failure = CompileModule(module, Target);
  
  if (failure) {
      cerr << "Error compiling "
	   << InputFilename.getValue() << "!\n";
      delete module;
      return 1;
    }
  
  // Okay, we're done now... write out result...
  // WriteBytecodeToFile(module, 
  // 		      OutputFilename.getValue());
  
  // Clean up and exit
  delete module;
  return 0;
}
