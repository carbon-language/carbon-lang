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

//************************** System Include Files **************************/

//*************************** User Include Files ***************************/

#include "llvm/Module.h"
#include "llvm/Method.h"
#include "llvm/Bytecode/Reader.h"
#include "llvm/Bytecode/Writer.h"
#include "llvm/Codegen/InstrForest.h"
#include "llvm/Codegen/InstrSelection.h"
#include "llvm/LLC/LLCOptions.h"
#include "llvm/LLC/CompileContext.h"

//************************** Forward Declarations **************************/

class Module;
class CompileContext;


static bool	CompileModule	(Module *module,
				 CompileContext& compileContext);

int DebugInstrSelectLevel = DEBUG_INSTR_TREES;


//---------------------------------------------------------------------------
// Function main()
// 
// Entry point for the driver.
//---------------------------------------------------------------------------


int
main(int argc, const char** argv, const char** envp)
{
  CompileContext compileContext(argc, argv, envp);
  
  Module *module =
    ParseBytecodeFile(compileContext.getOptions().getInputFileName());
  
  if (module == 0) {
    cerr << "bytecode didn't read correctly.\n";
    return 1;
  }
  
  bool failure = CompileModule(module, compileContext);
  
  if (failure)
    {
      cerr << "Error compiling "
	   << compileContext.getOptions().getInputFileName() << "!\n";
      delete module;
      return 1;
    }
  
  // Okay, we're done now... write out result...
  // WriteBytecodeToFile(module, 
  // 		      compileContext.getOptions().getOutputFileName);
  
  // Clean up and exit
  delete module;
  return 0;
}


static bool
CompileModule(Module *module,
	      CompileContext& ccontext)
{
  bool failed = false;
  
  for (Module::MethodListType::const_iterator
	 methodIter = module->getMethodList().begin();
       methodIter != module->getMethodList().end();
       ++methodIter)
    {
      Method* method = *methodIter;
      
      if (SelectInstructionsForMethod(method, ccontext))
	{
	  failed = true;
	  cerr << "Instruction selection failed for method "
	       << (method->hasName()? method->getName() : "")
	       << endl << endl;
	}
    }
  
  return failed;
}

