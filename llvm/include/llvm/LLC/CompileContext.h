// $Id$ -*-c++-*-
//***************************************************************************
// class CompileContext
// 
// Purpose:
//	Holds the common option and target information for a compilation run.
// 
// History:
//     07/15/01 - vadve - Created
//
//**************************************************************************/

#ifndef LLVM_LLC_COMPILECONTEXT_H
#define LLVM_LLC_COMPILECONTEXT_H

//************************** System Include Files **************************/

#include <string>

//*************************** User Include Files ***************************/

#include "llvm/CodeGen/Sparc.h"
#include "llvm/LLC/LLCOptions.h"

//************************** Forward Declarations **************************/

class ProgramOptions;
class TargetMachine;


//---------------------------------------------------------------------------
// class CompileContext
//---------------------------------------------------------------------------

class CompileContext: public Unique
{
private:
  LLCOptions*		options;
  TargetMachine*	targetMachine;
  
public:
  /*ctor*/		CompileContext	(int argc, const char **argv, const char** envp);
  /*dtor*/ virtual	~CompileContext	();
  
  const LLCOptions&	getOptions	() const { return *options; }
  
  const TargetMachine&	getTarget	() const { return *targetMachine; }
  TargetMachine&	getTarget	()	 { return *targetMachine; }
};


inline
CompileContext::CompileContext(int argc, const char **argv, const char** envp)
{
  options = new LLCOptions(argc, argv, envp);
  targetMachine = new UltraSparc;
}


inline
CompileContext::~CompileContext()
{
  delete options;
  delete targetMachine;
}

//**************************************************************************/

#endif
