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

#include "llvm/Support/Unique.h"
class TargetMachine;

//---------------------------------------------------------------------------
// class CompileContext
//---------------------------------------------------------------------------

class CompileContext: public Unique {
private:
  TargetMachine*	targetMachine;
  
public:
  CompileContext(TargetMachine *Target) : targetMachine(Target) {}
  ~CompileContext();
  
  const TargetMachine&	getTarget	() const { return *targetMachine; }
  TargetMachine&	getTarget	()	 { return *targetMachine; }
};

#endif
