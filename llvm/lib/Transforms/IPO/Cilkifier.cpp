//===- Cilkifier.cpp - Support routines for Cilk code gen. ------*- C++ -*-===//
//
// This is located here so that the code generator (dis) does not have to
// include and link with the libtipo.a archive containing class Cilkifier
// and the rest of the automatic parallelization code.
//===----------------------------------------------------------------------===//


#include "Cilkifier.h"
#include "llvm/Function.h"
#include "llvm/iOther.h"
#include "llvm/DerivedTypes.h"

//---------------------------------------------------------------------------- 
// Global constants used in marking Cilk functions and function calls.
// These should be used only by the auto-parallelization pass.
//---------------------------------------------------------------------------- 

const std::string CilkSuffix(".llvm2cilk");
const std::string DummySyncFuncName("__sync.llvm2cilk");

//---------------------------------------------------------------------------- 
// Routines to identify Cilk functions, calls to Cilk functions, and syncs.
//---------------------------------------------------------------------------- 

bool isCilk(const Function& F)
{
  assert(F.hasName());
  return (F.getName().rfind(CilkSuffix) ==
          F.getName().size() - CilkSuffix.size());
}

bool isCilkMain(const Function& F)
{
  assert(F.hasName());
  return (F.getName() == std::string("main") + CilkSuffix);
}


bool isCilk(const CallInst& CI)
{
  return (CI.getCalledFunction() != NULL && isCilk(*CI.getCalledFunction()));
}

bool isSync(const CallInst& CI)
{ 
  return (CI.getCalledFunction() != NULL &&
          CI.getCalledFunction()->getName() == DummySyncFuncName);
}


//---------------------------------------------------------------------------- 
