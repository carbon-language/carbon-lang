//===-- EmitFunctions.cpp - interface to insert instrumentation --*- C++ -*--=//
//
// This inserts a global constant table with function pointers all along
//
//===----------------------------------------------------------------------===//

//#include "llvm/Transforms/Instrumentation/ProfilePaths.h"
//#include "llvm/Transforms/Utils/UnifyFunctionExitNodes.h"
#include "llvm/Support/CFG.h"
#include "llvm/Transforms/Instrumentation/EmitFunctions.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/iMemory.h"
#include "llvm/Constants.h"
#include "llvm/Module.h"
#include <iostream>
#include <fstream>

using std::vector;

struct EmitFunctionTable : public Pass {
  const char *getPassName() const { return "EmitFunctionTablePass"; }

  bool run(Module &M);
};

//  Create a new pass to add function table
//
Pass *createEmitFunctionTablePass() {
  return new EmitFunctionTable();
}

//Per Module pass for inserting function table
bool EmitFunctionTable::run(Module &M){
  vector<const Type*> vType;
  vector<Constant *> vConsts;
  for(Module::iterator MI = M.begin(), ME = M.end(); MI!=ME; ++MI){
    Function *F = MI;
    if(F->size()<=1) continue;
    ConstantPointerRef *cpref = ConstantPointerRef::get(F);
    vType.push_back(MI->getType());
    vConsts.push_back(cpref);
  }
  
  StructType *sttype = StructType::get(vType);
  ConstantStruct *cstruct = ConstantStruct::get(sttype, vConsts);

  GlobalVariable *gb = new GlobalVariable(cstruct->getType(), true, true, 
                                          cstruct, "llvmFunctionTable");
  M.getGlobalList().push_back(gb);
  return true;  // Always modifies function
}
