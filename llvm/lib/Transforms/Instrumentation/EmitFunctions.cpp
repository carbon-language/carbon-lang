//===-- EmitFunctions.cpp - interface to insert instrumentation --*- C++ -*--=//
//
// This inserts a global constant table with function pointers all along
//
//===----------------------------------------------------------------------===//

#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Constants.h"
#include "llvm/Module.h"

using std::vector;

namespace {
  struct EmitFunctionTable : public Pass {
    bool run(Module &M);
  };
  
  RegisterOpt<EmitFunctionTable> X("emitfuncs", "Emit a Function Table");
}

// Per Module pass for inserting function table
bool EmitFunctionTable::run(Module &M){
  vector<const Type*> vType;
  vector<Constant *> vConsts;
  for(Module::iterator MI = M.begin(), ME = M.end(); MI!=ME; ++MI)
    if (!MI->isExternal()) {
      ConstantPointerRef *CP = ConstantPointerRef::get(MI);
      vType.push_back(MI->getType());
      vConsts.push_back(CP);
    }
  
  StructType *sttype = StructType::get(vType);
  ConstantStruct *cstruct = ConstantStruct::get(sttype, vConsts);

  GlobalVariable *gb = new GlobalVariable(cstruct->getType(), true, false, 
                                          cstruct, "llvmFunctionTable");
  M.getGlobalList().push_back(gb);
  return true;  // Always modifies program
}
