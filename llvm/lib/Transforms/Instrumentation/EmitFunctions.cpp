//===-- EmitFunctions.cpp - interface to insert instrumentation --*- C++ -*--=//
//
// This inserts a global constant table with function pointers all along
//
//===----------------------------------------------------------------------===//

#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"

namespace {
  struct EmitFunctionTable : public Pass {
    bool run(Module &M);
  };
  
  RegisterOpt<EmitFunctionTable> X("emitfuncs", "Emit a Function Table");
}

// Per Module pass for inserting function table
bool EmitFunctionTable::run(Module &M){
  std::vector<const Type*> vType;
  std::vector<Constant *> vConsts;
  unsigned char counter = 0;
  for(Module::iterator MI = M.begin(), ME = M.end(); MI != ME; ++MI)
    if (!MI->isExternal()) {
      vType.push_back(MI->getType());
      vConsts.push_back(ConstantPointerRef::get(MI));
      counter++;
    }
  
  StructType *sttype = StructType::get(vType);
  ConstantStruct *cstruct = ConstantStruct::get(sttype, vConsts);

  GlobalVariable *gb = new GlobalVariable(cstruct->getType(), true,
                                          GlobalValue::ExternalLinkage, 
                                          cstruct, "llvmFunctionTable");
  M.getGlobalList().push_back(gb);

  ConstantInt *cnst = ConstantSInt::get(Type::IntTy, counter); 
  GlobalVariable *fnCount = new GlobalVariable(Type::IntTy, true, 
					       GlobalValue::ExternalLinkage, 
					       cnst, "llvmFunctionCount");
  M.getGlobalList().push_back(fnCount);
  return true;  // Always modifies program
}
