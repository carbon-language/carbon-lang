//===-- EmitFunctions.cpp - interface to insert instrumentation -----------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This inserts a global constant table with function pointers all along
//
//===----------------------------------------------------------------------===//

#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/CFG.h"

namespace llvm {

enum Color{
  WHITE,
  GREY,
  BLACK
};

namespace {
  struct EmitFunctionTable : public Pass {
    bool run(Module &M);
  };
  
  RegisterOpt<EmitFunctionTable> X("emitfuncs", "Emit a Function Table");
}

char doDFS(BasicBlock * node,std::map<BasicBlock *, Color > &color){
  color[node] = GREY;

  for(succ_iterator vl = succ_begin(node), ve = succ_end(node); vl != ve; ++vl){
   
    BasicBlock *BB = *vl; 
    
    if(color[BB]!=GREY && color[BB]!=BLACK){
      if(!doDFS(BB, color)){
	return 0;
      }
    }

    //if has backedge
    else if(color[BB]==GREY)
      return 0;

  }

  color[node] = BLACK;
  return 1;
}

char hasBackEdge(Function *F){
  std::map<BasicBlock *, Color > color;
  return doDFS(F->begin(), color);
}

// Per Module pass for inserting function table
bool EmitFunctionTable::run(Module &M){
  std::vector<const Type*> vType;
 
  std::vector<Constant *> vConsts;
  std::vector<Constant *> sBCons;

  unsigned int counter = 0;
  for(Module::iterator MI = M.begin(), ME = M.end(); MI != ME; ++MI)
    if (!MI->isExternal()) {
      vType.push_back(MI->getType());
    
      //std::cerr<<MI;

      vConsts.push_back(ConstantPointerRef::get(MI));
      sBCons.push_back(ConstantInt::get(Type::SByteTy, hasBackEdge(MI)));
      
      counter++;
    }
  
  StructType *sttype = StructType::get(vType);
  Constant *cstruct = ConstantStruct::get(sttype, vConsts);

  GlobalVariable *gb = new GlobalVariable(cstruct->getType(), true,
                                          GlobalValue::ExternalLinkage, 
                                          cstruct, "llvmFunctionTable");
  M.getGlobalList().push_back(gb);

  Constant *constArray = ConstantArray::get(ArrayType::get(Type::SByteTy, 
								sBCons.size()),
						 sBCons);

  GlobalVariable *funcArray = new GlobalVariable(constArray->getType(), true,
					      GlobalValue::ExternalLinkage,
					      constArray, "llvmSimpleFunction");

  M.getGlobalList().push_back(funcArray);

  ConstantInt *cnst = ConstantSInt::get(Type::IntTy, counter); 
  GlobalVariable *fnCount = new GlobalVariable(Type::IntTy, true, 
					       GlobalValue::ExternalLinkage, 
					       cnst, "llvmFunctionCount");
  M.getGlobalList().push_back(fnCount);
  return true;  // Always modifies program
}

} // End llvm namespace
