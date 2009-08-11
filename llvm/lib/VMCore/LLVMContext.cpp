//===-- LLVMContext.cpp - Implement LLVMContext -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements LLVMContext, as a wrapper around the opaque
// struct LLVMContextImpl.
//
//===----------------------------------------------------------------------===//

#include "llvm/LLVMContext.h"
#include "llvm/Constants.h"
#include "llvm/Instruction.h"
#include "llvm/Support/ManagedStatic.h"
#include "LLVMContextImpl.h"
#include <set>

using namespace llvm;

static ManagedStatic<LLVMContext> GlobalContext;

LLVMContext& llvm::getGlobalContext() {
  return *GlobalContext;
}

LLVMContext::LLVMContext() : pImpl(new LLVMContextImpl()) { }
LLVMContext::~LLVMContext() { delete pImpl; }

GetElementPtrConstantExpr::GetElementPtrConstantExpr
  (Constant *C,
   const std::vector<Constant*> &IdxList,
   const Type *DestTy)
    : ConstantExpr(DestTy, Instruction::GetElementPtr,
                   OperandTraits<GetElementPtrConstantExpr>::op_end(this)
                   - (IdxList.size()+1),
                   IdxList.size()+1) {
  OperandList[0] = C;
  for (unsigned i = 0, E = IdxList.size(); i != E; ++i)
    OperandList[i+1] = IdxList[i];
}

bool LLVMContext::RemoveDeadMetadata() {
  std::vector<const MDNode *> DeadMDNodes;
  bool Changed = false;
  while (1) {

    for (LLVMContextImpl::MDNodeMapTy::MapTy::iterator
           I = pImpl->MDNodes.map_begin(),
           E = pImpl->MDNodes.map_end(); I != E; ++I) {
      const MDNode *N = cast<MDNode>(I->second);
      if (N->use_empty()) 
        DeadMDNodes.push_back(N);
    }
    
    if (DeadMDNodes.empty())
      return Changed;

    while (!DeadMDNodes.empty()) {
      const MDNode *N = DeadMDNodes.back(); DeadMDNodes.pop_back();
      delete N;
    }
  }
  return Changed;
}
