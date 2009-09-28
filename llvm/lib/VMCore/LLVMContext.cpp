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
//  class LLVMContextImpl.
//
//===----------------------------------------------------------------------===//

#include "llvm/LLVMContext.h"
#include "llvm/Metadata.h"
#include "llvm/Constants.h"
#include "llvm/Instruction.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/ValueHandle.h"
#include "LLVMContextImpl.h"
#include <set>

using namespace llvm;

static ManagedStatic<LLVMContext> GlobalContext;

LLVMContext& llvm::getGlobalContext() {
  return *GlobalContext;
}

LLVMContext::LLVMContext() : pImpl(new LLVMContextImpl(*this)) { }
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
  std::vector<WeakVH> DeadMDNodes;
  bool Changed = false;
  while (1) {

    for (FoldingSet<MDNode>::iterator 
           I = pImpl->MDNodeSet.begin(),
           E = pImpl->MDNodeSet.end(); I != E; ++I) {
      MDNode *N = &(*I);
      if (N->use_empty()) 
        DeadMDNodes.push_back(WeakVH(N));
    }
    
    if (DeadMDNodes.empty())
      return Changed;

    while (!DeadMDNodes.empty()) {
      Value *V = DeadMDNodes.back(); DeadMDNodes.pop_back();
      if (const MDNode *N = dyn_cast_or_null<MDNode>(V))
        if (N->use_empty())
          delete N;
    }
  }
  return Changed;
}

MetadataContext &LLVMContext::getMetadata() {
  return pImpl->TheMetadata;
}
