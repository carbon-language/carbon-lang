//===-- IndMemRemoval.cpp - Remove indirect allocations and frees ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass finds places where memory allocation functions may escape into
// indirect land.  Some transforms are much easier (aka possible) only if free 
// or malloc are not called indirectly.
// Thus find places where the address of memory functions are taken and 
// construct bounce functions with direct calls of those functions.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "indmemrem"
#include "llvm/Transforms/IPO.h"
#include "llvm/Pass.h"
#include "llvm/Module.h"
#include "llvm/Instructions.h"
#include "llvm/Type.h"
#include "llvm/DerivedTypes.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Target/TargetData.h"
using namespace llvm;

STATISTIC(NumBounceSites, "Number of sites modified");
STATISTIC(NumBounce     , "Number of bounce functions created");

namespace {
  class VISIBILITY_HIDDEN IndMemRemPass : public ModulePass {
  public:
    static char ID; // Pass identification, replacement for typeid
    IndMemRemPass() : ModulePass(&ID) {}

    virtual bool runOnModule(Module &M);
  };
} // end anonymous namespace

char IndMemRemPass::ID = 0;
static RegisterPass<IndMemRemPass>
X("indmemrem","Indirect Malloc and Free Removal");

bool IndMemRemPass::runOnModule(Module &M) {
  // In theory, all direct calls of malloc and free should be promoted
  // to intrinsics.  Therefore, this goes through and finds where the
  // address of free or malloc are taken and replaces those with bounce
  // functions, ensuring that all malloc and free that might happen
  // happen through intrinsics.
  bool changed = false;
  if (Function* F = M.getFunction("free")) {
    if (F->isDeclaration() && F->arg_size() == 1 && !F->use_empty()) {
      Function* FN = Function::Create(F->getFunctionType(),
                                      GlobalValue::LinkOnceAnyLinkage,
                                      "free_llvm_bounce", &M);
      BasicBlock* bb = BasicBlock::Create(M.getContext(), "entry",FN);
      Instruction* R = ReturnInst::Create(M.getContext(), bb);
      new FreeInst(FN->arg_begin(), R);
      ++NumBounce;
      NumBounceSites += F->getNumUses();
      F->replaceAllUsesWith(FN);
      changed = true;
    }
  }
  if (Function* F = M.getFunction("malloc")) {
    if (F->isDeclaration() && F->arg_size() == 1 && !F->use_empty()) {
      TargetData* TD = getAnalysisIfAvailable<TargetData>();
      if (TD) { 
        Function* FN = Function::Create(F->getFunctionType(), 
                                        GlobalValue::LinkOnceAnyLinkage,
                                        "malloc_llvm_bounce", &M);
        F->replaceAllUsesWith(FN);
        FN->setDoesNotAlias(0);
        BasicBlock* bb = BasicBlock::Create(M.getContext(), "entry", FN);
        const Type* IntPtrTy = TD->getIntPtrType(M.getContext());
        Value* c = FN->arg_begin();
        if (FN->arg_begin()->getType() != IntPtrTy)
          c = CastInst::CreateIntegerCast(FN->arg_begin(), IntPtrTy, false,
                                          "c", bb);
        Value* a = CallInst::CreateMalloc(bb, IntPtrTy,
                                          Type::getInt8Ty(M.getContext()),
                                          c, NULL, "m");
        bb->getInstList().push_back(cast<Instruction>(a));
        ReturnInst::Create(M.getContext(), a, bb);
        ++NumBounce;
        NumBounceSites += F->getNumUses();
        changed = true;
      }
    }
  }
  return changed;
}

ModulePass *llvm::createIndMemRemPass() {
  return new IndMemRemPass();
}
