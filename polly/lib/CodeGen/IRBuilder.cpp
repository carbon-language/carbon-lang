//===------ PollyIRBuilder.cpp --------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// The Polly IRBuilder file contains Polly specific extensions for the IRBuilder
// that are used e.g. to emit the llvm.loop.parallel metadata.
//
//===----------------------------------------------------------------------===//

#include "polly/CodeGen/IRBuilder.h"

#include "polly/ScopInfo.h"
#include "polly/Support/ScopHelper.h"

#include "llvm/IR/Metadata.h"
#include "llvm/Support/Debug.h"

using namespace llvm;
using namespace polly;

/// @brief Get a self referencing id metadata node.
///
/// The MDNode looks like this (if arg0/arg1 are not null):
///
///    '!n = metadata !{metadata !n, arg0, arg1}'
///
/// @return The self referencing id metadata node.
static MDNode *getID(LLVMContext &Ctx, Value *arg0 = nullptr,
                     Value *arg1 = nullptr) {
  MDNode *ID;
  SmallVector<Value *, 3> Args;
  Args.push_back(nullptr);

  if (arg0)
    Args.push_back(arg0);
  if (arg1)
    Args.push_back(arg1);

  ID = MDNode::get(Ctx, Args);
  ID->replaceOperandWith(0, ID);
  return ID;
}

LoopAnnotator::LoopAnnotator() : SE(nullptr), AliasScopeDomain(nullptr) {}

void LoopAnnotator::buildAliasScopes(Scop &S) {
  SE = S.getSE();

  LLVMContext &Ctx = SE->getContext();
  AliasScopeDomain = getID(Ctx, MDString::get(Ctx, "polly.alias.scope.domain"));

  AliasScopeMap.clear();
  OtherAliasScopeListMap.clear();

  SetVector<Value *> BasePtrs;
  for (ScopStmt *Stmt : S)
    for (MemoryAccess *MA : *Stmt)
      BasePtrs.insert(MA->getBaseAddr());

  std::string AliasScopeStr = "polly.alias.scope.";
  for (Value *BasePtr : BasePtrs)
    AliasScopeMap[BasePtr] = getID(
        Ctx, AliasScopeDomain,
        MDString::get(Ctx, (AliasScopeStr + BasePtr->getName()).str().c_str()));

  for (Value *BasePtr : BasePtrs) {
    MDNode *AliasScopeList = MDNode::get(Ctx, {});
    for (const auto &AliasScopePair : AliasScopeMap) {
      if (BasePtr == AliasScopePair.first)
        continue;

      Value *Args = {AliasScopePair.second};
      AliasScopeList =
          MDNode::concatenate(AliasScopeList, MDNode::get(Ctx, Args));
    }

    OtherAliasScopeListMap[BasePtr] = AliasScopeList;
  }
}

void polly::LoopAnnotator::pushLoop(Loop *L, bool IsParallel) {

  ActiveLoops.push_back(L);
  if (!IsParallel)
    return;

  BasicBlock *Header = L->getHeader();
  MDNode *Id = getID(Header->getContext());
  Value *Args[] = {Id};
  MDNode *Ids = ParallelLoops.empty()
                    ? MDNode::get(Header->getContext(), Args)
                    : MDNode::concatenate(ParallelLoops.back(), Id);
  ParallelLoops.push_back(Ids);
}

void polly::LoopAnnotator::popLoop(bool IsParallel) {
  ActiveLoops.pop_back();
  if (!IsParallel)
    return;

  assert(!ParallelLoops.empty() && "Expected a parallel loop to pop");
  ParallelLoops.pop_back();
}

void polly::LoopAnnotator::annotateLoopLatch(BranchInst *B, Loop *L,
                                             bool IsParallel) const {
  if (!IsParallel)
    return;

  assert(!ParallelLoops.empty() && "Expected a parallel loop to annotate");
  MDNode *Ids = ParallelLoops.back();
  MDNode *Id = cast<MDNode>(Ids->getOperand(Ids->getNumOperands() - 1));
  B->setMetadata("llvm.loop", Id);
}

void polly::LoopAnnotator::annotate(Instruction *Inst) {
  if (!Inst->mayReadOrWriteMemory())
    return;

  // TODO: Use the ScopArrayInfo once available here.
  if (AliasScopeDomain) {
    Value *BasePtr = nullptr;
    if (isa<StoreInst>(Inst) || isa<LoadInst>(Inst)) {
      const SCEV *PtrSCEV = SE->getSCEV(getPointerOperand(*Inst));
      const SCEV *BaseSCEV = SE->getPointerBase(PtrSCEV);
      if (const SCEVUnknown *SU = dyn_cast<SCEVUnknown>(BaseSCEV))
        BasePtr = SU->getValue();
    }

    if (BasePtr) {
      Inst->setMetadata("alias.scope", AliasScopeMap[BasePtr]);
      Inst->setMetadata("noalias", OtherAliasScopeListMap[BasePtr]);
    }
  }

  if (ParallelLoops.empty())
    return;

  Inst->setMetadata("llvm.mem.parallel_loop_access", ParallelLoops.back());
}
