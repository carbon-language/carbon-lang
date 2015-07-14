//===---- CGLoopInfo.cpp - LLVM CodeGen for loop metadata -*- C++ -*-------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CGLoopInfo.h"
#include "clang/AST/Attr.h"
#include "clang/Sema/LoopHint.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
using namespace clang::CodeGen;
using namespace llvm;

static MDNode *createMetadata(LLVMContext &Ctx, const LoopAttributes &Attrs) {

  if (!Attrs.IsParallel && Attrs.VectorizeWidth == 0 &&
      Attrs.InterleaveCount == 0 &&
      Attrs.VectorizeEnable == LoopAttributes::Unspecified)
    return nullptr;

  SmallVector<Metadata *, 4> Args;
  // Reserve operand 0 for loop id self reference.
  auto TempNode = MDNode::getTemporary(Ctx, None);
  Args.push_back(TempNode.get());

  // Setting vectorize.width
  if (Attrs.VectorizeWidth > 0) {
    Metadata *Vals[] = {MDString::get(Ctx, "llvm.loop.vectorize.width"),
                        ConstantAsMetadata::get(ConstantInt::get(
                            Type::getInt32Ty(Ctx), Attrs.VectorizeWidth))};
    Args.push_back(MDNode::get(Ctx, Vals));
  }

  // Setting interleave.count
  if (Attrs.InterleaveCount > 0) {
    Metadata *Vals[] = {MDString::get(Ctx, "llvm.loop.interleave.count"),
                        ConstantAsMetadata::get(ConstantInt::get(
                            Type::getInt32Ty(Ctx), Attrs.InterleaveCount))};
    Args.push_back(MDNode::get(Ctx, Vals));
  }

  // Setting vectorize.enable
  if (Attrs.VectorizeEnable != LoopAttributes::Unspecified) {
    Metadata *Vals[] = {MDString::get(Ctx, "llvm.loop.vectorize.enable"),
                        ConstantAsMetadata::get(ConstantInt::get(
                            Type::getInt1Ty(Ctx), (Attrs.VectorizeEnable ==
                                                   LoopAttributes::Enable)))};
    Args.push_back(MDNode::get(Ctx, Vals));
  }

  // Set the first operand to itself.
  MDNode *LoopID = MDNode::get(Ctx, Args);
  LoopID->replaceOperandWith(0, LoopID);
  return LoopID;
}

LoopAttributes::LoopAttributes(bool IsParallel)
    : IsParallel(IsParallel), VectorizeEnable(LoopAttributes::Unspecified),
      VectorizeWidth(0), InterleaveCount(0) {}

void LoopAttributes::clear() {
  IsParallel = false;
  VectorizeWidth = 0;
  InterleaveCount = 0;
  VectorizeEnable = LoopAttributes::Unspecified;
}

LoopInfo::LoopInfo(BasicBlock *Header, const LoopAttributes &Attrs)
    : LoopID(nullptr), Header(Header), Attrs(Attrs) {
  LoopID = createMetadata(Header->getContext(), Attrs);
}

void LoopInfoStack::push(BasicBlock *Header,
                         ArrayRef<const clang::Attr *> Attrs) {
  for (const auto *Attr : Attrs) {
    const LoopHintAttr *LH = dyn_cast<LoopHintAttr>(Attr);

    // Skip non loop hint attributes
    if (!LH)
      continue;

    LoopHintAttr::OptionType Option = LH->getOption();
    LoopHintAttr::LoopHintState State = LH->getState();
    switch (Option) {
    case LoopHintAttr::Vectorize:
    case LoopHintAttr::Interleave:
      if (State == LoopHintAttr::AssumeSafety) {
        // Apply "llvm.mem.parallel_loop_access" metadata to load/stores.
        setParallel(true);
      }
      break;
    case LoopHintAttr::VectorizeWidth:
    case LoopHintAttr::InterleaveCount:
    case LoopHintAttr::Unroll:
    case LoopHintAttr::UnrollCount:
      // Nothing to do here for these loop hints.
      break;
    }
  }

  Active.push_back(LoopInfo(Header, StagedAttrs));
  // Clear the attributes so nested loops do not inherit them.
  StagedAttrs.clear();
}

void LoopInfoStack::pop() {
  assert(!Active.empty() && "No active loops to pop");
  Active.pop_back();
}

void LoopInfoStack::InsertHelper(Instruction *I) const {
  if (!hasInfo())
    return;

  const LoopInfo &L = getInfo();
  if (!L.getLoopID())
    return;

  if (TerminatorInst *TI = dyn_cast<TerminatorInst>(I)) {
    for (unsigned i = 0, ie = TI->getNumSuccessors(); i < ie; ++i)
      if (TI->getSuccessor(i) == L.getHeader()) {
        TI->setMetadata("llvm.loop", L.getLoopID());
        break;
      }
    return;
  }

  if (L.getAttributes().IsParallel && I->mayReadOrWriteMemory())
    I->setMetadata("llvm.mem.parallel_loop_access", L.getLoopID());
}
