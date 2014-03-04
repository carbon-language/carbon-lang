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
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/Debug.h"

using namespace llvm;
using namespace polly;

llvm::MDNode *polly::PollyLoopInfo::GetLoopID() const {
  if (LoopID)
    return LoopID;

  llvm::Value *Args[] = {0};
  LoopID = llvm::MDNode::get(Header->getContext(), Args);
  LoopID->replaceOperandWith(0, LoopID);
  return LoopID;
}

void polly::LoopAnnotator::Begin(llvm::BasicBlock *Header) {
  Active.push_back(PollyLoopInfo(Header));
}

void polly::LoopAnnotator::End() { Active.pop_back(); }

void polly::LoopAnnotator::SetCurrentParallel() {
  Active.back().SetParallel(true);
}

void polly::LoopAnnotator::Annotate(llvm::Instruction *Inst) {
  if (Active.empty())
    return;

  const PollyLoopInfo &L = Active.back();
  if (!L.IsParallel())
    return;

  if (TerminatorInst *TI = dyn_cast<llvm::TerminatorInst>(Inst)) {
    for (unsigned i = 0, ie = TI->getNumSuccessors(); i != ie; ++i)
      if (TI->getSuccessor(i) == L.GetHeader()) {
        TI->setMetadata("llvm.loop", L.GetLoopID());
        break;
      }
  } else if (Inst->mayReadOrWriteMemory()) {
    Inst->setMetadata("llvm.mem.parallel_loop_access", L.GetLoopID());
  }
}
