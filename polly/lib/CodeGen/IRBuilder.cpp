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

#include "llvm/IR/Metadata.h"
#include "llvm/Support/Debug.h"

using namespace llvm;
using namespace polly;

/// @brief Get the loop id metadata node.
///
/// Each loop is identified by a self referencing metadata node of the form:
///
///    '!n = metadata !{metadata !n}'
///
/// This functions creates such metadata on demand if not yet available.
///
/// @return The loop id metadata node.
static MDNode *getLoopID(Loop *L) {
  Value *Args[] = {0};
  MDNode *LoopID = MDNode::get(L->getHeader()->getContext(), Args);
  LoopID->replaceOperandWith(0, LoopID);
  return LoopID;
}

void polly::LoopAnnotator::pushLoop(Loop *L, bool IsParallel) {
  ActiveLoops.push_back(L);
  if (!IsParallel)
    return;

  BasicBlock *Header = L->getHeader();
  MDNode *Id = getLoopID(L);
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
  if (!Inst->mayReadOrWriteMemory() || ParallelLoops.empty())
    return;

  Inst->setMetadata("llvm.mem.parallel_loop_access", ParallelLoops.back());
}
