//===-- InstructionPrecedenceTracking.cpp -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Implements a class that is able to define some instructions as "special"
// (e.g. as having implicit control flow, or writing memory, or having another
// interesting property) and then efficiently answers queries of the types:
// 1. Are there any special instructions in the block of interest?
// 2. Return first of the special instructions in the given block;
// 3. Check if the given instruction is preceeded by the first special
//    instruction in the same block.
// The class provides caching that allows to answer these queries quickly. The
// user must make sure that the cached data is invalidated properly whenever
// a content of some tracked block is changed.
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/InstructionPrecedenceTracking.h"
#include "llvm/Analysis/ValueTracking.h"

using namespace llvm;

#ifndef NDEBUG
static cl::opt<bool> ExpensiveAsserts(
    "ipt-expensive-asserts",
    cl::desc("Perform expensive assert validation on every query to Instruction"
             " Precedence Tracking"),
    cl::init(false), cl::Hidden);
#endif

const Instruction *InstructionPrecedenceTracking::getFirstSpecialInstruction(
    const BasicBlock *BB) {
#ifndef NDEBUG
  // If there is a bug connected to invalid cache, turn on ExpensiveAsserts to
  // catch this situation as early as possible.
  if (ExpensiveAsserts)
    validateAll();
  else
    validate(BB);
#endif

  if (!KnownBlocks.count(BB))
    fill(BB);
  auto *FirstICF = FirstSpecialInsts.lookup(BB);
  assert((!FirstICF || FirstICF->getParent() == BB) && "Inconsistent cache!");
  return FirstICF;
}

bool InstructionPrecedenceTracking::hasSpecialInstructions(
    const BasicBlock *BB) {
  return getFirstSpecialInstruction(BB) != nullptr;
}

bool InstructionPrecedenceTracking::isPreceededBySpecialInstruction(
    const Instruction *Insn) {
  const Instruction *MaybeFirstICF =
      getFirstSpecialInstruction(Insn->getParent());
  return MaybeFirstICF && OI.dominates(MaybeFirstICF, Insn);
}

void InstructionPrecedenceTracking::fill(const BasicBlock *BB) {
  FirstSpecialInsts.erase(BB);
  for (auto &I : *BB)
    if (isSpecialInstruction(&I)) {
      FirstSpecialInsts[BB] = &I;
      break;
    }

  // Mark this block as having a known result.
  KnownBlocks.insert(BB);
}

#ifndef NDEBUG
void InstructionPrecedenceTracking::validate(const BasicBlock *BB) const {
  // If we don't know anything about this block, make sure we don't store
  // a bucket for it in FirstSpecialInsts map.
  if (!KnownBlocks.count(BB)) {
    assert(FirstSpecialInsts.find(BB) == FirstSpecialInsts.end() && "Must be!");
    return;
  }

  auto It = FirstSpecialInsts.find(BB);
  bool BlockHasSpecialInsns = false;
  for (const Instruction &Insn : *BB) {
    if (isSpecialInstruction(&Insn)) {
      assert(It != FirstSpecialInsts.end() &&
             "Blocked marked as known but we have no cached value for it!");
      assert(It->second == &Insn &&
             "Cached first special instruction is wrong!");
      BlockHasSpecialInsns = true;
      break;
    }
  }
  if (!BlockHasSpecialInsns)
    assert(It == FirstSpecialInsts.end() &&
           "Block is marked as having special instructions but in fact it "
           "has none!");
}

void InstructionPrecedenceTracking::validateAll() const {
  // Check that for every known block the cached value is correct.
  for (auto *BB : KnownBlocks)
    validate(BB);

  // Check that all blocks with cached values are marked as known.
  for (auto &It : FirstSpecialInsts)
    assert(KnownBlocks.count(It.first) &&
           "We have a cached value but the block is not marked as known?");
}
#endif

void InstructionPrecedenceTracking::invalidateBlock(const BasicBlock *BB) {
  OI.invalidateBlock(BB);
  FirstSpecialInsts.erase(BB);
  KnownBlocks.erase(BB);
}

void InstructionPrecedenceTracking::clear() {
  for (auto It : FirstSpecialInsts)
    OI.invalidateBlock(It.first);
  FirstSpecialInsts.clear();
  KnownBlocks.clear();
#ifndef NDEBUG
  // The map should be valid after clearing (at least empty).
  validateAll();
#endif
}

bool ImplicitControlFlowTracking::isSpecialInstruction(
    const Instruction *Insn) const {
  // If a block's instruction doesn't always pass the control to its successor
  // instruction, mark the block as having implicit control flow. We use them
  // to avoid wrong assumptions of sort "if A is executed and B post-dominates
  // A, then B is also executed". This is not true is there is an implicit
  // control flow instruction (e.g. a guard) between them.
  //
  // TODO: Currently, isGuaranteedToTransferExecutionToSuccessor returns false
  // for volatile stores and loads because they can trap. The discussion on
  // whether or not it is correct is still ongoing. We might want to get rid
  // of this logic in the future. Anyways, trapping instructions shouldn't
  // introduce implicit control flow, so we explicitly allow them here. This
  // must be removed once isGuaranteedToTransferExecutionToSuccessor is fixed.
  if (isGuaranteedToTransferExecutionToSuccessor(Insn))
    return false;
  if (isa<LoadInst>(Insn)) {
    assert(cast<LoadInst>(Insn)->isVolatile() &&
           "Non-volatile load should transfer execution to successor!");
    return false;
  }
  if (isa<StoreInst>(Insn)) {
    assert(cast<StoreInst>(Insn)->isVolatile() &&
           "Non-volatile store should transfer execution to successor!");
    return false;
  }
  return true;
}
