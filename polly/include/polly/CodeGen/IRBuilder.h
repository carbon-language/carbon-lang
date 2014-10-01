//===- Codegen/IRBuilder.h - The IR builder used by Polly -*- C++ -*-===//
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

#ifndef POLLY_CODEGEN_IRBUILDER_H
#define POLLY_CODEGEN_IRBUILDER_H

#include "llvm/IR/IRBuilder.h"
#include "llvm/Analysis/LoopInfo.h"

namespace polly {

/// @brief Helper class to annotate newly generated loops with metadata.
///
/// This stack-like structure will keep track of all loops, and annotate
/// memory instructions and loop headers according to all parallel loops.
class LoopAnnotator {
public:
  /// @brief Add a new loop @p L which is parallel if @p IsParallel is true.
  void pushLoop(llvm::Loop *L, bool IsParallel);

  /// @brief Remove the last added loop.
  void popLoop(bool isParallel);

  /// @brief Annotate the new instruction @p I for all parallel loops.
  void annotate(llvm::Instruction *I);

  /// @brief Annotate the loop latch @p B wrt. @p L.
  void annotateLoopLatch(llvm::BranchInst *B, llvm::Loop *L,
                         bool IsParallel) const;

private:
  /// @brief All loops currently under construction.
  llvm::SmallVector<llvm::Loop *, 8> ActiveLoops;

  /// @brief Metadata pointing to parallel loops currently under construction.
  llvm::SmallVector<llvm::MDNode *, 8> ParallelLoops;
};

/// @brief Add Polly specifics when running IRBuilder.
///
/// This is used to add additional items such as e.g. the llvm.loop.parallel
/// metadata.
template <bool PreserveNames>
class PollyBuilderInserter
    : protected llvm::IRBuilderDefaultInserter<PreserveNames> {
public:
  PollyBuilderInserter() : Annotator(0) {}
  PollyBuilderInserter(class LoopAnnotator &A) : Annotator(&A) {}

protected:
  void InsertHelper(llvm::Instruction *I, const llvm::Twine &Name,
                    llvm::BasicBlock *BB,
                    llvm::BasicBlock::iterator InsertPt) const {
    llvm::IRBuilderDefaultInserter<PreserveNames>::InsertHelper(I, Name, BB,
                                                                InsertPt);
    if (Annotator)
      Annotator->annotate(I);
  }

private:
  class LoopAnnotator *Annotator;
};

// TODO: We should not name instructions in NDEBUG builds.
//
// We currently always name instructions, as the polly test suite currently
// matches for certain names.
typedef PollyBuilderInserter<true> IRInserter;
typedef llvm::IRBuilder<true, llvm::ConstantFolder, IRInserter> PollyIRBuilder;

/// @brief Return an IR builder pointed before the @p BB terminator.
static inline PollyIRBuilder createPollyIRBuilder(llvm::BasicBlock *BB,
                                                  LoopAnnotator &LA) {
  PollyIRBuilder Builder(BB->getContext(), llvm::ConstantFolder(),
                         polly::IRInserter(LA));
  Builder.SetInsertPoint(BB->getTerminator());
  return Builder;
}
}
#endif
