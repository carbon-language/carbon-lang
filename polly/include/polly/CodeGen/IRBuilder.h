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
namespace polly {

/// @brief Keeps information about generated loops.
class PollyLoopInfo {
public:
  PollyLoopInfo(llvm::BasicBlock *Header)
      : LoopID(0), Header(Header), Parallel(false) {}

  /// @brief Get the loop id metadata node.
  ///
  /// Each loop is identified by a self referencing metadata node of the form:
  ///
  ///    '!n = metadata !{metadata !n}'
  ///
  /// This functions creates such metadata on demand if not yet available.
  ///
  /// @return The loop id metadata node.
  llvm::MDNode *GetLoopID() const;

  /// @brief Get the head basic block of this loop.
  llvm::BasicBlock *GetHeader() const { return Header; }

  /// @brief Check if the loop is parallel.
  ///
  /// @return True, if the loop is parallel.
  bool IsParallel() const { return Parallel; }

  /// @brief Set a loop as parallel.
  ///
  /// @IsParallel True, if the loop is to be marked as parallel. False, if the
  //              loop should be marked sequential.
  void SetParallel(bool IsParallel = true) { Parallel = IsParallel; }

private:
  mutable llvm::MDNode *LoopID;
  llvm::BasicBlock *Header;
  bool Parallel;
};

class LoopAnnotator {
public:
  void Begin(llvm::BasicBlock *Header);
  void SetCurrentParallel();
  void End();
  void Annotate(llvm::Instruction *I);

private:
  std::vector<PollyLoopInfo> Active;
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
      Annotator->Annotate(I);
  }

private:
  class LoopAnnotator *Annotator;
};

// TODO: We should not name instructions in NDEBUG builds.
//
// We currently always name instructions, as the polly test suite currently
// matches for certain names.
//
// typedef PollyBuilderInserter<false> IRInserter;
// typedef llvm::IRBuilder<false, llvm::ConstantFolder, IRInserter>
// PollyIRBuilder;
typedef PollyBuilderInserter<true> IRInserter;
typedef llvm::IRBuilder<true, llvm::ConstantFolder, IRInserter> PollyIRBuilder;
}
#endif
