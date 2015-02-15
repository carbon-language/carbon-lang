//===---- CGLoopInfo.h - LLVM CodeGen for loop metadata -*- C++ -*---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is the internal state used for llvm translation for loop statement
// metadata.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CODEGEN_CGLOOPINFO_H
#define LLVM_CLANG_LIB_CODEGEN_CGLOOPINFO_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Compiler.h"

namespace llvm {
class BasicBlock;
class Instruction;
class MDNode;
} // end namespace llvm

namespace clang {
namespace CodeGen {

/// \brief Attributes that may be specified on loops.
struct LoopAttributes {
  explicit LoopAttributes(bool IsParallel = false);
  void clear();

  /// \brief Generate llvm.loop.parallel metadata for loads and stores.
  bool IsParallel;

  /// \brief Values of llvm.loop.vectorize.enable metadata.
  enum LVEnableState { VecUnspecified, VecEnable, VecDisable };

  /// \brief llvm.loop.vectorize.enable
  LVEnableState VectorizerEnable;

  /// \brief llvm.loop.vectorize.width
  unsigned VectorizerWidth;

  /// \brief llvm.loop.interleave.count
  unsigned VectorizerUnroll;
};

/// \brief Information used when generating a structured loop.
class LoopInfo {
public:
  /// \brief Construct a new LoopInfo for the loop with entry Header.
  LoopInfo(llvm::BasicBlock *Header, const LoopAttributes &Attrs);

  /// \brief Get the loop id metadata for this loop.
  llvm::MDNode *getLoopID() const { return LoopID; }

  /// \brief Get the header block of this loop.
  llvm::BasicBlock *getHeader() const { return Header; }

  /// \brief Get the set of attributes active for this loop.
  const LoopAttributes &getAttributes() const { return Attrs; }

private:
  /// \brief Loop ID metadata.
  llvm::MDNode *LoopID;
  /// \brief Header block of this loop.
  llvm::BasicBlock *Header;
  /// \brief The attributes for this loop.
  LoopAttributes Attrs;
};

/// \brief A stack of loop information corresponding to loop nesting levels.
/// This stack can be used to prepare attributes which are applied when a loop
/// is emitted.
class LoopInfoStack {
  LoopInfoStack(const LoopInfoStack &) = delete;
  void operator=(const LoopInfoStack &) = delete;

public:
  LoopInfoStack() {}

  /// \brief Begin a new structured loop. The set of staged attributes will be
  /// applied to the loop and then cleared.
  void push(llvm::BasicBlock *Header);

  /// \brief End the current loop.
  void pop();

  /// \brief Return the top loop id metadata.
  llvm::MDNode *getCurLoopID() const { return getInfo().getLoopID(); }

  /// \brief Return true if the top loop is parallel.
  bool getCurLoopParallel() const {
    return hasInfo() ? getInfo().getAttributes().IsParallel : false;
  }

  /// \brief Function called by the CodeGenFunction when an instruction is
  /// created.
  void InsertHelper(llvm::Instruction *I) const;

  /// \brief Set the next pushed loop as parallel.
  void setParallel(bool Enable = true) { StagedAttrs.IsParallel = Enable; }

  /// \brief Set the next pushed loop 'vectorizer.enable'
  void setVectorizerEnable(bool Enable = true) {
    StagedAttrs.VectorizerEnable =
        Enable ? LoopAttributes::VecEnable : LoopAttributes::VecDisable;
  }

  /// \brief Set the vectorizer width for the next loop pushed.
  void setVectorizerWidth(unsigned W) { StagedAttrs.VectorizerWidth = W; }

  /// \brief Set the vectorizer unroll for the next loop pushed.
  void setVectorizerUnroll(unsigned U) { StagedAttrs.VectorizerUnroll = U; }

private:
  /// \brief Returns true if there is LoopInfo on the stack.
  bool hasInfo() const { return !Active.empty(); }
  /// \brief Return the LoopInfo for the current loop. HasInfo should be called
  /// first to ensure LoopInfo is present.
  const LoopInfo &getInfo() const { return Active.back(); }
  /// \brief The set of attributes that will be applied to the next pushed loop.
  LoopAttributes StagedAttrs;
  /// \brief Stack of active loops.
  llvm::SmallVector<LoopInfo, 4> Active;
};

} // end namespace CodeGen
} // end namespace clang

#endif
