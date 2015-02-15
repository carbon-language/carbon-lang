//===-- CGBuilder.h - Choose IRBuilder implementation  ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CODEGEN_CGBUILDER_H
#define LLVM_CLANG_LIB_CODEGEN_CGBUILDER_H

#include "llvm/IR/IRBuilder.h"

namespace clang {
namespace CodeGen {

class CodeGenFunction;

/// \brief This is an IRBuilder insertion helper that forwards to
/// CodeGenFunction::InsertHelper, which adds necessary metadata to
/// instructions.
template <bool PreserveNames>
class CGBuilderInserter
  : protected llvm::IRBuilderDefaultInserter<PreserveNames> {
public:
  CGBuilderInserter() : CGF(nullptr) {}
  explicit CGBuilderInserter(CodeGenFunction *CGF) : CGF(CGF) {}

protected:
  /// \brief This forwards to CodeGenFunction::InsertHelper.
  void InsertHelper(llvm::Instruction *I, const llvm::Twine &Name,
                    llvm::BasicBlock *BB,
                    llvm::BasicBlock::iterator InsertPt) const;
private:
  void operator=(const CGBuilderInserter &) = delete;

  CodeGenFunction *CGF;
};

// Don't preserve names on values in an optimized build.
#ifdef NDEBUG
#define PreserveNames false
#else
#define PreserveNames true
#endif
typedef CGBuilderInserter<PreserveNames> CGBuilderInserterTy;
typedef llvm::IRBuilder<PreserveNames, llvm::ConstantFolder,
                        CGBuilderInserterTy> CGBuilderTy;
#undef PreserveNames

}  // end namespace CodeGen
}  // end namespace clang

#endif
