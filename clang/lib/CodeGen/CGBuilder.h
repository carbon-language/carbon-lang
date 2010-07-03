//===-- CGBuilder.h - Choose IRBuilder implementation  ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CODEGEN_CGBUILDER_H
#define CLANG_CODEGEN_CGBUILDER_H

#include "llvm/Support/IRBuilder.h"

namespace clang {
namespace CodeGen {

// Don't preserve names on values in an optimized build.
#ifdef NDEBUG
typedef llvm::IRBuilder<false> CGBuilderSuperTy;
#else
typedef llvm::IRBuilder<> CGBuilderSuperTy;
#endif

/// IR generation's wrapper around an LLVM IRBuilder.
class CGBuilderTy : public CGBuilderSuperTy {
public:
  CGBuilderTy(llvm::LLVMContext &Context) : CGBuilderSuperTy(Context) {}
  CGBuilderTy(llvm::BasicBlock *Block) : CGBuilderSuperTy(Block) {}
  CGBuilderTy(llvm::BasicBlock *Block, llvm::BasicBlock::iterator Point)
    : CGBuilderSuperTy(Block, Point) {}

  CGBuilderTy(const CGBuilderTy &Builder)
    : CGBuilderSuperTy(Builder.getContext()) {

    if (Builder.GetInsertBlock())
      SetInsertPoint(Builder.GetInsertBlock(), Builder.GetInsertPoint());
  }

  /// A saved insertion point.
  class InsertPoint {
    llvm::BasicBlock *Block;
    llvm::BasicBlock::iterator Point;

  public:
    InsertPoint(llvm::BasicBlock *Block, llvm::BasicBlock::iterator Point)
      : Block(Block), Point(Point) {}

    bool isSet() const { return (Block != 0); }
    llvm::BasicBlock *getBlock() const { return Block; }
    llvm::BasicBlock::iterator getPoint() const { return Point; }
  };

  InsertPoint saveIP() const {
    return InsertPoint(GetInsertBlock(), GetInsertPoint());
  }

  InsertPoint saveAndClearIP() {
    InsertPoint IP(GetInsertBlock(), GetInsertPoint());
    ClearInsertionPoint();
    return IP;
  }

  void restoreIP(InsertPoint IP) {
    if (IP.isSet())
      SetInsertPoint(IP.getBlock(), IP.getPoint());
    else
      ClearInsertionPoint();
  }
};

}  // end namespace CodeGen
}  // end namespace clang

#endif
