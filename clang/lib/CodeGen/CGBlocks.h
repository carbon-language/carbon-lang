//===-- CGBlocks.h - state for LLVM CodeGen for blocks ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is the internal state used for llvm translation for block literals.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CODEGEN_CGBLOCKS_H
#define CLANG_CODEGEN_CGBLOCKS_H

#include "CodeGenTypes.h"
#include "clang/AST/Type.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"

#include <vector>
#include <map>

#include "CGBuilder.h"
#include "CGCall.h"
#include "CGValue.h"

namespace llvm {
  class Module;
  class Constant;
  class Function;
  class GlobalValue;
  class TargetData;
  class FunctionType;
  class Value;
}

namespace clang {

namespace CodeGen {
class CodeGenModule;

class BlockBase {
public:
    enum {
        BLOCK_NEEDS_FREE =        (1 << 24),
        BLOCK_HAS_COPY_DISPOSE =  (1 << 25),
        BLOCK_HAS_CXX_OBJ =       (1 << 26),
        BLOCK_IS_GC =             (1 << 27),
        BLOCK_IS_GLOBAL =         (1 << 28),
        BLOCK_HAS_DESCRIPTOR =    (1 << 29)
    };
};

class BlockModule : public BlockBase {
  ASTContext &Context;
  llvm::Module &TheModule;
  const llvm::TargetData &TheTargetData;
  CodeGenTypes &Types;
  CodeGenModule &CGM;
  
  ASTContext &getContext() const { return Context; }
  llvm::Module &getModule() const { return TheModule; }
  CodeGenTypes &getTypes() { return Types; }
  const llvm::TargetData &getTargetData() const { return TheTargetData; }
public:
  llvm::Constant *getNSConcreteGlobalBlock();
  llvm::Constant *getNSConcreteStackBlock();
  int getGlobalUniqueCount() { return ++Block.GlobalUniqueCount; }
  const llvm::Type *getBlockDescriptorType();

  const llvm::Type *getGenericBlockLiteralType();
  const llvm::Type *getGenericExtendedBlockLiteralType();

  llvm::Constant *GetAddrOfGlobalBlock(const BlockExpr *BE, const char *);

  /// NSConcreteGlobalBlock - Cached reference to the class pointer for global
  /// blocks.
  llvm::Constant *NSConcreteGlobalBlock;

  /// NSConcreteStackBlock - Cached reference to the class poinnter for stack
  /// blocks.
  llvm::Constant *NSConcreteStackBlock;
  
  const llvm::Type *BlockDescriptorType;
  const llvm::Type *GenericBlockLiteralType;
  const llvm::Type *GenericExtendedBlockLiteralType;
  struct {
    int GlobalUniqueCount;
  } Block;

  llvm::Value *BlockObjectDispose;
  const llvm::Type *PtrToInt8Ty;

  BlockModule(ASTContext &C, llvm::Module &M, const llvm::TargetData &TD,
              CodeGenTypes &T, CodeGenModule &CodeGen)
    : Context(C), TheModule(M), TheTargetData(TD), Types(T),
      CGM(CodeGen),
      NSConcreteGlobalBlock(0), NSConcreteStackBlock(0), BlockDescriptorType(0),
      GenericBlockLiteralType(0), BlockObjectDispose(0) {
    Block.GlobalUniqueCount = 0;
    PtrToInt8Ty = llvm::PointerType::getUnqual(llvm::Type::Int8Ty);
  }
};

class BlockFunction : public BlockBase {
  CodeGenModule &CGM;

public:
  const llvm::Type *PtrToInt8Ty;
  enum {
    BLOCK_FIELD_IS_OBJECT   =  3,  /* id, NSObject, __attribute__((NSObject)),
                                      block, ... */
    BLOCK_FIELD_IS_BLOCK    =  7,  /* a block variable */
    BLOCK_FIELD_IS_BYREF    =  8,  /* the on stack structure holding the __block
                                      variable */
    BLOCK_FIELD_IS_WEAK     = 16,  /* declared __weak, only used in byref copy
                                      helpers */
    BLOCK_BYREF_CALLER      = 128  /* called from __block (byref) copy/dispose
                                      support routines */
  };

  /// BlockInfo - Information to generate a block literal.
  struct BlockInfo {
    /// BlockLiteralTy - The type of the block literal.
    const llvm::Type *BlockLiteralTy;

    /// Name - the name of the function this block was created for, if any
    const char *Name;

    /// ByCopyDeclRefs - Variables from parent scopes that have been imported
    /// into this block.
    llvm::SmallVector<const BlockDeclRefExpr *, 8> ByCopyDeclRefs;
    
    // ByRefDeclRefs - __block variables from parent scopes that have been 
    // imported into this block.
    llvm::SmallVector<const BlockDeclRefExpr *, 8> ByRefDeclRefs;
    
    BlockInfo(const llvm::Type *blt, const char *n)
      : BlockLiteralTy(blt), Name(n) {}
  };

  CGBuilderTy &Builder;

  BlockFunction(CodeGenModule &cgm, CGBuilderTy &B)
    : CGM(cgm), Builder(B) {
    PtrToInt8Ty = llvm::PointerType::getUnqual(llvm::Type::Int8Ty);
  }

  ImplicitParamDecl *BlockStructDecl;
  ImplicitParamDecl *getBlockStructDecl() { return BlockStructDecl; }

  llvm::Value *BuildCopyHelper(int flag);
  llvm::Value *BuildDestroyHelper(int flag);

  llvm::Value *getBlockObjectDispose();
  void BuildBlockRelease(const VarDecl &D, llvm::Value *DeclPtr);
};

}  // end namespace CodeGen
}  // end namespace clang

#endif
