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
#include "llvm/Module.h"
#include "llvm/ADT/SmallVector.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/AST/CharUnits.h"
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
  class PointerType;
  class Value;
  class LLVMContext;
}

namespace clang {

namespace CodeGen {
class CodeGenModule;
class CGBlockInfo;

class BlockBase {
public:
    enum {
        BLOCK_HAS_COPY_DISPOSE =  (1 << 25),
        BLOCK_HAS_CXX_OBJ =       (1 << 26),
        BLOCK_IS_GLOBAL =         (1 << 28),
        BLOCK_USE_STRET =         (1 << 29),
        BLOCK_HAS_SIGNATURE  =    (1 << 30)
    };
};


class BlockModule : public BlockBase {
  ASTContext &Context;
  llvm::Module &TheModule;
  const llvm::TargetData &TheTargetData;
  CodeGenTypes &Types;
  CodeGenModule &CGM;
  llvm::LLVMContext &VMContext;

  ASTContext &getContext() const { return Context; }
  llvm::Module &getModule() const { return TheModule; }
  CodeGenTypes &getTypes() { return Types; }
  const llvm::TargetData &getTargetData() const { return TheTargetData; }
public:
  int getGlobalUniqueCount() { return ++Block.GlobalUniqueCount; }
  const llvm::Type *getBlockDescriptorType();

  const llvm::Type *getGenericBlockLiteralType();

  llvm::Constant *GetAddrOfGlobalBlock(const BlockExpr *BE, const char *);

  const llvm::Type *BlockDescriptorType;
  const llvm::Type *GenericBlockLiteralType;

  struct {
    int GlobalUniqueCount;
  } Block;

  const llvm::PointerType *PtrToInt8Ty;

  std::map<uint64_t, llvm::Constant *> AssignCache;
  std::map<uint64_t, llvm::Constant *> DestroyCache;

  BlockModule(ASTContext &C, llvm::Module &M, const llvm::TargetData &TD,
              CodeGenTypes &T, CodeGenModule &CodeGen)
    : Context(C), TheModule(M), TheTargetData(TD), Types(T),
      CGM(CodeGen), VMContext(M.getContext()),
      BlockDescriptorType(0), GenericBlockLiteralType(0) {
    Block.GlobalUniqueCount = 0;
    PtrToInt8Ty = llvm::Type::getInt8PtrTy(M.getContext());
  }
};

class BlockFunction : public BlockBase {
  CodeGenModule &CGM;
  ASTContext &getContext() const;

protected:
  llvm::LLVMContext &VMContext;

public:
  CodeGenFunction &CGF;

  const CodeGen::CGBlockInfo *BlockInfo;
  llvm::Value *BlockPointer;

  const llvm::PointerType *PtrToInt8Ty;
  struct HelperInfo {
    int index;
    int flag;
    const BlockDeclRefExpr *cxxvar_import;
    bool RequiresCopying;
  };

  enum {
    BLOCK_FIELD_IS_OBJECT   =  3,  /* id, NSObject, __attribute__((NSObject)),
                                      block, ... */
    BLOCK_FIELD_IS_BLOCK    =  7,  /* a block variable */
    BLOCK_FIELD_IS_BYREF    =  8,  /* the on stack structure holding the __block
                                      variable */
    BLOCK_FIELD_IS_WEAK     = 16,  /* declared __weak, only used in byref copy
                                      helpers */
    BLOCK_BYREF_CALLER      = 128,  /* called from __block (byref) copy/dispose
                                      support routines */
    BLOCK_BYREF_CURRENT_MAX = 256
  };

  CGBuilderTy &Builder;

  BlockFunction(CodeGenModule &cgm, CodeGenFunction &cgf, CGBuilderTy &B);

  llvm::Constant *GenerateCopyHelperFunction(const CGBlockInfo &blockInfo);
  llvm::Constant *GenerateDestroyHelperFunction(const CGBlockInfo &blockInfo);

  llvm::Constant *GeneratebyrefCopyHelperFunction(const llvm::Type *, int flag,
                                                  const VarDecl *BD);
  llvm::Constant *GeneratebyrefDestroyHelperFunction(const llvm::Type *T, 
                                                     int flag, 
                                                     const VarDecl *BD);

  llvm::Constant *BuildbyrefCopyHelper(const llvm::Type *T, uint32_t flags,
                                       unsigned Align, const VarDecl *BD);
  llvm::Constant *BuildbyrefDestroyHelper(const llvm::Type *T, uint32_t flags,
                                          unsigned Align, const VarDecl *BD);

  void BuildBlockRelease(llvm::Value *DeclPtr, uint32_t flags);
};

}  // end namespace CodeGen
}  // end namespace clang

#endif
