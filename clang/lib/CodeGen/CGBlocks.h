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
  llvm::Constant *getNSConcreteGlobalBlock();
  llvm::Constant *getNSConcreteStackBlock();
  int getGlobalUniqueCount() { return ++Block.GlobalUniqueCount; }
  const llvm::Type *getBlockDescriptorType();

  const llvm::Type *getGenericBlockLiteralType();

  llvm::Constant *GetAddrOfGlobalBlock(const BlockExpr *BE, const char *);

  /// NSConcreteGlobalBlock - Cached reference to the class pointer for global
  /// blocks.
  llvm::Constant *NSConcreteGlobalBlock;

  /// NSConcreteStackBlock - Cached reference to the class poinnter for stack
  /// blocks.
  llvm::Constant *NSConcreteStackBlock;

  const llvm::Type *BlockDescriptorType;
  const llvm::Type *GenericBlockLiteralType;

  struct {
    int GlobalUniqueCount;
  } Block;

  llvm::Value *BlockObjectAssign;
  llvm::Value *BlockObjectDispose;
  const llvm::Type *PtrToInt8Ty;

  std::map<uint64_t, llvm::Constant *> AssignCache;
  std::map<uint64_t, llvm::Constant *> DestroyCache;

  BlockModule(ASTContext &C, llvm::Module &M, const llvm::TargetData &TD,
              CodeGenTypes &T, CodeGenModule &CodeGen)
    : Context(C), TheModule(M), TheTargetData(TD), Types(T),
      CGM(CodeGen), VMContext(M.getContext()),
      NSConcreteGlobalBlock(0), NSConcreteStackBlock(0), BlockDescriptorType(0),
      GenericBlockLiteralType(0),
      BlockObjectAssign(0), BlockObjectDispose(0) {
    Block.GlobalUniqueCount = 0;
    PtrToInt8Ty = llvm::Type::getInt8PtrTy(M.getContext());
  }

  bool BlockRequiresCopying(QualType Ty)
    { return getContext().BlockRequiresCopying(Ty); }
};

class BlockFunction : public BlockBase {
  CodeGenModule &CGM;
  CodeGenFunction &CGF;
  ASTContext &getContext() const;

protected:
  llvm::LLVMContext &VMContext;

public:
  const llvm::PointerType *PtrToInt8Ty;
  struct HelperInfo {
    int index;
    int flag;
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

  /// BlockOffset - The offset in bytes for the next allocation of an
  /// imported block variable.
  CharUnits BlockOffset;
  /// BlockAlign - Maximal alignment needed for the Block expressed in 
  /// characters.
  CharUnits BlockAlign;

  /// getBlockOffset - Allocate a location within the block's storage
  /// for a value with the given size and alignment requirements.
  CharUnits getBlockOffset(CharUnits Size, CharUnits Align);

  /// BlockHasCopyDispose - True iff the block uses copy/dispose.
  bool BlockHasCopyDispose;

  /// BlockLayout - The layout of the block's storage, represented as
  /// a sequence of expressions which require such storage.  The
  /// expressions can be:
  /// - a BlockDeclRefExpr, indicating that the given declaration
  ///   from an enclosing scope is needed by the block;
  /// - a DeclRefExpr, which always wraps an anonymous VarDecl with
  ///   array type, used to insert padding into the block; or
  /// - a CXXThisExpr, indicating that the C++ 'this' value should
  ///   propagate from the parent to the block.
  /// This is a really silly representation.
  llvm::SmallVector<const Expr *, 8> BlockLayout;

  /// BlockDecls - Offsets for all Decls in BlockDeclRefExprs.
  llvm::DenseMap<const Decl*, CharUnits> BlockDecls;

  /// BlockCXXThisOffset - The offset of the C++ 'this' value within
  /// the block structure.
  CharUnits BlockCXXThisOffset;

  ImplicitParamDecl *BlockStructDecl;
  ImplicitParamDecl *getBlockStructDecl() { return BlockStructDecl; }

  llvm::Constant *GenerateCopyHelperFunction(bool, const llvm::StructType *,
                                             std::vector<HelperInfo> *);
  llvm::Constant *GenerateDestroyHelperFunction(bool, const llvm::StructType *,
                                                std::vector<HelperInfo> *);

  llvm::Constant *BuildCopyHelper(const llvm::StructType *,
                                  std::vector<HelperInfo> *);
  llvm::Constant *BuildDestroyHelper(const llvm::StructType *,
                                     std::vector<HelperInfo> *);

  llvm::Constant *GeneratebyrefCopyHelperFunction(const llvm::Type *, int flag);
  llvm::Constant *GeneratebyrefDestroyHelperFunction(const llvm::Type *T, int);

  llvm::Constant *BuildbyrefCopyHelper(const llvm::Type *T, int flag,
                                       unsigned Align);
  llvm::Constant *BuildbyrefDestroyHelper(const llvm::Type *T, int flag,
                                          unsigned Align);

  llvm::Value *getBlockObjectAssign();
  llvm::Value *getBlockObjectDispose();
  void BuildBlockRelease(llvm::Value *DeclPtr, int flag = BLOCK_FIELD_IS_BYREF);

  bool BlockRequiresCopying(QualType Ty)
    { return getContext().BlockRequiresCopying(Ty); }
};

}  // end namespace CodeGen
}  // end namespace clang

#endif
