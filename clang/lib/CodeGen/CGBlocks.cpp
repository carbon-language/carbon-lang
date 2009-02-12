//===--- CGBlocks.cpp - Emit LLVM Code for declarations -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit blocks.
//
//===----------------------------------------------------------------------===//

#include "CodeGenFunction.h"
#include "CodeGenModule.h"
#include "llvm/Module.h"

#include <algorithm>

using namespace clang;
using namespace CodeGen;

static const llvm::Type *getBlockDescriptorType(CodeGenFunction &CGF) {
  static const llvm::Type *Ty = 0;
    
  if (!Ty) {
    const llvm::Type *UnsignedLongTy = 
      CGF.ConvertType(CGF.getContext().UnsignedLongTy);
        
    // struct __block_descriptor {
    //   unsigned long reserved;
    //   unsigned long block_size;
    // };
    Ty = llvm::StructType::get(UnsignedLongTy, 
                               UnsignedLongTy, 
                               NULL);
        
    CGF.CGM.getModule().addTypeName("struct.__block_descriptor", Ty);
  }
    
  return Ty;
}

static const llvm::Type *getGenericBlockLiteralType(CodeGenFunction &CGF) {
  static const llvm::Type *Ty = 0;
    
  if (!Ty) {
    const llvm::Type *Int8PtrTy = 
      llvm::PointerType::getUnqual(llvm::Type::Int8Ty);
        
    const llvm::Type *BlockDescPtrTy = 
      llvm::PointerType::getUnqual(getBlockDescriptorType(CGF));
        
    // struct __block_literal_generic {
    //   void *isa;
    //   int flags;
    //   int reserved;
    //   void (*invoke)(void *);
    //   struct __block_descriptor *descriptor;
    // };
    Ty = llvm::StructType::get(Int8PtrTy,
                               llvm::Type::Int32Ty,
                               llvm::Type::Int32Ty,
                               Int8PtrTy,
                               BlockDescPtrTy,
                               NULL);
        
    CGF.CGM.getModule().addTypeName("struct.__block_literal_generic", Ty);
  }
  
  return Ty;
}

/// getBlockFunctionType - Given a BlockPointerType, will return the 
/// function type for the block, including the first block literal argument.
static QualType getBlockFunctionType(ASTContext &Ctx,
                                     const BlockPointerType *BPT)
{
  const FunctionTypeProto *FTy = cast<FunctionTypeProto>(BPT->getPointeeType());
  
  llvm::SmallVector<QualType, 8> Types;
  Types.push_back(Ctx.getPointerType(Ctx.VoidTy));
  
  for (FunctionTypeProto::arg_type_iterator i = FTy->arg_type_begin(),
       e = FTy->arg_type_end(); i != e; ++i)
    Types.push_back(*i);
  
  return Ctx.getFunctionType(FTy->getResultType(),
                             &Types[0], Types.size(), 
                             FTy->isVariadic(), 0);
}

RValue CodeGenFunction::EmitBlockCallExpr(const CallExpr* E)
{
  const BlockPointerType *BPT = 
    E->getCallee()->getType()->getAsBlockPointerType();
  
  llvm::Value *Callee = EmitScalarExpr(E->getCallee());

  // Get a pointer to the generic block literal.
  const llvm::Type *BlockLiteralTy =
    llvm::PointerType::getUnqual(getGenericBlockLiteralType(*this));

  // Bitcast the callee to a block literal.
  llvm::Value *BlockLiteral = 
    Builder.CreateBitCast(Callee, BlockLiteralTy, "block.literal");

  // Get the function pointer from the literal.
  llvm::Value *FuncPtr = Builder.CreateStructGEP(BlockLiteral, 3, "tmp");
  llvm::Value *Func = Builder.CreateLoad(FuncPtr, FuncPtr, "tmp");

  // Cast the function pointer to the right type.
  const llvm::Type *BlockFTy = 
    ConvertType(getBlockFunctionType(getContext(), BPT));
  const llvm::Type *BlockFTyPtr = llvm::PointerType::getUnqual(BlockFTy);
  Func = Builder.CreateBitCast(Func, BlockFTyPtr);

  BlockLiteral = 
    Builder.CreateBitCast(BlockLiteral, 
                          llvm::PointerType::getUnqual(llvm::Type::Int8Ty),
                          "tmp");
  
  // Add the block literal.
  QualType VoidPtrTy = getContext().getPointerType(getContext().VoidTy);
  CallArgList Args;
  Args.push_back(std::make_pair(RValue::get(BlockLiteral), VoidPtrTy));
  
  // And the rest of the arguments.
  for (CallExpr::const_arg_iterator i = E->arg_begin(), e = E->arg_end(); 
       i != e; ++i)
    Args.push_back(std::make_pair(EmitAnyExprToTemp(*i), 
                                  i->getType()));
  
  // And call the block.
  return EmitCall(CGM.getTypes().getFunctionInfo(E->getType(), Args), 
                  Func, Args);
}
