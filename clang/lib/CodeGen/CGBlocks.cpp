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
#include "llvm/Target/TargetData.h"

#include <algorithm>

using namespace clang;
using namespace CodeGen;

// Block flags
enum {
  IsGlobal = 1 << 28
};

static const llvm::Type *getBlockDescriptorType(CodeGenModule &CGM) {
  static const llvm::Type *Ty = 0;
    
  if (!Ty) {
    const llvm::Type *UnsignedLongTy = 
      CGM.getTypes().ConvertType(CGM.getContext().UnsignedLongTy);
        
    // struct __block_descriptor {
    //   unsigned long reserved;
    //   unsigned long block_size;
    // };
    Ty = llvm::StructType::get(UnsignedLongTy, 
                               UnsignedLongTy, 
                               NULL);
        
    CGM.getModule().addTypeName("struct.__block_descriptor", Ty);
  }
    
  return Ty;
}

static const llvm::Type *getGenericBlockLiteralType(CodeGenModule &CGM) {
  static const llvm::Type *Ty = 0;
    
  if (!Ty) {
    const llvm::Type *Int8PtrTy = 
      llvm::PointerType::getUnqual(llvm::Type::Int8Ty);
        
    const llvm::Type *BlockDescPtrTy = 
      llvm::PointerType::getUnqual(getBlockDescriptorType(CGM));
        
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
        
    CGM.getModule().addTypeName("struct.__block_literal_generic", Ty);
  }
  
  return Ty;
}

/// getBlockFunctionType - Given a BlockPointerType, will return the 
/// function type for the block, including the first block literal argument.
static QualType getBlockFunctionType(ASTContext &Ctx,
                                     const BlockPointerType *BPT) {
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

RValue CodeGenFunction::EmitBlockCallExpr(const CallExpr* E) {
  const BlockPointerType *BPT = 
    E->getCallee()->getType()->getAsBlockPointerType();
  
  llvm::Value *Callee = EmitScalarExpr(E->getCallee());

  // Get a pointer to the generic block literal.
  const llvm::Type *BlockLiteralTy =
    llvm::PointerType::getUnqual(getGenericBlockLiteralType(CGM));

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

llvm::Constant *CodeGenModule::GetAddrOfGlobalBlock(const BlockExpr *BE) {
  if (!NSConcreteGlobalBlock) {
    const llvm::Type *Ty = llvm::PointerType::getUnqual(llvm::Type::Int8Ty);

    // FIXME: Wee should have a CodeGenModule::AddRuntimeVariable that does the 
    // same thing as CreateRuntimeFunction if there's already a variable with
    // the same name.
    NSConcreteGlobalBlock = 
      new llvm::GlobalVariable(Ty, false,
                              llvm::GlobalVariable::ExternalLinkage, 0, 
                              "_NSConcreteGlobalBlock", &getModule());      
  }

  // Generate the block descriptor.
  const llvm::Type *UnsignedLongTy = Types.ConvertType(Context.UnsignedLongTy);
  
  llvm::Constant *DescriptorFields[2];
  
  // Reserved
  DescriptorFields[0] = llvm::Constant::getNullValue(UnsignedLongTy);
  
  // Block literal size. For global blocks we just use the size of the generic
  // block literal struct.
  uint64_t BlockLiteralSize = 
    TheTargetData.getTypeStoreSizeInBits(getGenericBlockLiteralType(*this)) / 8;
  DescriptorFields[1] = llvm::ConstantInt::get(UnsignedLongTy,BlockLiteralSize);
  
  llvm::Constant *DescriptorStruct = 
    llvm::ConstantStruct::get(&DescriptorFields[0], 2);
  
  llvm::GlobalVariable *Descriptor =
    new llvm::GlobalVariable(DescriptorStruct->getType(), true,
                             llvm::GlobalVariable::InternalLinkage, 
                             DescriptorStruct, "__block_descriptor_global", 
                             &getModule());
  
  // Generate the constants for the block literal.
  llvm::Constant *LiteralFields[5];
  
  CodeGenFunction::BlockInfo Info(0, "global");
  llvm::Function *Fn = CodeGenFunction(*this).GenerateBlockFunction(BE, Info);
  
  // isa
  LiteralFields[0] = NSConcreteGlobalBlock;
  
  // Flags
  LiteralFields[1] = llvm::ConstantInt::get(llvm::Type::Int32Ty, IsGlobal);
  
  // Reserved
  LiteralFields[2] = llvm::Constant::getNullValue(llvm::Type::Int32Ty);
  
  // Function
  LiteralFields[3] = Fn;
  
  // Descriptor
  LiteralFields[4] = Descriptor;
  
  llvm::Constant *BlockLiteralStruct = 
    llvm::ConstantStruct::get(&LiteralFields[0], 5);
  
  llvm::GlobalVariable *BlockLiteral = 
    new llvm::GlobalVariable(BlockLiteralStruct->getType(), true,
                             llvm::GlobalVariable::InternalLinkage, 
                             BlockLiteralStruct, "__block_literal_global", 
                             &getModule());
  
  return BlockLiteral;
}

llvm::Function *CodeGenFunction::GenerateBlockFunction(const BlockExpr *Expr,
                                                       const BlockInfo& Info)
{
  const FunctionTypeProto *FTy = 
    cast<FunctionTypeProto>(Expr->getFunctionType());
  
  FunctionArgList Args;
  
  const BlockDecl *BD = Expr->getBlockDecl();

  // FIXME: This leaks
  ImplicitParamDecl *SelfDecl = 
    ImplicitParamDecl::Create(getContext(), 0,
                              SourceLocation(), 0,
                              getContext().getPointerType(getContext().VoidTy));
  
  Args.push_back(std::make_pair(SelfDecl, SelfDecl->getType()));
  
  for (BlockDecl::param_iterator i = BD->param_begin(), 
       e = BD->param_end(); i != e; ++i)
    Args.push_back(std::make_pair(*e, (*e)->getType()));
  
  const CGFunctionInfo &FI = 
    CGM.getTypes().getFunctionInfo(FTy->getResultType(), Args);

  std::string Name = std::string("__block_function_") + Info.NameSuffix;

  CodeGenTypes &Types = CGM.getTypes();
  const llvm::FunctionType *LTy = Types.GetFunctionType(FI, FTy->isVariadic());
  
  llvm::Function *Fn = 
    llvm::Function::Create(LTy, llvm::GlobalValue::InternalLinkage,
                           Name,
                           &CGM.getModule());
  
  StartFunction(BD, FTy->getResultType(), Fn, Args, 
                Expr->getBody()->getLocEnd());
  EmitStmt(Expr->getBody());
  FinishFunction(cast<CompoundStmt>(Expr->getBody())->getRBracLoc());

  return Fn;
}


