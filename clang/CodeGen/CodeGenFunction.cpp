//===--- CodeGenFunction.cpp - Emit LLVM Code from ASTs for a Function ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This coordinates the per-function state used while generating code.
//
//===----------------------------------------------------------------------===//

#include "CodeGenFunction.h"
#include "CodeGenModule.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/AST/AST.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Support/CFG.h"
using namespace clang;
using namespace CodeGen;

CodeGenFunction::CodeGenFunction(CodeGenModule &cgm) 
  : CGM(cgm), Target(CGM.getContext().Target) {}

ASTContext &CodeGenFunction::getContext() const {
  return CGM.getContext();
}


llvm::BasicBlock *CodeGenFunction::getBasicBlockForLabel(const LabelStmt *S) {
  llvm::BasicBlock *&BB = LabelMap[S];
  if (BB) return BB;
  
  // Create, but don't insert, the new block.
  return BB = new llvm::BasicBlock(S->getName());
}


const llvm::Type *CodeGenFunction::ConvertType(QualType T) {
  return CGM.getTypes().ConvertType(T);
}

bool CodeGenFunction::hasAggregateLLVMType(QualType T) {
  return !T->isRealType() && !T->isPointerType() && !T->isVoidType() &&
         !T->isVectorType() && !T->isFunctionType();
}


void CodeGenFunction::GenerateCode(const FunctionDecl *FD) {
  LLVMIntTy = ConvertType(getContext().IntTy);
  LLVMPointerWidth = static_cast<unsigned>(
    getContext().getTypeSize(getContext().getPointerType(getContext().VoidTy),
                             SourceLocation()));
  
  CurFn = cast<llvm::Function>(CGM.GetAddrOfGlobalDecl(FD));
  CurFuncDecl = FD;
  
  // TODO: Set up linkage and many other things.
  assert(CurFn->isDeclaration() && "Function already has body?");
  
  llvm::BasicBlock *EntryBB = new llvm::BasicBlock("entry", CurFn);
  
  Builder.SetInsertPoint(EntryBB);

  // Create a marker to make it easy to insert allocas into the entryblock
  // later.
  llvm::Value *Undef = llvm::UndefValue::get(llvm::Type::Int32Ty);
  AllocaInsertPt = Builder.CreateBitCast(Undef,llvm::Type::Int32Ty, "allocapt");
  
  // Emit allocs for param decls.  Give the LLVM Argument nodes names.
  llvm::Function::arg_iterator AI = CurFn->arg_begin();
  
  // Name the struct return argument.
  if (hasAggregateLLVMType(FD->getResultType())) {
    AI->setName("agg.result");
    ++AI;
  }
  
  for (unsigned i = 0, e = FD->getNumParams(); i != e; ++i, ++AI) {
    assert(AI != CurFn->arg_end() && "Argument mismatch!");
    EmitParmDecl(*FD->getParamDecl(i), AI);
  }
  
  // Emit the function body.
  EmitStmt(FD->getBody());
  
  // Emit a return for code that falls off the end. If insert point
  // is a dummy block with no predecessors then remove the block itself.
  llvm::BasicBlock *BB = Builder.GetInsertBlock();
  if (isDummyBlock(BB))
    BB->eraseFromParent();
  else {
    // FIXME: if this is C++ main, this should return 0.
    if (CurFn->getReturnType() == llvm::Type::VoidTy)
      Builder.CreateRetVoid();
    else
      Builder.CreateRet(llvm::UndefValue::get(CurFn->getReturnType()));
  }
  assert(BreakContinueStack.empty() &&
         "mismatched push/pop in break/continue stack!");
  
  // Verify that the function is well formed.
  assert(!verifyFunction(*CurFn));
}

/// isDummyBlock - Return true if BB is an empty basic block
/// with no predecessors.
bool CodeGenFunction::isDummyBlock(const llvm::BasicBlock *BB) {
  if (BB->empty() && pred_begin(BB) == pred_end(BB)) 
    return true;
  return false;
}

/// StartBlock - Start new block named N. If insert block is a dummy block
/// then reuse it.
void CodeGenFunction::StartBlock(const char *N) {
  llvm::BasicBlock *BB = Builder.GetInsertBlock();
  if (!isDummyBlock(BB))
    EmitBlock(new llvm::BasicBlock(N));
  else
    BB->setName(N);
}

