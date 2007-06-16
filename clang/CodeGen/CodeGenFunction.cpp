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


const llvm::Type *CodeGenFunction::ConvertType(QualType T, SourceLocation Loc) {
  return CGM.getTypes().ConvertType(T, Loc);
}

void CodeGenFunction::GenerateCode(const FunctionDecl *FD) {
  LLVMIntTy = ConvertType(getContext().IntTy, FD->getLocation());
  LLVMPointerWidth = Target.getPointerWidth(FD->getLocation());
  
  const llvm::FunctionType *Ty = 
    cast<llvm::FunctionType>(ConvertType(FD->getType(), FD->getLocation()));
  
  // FIXME: param attributes for sext/zext etc.
  
  CurFuncDecl = FD;
  CurFn = new llvm::Function(Ty, llvm::Function::ExternalLinkage,
                             FD->getName(), &CGM.getModule());
  
  llvm::BasicBlock *EntryBB = new llvm::BasicBlock("entry", CurFn);
  
  Builder.SetInsertPoint(EntryBB);

  // Create a marker to make it easy to insert allocas into the entryblock
  // later.
  llvm::Value *Undef = llvm::UndefValue::get(llvm::Type::Int32Ty);
  AllocaInsertPt = Builder.CreateBitCast(Undef,llvm::Type::Int32Ty, "allocapt");
  
  // Emit allocs for param decls. 
  llvm::Function::arg_iterator AI = CurFn->arg_begin();
  for (unsigned i = 0, e = FD->getNumParams(); i != e; ++i, ++AI) {
    assert(AI != CurFn->arg_end() && "Argument mismatch!");
    EmitParmDecl(*FD->getParamDecl(i), AI);
  }
  
  // Emit the function body.
  EmitStmt(FD->getBody());
  
  // Emit a return for code that falls off the end.
  // FIXME: if this is C++ main, this should return 0.
  if (Ty->getReturnType() == llvm::Type::VoidTy)
    Builder.CreateRetVoid();
  else
    Builder.CreateRet(llvm::UndefValue::get(Ty->getReturnType()));
  
  // Verify that the function is well formed.
  assert(!verifyFunction(*CurFn));
}

