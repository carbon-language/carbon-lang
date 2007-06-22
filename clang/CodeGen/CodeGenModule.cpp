//===--- CodeGenModule.cpp - Emit LLVM Code from ASTs for a Module --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This coordinates the per-module state used while generating code.
//
//===----------------------------------------------------------------------===//

#include "CodeGenModule.h"
#include "CodeGenFunction.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Intrinsics.h"
using namespace clang;
using namespace CodeGen;


CodeGenModule::CodeGenModule(ASTContext &C, llvm::Module &M)
  : Context(C), TheModule(M), Types(C.Target) {}

llvm::Constant *CodeGenModule::GetAddrOfGlobalDecl(const Decl *D) {
  // See if it is already in the map.
  llvm::Constant *&Entry = GlobalDeclMap[D];
  if (Entry) return Entry;
  
  QualType ASTTy = cast<ValueDecl>(D)->getType();
  const llvm::Type *Ty = getTypes().ConvertType(ASTTy, D->getLocation());
  if (isa<FunctionDecl>(D)) {
    const llvm::FunctionType *FTy = cast<llvm::FunctionType>(Ty);
    // FIXME: param attributes for sext/zext etc.
    return Entry = new llvm::Function(FTy, llvm::Function::ExternalLinkage,
                                      D->getName(), &getModule());
  }
  
  assert(isa<FileVarDecl>(D) && "Unknown global decl!");
  
  return Entry = new llvm::GlobalVariable(Ty, false, 
                                          llvm::GlobalValue::ExternalLinkage,
                                          0, D->getName(), &getModule());
}

void CodeGenModule::EmitFunction(FunctionDecl *FD) {
  // If this is not a prototype, emit the body.
  if (FD->getBody())
    CodeGenFunction(*this).GenerateCode(FD);
}



llvm::Function *CodeGenModule::getMemCpyFn() {
  if (MemCpyFn) return MemCpyFn;
  llvm::Intrinsic::ID IID;
  switch (Context.Target.getPointerWidth(SourceLocation())) {
  default: assert(0 && "Unknown ptr width");
  case 32: IID = llvm::Intrinsic::memcpy_i32; break;
  case 64: IID = llvm::Intrinsic::memcpy_i64; break;
  }
  return MemCpyFn = llvm::Intrinsic::getDeclaration(&TheModule, IID);
}
