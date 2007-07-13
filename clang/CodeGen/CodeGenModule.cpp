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
  const llvm::Type *Ty = getTypes().ConvertType(ASTTy);
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

void CodeGenModule::EmitFunction(const FunctionDecl *FD) {
  // If this is not a prototype, emit the body.
  if (FD->getBody())
    CodeGenFunction(*this).GenerateCode(FD);
}

void CodeGenModule::EmitGlobalVar(const FileVarDecl *D) {
  llvm::GlobalVariable *GV = cast<llvm::GlobalVariable>(GetAddrOfGlobalDecl(D));
  
  // If the storage class is external and there is no initializer, just leave it
  // as a declaration.
  if (D->getStorageClass() == VarDecl::Extern && D->getInit() == 0)
    return;

  // Otherwise, convert the initializer, or use zero if appropriate.
  llvm::Constant *Init;
  if (D->getInit() == 0)
    Init = llvm::Constant::getNullValue(GV->getType()->getElementType());
  else
    assert(D->getInit() == 0 && "FIXME: Global variable initializers unimp!");
    
  GV->setInitializer(Init);
  
  // Set the llvm linkage type as appropriate.
  // FIXME: This isn't right.  This should handle common linkage and other
  // stuff.
  switch (D->getStorageClass()) {
  case VarDecl::Auto:
  case VarDecl::Register:
    assert(0 && "Can't have auto or register globals");
  case VarDecl::None:
  case VarDecl::Extern:
    // todo: common
    break;
  case VarDecl::Static:
    GV->setLinkage(llvm::GlobalVariable::InternalLinkage);
    break;
  }
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
