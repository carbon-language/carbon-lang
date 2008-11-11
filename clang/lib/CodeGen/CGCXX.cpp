//===--- CGDecl.cpp - Emit LLVM Code for declarations ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code dealing with C++ code generation.
//
//===----------------------------------------------------------------------===//

// We might split this into multiple files if it gets too unwieldy 

#include "CodeGenFunction.h"
#include "CodeGenModule.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclObjC.h"
#include "llvm/ADT/StringExtras.h"

using namespace clang;
using namespace CodeGen;

using llvm::utostr;


// FIXME: Name mangling should be moved to a separate class.

static void mangleDeclContextInternal(const DeclContext *D, std::string &S)
{
  // FIXME: Should ObjcMethodDecl have the TranslationUnitDecl as its parent?
  assert(!D->getParent() || isa<TranslationUnitDecl>(D->getParent()) && 
         "Only one level of decl context mangling is currently supported!");
  
  if (const FunctionDecl* FD = dyn_cast<FunctionDecl>(D)) {
    S += utostr(FD->getIdentifier()->getLength());
    S += FD->getIdentifier()->getName();
    
    if (FD->param_size() == 0)
      S += 'v';
    else
      assert(0 && "mangling of types not supported yet!");
  } else if (const ObjCMethodDecl* MD = dyn_cast<ObjCMethodDecl>(D)) {
    
    // FIXME: This should really use GetNameForMethod from CGObjCMac.
    std::string Name;
    Name += MD->isInstance() ? '-' : '+';
    Name += '[';
    Name += MD->getClassInterface()->getName();
    Name += ' ';
    Name += MD->getSelector().getName();
    Name += ']';
    S += utostr(Name.length());
    S += Name;
  } else 
    assert(0 && "Unsupported decl type!");
}

static void mangleVarDeclInternal(const VarDecl &D, std::string &S)
{
  S += 'Z';
  mangleDeclContextInternal(D.getDeclContext(), S);
  S += 'E';
  
  S += utostr(D.getIdentifier()->getLength());
  S += D.getIdentifier()->getName();
}

static std::string mangleVarDecl(const VarDecl& D)
{
  std::string S = "_Z";
  
  mangleVarDeclInternal(D, S);
  
  return S;
}

static std::string mangleGuardVariable(const VarDecl& D)
{
  std::string S = "_ZGV";

  mangleVarDeclInternal(D, S);
  
  return S;
}

llvm::GlobalValue *
CodeGenFunction::GenerateStaticCXXBlockVarDecl(const VarDecl &D)
{
  assert(!getContext().getLangOptions().ThreadsafeStatics &&
         "thread safe statics are currently not supported!");
  const llvm::Type *LTy = CGM.getTypes().ConvertTypeForMem(D.getType());

  // FIXME: If the function is inline, the linkage should be weak.
  llvm::GlobalValue::LinkageTypes linkage = llvm::GlobalValue::InternalLinkage;
  
  // Create the guard variable.
  llvm::GlobalValue *GuardV = 
    new llvm::GlobalVariable(llvm::Type::Int64Ty, false,
                             linkage,
                             llvm::Constant::getNullValue(llvm::Type::Int64Ty),
                             mangleGuardVariable(D),
                             &CGM.getModule());
  
  // FIXME: Address space.
  const llvm::Type *PtrTy = llvm::PointerType::get(llvm::Type::Int8Ty, 0);

  // Load the first byte of the guard variable.
  llvm::Value *V = Builder.CreateLoad(Builder.CreateBitCast(GuardV, PtrTy), 
                                      "tmp");
  
  // Compare it against 0.
  llvm::Value *nullValue = llvm::Constant::getNullValue(llvm::Type::Int8Ty);
  llvm::Value *ICmp = Builder.CreateICmpEQ(V, nullValue , "tobool");
  
  llvm::BasicBlock *InitBlock = createBasicBlock("init");
  llvm::BasicBlock *EndBlock = createBasicBlock("initend");

  // If the guard variable is 0, jump to the initializer code.
  Builder.CreateCondBr(ICmp, InitBlock, EndBlock);
                         
  EmitBlock(InitBlock);

  llvm::GlobalValue *GV =
    new llvm::GlobalVariable(LTy, false,
                             llvm::GlobalValue::InternalLinkage,
                             llvm::Constant::getNullValue(LTy), 
                             mangleVarDecl(D),
                             &CGM.getModule(), 0, 
                             D.getType().getAddressSpace());
    
  const Expr *Init = D.getInit();
  if (!hasAggregateLLVMType(Init->getType())) {
    llvm::Value *V = EmitScalarExpr(Init);
    Builder.CreateStore(V, GV, D.getType().isVolatileQualified());
  } else if (Init->getType()->isAnyComplexType()) {
    EmitComplexExprIntoAddr(Init, GV, D.getType().isVolatileQualified());
  } else {
    EmitAggExpr(Init, GV, D.getType().isVolatileQualified());
  }
    
  Builder.CreateStore(llvm::ConstantInt::get(llvm::Type::Int8Ty, 1),
                      Builder.CreateBitCast(GuardV, PtrTy));
                      
  EmitBlock(EndBlock);
  return GV;
}

