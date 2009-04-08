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
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "llvm/ADT/StringExtras.h"
using namespace clang;
using namespace CodeGen;


// FIXME: Name mangling should be moved to a separate class.

static void mangleDeclContextInternal(const DeclContext *D, std::string &S)
{
  // FIXME: Should ObjcMethodDecl have the TranslationUnitDecl as its parent?
  assert((!D->getParent() || isa<TranslationUnitDecl>(D->getParent())) && 
         "Only one level of decl context mangling is currently supported!");
  
  if (const FunctionDecl* FD = dyn_cast<FunctionDecl>(D)) {
    S += llvm::utostr(FD->getIdentifier()->getLength());
    S += FD->getIdentifier()->getName();
    
    if (FD->param_size() == 0)
      S += 'v';
    else
      assert(0 && "mangling of types not supported yet!");
  } else if (const ObjCMethodDecl* MD = dyn_cast<ObjCMethodDecl>(D)) {
    
    // FIXME: This should really use GetNameForMethod from CGObjCMac.
    std::string Name;
    Name += MD->isInstanceMethod() ? '-' : '+';
    Name += '[';
    Name += MD->getClassInterface()->getNameAsString();
    Name += ' ';
    Name += MD->getSelector().getAsString();
    Name += ']';
    S += llvm::utostr(Name.length());
    S += Name;
  } else 
    assert(0 && "Unsupported decl type!");
}

static void mangleVarDeclInternal(const VarDecl &D, std::string &S)
{
  S += 'Z';
  mangleDeclContextInternal(D.getDeclContext(), S);
  S += 'E';
  
  S += llvm::utostr(D.getIdentifier()->getLength());
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

void 
CodeGenFunction::GenerateStaticCXXBlockVarDeclInit(const VarDecl &D, 
                                                   llvm::GlobalVariable *GV) {
  // FIXME: This should use __cxa_guard_{acquire,release}?

  assert(!getContext().getLangOptions().ThreadsafeStatics &&
         "thread safe statics are currently not supported!");

  // Create the guard variable.
  llvm::GlobalValue *GuardV = 
    new llvm::GlobalVariable(llvm::Type::Int64Ty, false,
                             GV->getLinkage(),
                             llvm::Constant::getNullValue(llvm::Type::Int64Ty),
                             mangleGuardVariable(D),
                             &CGM.getModule());
  
  // Load the first byte of the guard variable.
  const llvm::Type *PtrTy = llvm::PointerType::get(llvm::Type::Int8Ty, 0);
  llvm::Value *V = Builder.CreateLoad(Builder.CreateBitCast(GuardV, PtrTy), 
                                      "tmp");
  
  // Compare it against 0.
  llvm::Value *nullValue = llvm::Constant::getNullValue(llvm::Type::Int8Ty);
  llvm::Value *ICmp = Builder.CreateICmpEQ(V, nullValue , "tobool");
  
  llvm::BasicBlock *InitBlock = createBasicBlock("init");
  llvm::BasicBlock *EndBlock = createBasicBlock("init.end");

  // If the guard variable is 0, jump to the initializer code.
  Builder.CreateCondBr(ICmp, InitBlock, EndBlock);
                         
  EmitBlock(InitBlock);

  // Patch the name. FIXME: We shouldn't need to do this.
  GV->setName(mangleVarDecl(D));
    
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
}

RValue CodeGenFunction::EmitCXXMemberCallExpr(const CXXMemberCallExpr *CE) {
  const MemberExpr *ME = cast<MemberExpr>(CE->getCallee());
  const CXXMethodDecl *MD = cast<CXXMethodDecl>(ME->getMemberDecl());
  assert(MD->isInstance() && 
         "Trying to emit a member call expr on a static method!");
  
  const FunctionProtoType *FPT = MD->getType()->getAsFunctionProtoType();
  const llvm::Type *Ty = 
    CGM.getTypes().GetFunctionType(CGM.getTypes().getFunctionInfo(MD), 
                                   FPT->isVariadic());
  llvm::Constant *Callee = CGM.GetAddrOfFunction(MD, Ty);
  
  llvm::Value *BaseValue = 0;
  
  // There's a deref operator node added in Sema::BuildCallToMemberFunction
  // that's giving the wrong type for -> call exprs so we just ignore them
  // for now.
  if (ME->isArrow())
    return EmitUnsupportedRValue(CE, "C++ member call expr");
  else {
    LValue BaseLV = EmitLValue(ME->getBase());
    BaseValue = BaseLV.getAddress();
  }
  
  CallArgList Args;
  
  // Push the 'this' pointer.
  Args.push_back(std::make_pair(RValue::get(BaseValue), 
                                MD->getThisType(getContext())));
  
  EmitCallArgs(Args, FPT, CE->arg_begin(), CE->arg_end());
  
  QualType ResultType = MD->getType()->getAsFunctionType()->getResultType();
  return EmitCall(CGM.getTypes().getFunctionInfo(ResultType, Args), 
                  Callee, Args, MD);
}
