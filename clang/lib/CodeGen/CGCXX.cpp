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
#include "Mangle.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "llvm/ADT/StringExtras.h"
using namespace clang;
using namespace CodeGen;

void 
CodeGenFunction::GenerateStaticCXXBlockVarDeclInit(const VarDecl &D, 
                                                   llvm::GlobalVariable *GV) {
  // FIXME: This should use __cxa_guard_{acquire,release}?

  assert(!getContext().getLangOptions().ThreadsafeStatics &&
         "thread safe statics are currently not supported!");

  llvm::SmallString<256> GuardVName;
  llvm::raw_svector_ostream GuardVOut(GuardVName);
  mangleGuardVariable(&D, getContext(), GuardVOut);
  
  // Create the guard variable.
  llvm::GlobalValue *GuardV = 
    new llvm::GlobalVariable(llvm::Type::Int64Ty, false,
                             GV->getLinkage(),
                             llvm::Constant::getNullValue(llvm::Type::Int64Ty),
                             GuardVName.c_str(),
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
  
  if (ME->isArrow())
    BaseValue = EmitScalarExpr(ME->getBase());
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

llvm::Value *CodeGenFunction::LoadCXXThis() {
  assert(isa<CXXMethodDecl>(CurFuncDecl) && 
         "Must be in a C++ member function decl to load 'this'");
  assert(cast<CXXMethodDecl>(CurFuncDecl)->isInstance() &&
         "Must be in a C++ member function decl to load 'this'");
  
  // FIXME: What if we're inside a block?
  return Builder.CreateLoad(LocalDeclMap[CXXThisDecl], "this");
}

void
CodeGenFunction::EmitCXXConstructorCall(const CXXConstructorDecl *D, 
                                        CXXCtorType Type, 
                                        llvm::Value *This,
                                        CallExpr::const_arg_iterator ArgBeg,
                                        CallExpr::const_arg_iterator ArgEnd) {
  CallArgList Args;
  
  // Push the 'this' pointer.
  Args.push_back(std::make_pair(RValue::get(This), 
                                D->getThisType(getContext())));
  
  EmitCallArgs(Args, D->getType()->getAsFunctionProtoType(), ArgBeg, ArgEnd);
  
  EmitCall(CGM.getTypes().getFunctionInfo(getContext().VoidTy, Args), 
           CGM.GetAddrOfCXXConstructor(D, Type), Args, D);
}

void 
CodeGenFunction::EmitCXXConstructExpr(llvm::Value *Dest, 
                                      const CXXConstructExpr *E) {
  assert(Dest && "Must have a destination!");
  
  const CXXRecordDecl *RD = 
  cast<CXXRecordDecl>(E->getType()->getAsRecordType()->getDecl());
  if (RD->hasTrivialConstructor())
    return;
  
  // Call the constructor.
  EmitCXXConstructorCall(E->getConstructor(), Ctor_Complete, Dest, 
                         E->arg_begin(), E->arg_end());
}

static bool canGenerateCXXstructor(const CXXRecordDecl *RD, 
                                   ASTContext &Context) {
  // The class has base classes - we don't support that right now.
  if (RD->getNumBases() > 0)
    return false;
  
  for (CXXRecordDecl::field_iterator I = RD->field_begin(Context), 
       E = RD->field_end(Context); I != E; ++I) {
    // We don't support ctors for fields that aren't POD.
    if (!I->getType()->isPODType())
      return false;
  }
  
  return true;
}

void CodeGenModule::EmitCXXConstructors(const CXXConstructorDecl *D) {
  if (!canGenerateCXXstructor(D->getParent(), getContext())) {
    ErrorUnsupported(D, "C++ constructor", true);
    return;
  }

  EmitCXXConstructor(D, Ctor_Complete);
  EmitCXXConstructor(D, Ctor_Base);
}

void CodeGenModule::EmitCXXConstructor(const CXXConstructorDecl *D, 
                                       CXXCtorType Type) {
  
  llvm::Function *Fn = GetAddrOfCXXConstructor(D, Type);
  
  CodeGenFunction(*this).GenerateCode(D, Fn);
  
  SetFunctionDefinitionAttributes(D, Fn);
  SetLLVMFunctionAttributesForDefinition(D, Fn);
}

llvm::Function *
CodeGenModule::GetAddrOfCXXConstructor(const CXXConstructorDecl *D, 
                                       CXXCtorType Type) {
  const llvm::FunctionType *FTy =
    getTypes().GetFunctionType(getTypes().getFunctionInfo(D), false);
  
  const char *Name = getMangledCXXCtorName(D, Type);
  return cast<llvm::Function>(GetOrCreateLLVMFunction(Name, FTy, D));
}

const char *CodeGenModule::getMangledCXXCtorName(const CXXConstructorDecl *D, 
                                                 CXXCtorType Type) {
  llvm::SmallString<256> Name;
  llvm::raw_svector_ostream Out(Name);
  mangleCXXCtor(D, Type, Context, Out);
  
  Name += '\0';
  return UniqueMangledName(Name.begin(), Name.end());
}

void CodeGenModule::EmitCXXDestructors(const CXXDestructorDecl *D) {
  if (!canGenerateCXXstructor(D->getParent(), getContext())) {
    ErrorUnsupported(D, "C++ destructor", true);
    return;
  }
  
  EmitCXXDestructor(D, Dtor_Complete);
  EmitCXXDestructor(D, Dtor_Base);
}

void CodeGenModule::EmitCXXDestructor(const CXXDestructorDecl *D, 
                                      CXXDtorType Type) {
  llvm::Function *Fn = GetAddrOfCXXDestructor(D, Type);
  
  CodeGenFunction(*this).GenerateCode(D, Fn);
  
  SetFunctionDefinitionAttributes(D, Fn);
  SetLLVMFunctionAttributesForDefinition(D, Fn);
}

llvm::Function *
CodeGenModule::GetAddrOfCXXDestructor(const CXXDestructorDecl *D, 
                                      CXXDtorType Type) {
  const llvm::FunctionType *FTy =
    getTypes().GetFunctionType(getTypes().getFunctionInfo(D), false);
  
  const char *Name = getMangledCXXDtorName(D, Type);
  return cast<llvm::Function>(GetOrCreateLLVMFunction(Name, FTy, D));
}

const char *CodeGenModule::getMangledCXXDtorName(const CXXDestructorDecl *D, 
                                                 CXXDtorType Type) {
  llvm::SmallString<256> Name;
  llvm::raw_svector_ostream Out(Name);
  mangleCXXDtor(D, Type, Context, Out);
  
  Name += '\0';
  return UniqueMangledName(Name.begin(), Name.end());
}

