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
    new llvm::GlobalVariable(CGM.getModule(), llvm::Type::Int64Ty, false,
                             GV->getLinkage(),
                             llvm::Constant::getNullValue(llvm::Type::Int64Ty),
                             GuardVName.c_str());
  
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

RValue CodeGenFunction::EmitCXXMemberCall(const CXXMethodDecl *MD,
                                          llvm::Value *Callee,
                                          llvm::Value *This,
                                          CallExpr::const_arg_iterator ArgBeg,
                                          CallExpr::const_arg_iterator ArgEnd) {
  assert(MD->isInstance() && 
         "Trying to emit a member call expr on a static method!");

  const FunctionProtoType *FPT = MD->getType()->getAsFunctionProtoType();
  
  CallArgList Args;
  
  // Push the this ptr.
  Args.push_back(std::make_pair(RValue::get(This),
                                MD->getThisType(getContext())));
  
  // And the rest of the call args
  EmitCallArgs(Args, FPT, ArgBeg, ArgEnd);
  
  QualType ResultType = MD->getType()->getAsFunctionType()->getResultType();
  return EmitCall(CGM.getTypes().getFunctionInfo(ResultType, Args),
                  Callee, Args, MD);
}

RValue CodeGenFunction::EmitCXXMemberCallExpr(const CXXMemberCallExpr *CE) {
  const MemberExpr *ME = cast<MemberExpr>(CE->getCallee());
  const CXXMethodDecl *MD = cast<CXXMethodDecl>(ME->getMemberDecl());

  const FunctionProtoType *FPT = MD->getType()->getAsFunctionProtoType();
  const llvm::Type *Ty = 
    CGM.getTypes().GetFunctionType(CGM.getTypes().getFunctionInfo(MD), 
                                   FPT->isVariadic());
  llvm::Constant *Callee = CGM.GetAddrOfFunction(GlobalDecl(MD), Ty);
  
  llvm::Value *This;
  
  if (ME->isArrow())
    This = EmitScalarExpr(ME->getBase());
  else {
    LValue BaseLV = EmitLValue(ME->getBase());
    This = BaseLV.getAddress();
  }
  
  return EmitCXXMemberCall(MD, Callee, This, 
                           CE->arg_begin(), CE->arg_end());
}

RValue 
CodeGenFunction::EmitCXXOperatorMemberCallExpr(const CXXOperatorCallExpr *E,
                                               const CXXMethodDecl *MD) {
  assert(MD->isInstance() && 
         "Trying to emit a member call expr on a static method!");
  
  
  const FunctionProtoType *FPT = MD->getType()->getAsFunctionProtoType();
  const llvm::Type *Ty = 
  CGM.getTypes().GetFunctionType(CGM.getTypes().getFunctionInfo(MD), 
                                 FPT->isVariadic());
  llvm::Constant *Callee = CGM.GetAddrOfFunction(GlobalDecl(MD), Ty);
  
  llvm::Value *This = EmitLValue(E->getArg(0)).getAddress();
  
  return EmitCXXMemberCall(MD, Callee, This,
                           E->arg_begin() + 1, E->arg_end());
}

llvm::Value *CodeGenFunction::LoadCXXThis() {
  assert(isa<CXXMethodDecl>(CurFuncDecl) && 
         "Must be in a C++ member function decl to load 'this'");
  assert(cast<CXXMethodDecl>(CurFuncDecl)->isInstance() &&
         "Must be in a C++ member function decl to load 'this'");
  
  // FIXME: What if we're inside a block?
  // ans: See how CodeGenFunction::LoadObjCSelf() uses
  // CodeGenFunction::BlockForwardSelf() for how to do this.
  return Builder.CreateLoad(LocalDeclMap[CXXThisDecl], "this");
}

void
CodeGenFunction::EmitCXXConstructorCall(const CXXConstructorDecl *D, 
                                        CXXCtorType Type, 
                                        llvm::Value *This,
                                        CallExpr::const_arg_iterator ArgBeg,
                                        CallExpr::const_arg_iterator ArgEnd) {
  llvm::Value *Callee = CGM.GetAddrOfCXXConstructor(D, Type);

  EmitCXXMemberCall(D, Callee, This, ArgBeg, ArgEnd);
}

void CodeGenFunction::EmitCXXDestructorCall(const CXXDestructorDecl *D, 
                                            CXXDtorType Type,
                                            llvm::Value *This) {
  llvm::Value *Callee = CGM.GetAddrOfCXXDestructor(D, Type);
  
  EmitCXXMemberCall(D, Callee, This, 0, 0);
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

llvm::Value *CodeGenFunction::EmitCXXNewExpr(const CXXNewExpr *E) {
  if (E->isArray()) {
    ErrorUnsupported(E, "new[] expression");
    return llvm::UndefValue::get(ConvertType(E->getType()));
  }
  
  QualType AllocType = E->getAllocatedType();
  FunctionDecl *NewFD = E->getOperatorNew();
  const FunctionProtoType *NewFTy = NewFD->getType()->getAsFunctionProtoType();
  
  CallArgList NewArgs;

  // The allocation size is the first argument.
  QualType SizeTy = getContext().getSizeType();
  llvm::Value *AllocSize = 
    llvm::ConstantInt::get(ConvertType(SizeTy), 
                           getContext().getTypeSize(AllocType) / 8);

  NewArgs.push_back(std::make_pair(RValue::get(AllocSize), SizeTy));
  
  // Emit the rest of the arguments.
  // FIXME: Ideally, this should just use EmitCallArgs.
  CXXNewExpr::const_arg_iterator NewArg = E->placement_arg_begin();

  // First, use the types from the function type.
  // We start at 1 here because the first argument (the allocation size)
  // has already been emitted.
  for (unsigned i = 1, e = NewFTy->getNumArgs(); i != e; ++i, ++NewArg) {
    QualType ArgType = NewFTy->getArgType(i);
    
    assert(getContext().getCanonicalType(ArgType.getNonReferenceType()).
           getTypePtr() == 
           getContext().getCanonicalType(NewArg->getType()).getTypePtr() && 
           "type mismatch in call argument!");
    
    NewArgs.push_back(std::make_pair(EmitCallArg(*NewArg, ArgType), 
                                     ArgType));
    
  }
  
  // Either we've emitted all the call args, or we have a call to a 
  // variadic function.
  assert((NewArg == E->placement_arg_end() || NewFTy->isVariadic()) && 
         "Extra arguments in non-variadic function!");
  
  // If we still have any arguments, emit them using the type of the argument.
  for (CXXNewExpr::const_arg_iterator NewArgEnd = E->placement_arg_end(); 
       NewArg != NewArgEnd; ++NewArg) {
    QualType ArgType = NewArg->getType();
    NewArgs.push_back(std::make_pair(EmitCallArg(*NewArg, ArgType),
                                     ArgType));
  }

  // Emit the call to new.
  RValue RV = 
    EmitCall(CGM.getTypes().getFunctionInfo(NewFTy->getResultType(), NewArgs),
             CGM.GetAddrOfFunction(GlobalDecl(NewFD)),
             NewArgs, NewFD);

  // If an allocation function is declared with an empty exception specification
  // it returns null to indicate failure to allocate storage. [expr.new]p13.
  // (We don't need to check for null when there's no new initializer and
  // we're allocating a POD type).
  bool NullCheckResult = NewFTy->hasEmptyExceptionSpec() &&
    !(AllocType->isPODType() && !E->hasInitializer());

  llvm::BasicBlock *NewNull = 0;
  llvm::BasicBlock *NewNotNull = 0;
  llvm::BasicBlock *NewEnd = 0;

  llvm::Value *NewPtr = RV.getScalarVal();

  if (NullCheckResult) {
    NewNull = createBasicBlock("new.null");
    NewNotNull = createBasicBlock("new.notnull");
    NewEnd = createBasicBlock("new.end");
    
    llvm::Value *IsNull = 
      Builder.CreateICmpEQ(NewPtr, 
                           llvm::Constant::getNullValue(NewPtr->getType()),
                           "isnull");
    
    Builder.CreateCondBr(IsNull, NewNull, NewNotNull);
    EmitBlock(NewNotNull);
  }
  
  NewPtr = Builder.CreateBitCast(NewPtr, ConvertType(E->getType()));
  
  if (AllocType->isPODType()) {
    if (E->getNumConstructorArgs() > 0) {
      assert(E->getNumConstructorArgs() == 1 && 
             "Can only have one argument to initializer of POD type.");

      const Expr *Init = E->getConstructorArg(0);
    
      if (!hasAggregateLLVMType(AllocType)) 
        Builder.CreateStore(EmitScalarExpr(Init), NewPtr);
      else if (AllocType->isAnyComplexType())
        EmitComplexExprIntoAddr(Init, NewPtr, AllocType.isVolatileQualified());
      else
        EmitAggExpr(Init, NewPtr, AllocType.isVolatileQualified());
    }
  } else {
    // Call the constructor.    
    CXXConstructorDecl *Ctor = E->getConstructor();
    
    EmitCXXConstructorCall(Ctor, Ctor_Complete, NewPtr, 
                           E->constructor_arg_begin(), 
                           E->constructor_arg_end());
  }

  if (NullCheckResult) {
    Builder.CreateBr(NewEnd);
    EmitBlock(NewNull);
    Builder.CreateBr(NewEnd);
    EmitBlock(NewEnd);
  
    llvm::PHINode *PHI = Builder.CreatePHI(NewPtr->getType());
    PHI->reserveOperandSpace(2);
    PHI->addIncoming(NewPtr, NewNotNull);
    PHI->addIncoming(llvm::Constant::getNullValue(NewPtr->getType()), NewNull);
    
    NewPtr = PHI;
  }
    
  return NewPtr;
}

static bool canGenerateCXXstructor(const CXXRecordDecl *RD, 
                                   ASTContext &Context) {
  // The class has base classes - we don't support that right now.
  if (RD->getNumBases() > 0)
    return false;
  
  for (CXXRecordDecl::field_iterator I = RD->field_begin(), E = RD->field_end();
         I != E; ++I) {
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

  EmitGlobal(GlobalDecl(D, Ctor_Complete));
  EmitGlobal(GlobalDecl(D, Ctor_Base));
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
  return cast<llvm::Function>(
                      GetOrCreateLLVMFunction(Name, FTy, GlobalDecl(D, Type)));
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
  return cast<llvm::Function>(
                      GetOrCreateLLVMFunction(Name, FTy, GlobalDecl(D, Type)));
}

const char *CodeGenModule::getMangledCXXDtorName(const CXXDestructorDecl *D, 
                                                 CXXDtorType Type) {
  llvm::SmallString<256> Name;
  llvm::raw_svector_ostream Out(Name);
  mangleCXXDtor(D, Type, Context, Out);
  
  Name += '\0';
  return UniqueMangledName(Name.begin(), Name.end());
}
