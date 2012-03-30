//===--- CGDeclCXX.cpp - Emit LLVM Code for C++ declarations --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code dealing with code generation of C++ declarations
//
//===----------------------------------------------------------------------===//

#include "CodeGenFunction.h"
#include "CGObjCRuntime.h"
#include "CGCXXABI.h"
#include "clang/Frontend/CodeGenOptions.h"
#include "llvm/Intrinsics.h"

using namespace clang;
using namespace CodeGen;

static void EmitDeclInit(CodeGenFunction &CGF, const VarDecl &D,
                         llvm::Constant *DeclPtr) {
  assert(D.hasGlobalStorage() && "VarDecl must have global storage!");
  assert(!D.getType()->isReferenceType() && 
         "Should not call EmitDeclInit on a reference!");
  
  ASTContext &Context = CGF.getContext();

  CharUnits alignment = Context.getDeclAlign(&D);
  QualType type = D.getType();
  LValue lv = CGF.MakeAddrLValue(DeclPtr, type, alignment);

  const Expr *Init = D.getInit();
  if (!CGF.hasAggregateLLVMType(type)) {
    CodeGenModule &CGM = CGF.CGM;
    if (lv.isObjCStrong())
      CGM.getObjCRuntime().EmitObjCGlobalAssign(CGF, CGF.EmitScalarExpr(Init),
                                                DeclPtr, D.isThreadSpecified());
    else if (lv.isObjCWeak())
      CGM.getObjCRuntime().EmitObjCWeakAssign(CGF, CGF.EmitScalarExpr(Init),
                                              DeclPtr);
    else
      CGF.EmitScalarInit(Init, &D, lv, false);
  } else if (type->isAnyComplexType()) {
    CGF.EmitComplexExprIntoAddr(Init, DeclPtr, lv.isVolatile());
  } else {
    CGF.EmitAggExpr(Init, AggValueSlot::forLValue(lv,AggValueSlot::IsDestructed,
                                          AggValueSlot::DoesNotNeedGCBarriers,
                                                  AggValueSlot::IsNotAliased));
  }
}

/// Emit code to cause the destruction of the given variable with
/// static storage duration.
static void EmitDeclDestroy(CodeGenFunction &CGF, const VarDecl &D,
                            llvm::Constant *addr) {
  CodeGenModule &CGM = CGF.CGM;

  // FIXME:  __attribute__((cleanup)) ?
  
  QualType type = D.getType();
  QualType::DestructionKind dtorKind = type.isDestructedType();

  switch (dtorKind) {
  case QualType::DK_none:
    return;

  case QualType::DK_cxx_destructor:
    break;

  case QualType::DK_objc_strong_lifetime:
  case QualType::DK_objc_weak_lifetime:
    // We don't care about releasing objects during process teardown.
    return;
  }

  llvm::Constant *function;
  llvm::Constant *argument;

  // Special-case non-array C++ destructors, where there's a function
  // with the right signature that we can just call.
  const CXXRecordDecl *record = 0;
  if (dtorKind == QualType::DK_cxx_destructor &&
      (record = type->getAsCXXRecordDecl())) {
    assert(!record->hasTrivialDestructor());
    CXXDestructorDecl *dtor = record->getDestructor();

    function = CGM.GetAddrOfCXXDestructor(dtor, Dtor_Complete);
    argument = addr;

  // Otherwise, the standard logic requires a helper function.
  } else {
    function = CodeGenFunction(CGM).generateDestroyHelper(addr, type,
                                                  CGF.getDestroyer(dtorKind),
                                                  CGF.needsEHCleanup(dtorKind));
    argument = llvm::Constant::getNullValue(CGF.Int8PtrTy);
  }

  CGF.EmitCXXGlobalDtorRegistration(function, argument);
}

/// Emit code to cause the variable at the given address to be considered as
/// constant from this point onwards.
static void EmitDeclInvariant(CodeGenFunction &CGF, const VarDecl &D,
                              llvm::Constant *Addr) {
  // Don't emit the intrinsic if we're not optimizing.
  if (!CGF.CGM.getCodeGenOpts().OptimizationLevel)
    return;

  // Grab the llvm.invariant.start intrinsic.
  llvm::Intrinsic::ID InvStartID = llvm::Intrinsic::invariant_start;
  llvm::Constant *InvariantStart = CGF.CGM.getIntrinsic(InvStartID);

  // Emit a call with the size in bytes of the object.
  CharUnits WidthChars = CGF.getContext().getTypeSizeInChars(D.getType());
  uint64_t Width = WidthChars.getQuantity();
  llvm::Value *Args[2] = { llvm::ConstantInt::getSigned(CGF.Int64Ty, Width),
                           llvm::ConstantExpr::getBitCast(Addr, CGF.Int8PtrTy)};
  CGF.Builder.CreateCall(InvariantStart, Args);
}

void CodeGenFunction::EmitCXXGlobalVarDeclInit(const VarDecl &D,
                                               llvm::Constant *DeclPtr,
                                               bool PerformInit) {

  const Expr *Init = D.getInit();
  QualType T = D.getType();

  if (!T->isReferenceType()) {
    if (PerformInit)
      EmitDeclInit(*this, D, DeclPtr);
    if (CGM.isTypeConstant(D.getType(), true))
      EmitDeclInvariant(*this, D, DeclPtr);
    else
      EmitDeclDestroy(*this, D, DeclPtr);
    return;
  }

  assert(PerformInit && "cannot have constant initializer which needs "
         "destruction for reference");
  unsigned Alignment = getContext().getDeclAlign(&D).getQuantity();
  RValue RV = EmitReferenceBindingToExpr(Init, &D);
  EmitStoreOfScalar(RV.getScalarVal(), DeclPtr, false, Alignment, T);
}

void
CodeGenFunction::EmitCXXGlobalDtorRegistration(llvm::Constant *DtorFn,
                                               llvm::Constant *DeclPtr) {
  // Generate a global destructor entry if not using __cxa_atexit.
  if (!CGM.getCodeGenOpts().CXAAtExit) {
    CGM.AddCXXDtorEntry(DtorFn, DeclPtr);
    return;
  }

  // Get the destructor function type
  llvm::Type *DtorFnTy = llvm::FunctionType::get(VoidTy, Int8PtrTy, false);
  DtorFnTy = llvm::PointerType::getUnqual(DtorFnTy);

  llvm::Type *Params[] = { DtorFnTy, Int8PtrTy, Int8PtrTy };

  // Get the __cxa_atexit function type
  // extern "C" int __cxa_atexit ( void (*f)(void *), void *p, void *d );
  llvm::FunctionType *AtExitFnTy =
    llvm::FunctionType::get(ConvertType(getContext().IntTy), Params, false);

  llvm::Constant *AtExitFn = CGM.CreateRuntimeFunction(AtExitFnTy,
                                                       "__cxa_atexit");
  if (llvm::Function *Fn = dyn_cast<llvm::Function>(AtExitFn))
    Fn->setDoesNotThrow();

  llvm::Constant *Handle = CGM.CreateRuntimeVariable(Int8PtrTy,
                                                     "__dso_handle");
  llvm::Value *Args[3] = { llvm::ConstantExpr::getBitCast(DtorFn, DtorFnTy),
                           llvm::ConstantExpr::getBitCast(DeclPtr, Int8PtrTy),
                           llvm::ConstantExpr::getBitCast(Handle, Int8PtrTy) };
  Builder.CreateCall(AtExitFn, Args);
}

void CodeGenFunction::EmitCXXGuardedInit(const VarDecl &D,
                                         llvm::Constant *addr,
                                         bool PerformInit) {
  // If we've been asked to forbid guard variables, emit an error now.
  // This diagnostic is hard-coded for Darwin's use case;  we can find
  // better phrasing if someone else needs it.
  if (CGM.getCodeGenOpts().ForbidGuardVariables)
    CGM.Error(D.getLocation(),
              "this initialization requires a guard variable, which "
              "the kernel does not support");

  CGM.getCXXABI().EmitGuardedInit(*this, D, addr, PerformInit);
}

static llvm::Function *
CreateGlobalInitOrDestructFunction(CodeGenModule &CGM,
                                   llvm::FunctionType *FTy,
                                   const Twine &Name) {
  llvm::Function *Fn =
    llvm::Function::Create(FTy, llvm::GlobalValue::InternalLinkage,
                           Name, &CGM.getModule());
  if (!CGM.getContext().getLangOpts().AppleKext) {
    // Set the section if needed.
    if (const char *Section = 
          CGM.getContext().getTargetInfo().getStaticInitSectionSpecifier())
      Fn->setSection(Section);
  }

  if (!CGM.getLangOpts().Exceptions)
    Fn->setDoesNotThrow();

  return Fn;
}

void
CodeGenModule::EmitCXXGlobalVarDeclInitFunc(const VarDecl *D,
                                            llvm::GlobalVariable *Addr,
                                            bool PerformInit) {
  llvm::FunctionType *FTy = llvm::FunctionType::get(VoidTy, false);

  // Create a variable initialization function.
  llvm::Function *Fn =
    CreateGlobalInitOrDestructFunction(*this, FTy, "__cxx_global_var_init");

  CodeGenFunction(*this).GenerateCXXGlobalVarDeclInitFunc(Fn, D, Addr,
                                                          PerformInit);

  if (D->hasAttr<InitPriorityAttr>()) {
    unsigned int order = D->getAttr<InitPriorityAttr>()->getPriority();
    OrderGlobalInits Key(order, PrioritizedCXXGlobalInits.size());
    PrioritizedCXXGlobalInits.push_back(std::make_pair(Key, Fn));
    DelayedCXXInitPosition.erase(D);
  }
  else {
    llvm::DenseMap<const Decl *, unsigned>::iterator I =
      DelayedCXXInitPosition.find(D);
    if (I == DelayedCXXInitPosition.end()) {
      CXXGlobalInits.push_back(Fn);
    } else {
      assert(CXXGlobalInits[I->second] == 0);
      CXXGlobalInits[I->second] = Fn;
      DelayedCXXInitPosition.erase(I);
    }
  }
}

void
CodeGenModule::EmitCXXGlobalInitFunc() {
  while (!CXXGlobalInits.empty() && !CXXGlobalInits.back())
    CXXGlobalInits.pop_back();

  if (CXXGlobalInits.empty() && PrioritizedCXXGlobalInits.empty())
    return;

  llvm::FunctionType *FTy = llvm::FunctionType::get(VoidTy, false);

  // Create our global initialization function.
  llvm::Function *Fn = 
    CreateGlobalInitOrDestructFunction(*this, FTy, "_GLOBAL__I_a");

  if (!PrioritizedCXXGlobalInits.empty()) {
    SmallVector<llvm::Constant*, 8> LocalCXXGlobalInits;
    llvm::array_pod_sort(PrioritizedCXXGlobalInits.begin(), 
                         PrioritizedCXXGlobalInits.end()); 
    for (unsigned i = 0; i < PrioritizedCXXGlobalInits.size(); i++) {
      llvm::Function *Fn = PrioritizedCXXGlobalInits[i].second;
      LocalCXXGlobalInits.push_back(Fn);
    }
    LocalCXXGlobalInits.append(CXXGlobalInits.begin(), CXXGlobalInits.end());
    CodeGenFunction(*this).GenerateCXXGlobalInitFunc(Fn,
                                                    &LocalCXXGlobalInits[0],
                                                    LocalCXXGlobalInits.size());
  }
  else
    CodeGenFunction(*this).GenerateCXXGlobalInitFunc(Fn,
                                                     &CXXGlobalInits[0],
                                                     CXXGlobalInits.size());
  AddGlobalCtor(Fn);
  CXXGlobalInits.clear();
  PrioritizedCXXGlobalInits.clear();
}

void CodeGenModule::EmitCXXGlobalDtorFunc() {
  if (CXXGlobalDtors.empty())
    return;

  llvm::FunctionType *FTy = llvm::FunctionType::get(VoidTy, false);

  // Create our global destructor function.
  llvm::Function *Fn =
    CreateGlobalInitOrDestructFunction(*this, FTy, "_GLOBAL__D_a");

  CodeGenFunction(*this).GenerateCXXGlobalDtorFunc(Fn, CXXGlobalDtors);
  AddGlobalDtor(Fn);
}

/// Emit the code necessary to initialize the given global variable.
void CodeGenFunction::GenerateCXXGlobalVarDeclInitFunc(llvm::Function *Fn,
                                                       const VarDecl *D,
                                                 llvm::GlobalVariable *Addr,
                                                       bool PerformInit) {
  StartFunction(GlobalDecl(), getContext().VoidTy, Fn,
                getTypes().arrangeNullaryFunction(),
                FunctionArgList(), SourceLocation());

  // Use guarded initialization if the global variable is weak. This
  // occurs for, e.g., instantiated static data members and
  // definitions explicitly marked weak.
  if (Addr->getLinkage() == llvm::GlobalValue::WeakODRLinkage ||
      Addr->getLinkage() == llvm::GlobalValue::WeakAnyLinkage) {
    EmitCXXGuardedInit(*D, Addr, PerformInit);
  } else {
    EmitCXXGlobalVarDeclInit(*D, Addr, PerformInit);
  }

  FinishFunction();
}

void CodeGenFunction::GenerateCXXGlobalInitFunc(llvm::Function *Fn,
                                                llvm::Constant **Decls,
                                                unsigned NumDecls) {
  StartFunction(GlobalDecl(), getContext().VoidTy, Fn,
                getTypes().arrangeNullaryFunction(),
                FunctionArgList(), SourceLocation());

  RunCleanupsScope Scope(*this);

  // When building in Objective-C++ ARC mode, create an autorelease pool
  // around the global initializers.
  if (getLangOpts().ObjCAutoRefCount && getLangOpts().CPlusPlus) {    
    llvm::Value *token = EmitObjCAutoreleasePoolPush();
    EmitObjCAutoreleasePoolCleanup(token);
  }
  
  for (unsigned i = 0; i != NumDecls; ++i)
    if (Decls[i])
      Builder.CreateCall(Decls[i]);    

  Scope.ForceCleanup();
  
  FinishFunction();
}

void CodeGenFunction::GenerateCXXGlobalDtorFunc(llvm::Function *Fn,
                  const std::vector<std::pair<llvm::WeakVH, llvm::Constant*> >
                                                &DtorsAndObjects) {
  StartFunction(GlobalDecl(), getContext().VoidTy, Fn,
                getTypes().arrangeNullaryFunction(),
                FunctionArgList(), SourceLocation());

  // Emit the dtors, in reverse order from construction.
  for (unsigned i = 0, e = DtorsAndObjects.size(); i != e; ++i) {
    llvm::Value *Callee = DtorsAndObjects[e - i - 1].first;
    llvm::CallInst *CI = Builder.CreateCall(Callee,
                                            DtorsAndObjects[e - i - 1].second);
    // Make sure the call and the callee agree on calling convention.
    if (llvm::Function *F = dyn_cast<llvm::Function>(Callee))
      CI->setCallingConv(F->getCallingConv());
  }

  FinishFunction();
}

/// generateDestroyHelper - Generates a helper function which, when
/// invoked, destroys the given object.
llvm::Function * 
CodeGenFunction::generateDestroyHelper(llvm::Constant *addr,
                                       QualType type,
                                       Destroyer *destroyer,
                                       bool useEHCleanupForArray) {
  FunctionArgList args;
  ImplicitParamDecl dst(0, SourceLocation(), 0, getContext().VoidPtrTy);
  args.push_back(&dst);
  
  const CGFunctionInfo &FI = 
    CGM.getTypes().arrangeFunctionDeclaration(getContext().VoidTy, args,
                                              FunctionType::ExtInfo(),
                                              /*variadic*/ false);
  llvm::FunctionType *FTy = CGM.getTypes().GetFunctionType(FI);
  llvm::Function *fn = 
    CreateGlobalInitOrDestructFunction(CGM, FTy, "__cxx_global_array_dtor");

  StartFunction(GlobalDecl(), getContext().VoidTy, fn, FI, args,
                SourceLocation());

  emitDestroy(addr, type, destroyer, useEHCleanupForArray);
  
  FinishFunction();
  
  return fn;
}
