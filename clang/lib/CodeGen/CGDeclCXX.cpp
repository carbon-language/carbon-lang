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
#include "clang/CodeGen/CodeGenOptions.h"
using namespace clang;
using namespace CodeGen;

static void EmitDeclInit(CodeGenFunction &CGF, const VarDecl &D,
                         llvm::Constant *DeclPtr) {
  assert(D.hasGlobalStorage() && "VarDecl must have global storage!");
  assert(!D.getType()->isReferenceType() && 
         "Should not call EmitDeclInit on a reference!");
  
  CodeGenModule &CGM = CGF.CGM;
  ASTContext &Context = CGF.getContext();
    
  const Expr *Init = D.getInit();
  QualType T = D.getType();
  bool isVolatile = Context.getCanonicalType(T).isVolatileQualified();

  if (!CGF.hasAggregateLLVMType(T)) {
    llvm::Value *V = CGF.EmitScalarExpr(Init);
    CGF.EmitStoreOfScalar(V, DeclPtr, isVolatile, T);
  } else if (T->isAnyComplexType()) {
    CGF.EmitComplexExprIntoAddr(Init, DeclPtr, isVolatile);
  } else {
    CGF.EmitAggExpr(Init, DeclPtr, isVolatile);
    
    // Avoid generating destructor(s) for initialized objects. 
    if (!isa<CXXConstructExpr>(Init))
      return;
    
    const ConstantArrayType *Array = Context.getAsConstantArrayType(T);
    if (Array)
      T = Context.getBaseElementType(Array);
    
    const RecordType *RT = T->getAs<RecordType>();
    if (!RT)
      return;
    
    CXXRecordDecl *RD = cast<CXXRecordDecl>(RT->getDecl());
    if (RD->hasTrivialDestructor())
      return;
    
    CXXDestructorDecl *Dtor = RD->getDestructor(Context);
    
    llvm::Constant *DtorFn;
    if (Array) {
      DtorFn = 
        CodeGenFunction(CGM).GenerateCXXAggrDestructorHelper(Dtor, 
                                                             Array, 
                                                             DeclPtr);
      const llvm::Type *Int8PtrTy =
        llvm::Type::getInt8PtrTy(CGM.getLLVMContext());
      DeclPtr = llvm::Constant::getNullValue(Int8PtrTy);
     } else
      DtorFn = CGM.GetAddrOfCXXDestructor(Dtor, Dtor_Complete);

    CGF.EmitCXXGlobalDtorRegistration(DtorFn, DeclPtr);
  }
}

void CodeGenFunction::EmitCXXGlobalVarDeclInit(const VarDecl &D,
                                               llvm::Constant *DeclPtr) {

  const Expr *Init = D.getInit();
  QualType T = D.getType();

  if (!T->isReferenceType()) {
    EmitDeclInit(*this, D, DeclPtr);
    return;
  }
  if (Init->isLvalue(getContext()) == Expr::LV_Valid) {
    RValue RV = EmitReferenceBindingToExpr(Init, /*IsInitializer=*/true);
    EmitStoreOfScalar(RV.getScalarVal(), DeclPtr, false, T);
    return;
  }
  ErrorUnsupported(Init, 
                   "global variable that binds reference to a non-lvalue");
}

void
CodeGenFunction::EmitCXXGlobalDtorRegistration(llvm::Constant *DtorFn,
                                               llvm::Constant *DeclPtr) {
  // Generate a global destructor entry if not using __cxa_atexit.
  if (!CGM.getCodeGenOpts().CXAAtExit) {
    CGM.AddCXXDtorEntry(DtorFn, DeclPtr);
    return;
  }

  const llvm::Type *Int8PtrTy = 
    llvm::Type::getInt8Ty(VMContext)->getPointerTo();

  std::vector<const llvm::Type *> Params;
  Params.push_back(Int8PtrTy);

  // Get the destructor function type
  const llvm::Type *DtorFnTy =
    llvm::FunctionType::get(llvm::Type::getVoidTy(VMContext), Params, false);
  DtorFnTy = llvm::PointerType::getUnqual(DtorFnTy);

  Params.clear();
  Params.push_back(DtorFnTy);
  Params.push_back(Int8PtrTy);
  Params.push_back(Int8PtrTy);

  // Get the __cxa_atexit function type
  // extern "C" int __cxa_atexit ( void (*f)(void *), void *p, void *d );
  const llvm::FunctionType *AtExitFnTy =
    llvm::FunctionType::get(ConvertType(getContext().IntTy), Params, false);

  llvm::Constant *AtExitFn = CGM.CreateRuntimeFunction(AtExitFnTy,
                                                       "__cxa_atexit");

  llvm::Constant *Handle = CGM.CreateRuntimeVariable(Int8PtrTy,
                                                     "__dso_handle");
  llvm::Value *Args[3] = { llvm::ConstantExpr::getBitCast(DtorFn, DtorFnTy),
                           llvm::ConstantExpr::getBitCast(DeclPtr, Int8PtrTy),
                           llvm::ConstantExpr::getBitCast(Handle, Int8PtrTy) };
  Builder.CreateCall(AtExitFn, &Args[0], llvm::array_endof(Args));
}

void
CodeGenModule::EmitCXXGlobalVarDeclInitFunc(const VarDecl *D) {
  const llvm::FunctionType *FTy
    = llvm::FunctionType::get(llvm::Type::getVoidTy(VMContext),
                              false);

  // Create a variable initialization function.
  llvm::Function *Fn =
    llvm::Function::Create(FTy, llvm::GlobalValue::InternalLinkage,
                           "__cxx_global_var_init", &TheModule);

  CodeGenFunction(*this).GenerateCXXGlobalVarDeclInitFunc(Fn, D);

  CXXGlobalInits.push_back(Fn);
}

void
CodeGenModule::EmitCXXGlobalInitFunc() {
  if (CXXGlobalInits.empty())
    return;

  const llvm::FunctionType *FTy
    = llvm::FunctionType::get(llvm::Type::getVoidTy(VMContext),
                              false);

  // Create our global initialization function.
  llvm::Function *Fn =
    llvm::Function::Create(FTy, llvm::GlobalValue::InternalLinkage,
                           "_GLOBAL__I_a", &TheModule);

  CodeGenFunction(*this).GenerateCXXGlobalInitFunc(Fn,
                                                   &CXXGlobalInits[0],
                                                   CXXGlobalInits.size());
  AddGlobalCtor(Fn);
}

void CodeGenModule::AddCXXDtorEntry(llvm::Constant *DtorFn,
                                    llvm::Constant *Object) {
  CXXGlobalDtors.push_back(std::make_pair(DtorFn, Object));
}

void CodeGenModule::EmitCXXGlobalDtorFunc() {
  if (CXXGlobalDtors.empty())
    return;

  const llvm::FunctionType *FTy
    = llvm::FunctionType::get(llvm::Type::getVoidTy(VMContext),
                              false);

  // Create our global destructor function.
  llvm::Function *Fn =
    llvm::Function::Create(FTy, llvm::GlobalValue::InternalLinkage,
                           "_GLOBAL__D_a", &TheModule);

  CodeGenFunction(*this).GenerateCXXGlobalDtorFunc(Fn, CXXGlobalDtors);
  AddGlobalDtor(Fn);
}

void CodeGenFunction::GenerateCXXGlobalVarDeclInitFunc(llvm::Function *Fn,
                                                       const VarDecl *D) {
  StartFunction(GlobalDecl(), getContext().VoidTy, Fn, FunctionArgList(),
                SourceLocation());

  llvm::Constant *DeclPtr = CGM.GetAddrOfGlobalVar(D);
  EmitCXXGlobalVarDeclInit(*D, DeclPtr);

  FinishFunction();
}

void CodeGenFunction::GenerateCXXGlobalInitFunc(llvm::Function *Fn,
                                                llvm::Constant **Decls,
                                                unsigned NumDecls) {
  StartFunction(GlobalDecl(), getContext().VoidTy, Fn, FunctionArgList(),
                SourceLocation());

  for (unsigned i = 0; i != NumDecls; ++i)
    Builder.CreateCall(Decls[i]);

  FinishFunction();
}

void CodeGenFunction::GenerateCXXGlobalDtorFunc(llvm::Function *Fn,
                const std::vector<std::pair<llvm::Constant*, llvm::Constant*> >
                                                &DtorsAndObjects) {
  StartFunction(GlobalDecl(), getContext().VoidTy, Fn, FunctionArgList(),
                SourceLocation());

  // Emit the dtors, in reverse order from construction.
  for (unsigned i = 0, e = DtorsAndObjects.size(); i != e; ++i) {
    llvm::Constant *Callee = DtorsAndObjects[e - i - 1].first;
    llvm::CallInst *CI = Builder.CreateCall(Callee,
                                            DtorsAndObjects[e - i - 1].second);
    // Make sure the call and the callee agree on calling convention.
    if (llvm::Function *F = dyn_cast<llvm::Function>(Callee))
      CI->setCallingConv(F->getCallingConv());
  }

  FinishFunction();
}

static llvm::Constant *getGuardAcquireFn(CodeGenFunction &CGF) {
  // int __cxa_guard_acquire(__int64_t *guard_object);
  
  const llvm::Type *Int64PtrTy = 
    llvm::Type::getInt64PtrTy(CGF.getLLVMContext());
  
  std::vector<const llvm::Type*> Args(1, Int64PtrTy);
  
  const llvm::FunctionType *FTy =
    llvm::FunctionType::get(CGF.ConvertType(CGF.getContext().IntTy),
                            Args, /*isVarArg=*/false);
  
  return CGF.CGM.CreateRuntimeFunction(FTy, "__cxa_guard_acquire");
}

static llvm::Constant *getGuardReleaseFn(CodeGenFunction &CGF) {
  // void __cxa_guard_release(__int64_t *guard_object);
  
  const llvm::Type *Int64PtrTy = 
    llvm::Type::getInt64PtrTy(CGF.getLLVMContext());
  
  std::vector<const llvm::Type*> Args(1, Int64PtrTy);
  
  const llvm::FunctionType *FTy =
  llvm::FunctionType::get(llvm::Type::getVoidTy(CGF.getLLVMContext()),
                          Args, /*isVarArg=*/false);
  
  return CGF.CGM.CreateRuntimeFunction(FTy, "__cxa_guard_release");
}

static llvm::Constant *getGuardAbortFn(CodeGenFunction &CGF) {
  // void __cxa_guard_abort(__int64_t *guard_object);
  
  const llvm::Type *Int64PtrTy = 
    llvm::Type::getInt64PtrTy(CGF.getLLVMContext());
  
  std::vector<const llvm::Type*> Args(1, Int64PtrTy);
  
  const llvm::FunctionType *FTy =
  llvm::FunctionType::get(llvm::Type::getVoidTy(CGF.getLLVMContext()),
                          Args, /*isVarArg=*/false);
  
  return CGF.CGM.CreateRuntimeFunction(FTy, "__cxa_guard_abort");
}

void
CodeGenFunction::EmitStaticCXXBlockVarDeclInit(const VarDecl &D,
                                               llvm::GlobalVariable *GV) {
  // Bail out early if this initializer isn't reachable.
  if (!Builder.GetInsertBlock()) return;

  bool ThreadsafeStatics = getContext().getLangOptions().ThreadsafeStatics;
  
  llvm::SmallString<256> GuardVName;
  CGM.getMangleContext().mangleGuardVariable(&D, GuardVName);

  // Create the guard variable.
  const llvm::Type *Int64Ty = llvm::Type::getInt64Ty(VMContext);
  llvm::GlobalValue *GuardVariable =
    new llvm::GlobalVariable(CGM.getModule(), Int64Ty,
                             false, GV->getLinkage(),
                             llvm::Constant::getNullValue(Int64Ty),
                             GuardVName.str());

  // Load the first byte of the guard variable.
  const llvm::Type *PtrTy
    = llvm::PointerType::get(llvm::Type::getInt8Ty(VMContext), 0);
  llvm::Value *V = 
    Builder.CreateLoad(Builder.CreateBitCast(GuardVariable, PtrTy), "tmp");

  llvm::BasicBlock *InitCheckBlock = createBasicBlock("init.check");
  llvm::BasicBlock *EndBlock = createBasicBlock("init.end");

  // Check if the first byte of the guard variable is zero.
  Builder.CreateCondBr(Builder.CreateIsNull(V, "tobool"), 
                       InitCheckBlock, EndBlock);

  EmitBlock(InitCheckBlock);

  if (ThreadsafeStatics) {
    // Call __cxa_guard_acquire.
    V = Builder.CreateCall(getGuardAcquireFn(*this), GuardVariable);
               
    llvm::BasicBlock *InitBlock = createBasicBlock("init");
  
    Builder.CreateCondBr(Builder.CreateIsNotNull(V, "tobool"),
                         InitBlock, EndBlock);
  
    EmitBlock(InitBlock);

    if (Exceptions) {
      EHCleanupBlock Cleanup(*this);
    
      // Call __cxa_guard_abort.
      Builder.CreateCall(getGuardAbortFn(*this), GuardVariable);
    }
  }

  if (D.getType()->isReferenceType()) {
    QualType T = D.getType();
    // We don't want to pass true for IsInitializer here, because a static
    // reference to a temporary does not extend its lifetime.
    RValue RV = EmitReferenceBindingToExpr(D.getInit(),
                                           /*IsInitializer=*/false);
    EmitStoreOfScalar(RV.getScalarVal(), GV, /*Volatile=*/false, T);

  } else
    EmitDeclInit(*this, D, GV);

  if (ThreadsafeStatics) {
    // Call __cxa_guard_release.
    Builder.CreateCall(getGuardReleaseFn(*this), GuardVariable);
  } else {
    llvm::Value *One = 
      llvm::ConstantInt::get(llvm::Type::getInt8Ty(VMContext), 1);
    Builder.CreateStore(One, Builder.CreateBitCast(GuardVariable, PtrTy));
  }

  EmitBlock(EndBlock);
}
