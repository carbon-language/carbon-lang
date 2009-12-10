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

  ErrorUnsupported(Init, "global variable that binds to a reference");
}

void
CodeGenFunction::EmitCXXGlobalDtorRegistration(llvm::Constant *DtorFn,
                                               llvm::Constant *DeclPtr) {
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
CodeGenModule::EmitCXXGlobalInitFunc() {
  if (CXXGlobalInits.empty())
    return;

  const llvm::FunctionType *FTy
    = llvm::FunctionType::get(llvm::Type::getVoidTy(VMContext),
                              false);

  // Create our global initialization function.
  // FIXME: Should this be tweakable by targets?
  llvm::Function *Fn =
    llvm::Function::Create(FTy, llvm::GlobalValue::InternalLinkage,
                           "__cxx_global_initialization", &TheModule);

  CodeGenFunction(*this).GenerateCXXGlobalInitFunc(Fn,
                                                   &CXXGlobalInits[0],
                                                   CXXGlobalInits.size());
  AddGlobalCtor(Fn);
}

void CodeGenFunction::GenerateCXXGlobalInitFunc(llvm::Function *Fn,
                                                const VarDecl **Decls,
                                                unsigned NumDecls) {
  StartFunction(GlobalDecl(), getContext().VoidTy, Fn, FunctionArgList(),
                SourceLocation());

  for (unsigned i = 0; i != NumDecls; ++i) {
    const VarDecl *D = Decls[i];

    llvm::Constant *DeclPtr = CGM.GetAddrOfGlobalVar(D);
    EmitCXXGlobalVarDeclInit(*D, DeclPtr);
  }
  FinishFunction();
}

void
CodeGenFunction::EmitStaticCXXBlockVarDeclInit(const VarDecl &D,
                                               llvm::GlobalVariable *GV) {
  // FIXME: This should use __cxa_guard_{acquire,release}?

  assert(!getContext().getLangOptions().ThreadsafeStatics &&
         "thread safe statics are currently not supported!");

  llvm::SmallString<256> GuardVName;
  CGM.getMangleContext().mangleGuardVariable(&D, GuardVName);

  // Create the guard variable.
  llvm::GlobalValue *GuardV =
    new llvm::GlobalVariable(CGM.getModule(), llvm::Type::getInt64Ty(VMContext),
                             false, GV->getLinkage(),
                llvm::Constant::getNullValue(llvm::Type::getInt64Ty(VMContext)),
                             GuardVName.str());

  // Load the first byte of the guard variable.
  const llvm::Type *PtrTy
    = llvm::PointerType::get(llvm::Type::getInt8Ty(VMContext), 0);
  llvm::Value *V = Builder.CreateLoad(Builder.CreateBitCast(GuardV, PtrTy),
                                      "tmp");

  // Compare it against 0.
  llvm::Value *nullValue
    = llvm::Constant::getNullValue(llvm::Type::getInt8Ty(VMContext));
  llvm::Value *ICmp = Builder.CreateICmpEQ(V, nullValue , "tobool");

  llvm::BasicBlock *InitBlock = createBasicBlock("init");
  llvm::BasicBlock *EndBlock = createBasicBlock("init.end");

  // If the guard variable is 0, jump to the initializer code.
  Builder.CreateCondBr(ICmp, InitBlock, EndBlock);

  EmitBlock(InitBlock);

  if (D.getType()->isReferenceType()) {
    QualType T = D.getType();
    // We don't want to pass true for IsInitializer here, because a static
    // reference to a temporary does not extend its lifetime.
    RValue RV = EmitReferenceBindingToExpr(D.getInit(), T,
                                           /*IsInitializer=*/false);
    EmitStoreOfScalar(RV.getScalarVal(), GV, /*Volatile=*/false, T);

  } else
    EmitDeclInit(*this, D, GV);

  Builder.CreateStore(llvm::ConstantInt::get(llvm::Type::getInt8Ty(VMContext),
                                             1),
                      Builder.CreateBitCast(GuardV, PtrTy));

  EmitBlock(EndBlock);
}
