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

void CodeGenFunction::EmitCXXGlobalVarDeclInit(const VarDecl &D,
                                               llvm::Constant *DeclPtr) {
  assert(D.hasGlobalStorage() &&
         "VarDecl must have global storage!");

  const Expr *Init = D.getInit();
  QualType T = D.getType();
  bool isVolatile = getContext().getCanonicalType(T).isVolatileQualified();

  if (T->isReferenceType()) {
    ErrorUnsupported(Init, "global variable that binds to a reference");
  } else if (!hasAggregateLLVMType(T)) {
    llvm::Value *V = EmitScalarExpr(Init);
    EmitStoreOfScalar(V, DeclPtr, isVolatile, T);
  } else if (T->isAnyComplexType()) {
    EmitComplexExprIntoAddr(Init, DeclPtr, isVolatile);
  } else {
    EmitAggExpr(Init, DeclPtr, isVolatile);
    // Avoid generating destructor(s) for initialized objects. 
    if (!isa<CXXConstructExpr>(Init))
      return;
    const ConstantArrayType *Array = getContext().getAsConstantArrayType(T);
    if (Array)
      T = getContext().getBaseElementType(Array);
    
    if (const RecordType *RT = T->getAs<RecordType>()) {
      CXXRecordDecl *RD = cast<CXXRecordDecl>(RT->getDecl());
      if (!RD->hasTrivialDestructor()) {
        llvm::Constant *DtorFn;
        if (Array) {
          DtorFn = CodeGenFunction(CGM).GenerateCXXAggrDestructorHelper(
                                                RD->getDestructor(getContext()), 
                                                Array, DeclPtr);
          DeclPtr = 
            llvm::Constant::getNullValue(llvm::Type::getInt8PtrTy(VMContext));
        }
        else
          DtorFn = CGM.GetAddrOfCXXDestructor(RD->getDestructor(getContext()), 
                                              Dtor_Complete);                                
        EmitCXXGlobalDtorRegistration(DtorFn, DeclPtr);
      }
    }
  }
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

  EmitCXXGlobalVarDeclInit(D, GV);

  Builder.CreateStore(llvm::ConstantInt::get(llvm::Type::getInt8Ty(VMContext),
                                             1),
                      Builder.CreateBitCast(GuardV, PtrTy));

  EmitBlock(EndBlock);
}
