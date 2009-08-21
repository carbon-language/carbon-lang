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
#include "clang/AST/RecordLayout.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "llvm/ADT/StringExtras.h"
using namespace clang;
using namespace CodeGen;

void 
CodeGenFunction::EmitCXXGlobalDtorRegistration(const CXXDestructorDecl *Dtor,
                                               llvm::Constant *DeclPtr) {
  // FIXME: This is ABI dependent and we use the Itanium ABI.
  
  const llvm::Type *Int8PtrTy = 
    llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(VMContext));
  
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
  
  llvm::Constant *DtorFn = CGM.GetAddrOfCXXDestructor(Dtor, Dtor_Complete);
  
  llvm::Value *Args[3] = { llvm::ConstantExpr::getBitCast(DtorFn, DtorFnTy),
                           llvm::ConstantExpr::getBitCast(DeclPtr, Int8PtrTy),
                           llvm::ConstantExpr::getBitCast(Handle, Int8PtrTy) };
  Builder.CreateCall(AtExitFn, &Args[0], llvm::array_endof(Args));
}

void CodeGenFunction::EmitCXXGlobalVarDeclInit(const VarDecl &D, 
                                               llvm::Constant *DeclPtr) {
  assert(D.hasGlobalStorage() &&
         "VarDecl must have global storage!");
  
  const Expr *Init = D.getInit();
  QualType T = D.getType();
  
  if (T->isReferenceType()) {
    ErrorUnsupported(Init, "global variable that binds to a reference");
  } else if (!hasAggregateLLVMType(T)) {
    llvm::Value *V = EmitScalarExpr(Init);
    EmitStoreOfScalar(V, DeclPtr, T.isVolatileQualified(), T);
  } else if (T->isAnyComplexType()) {
    EmitComplexExprIntoAddr(Init, DeclPtr, T.isVolatileQualified());
  } else {
    EmitAggExpr(Init, DeclPtr, T.isVolatileQualified());
    
    if (const RecordType *RT = T->getAs<RecordType>()) {
      CXXRecordDecl *RD = cast<CXXRecordDecl>(RT->getDecl());
      if (!RD->hasTrivialDestructor())
        EmitCXXGlobalDtorRegistration(RD->getDestructor(getContext()), DeclPtr);
    }
  }
}

void
CodeGenModule::EmitCXXGlobalInitFunc() {
  if (CXXGlobalInits.empty())
    return;
  
  const llvm::FunctionType *FTy = llvm::FunctionType::get(llvm::Type::getVoidTy(VMContext),
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
  StartFunction(0, getContext().VoidTy, Fn, FunctionArgList(), 
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
  llvm::raw_svector_ostream GuardVOut(GuardVName);
  mangleGuardVariable(&D, getContext(), GuardVOut);
  
  // Create the guard variable.
  llvm::GlobalValue *GuardV = 
    new llvm::GlobalVariable(CGM.getModule(), llvm::Type::getInt64Ty(VMContext), false,
                             GV->getLinkage(),
                             llvm::Constant::getNullValue(llvm::Type::getInt64Ty(VMContext)),
                             GuardVName.str());
  
  // Load the first byte of the guard variable.
  const llvm::Type *PtrTy = llvm::PointerType::get(llvm::Type::getInt8Ty(VMContext), 0);
  llvm::Value *V = Builder.CreateLoad(Builder.CreateBitCast(GuardV, PtrTy), 
                                      "tmp");
  
  // Compare it against 0.
  llvm::Value *nullValue = llvm::Constant::getNullValue(llvm::Type::getInt8Ty(VMContext));
  llvm::Value *ICmp = Builder.CreateICmpEQ(V, nullValue , "tobool");
  
  llvm::BasicBlock *InitBlock = createBasicBlock("init");
  llvm::BasicBlock *EndBlock = createBasicBlock("init.end");

  // If the guard variable is 0, jump to the initializer code.
  Builder.CreateCondBr(ICmp, InitBlock, EndBlock);
                         
  EmitBlock(InitBlock);

  EmitCXXGlobalVarDeclInit(D, GV);

  Builder.CreateStore(llvm::ConstantInt::get(llvm::Type::getInt8Ty(VMContext), 1),
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

  if (MD->isVirtual()) {
    ErrorUnsupported(CE, "virtual dispatch");
  }

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
  
  if (MD->isCopyAssignment()) {
    const CXXRecordDecl *ClassDecl = cast<CXXRecordDecl>(MD->getDeclContext());
    if (ClassDecl->hasTrivialCopyAssignment()) {
      assert(!ClassDecl->hasUserDeclaredCopyAssignment() &&
             "EmitCXXOperatorMemberCallExpr - user declared copy assignment");
      llvm::Value *This = EmitLValue(E->getArg(0)).getAddress();
      llvm::Value *Src = EmitLValue(E->getArg(1)).getAddress();
      QualType Ty = E->getType();
      EmitAggregateCopy(This, Src, Ty);
      return RValue::get(This);
    }
  }
  
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

static bool
GetNestedPaths(llvm::SmallVectorImpl<const CXXRecordDecl *> &NestedBasePaths,
               const CXXRecordDecl *ClassDecl,
               const CXXRecordDecl *BaseClassDecl) {
  for (CXXRecordDecl::base_class_const_iterator i = ClassDecl->bases_begin(),
      e = ClassDecl->bases_end(); i != e; ++i) {
    if (i->isVirtual())
      continue;
    const CXXRecordDecl *Base = 
      cast<CXXRecordDecl>(i->getType()->getAs<RecordType>()->getDecl());
    if (Base == BaseClassDecl) {
      NestedBasePaths.push_back(BaseClassDecl);
      return true;
    }
  }
  // BaseClassDecl not an immediate base of ClassDecl.
  for (CXXRecordDecl::base_class_const_iterator i = ClassDecl->bases_begin(),
       e = ClassDecl->bases_end(); i != e; ++i) {
    if (i->isVirtual())
      continue;
    const CXXRecordDecl *Base = 
      cast<CXXRecordDecl>(i->getType()->getAs<RecordType>()->getDecl());
    if (GetNestedPaths(NestedBasePaths, Base, BaseClassDecl)) {
      NestedBasePaths.push_back(Base);
      return true;
    }
  }
  return false;
}

llvm::Value *CodeGenFunction::AddressCXXOfBaseClass(llvm::Value *BaseValue,
                                          const CXXRecordDecl *ClassDecl, 
                                          const CXXRecordDecl *BaseClassDecl) {
  if (ClassDecl == BaseClassDecl)
    return BaseValue;
  
  llvm::Type *I8Ptr = llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(VMContext));
  llvm::SmallVector<const CXXRecordDecl *, 16> NestedBasePaths;
  GetNestedPaths(NestedBasePaths, ClassDecl, BaseClassDecl);
  assert(NestedBasePaths.size() > 0 && 
         "AddressCXXOfBaseClass - inheritence path failed");
  NestedBasePaths.push_back(ClassDecl);
  uint64_t Offset = 0;
  
  // Accessing a member of the base class. Must add delata to
  // the load of 'this'.
  for (unsigned i = NestedBasePaths.size()-1; i > 0; i--) {
    const CXXRecordDecl *DerivedClass = NestedBasePaths[i];
    const CXXRecordDecl *BaseClass = NestedBasePaths[i-1];
    const ASTRecordLayout &Layout = 
      getContext().getASTRecordLayout(DerivedClass);
    Offset += Layout.getBaseClassOffset(BaseClass) / 8;
  }
  llvm::Value *OffsetVal = 
    llvm::ConstantInt::get(
                  CGM.getTypes().ConvertType(CGM.getContext().LongTy), Offset);
  BaseValue = Builder.CreateBitCast(BaseValue, I8Ptr);
  BaseValue = Builder.CreateGEP(BaseValue, OffsetVal, "add.ptr");
  QualType BTy = 
    getContext().getCanonicalType(
      getContext().getTypeDeclType(const_cast<CXXRecordDecl*>(BaseClassDecl)));
  const llvm::Type *BasePtr = ConvertType(BTy);
  BasePtr = llvm::PointerType::getUnqual(BasePtr);
  BaseValue = Builder.CreateBitCast(BaseValue, BasePtr);
  return BaseValue;
}

/// EmitCXXAggrConstructorCall - This routine essentially creates a (nested)
/// for-loop to call the default constructor on individual members of the
/// array. 'Array' is the array type, 'This' is llvm pointer of the start
/// of the array and 'D' is the default costructor Decl for elements of the
/// array. It is assumed that all relevant checks have been made by the
/// caller.
void
CodeGenFunction::EmitCXXAggrConstructorCall(const CXXConstructorDecl *D,
                                            const ArrayType *Array,
                                            llvm::Value *This) {
  const ConstantArrayType *CA = dyn_cast<ConstantArrayType>(Array);
  assert(CA && "Do we support VLA for construction ?");
  
  // Create a temporary for the loop index and initialize it with 0.
  llvm::Value *IndexPtr = CreateTempAlloca(llvm::Type::getInt64Ty(VMContext),
                                           "loop.index");
  llvm::Value* zeroConstant = 
    llvm::Constant::getNullValue(llvm::Type::getInt64Ty(VMContext));
  Builder.CreateStore(zeroConstant, IndexPtr, false);
  
  // Start the loop with a block that tests the condition.
  llvm::BasicBlock *CondBlock = createBasicBlock("for.cond");
  llvm::BasicBlock *AfterFor = createBasicBlock("for.end");
  
  EmitBlock(CondBlock);
  
  llvm::BasicBlock *ForBody = createBasicBlock("for.body");
  
  // Generate: if (loop-index < number-of-elements fall to the loop body,
  // otherwise, go to the block after the for-loop.
  uint64_t NumElements = CA->getSize().getZExtValue();
  llvm::Value * NumElementsPtr = 
  llvm::ConstantInt::get(llvm::Type::getInt64Ty(VMContext), NumElements);
  llvm::Value *Counter = Builder.CreateLoad(IndexPtr);
  llvm::Value *IsLess = Builder.CreateICmpULT(Counter, NumElementsPtr, 
                                              "isless");
  // If the condition is true, execute the body.
  Builder.CreateCondBr(IsLess, ForBody, AfterFor);
  
  EmitBlock(ForBody);
  
  llvm::BasicBlock *ContinueBlock = createBasicBlock("for.inc");
  // Inside the loop body, emit the constructor call on the array element.
  Counter = Builder.CreateLoad(IndexPtr);
  if (const ConstantArrayType *CAT = 
      dyn_cast<ConstantArrayType>(Array->getElementType())) {
    uint64_t delta = getContext().getConstantArrayElementCount(CAT);
    // Address = This + delta*Counter for current loop iteration.
    llvm::Value *DeltaPtr = 
      llvm::ConstantInt::get(llvm::Type::getInt64Ty(VMContext), delta);
    DeltaPtr = Builder.CreateMul(Counter, DeltaPtr, "mul");
    llvm::Value *Address = 
      Builder.CreateInBoundsGEP(This, DeltaPtr, "arrayidx");
    EmitCXXAggrConstructorCall(D, CAT, Address);
  } 
  else {
    llvm::Value *Address = Builder.CreateInBoundsGEP(This, Counter, "arrayidx");
    EmitCXXConstructorCall(D, Ctor_Complete, Address, 0, 0);
  }
    
  EmitBlock(ContinueBlock);
  
  // Emit the increment of the loop counter.
  llvm::Value *NextVal = llvm::ConstantInt::get(Counter->getType(), 1);
  Counter = Builder.CreateLoad(IndexPtr);
  NextVal = Builder.CreateAdd(Counter, NextVal, "inc");
  Builder.CreateStore(NextVal, IndexPtr, false);
  
  // Finally, branch back up to the condition for the next iteration.
  EmitBranch(CondBlock);
  
  // Emit the fall-through block.
  EmitBlock(AfterFor, true);
}

/// EmitCXXAggrDestructorCall - calls the default destructor on array
/// elements in reverse order of construction.
void
CodeGenFunction::EmitCXXAggrDestructorCall(const CXXDestructorDecl *D,
                                           const ArrayType *Array,
                                           llvm::Value *This) {
  const ConstantArrayType *CA = dyn_cast<ConstantArrayType>(Array);
  assert(CA && "Do we support VLA for destruction ?");
  llvm::Value *One = llvm::ConstantInt::get(llvm::Type::getInt64Ty(VMContext), 
                                            1);
  uint64_t ElementCount = getContext().getConstantArrayElementCount(CA);
  // Create a temporary for the loop index and initialize it with count of
  // array elements.
  llvm::Value *IndexPtr = CreateTempAlloca(llvm::Type::getInt64Ty(VMContext),
                                           "loop.index");
  // Index = ElementCount;
  llvm::Value* UpperCount = 
    llvm::ConstantInt::get(llvm::Type::getInt64Ty(VMContext), ElementCount);
  Builder.CreateStore(UpperCount, IndexPtr, false);
  
  // Start the loop with a block that tests the condition.
  llvm::BasicBlock *CondBlock = createBasicBlock("for.cond");
  llvm::BasicBlock *AfterFor = createBasicBlock("for.end");
  
  EmitBlock(CondBlock);
  
  llvm::BasicBlock *ForBody = createBasicBlock("for.body");
  
  // Generate: if (loop-index != 0 fall to the loop body,
  // otherwise, go to the block after the for-loop.
  llvm::Value* zeroConstant = 
    llvm::Constant::getNullValue(llvm::Type::getInt64Ty(VMContext));
  llvm::Value *Counter = Builder.CreateLoad(IndexPtr);
  llvm::Value *IsNE = Builder.CreateICmpNE(Counter, zeroConstant,
                                            "isne");
  // If the condition is true, execute the body.
  Builder.CreateCondBr(IsNE, ForBody, AfterFor);
  
  EmitBlock(ForBody);
  
  llvm::BasicBlock *ContinueBlock = createBasicBlock("for.inc");
  // Inside the loop body, emit the constructor call on the array element.
  Counter = Builder.CreateLoad(IndexPtr);
  Counter = Builder.CreateSub(Counter, One);
  llvm::Value *Address = Builder.CreateInBoundsGEP(This, Counter, "arrayidx");
  EmitCXXDestructorCall(D, Dtor_Complete, Address);
  
  EmitBlock(ContinueBlock);
  
  // Emit the decrement of the loop counter.
  Counter = Builder.CreateLoad(IndexPtr);
  Counter = Builder.CreateSub(Counter, One, "dec");
  Builder.CreateStore(Counter, IndexPtr, false);
  
  // Finally, branch back up to the condition for the next iteration.
  EmitBranch(CondBlock);
  
  // Emit the fall-through block.
  EmitBlock(AfterFor, true);
}

void
CodeGenFunction::EmitCXXConstructorCall(const CXXConstructorDecl *D, 
                                        CXXCtorType Type, 
                                        llvm::Value *This,
                                        CallExpr::const_arg_iterator ArgBeg,
                                        CallExpr::const_arg_iterator ArgEnd) {
  if (D->isCopyConstructor(getContext())) {
    const CXXRecordDecl *ClassDecl = cast<CXXRecordDecl>(D->getDeclContext());
    if (ClassDecl->hasTrivialCopyConstructor()) {
      assert(!ClassDecl->hasUserDeclaredCopyConstructor() &&
             "EmitCXXConstructorCall - user declared copy constructor");
      const Expr *E = (*ArgBeg);
      QualType Ty = E->getType();
      llvm::Value *Src = EmitLValue(E).getAddress();
      EmitAggregateCopy(This, Src, Ty);
      return;
    }
  }
  
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
  cast<CXXRecordDecl>(E->getType()->getAs<RecordType>()->getDecl());
  if (RD->hasTrivialConstructor())
    return;

  // Code gen optimization to eliminate copy constructor and return 
  // its first argument instead.
  if (E->isElidable()) {
    CXXConstructExpr::const_arg_iterator i = E->arg_begin();
    EmitAggExpr((*i), Dest, false);
    return;
  }
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

void CodeGenFunction::EmitCXXDeleteExpr(const CXXDeleteExpr *E) {
  if (E->isArrayForm()) {
    ErrorUnsupported(E, "delete[] expression");
    return;
  };

  QualType DeleteTy = 
    E->getArgument()->getType()->getAs<PointerType>()->getPointeeType();
  
  llvm::Value *Ptr = EmitScalarExpr(E->getArgument());
  
  // Null check the pointer.
  llvm::BasicBlock *DeleteNotNull = createBasicBlock("delete.notnull");
  llvm::BasicBlock *DeleteEnd = createBasicBlock("delete.end");

  llvm::Value *IsNull = 
    Builder.CreateICmpEQ(Ptr, llvm::Constant::getNullValue(Ptr->getType()),
                         "isnull");
    
  Builder.CreateCondBr(IsNull, DeleteEnd, DeleteNotNull);
  EmitBlock(DeleteNotNull);
    
  // Call the destructor if necessary.
  if (const RecordType *RT = DeleteTy->getAs<RecordType>()) {
    if (CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(RT->getDecl())) {
      if (!RD->hasTrivialDestructor()) {
        const CXXDestructorDecl *Dtor = RD->getDestructor(getContext());
        if (Dtor->isVirtual()) {
          ErrorUnsupported(E, "delete expression with virtual destructor");
          return;
        }
        
        EmitCXXDestructorCall(Dtor, Dtor_Complete, Ptr);
      }
    }
  }
  
  // Call delete.
  FunctionDecl *DeleteFD = E->getOperatorDelete();
  const FunctionProtoType *DeleteFTy = 
    DeleteFD->getType()->getAsFunctionProtoType();
  
  CallArgList DeleteArgs;

  QualType ArgTy = DeleteFTy->getArgType(0);
  llvm::Value *DeletePtr = Builder.CreateBitCast(Ptr, ConvertType(ArgTy));
  DeleteArgs.push_back(std::make_pair(RValue::get(DeletePtr), ArgTy));
  
  // Emit the call to delete.
  EmitCall(CGM.getTypes().getFunctionInfo(DeleteFTy->getResultType(), 
                                          DeleteArgs),
           CGM.GetAddrOfFunction(GlobalDecl(DeleteFD)),
           DeleteArgs, DeleteFD);
  
  EmitBlock(DeleteEnd);
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

llvm::Constant *CodeGenModule::GenerateRtti(const CXXRecordDecl *RD) {
  llvm::Type *Ptr8Ty;
  Ptr8Ty = llvm::PointerType::get(llvm::Type::getInt8Ty(VMContext), 0);
  llvm::Constant *Rtti = llvm::Constant::getNullValue(Ptr8Ty);

  if (!getContext().getLangOptions().Rtti)
    return Rtti;

  llvm::SmallString<256> OutName;
  llvm::raw_svector_ostream Out(OutName);
  QualType ClassTy;
  ClassTy = getContext().getTagDeclType(RD);
  mangleCXXRtti(ClassTy, getContext(), Out);
  llvm::GlobalVariable::LinkageTypes linktype;
  linktype = llvm::GlobalValue::WeakAnyLinkage;
  std::vector<llvm::Constant *> info;
  // assert(0 && "FIXME: implement rtti descriptor");
  // FIXME: descriptor
  info.push_back(llvm::Constant::getNullValue(Ptr8Ty));
  // assert(0 && "FIXME: implement rtti ts");
  // FIXME: TS
  info.push_back(llvm::Constant::getNullValue(Ptr8Ty));

  llvm::Constant *C;
  llvm::ArrayType *type = llvm::ArrayType::get(Ptr8Ty, info.size());
  C = llvm::ConstantArray::get(type, info);
  Rtti = new llvm::GlobalVariable(getModule(), type, true, linktype, C,
                                  Out.str());
  Rtti = llvm::ConstantExpr::getBitCast(Rtti, Ptr8Ty);
  return Rtti;
}

class VtableBuilder {
  std::vector<llvm::Constant *> &methods;
  llvm::Type *Ptr8Ty;
  /// Class - The most derived class that this vtable is being built for.
  const CXXRecordDecl *Class;
  /// BLayout - Layout for the most derived class that this vtable is being
  /// built for.
  const ASTRecordLayout &BLayout;
  llvm::SmallSet<const CXXRecordDecl *, 32> IndirectPrimary;
  llvm::SmallSet<const CXXRecordDecl *, 32> SeenVBase;
  llvm::Constant *rtti;
  llvm::LLVMContext &VMContext;
  CodeGenModule &CGM;  // Per-module state.
  /// Index - Maps a method decl into a vtable index.  Useful for virtual
  /// dispatch codegen.
  llvm::DenseMap<const CXXMethodDecl *, int32_t> Index;
  typedef CXXRecordDecl::method_iterator method_iter;
public:
  VtableBuilder(std::vector<llvm::Constant *> &meth,
                const CXXRecordDecl *c,
                CodeGenModule &cgm)
    : methods(meth), Class(c), BLayout(cgm.getContext().getASTRecordLayout(c)),
      rtti(cgm.GenerateRtti(c)), VMContext(cgm.getModule().getContext()),
      CGM(cgm) {
    Ptr8Ty = llvm::PointerType::get(llvm::Type::getInt8Ty(VMContext), 0);
  }

  llvm::Constant *GenerateVcall(const CXXMethodDecl *MD,
                                const CXXRecordDecl *RD,
                                bool VBoundary,
                                bool SecondaryVirtual) {
    llvm::Constant *m = 0;

    // FIXME: vcall: offset for virtual base for this function
    if (SecondaryVirtual || VBoundary)
      m = llvm::Constant::getNullValue(Ptr8Ty);
    return m;
  }

  void GenerateVcalls(const CXXRecordDecl *RD, bool VBoundary,
                      bool SecondaryVirtual) {
    llvm::Constant *m;

    for (method_iter mi = RD->method_begin(),
           me = RD->method_end(); mi != me; ++mi) {
      if (mi->isVirtual()) {
        m = GenerateVcall(*mi, RD, VBoundary, SecondaryVirtual);
        if (m)
          methods.push_back(m);
      }
    }
  }

  void GenerateVBaseOffsets(std::vector<llvm::Constant *> &offsets,
                            const CXXRecordDecl *RD, uint64_t Offset) {
    for (CXXRecordDecl::base_class_const_iterator i =RD->bases_begin(),
           e = RD->bases_end(); i != e; ++i) {
      const CXXRecordDecl *Base = 
        cast<CXXRecordDecl>(i->getType()->getAs<RecordType>()->getDecl());
      if (i->isVirtual() && !SeenVBase.count(Base)) {
        SeenVBase.insert(Base);
        int64_t BaseOffset = -(Offset/8) + BLayout.getVBaseClassOffset(Base)/8;
        llvm::Constant *m;
        m = llvm::ConstantInt::get(llvm::Type::getInt64Ty(VMContext),BaseOffset);
        m = llvm::ConstantExpr::getIntToPtr(m, Ptr8Ty);
        offsets.push_back(m);
      }
      GenerateVBaseOffsets(offsets, Base, Offset);
    }
  }

  void StartNewTable() {
    SeenVBase.clear();
  }

  inline uint32_t nottoobig(uint64_t t) {
    assert(t < (uint32_t)-1ULL || "vtable too big");
    return t;
  }
#if 0
  inline uint32_t nottoobig(uint32_t t) {
    return t;
  }
#endif

  void AddMethod(const CXXMethodDecl *MD, int32_t FirstIndex) {
    typedef CXXMethodDecl::method_iterator meth_iter;

    llvm::Constant *m;
    m = CGM.GetAddrOfFunction(GlobalDecl(MD), Ptr8Ty);
    m = llvm::ConstantExpr::getBitCast(m, Ptr8Ty);

    // FIXME: Don't like the nested loops.  For very large inheritance
    // heirarchies we could have a table on the side with the final overridder
    // and just replace each instance of an overridden method once.  Would be
    // nice to measure the cost/benefit on real code.

    // If we can find a previously allocated slot for this, reuse it.
    for (meth_iter mi = MD->begin_overridden_methods(),
           e = MD->end_overridden_methods();
         mi != e; ++mi) {
      const CXXMethodDecl *OMD = *mi;
      llvm::Constant *om;
      om = CGM.GetAddrOfFunction(GlobalDecl(OMD), Ptr8Ty);
      om = llvm::ConstantExpr::getBitCast(om, Ptr8Ty);

      for (int32_t i = FirstIndex, e = nottoobig(methods.size()); i != e; ++i) {
        // FIXME: begin_overridden_methods might be too lax, covariance */
        if (methods[i] == om) {
          methods[i] = m;
          Index[MD] = i;
          return;
        }
      }
    }

    // else allocate a new slot.
    Index[MD] = methods.size();
    methods.push_back(m);
  }

  void GenerateMethods(const CXXRecordDecl *RD, int32_t FirstIndex) {
    for (method_iter mi = RD->method_begin(), me = RD->method_end(); mi != me;
         ++mi)
      if (mi->isVirtual())
        AddMethod(*mi, FirstIndex);
  }

  void GenerateVtableForBase(const CXXRecordDecl *RD,
                             bool forPrimary,
                             bool VBoundary,
                             int64_t Offset,
                             bool ForVirtualBase,
                             int32_t FirstIndex) {
    llvm::Constant *m = llvm::Constant::getNullValue(Ptr8Ty);

    if (RD && !RD->isDynamicClass())
      return;

    const ASTRecordLayout &Layout = CGM.getContext().getASTRecordLayout(RD);
    const CXXRecordDecl *PrimaryBase = Layout.getPrimaryBase(); 
    const bool PrimaryBaseWasVirtual = Layout.getPrimaryBaseWasVirtual();

    if (VBoundary || forPrimary || ForVirtualBase) {
      // then comes the the vcall offsets for all our functions...
      GenerateVcalls(RD, VBoundary, !forPrimary && ForVirtualBase);
    }

    // The virtual base offsets come first...
    // FIXME: Audit, is this right?
    if (PrimaryBase == 0 || forPrimary || !PrimaryBaseWasVirtual) {
      std::vector<llvm::Constant *> offsets;
      GenerateVBaseOffsets(offsets, RD, Offset);
      for (std::vector<llvm::Constant *>::reverse_iterator i = offsets.rbegin(),
             e = offsets.rend(); i != e; ++i)
        methods.push_back(*i);
    }

    bool Top = true;

    // vtables are composed from the chain of primaries.
    if (PrimaryBase) {
      if (PrimaryBaseWasVirtual)
        IndirectPrimary.insert(PrimaryBase);
      Top = false;
      GenerateVtableForBase(PrimaryBase, true, PrimaryBaseWasVirtual|VBoundary,
                            Offset, PrimaryBaseWasVirtual, FirstIndex);
    }

    if (Top) {
      int64_t BaseOffset;
      if (ForVirtualBase) {
        BaseOffset = -(BLayout.getVBaseClassOffset(RD) / 8);
      } else
        BaseOffset = -Offset/8;
      m = llvm::ConstantInt::get(llvm::Type::getInt64Ty(VMContext), BaseOffset);
      m = llvm::ConstantExpr::getIntToPtr(m, Ptr8Ty);
      methods.push_back(m);
      methods.push_back(rtti);
    }

    // And add the virtuals for the class to the primary vtable.
    GenerateMethods(RD, FirstIndex);

    // and then the non-virtual bases.
    for (CXXRecordDecl::base_class_const_iterator i = RD->bases_begin(),
           e = RD->bases_end(); i != e; ++i) {
      if (i->isVirtual())
        continue;
      const CXXRecordDecl *Base = 
        cast<CXXRecordDecl>(i->getType()->getAs<RecordType>()->getDecl());
      if (Base != PrimaryBase || PrimaryBaseWasVirtual) {
        uint64_t o = Offset + Layout.getBaseClassOffset(Base);
        StartNewTable();
        FirstIndex = methods.size();
        GenerateVtableForBase(Base, true, false, o, false, FirstIndex);
      }
    }
  }

  void GenerateVtableForVBases(const CXXRecordDecl *RD,
                               const CXXRecordDecl *Class) {
    for (CXXRecordDecl::base_class_const_iterator i = RD->bases_begin(),
           e = RD->bases_end(); i != e; ++i) {
      const CXXRecordDecl *Base = 
        cast<CXXRecordDecl>(i->getType()->getAs<RecordType>()->getDecl());
      if (i->isVirtual() && !IndirectPrimary.count(Base)) {
        // Mark it so we don't output it twice.
        IndirectPrimary.insert(Base);
        StartNewTable();
        int64_t BaseOffset = BLayout.getVBaseClassOffset(Base);
        int32_t FirstIndex = methods.size();
        GenerateVtableForBase(Base, false, true, BaseOffset, true, FirstIndex);
      }
      if (Base->getNumVBases())
        GenerateVtableForVBases(Base, Class);
    }
  }
};

llvm::Value *CodeGenFunction::GenerateVtable(const CXXRecordDecl *RD) {
  llvm::SmallString<256> OutName;
  llvm::raw_svector_ostream Out(OutName);
  QualType ClassTy;
  ClassTy = getContext().getTagDeclType(RD);
  mangleCXXVtable(ClassTy, getContext(), Out);
  llvm::GlobalVariable::LinkageTypes linktype;
  linktype = llvm::GlobalValue::WeakAnyLinkage;
  std::vector<llvm::Constant *> methods;
  llvm::Type *Ptr8Ty=llvm::PointerType::get(llvm::Type::getInt8Ty(VMContext),0);
  int64_t Offset = 0;

  Offset += LLVMPointerWidth;
  Offset += LLVMPointerWidth;

  VtableBuilder b(methods, RD, CGM);

  // First comes the vtables for all the non-virtual bases...
  b.GenerateVtableForBase(RD, true, false, 0, false, 0);

  // then the vtables for all the virtual bases.
  b.GenerateVtableForVBases(RD, RD);

  llvm::Constant *C;
  llvm::ArrayType *type = llvm::ArrayType::get(Ptr8Ty, methods.size());
  C = llvm::ConstantArray::get(type, methods);
  llvm::Value *vtable = new llvm::GlobalVariable(CGM.getModule(), type, true,
                                                 linktype, C, Out.str());
  vtable = Builder.CreateBitCast(vtable, Ptr8Ty);
  vtable = Builder.CreateGEP(vtable,
                       llvm::ConstantInt::get(llvm::Type::getInt64Ty(VMContext),
                                                    Offset/8));
  return vtable;
}

/// EmitClassAggrMemberwiseCopy - This routine generates code to copy a class
/// array of objects from SrcValue to DestValue. Copying can be either a bitwise
/// copy or via a copy constructor call.
void CodeGenFunction::EmitClassAggrMemberwiseCopy(llvm::Value *Dest, 
                                            llvm::Value *Src,
                                            const ArrayType *Array,
                                            const CXXRecordDecl *BaseClassDecl, 
                                            QualType Ty) {
  const ConstantArrayType *CA = dyn_cast<ConstantArrayType>(Array);
  assert(CA && "VLA cannot be copied over");
  bool BitwiseCopy = BaseClassDecl->hasTrivialCopyConstructor();
  
  // Create a temporary for the loop index and initialize it with 0.
  llvm::Value *IndexPtr = CreateTempAlloca(llvm::Type::getInt64Ty(VMContext),
                                           "loop.index");
  llvm::Value* zeroConstant = 
    llvm::Constant::getNullValue(llvm::Type::getInt64Ty(VMContext));
    Builder.CreateStore(zeroConstant, IndexPtr, false);
  // Start the loop with a block that tests the condition.
  llvm::BasicBlock *CondBlock = createBasicBlock("for.cond");
  llvm::BasicBlock *AfterFor = createBasicBlock("for.end");
  
  EmitBlock(CondBlock);
  
  llvm::BasicBlock *ForBody = createBasicBlock("for.body");
  // Generate: if (loop-index < number-of-elements fall to the loop body,
  // otherwise, go to the block after the for-loop.
  uint64_t NumElements = getContext().getConstantArrayElementCount(CA);
  llvm::Value * NumElementsPtr = 
    llvm::ConstantInt::get(llvm::Type::getInt64Ty(VMContext), NumElements);
  llvm::Value *Counter = Builder.CreateLoad(IndexPtr);
  llvm::Value *IsLess = Builder.CreateICmpULT(Counter, NumElementsPtr, 
                                              "isless");
  // If the condition is true, execute the body.
  Builder.CreateCondBr(IsLess, ForBody, AfterFor);
  
  EmitBlock(ForBody);
  llvm::BasicBlock *ContinueBlock = createBasicBlock("for.inc");
  // Inside the loop body, emit the constructor call on the array element.
  Counter = Builder.CreateLoad(IndexPtr);
  Src = Builder.CreateInBoundsGEP(Src, Counter, "srcaddress");
  Dest = Builder.CreateInBoundsGEP(Dest, Counter, "destaddress");
  if (BitwiseCopy)
    EmitAggregateCopy(Dest, Src, Ty);
  else if (CXXConstructorDecl *BaseCopyCtor = 
           BaseClassDecl->getCopyConstructor(getContext(), 0)) {
    llvm::Value *Callee = CGM.GetAddrOfCXXConstructor(BaseCopyCtor, 
                                                      Ctor_Complete);
    CallArgList CallArgs;
    // Push the this (Dest) ptr.
    CallArgs.push_back(std::make_pair(RValue::get(Dest),
                                      BaseCopyCtor->getThisType(getContext())));
    
    // Push the Src ptr.
    CallArgs.push_back(std::make_pair(RValue::get(Src),
                                      BaseCopyCtor->getParamDecl(0)->getType()));
    QualType ResultType = 
      BaseCopyCtor->getType()->getAsFunctionType()->getResultType();
    EmitCall(CGM.getTypes().getFunctionInfo(ResultType, CallArgs),
             Callee, CallArgs, BaseCopyCtor);
  }
  EmitBlock(ContinueBlock);
  
  // Emit the increment of the loop counter.
  llvm::Value *NextVal = llvm::ConstantInt::get(Counter->getType(), 1);
  Counter = Builder.CreateLoad(IndexPtr);
  NextVal = Builder.CreateAdd(Counter, NextVal, "inc");
  Builder.CreateStore(NextVal, IndexPtr, false);
  
  // Finally, branch back up to the condition for the next iteration.
  EmitBranch(CondBlock);
  
  // Emit the fall-through block.
  EmitBlock(AfterFor, true);
}

/// EmitClassMemberwiseCopy - This routine generates code to copy a class
/// object from SrcValue to DestValue. Copying can be either a bitwise copy
/// or via a copy constructor call.
void CodeGenFunction::EmitClassMemberwiseCopy(
                        llvm::Value *Dest, llvm::Value *Src,
                        const CXXRecordDecl *ClassDecl, 
                        const CXXRecordDecl *BaseClassDecl, QualType Ty) {
  if (ClassDecl) {
    Dest = AddressCXXOfBaseClass(Dest, ClassDecl, BaseClassDecl);
    Src = AddressCXXOfBaseClass(Src, ClassDecl, BaseClassDecl) ;
  }
  if (BaseClassDecl->hasTrivialCopyConstructor()) {
    EmitAggregateCopy(Dest, Src, Ty);
    return;
  }
  
  if (CXXConstructorDecl *BaseCopyCtor = 
      BaseClassDecl->getCopyConstructor(getContext(), 0)) {
    llvm::Value *Callee = CGM.GetAddrOfCXXConstructor(BaseCopyCtor, 
                                                      Ctor_Complete);
    CallArgList CallArgs;
    // Push the this (Dest) ptr.
    CallArgs.push_back(std::make_pair(RValue::get(Dest),
                                      BaseCopyCtor->getThisType(getContext())));
    
    // Push the Src ptr.
    CallArgs.push_back(std::make_pair(RValue::get(Src),
                       BaseCopyCtor->getParamDecl(0)->getType()));
    QualType ResultType = 
    BaseCopyCtor->getType()->getAsFunctionType()->getResultType();
    EmitCall(CGM.getTypes().getFunctionInfo(ResultType, CallArgs),
             Callee, CallArgs, BaseCopyCtor);
  }
}

/// EmitClassCopyAssignment - This routine generates code to copy assign a class
/// object from SrcValue to DestValue. Assignment can be either a bitwise 
/// assignment of via an assignment operator call.
void CodeGenFunction::EmitClassCopyAssignment(
                                        llvm::Value *Dest, llvm::Value *Src,
                                        const CXXRecordDecl *ClassDecl, 
                                        const CXXRecordDecl *BaseClassDecl, 
                                        QualType Ty) {
  if (ClassDecl) {
    Dest = AddressCXXOfBaseClass(Dest, ClassDecl, BaseClassDecl);
    Src = AddressCXXOfBaseClass(Src, ClassDecl, BaseClassDecl) ;
  }
  if (BaseClassDecl->hasTrivialCopyAssignment()) {
    EmitAggregateCopy(Dest, Src, Ty);
    return;
  }
  
  const CXXMethodDecl *MD = 0;
  bool ConstCopyAssignOp = BaseClassDecl->hasConstCopyAssignment(getContext(), 
                                                                 MD);
  assert(ConstCopyAssignOp && "EmitClassCopyAssignment - missing copy assign");
  (void)ConstCopyAssignOp;

  const FunctionProtoType *FPT = MD->getType()->getAsFunctionProtoType();
  const llvm::Type *LTy = 
    CGM.getTypes().GetFunctionType(CGM.getTypes().getFunctionInfo(MD), 
                                   FPT->isVariadic());
  llvm::Constant *Callee = CGM.GetAddrOfFunction(GlobalDecl(MD), LTy);
    
  CallArgList CallArgs;
  // Push the this (Dest) ptr.
  CallArgs.push_back(std::make_pair(RValue::get(Dest),
                                    MD->getThisType(getContext())));
    
  // Push the Src ptr.
  CallArgs.push_back(std::make_pair(RValue::get(Src),
                                    MD->getParamDecl(0)->getType()));
  QualType ResultType = 
    MD->getType()->getAsFunctionType()->getResultType();
  EmitCall(CGM.getTypes().getFunctionInfo(ResultType, CallArgs),
           Callee, CallArgs, MD);
}

/// SynthesizeDefaultConstructor - synthesize a default constructor
void 
CodeGenFunction::SynthesizeDefaultConstructor(const CXXConstructorDecl *CD,
                                              const FunctionDecl *FD,
                                              llvm::Function *Fn,
                                              const FunctionArgList &Args) {
  StartFunction(FD, FD->getResultType(), Fn, Args, SourceLocation());
  EmitCtorPrologue(CD);
  FinishFunction();
}

/// SynthesizeCXXCopyConstructor - This routine implicitly defines body of a copy
/// constructor, in accordance with section 12.8 (p7 and p8) of C++03
/// The implicitly-defined copy constructor for class X performs a memberwise 
/// copy of its subobjects. The order of copying is the same as the order 
/// of initialization of bases and members in a user-defined constructor
/// Each subobject is copied in the manner appropriate to its type:
///  if the subobject is of class type, the copy constructor for the class is 
///  used;
///  if the subobject is an array, each element is copied, in the manner 
///  appropriate to the element type;
///  if the subobject is of scalar type, the built-in assignment operator is 
///  used.
/// Virtual base class subobjects shall be copied only once by the 
/// implicitly-defined copy constructor 

void CodeGenFunction::SynthesizeCXXCopyConstructor(const CXXConstructorDecl *CD,
                                       const FunctionDecl *FD,
                                       llvm::Function *Fn,
                                       const FunctionArgList &Args) {
  const CXXRecordDecl *ClassDecl = cast<CXXRecordDecl>(CD->getDeclContext());
  assert(!ClassDecl->hasUserDeclaredCopyConstructor() &&
         "SynthesizeCXXCopyConstructor - copy constructor has definition already");
  StartFunction(FD, FD->getResultType(), Fn, Args, SourceLocation());
 
  FunctionArgList::const_iterator i = Args.begin();
  const VarDecl *ThisArg = i->first;
  llvm::Value *ThisObj = GetAddrOfLocalVar(ThisArg);
  llvm::Value *LoadOfThis = Builder.CreateLoad(ThisObj, "this");
  const VarDecl *SrcArg = (i+1)->first;
  llvm::Value *SrcObj = GetAddrOfLocalVar(SrcArg);
  llvm::Value *LoadOfSrc = Builder.CreateLoad(SrcObj);
  
  for (CXXRecordDecl::base_class_const_iterator Base = ClassDecl->bases_begin();
       Base != ClassDecl->bases_end(); ++Base) {
    // FIXME. copy constrution of virtual base NYI
    if (Base->isVirtual())
      continue;
    
    CXXRecordDecl *BaseClassDecl
      = cast<CXXRecordDecl>(Base->getType()->getAs<RecordType>()->getDecl());
    EmitClassMemberwiseCopy(LoadOfThis, LoadOfSrc, ClassDecl, BaseClassDecl,
                            Base->getType());
  }
  
  for (CXXRecordDecl::field_iterator Field = ClassDecl->field_begin(),
       FieldEnd = ClassDecl->field_end();
       Field != FieldEnd; ++Field) {
    QualType FieldType = getContext().getCanonicalType((*Field)->getType());
    const ConstantArrayType *Array = 
      getContext().getAsConstantArrayType(FieldType);
    if (Array)
      FieldType = getContext().getBaseElementType(FieldType);
        
    if (const RecordType *FieldClassType = FieldType->getAs<RecordType>()) {
      CXXRecordDecl *FieldClassDecl
        = cast<CXXRecordDecl>(FieldClassType->getDecl());
      LValue LHS = EmitLValueForField(LoadOfThis, *Field, false, 0);
      LValue RHS = EmitLValueForField(LoadOfSrc, *Field, false, 0);
      if (Array) {
        const llvm::Type *BasePtr = ConvertType(FieldType);
        BasePtr = llvm::PointerType::getUnqual(BasePtr);
        llvm::Value *DestBaseAddrPtr = 
          Builder.CreateBitCast(LHS.getAddress(), BasePtr);
        llvm::Value *SrcBaseAddrPtr = 
          Builder.CreateBitCast(RHS.getAddress(), BasePtr);
        EmitClassAggrMemberwiseCopy(DestBaseAddrPtr, SrcBaseAddrPtr, Array,
                                    FieldClassDecl, FieldType);
      }
      else        
        EmitClassMemberwiseCopy(LHS.getAddress(), RHS.getAddress(), 
                                0 /*ClassDecl*/, FieldClassDecl, FieldType);
      continue;
    }
    // Do a built-in assignment of scalar data members.
    LValue LHS = EmitLValueForField(LoadOfThis, *Field, false, 0);
    LValue RHS = EmitLValueForField(LoadOfSrc, *Field, false, 0);
    RValue RVRHS = EmitLoadOfLValue(RHS, FieldType);
    EmitStoreThroughLValue(RVRHS, LHS, FieldType);
  }
  FinishFunction();
}  

/// SynthesizeCXXCopyAssignment - Implicitly define copy assignment operator.
/// Before the implicitly-declared copy assignment operator for a class is 
/// implicitly defined, all implicitly- declared copy assignment operators for 
/// its direct base classes and its nonstatic data members shall have been 
/// implicitly defined. [12.8-p12]
/// The implicitly-defined copy assignment operator for class X performs 
/// memberwise assignment of its subob- jects. The direct base classes of X are 
/// assigned first, in the order of their declaration in 
/// the base-specifier-list, and then the immediate nonstatic data members of X 
/// are assigned, in the order in which they were declared in the class 
/// definition.Each subobject is assigned in the manner appropriate to its type:
///   if the subobject is of class type, the copy assignment operator for the 
///   class is used (as if by explicit qualification; that is, ignoring any 
///   possible virtual overriding functions in more derived classes);
///
///   if the subobject is an array, each element is assigned, in the manner 
///   appropriate to the element type;
///
///   if the subobject is of scalar type, the built-in assignment operator is 
///   used.
void CodeGenFunction::SynthesizeCXXCopyAssignment(const CXXMethodDecl *CD,
                                                  const FunctionDecl *FD,
                                                  llvm::Function *Fn,
                                                  const FunctionArgList &Args) {

  const CXXRecordDecl *ClassDecl = cast<CXXRecordDecl>(CD->getDeclContext());
  assert(!ClassDecl->hasUserDeclaredCopyAssignment() &&
         "SynthesizeCXXCopyAssignment - copy assignment has user declaration");
  StartFunction(FD, FD->getResultType(), Fn, Args, SourceLocation());
  
  FunctionArgList::const_iterator i = Args.begin();
  const VarDecl *ThisArg = i->first;
  llvm::Value *ThisObj = GetAddrOfLocalVar(ThisArg);
  llvm::Value *LoadOfThis = Builder.CreateLoad(ThisObj, "this");
  const VarDecl *SrcArg = (i+1)->first;
  llvm::Value *SrcObj = GetAddrOfLocalVar(SrcArg);
  llvm::Value *LoadOfSrc = Builder.CreateLoad(SrcObj);
  
  for (CXXRecordDecl::base_class_const_iterator Base = ClassDecl->bases_begin();
       Base != ClassDecl->bases_end(); ++Base) {
    // FIXME. copy assignment of virtual base NYI
    if (Base->isVirtual())
      continue;
    
    CXXRecordDecl *BaseClassDecl
      = cast<CXXRecordDecl>(Base->getType()->getAs<RecordType>()->getDecl());
    EmitClassCopyAssignment(LoadOfThis, LoadOfSrc, ClassDecl, BaseClassDecl,
                            Base->getType());
  }
  
  for (CXXRecordDecl::field_iterator Field = ClassDecl->field_begin(),
       FieldEnd = ClassDecl->field_end();
       Field != FieldEnd; ++Field) {
    QualType FieldType = getContext().getCanonicalType((*Field)->getType());
    
    // FIXME. How about copy assignment of  arrays!
    assert(!getContext().getAsArrayType(FieldType) &&
           "FIXME. Copy assignment of arrays NYI");
    
    if (const RecordType *FieldClassType = FieldType->getAs<RecordType>()) {
      CXXRecordDecl *FieldClassDecl
      = cast<CXXRecordDecl>(FieldClassType->getDecl());
      LValue LHS = EmitLValueForField(LoadOfThis, *Field, false, 0);
      LValue RHS = EmitLValueForField(LoadOfSrc, *Field, false, 0);
      
      EmitClassCopyAssignment(LHS.getAddress(), RHS.getAddress(), 
                              0 /*ClassDecl*/, FieldClassDecl, FieldType);
      continue;
    }
    // Do a built-in assignment of scalar data members.
    LValue LHS = EmitLValueForField(LoadOfThis, *Field, false, 0);
    LValue RHS = EmitLValueForField(LoadOfSrc, *Field, false, 0);
    RValue RVRHS = EmitLoadOfLValue(RHS, FieldType);
    EmitStoreThroughLValue(RVRHS, LHS, FieldType);
  }
  
  // return *this;
  Builder.CreateStore(LoadOfThis, ReturnValue);
  
  FinishFunction();
}  

/// EmitCtorPrologue - This routine generates necessary code to initialize
/// base classes and non-static data members belonging to this constructor.
void CodeGenFunction::EmitCtorPrologue(const CXXConstructorDecl *CD) {
  const CXXRecordDecl *ClassDecl = cast<CXXRecordDecl>(CD->getDeclContext());
  // FIXME: Add vbase initialization
  llvm::Value *LoadOfThis = 0;
  
  for (CXXConstructorDecl::init_const_iterator B = CD->init_begin(),
       E = CD->init_end();
       B != E; ++B) {
    CXXBaseOrMemberInitializer *Member = (*B);
    if (Member->isBaseInitializer()) {
      LoadOfThis = LoadCXXThis();
      Type *BaseType = Member->getBaseClass();
      CXXRecordDecl *BaseClassDecl = 
        cast<CXXRecordDecl>(BaseType->getAs<RecordType>()->getDecl());
      llvm::Value *V = AddressCXXOfBaseClass(LoadOfThis, ClassDecl, 
                                             BaseClassDecl);
      EmitCXXConstructorCall(Member->getConstructor(),
                             Ctor_Complete, V,
                             Member->const_arg_begin(), 
                             Member->const_arg_end());
    } else {
      // non-static data member initilaizers.
      FieldDecl *Field = Member->getMember();
      QualType FieldType = getContext().getCanonicalType((Field)->getType());
      const ConstantArrayType *Array = 
        getContext().getAsConstantArrayType(FieldType);
      if (Array)
        FieldType = getContext().getBaseElementType(FieldType);
      
      LoadOfThis = LoadCXXThis();
      LValue LHS = EmitLValueForField(LoadOfThis, Field, false, 0);
      if (FieldType->getAs<RecordType>()) {
        if (!Field->isAnonymousStructOrUnion()) {
          assert(Member->getConstructor() && 
                 "EmitCtorPrologue - no constructor to initialize member");
          if (Array) {
            const llvm::Type *BasePtr = ConvertType(FieldType);
            BasePtr = llvm::PointerType::getUnqual(BasePtr);
            llvm::Value *BaseAddrPtr = 
            Builder.CreateBitCast(LHS.getAddress(), BasePtr);
            EmitCXXAggrConstructorCall(Member->getConstructor(), 
                                       Array, BaseAddrPtr);
          }
          else
            EmitCXXConstructorCall(Member->getConstructor(),
                                   Ctor_Complete, LHS.getAddress(),
                                   Member->const_arg_begin(), 
                                   Member->const_arg_end());
          continue;
        }
        else {
          // Initializing an anonymous union data member.
          FieldDecl *anonMember = Member->getAnonUnionMember();
          LHS = EmitLValueForField(LHS.getAddress(), anonMember, false, 0);
          FieldType = anonMember->getType();
        }
      }
      
      assert(Member->getNumArgs() == 1 && "Initializer count must be 1 only");
      Expr *RhsExpr = *Member->arg_begin();
      llvm::Value *RHS = EmitScalarExpr(RhsExpr, true);
      EmitStoreThroughLValue(RValue::get(RHS), LHS, FieldType);
    }
  }

  if (!CD->getNumBaseOrMemberInitializers() && !CD->isTrivial()) {
    // Nontrivial default constructor with no initializer list. It may still
    // have bases classes and/or contain non-static data members which require 
    // construction.
    for (CXXRecordDecl::base_class_const_iterator Base = 
          ClassDecl->bases_begin();
          Base != ClassDecl->bases_end(); ++Base) {
      // FIXME. copy assignment of virtual base NYI
      if (Base->isVirtual())
        continue;
    
      CXXRecordDecl *BaseClassDecl
        = cast<CXXRecordDecl>(Base->getType()->getAs<RecordType>()->getDecl());
      if (BaseClassDecl->hasTrivialConstructor())
        continue;
      if (CXXConstructorDecl *BaseCX = 
            BaseClassDecl->getDefaultConstructor(getContext())) {
        LoadOfThis = LoadCXXThis();
        llvm::Value *V = AddressCXXOfBaseClass(LoadOfThis, ClassDecl,
                                               BaseClassDecl);
        EmitCXXConstructorCall(BaseCX, Ctor_Complete, V, 0, 0);
      }
    }
  
    for (CXXRecordDecl::field_iterator Field = ClassDecl->field_begin(),
         FieldEnd = ClassDecl->field_end();
         Field != FieldEnd; ++Field) {
      QualType FieldType = getContext().getCanonicalType((*Field)->getType());
      const ConstantArrayType *Array = 
        getContext().getAsConstantArrayType(FieldType);
      if (Array)
        FieldType = getContext().getBaseElementType(FieldType);
      if (!FieldType->getAs<RecordType>() || Field->isAnonymousStructOrUnion())
        continue;
      const RecordType *ClassRec = FieldType->getAs<RecordType>();
      CXXRecordDecl *MemberClassDecl = 
        dyn_cast<CXXRecordDecl>(ClassRec->getDecl());
      if (!MemberClassDecl || MemberClassDecl->hasTrivialConstructor())
        continue;
      if (CXXConstructorDecl *MamberCX = 
            MemberClassDecl->getDefaultConstructor(getContext())) {
        LoadOfThis = LoadCXXThis();
        LValue LHS = EmitLValueForField(LoadOfThis, *Field, false, 0);
        if (Array) {
          const llvm::Type *BasePtr = ConvertType(FieldType);
          BasePtr = llvm::PointerType::getUnqual(BasePtr);
          llvm::Value *BaseAddrPtr = 
            Builder.CreateBitCast(LHS.getAddress(), BasePtr);
          EmitCXXAggrConstructorCall(MamberCX, Array, BaseAddrPtr);
        }
        else
          EmitCXXConstructorCall(MamberCX, Ctor_Complete, LHS.getAddress(), 
                                 0, 0);
      }
    }
  }
  
  // Initialize the vtable pointer
  if (ClassDecl->isDynamicClass()) {
    if (!LoadOfThis)
      LoadOfThis = LoadCXXThis();
    llvm::Value *VtableField;
    llvm::Type *Ptr8Ty, *PtrPtr8Ty;
    Ptr8Ty = llvm::PointerType::get(llvm::Type::getInt8Ty(VMContext), 0);
    PtrPtr8Ty = llvm::PointerType::get(Ptr8Ty, 0);
    VtableField = Builder.CreateBitCast(LoadOfThis, PtrPtr8Ty);
    llvm::Value *vtable = GenerateVtable(ClassDecl);
    Builder.CreateStore(vtable, VtableField);
  }
}

/// EmitDtorEpilogue - Emit all code that comes at the end of class's
/// destructor. This is to call destructors on members and base classes 
/// in reverse order of their construction.
void CodeGenFunction::EmitDtorEpilogue(const CXXDestructorDecl *DD) {
  const CXXRecordDecl *ClassDecl = cast<CXXRecordDecl>(DD->getDeclContext());
  assert(!ClassDecl->isPolymorphic() &&
         "FIXME. polymorphic destruction not supported");
  (void)ClassDecl;  // prevent warning.
  
  for (CXXDestructorDecl::destr_const_iterator *B = DD->destr_begin(),
       *E = DD->destr_end(); B != E; ++B) {
    uintptr_t BaseOrMember = (*B);
    if (DD->isMemberToDestroy(BaseOrMember)) {
      FieldDecl *FD = DD->getMemberToDestroy(BaseOrMember);
      QualType FieldType = getContext().getCanonicalType((FD)->getType());
      const ConstantArrayType *Array = 
        getContext().getAsConstantArrayType(FieldType);
      if (Array)
        FieldType = getContext().getBaseElementType(FieldType);
      const RecordType *RT = FieldType->getAs<RecordType>();
      CXXRecordDecl *FieldClassDecl = cast<CXXRecordDecl>(RT->getDecl());
      if (FieldClassDecl->hasTrivialDestructor())
        continue;
      llvm::Value *LoadOfThis = LoadCXXThis();
      LValue LHS = EmitLValueForField(LoadOfThis, FD, false, 0);
      if (Array) {
        const llvm::Type *BasePtr = ConvertType(FieldType);
        BasePtr = llvm::PointerType::getUnqual(BasePtr);
        llvm::Value *BaseAddrPtr = 
          Builder.CreateBitCast(LHS.getAddress(), BasePtr);
        EmitCXXAggrDestructorCall(FieldClassDecl->getDestructor(getContext()), 
                                  Array, BaseAddrPtr);
      }
      else
        EmitCXXDestructorCall(FieldClassDecl->getDestructor(getContext()),
                              Dtor_Complete, LHS.getAddress());
    } else {
      const RecordType *RT =
        DD->getAnyBaseClassToDestroy(BaseOrMember)->getAs<RecordType>();
      CXXRecordDecl *BaseClassDecl = cast<CXXRecordDecl>(RT->getDecl());
      if (BaseClassDecl->hasTrivialDestructor())
        continue;
      llvm::Value *V = AddressCXXOfBaseClass(LoadCXXThis(), 
                                             ClassDecl,BaseClassDecl);
      EmitCXXDestructorCall(BaseClassDecl->getDestructor(getContext()),
                            Dtor_Complete, V);
    }
  }
  if (DD->getNumBaseOrMemberDestructions() || DD->isTrivial())
    return;
  // Case of destructor synthesis with fields and base classes
  // which have non-trivial destructors. They must be destructed in 
  // reverse order of their construction.
  llvm::SmallVector<FieldDecl *, 16> DestructedFields;
  
  for (CXXRecordDecl::field_iterator Field = ClassDecl->field_begin(),
       FieldEnd = ClassDecl->field_end();
       Field != FieldEnd; ++Field) {
    QualType FieldType = getContext().getCanonicalType((*Field)->getType());
    if (getContext().getAsConstantArrayType(FieldType))
      FieldType = getContext().getBaseElementType(FieldType);
    if (const RecordType *RT = FieldType->getAs<RecordType>()) {
      CXXRecordDecl *FieldClassDecl = cast<CXXRecordDecl>(RT->getDecl());
      if (FieldClassDecl->hasTrivialDestructor())
        continue;
      DestructedFields.push_back(*Field);
    }
  }
  if (!DestructedFields.empty())
    for (int i = DestructedFields.size() -1; i >= 0; --i) {
      FieldDecl *Field = DestructedFields[i];
      QualType FieldType = Field->getType();
      const ConstantArrayType *Array = 
        getContext().getAsConstantArrayType(FieldType);
        if (Array)
          FieldType = getContext().getBaseElementType(FieldType);
      const RecordType *RT = FieldType->getAs<RecordType>();
      CXXRecordDecl *FieldClassDecl = cast<CXXRecordDecl>(RT->getDecl());
      llvm::Value *LoadOfThis = LoadCXXThis();
      LValue LHS = EmitLValueForField(LoadOfThis, Field, false, 0);
      if (Array) {
        const llvm::Type *BasePtr = ConvertType(FieldType);
        BasePtr = llvm::PointerType::getUnqual(BasePtr);
        llvm::Value *BaseAddrPtr = 
        Builder.CreateBitCast(LHS.getAddress(), BasePtr);
        EmitCXXAggrDestructorCall(FieldClassDecl->getDestructor(getContext()), 
                                  Array, BaseAddrPtr);
      }
      else
        EmitCXXDestructorCall(FieldClassDecl->getDestructor(getContext()),
                              Dtor_Complete, LHS.getAddress());
    }
  
  llvm::SmallVector<CXXRecordDecl*, 4> DestructedBases;
  for (CXXRecordDecl::base_class_const_iterator Base = ClassDecl->bases_begin();
       Base != ClassDecl->bases_end(); ++Base) {
    // FIXME. copy assignment of virtual base NYI
    if (Base->isVirtual())
      continue;
    
    CXXRecordDecl *BaseClassDecl
      = cast<CXXRecordDecl>(Base->getType()->getAs<RecordType>()->getDecl());
    if (BaseClassDecl->hasTrivialDestructor())
      continue;
    DestructedBases.push_back(BaseClassDecl);
  }
  if (DestructedBases.empty())
    return;
  for (int i = DestructedBases.size() -1; i >= 0; --i) {
    CXXRecordDecl *BaseClassDecl = DestructedBases[i];
    llvm::Value *V = AddressCXXOfBaseClass(LoadCXXThis(), 
                                           ClassDecl,BaseClassDecl);
    EmitCXXDestructorCall(BaseClassDecl->getDestructor(getContext()),
                          Dtor_Complete, V);
  }
}

void CodeGenFunction::SynthesizeDefaultDestructor(const CXXDestructorDecl *CD,
                                                  const FunctionDecl *FD,
                                                  llvm::Function *Fn,
                                                  const FunctionArgList &Args) {
  
  const CXXRecordDecl *ClassDecl = cast<CXXRecordDecl>(CD->getDeclContext());
  assert(!ClassDecl->hasUserDeclaredDestructor() &&
         "SynthesizeDefaultDestructor - destructor has user declaration");
  (void) ClassDecl;
  
  StartFunction(FD, FD->getResultType(), Fn, Args, SourceLocation());
  EmitDtorEpilogue(CD);
  FinishFunction();
}  
