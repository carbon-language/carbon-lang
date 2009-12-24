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
#include "clang/AST/StmtCXX.h"
#include "llvm/ADT/StringExtras.h"
using namespace clang;
using namespace CodeGen;

RValue CodeGenFunction::EmitCXXMemberCall(const CXXMethodDecl *MD,
                                          llvm::Value *Callee,
                                          ReturnValueSlot ReturnValue,
                                          llvm::Value *This,
                                          CallExpr::const_arg_iterator ArgBeg,
                                          CallExpr::const_arg_iterator ArgEnd) {
  assert(MD->isInstance() &&
         "Trying to emit a member call expr on a static method!");

  const FunctionProtoType *FPT = MD->getType()->getAs<FunctionProtoType>();

  CallArgList Args;

  // Push the this ptr.
  Args.push_back(std::make_pair(RValue::get(This),
                                MD->getThisType(getContext())));

  // And the rest of the call args
  EmitCallArgs(Args, FPT, ArgBeg, ArgEnd);

  QualType ResultType = MD->getType()->getAs<FunctionType>()->getResultType();
  return EmitCall(CGM.getTypes().getFunctionInfo(ResultType, Args), Callee, 
                  ReturnValue, Args, MD);
}

/// canDevirtualizeMemberFunctionCalls - Checks whether virtual calls on given
/// expr can be devirtualized.
static bool canDevirtualizeMemberFunctionCalls(const Expr *Base) {
  if (const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(Base)) {
    if (const VarDecl *VD = dyn_cast<VarDecl>(DRE->getDecl())) {
      // This is a record decl. We know the type and can devirtualize it.
      return VD->getType()->isRecordType();
    }
    
    return false;
  }
  
  // We can always devirtualize calls on temporary object expressions.
  if (isa<CXXTemporaryObjectExpr>(Base))
    return true;
  
  // And calls on bound temporaries.
  if (isa<CXXBindTemporaryExpr>(Base))
    return true;
  
  // Check if this is a call expr that returns a record type.
  if (const CallExpr *CE = dyn_cast<CallExpr>(Base))
    return CE->getCallReturnType()->isRecordType();
  
  // We can't devirtualize the call.
  return false;
}

RValue CodeGenFunction::EmitCXXMemberCallExpr(const CXXMemberCallExpr *CE,
                                              ReturnValueSlot ReturnValue) {
  if (isa<BinaryOperator>(CE->getCallee()->IgnoreParens())) 
    return EmitCXXMemberPointerCallExpr(CE, ReturnValue);
      
  const MemberExpr *ME = cast<MemberExpr>(CE->getCallee()->IgnoreParens());
  const CXXMethodDecl *MD = cast<CXXMethodDecl>(ME->getMemberDecl());

  if (MD->isStatic()) {
    // The method is static, emit it as we would a regular call.
    llvm::Value *Callee = CGM.GetAddrOfFunction(MD);
    return EmitCall(getContext().getPointerType(MD->getType()), Callee,
                    ReturnValue, CE->arg_begin(), CE->arg_end());
  }
  
  const FunctionProtoType *FPT = MD->getType()->getAs<FunctionProtoType>();

  const llvm::Type *Ty =
    CGM.getTypes().GetFunctionType(CGM.getTypes().getFunctionInfo(MD),
                                   FPT->isVariadic());
  llvm::Value *This;

  if (ME->isArrow())
    This = EmitScalarExpr(ME->getBase());
  else {
    LValue BaseLV = EmitLValue(ME->getBase());
    This = BaseLV.getAddress();
  }

  if (MD->isCopyAssignment() && MD->isTrivial()) {
    // We don't like to generate the trivial copy assignment operator when
    // it isn't necessary; just produce the proper effect here.
    llvm::Value *RHS = EmitLValue(*CE->arg_begin()).getAddress();
    EmitAggregateCopy(This, RHS, CE->getType());
    return RValue::get(This);
  }

  // C++ [class.virtual]p12:
  //   Explicit qualification with the scope operator (5.1) suppresses the
  //   virtual call mechanism.
  //
  // We also don't emit a virtual call if the base expression has a record type
  // because then we know what the type is.
  llvm::Value *Callee;
  if (const CXXDestructorDecl *Destructor
             = dyn_cast<CXXDestructorDecl>(MD)) {
    if (Destructor->isTrivial())
      return RValue::get(0);
    if (MD->isVirtual() && !ME->hasQualifier() && 
        !canDevirtualizeMemberFunctionCalls(ME->getBase())) {
      Callee = BuildVirtualCall(Destructor, Dtor_Complete, This, Ty); 
    } else {
      Callee = CGM.GetAddrOfFunction(GlobalDecl(Destructor, Dtor_Complete), Ty);
    }
  } else if (MD->isVirtual() && !ME->hasQualifier() && 
             !canDevirtualizeMemberFunctionCalls(ME->getBase())) {
    Callee = BuildVirtualCall(MD, This, Ty); 
  } else {
    Callee = CGM.GetAddrOfFunction(MD, Ty);
  }

  return EmitCXXMemberCall(MD, Callee, ReturnValue, This,
                           CE->arg_begin(), CE->arg_end());
}

RValue
CodeGenFunction::EmitCXXMemberPointerCallExpr(const CXXMemberCallExpr *E,
                                              ReturnValueSlot ReturnValue) {
  const BinaryOperator *BO =
      cast<BinaryOperator>(E->getCallee()->IgnoreParens());
  const Expr *BaseExpr = BO->getLHS();
  const Expr *MemFnExpr = BO->getRHS();
  
  const MemberPointerType *MPT = 
    MemFnExpr->getType()->getAs<MemberPointerType>();
  const FunctionProtoType *FPT = 
    MPT->getPointeeType()->getAs<FunctionProtoType>();
  const CXXRecordDecl *RD = 
    cast<CXXRecordDecl>(MPT->getClass()->getAs<RecordType>()->getDecl());

  const llvm::FunctionType *FTy = 
    CGM.getTypes().GetFunctionType(CGM.getTypes().getFunctionInfo(RD, FPT),
                                   FPT->isVariadic());

  const llvm::Type *Int8PtrTy = 
    llvm::Type::getInt8Ty(VMContext)->getPointerTo();

  // Get the member function pointer.
  llvm::Value *MemFnPtr = 
    CreateTempAlloca(ConvertType(MemFnExpr->getType()), "mem.fn");
  EmitAggExpr(MemFnExpr, MemFnPtr, /*VolatileDest=*/false);

  // Emit the 'this' pointer.
  llvm::Value *This;
  
  if (BO->getOpcode() == BinaryOperator::PtrMemI)
    This = EmitScalarExpr(BaseExpr);
  else 
    This = EmitLValue(BaseExpr).getAddress();
  
  // Adjust it.
  llvm::Value *Adj = Builder.CreateStructGEP(MemFnPtr, 1);
  Adj = Builder.CreateLoad(Adj, "mem.fn.adj");
  
  llvm::Value *Ptr = Builder.CreateBitCast(This, Int8PtrTy, "ptr");
  Ptr = Builder.CreateGEP(Ptr, Adj, "adj");
  
  This = Builder.CreateBitCast(Ptr, This->getType(), "this");
  
  llvm::Value *FnPtr = Builder.CreateStructGEP(MemFnPtr, 0, "mem.fn.ptr");
  
  const llvm::Type *PtrDiffTy = ConvertType(getContext().getPointerDiffType());

  llvm::Value *FnAsInt = Builder.CreateLoad(FnPtr, "fn");
  
  // If the LSB in the function pointer is 1, the function pointer points to
  // a virtual function.
  llvm::Value *IsVirtual 
    = Builder.CreateAnd(FnAsInt, llvm::ConstantInt::get(PtrDiffTy, 1),
                        "and");
  
  IsVirtual = Builder.CreateTrunc(IsVirtual,
                                  llvm::Type::getInt1Ty(VMContext));
  
  llvm::BasicBlock *FnVirtual = createBasicBlock("fn.virtual");
  llvm::BasicBlock *FnNonVirtual = createBasicBlock("fn.nonvirtual");
  llvm::BasicBlock *FnEnd = createBasicBlock("fn.end");
  
  Builder.CreateCondBr(IsVirtual, FnVirtual, FnNonVirtual);
  EmitBlock(FnVirtual);
  
  const llvm::Type *VTableTy = 
    FTy->getPointerTo()->getPointerTo()->getPointerTo();

  llvm::Value *VTable = Builder.CreateBitCast(This, VTableTy);
  VTable = Builder.CreateLoad(VTable);
  
  VTable = Builder.CreateGEP(VTable, FnAsInt, "fn");
  
  // Since the function pointer is 1 plus the virtual table offset, we
  // subtract 1 by using a GEP.
  VTable = Builder.CreateConstGEP1_64(VTable, (uint64_t)-1);
  
  llvm::Value *VirtualFn = Builder.CreateLoad(VTable, "virtualfn");
  
  EmitBranch(FnEnd);
  EmitBlock(FnNonVirtual);
  
  // If the function is not virtual, just load the pointer.
  llvm::Value *NonVirtualFn = Builder.CreateLoad(FnPtr, "fn");
  NonVirtualFn = Builder.CreateIntToPtr(NonVirtualFn, FTy->getPointerTo());
  
  EmitBlock(FnEnd);

  llvm::PHINode *Callee = Builder.CreatePHI(FTy->getPointerTo());
  Callee->reserveOperandSpace(2);
  Callee->addIncoming(VirtualFn, FnVirtual);
  Callee->addIncoming(NonVirtualFn, FnNonVirtual);

  CallArgList Args;

  QualType ThisType = 
    getContext().getPointerType(getContext().getTagDeclType(RD));

  // Push the this ptr.
  Args.push_back(std::make_pair(RValue::get(This), ThisType));
  
  // And the rest of the call args
  EmitCallArgs(Args, FPT, E->arg_begin(), E->arg_end());
  QualType ResultType = BO->getType()->getAs<FunctionType>()->getResultType();
  return EmitCall(CGM.getTypes().getFunctionInfo(ResultType, Args), Callee, 
                  ReturnValue, Args);
}

RValue
CodeGenFunction::EmitCXXOperatorMemberCallExpr(const CXXOperatorCallExpr *E,
                                               const CXXMethodDecl *MD,
                                               ReturnValueSlot ReturnValue) {
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

  const FunctionProtoType *FPT = MD->getType()->getAs<FunctionProtoType>();
  const llvm::Type *Ty =
    CGM.getTypes().GetFunctionType(CGM.getTypes().getFunctionInfo(MD),
                                   FPT->isVariadic());

  llvm::Value *This = EmitLValue(E->getArg(0)).getAddress();

  llvm::Value *Callee;
  if (MD->isVirtual() && !canDevirtualizeMemberFunctionCalls(E->getArg(0)))
    Callee = BuildVirtualCall(MD, This, Ty);
  else
    Callee = CGM.GetAddrOfFunction(MD, Ty);

  return EmitCXXMemberCall(MD, Callee, ReturnValue, This, 
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

/// EmitCXXAggrConstructorCall - This routine essentially creates a (nested)
/// for-loop to call the default constructor on individual members of the
/// array. 
/// 'D' is the default constructor for elements of the array, 'ArrayTy' is the
/// array type and 'ArrayPtr' points to the beginning fo the array.
/// It is assumed that all relevant checks have been made by the caller.
void
CodeGenFunction::EmitCXXAggrConstructorCall(const CXXConstructorDecl *D,
                                          const ConstantArrayType *ArrayTy,
                                          llvm::Value *ArrayPtr,
                                          CallExpr::const_arg_iterator ArgBeg,
                                          CallExpr::const_arg_iterator ArgEnd) {

  const llvm::Type *SizeTy = ConvertType(getContext().getSizeType());
  llvm::Value * NumElements =
    llvm::ConstantInt::get(SizeTy, 
                           getContext().getConstantArrayElementCount(ArrayTy));

  EmitCXXAggrConstructorCall(D, NumElements, ArrayPtr, ArgBeg, ArgEnd);
}

void
CodeGenFunction::EmitCXXAggrConstructorCall(const CXXConstructorDecl *D,
                                          llvm::Value *NumElements,
                                          llvm::Value *ArrayPtr,
                                          CallExpr::const_arg_iterator ArgBeg,
                                          CallExpr::const_arg_iterator ArgEnd) {
  const llvm::Type *SizeTy = ConvertType(getContext().getSizeType());

  // Create a temporary for the loop index and initialize it with 0.
  llvm::Value *IndexPtr = CreateTempAlloca(SizeTy, "loop.index");
  llvm::Value *Zero = llvm::Constant::getNullValue(SizeTy);
  Builder.CreateStore(Zero, IndexPtr);

  // Start the loop with a block that tests the condition.
  llvm::BasicBlock *CondBlock = createBasicBlock("for.cond");
  llvm::BasicBlock *AfterFor = createBasicBlock("for.end");

  EmitBlock(CondBlock);

  llvm::BasicBlock *ForBody = createBasicBlock("for.body");

  // Generate: if (loop-index < number-of-elements fall to the loop body,
  // otherwise, go to the block after the for-loop.
  llvm::Value *Counter = Builder.CreateLoad(IndexPtr);
  llvm::Value *IsLess = Builder.CreateICmpULT(Counter, NumElements, "isless");
  // If the condition is true, execute the body.
  Builder.CreateCondBr(IsLess, ForBody, AfterFor);

  EmitBlock(ForBody);

  llvm::BasicBlock *ContinueBlock = createBasicBlock("for.inc");
  // Inside the loop body, emit the constructor call on the array element.
  Counter = Builder.CreateLoad(IndexPtr);
  llvm::Value *Address = Builder.CreateInBoundsGEP(ArrayPtr, Counter, 
                                                   "arrayidx");

  // C++ [class.temporary]p4: 
  // There are two contexts in which temporaries are destroyed at a different
  // point than the end of the full-expression. The first context is when a
  // default constructor is called to initialize an element of an array. 
  // If the constructor has one or more default arguments, the destruction of 
  // every temporary created in a default argument expression is sequenced 
  // before the construction of the next array element, if any.
  
  // Keep track of the current number of live temporaries.
  unsigned OldNumLiveTemporaries = LiveTemporaries.size();

  EmitCXXConstructorCall(D, Ctor_Complete, Address, ArgBeg, ArgEnd);

  // Pop temporaries.
  while (LiveTemporaries.size() > OldNumLiveTemporaries)
    PopCXXTemporary();
  
  EmitBlock(ContinueBlock);

  // Emit the increment of the loop counter.
  llvm::Value *NextVal = llvm::ConstantInt::get(SizeTy, 1);
  Counter = Builder.CreateLoad(IndexPtr);
  NextVal = Builder.CreateAdd(Counter, NextVal, "inc");
  Builder.CreateStore(NextVal, IndexPtr);

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
  uint64_t ElementCount = getContext().getConstantArrayElementCount(CA);
  
  const llvm::Type *SizeLTy = ConvertType(getContext().getSizeType());
  llvm::Value* ElementCountPtr = llvm::ConstantInt::get(SizeLTy, ElementCount);
  EmitCXXAggrDestructorCall(D, ElementCountPtr, This);
}

/// EmitCXXAggrDestructorCall - calls the default destructor on array
/// elements in reverse order of construction.
void
CodeGenFunction::EmitCXXAggrDestructorCall(const CXXDestructorDecl *D,
                                           llvm::Value *UpperCount,
                                           llvm::Value *This) {
  const llvm::Type *SizeLTy = ConvertType(getContext().getSizeType());
  llvm::Value *One = llvm::ConstantInt::get(SizeLTy, 1);
  
  // Create a temporary for the loop index and initialize it with count of
  // array elements.
  llvm::Value *IndexPtr = CreateTempAlloca(SizeLTy, "loop.index");

  // Store the number of elements in the index pointer.
  Builder.CreateStore(UpperCount, IndexPtr);

  // Start the loop with a block that tests the condition.
  llvm::BasicBlock *CondBlock = createBasicBlock("for.cond");
  llvm::BasicBlock *AfterFor = createBasicBlock("for.end");

  EmitBlock(CondBlock);

  llvm::BasicBlock *ForBody = createBasicBlock("for.body");

  // Generate: if (loop-index != 0 fall to the loop body,
  // otherwise, go to the block after the for-loop.
  llvm::Value* zeroConstant =
    llvm::Constant::getNullValue(SizeLTy);
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
  Builder.CreateStore(Counter, IndexPtr);

  // Finally, branch back up to the condition for the next iteration.
  EmitBranch(CondBlock);

  // Emit the fall-through block.
  EmitBlock(AfterFor, true);
}

/// GenerateCXXAggrDestructorHelper - Generates a helper function which when
/// invoked, calls the default destructor on array elements in reverse order of
/// construction.
llvm::Constant * 
CodeGenFunction::GenerateCXXAggrDestructorHelper(const CXXDestructorDecl *D,
                                                 const ArrayType *Array,
                                                 llvm::Value *This) {
  FunctionArgList Args;
  ImplicitParamDecl *Dst =
    ImplicitParamDecl::Create(getContext(), 0,
                              SourceLocation(), 0,
                              getContext().getPointerType(getContext().VoidTy));
  Args.push_back(std::make_pair(Dst, Dst->getType()));
  
  llvm::SmallString<16> Name;
  llvm::raw_svector_ostream(Name) << "__tcf_" << (++UniqueAggrDestructorCount);
  QualType R = getContext().VoidTy;
  const CGFunctionInfo &FI = CGM.getTypes().getFunctionInfo(R, Args);
  const llvm::FunctionType *FTy = CGM.getTypes().GetFunctionType(FI, false);
  llvm::Function *Fn =
    llvm::Function::Create(FTy, llvm::GlobalValue::InternalLinkage,
                           Name.str(),
                           &CGM.getModule());
  IdentifierInfo *II = &CGM.getContext().Idents.get(Name.str());
  FunctionDecl *FD = FunctionDecl::Create(getContext(),
                                          getContext().getTranslationUnitDecl(),
                                          SourceLocation(), II, R, 0,
                                          FunctionDecl::Static,
                                          false, true);
  StartFunction(FD, R, Fn, Args, SourceLocation());
  QualType BaseElementTy = getContext().getBaseElementType(Array);
  const llvm::Type *BasePtr = ConvertType(BaseElementTy);
  BasePtr = llvm::PointerType::getUnqual(BasePtr);
  llvm::Value *BaseAddrPtr = Builder.CreateBitCast(This, BasePtr);
  EmitCXXAggrDestructorCall(D, Array, BaseAddrPtr);
  FinishFunction();
  llvm::Type *Ptr8Ty = llvm::PointerType::get(llvm::Type::getInt8Ty(VMContext),
                                              0);
  llvm::Constant *m = llvm::ConstantExpr::getBitCast(Fn, Ptr8Ty);
  return m;
}

void
CodeGenFunction::EmitCXXConstructorCall(const CXXConstructorDecl *D,
                                        CXXCtorType Type,
                                        llvm::Value *This,
                                        CallExpr::const_arg_iterator ArgBeg,
                                        CallExpr::const_arg_iterator ArgEnd) {
  if (D->isCopyConstructor()) {
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
  } else if (D->isTrivial()) {
    // FIXME: Track down why we're trying to generate calls to the trivial
    // default constructor!
    return;
  }

  llvm::Value *Callee = CGM.GetAddrOfCXXConstructor(D, Type);

  EmitCXXMemberCall(D, Callee, ReturnValueSlot(), This, ArgBeg, ArgEnd);
}

void CodeGenFunction::EmitCXXDestructorCall(const CXXDestructorDecl *DD,
                                            CXXDtorType Type,
                                            llvm::Value *This) {
  llvm::Value *Callee = CGM.GetAddrOfCXXDestructor(DD, Type);
  
  CallArgList Args;

  // Push the this ptr.
  Args.push_back(std::make_pair(RValue::get(This),
                                DD->getThisType(getContext())));
  
  // Add a VTT parameter if necessary.
  // FIXME: This should not be a dummy null parameter!
  if (Type == Dtor_Base && DD->getParent()->getNumVBases() != 0) {
    QualType T = getContext().getPointerType(getContext().VoidPtrTy);
    
    Args.push_back(std::make_pair(RValue::get(CGM.EmitNullConstant(T)), T));
  }

  // FIXME: We should try to share this code with EmitCXXMemberCall.
  
  QualType ResultType = DD->getType()->getAs<FunctionType>()->getResultType();
  EmitCall(CGM.getTypes().getFunctionInfo(ResultType, Args), Callee, 
           ReturnValueSlot(), Args, DD);
}

void
CodeGenFunction::EmitCXXConstructExpr(llvm::Value *Dest,
                                      const CXXConstructExpr *E) {
  assert(Dest && "Must have a destination!");
  const CXXConstructorDecl *CD = E->getConstructor();
  const ConstantArrayType *Array =
    getContext().getAsConstantArrayType(E->getType());
  // For a copy constructor, even if it is trivial, must fall thru so
  // its argument is code-gen'ed.
  if (!CD->isCopyConstructor()) {
    QualType InitType = E->getType();
    if (Array)
      InitType = getContext().getBaseElementType(Array);
    const CXXRecordDecl *RD =
      cast<CXXRecordDecl>(InitType->getAs<RecordType>()->getDecl());
    if (RD->hasTrivialConstructor())
      return;
  }
  // Code gen optimization to eliminate copy constructor and return
  // its first argument instead.
  if (getContext().getLangOptions().ElideConstructors && E->isElidable()) {
    const Expr *Arg = E->getArg(0);

    if (const ImplicitCastExpr *ICE = dyn_cast<ImplicitCastExpr>(Arg)) {
      assert((ICE->getCastKind() == CastExpr::CK_NoOp ||
              ICE->getCastKind() == CastExpr::CK_ConstructorConversion ||
              ICE->getCastKind() == CastExpr::CK_UserDefinedConversion) &&
             "Unknown implicit cast kind in constructor elision");
      Arg = ICE->getSubExpr();
    }

    if (const CXXFunctionalCastExpr *FCE = dyn_cast<CXXFunctionalCastExpr>(Arg))
      Arg = FCE->getSubExpr();

    if (const CXXBindTemporaryExpr *BindExpr = 
           dyn_cast<CXXBindTemporaryExpr>(Arg))
      Arg = BindExpr->getSubExpr();

    EmitAggExpr(Arg, Dest, false);
    return;
  }
  if (Array) {
    QualType BaseElementTy = getContext().getBaseElementType(Array);
    const llvm::Type *BasePtr = ConvertType(BaseElementTy);
    BasePtr = llvm::PointerType::getUnqual(BasePtr);
    llvm::Value *BaseAddrPtr =
      Builder.CreateBitCast(Dest, BasePtr);

    EmitCXXAggrConstructorCall(CD, Array, BaseAddrPtr, 
                               E->arg_begin(), E->arg_end());
  }
  else
    // Call the constructor.
    EmitCXXConstructorCall(CD, Ctor_Complete, Dest,
                           E->arg_begin(), E->arg_end());
}

void CodeGenModule::EmitCXXConstructors(const CXXConstructorDecl *D) {
  EmitGlobal(GlobalDecl(D, Ctor_Complete));
  EmitGlobal(GlobalDecl(D, Ctor_Base));
}

void CodeGenModule::EmitCXXConstructor(const CXXConstructorDecl *D,
                                       CXXCtorType Type) {

  llvm::Function *Fn = GetAddrOfCXXConstructor(D, Type);

  CodeGenFunction(*this).GenerateCode(GlobalDecl(D, Type), Fn);

  SetFunctionDefinitionAttributes(D, Fn);
  SetLLVMFunctionAttributesForDefinition(D, Fn);
}

llvm::Function *
CodeGenModule::GetAddrOfCXXConstructor(const CXXConstructorDecl *D,
                                       CXXCtorType Type) {
  const FunctionProtoType *FPT = D->getType()->getAs<FunctionProtoType>();
  const llvm::FunctionType *FTy =
    getTypes().GetFunctionType(getTypes().getFunctionInfo(D, Type), 
                               FPT->isVariadic());

  const char *Name = getMangledCXXCtorName(D, Type);
  return cast<llvm::Function>(
                      GetOrCreateLLVMFunction(Name, FTy, GlobalDecl(D, Type)));
}

const char *CodeGenModule::getMangledCXXCtorName(const CXXConstructorDecl *D,
                                                 CXXCtorType Type) {
  llvm::SmallString<256> Name;
  getMangleContext().mangleCXXCtor(D, Type, Name);

  Name += '\0';
  return UniqueMangledName(Name.begin(), Name.end());
}

void CodeGenModule::EmitCXXDestructors(const CXXDestructorDecl *D) {
  if (D->isVirtual())
    EmitGlobal(GlobalDecl(D, Dtor_Deleting));
  EmitGlobal(GlobalDecl(D, Dtor_Complete));
  EmitGlobal(GlobalDecl(D, Dtor_Base));
}

void CodeGenModule::EmitCXXDestructor(const CXXDestructorDecl *D,
                                      CXXDtorType Type) {
  llvm::Function *Fn = GetAddrOfCXXDestructor(D, Type);

  CodeGenFunction(*this).GenerateCode(GlobalDecl(D, Type), Fn);

  SetFunctionDefinitionAttributes(D, Fn);
  SetLLVMFunctionAttributesForDefinition(D, Fn);
}

llvm::Function *
CodeGenModule::GetAddrOfCXXDestructor(const CXXDestructorDecl *D,
                                      CXXDtorType Type) {
  const llvm::FunctionType *FTy =
    getTypes().GetFunctionType(getTypes().getFunctionInfo(D, Type), false);

  const char *Name = getMangledCXXDtorName(D, Type);
  return cast<llvm::Function>(
                      GetOrCreateLLVMFunction(Name, FTy, GlobalDecl(D, Type)));
}

const char *CodeGenModule::getMangledCXXDtorName(const CXXDestructorDecl *D,
                                                 CXXDtorType Type) {
  llvm::SmallString<256> Name;
  getMangleContext().mangleCXXDtor(D, Type, Name);

  Name += '\0';
  return UniqueMangledName(Name.begin(), Name.end());
}

llvm::Constant *
CodeGenFunction::GenerateThunk(llvm::Function *Fn, GlobalDecl GD,
                               bool Extern, 
                               const ThunkAdjustment &ThisAdjustment) {
  return GenerateCovariantThunk(Fn, GD, Extern,
                                CovariantThunkAdjustment(ThisAdjustment,
                                                         ThunkAdjustment()));
}

llvm::Value *
CodeGenFunction::DynamicTypeAdjust(llvm::Value *V, 
                                   const ThunkAdjustment &Adjustment) {
  const llvm::Type *Int8PtrTy = llvm::Type::getInt8PtrTy(VMContext);

  const llvm::Type *OrigTy = V->getType();
  if (Adjustment.NonVirtual) {
    // Do the non-virtual adjustment
    V = Builder.CreateBitCast(V, Int8PtrTy);
    V = Builder.CreateConstInBoundsGEP1_64(V, Adjustment.NonVirtual);
    V = Builder.CreateBitCast(V, OrigTy);
  }
  
  if (!Adjustment.Virtual)
    return V;

  assert(Adjustment.Virtual % (LLVMPointerWidth / 8) == 0 && 
         "vtable entry unaligned");

  // Do the virtual this adjustment
  const llvm::Type *PtrDiffTy = ConvertType(getContext().getPointerDiffType());
  const llvm::Type *PtrDiffPtrTy = PtrDiffTy->getPointerTo();
  
  llvm::Value *ThisVal = Builder.CreateBitCast(V, Int8PtrTy);
  V = Builder.CreateBitCast(V, PtrDiffPtrTy->getPointerTo());
  V = Builder.CreateLoad(V, "vtable");
  
  llvm::Value *VTablePtr = V;
  uint64_t VirtualAdjustment = Adjustment.Virtual / (LLVMPointerWidth / 8);
  V = Builder.CreateConstInBoundsGEP1_64(VTablePtr, VirtualAdjustment);
  V = Builder.CreateLoad(V);
  V = Builder.CreateGEP(ThisVal, V);
  
  return Builder.CreateBitCast(V, OrigTy);
}

llvm::Constant *
CodeGenFunction::GenerateCovariantThunk(llvm::Function *Fn,
                                   GlobalDecl GD, bool Extern,
                                   const CovariantThunkAdjustment &Adjustment) {
  const CXXMethodDecl *MD = cast<CXXMethodDecl>(GD.getDecl());
  QualType ResultType = MD->getType()->getAs<FunctionType>()->getResultType();

  FunctionArgList Args;
  ImplicitParamDecl *ThisDecl =
    ImplicitParamDecl::Create(getContext(), 0, SourceLocation(), 0,
                              MD->getThisType(getContext()));
  Args.push_back(std::make_pair(ThisDecl, ThisDecl->getType()));
  for (FunctionDecl::param_const_iterator i = MD->param_begin(),
         e = MD->param_end();
       i != e; ++i) {
    ParmVarDecl *D = *i;
    Args.push_back(std::make_pair(D, D->getType()));
  }
  IdentifierInfo *II
    = &CGM.getContext().Idents.get("__thunk_named_foo_");
  FunctionDecl *FD = FunctionDecl::Create(getContext(),
                                          getContext().getTranslationUnitDecl(),
                                          SourceLocation(), II, ResultType, 0,
                                          Extern
                                            ? FunctionDecl::Extern
                                            : FunctionDecl::Static,
                                          false, true);
  StartFunction(FD, ResultType, Fn, Args, SourceLocation());

  // generate body
  const FunctionProtoType *FPT = MD->getType()->getAs<FunctionProtoType>();
  const llvm::Type *Ty =
    CGM.getTypes().GetFunctionType(CGM.getTypes().getFunctionInfo(MD),
                                   FPT->isVariadic());
  llvm::Value *Callee = CGM.GetAddrOfFunction(GD, Ty);

  CallArgList CallArgs;

  bool ShouldAdjustReturnPointer = true;
  QualType ArgType = MD->getThisType(getContext());
  llvm::Value *Arg = Builder.CreateLoad(LocalDeclMap[ThisDecl], "this");
  if (!Adjustment.ThisAdjustment.isEmpty()) {
    // Do the this adjustment.
    const llvm::Type *OrigTy = Callee->getType();
    Arg = DynamicTypeAdjust(Arg, Adjustment.ThisAdjustment);
    
    if (!Adjustment.ReturnAdjustment.isEmpty()) {
      const CovariantThunkAdjustment &ReturnAdjustment = 
        CovariantThunkAdjustment(ThunkAdjustment(),
                                 Adjustment.ReturnAdjustment);
      
      Callee = CGM.BuildCovariantThunk(GD, Extern, ReturnAdjustment);
      
      Callee = Builder.CreateBitCast(Callee, OrigTy);
      ShouldAdjustReturnPointer = false;
    }
  }    

  CallArgs.push_back(std::make_pair(RValue::get(Arg), ArgType));

  for (FunctionDecl::param_const_iterator i = MD->param_begin(),
         e = MD->param_end();
       i != e; ++i) {
    ParmVarDecl *D = *i;
    QualType ArgType = D->getType();

    // llvm::Value *Arg = CGF.GetAddrOfLocalVar(Dst);
    Expr *Arg = new (getContext()) DeclRefExpr(D, ArgType.getNonReferenceType(),
                                               SourceLocation());
    CallArgs.push_back(std::make_pair(EmitCallArg(Arg, ArgType), ArgType));
  }

  RValue RV = EmitCall(CGM.getTypes().getFunctionInfo(ResultType, CallArgs),
                       Callee, ReturnValueSlot(), CallArgs, MD);
  if (ShouldAdjustReturnPointer && !Adjustment.ReturnAdjustment.isEmpty()) {
    bool CanBeZero = !(ResultType->isReferenceType()
    // FIXME: attr nonnull can't be zero either
                       /* || ResultType->hasAttr<NonNullAttr>() */ );
    // Do the return result adjustment.
    if (CanBeZero) {
      llvm::BasicBlock *NonZeroBlock = createBasicBlock();
      llvm::BasicBlock *ZeroBlock = createBasicBlock();
      llvm::BasicBlock *ContBlock = createBasicBlock();

      const llvm::Type *Ty = RV.getScalarVal()->getType();
      llvm::Value *Zero = llvm::Constant::getNullValue(Ty);
      Builder.CreateCondBr(Builder.CreateICmpNE(RV.getScalarVal(), Zero),
                           NonZeroBlock, ZeroBlock);
      EmitBlock(NonZeroBlock);
      llvm::Value *NZ = 
        DynamicTypeAdjust(RV.getScalarVal(), Adjustment.ReturnAdjustment);
      EmitBranch(ContBlock);
      EmitBlock(ZeroBlock);
      llvm::Value *Z = RV.getScalarVal();
      EmitBlock(ContBlock);
      llvm::PHINode *RVOrZero = Builder.CreatePHI(Ty);
      RVOrZero->reserveOperandSpace(2);
      RVOrZero->addIncoming(NZ, NonZeroBlock);
      RVOrZero->addIncoming(Z, ZeroBlock);
      RV = RValue::get(RVOrZero);
    } else
      RV = RValue::get(DynamicTypeAdjust(RV.getScalarVal(), 
                                         Adjustment.ReturnAdjustment));
  }

  if (!ResultType->isVoidType())
    EmitReturnOfRValue(RV, ResultType);

  FinishFunction();
  return Fn;
}

llvm::Constant *
CodeGenModule::GetAddrOfThunk(GlobalDecl GD,
                              const ThunkAdjustment &ThisAdjustment) {
  const CXXMethodDecl *MD = cast<CXXMethodDecl>(GD.getDecl());

  // Compute mangled name
  llvm::SmallString<256> OutName;
  if (const CXXDestructorDecl* DD = dyn_cast<CXXDestructorDecl>(MD))
    getMangleContext().mangleCXXDtorThunk(DD, GD.getDtorType(), ThisAdjustment,
                                          OutName);
  else
    getMangleContext().mangleThunk(MD, ThisAdjustment, OutName);
  OutName += '\0';
  const char* Name = UniqueMangledName(OutName.begin(), OutName.end());

  // Get function for mangled name
  const llvm::Type *Ty = getTypes().GetFunctionTypeForVtable(MD);
  return GetOrCreateLLVMFunction(Name, Ty, GlobalDecl());
}

llvm::Constant *
CodeGenModule::GetAddrOfCovariantThunk(GlobalDecl GD,
                                   const CovariantThunkAdjustment &Adjustment) {
  const CXXMethodDecl *MD = cast<CXXMethodDecl>(GD.getDecl());

  // Compute mangled name
  llvm::SmallString<256> OutName;
  getMangleContext().mangleCovariantThunk(MD, Adjustment, OutName);
  OutName += '\0';
  const char* Name = UniqueMangledName(OutName.begin(), OutName.end());

  // Get function for mangled name
  const llvm::Type *Ty = getTypes().GetFunctionTypeForVtable(MD);
  return GetOrCreateLLVMFunction(Name, Ty, GlobalDecl());
}

void CodeGenModule::BuildThunksForVirtual(GlobalDecl GD) {
  CGVtableInfo::AdjustmentVectorTy *AdjPtr = getVtableInfo().getAdjustments(GD);
  if (!AdjPtr)
    return;
  CGVtableInfo::AdjustmentVectorTy &Adj = *AdjPtr;
  const CXXMethodDecl *MD = cast<CXXMethodDecl>(GD.getDecl());
  for (unsigned i = 0; i < Adj.size(); i++) {
    GlobalDecl OGD = Adj[i].first;
    const CXXMethodDecl *OMD = cast<CXXMethodDecl>(OGD.getDecl());
    QualType nc_oret = OMD->getType()->getAs<FunctionType>()->getResultType();
    CanQualType oret = getContext().getCanonicalType(nc_oret);
    QualType nc_ret = MD->getType()->getAs<FunctionType>()->getResultType();
    CanQualType ret = getContext().getCanonicalType(nc_ret);
    ThunkAdjustment ReturnAdjustment;
    if (oret != ret) {
      QualType qD = nc_ret->getPointeeType();
      QualType qB = nc_oret->getPointeeType();
      CXXRecordDecl *D = cast<CXXRecordDecl>(qD->getAs<RecordType>()->getDecl());
      CXXRecordDecl *B = cast<CXXRecordDecl>(qB->getAs<RecordType>()->getDecl());
      ReturnAdjustment = ComputeThunkAdjustment(D, B);
    }
    ThunkAdjustment ThisAdjustment = Adj[i].second;
    bool Extern = !cast<CXXRecordDecl>(OMD->getDeclContext())->isInAnonymousNamespace();
    if (!ReturnAdjustment.isEmpty() || !ThisAdjustment.isEmpty()) {
      CovariantThunkAdjustment CoAdj(ThisAdjustment, ReturnAdjustment);
      llvm::Constant *FnConst;
      if (!ReturnAdjustment.isEmpty())
        FnConst = GetAddrOfCovariantThunk(GD, CoAdj);
      else
        FnConst = GetAddrOfThunk(GD, ThisAdjustment);
      if (!isa<llvm::Function>(FnConst)) {
        llvm::Constant *SubExpr =
            cast<llvm::ConstantExpr>(FnConst)->getOperand(0);
        llvm::Function *OldFn = cast<llvm::Function>(SubExpr);
        std::string Name = OldFn->getNameStr();
        GlobalDeclMap.erase(UniqueMangledName(Name.data(),
                                              Name.data() + Name.size() + 1));
        llvm::Constant *NewFnConst;
        if (!ReturnAdjustment.isEmpty())
          NewFnConst = GetAddrOfCovariantThunk(GD, CoAdj);
        else
          NewFnConst = GetAddrOfThunk(GD, ThisAdjustment);
        llvm::Function *NewFn = cast<llvm::Function>(NewFnConst);
        NewFn->takeName(OldFn);
        llvm::Constant *NewPtrForOldDecl =
            llvm::ConstantExpr::getBitCast(NewFn, OldFn->getType());
        OldFn->replaceAllUsesWith(NewPtrForOldDecl);
        OldFn->eraseFromParent();
        FnConst = NewFn;
      }
      llvm::Function *Fn = cast<llvm::Function>(FnConst);
      if (Fn->isDeclaration()) {
        llvm::GlobalVariable::LinkageTypes linktype;
        linktype = llvm::GlobalValue::WeakAnyLinkage;
        if (!Extern)
          linktype = llvm::GlobalValue::InternalLinkage;
        Fn->setLinkage(linktype);
        if (!Features.Exceptions && !Features.ObjCNonFragileABI)
          Fn->addFnAttr(llvm::Attribute::NoUnwind);
        Fn->setAlignment(2);
        CodeGenFunction(*this).GenerateCovariantThunk(Fn, GD, Extern, CoAdj);
      }
    }
  }
}

llvm::Constant *
CodeGenModule::BuildThunk(GlobalDecl GD, bool Extern,
                          const ThunkAdjustment &ThisAdjustment) {
  const CXXMethodDecl *MD = cast<CXXMethodDecl>(GD.getDecl());
  llvm::SmallString<256> OutName;
  if (const CXXDestructorDecl *D = dyn_cast<CXXDestructorDecl>(MD)) {
    getMangleContext().mangleCXXDtorThunk(D, GD.getDtorType(), ThisAdjustment,
                                          OutName);
  } else 
    getMangleContext().mangleThunk(MD, ThisAdjustment, OutName);
  
  llvm::GlobalVariable::LinkageTypes linktype;
  linktype = llvm::GlobalValue::WeakAnyLinkage;
  if (!Extern)
    linktype = llvm::GlobalValue::InternalLinkage;
  llvm::Type *Ptr8Ty=llvm::PointerType::get(llvm::Type::getInt8Ty(VMContext),0);
  const FunctionProtoType *FPT = MD->getType()->getAs<FunctionProtoType>();
  const llvm::FunctionType *FTy =
    getTypes().GetFunctionType(getTypes().getFunctionInfo(MD),
                               FPT->isVariadic());

  llvm::Function *Fn = llvm::Function::Create(FTy, linktype, OutName.str(),
                                              &getModule());
  CodeGenFunction(*this).GenerateThunk(Fn, GD, Extern, ThisAdjustment);
  llvm::Constant *m = llvm::ConstantExpr::getBitCast(Fn, Ptr8Ty);
  return m;
}

llvm::Constant *
CodeGenModule::BuildCovariantThunk(const GlobalDecl &GD, bool Extern,
                                   const CovariantThunkAdjustment &Adjustment) {
  const CXXMethodDecl *MD = cast<CXXMethodDecl>(GD.getDecl());
  llvm::SmallString<256> OutName;
  getMangleContext().mangleCovariantThunk(MD, Adjustment, OutName);
  llvm::GlobalVariable::LinkageTypes linktype;
  linktype = llvm::GlobalValue::WeakAnyLinkage;
  if (!Extern)
    linktype = llvm::GlobalValue::InternalLinkage;
  llvm::Type *Ptr8Ty=llvm::PointerType::get(llvm::Type::getInt8Ty(VMContext),0);
  const FunctionProtoType *FPT = MD->getType()->getAs<FunctionProtoType>();
  const llvm::FunctionType *FTy =
    getTypes().GetFunctionType(getTypes().getFunctionInfo(MD),
                               FPT->isVariadic());

  llvm::Function *Fn = llvm::Function::Create(FTy, linktype, OutName.str(),
                                              &getModule());
  CodeGenFunction(*this).GenerateCovariantThunk(Fn, MD, Extern, Adjustment);
  llvm::Constant *m = llvm::ConstantExpr::getBitCast(Fn, Ptr8Ty);
  return m;
}

llvm::Value *
CodeGenFunction::GetVirtualCXXBaseClassOffset(llvm::Value *This,
                                              const CXXRecordDecl *ClassDecl,
                                           const CXXRecordDecl *BaseClassDecl) {
  const llvm::Type *Int8PtrTy = 
    llvm::Type::getInt8Ty(VMContext)->getPointerTo();

  llvm::Value *VTablePtr = Builder.CreateBitCast(This, 
                                                 Int8PtrTy->getPointerTo());
  VTablePtr = Builder.CreateLoad(VTablePtr, "vtable");

  int64_t VBaseOffsetIndex = 
    CGM.getVtableInfo().getVirtualBaseOffsetIndex(ClassDecl, BaseClassDecl);
  
  llvm::Value *VBaseOffsetPtr = 
    Builder.CreateConstGEP1_64(VTablePtr, VBaseOffsetIndex, "vbase.offset.ptr");
  const llvm::Type *PtrDiffTy = 
    ConvertType(getContext().getPointerDiffType());
  
  VBaseOffsetPtr = Builder.CreateBitCast(VBaseOffsetPtr, 
                                         PtrDiffTy->getPointerTo());
                                         
  llvm::Value *VBaseOffset = Builder.CreateLoad(VBaseOffsetPtr, "vbase.offset");
  
  return VBaseOffset;
}

static llvm::Value *BuildVirtualCall(CodeGenFunction &CGF, uint64_t VtableIndex, 
                                     llvm::Value *This, const llvm::Type *Ty) {
  Ty = Ty->getPointerTo()->getPointerTo()->getPointerTo();
  
  llvm::Value *Vtable = CGF.Builder.CreateBitCast(This, Ty);
  Vtable = CGF.Builder.CreateLoad(Vtable);
  
  llvm::Value *VFuncPtr = 
    CGF.Builder.CreateConstInBoundsGEP1_64(Vtable, VtableIndex, "vfn");
  return CGF.Builder.CreateLoad(VFuncPtr);
}

llvm::Value *
CodeGenFunction::BuildVirtualCall(const CXXMethodDecl *MD, llvm::Value *This,
                                  const llvm::Type *Ty) {
  MD = MD->getCanonicalDecl();
  uint64_t VtableIndex = CGM.getVtableInfo().getMethodVtableIndex(MD);
  
  return ::BuildVirtualCall(*this, VtableIndex, This, Ty);
}

llvm::Value *
CodeGenFunction::BuildVirtualCall(const CXXDestructorDecl *DD, CXXDtorType Type, 
                                  llvm::Value *&This, const llvm::Type *Ty) {
  DD = cast<CXXDestructorDecl>(DD->getCanonicalDecl());
  uint64_t VtableIndex = 
    CGM.getVtableInfo().getMethodVtableIndex(GlobalDecl(DD, Type));

  return ::BuildVirtualCall(*this, VtableIndex, This, Ty);
}

void CodeGenFunction::InitializeVtablePtrs(const CXXRecordDecl *ClassDecl) {
  if (!ClassDecl->isDynamicClass())
    return;

  llvm::Constant *Vtable = CGM.getVtableInfo().getVtable(ClassDecl);
  CodeGenModule::AddrSubMap_t& AddressPoints =
      *(*CGM.AddressPoints[ClassDecl])[ClassDecl];
  llvm::Value *ThisPtr = LoadCXXThis();
  const ASTRecordLayout &Layout = getContext().getASTRecordLayout(ClassDecl);

  // Store address points for virtual bases
  for (CXXRecordDecl::base_class_const_iterator I = 
       ClassDecl->vbases_begin(), E = ClassDecl->vbases_end(); I != E; ++I) {
    const CXXBaseSpecifier &Base = *I;
    CXXRecordDecl *BaseClassDecl
      = cast<CXXRecordDecl>(Base.getType()->getAs<RecordType>()->getDecl());
    uint64_t Offset = Layout.getVBaseClassOffset(BaseClassDecl);
    InitializeVtablePtrsRecursive(BaseClassDecl, Vtable, AddressPoints,
                                  ThisPtr, Offset);
  }

  // Store address points for non-virtual bases and current class
  InitializeVtablePtrsRecursive(ClassDecl, Vtable, AddressPoints, ThisPtr, 0);
}

void CodeGenFunction::InitializeVtablePtrsRecursive(
        const CXXRecordDecl *ClassDecl,
        llvm::Constant *Vtable,
        CodeGenModule::AddrSubMap_t& AddressPoints,
        llvm::Value *ThisPtr,
        uint64_t Offset) {
  if (!ClassDecl->isDynamicClass())
    return;

  // Store address points for non-virtual bases
  const ASTRecordLayout &Layout = getContext().getASTRecordLayout(ClassDecl);
  for (CXXRecordDecl::base_class_const_iterator I = 
       ClassDecl->bases_begin(), E = ClassDecl->bases_end(); I != E; ++I) {
    const CXXBaseSpecifier &Base = *I;
    if (Base.isVirtual())
      continue;
    CXXRecordDecl *BaseClassDecl
      = cast<CXXRecordDecl>(Base.getType()->getAs<RecordType>()->getDecl());
    uint64_t NewOffset = Offset + Layout.getBaseClassOffset(BaseClassDecl);
    InitializeVtablePtrsRecursive(BaseClassDecl, Vtable, AddressPoints,
                                  ThisPtr, NewOffset);
  }

  // Compute the address point
  uint64_t AddressPoint = AddressPoints[std::make_pair(ClassDecl, Offset)];
  llvm::Value *VtableAddressPoint =
      Builder.CreateConstInBoundsGEP2_64(Vtable, 0, AddressPoint);

  // Compute the address to store the address point
  const llvm::Type *Int8PtrTy = llvm::Type::getInt8PtrTy(CGM.getLLVMContext());
  llvm::Value *VtableField = Builder.CreateBitCast(ThisPtr, Int8PtrTy);
  VtableField = Builder.CreateConstInBoundsGEP1_64(VtableField, Offset/8);
  const llvm::Type *AddressPointPtrTy =
      VtableAddressPoint->getType()->getPointerTo();
  VtableField = Builder.CreateBitCast(ThisPtr, AddressPointPtrTy);

  // Store address point
  Builder.CreateStore(VtableAddressPoint, VtableField);
}

