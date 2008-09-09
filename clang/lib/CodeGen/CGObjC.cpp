//===---- CGBuiltin.cpp - Emit LLVM Code for builtins ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit Objective-C code as LLVM code.
//
//===----------------------------------------------------------------------===//

#include "CGObjCRuntime.h"
#include "CodeGenFunction.h"
#include "CodeGenModule.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclObjC.h"
#include "clang/Basic/Diagnostic.h"
#include "llvm/ADT/STLExtras.h"

using namespace clang;
using namespace CodeGen;

/// Emits an instance of NSConstantString representing the object.
llvm::Value *CodeGenFunction::EmitObjCStringLiteral(const ObjCStringLiteral *E) {
  std::string String(E->getString()->getStrData(), E->getString()->getByteLength());
  llvm::Constant *C = CGM.getObjCRuntime().GenerateConstantString(String);
  // FIXME: This bitcast should just be made an invariant on the Runtime.
  return llvm::ConstantExpr::getBitCast(C, ConvertType(E->getType()));
}

/// Emit a selector.
llvm::Value *CodeGenFunction::EmitObjCSelectorExpr(const ObjCSelectorExpr *E) {
  // Untyped selector.
  // Note that this implementation allows for non-constant strings to be passed
  // as arguments to @selector().  Currently, the only thing preventing this
  // behaviour is the type checking in the front end.
  return CGM.getObjCRuntime().GetSelector(Builder, E->getSelector());
}

llvm::Value *CodeGenFunction::EmitObjCProtocolExpr(const ObjCProtocolExpr *E) {
  // FIXME: This should pass the Decl not the name.
  return CGM.getObjCRuntime().GenerateProtocolRef(Builder, E->getProtocol());
}


RValue CodeGenFunction::EmitObjCMessageExpr(const ObjCMessageExpr *E) {
  // Only the lookup mechanism and first two arguments of the method
  // implementation vary between runtimes.  We can get the receiver and
  // arguments in generic code.
  
  CGObjCRuntime &Runtime = CGM.getObjCRuntime();
  const Expr *ReceiverExpr = E->getReceiver();
  bool isSuperMessage = false;
  bool isClassMessage = false;
  // Find the receiver
  llvm::Value *Receiver;
  if (!ReceiverExpr) {
    const ObjCInterfaceDecl *OID = E->getClassInfo().first;

    // Very special case, super send in class method. The receiver is
    // self (the class object) and the send uses super semantics.
    if (!OID) {
      assert(!strcmp(E->getClassName()->getName(), "super") &&
             "Unexpected missing class interface in message send.");
      isSuperMessage = true;
      Receiver = LoadObjCSelf();
    } else {
      Receiver = Runtime.GetClass(Builder, OID);
    }
    
    isClassMessage = true;
  } else if (const PredefinedExpr *PDE =
               dyn_cast<PredefinedExpr>(E->getReceiver())) {
    assert(PDE->getIdentType() == PredefinedExpr::ObjCSuper);
    isSuperMessage = true;
    Receiver = LoadObjCSelf();
  } else {
    Receiver = EmitScalarExpr(E->getReceiver());
  }

  CallArgList Args;
  for (CallExpr::const_arg_iterator i = E->arg_begin(), e = E->arg_end(); 
       i != e; ++i)
    Args.push_back(std::make_pair(EmitAnyExprToTemp(*i), (*i)->getType()));
  
  if (isSuperMessage) {
    // super is only valid in an Objective-C method
    const ObjCMethodDecl *OMD = cast<ObjCMethodDecl>(CurFuncDecl);
    return Runtime.GenerateMessageSendSuper(*this, E->getType(),
                                            E->getSelector(),
                                            OMD->getClassInterface(),
                                            Receiver,
                                            isClassMessage,
                                            Args);
  }
  return Runtime.GenerateMessageSend(*this, E->getType(), E->getSelector(), 
                                     Receiver, isClassMessage, Args);
}

/// StartObjCMethod - Begin emission of an ObjCMethod. This generates
/// the LLVM function and sets the other context used by
/// CodeGenFunction.

// FIXME: This should really be merged with GenerateCode.
void CodeGenFunction::StartObjCMethod(const ObjCMethodDecl *OMD) {
  CurFn = CGM.getObjCRuntime().GenerateMethod(OMD);

  CGM.SetMethodAttributes(OMD, CurFn);
  
  llvm::BasicBlock *EntryBB = llvm::BasicBlock::Create("entry", CurFn);
  
  // Create a marker to make it easy to insert allocas into the entryblock
  // later.  Don't create this with the builder, because we don't want it
  // folded.
  llvm::Value *Undef = llvm::UndefValue::get(llvm::Type::Int32Ty);
  AllocaInsertPt = new llvm::BitCastInst(Undef, llvm::Type::Int32Ty, "allocapt",
                                         EntryBB);

  FnRetTy = OMD->getResultType();
  CurFuncDecl = OMD;

  ReturnBlock = llvm::BasicBlock::Create("return", CurFn);
  ReturnValue = 0;
  if (!FnRetTy->isVoidType())
    ReturnValue = CreateTempAlloca(ConvertType(FnRetTy), "retval");

  Builder.SetInsertPoint(EntryBB);
  
  // Emit allocs for param decls.  Give the LLVM Argument nodes names.
  llvm::Function::arg_iterator AI = CurFn->arg_begin();
  
  // Name the struct return argument.
  if (hasAggregateLLVMType(OMD->getResultType())) {
    AI->setName("agg.result");
    ++AI;
  }

  // Add implicit parameters to the decl map.
  EmitParmDecl(*OMD->getSelfDecl(), AI); 
  ++AI;

  EmitParmDecl(*OMD->getCmdDecl(), AI); 
  ++AI;

  for (unsigned i = 0, e = OMD->getNumParams(); i != e; ++i, ++AI) {
    assert(AI != CurFn->arg_end() && "Argument mismatch!");
    EmitParmDecl(*OMD->getParamDecl(i), AI);
  }
  assert(AI == CurFn->arg_end() && "Argument mismatch");
}

/// Generate an Objective-C method.  An Objective-C method is a C function with
/// its pointer, name, and types registered in the class struture.  
void CodeGenFunction::GenerateObjCMethod(const ObjCMethodDecl *OMD) {
  StartObjCMethod(OMD);
  EmitStmt(OMD->getBody());

  const CompoundStmt *S = dyn_cast<CompoundStmt>(OMD->getBody());
  if (S) {
    FinishFunction(S->getRBracLoc());
  } else {
    FinishFunction();
  }
}

// FIXME: I wasn't sure about the synthesis approach. If we end up
// generating an AST for the whole body we can just fall back to
// having a GenerateFunction which takes the body Stmt.

/// GenerateObjCGetter - Generate an Objective-C property getter
/// function. The given Decl must be either an ObjCCategoryImplDecl
/// or an ObjCImplementationDecl.
void CodeGenFunction::GenerateObjCGetter(const ObjCPropertyImplDecl *PID) {
  const ObjCPropertyDecl *PD = PID->getPropertyDecl();
  ObjCMethodDecl *OMD = PD->getGetterMethodDecl();
  assert(OMD && "Invalid call to generate getter (empty method)");
  // FIXME: This is rather murky, we create this here since they will
  // not have been created by Sema for us.
  OMD->createImplicitParams(getContext());
  StartObjCMethod(OMD);

  // FIXME: What about nonatomic?
  SourceLocation Loc = PD->getLocation();
  ValueDecl *Self = OMD->getSelfDecl();
  ObjCIvarDecl *Ivar = PID->getPropertyIvarDecl();
  DeclRefExpr Base(Self, Self->getType(), Loc);
  ObjCIvarRefExpr IvarRef(Ivar, Ivar->getType(), Loc, &Base,
                          true, true);
  ReturnStmt Return(Loc, &IvarRef);
  EmitStmt(&Return);

  FinishFunction();
}

/// GenerateObjCSetter - Generate an Objective-C property setter
/// function. The given Decl must be either an ObjCCategoryImplDecl
/// or an ObjCImplementationDecl.
void CodeGenFunction::GenerateObjCSetter(const ObjCPropertyImplDecl *PID) {
  const ObjCPropertyDecl *PD = PID->getPropertyDecl();
  ObjCMethodDecl *OMD = PD->getSetterMethodDecl();
  assert(OMD && "Invalid call to generate setter (empty method)");
  // FIXME: This is rather murky, we create this here since they will
  // not have been created by Sema for us.  
  OMD->createImplicitParams(getContext());
  StartObjCMethod(OMD);
  
  switch (PD->getSetterKind()) {
  case ObjCPropertyDecl::Assign: break;
  case ObjCPropertyDecl::Copy:
      CGM.ErrorUnsupported(PID, "Obj-C setter with 'copy'");
      break;
  case ObjCPropertyDecl::Retain:
      CGM.ErrorUnsupported(PID, "Obj-C setter with 'retain'");
      break;
  }

  // FIXME: What about nonatomic?
  SourceLocation Loc = PD->getLocation();
  ValueDecl *Self = OMD->getSelfDecl();
  ObjCIvarDecl *Ivar = PID->getPropertyIvarDecl();
  DeclRefExpr Base(Self, Self->getType(), Loc);
  ParmVarDecl *ArgDecl = OMD->getParamDecl(0);
  DeclRefExpr Arg(ArgDecl, ArgDecl->getType(), Loc);
  ObjCIvarRefExpr IvarRef(Ivar, Ivar->getType(), Loc, &Base,
                          true, true);
  BinaryOperator Assign(&IvarRef, &Arg, BinaryOperator::Assign,
                        Ivar->getType(), Loc);
  EmitStmt(&Assign);

  FinishFunction();
}

llvm::Value *CodeGenFunction::LoadObjCSelf(void) {
  const ObjCMethodDecl *OMD = cast<ObjCMethodDecl>(CurFuncDecl);
  return Builder.CreateLoad(LocalDeclMap[OMD->getSelfDecl()], "self");
}

RValue CodeGenFunction::EmitObjCPropertyGet(const ObjCPropertyRefExpr *E) {
  // Determine getter selector.
  Selector S;
  if (E->getKind() == ObjCPropertyRefExpr::MethodRef) {
    S = E->getGetterMethod()->getSelector();
  } else {
    S = E->getProperty()->getGetterName();
  }

  return CGM.getObjCRuntime().
    GenerateMessageSend(*this, E->getType(), S, 
                        EmitScalarExpr(E->getBase()), 
                        false, CallArgList());
}

void CodeGenFunction::EmitObjCPropertySet(const ObjCPropertyRefExpr *E,
                                          RValue Src) {
  Selector S;
  if (E->getKind() == ObjCPropertyRefExpr::MethodRef) {
    ObjCMethodDecl *Setter = E->getSetterMethod(); 
    
    if (Setter) {
      S = Setter->getSelector();
    } else {
      // FIXME: This should be diagnosed by sema.
      SourceRange Range = E->getSourceRange();
      CGM.getDiags().Report(getContext().getFullLoc(E->getLocStart()),
                            diag::err_typecheck_assign_const, 0, 0,
                            &Range, 1);
      return;
    }
  } else {
    S = E->getProperty()->getSetterName();
  }

  CallArgList Args;
  Args.push_back(std::make_pair(Src, E->getType()));
  CGM.getObjCRuntime().GenerateMessageSend(*this, getContext().VoidTy, S, 
                                           EmitScalarExpr(E->getBase()), 
                                           false, Args);
}

void CodeGenFunction::EmitObjCForCollectionStmt(const ObjCForCollectionStmt &S)
{
  llvm::Value *DeclAddress;
  QualType ElementTy;
  
  if (const DeclStmt *SD = dyn_cast<DeclStmt>(S.getElement())) {
    EmitStmt(SD);
    
    ElementTy = cast<ValueDecl>(SD->getDecl())->getType();
    DeclAddress = LocalDeclMap[SD->getDecl()];    
  } else {
    ElementTy = cast<Expr>(S.getElement())->getType();
    DeclAddress = 0;
  }
  
  // Fast enumeration state.
  QualType StateTy = getContext().getObjCFastEnumerationStateType();
  llvm::AllocaInst *StatePtr = CreateTempAlloca(ConvertType(StateTy), 
                                                "state.ptr");
  StatePtr->setAlignment(getContext().getTypeAlign(StateTy) >> 3);  
  EmitMemSetToZero(StatePtr, StateTy);
  
  // Number of elements in the items array.
  static const unsigned NumItems = 16;
  
  // Get selector
  llvm::SmallVector<IdentifierInfo*, 3> II;
  II.push_back(&CGM.getContext().Idents.get("countByEnumeratingWithState"));
  II.push_back(&CGM.getContext().Idents.get("objects"));
  II.push_back(&CGM.getContext().Idents.get("count"));
  Selector FastEnumSel = CGM.getContext().Selectors.getSelector(II.size(), 
                                                                &II[0]);

  QualType ItemsTy =
    getContext().getConstantArrayType(getContext().getObjCIdType(),
                                      llvm::APInt(32, NumItems), 
                                      ArrayType::Normal, 0);
  llvm::Value *ItemsPtr = CreateTempAlloca(ConvertType(ItemsTy), "items.ptr");
  
  llvm::Value *Collection = EmitScalarExpr(S.getCollection());
  
  CallArgList Args;
  Args.push_back(std::make_pair(RValue::get(StatePtr), 
                                getContext().getPointerType(StateTy)));
  
  Args.push_back(std::make_pair(RValue::get(ItemsPtr), 
                                getContext().getPointerType(ItemsTy)));
  
  const llvm::Type *UnsignedLongLTy = ConvertType(getContext().UnsignedLongTy);
  llvm::Constant *Count = llvm::ConstantInt::get(UnsignedLongLTy, NumItems);
  Args.push_back(std::make_pair(RValue::get(Count), 
                                getContext().UnsignedLongTy));
  
  RValue CountRV = 
    CGM.getObjCRuntime().GenerateMessageSend(*this, 
                                             getContext().UnsignedLongTy,
                                             FastEnumSel,
                                             Collection, false, Args);

  llvm::Value *LimitPtr = CreateTempAlloca(UnsignedLongLTy, "limit.ptr");
  Builder.CreateStore(CountRV.getScalarVal(), LimitPtr);
  
  llvm::BasicBlock *NoElements = llvm::BasicBlock::Create("noelements");
  llvm::BasicBlock *SetStartMutations = 
    llvm::BasicBlock::Create("setstartmutations");
  
  llvm::Value *Limit = Builder.CreateLoad(LimitPtr);
  llvm::Value *Zero = llvm::Constant::getNullValue(UnsignedLongLTy);

  llvm::Value *IsZero = Builder.CreateICmpEQ(Limit, Zero, "iszero");
  Builder.CreateCondBr(IsZero, NoElements, SetStartMutations);

  EmitBlock(SetStartMutations);
  
  llvm::Value *StartMutationsPtr = 
    CreateTempAlloca(UnsignedLongLTy);
  
  llvm::Value *StateMutationsPtrPtr = 
    Builder.CreateStructGEP(StatePtr, 2, "mutationsptr.ptr");
  llvm::Value *StateMutationsPtr = Builder.CreateLoad(StateMutationsPtrPtr, 
                                                      "mutationsptr");
  
  llvm::Value *StateMutations = Builder.CreateLoad(StateMutationsPtr, 
                                                   "mutations");
  
  Builder.CreateStore(StateMutations, StartMutationsPtr);
  
  llvm::BasicBlock *LoopStart = llvm::BasicBlock::Create("loopstart");
  EmitBlock(LoopStart);

  llvm::Value *CounterPtr = CreateTempAlloca(UnsignedLongLTy, "counter.ptr");
  Builder.CreateStore(Zero, CounterPtr);
  
  llvm::BasicBlock *LoopBody = llvm::BasicBlock::Create("loopbody"); 
  EmitBlock(LoopBody);

  StateMutationsPtr = Builder.CreateLoad(StateMutationsPtrPtr, "mutationsptr");
  StateMutations = Builder.CreateLoad(StateMutationsPtr, "statemutations");

  llvm::Value *StartMutations = Builder.CreateLoad(StartMutationsPtr, 
                                                   "mutations");
  llvm::Value *MutationsEqual = Builder.CreateICmpEQ(StateMutations, 
                                                     StartMutations,
                                                     "tobool");
  
  
  llvm::BasicBlock *WasMutated = llvm::BasicBlock::Create("wasmutated");
  llvm::BasicBlock *WasNotMutated = llvm::BasicBlock::Create("wasnotmutated");
  
  Builder.CreateCondBr(MutationsEqual, WasNotMutated, WasMutated);
  
  EmitBlock(WasMutated);
  llvm::Value *V =
    Builder.CreateBitCast(Collection, 
                          ConvertType(getContext().getObjCIdType()),
                          "tmp");
  Builder.CreateCall(CGM.getObjCRuntime().EnumerationMutationFunction(),
                     V);
  
  EmitBlock(WasNotMutated);
  
  llvm::Value *StateItemsPtr = 
    Builder.CreateStructGEP(StatePtr, 1, "stateitems.ptr");

  llvm::Value *Counter = Builder.CreateLoad(CounterPtr, "counter");

  llvm::Value *EnumStateItems = Builder.CreateLoad(StateItemsPtr,
                                                   "stateitems");

  llvm::Value *CurrentItemPtr = 
    Builder.CreateGEP(EnumStateItems, Counter, "currentitem.ptr");
  
  llvm::Value *CurrentItem = Builder.CreateLoad(CurrentItemPtr, "currentitem");
  
  // Cast the item to the right type.
  CurrentItem = Builder.CreateBitCast(CurrentItem,
                                      ConvertType(ElementTy), "tmp");
  
  if (!DeclAddress) {
    LValue LV = EmitLValue(cast<Expr>(S.getElement()));
    
    // Set the value to null.
    Builder.CreateStore(CurrentItem, LV.getAddress());
  } else
    Builder.CreateStore(CurrentItem, DeclAddress);
  
  // Increment the counter.
  Counter = Builder.CreateAdd(Counter, 
                              llvm::ConstantInt::get(UnsignedLongLTy, 1));
  Builder.CreateStore(Counter, CounterPtr);
  
  llvm::BasicBlock *LoopEnd = llvm::BasicBlock::Create("loopend");
  llvm::BasicBlock *AfterBody = llvm::BasicBlock::Create("afterbody");
  
  BreakContinueStack.push_back(BreakContinue(LoopEnd, AfterBody));

  EmitStmt(S.getBody());
  
  BreakContinueStack.pop_back();
  
  EmitBlock(AfterBody);
  
  llvm::BasicBlock *FetchMore = llvm::BasicBlock::Create("fetchmore");
  
  llvm::Value *IsLess = Builder.CreateICmpULT(Counter, Limit, "isless");
  Builder.CreateCondBr(IsLess, LoopBody, FetchMore);

  // Fetch more elements.
  EmitBlock(FetchMore);
  
  CountRV = 
    CGM.getObjCRuntime().GenerateMessageSend(*this, 
                                             getContext().UnsignedLongTy,
                                             FastEnumSel, 
                                             Collection, false, Args);
  Builder.CreateStore(CountRV.getScalarVal(), LimitPtr);
  Limit = Builder.CreateLoad(LimitPtr);
  
  IsZero = Builder.CreateICmpEQ(Limit, Zero, "iszero");
  Builder.CreateCondBr(IsZero, NoElements, LoopStart);
  
  // No more elements.
  EmitBlock(NoElements);

  if (!DeclAddress) {
    // If the element was not a declaration, set it to be null.

    LValue LV = EmitLValue(cast<Expr>(S.getElement()));
    
    // Set the value to null.
    Builder.CreateStore(llvm::Constant::getNullValue(ConvertType(ElementTy)),
                        LV.getAddress());
  }

  EmitBlock(LoopEnd);
}

void CodeGenFunction::EmitObjCAtTryStmt(const ObjCAtTryStmt &S)
{
  CGM.getObjCRuntime().EmitTryStmt(*this, S);
}

void CodeGenFunction::EmitObjCAtThrowStmt(const ObjCAtThrowStmt &S)
{
  CGM.getObjCRuntime().EmitThrowStmt(*this, S);
}

CGObjCRuntime::~CGObjCRuntime() {}
