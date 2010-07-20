//===--- CodeGenFunction.cpp - Emit LLVM Code from ASTs for a Function ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This coordinates the per-function state used while generating code.
//
//===----------------------------------------------------------------------===//

#include "CodeGenFunction.h"
#include "CodeGenModule.h"
#include "CGDebugInfo.h"
#include "CGException.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/AST/APValue.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/StmtCXX.h"
#include "clang/Frontend/CodeGenOptions.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Intrinsics.h"
using namespace clang;
using namespace CodeGen;

CodeGenFunction::CodeGenFunction(CodeGenModule &cgm)
  : BlockFunction(cgm, *this, Builder), CGM(cgm),
    Target(CGM.getContext().Target),
    Builder(cgm.getModule().getContext()),
    ExceptionSlot(0), DebugInfo(0), IndirectBranch(0),
    SwitchInsn(0), CaseRangeBlock(0), InvokeDest(0),
    DidCallStackSave(false), UnreachableBlock(0),
    CXXThisDecl(0), CXXThisValue(0), CXXVTTDecl(0), CXXVTTValue(0),
    ConditionalBranchLevel(0), TerminateLandingPad(0), TerminateHandler(0),
    TrapBB(0) {
      
  // Get some frequently used types.
  LLVMPointerWidth = Target.getPointerWidth(0);
  llvm::LLVMContext &LLVMContext = CGM.getLLVMContext();
  IntPtrTy = llvm::IntegerType::get(LLVMContext, LLVMPointerWidth);
  Int32Ty  = llvm::Type::getInt32Ty(LLVMContext);
  Int64Ty  = llvm::Type::getInt64Ty(LLVMContext);
      
  Exceptions = getContext().getLangOptions().Exceptions;
  CatchUndefined = getContext().getLangOptions().CatchUndefined;
  CGM.getMangleContext().startNewFunction();
}

ASTContext &CodeGenFunction::getContext() const {
  return CGM.getContext();
}


llvm::Value *CodeGenFunction::GetAddrOfLocalVar(const VarDecl *VD) {
  llvm::Value *Res = LocalDeclMap[VD];
  assert(Res && "Invalid argument to GetAddrOfLocalVar(), no decl!");
  return Res;
}

llvm::Constant *
CodeGenFunction::GetAddrOfStaticLocalVar(const VarDecl *BVD) {
  return cast<llvm::Constant>(GetAddrOfLocalVar(BVD));
}

const llvm::Type *CodeGenFunction::ConvertTypeForMem(QualType T) {
  return CGM.getTypes().ConvertTypeForMem(T);
}

const llvm::Type *CodeGenFunction::ConvertType(QualType T) {
  return CGM.getTypes().ConvertType(T);
}

bool CodeGenFunction::hasAggregateLLVMType(QualType T) {
  return T->isRecordType() || T->isArrayType() || T->isAnyComplexType() ||
    T->isMemberFunctionPointerType();
}

void CodeGenFunction::EmitReturnBlock() {
  // For cleanliness, we try to avoid emitting the return block for
  // simple cases.
  llvm::BasicBlock *CurBB = Builder.GetInsertBlock();

  if (CurBB) {
    assert(!CurBB->getTerminator() && "Unexpected terminated block.");

    // We have a valid insert point, reuse it if it is empty or there are no
    // explicit jumps to the return block.
    if (CurBB->empty() || ReturnBlock.Block->use_empty()) {
      ReturnBlock.Block->replaceAllUsesWith(CurBB);
      delete ReturnBlock.Block;
    } else
      EmitBlock(ReturnBlock.Block);
    return;
  }

  // Otherwise, if the return block is the target of a single direct
  // branch then we can just put the code in that block instead. This
  // cleans up functions which started with a unified return block.
  if (ReturnBlock.Block->hasOneUse()) {
    llvm::BranchInst *BI =
      dyn_cast<llvm::BranchInst>(*ReturnBlock.Block->use_begin());
    if (BI && BI->isUnconditional() &&
        BI->getSuccessor(0) == ReturnBlock.Block) {
      // Reset insertion point and delete the branch.
      Builder.SetInsertPoint(BI->getParent());
      BI->eraseFromParent();
      delete ReturnBlock.Block;
      return;
    }
  }

  // FIXME: We are at an unreachable point, there is no reason to emit the block
  // unless it has uses. However, we still need a place to put the debug
  // region.end for now.

  EmitBlock(ReturnBlock.Block);
}

static void EmitIfUsed(CodeGenFunction &CGF, llvm::BasicBlock *BB) {
  if (!BB) return;
  if (!BB->use_empty())
    return CGF.CurFn->getBasicBlockList().push_back(BB);
  delete BB;
}

void CodeGenFunction::FinishFunction(SourceLocation EndLoc) {
  assert(BreakContinueStack.empty() &&
         "mismatched push/pop in break/continue stack!");

  // Emit function epilog (to return).
  EmitReturnBlock();

  EmitFunctionInstrumentation("__cyg_profile_func_exit");

  // Emit debug descriptor for function end.
  if (CGDebugInfo *DI = getDebugInfo()) {
    DI->setLocation(EndLoc);
    DI->EmitRegionEnd(CurFn, Builder);
  }

  EmitFunctionEpilog(*CurFnInfo);
  EmitEndEHSpec(CurCodeDecl);

  assert(EHStack.empty() &&
         "did not remove all scopes from cleanup stack!");

  // If someone did an indirect goto, emit the indirect goto block at the end of
  // the function.
  if (IndirectBranch) {
    EmitBlock(IndirectBranch->getParent());
    Builder.ClearInsertionPoint();
  }
  
  // Remove the AllocaInsertPt instruction, which is just a convenience for us.
  llvm::Instruction *Ptr = AllocaInsertPt;
  AllocaInsertPt = 0;
  Ptr->eraseFromParent();
  
  // If someone took the address of a label but never did an indirect goto, we
  // made a zero entry PHI node, which is illegal, zap it now.
  if (IndirectBranch) {
    llvm::PHINode *PN = cast<llvm::PHINode>(IndirectBranch->getAddress());
    if (PN->getNumIncomingValues() == 0) {
      PN->replaceAllUsesWith(llvm::UndefValue::get(PN->getType()));
      PN->eraseFromParent();
    }
  }

  EmitIfUsed(*this, TerminateLandingPad);
  EmitIfUsed(*this, TerminateHandler);
  EmitIfUsed(*this, UnreachableBlock);

  if (CGM.getCodeGenOpts().EmitDeclMetadata)
    EmitDeclMetadata();
}

/// ShouldInstrumentFunction - Return true if the current function should be
/// instrumented with __cyg_profile_func_* calls
bool CodeGenFunction::ShouldInstrumentFunction() {
  if (!CGM.getCodeGenOpts().InstrumentFunctions)
    return false;
  if (CurFuncDecl->hasAttr<NoInstrumentFunctionAttr>())
    return false;
  return true;
}

/// EmitFunctionInstrumentation - Emit LLVM code to call the specified
/// instrumentation function with the current function and the call site, if
/// function instrumentation is enabled.
void CodeGenFunction::EmitFunctionInstrumentation(const char *Fn) {
  if (!ShouldInstrumentFunction())
    return;

  const llvm::PointerType *PointerTy;
  const llvm::FunctionType *FunctionTy;
  std::vector<const llvm::Type*> ProfileFuncArgs;

  // void __cyg_profile_func_{enter,exit} (void *this_fn, void *call_site);
  PointerTy = llvm::Type::getInt8PtrTy(VMContext);
  ProfileFuncArgs.push_back(PointerTy);
  ProfileFuncArgs.push_back(PointerTy);
  FunctionTy = llvm::FunctionType::get(
    llvm::Type::getVoidTy(VMContext),
    ProfileFuncArgs, false);

  llvm::Constant *F = CGM.CreateRuntimeFunction(FunctionTy, Fn);
  llvm::CallInst *CallSite = Builder.CreateCall(
    CGM.getIntrinsic(llvm::Intrinsic::returnaddress, 0, 0),
    llvm::ConstantInt::get(Int32Ty, 0),
    "callsite");

  Builder.CreateCall2(F,
                      llvm::ConstantExpr::getBitCast(CurFn, PointerTy),
                      CallSite);
}

void CodeGenFunction::StartFunction(GlobalDecl GD, QualType RetTy,
                                    llvm::Function *Fn,
                                    const FunctionArgList &Args,
                                    SourceLocation StartLoc) {
  const Decl *D = GD.getDecl();
  
  DidCallStackSave = false;
  CurCodeDecl = CurFuncDecl = D;
  FnRetTy = RetTy;
  CurFn = Fn;
  assert(CurFn->isDeclaration() && "Function already has body?");

  // Pass inline keyword to optimizer if it appears explicitly on any
  // declaration.
  if (const FunctionDecl *FD = dyn_cast_or_null<FunctionDecl>(D))
    for (FunctionDecl::redecl_iterator RI = FD->redecls_begin(),
           RE = FD->redecls_end(); RI != RE; ++RI)
      if (RI->isInlineSpecified()) {
        Fn->addFnAttr(llvm::Attribute::InlineHint);
        break;
      }

  llvm::BasicBlock *EntryBB = createBasicBlock("entry", CurFn);

  // Create a marker to make it easy to insert allocas into the entryblock
  // later.  Don't create this with the builder, because we don't want it
  // folded.
  llvm::Value *Undef = llvm::UndefValue::get(Int32Ty);
  AllocaInsertPt = new llvm::BitCastInst(Undef, Int32Ty, "", EntryBB);
  if (Builder.isNamePreserving())
    AllocaInsertPt->setName("allocapt");

  ReturnBlock = getJumpDestInCurrentScope("return");

  Builder.SetInsertPoint(EntryBB);

  QualType FnType = getContext().getFunctionType(RetTy, 0, 0, false, 0,
                                                 false, false, 0, 0,
                                                 /*FIXME?*/
                                                 FunctionType::ExtInfo());

  // Emit subprogram debug descriptor.
  if (CGDebugInfo *DI = getDebugInfo()) {
    DI->setLocation(StartLoc);
    DI->EmitFunctionStart(GD, FnType, CurFn, Builder);
  }

  EmitFunctionInstrumentation("__cyg_profile_func_enter");

  // FIXME: Leaked.
  // CC info is ignored, hopefully?
  CurFnInfo = &CGM.getTypes().getFunctionInfo(FnRetTy, Args,
                                              FunctionType::ExtInfo());

  if (RetTy->isVoidType()) {
    // Void type; nothing to return.
    ReturnValue = 0;
  } else if (CurFnInfo->getReturnInfo().getKind() == ABIArgInfo::Indirect &&
             hasAggregateLLVMType(CurFnInfo->getReturnType())) {
    // Indirect aggregate return; emit returned value directly into sret slot.
    // This reduces code size, and affects correctness in C++.
    ReturnValue = CurFn->arg_begin();
  } else {
    ReturnValue = CreateIRTemp(RetTy, "retval");
  }

  EmitStartEHSpec(CurCodeDecl);
  EmitFunctionProlog(*CurFnInfo, CurFn, Args);

  if (CXXThisDecl)
    CXXThisValue = Builder.CreateLoad(LocalDeclMap[CXXThisDecl], "this");
  if (CXXVTTDecl)
    CXXVTTValue = Builder.CreateLoad(LocalDeclMap[CXXVTTDecl], "vtt");

  // If any of the arguments have a variably modified type, make sure to
  // emit the type size.
  for (FunctionArgList::const_iterator i = Args.begin(), e = Args.end();
       i != e; ++i) {
    QualType Ty = i->second;

    if (Ty->isVariablyModifiedType())
      EmitVLASize(Ty);
  }
}

void CodeGenFunction::EmitFunctionBody(FunctionArgList &Args) {
  const FunctionDecl *FD = cast<FunctionDecl>(CurGD.getDecl());
  assert(FD->getBody());
  EmitStmt(FD->getBody());
}

void CodeGenFunction::GenerateCode(GlobalDecl GD, llvm::Function *Fn) {
  const FunctionDecl *FD = cast<FunctionDecl>(GD.getDecl());
  
  // Check if we should generate debug info for this function.
  if (CGM.getDebugInfo() && !FD->hasAttr<NoDebugAttr>())
    DebugInfo = CGM.getDebugInfo();

  FunctionArgList Args;

  CurGD = GD;
  if (const CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(FD)) {
    if (MD->isInstance()) {
      // Create the implicit 'this' decl.
      // FIXME: I'm not entirely sure I like using a fake decl just for code
      // generation. Maybe we can come up with a better way?
      CXXThisDecl = ImplicitParamDecl::Create(getContext(), 0,
                                              FD->getLocation(),
                                              &getContext().Idents.get("this"),
                                              MD->getThisType(getContext()));
      Args.push_back(std::make_pair(CXXThisDecl, CXXThisDecl->getType()));
      
      // Check if we need a VTT parameter as well.
      if (CodeGenVTables::needsVTTParameter(GD)) {
        // FIXME: The comment about using a fake decl above applies here too.
        QualType T = getContext().getPointerType(getContext().VoidPtrTy);
        CXXVTTDecl = 
          ImplicitParamDecl::Create(getContext(), 0, FD->getLocation(),
                                    &getContext().Idents.get("vtt"), T);
        Args.push_back(std::make_pair(CXXVTTDecl, CXXVTTDecl->getType()));
      }
    }
  }

  if (FD->getNumParams()) {
    const FunctionProtoType* FProto = FD->getType()->getAs<FunctionProtoType>();
    assert(FProto && "Function def must have prototype!");

    for (unsigned i = 0, e = FD->getNumParams(); i != e; ++i)
      Args.push_back(std::make_pair(FD->getParamDecl(i),
                                    FProto->getArgType(i)));
  }

  SourceRange BodyRange;
  if (Stmt *Body = FD->getBody()) BodyRange = Body->getSourceRange();

  // Emit the standard function prologue.
  StartFunction(GD, FD->getResultType(), Fn, Args, BodyRange.getBegin());

  // Generate the body of the function.
  if (isa<CXXDestructorDecl>(FD))
    EmitDestructorBody(Args);
  else if (isa<CXXConstructorDecl>(FD))
    EmitConstructorBody(Args);
  else
    EmitFunctionBody(Args);

  // Emit the standard function epilogue.
  FinishFunction(BodyRange.getEnd());

  // Destroy the 'this' declaration.
  if (CXXThisDecl)
    CXXThisDecl->Destroy(getContext());
  
  // Destroy the VTT declaration.
  if (CXXVTTDecl)
    CXXVTTDecl->Destroy(getContext());
}

/// ContainsLabel - Return true if the statement contains a label in it.  If
/// this statement is not executed normally, it not containing a label means
/// that we can just remove the code.
bool CodeGenFunction::ContainsLabel(const Stmt *S, bool IgnoreCaseStmts) {
  // Null statement, not a label!
  if (S == 0) return false;

  // If this is a label, we have to emit the code, consider something like:
  // if (0) {  ...  foo:  bar(); }  goto foo;
  if (isa<LabelStmt>(S))
    return true;

  // If this is a case/default statement, and we haven't seen a switch, we have
  // to emit the code.
  if (isa<SwitchCase>(S) && !IgnoreCaseStmts)
    return true;

  // If this is a switch statement, we want to ignore cases below it.
  if (isa<SwitchStmt>(S))
    IgnoreCaseStmts = true;

  // Scan subexpressions for verboten labels.
  for (Stmt::const_child_iterator I = S->child_begin(), E = S->child_end();
       I != E; ++I)
    if (ContainsLabel(*I, IgnoreCaseStmts))
      return true;

  return false;
}


/// ConstantFoldsToSimpleInteger - If the sepcified expression does not fold to
/// a constant, or if it does but contains a label, return 0.  If it constant
/// folds to 'true' and does not contain a label, return 1, if it constant folds
/// to 'false' and does not contain a label, return -1.
int CodeGenFunction::ConstantFoldsToSimpleInteger(const Expr *Cond) {
  // FIXME: Rename and handle conversion of other evaluatable things
  // to bool.
  Expr::EvalResult Result;
  if (!Cond->Evaluate(Result, getContext()) || !Result.Val.isInt() ||
      Result.HasSideEffects)
    return 0;  // Not foldable, not integer or not fully evaluatable.

  if (CodeGenFunction::ContainsLabel(Cond))
    return 0;  // Contains a label.

  return Result.Val.getInt().getBoolValue() ? 1 : -1;
}


/// EmitBranchOnBoolExpr - Emit a branch on a boolean condition (e.g. for an if
/// statement) to the specified blocks.  Based on the condition, this might try
/// to simplify the codegen of the conditional based on the branch.
///
void CodeGenFunction::EmitBranchOnBoolExpr(const Expr *Cond,
                                           llvm::BasicBlock *TrueBlock,
                                           llvm::BasicBlock *FalseBlock) {
  if (const ParenExpr *PE = dyn_cast<ParenExpr>(Cond))
    return EmitBranchOnBoolExpr(PE->getSubExpr(), TrueBlock, FalseBlock);

  if (const BinaryOperator *CondBOp = dyn_cast<BinaryOperator>(Cond)) {
    // Handle X && Y in a condition.
    if (CondBOp->getOpcode() == BinaryOperator::LAnd) {
      // If we have "1 && X", simplify the code.  "0 && X" would have constant
      // folded if the case was simple enough.
      if (ConstantFoldsToSimpleInteger(CondBOp->getLHS()) == 1) {
        // br(1 && X) -> br(X).
        return EmitBranchOnBoolExpr(CondBOp->getRHS(), TrueBlock, FalseBlock);
      }

      // If we have "X && 1", simplify the code to use an uncond branch.
      // "X && 0" would have been constant folded to 0.
      if (ConstantFoldsToSimpleInteger(CondBOp->getRHS()) == 1) {
        // br(X && 1) -> br(X).
        return EmitBranchOnBoolExpr(CondBOp->getLHS(), TrueBlock, FalseBlock);
      }

      // Emit the LHS as a conditional.  If the LHS conditional is false, we
      // want to jump to the FalseBlock.
      llvm::BasicBlock *LHSTrue = createBasicBlock("land.lhs.true");
      EmitBranchOnBoolExpr(CondBOp->getLHS(), LHSTrue, FalseBlock);
      EmitBlock(LHSTrue);

      // Any temporaries created here are conditional.
      BeginConditionalBranch();
      EmitBranchOnBoolExpr(CondBOp->getRHS(), TrueBlock, FalseBlock);
      EndConditionalBranch();

      return;
    } else if (CondBOp->getOpcode() == BinaryOperator::LOr) {
      // If we have "0 || X", simplify the code.  "1 || X" would have constant
      // folded if the case was simple enough.
      if (ConstantFoldsToSimpleInteger(CondBOp->getLHS()) == -1) {
        // br(0 || X) -> br(X).
        return EmitBranchOnBoolExpr(CondBOp->getRHS(), TrueBlock, FalseBlock);
      }

      // If we have "X || 0", simplify the code to use an uncond branch.
      // "X || 1" would have been constant folded to 1.
      if (ConstantFoldsToSimpleInteger(CondBOp->getRHS()) == -1) {
        // br(X || 0) -> br(X).
        return EmitBranchOnBoolExpr(CondBOp->getLHS(), TrueBlock, FalseBlock);
      }

      // Emit the LHS as a conditional.  If the LHS conditional is true, we
      // want to jump to the TrueBlock.
      llvm::BasicBlock *LHSFalse = createBasicBlock("lor.lhs.false");
      EmitBranchOnBoolExpr(CondBOp->getLHS(), TrueBlock, LHSFalse);
      EmitBlock(LHSFalse);

      // Any temporaries created here are conditional.
      BeginConditionalBranch();
      EmitBranchOnBoolExpr(CondBOp->getRHS(), TrueBlock, FalseBlock);
      EndConditionalBranch();

      return;
    }
  }

  if (const UnaryOperator *CondUOp = dyn_cast<UnaryOperator>(Cond)) {
    // br(!x, t, f) -> br(x, f, t)
    if (CondUOp->getOpcode() == UnaryOperator::LNot)
      return EmitBranchOnBoolExpr(CondUOp->getSubExpr(), FalseBlock, TrueBlock);
  }

  if (const ConditionalOperator *CondOp = dyn_cast<ConditionalOperator>(Cond)) {
    // Handle ?: operator.

    // Just ignore GNU ?: extension.
    if (CondOp->getLHS()) {
      // br(c ? x : y, t, f) -> br(c, br(x, t, f), br(y, t, f))
      llvm::BasicBlock *LHSBlock = createBasicBlock("cond.true");
      llvm::BasicBlock *RHSBlock = createBasicBlock("cond.false");
      EmitBranchOnBoolExpr(CondOp->getCond(), LHSBlock, RHSBlock);
      EmitBlock(LHSBlock);
      EmitBranchOnBoolExpr(CondOp->getLHS(), TrueBlock, FalseBlock);
      EmitBlock(RHSBlock);
      EmitBranchOnBoolExpr(CondOp->getRHS(), TrueBlock, FalseBlock);
      return;
    }
  }

  // Emit the code with the fully general case.
  llvm::Value *CondV = EvaluateExprAsBool(Cond);
  Builder.CreateCondBr(CondV, TrueBlock, FalseBlock);
}

/// ErrorUnsupported - Print out an error that codegen doesn't support the
/// specified stmt yet.
void CodeGenFunction::ErrorUnsupported(const Stmt *S, const char *Type,
                                       bool OmitOnError) {
  CGM.ErrorUnsupported(S, Type, OmitOnError);
}

void
CodeGenFunction::EmitNullInitialization(llvm::Value *DestPtr, QualType Ty) {
  // If the type contains a pointer to data member we can't memset it to zero.
  // Instead, create a null constant and copy it to the destination.
  if (CGM.getTypes().ContainsPointerToDataMember(Ty)) {
    llvm::Constant *NullConstant = CGM.EmitNullConstant(Ty);
    
    llvm::GlobalVariable *NullVariable = 
      new llvm::GlobalVariable(CGM.getModule(), NullConstant->getType(),
                               /*isConstant=*/true, 
                               llvm::GlobalVariable::PrivateLinkage,
                               NullConstant, llvm::Twine());
    EmitAggregateCopy(DestPtr, NullVariable, Ty, /*isVolatile=*/false);
    return;
  } 
  

  // Ignore empty classes in C++.
  if (getContext().getLangOptions().CPlusPlus) {
    if (const RecordType *RT = Ty->getAs<RecordType>()) {
      if (cast<CXXRecordDecl>(RT->getDecl())->isEmpty())
        return;
    }
  }
  
  // Otherwise, just memset the whole thing to zero.  This is legal
  // because in LLVM, all default initializers (other than the ones we just
  // handled above) are guaranteed to have a bit pattern of all zeros.
  const llvm::Type *BP = llvm::Type::getInt8PtrTy(VMContext);
  if (DestPtr->getType() != BP)
    DestPtr = Builder.CreateBitCast(DestPtr, BP, "tmp");

  // Get size and alignment info for this aggregate.
  std::pair<uint64_t, unsigned> TypeInfo = getContext().getTypeInfo(Ty);

  // Don't bother emitting a zero-byte memset.
  if (TypeInfo.first == 0)
    return;

  // FIXME: Handle variable sized types.
  Builder.CreateCall5(CGM.getMemSetFn(BP, IntPtrTy), DestPtr,
                 llvm::Constant::getNullValue(llvm::Type::getInt8Ty(VMContext)),
                      // TypeInfo.first describes size in bits.
                      llvm::ConstantInt::get(IntPtrTy, TypeInfo.first/8),
                      llvm::ConstantInt::get(Int32Ty, TypeInfo.second/8),
                      llvm::ConstantInt::get(llvm::Type::getInt1Ty(VMContext),
                                             0));
}

llvm::BlockAddress *CodeGenFunction::GetAddrOfLabel(const LabelStmt *L) {
  // Make sure that there is a block for the indirect goto.
  if (IndirectBranch == 0)
    GetIndirectGotoBlock();
  
  llvm::BasicBlock *BB = getJumpDestForLabel(L).Block;
  
  // Make sure the indirect branch includes all of the address-taken blocks.
  IndirectBranch->addDestination(BB);
  return llvm::BlockAddress::get(CurFn, BB);
}

llvm::BasicBlock *CodeGenFunction::GetIndirectGotoBlock() {
  // If we already made the indirect branch for indirect goto, return its block.
  if (IndirectBranch) return IndirectBranch->getParent();
  
  CGBuilderTy TmpBuilder(createBasicBlock("indirectgoto"));
  
  const llvm::Type *Int8PtrTy = llvm::Type::getInt8PtrTy(VMContext);

  // Create the PHI node that indirect gotos will add entries to.
  llvm::Value *DestVal = TmpBuilder.CreatePHI(Int8PtrTy, "indirect.goto.dest");
  
  // Create the indirect branch instruction.
  IndirectBranch = TmpBuilder.CreateIndirectBr(DestVal);
  return IndirectBranch->getParent();
}

llvm::Value *CodeGenFunction::GetVLASize(const VariableArrayType *VAT) {
  llvm::Value *&SizeEntry = VLASizeMap[VAT->getSizeExpr()];

  assert(SizeEntry && "Did not emit size for type");
  return SizeEntry;
}

llvm::Value *CodeGenFunction::EmitVLASize(QualType Ty) {
  assert(Ty->isVariablyModifiedType() &&
         "Must pass variably modified type to EmitVLASizes!");

  EnsureInsertPoint();

  if (const VariableArrayType *VAT = getContext().getAsVariableArrayType(Ty)) {
    llvm::Value *&SizeEntry = VLASizeMap[VAT->getSizeExpr()];

    if (!SizeEntry) {
      const llvm::Type *SizeTy = ConvertType(getContext().getSizeType());

      // Get the element size;
      QualType ElemTy = VAT->getElementType();
      llvm::Value *ElemSize;
      if (ElemTy->isVariableArrayType())
        ElemSize = EmitVLASize(ElemTy);
      else
        ElemSize = llvm::ConstantInt::get(SizeTy,
            getContext().getTypeSizeInChars(ElemTy).getQuantity());

      llvm::Value *NumElements = EmitScalarExpr(VAT->getSizeExpr());
      NumElements = Builder.CreateIntCast(NumElements, SizeTy, false, "tmp");

      SizeEntry = Builder.CreateMul(ElemSize, NumElements);
    }

    return SizeEntry;
  }

  if (const ArrayType *AT = dyn_cast<ArrayType>(Ty)) {
    EmitVLASize(AT->getElementType());
    return 0;
  }

  const PointerType *PT = Ty->getAs<PointerType>();
  assert(PT && "unknown VM type!");
  EmitVLASize(PT->getPointeeType());
  return 0;
}

llvm::Value* CodeGenFunction::EmitVAListRef(const Expr* E) {
  if (CGM.getContext().getBuiltinVaListType()->isArrayType())
    return EmitScalarExpr(E);
  return EmitLValue(E).getAddress();
}

/// Pops cleanup blocks until the given savepoint is reached.
void CodeGenFunction::PopCleanupBlocks(EHScopeStack::stable_iterator Old) {
  assert(Old.isValid());

  EHScopeStack::iterator E = EHStack.find(Old);
  while (EHStack.begin() != E)
    PopCleanupBlock();
}

/// Destroys a cleanup if it was unused.
static void DestroyCleanup(CodeGenFunction &CGF,
                           llvm::BasicBlock *Entry,
                           llvm::BasicBlock *Exit) {
  assert(Entry->use_empty() && "destroying cleanup with uses!");
  assert(Exit->getTerminator() == 0 &&
         "exit has terminator but entry has no predecessors!");

  // This doesn't always remove the entire cleanup, but it's much
  // safer as long as we don't know what blocks belong to the cleanup.
  // A *much* better approach if we care about this inefficiency would
  // be to lazily emit the cleanup.

  // If the exit block is distinct from the entry, give it a branch to
  // an unreachable destination.  This preserves the well-formedness
  // of the IR.
  if (Entry != Exit)
    llvm::BranchInst::Create(CGF.getUnreachableBlock(), Exit);

  assert(!Entry->getParent() && "cleanup entry already positioned?");
  // We can't just delete the entry; we have to kill any references to
  // its instructions in other blocks.
  for (llvm::BasicBlock::iterator I = Entry->begin(), E = Entry->end();
         I != E; ++I)
    if (!I->use_empty())
      I->replaceAllUsesWith(llvm::UndefValue::get(I->getType()));
  delete Entry;
}

/// Creates a switch instruction to thread branches out of the given
/// block (which is the exit block of a cleanup).
static void CreateCleanupSwitch(CodeGenFunction &CGF,
                                llvm::BasicBlock *Block) {
  if (Block->getTerminator()) {
    assert(isa<llvm::SwitchInst>(Block->getTerminator()) &&
           "cleanup block already has a terminator, but it isn't a switch");
    return;
  }

  llvm::Value *DestCodePtr
    = CGF.CreateTempAlloca(CGF.Builder.getInt32Ty(), "cleanup.dst");
  CGBuilderTy Builder(Block);
  llvm::Value *DestCode = Builder.CreateLoad(DestCodePtr, "tmp");

  // Create a switch instruction to determine where to jump next.
  Builder.CreateSwitch(DestCode, CGF.getUnreachableBlock());
}

/// Attempts to reduce a cleanup's entry block to a fallthrough.  This
/// is basically llvm::MergeBlockIntoPredecessor, except
/// simplified/optimized for the tighter constraints on cleanup
/// blocks.
static void SimplifyCleanupEntry(CodeGenFunction &CGF,
                                 llvm::BasicBlock *Entry) {
  llvm::BasicBlock *Pred = Entry->getSinglePredecessor();
  if (!Pred) return;

  llvm::BranchInst *Br = dyn_cast<llvm::BranchInst>(Pred->getTerminator());
  if (!Br || Br->isConditional()) return;
  assert(Br->getSuccessor(0) == Entry);

  // If we were previously inserting at the end of the cleanup entry
  // block, we'll need to continue inserting at the end of the
  // predecessor.
  bool WasInsertBlock = CGF.Builder.GetInsertBlock() == Entry;
  assert(!WasInsertBlock || CGF.Builder.GetInsertPoint() == Entry->end());

  // Kill the branch.
  Br->eraseFromParent();

  // Merge the blocks.
  Pred->getInstList().splice(Pred->end(), Entry->getInstList());

  // Kill the entry block.
  Entry->eraseFromParent();

  if (WasInsertBlock)
    CGF.Builder.SetInsertPoint(Pred);
}

/// Attempts to reduce an cleanup's exit switch to an unconditional
/// branch.
static void SimplifyCleanupExit(llvm::BasicBlock *Exit) {
  llvm::TerminatorInst *Terminator = Exit->getTerminator();
  assert(Terminator && "completed cleanup exit has no terminator");

  llvm::SwitchInst *Switch = dyn_cast<llvm::SwitchInst>(Terminator);
  if (!Switch) return;
  if (Switch->getNumCases() != 2) return; // default + 1

  llvm::LoadInst *Cond = cast<llvm::LoadInst>(Switch->getCondition());
  llvm::AllocaInst *CondVar = cast<llvm::AllocaInst>(Cond->getPointerOperand());

  // Replace the switch instruction with an unconditional branch.
  llvm::BasicBlock *Dest = Switch->getSuccessor(1); // default is 0
  Switch->eraseFromParent();
  llvm::BranchInst::Create(Dest, Exit);

  // Delete all uses of the condition variable.
  Cond->eraseFromParent();
  while (!CondVar->use_empty())
    cast<llvm::StoreInst>(*CondVar->use_begin())->eraseFromParent();

  // Delete the condition variable itself.
  CondVar->eraseFromParent();
}

/// Threads a branch fixup through a cleanup block.
static void ThreadFixupThroughCleanup(CodeGenFunction &CGF,
                                      BranchFixup &Fixup,
                                      llvm::BasicBlock *Entry,
                                      llvm::BasicBlock *Exit) {
  if (!Exit->getTerminator())
    CreateCleanupSwitch(CGF, Exit);

  // Find the switch and its destination index alloca.
  llvm::SwitchInst *Switch = cast<llvm::SwitchInst>(Exit->getTerminator());
  llvm::Value *DestCodePtr =
    cast<llvm::LoadInst>(Switch->getCondition())->getPointerOperand();

  // Compute the index of the new case we're adding to the switch.
  unsigned Index = Switch->getNumCases();

  const llvm::IntegerType *i32 = llvm::Type::getInt32Ty(CGF.getLLVMContext());
  llvm::ConstantInt *IndexV = llvm::ConstantInt::get(i32, Index);

  // Set the index in the origin block.
  new llvm::StoreInst(IndexV, DestCodePtr, Fixup.Origin);

  // Add a case to the switch.
  Switch->addCase(IndexV, Fixup.Destination);

  // Change the last branch to point to the cleanup entry block.
  Fixup.LatestBranch->setSuccessor(Fixup.LatestBranchIndex, Entry);

  // And finally, update the fixup.
  Fixup.LatestBranch = Switch;
  Fixup.LatestBranchIndex = Index;
}

/// Try to simplify both the entry and exit edges of a cleanup.
static void SimplifyCleanupEdges(CodeGenFunction &CGF,
                                 llvm::BasicBlock *Entry,
                                 llvm::BasicBlock *Exit) {

  // Given their current implementations, it's important to run these
  // in this order: SimplifyCleanupEntry will delete Entry if it can
  // be merged into its predecessor, which will then break
  // SimplifyCleanupExit if (as is common) Entry == Exit.

  SimplifyCleanupExit(Exit);
  SimplifyCleanupEntry(CGF, Entry);  
}

static void EmitLazyCleanup(CodeGenFunction &CGF,
                            EHScopeStack::LazyCleanup *Fn,
                            bool ForEH) {
  if (ForEH) CGF.EHStack.pushTerminate();
  Fn->Emit(CGF, ForEH);
  if (ForEH) CGF.EHStack.popTerminate();
  assert(CGF.HaveInsertPoint() && "cleanup ended with no insertion point?");
}

static void SplitAndEmitLazyCleanup(CodeGenFunction &CGF,
                                    EHScopeStack::LazyCleanup *Fn,
                                    bool ForEH,
                                    llvm::BasicBlock *Entry) {
  assert(Entry && "no entry block for cleanup");

  // Remove the switch and load from the end of the entry block.
  llvm::Instruction *Switch = &Entry->getInstList().back();
  Entry->getInstList().remove(Switch);
  assert(isa<llvm::SwitchInst>(Switch));
  llvm::Instruction *Load = &Entry->getInstList().back();
  Entry->getInstList().remove(Load);
  assert(isa<llvm::LoadInst>(Load));

  assert(Entry->getInstList().empty() &&
         "lazy cleanup block not empty after removing load/switch pair?");

  // Emit the actual cleanup at the end of the entry block.
  CGF.Builder.SetInsertPoint(Entry);
  EmitLazyCleanup(CGF, Fn, ForEH);

  // Put the load and switch at the end of the exit block.
  llvm::BasicBlock *Exit = CGF.Builder.GetInsertBlock();
  Exit->getInstList().push_back(Load);
  Exit->getInstList().push_back(Switch);

  // Clean up the edges if possible.
  SimplifyCleanupEdges(CGF, Entry, Exit);

  CGF.Builder.ClearInsertionPoint();
}

static void PopLazyCleanupBlock(CodeGenFunction &CGF) {
  assert(isa<EHLazyCleanupScope>(*CGF.EHStack.begin()) && "top not a cleanup!");
  EHLazyCleanupScope &Scope = cast<EHLazyCleanupScope>(*CGF.EHStack.begin());
  assert(Scope.getFixupDepth() <= CGF.EHStack.getNumBranchFixups());

  // Check whether we need an EH cleanup.  This is only true if we've
  // generated a lazy EH cleanup block.
  llvm::BasicBlock *EHEntry = Scope.getEHBlock();
  bool RequiresEHCleanup = (EHEntry != 0);

  // Check the three conditions which might require a normal cleanup:

  // - whether there are branch fix-ups through this cleanup
  unsigned FixupDepth = Scope.getFixupDepth();
  bool HasFixups = CGF.EHStack.getNumBranchFixups() != FixupDepth;

  // - whether control has already been threaded through this cleanup
  llvm::BasicBlock *NormalEntry = Scope.getNormalBlock();
  bool HasExistingBranches = (NormalEntry != 0);

  // - whether there's a fallthrough
  llvm::BasicBlock *FallthroughSource = CGF.Builder.GetInsertBlock();
  bool HasFallthrough = (FallthroughSource != 0);

  bool RequiresNormalCleanup = false;
  if (Scope.isNormalCleanup() &&
      (HasFixups || HasExistingBranches || HasFallthrough)) {
    RequiresNormalCleanup = true;
  }

  // If we don't need the cleanup at all, we're done.
  if (!RequiresNormalCleanup && !RequiresEHCleanup) {
    CGF.EHStack.popCleanup();
    assert(CGF.EHStack.getNumBranchFixups() == 0 ||
           CGF.EHStack.hasNormalCleanups());
    return;
  }

  // Copy the cleanup emission data out.  Note that SmallVector
  // guarantees maximal alignment for its buffer regardless of its
  // type parameter.
  llvm::SmallVector<char, 8*sizeof(void*)> CleanupBuffer;
  CleanupBuffer.reserve(Scope.getCleanupSize());
  memcpy(CleanupBuffer.data(),
         Scope.getCleanupBuffer(), Scope.getCleanupSize());
  CleanupBuffer.set_size(Scope.getCleanupSize());
  EHScopeStack::LazyCleanup *Fn =
    reinterpret_cast<EHScopeStack::LazyCleanup*>(CleanupBuffer.data());

  // We're done with the scope; pop it off so we can emit the cleanups.
  CGF.EHStack.popCleanup();

  if (RequiresNormalCleanup) {
    // If we have a fallthrough and no other need for the cleanup,
    // emit it directly.
    if (HasFallthrough && !HasFixups && !HasExistingBranches) {
      EmitLazyCleanup(CGF, Fn, /*ForEH*/ false);

    // Otherwise, the best approach is to thread everything through
    // the cleanup block and then try to clean up after ourselves.
    } else {
      // Force the entry block to exist.
      if (!HasExistingBranches) {
        NormalEntry = CGF.createBasicBlock("cleanup");
        CreateCleanupSwitch(CGF, NormalEntry);
      }

      CGF.EmitBlock(NormalEntry);

      // Thread the fallthrough edge through the (momentarily trivial)
      // cleanup.
      llvm::BasicBlock *FallthroughDestination = 0;
      if (HasFallthrough) {
        assert(isa<llvm::BranchInst>(FallthroughSource->getTerminator()));
        FallthroughDestination = CGF.createBasicBlock("cleanup.cont");

        BranchFixup Fix;
        Fix.Destination = FallthroughDestination;
        Fix.LatestBranch = FallthroughSource->getTerminator();
        Fix.LatestBranchIndex = 0;
        Fix.Origin = Fix.LatestBranch;

        // Restore fixup invariant.  EmitBlock added a branch to the
        // cleanup which we need to redirect to the destination.
        cast<llvm::BranchInst>(Fix.LatestBranch)
          ->setSuccessor(0, Fix.Destination);

        ThreadFixupThroughCleanup(CGF, Fix, NormalEntry, NormalEntry);
      }

      // Thread any "real" fixups we need to thread.
      for (unsigned I = FixupDepth, E = CGF.EHStack.getNumBranchFixups();
           I != E; ++I)
        if (CGF.EHStack.getBranchFixup(I).Destination)
          ThreadFixupThroughCleanup(CGF, CGF.EHStack.getBranchFixup(I),
                                    NormalEntry, NormalEntry);

      SplitAndEmitLazyCleanup(CGF, Fn, /*ForEH*/ false, NormalEntry);

      if (HasFallthrough)
        CGF.EmitBlock(FallthroughDestination);
    }
  }

  // Emit the EH cleanup if required.
  if (RequiresEHCleanup) {
    CGBuilderTy::InsertPoint SavedIP = CGF.Builder.saveAndClearIP();
    CGF.EmitBlock(EHEntry);
    SplitAndEmitLazyCleanup(CGF, Fn, /*ForEH*/ true, EHEntry);
    CGF.Builder.restoreIP(SavedIP);
  }
}

/// Pops a cleanup block.  If the block includes a normal cleanup, the
/// current insertion point is threaded through the cleanup, as are
/// any branch fixups on the cleanup.
void CodeGenFunction::PopCleanupBlock() {
  assert(!EHStack.empty() && "cleanup stack is empty!");
  if (isa<EHLazyCleanupScope>(*EHStack.begin()))
    return PopLazyCleanupBlock(*this);

  assert(isa<EHCleanupScope>(*EHStack.begin()) && "top not a cleanup!");
  EHCleanupScope &Scope = cast<EHCleanupScope>(*EHStack.begin());
  assert(Scope.getFixupDepth() <= EHStack.getNumBranchFixups());

  // Handle the EH cleanup if (1) there is one and (2) it's different
  // from the normal cleanup.
  if (Scope.isEHCleanup() &&
      Scope.getEHEntry() != Scope.getNormalEntry()) {
    llvm::BasicBlock *EHEntry = Scope.getEHEntry();
    llvm::BasicBlock *EHExit = Scope.getEHExit();
    
    if (EHEntry->use_empty()) {
      DestroyCleanup(*this, EHEntry, EHExit);
    } else {
      // TODO: this isn't really the ideal location to put this EH
      // cleanup, but lazy emission is a better solution than trying
      // to pick a better spot.
      CGBuilderTy::InsertPoint SavedIP = Builder.saveAndClearIP();
      EmitBlock(EHEntry);
      Builder.restoreIP(SavedIP);

      SimplifyCleanupEdges(*this, EHEntry, EHExit);
    }
  }

  // If we only have an EH cleanup, we don't really need to do much
  // here.  Branch fixups just naturally drop down to the enclosing
  // cleanup scope.
  if (!Scope.isNormalCleanup()) {
    EHStack.popCleanup();
    assert(EHStack.getNumBranchFixups() == 0 || EHStack.hasNormalCleanups());
    return;
  }

  // Check whether the scope has any fixups that need to be threaded.
  unsigned FixupDepth = Scope.getFixupDepth();
  bool HasFixups = EHStack.getNumBranchFixups() != FixupDepth;

  // Grab the entry and exit blocks.
  llvm::BasicBlock *Entry = Scope.getNormalEntry();
  llvm::BasicBlock *Exit = Scope.getNormalExit();

  // Check whether anything's been threaded through the cleanup already.
  assert((Exit->getTerminator() == 0) == Entry->use_empty() &&
         "cleanup entry/exit mismatch");
  bool HasExistingBranches = !Entry->use_empty();

  // Check whether we need to emit a "fallthrough" branch through the
  // cleanup for the current insertion point.
  llvm::BasicBlock *FallThrough = Builder.GetInsertBlock();
  if (FallThrough && FallThrough->getTerminator())
    FallThrough = 0;

  // If *nothing* is using the cleanup, kill it.
  if (!FallThrough && !HasFixups && !HasExistingBranches) {
    EHStack.popCleanup();
    DestroyCleanup(*this, Entry, Exit);
    return;
  }

  // Otherwise, add the block to the function.
  EmitBlock(Entry);

  if (FallThrough)
    Builder.SetInsertPoint(Exit);
  else
    Builder.ClearInsertionPoint();

  // Fast case: if we don't have to add any fixups, and either
  // we don't have a fallthrough or the cleanup wasn't previously
  // used, then the setup above is sufficient.
  if (!HasFixups) {
    if (!FallThrough) {
      assert(HasExistingBranches && "no reason for cleanup but didn't kill before");
      EHStack.popCleanup();
      SimplifyCleanupEdges(*this, Entry, Exit);
      return;
    } else if (!HasExistingBranches) {
      assert(FallThrough && "no reason for cleanup but didn't kill before");
      // We can't simplify the exit edge in this case because we're
      // already inserting at the end of the exit block.
      EHStack.popCleanup();
      SimplifyCleanupEntry(*this, Entry);
      return;
    }
  }

  // Otherwise we're going to have to thread things through the cleanup.
  llvm::SmallVector<BranchFixup*, 8> Fixups;

  // Synthesize a fixup for the current insertion point.
  BranchFixup Cur;
  if (FallThrough) {
    Cur.Destination = createBasicBlock("cleanup.cont");
    Cur.LatestBranch = FallThrough->getTerminator();
    Cur.LatestBranchIndex = 0;
    Cur.Origin = Cur.LatestBranch;

    // Restore fixup invariant.  EmitBlock added a branch to the cleanup
    // which we need to redirect to the destination.
    cast<llvm::BranchInst>(Cur.LatestBranch)->setSuccessor(0, Cur.Destination);

    Fixups.push_back(&Cur);
  } else {
    Cur.Destination = 0;
  }

  // Collect any "real" fixups we need to thread.
  for (unsigned I = FixupDepth, E = EHStack.getNumBranchFixups();
        I != E; ++I)
    if (EHStack.getBranchFixup(I).Destination)
      Fixups.push_back(&EHStack.getBranchFixup(I));

  assert(!Fixups.empty() && "no fixups, invariants broken!");

  // If there's only a single fixup to thread through, do so with
  // unconditional branches.  This only happens if there's a single
  // branch and no fallthrough.
  if (Fixups.size() == 1 && !HasExistingBranches) {
    Fixups[0]->LatestBranch->setSuccessor(Fixups[0]->LatestBranchIndex, Entry);
    llvm::BranchInst *Br =
      llvm::BranchInst::Create(Fixups[0]->Destination, Exit);
    Fixups[0]->LatestBranch = Br;
    Fixups[0]->LatestBranchIndex = 0;

  // Otherwise, force a switch statement and thread everything through
  // the switch.
  } else {
    CreateCleanupSwitch(*this, Exit);
    for (unsigned I = 0, E = Fixups.size(); I != E; ++I)
      ThreadFixupThroughCleanup(*this, *Fixups[I], Entry, Exit);
  }

  // Emit the fallthrough destination block if necessary.
  if (Cur.Destination)
    EmitBlock(Cur.Destination);

  // We're finally done with the cleanup.
  EHStack.popCleanup();
}

void CodeGenFunction::EmitBranchThroughCleanup(JumpDest Dest) {
  if (!HaveInsertPoint())
    return;

  // Create the branch.
  llvm::BranchInst *BI = Builder.CreateBr(Dest.Block);

  // If we're not in a cleanup scope, we don't need to worry about
  // fixups.
  if (!EHStack.hasNormalCleanups()) {
    Builder.ClearInsertionPoint();
    return;
  }

  // Initialize a fixup.
  BranchFixup Fixup;
  Fixup.Destination = Dest.Block;
  Fixup.Origin = BI;
  Fixup.LatestBranch = BI;
  Fixup.LatestBranchIndex = 0;

  // If we can't resolve the destination cleanup scope, just add this
  // to the current cleanup scope.
  if (!Dest.ScopeDepth.isValid()) {
    EHStack.addBranchFixup() = Fixup;
    Builder.ClearInsertionPoint();
    return;
  }

  for (EHScopeStack::iterator I = EHStack.begin(),
         E = EHStack.find(Dest.ScopeDepth); I != E; ++I) {
    if (isa<EHCleanupScope>(*I)) {
      EHCleanupScope &Scope = cast<EHCleanupScope>(*I);
      if (Scope.isNormalCleanup())
        ThreadFixupThroughCleanup(*this, Fixup, Scope.getNormalEntry(),
                                  Scope.getNormalExit());
    } else if (isa<EHLazyCleanupScope>(*I)) {
      EHLazyCleanupScope &Scope = cast<EHLazyCleanupScope>(*I);
      if (Scope.isNormalCleanup()) {
        llvm::BasicBlock *Block = Scope.getNormalBlock();
        if (!Block) {
          Block = createBasicBlock("cleanup");
          Scope.setNormalBlock(Block);
        }
        ThreadFixupThroughCleanup(*this, Fixup, Block, Block);
      }
    }
  }
  
  Builder.ClearInsertionPoint();
}

void CodeGenFunction::EmitBranchThroughEHCleanup(JumpDest Dest) {
  if (!HaveInsertPoint())
    return;

  // Create the branch.
  llvm::BranchInst *BI = Builder.CreateBr(Dest.Block);

  // If we're not in a cleanup scope, we don't need to worry about
  // fixups.
  if (!EHStack.hasEHCleanups()) {
    Builder.ClearInsertionPoint();
    return;
  }

  // Initialize a fixup.
  BranchFixup Fixup;
  Fixup.Destination = Dest.Block;
  Fixup.Origin = BI;
  Fixup.LatestBranch = BI;
  Fixup.LatestBranchIndex = 0;

  // We should never get invalid scope depths for these: invalid scope
  // depths only arise for as-yet-unemitted labels, and we can't do an
  // EH-unwind to one of those.
  assert(Dest.ScopeDepth.isValid() && "invalid scope depth on EH dest?");

  for (EHScopeStack::iterator I = EHStack.begin(),
         E = EHStack.find(Dest.ScopeDepth); I != E; ++I) {
    if (isa<EHCleanupScope>(*I)) {
      EHCleanupScope &Scope = cast<EHCleanupScope>(*I);
      if (Scope.isEHCleanup())
        ThreadFixupThroughCleanup(*this, Fixup, Scope.getEHEntry(),
                                  Scope.getEHExit());
    } else if (isa<EHLazyCleanupScope>(*I)) {
      EHLazyCleanupScope &Scope = cast<EHLazyCleanupScope>(*I);
      if (Scope.isEHCleanup()) {
        llvm::BasicBlock *Block = Scope.getEHBlock();
        if (!Block) {
          Block = createBasicBlock("eh.cleanup");
          Scope.setEHBlock(Block);
        }
        ThreadFixupThroughCleanup(*this, Fixup, Block, Block);
      }
    }
  }
  
  Builder.ClearInsertionPoint();
}
