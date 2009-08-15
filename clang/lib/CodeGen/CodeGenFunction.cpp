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
#include "clang/Basic/TargetInfo.h"
#include "clang/AST/APValue.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "llvm/Target/TargetData.h"
using namespace clang;
using namespace CodeGen;

CodeGenFunction::CodeGenFunction(CodeGenModule &cgm) 
  : BlockFunction(cgm, *this, Builder), CGM(cgm),
    Target(CGM.getContext().Target),
    Builder(cgm.getModule().getContext()),
    DebugInfo(0), SwitchInsn(0), CaseRangeBlock(0), InvokeDest(0), 
    CXXThisDecl(0) {
  LLVMIntTy = ConvertType(getContext().IntTy);
  LLVMPointerWidth = Target.getPointerWidth(0);
}

ASTContext &CodeGenFunction::getContext() const {
  return CGM.getContext();
}


llvm::BasicBlock *CodeGenFunction::getBasicBlockForLabel(const LabelStmt *S) {
  llvm::BasicBlock *&BB = LabelMap[S];
  if (BB) return BB;
  
  // Create, but don't insert, the new block.
  return BB = createBasicBlock(S->getName());
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
  // FIXME: Use positive checks instead of negative ones to be more robust in
  // the face of extension.
  return !T->hasPointerRepresentation() && !T->isRealType() &&
    !T->isVoidType() && !T->isVectorType() && !T->isFunctionType() && 
    !T->isBlockPointerType() && !T->isMemberPointerType();
}

void CodeGenFunction::EmitReturnBlock() {
  // For cleanliness, we try to avoid emitting the return block for
  // simple cases.
  llvm::BasicBlock *CurBB = Builder.GetInsertBlock();

  if (CurBB) {
    assert(!CurBB->getTerminator() && "Unexpected terminated block.");

    // We have a valid insert point, reuse it if it is empty or there are no
    // explicit jumps to the return block.
    if (CurBB->empty() || ReturnBlock->use_empty()) {
      ReturnBlock->replaceAllUsesWith(CurBB);
      delete ReturnBlock;
    } else
      EmitBlock(ReturnBlock);
    return;
  }

  // Otherwise, if the return block is the target of a single direct
  // branch then we can just put the code in that block instead. This
  // cleans up functions which started with a unified return block.
  if (ReturnBlock->hasOneUse()) {
    llvm::BranchInst *BI = 
      dyn_cast<llvm::BranchInst>(*ReturnBlock->use_begin());
    if (BI && BI->isUnconditional() && BI->getSuccessor(0) == ReturnBlock) {
      // Reset insertion point and delete the branch.
      Builder.SetInsertPoint(BI->getParent());
      BI->eraseFromParent();
      delete ReturnBlock;
      return;
    }
  }

  // FIXME: We are at an unreachable point, there is no reason to emit the block
  // unless it has uses. However, we still need a place to put the debug
  // region.end for now.

  EmitBlock(ReturnBlock);
}

void CodeGenFunction::FinishFunction(SourceLocation EndLoc) {
  // Finish emission of indirect switches.
  EmitIndirectSwitches();

  assert(BreakContinueStack.empty() &&
         "mismatched push/pop in break/continue stack!");
  assert(BlockScopes.empty() &&
         "did not remove all blocks from block scope map!");
  assert(CleanupEntries.empty() &&
         "mismatched push/pop in cleanup stack!");
  
  // Emit function epilog (to return). 
  EmitReturnBlock();

  // Emit debug descriptor for function end.
  if (CGDebugInfo *DI = getDebugInfo()) {
    DI->setLocation(EndLoc);
    DI->EmitRegionEnd(CurFn, Builder);
  }

  EmitFunctionEpilog(*CurFnInfo, ReturnValue);

  // Remove the AllocaInsertPt instruction, which is just a convenience for us.
  llvm::Instruction *Ptr = AllocaInsertPt;
  AllocaInsertPt = 0;
  Ptr->eraseFromParent();
}

void CodeGenFunction::StartFunction(const Decl *D, QualType RetTy, 
                                    llvm::Function *Fn,
                                    const FunctionArgList &Args,
                                    SourceLocation StartLoc) {
  DidCallStackSave = false;
  CurCodeDecl = CurFuncDecl = D;
  FnRetTy = RetTy;
  CurFn = Fn;
  assert(CurFn->isDeclaration() && "Function already has body?");

  llvm::BasicBlock *EntryBB = createBasicBlock("entry", CurFn);

  // Create a marker to make it easy to insert allocas into the entryblock
  // later.  Don't create this with the builder, because we don't want it
  // folded.
  llvm::Value *Undef = llvm::UndefValue::get(llvm::Type::getInt32Ty(VMContext));
  AllocaInsertPt = new llvm::BitCastInst(Undef, llvm::Type::getInt32Ty(VMContext), "",
                                         EntryBB);
  if (Builder.isNamePreserving())
    AllocaInsertPt->setName("allocapt");
  
  ReturnBlock = createBasicBlock("return");
  ReturnValue = 0;
  if (!RetTy->isVoidType())
    ReturnValue = CreateTempAlloca(ConvertType(RetTy), "retval");
    
  Builder.SetInsertPoint(EntryBB);
  
  // Emit subprogram debug descriptor.
  // FIXME: The cast here is a huge hack.
  if (CGDebugInfo *DI = getDebugInfo()) {
    DI->setLocation(StartLoc);
    if (const FunctionDecl *FD = dyn_cast_or_null<FunctionDecl>(D)) {
      DI->EmitFunctionStart(CGM.getMangledName(FD), RetTy, CurFn, Builder);
    } else {
      // Just use LLVM function name.
      
      // FIXME: Remove unnecessary conversion to std::string when API settles.
      DI->EmitFunctionStart(std::string(Fn->getName()).c_str(), 
                            RetTy, CurFn, Builder);
    }
  }

  // FIXME: Leaked.
  CurFnInfo = &CGM.getTypes().getFunctionInfo(FnRetTy, Args);
  EmitFunctionProlog(*CurFnInfo, CurFn, Args);
  
  // If any of the arguments have a variably modified type, make sure to
  // emit the type size.
  for (FunctionArgList::const_iterator i = Args.begin(), e = Args.end();
       i != e; ++i) {
    QualType Ty = i->second;

    if (Ty->isVariablyModifiedType())
      EmitVLASize(Ty);
  }
}

void CodeGenFunction::GenerateCode(const FunctionDecl *FD,
                                   llvm::Function *Fn) {
  // Check if we should generate debug info for this function.
  if (CGM.getDebugInfo() && !FD->hasAttr<NodebugAttr>())
    DebugInfo = CGM.getDebugInfo();
  
  FunctionArgList Args;
  
  if (const CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(FD)) {
    if (MD->isInstance()) {
      // Create the implicit 'this' decl.
      // FIXME: I'm not entirely sure I like using a fake decl just for code
      // generation. Maybe we can come up with a better way?
      CXXThisDecl = ImplicitParamDecl::Create(getContext(), 0, SourceLocation(),
                                              &getContext().Idents.get("this"), 
                                              MD->getThisType(getContext()));
      Args.push_back(std::make_pair(CXXThisDecl, CXXThisDecl->getType()));
    }
  }
  
  if (FD->getNumParams()) {
    const FunctionProtoType* FProto = FD->getType()->getAsFunctionProtoType();
    assert(FProto && "Function def must have prototype!");

    for (unsigned i = 0, e = FD->getNumParams(); i != e; ++i)
      Args.push_back(std::make_pair(FD->getParamDecl(i), 
                                    FProto->getArgType(i)));
  }

  // FIXME: Support CXXTryStmt here, too.
  if (const CompoundStmt *S = FD->getCompoundBody()) {
    StartFunction(FD, FD->getResultType(), Fn, Args, S->getLBracLoc());
    if (const CXXConstructorDecl *CD = dyn_cast<CXXConstructorDecl>(FD))
      EmitCtorPrologue(CD);
    EmitStmt(S);
    if (const CXXDestructorDecl *DD = dyn_cast<CXXDestructorDecl>(FD))
      EmitDtorEpilogue(DD);
    FinishFunction(S->getRBracLoc());
  }
  else 
    if (const CXXConstructorDecl *CD = dyn_cast<CXXConstructorDecl>(FD)) {
      const CXXRecordDecl *ClassDecl = 
        cast<CXXRecordDecl>(CD->getDeclContext());
      (void) ClassDecl;
      if (CD->isCopyConstructor(getContext())) {
        assert(!ClassDecl->hasUserDeclaredCopyConstructor() &&
               "bogus constructor is being synthesize");
        SynthesizeCXXCopyConstructor(CD, FD, Fn, Args);
      }
      else {
        assert(!ClassDecl->hasUserDeclaredConstructor() &&
               "bogus constructor is being synthesize");
        SynthesizeDefaultConstructor(CD, FD, Fn, Args);
      }
    }
  else if (const CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(FD))
         if (MD->isCopyAssignment())
           SynthesizeCXXCopyAssignment(MD, FD, Fn, Args);
    
  // Destroy the 'this' declaration.
  if (CXXThisDecl)
    CXXThisDecl->Destroy(getContext());
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
      
      EmitBranchOnBoolExpr(CondBOp->getRHS(), TrueBlock, FalseBlock);
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
      
      EmitBranchOnBoolExpr(CondBOp->getRHS(), TrueBlock, FalseBlock);
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

/// getCGRecordLayout - Return record layout info.
const CGRecordLayout *CodeGenFunction::getCGRecordLayout(CodeGenTypes &CGT,
                                                         QualType Ty) {
  const RecordType *RTy = Ty->getAs<RecordType>();
  assert (RTy && "Unexpected type. RecordType expected here.");

  return CGT.getCGRecordLayout(RTy->getDecl());
}

/// ErrorUnsupported - Print out an error that codegen doesn't support the
/// specified stmt yet.
void CodeGenFunction::ErrorUnsupported(const Stmt *S, const char *Type,
                                       bool OmitOnError) {
  CGM.ErrorUnsupported(S, Type, OmitOnError);
}

unsigned CodeGenFunction::GetIDForAddrOfLabel(const LabelStmt *L) {
  // Use LabelIDs.size() as the new ID if one hasn't been assigned.
  return LabelIDs.insert(std::make_pair(L, LabelIDs.size())).first->second;
}

void CodeGenFunction::EmitMemSetToZero(llvm::Value *DestPtr, QualType Ty) {
  const llvm::Type *BP = llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(VMContext));
  if (DestPtr->getType() != BP)
    DestPtr = Builder.CreateBitCast(DestPtr, BP, "tmp");

  // Get size and alignment info for this aggregate.
  std::pair<uint64_t, unsigned> TypeInfo = getContext().getTypeInfo(Ty);

  // Don't bother emitting a zero-byte memset.
  if (TypeInfo.first == 0)
    return;
  
  // FIXME: Handle variable sized types.
  const llvm::Type *IntPtr = llvm::IntegerType::get(VMContext, 
                                                    LLVMPointerWidth);

  Builder.CreateCall4(CGM.getMemSetFn(), DestPtr,
                 llvm::Constant::getNullValue(llvm::Type::getInt8Ty(VMContext)),
                      // TypeInfo.first describes size in bits.
                      llvm::ConstantInt::get(IntPtr, TypeInfo.first/8),
                      llvm::ConstantInt::get(llvm::Type::getInt32Ty(VMContext), 
                                             TypeInfo.second/8));
}

void CodeGenFunction::EmitIndirectSwitches() {
  llvm::BasicBlock *Default;
  
  if (IndirectSwitches.empty())
    return;
  
  if (!LabelIDs.empty()) {
    Default = getBasicBlockForLabel(LabelIDs.begin()->first);
  } else {
    // No possible targets for indirect goto, just emit an infinite
    // loop.
    Default = createBasicBlock("indirectgoto.loop", CurFn);
    llvm::BranchInst::Create(Default, Default);
  }

  for (std::vector<llvm::SwitchInst*>::iterator i = IndirectSwitches.begin(),
         e = IndirectSwitches.end(); i != e; ++i) {
    llvm::SwitchInst *I = *i;
    
    I->setSuccessor(0, Default);
    for (std::map<const LabelStmt*,unsigned>::iterator LI = LabelIDs.begin(), 
           LE = LabelIDs.end(); LI != LE; ++LI) {
      I->addCase(llvm::ConstantInt::get(llvm::Type::getInt32Ty(VMContext),
                                        LI->second), 
                 getBasicBlockForLabel(LI->first));
    }
  }         
}

llvm::Value *CodeGenFunction::GetVLASize(const VariableArrayType *VAT) {
  llvm::Value *&SizeEntry = VLASizeMap[VAT];
  
  assert(SizeEntry && "Did not emit size for type");
  return SizeEntry;
}

llvm::Value *CodeGenFunction::EmitVLASize(QualType Ty) {
  assert(Ty->isVariablyModifiedType() &&
         "Must pass variably modified type to EmitVLASizes!");
  
  EnsureInsertPoint();
  
  if (const VariableArrayType *VAT = getContext().getAsVariableArrayType(Ty)) {
    llvm::Value *&SizeEntry = VLASizeMap[VAT];
    
    if (!SizeEntry) {
      const llvm::Type *SizeTy = ConvertType(getContext().getSizeType());
                                             
      // Get the element size;
      QualType ElemTy = VAT->getElementType();
      llvm::Value *ElemSize;
      if (ElemTy->isVariableArrayType())
        ElemSize = EmitVLASize(ElemTy);
      else
        ElemSize = llvm::ConstantInt::get(SizeTy,
                                          getContext().getTypeSize(ElemTy) / 8);
    
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
  if (CGM.getContext().getBuiltinVaListType()->isArrayType()) {
    return EmitScalarExpr(E);
  }
  return EmitLValue(E).getAddress();
}

void CodeGenFunction::PushCleanupBlock(llvm::BasicBlock *CleanupBlock)
{
  CleanupEntries.push_back(CleanupEntry(CleanupBlock));
}

void CodeGenFunction::EmitCleanupBlocks(size_t OldCleanupStackSize)
{
  assert(CleanupEntries.size() >= OldCleanupStackSize && 
         "Cleanup stack mismatch!");
  
  while (CleanupEntries.size() > OldCleanupStackSize)
    EmitCleanupBlock();
}

CodeGenFunction::CleanupBlockInfo CodeGenFunction::PopCleanupBlock()
{
  CleanupEntry &CE = CleanupEntries.back();
  
  llvm::BasicBlock *CleanupBlock = CE.CleanupBlock;
  
  std::vector<llvm::BasicBlock *> Blocks;
  std::swap(Blocks, CE.Blocks);
  
  std::vector<llvm::BranchInst *> BranchFixups;
  std::swap(BranchFixups, CE.BranchFixups);
  
  CleanupEntries.pop_back();

  // Check if any branch fixups pointed to the scope we just popped. If so,
  // we can remove them.
  for (size_t i = 0, e = BranchFixups.size(); i != e; ++i) {
    llvm::BasicBlock *Dest = BranchFixups[i]->getSuccessor(0);
    BlockScopeMap::iterator I = BlockScopes.find(Dest);
      
    if (I == BlockScopes.end())
      continue;
      
    assert(I->second <= CleanupEntries.size() && "Invalid branch fixup!");
      
    if (I->second == CleanupEntries.size()) {
      // We don't need to do this branch fixup.
      BranchFixups[i] = BranchFixups.back();
      BranchFixups.pop_back();
      i--;
      e--;
      continue;
    }
  }
  
  llvm::BasicBlock *SwitchBlock = 0;
  llvm::BasicBlock *EndBlock = 0;
  if (!BranchFixups.empty()) {
    SwitchBlock = createBasicBlock("cleanup.switch");
    EndBlock = createBasicBlock("cleanup.end");
    
    llvm::BasicBlock *CurBB = Builder.GetInsertBlock();
    
    Builder.SetInsertPoint(SwitchBlock);

    llvm::Value *DestCodePtr = CreateTempAlloca(llvm::Type::getInt32Ty(VMContext), 
                                                "cleanup.dst");
    llvm::Value *DestCode = Builder.CreateLoad(DestCodePtr, "tmp");
    
    // Create a switch instruction to determine where to jump next.
    llvm::SwitchInst *SI = Builder.CreateSwitch(DestCode, EndBlock, 
                                                BranchFixups.size());

    // Restore the current basic block (if any)
    if (CurBB) {
      Builder.SetInsertPoint(CurBB);
      
      // If we had a current basic block, we also need to emit an instruction
      // to initialize the cleanup destination.
      Builder.CreateStore(llvm::Constant::getNullValue(llvm::Type::getInt32Ty(VMContext)),
                          DestCodePtr);
    } else
      Builder.ClearInsertionPoint();

    for (size_t i = 0, e = BranchFixups.size(); i != e; ++i) {
      llvm::BranchInst *BI = BranchFixups[i];
      llvm::BasicBlock *Dest = BI->getSuccessor(0);
      
      // Fixup the branch instruction to point to the cleanup block.
      BI->setSuccessor(0, CleanupBlock);
      
      if (CleanupEntries.empty()) {
        llvm::ConstantInt *ID;
        
        // Check if we already have a destination for this block.
        if (Dest == SI->getDefaultDest())
          ID = llvm::ConstantInt::get(llvm::Type::getInt32Ty(VMContext), 0);
        else {
          ID = SI->findCaseDest(Dest);
          if (!ID) {
            // No code found, get a new unique one by using the number of
            // switch successors.
            ID = llvm::ConstantInt::get(llvm::Type::getInt32Ty(VMContext), 
                                        SI->getNumSuccessors());
            SI->addCase(ID, Dest);
          }
        }
        
        // Store the jump destination before the branch instruction.
        new llvm::StoreInst(ID, DestCodePtr, BI);
      } else {
        // We need to jump through another cleanup block. Create a pad block
        // with a branch instruction that jumps to the final destination and
        // add it as a branch fixup to the current cleanup scope.
        
        // Create the pad block.
        llvm::BasicBlock *CleanupPad = createBasicBlock("cleanup.pad", CurFn);

        // Create a unique case ID.
        llvm::ConstantInt *ID = llvm::ConstantInt::get(llvm::Type::getInt32Ty(VMContext), 
                                                       SI->getNumSuccessors());

        // Store the jump destination before the branch instruction.
        new llvm::StoreInst(ID, DestCodePtr, BI);

        // Add it as the destination.
        SI->addCase(ID, CleanupPad);
        
        // Create the branch to the final destination.
        llvm::BranchInst *BI = llvm::BranchInst::Create(Dest);
        CleanupPad->getInstList().push_back(BI);
        
        // And add it as a branch fixup.
        CleanupEntries.back().BranchFixups.push_back(BI);
      }
    }
  }
  
  // Remove all blocks from the block scope map.
  for (size_t i = 0, e = Blocks.size(); i != e; ++i) {
    assert(BlockScopes.count(Blocks[i]) &&
           "Did not find block in scope map!");
    
    BlockScopes.erase(Blocks[i]);
  }
  
  return CleanupBlockInfo(CleanupBlock, SwitchBlock, EndBlock);
}

void CodeGenFunction::EmitCleanupBlock()
{
  CleanupBlockInfo Info = PopCleanupBlock();
  
  llvm::BasicBlock *CurBB = Builder.GetInsertBlock();
  if (CurBB && !CurBB->getTerminator() && 
      Info.CleanupBlock->getNumUses() == 0) {
    CurBB->getInstList().splice(CurBB->end(), Info.CleanupBlock->getInstList());
    delete Info.CleanupBlock;
  } else 
    EmitBlock(Info.CleanupBlock);
  
  if (Info.SwitchBlock)
    EmitBlock(Info.SwitchBlock);
  if (Info.EndBlock)
    EmitBlock(Info.EndBlock);
}

void CodeGenFunction::AddBranchFixup(llvm::BranchInst *BI)
{
  assert(!CleanupEntries.empty() && 
         "Trying to add branch fixup without cleanup block!");
  
  // FIXME: We could be more clever here and check if there's already a branch
  // fixup for this destination and recycle it.
  CleanupEntries.back().BranchFixups.push_back(BI);
}

void CodeGenFunction::EmitBranchThroughCleanup(llvm::BasicBlock *Dest)
{
  if (!HaveInsertPoint())
    return;
  
  llvm::BranchInst* BI = Builder.CreateBr(Dest);
  
  Builder.ClearInsertionPoint();
  
  // The stack is empty, no need to do any cleanup.
  if (CleanupEntries.empty())
    return;
  
  if (!Dest->getParent()) {
    // We are trying to branch to a block that hasn't been inserted yet.
    AddBranchFixup(BI);
    return;
  }
  
  BlockScopeMap::iterator I = BlockScopes.find(Dest);
  if (I == BlockScopes.end()) {
    // We are trying to jump to a block that is outside of any cleanup scope.
    AddBranchFixup(BI);
    return;
  }
  
  assert(I->second < CleanupEntries.size() &&
         "Trying to branch into cleanup region");
  
  if (I->second == CleanupEntries.size() - 1) {
    // We have a branch to a block in the same scope.
    return;
  }
  
  AddBranchFixup(BI);
}
