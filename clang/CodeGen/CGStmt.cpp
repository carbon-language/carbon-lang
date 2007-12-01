//===--- CGStmt.cpp - Emit LLVM Code from Statements ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit Stmt nodes as LLVM code.
//
//===----------------------------------------------------------------------===//

#include "CodeGenFunction.h"
#include "clang/AST/AST.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
using namespace clang;
using namespace CodeGen;

//===----------------------------------------------------------------------===//
//                              Statement Emission
//===----------------------------------------------------------------------===//

void CodeGenFunction::EmitStmt(const Stmt *S) {
  assert(S && "Null statement?");
  
  switch (S->getStmtClass()) {
  default:
    // Must be an expression in a stmt context.  Emit the value (to get
    // side-effects) and ignore the result.
    if (const Expr *E = dyn_cast<Expr>(S)) {
      if (!hasAggregateLLVMType(E->getType()))
        EmitScalarExpr(E);
      else if (E->getType()->isComplexType())
        EmitComplexExpr(E);
      else
        EmitAggExpr(E, 0, false);
    } else {
      printf("Unimplemented stmt!\n");
      S->dump(getContext().SourceMgr);
    }
    break;
  case Stmt::NullStmtClass: break;
  case Stmt::CompoundStmtClass: EmitCompoundStmt(cast<CompoundStmt>(*S)); break;
  case Stmt::LabelStmtClass:    EmitLabelStmt(cast<LabelStmt>(*S));       break;
  case Stmt::GotoStmtClass:     EmitGotoStmt(cast<GotoStmt>(*S));         break;

  case Stmt::IfStmtClass:       EmitIfStmt(cast<IfStmt>(*S));             break;
  case Stmt::WhileStmtClass:    EmitWhileStmt(cast<WhileStmt>(*S));       break;
  case Stmt::DoStmtClass:       EmitDoStmt(cast<DoStmt>(*S));             break;
  case Stmt::ForStmtClass:      EmitForStmt(cast<ForStmt>(*S));           break;
    
  case Stmt::ReturnStmtClass:   EmitReturnStmt(cast<ReturnStmt>(*S));     break;
  case Stmt::DeclStmtClass:     EmitDeclStmt(cast<DeclStmt>(*S));         break;
      
  case Stmt::BreakStmtClass:    EmitBreakStmt();                          break;
  case Stmt::ContinueStmtClass: EmitContinueStmt();                       break;
  case Stmt::SwitchStmtClass:   EmitSwitchStmt(cast<SwitchStmt>(*S));     break;
  case Stmt::DefaultStmtClass:  EmitDefaultStmt(cast<DefaultStmt>(*S));   break;
  case Stmt::CaseStmtClass:     EmitCaseStmt(cast<CaseStmt>(*S));         break;
  }
}

/// EmitCompoundStmt - Emit a compound statement {..} node.  If GetLast is true,
/// this captures the expression result of the last sub-statement and returns it
/// (for use by the statement expression extension).
RValue CodeGenFunction::EmitCompoundStmt(const CompoundStmt &S, bool GetLast,
                                         llvm::Value *AggLoc, bool isAggVol) {
  // FIXME: handle vla's etc.
  if (S.body_empty() || !isa<Expr>(S.body_back())) GetLast = false;
  
  for (CompoundStmt::const_body_iterator I = S.body_begin(),
       E = S.body_end()-GetLast; I != E; ++I)
    EmitStmt(*I);
  
  
  if (!GetLast)
    return RValue::get(0);
  
  return EmitAnyExpr(cast<Expr>(S.body_back()), AggLoc);
}

void CodeGenFunction::EmitBlock(llvm::BasicBlock *BB) {
  // Emit a branch from this block to the next one if this was a real block.  If
  // this was just a fall-through block after a terminator, don't emit it.
  llvm::BasicBlock *LastBB = Builder.GetInsertBlock();
  
  if (LastBB->getTerminator()) {
    // If the previous block is already terminated, don't touch it.
  } else if (LastBB->empty() && LastBB->getValueName() == 0) {
    // If the last block was an empty placeholder, remove it now.
    // TODO: cache and reuse these.
    Builder.GetInsertBlock()->eraseFromParent();
  } else {
    // Otherwise, create a fall-through branch.
    Builder.CreateBr(BB);
  }
  CurFn->getBasicBlockList().push_back(BB);
  Builder.SetInsertPoint(BB);
}

void CodeGenFunction::EmitLabelStmt(const LabelStmt &S) {
  llvm::BasicBlock *NextBB = getBasicBlockForLabel(&S);
  
  EmitBlock(NextBB);
  EmitStmt(S.getSubStmt());
}

void CodeGenFunction::EmitGotoStmt(const GotoStmt &S) {
  Builder.CreateBr(getBasicBlockForLabel(S.getLabel()));
  
  // Emit a block after the branch so that dead code after a goto has some place
  // to go.
  Builder.SetInsertPoint(new llvm::BasicBlock("", CurFn));
}

void CodeGenFunction::EmitIfStmt(const IfStmt &S) {
  // C99 6.8.4.1: The first substatement is executed if the expression compares
  // unequal to 0.  The condition must be a scalar type.
  llvm::Value *BoolCondVal = EvaluateExprAsBool(S.getCond());
  
  llvm::BasicBlock *ContBlock = new llvm::BasicBlock("ifend");
  llvm::BasicBlock *ThenBlock = new llvm::BasicBlock("ifthen");
  llvm::BasicBlock *ElseBlock = ContBlock;
  
  if (S.getElse())
    ElseBlock = new llvm::BasicBlock("ifelse");
  
  // Insert the conditional branch.
  Builder.CreateCondBr(BoolCondVal, ThenBlock, ElseBlock);
  
  // Emit the 'then' code.
  EmitBlock(ThenBlock);
  EmitStmt(S.getThen());
  llvm::BasicBlock *BB = Builder.GetInsertBlock();
  if (isDummyBlock(BB)) {
    BB->eraseFromParent();
    Builder.SetInsertPoint(ThenBlock);
  }
  else
    Builder.CreateBr(ContBlock);
  
  // Emit the 'else' code if present.
  if (const Stmt *Else = S.getElse()) {
    EmitBlock(ElseBlock);
    EmitStmt(Else);
    llvm::BasicBlock *BB = Builder.GetInsertBlock();
    if (isDummyBlock(BB)) {
      BB->eraseFromParent();
      Builder.SetInsertPoint(ElseBlock);
    }
    else
      Builder.CreateBr(ContBlock);
  }
  
  // Emit the continuation block for code after the if.
  EmitBlock(ContBlock);
}

void CodeGenFunction::EmitWhileStmt(const WhileStmt &S) {
  // Emit the header for the loop, insert it, which will create an uncond br to
  // it.
  llvm::BasicBlock *LoopHeader = new llvm::BasicBlock("whilecond");
  EmitBlock(LoopHeader);
  
  // Evaluate the conditional in the while header.  C99 6.8.5.1: The evaluation
  // of the controlling expression takes place before each execution of the loop
  // body. 
  llvm::Value *BoolCondVal = EvaluateExprAsBool(S.getCond());

  // while(1) is common, avoid extra exit blocks.  Be sure
  // to correctly handle break/continue though.
  bool EmitBoolCondBranch = true;
  if (llvm::ConstantInt *C = dyn_cast<llvm::ConstantInt>(BoolCondVal)) 
    if (C->isOne())
      EmitBoolCondBranch = false;
  
  // Create an exit block for when the condition fails, create a block for the
  // body of the loop.
  llvm::BasicBlock *ExitBlock = new llvm::BasicBlock("whileexit");
  llvm::BasicBlock *LoopBody  = new llvm::BasicBlock("whilebody");
  
  // As long as the condition is true, go to the loop body.
  if (EmitBoolCondBranch)
    Builder.CreateCondBr(BoolCondVal, LoopBody, ExitBlock);

  // Store the blocks to use for break and continue.
  BreakContinueStack.push_back(BreakContinue(ExitBlock, LoopHeader));
  
  // Emit the loop body.
  EmitBlock(LoopBody);
  EmitStmt(S.getBody());

  BreakContinueStack.pop_back();
  
  // Cycle to the condition.
  Builder.CreateBr(LoopHeader);
  
  // Emit the exit block.
  EmitBlock(ExitBlock);

  // If LoopHeader is a simple forwarding block then eliminate it.
  if (!EmitBoolCondBranch 
      && &LoopHeader->front() == LoopHeader->getTerminator()) {
    LoopHeader->replaceAllUsesWith(LoopBody);
    LoopHeader->getTerminator()->eraseFromParent();
    LoopHeader->eraseFromParent();
  }
}

void CodeGenFunction::EmitDoStmt(const DoStmt &S) {
  // Emit the body for the loop, insert it, which will create an uncond br to
  // it.
  llvm::BasicBlock *LoopBody = new llvm::BasicBlock("dobody");
  llvm::BasicBlock *AfterDo = new llvm::BasicBlock("afterdo");
  EmitBlock(LoopBody);

  llvm::BasicBlock *DoCond = new llvm::BasicBlock("docond");
  
  // Store the blocks to use for break and continue.
  BreakContinueStack.push_back(BreakContinue(AfterDo, DoCond));
  
  // Emit the body of the loop into the block.
  EmitStmt(S.getBody());
  
  BreakContinueStack.pop_back();
  
  EmitBlock(DoCond);
  
  // C99 6.8.5.2: "The evaluation of the controlling expression takes place
  // after each execution of the loop body."
  
  // Evaluate the conditional in the while header.
  // C99 6.8.5p2/p4: The first substatement is executed if the expression
  // compares unequal to 0.  The condition must be a scalar type.
  llvm::Value *BoolCondVal = EvaluateExprAsBool(S.getCond());

  // "do {} while (0)" is common in macros, avoid extra blocks.  Be sure
  // to correctly handle break/continue though.
  bool EmitBoolCondBranch = true;
  if (llvm::ConstantInt *C = dyn_cast<llvm::ConstantInt>(BoolCondVal)) 
    if (C->isZero())
      EmitBoolCondBranch = false;

  // As long as the condition is true, iterate the loop.
  if (EmitBoolCondBranch)
    Builder.CreateCondBr(BoolCondVal, LoopBody, AfterDo);
  
  // Emit the exit block.
  EmitBlock(AfterDo);

  // If DoCond is a simple forwarding block then eliminate it.
  if (!EmitBoolCondBranch && &DoCond->front() == DoCond->getTerminator()) {
    DoCond->replaceAllUsesWith(AfterDo);
    DoCond->getTerminator()->eraseFromParent();
    DoCond->eraseFromParent();
  }
}

void CodeGenFunction::EmitForStmt(const ForStmt &S) {
  // FIXME: What do we do if the increment (f.e.) contains a stmt expression,
  // which contains a continue/break?
  // TODO: We could keep track of whether the loop body contains any
  // break/continue statements and not create unnecessary blocks (like
  // "afterfor" for a condless loop) if it doesn't.

  // Evaluate the first part before the loop.
  if (S.getInit())
    EmitStmt(S.getInit());

  // Start the loop with a block that tests the condition.
  llvm::BasicBlock *CondBlock = new llvm::BasicBlock("forcond");
  llvm::BasicBlock *AfterFor = new llvm::BasicBlock("afterfor");

  EmitBlock(CondBlock);

  // Evaluate the condition if present.  If not, treat it as a non-zero-constant
  // according to 6.8.5.3p2, aka, true.
  if (S.getCond()) {
    // C99 6.8.5p2/p4: The first substatement is executed if the expression
    // compares unequal to 0.  The condition must be a scalar type.
    llvm::Value *BoolCondVal = EvaluateExprAsBool(S.getCond());
    
    // As long as the condition is true, iterate the loop.
    llvm::BasicBlock *ForBody = new llvm::BasicBlock("forbody");
    Builder.CreateCondBr(BoolCondVal, ForBody, AfterFor);
    EmitBlock(ForBody);    
  } else {
    // Treat it as a non-zero constant.  Don't even create a new block for the
    // body, just fall into it.
  }

  // If the for loop doesn't have an increment we can just use the 
  // condition as the continue block.
  llvm::BasicBlock *ContinueBlock;
  if (S.getInc())
    ContinueBlock = new llvm::BasicBlock("forinc");
  else
    ContinueBlock = CondBlock;  
  
  // Store the blocks to use for break and continue.
  BreakContinueStack.push_back(BreakContinue(AfterFor, ContinueBlock));
  
  // If the condition is true, execute the body of the for stmt.
  EmitStmt(S.getBody());

  BreakContinueStack.pop_back();
  
  if (S.getInc())
    EmitBlock(ContinueBlock);
  
  // If there is an increment, emit it next.
  if (S.getInc())
    EmitStmt(S.getInc());
      
  // Finally, branch back up to the condition for the next iteration.
  Builder.CreateBr(CondBlock);

  // Emit the fall-through block.
  EmitBlock(AfterFor);
}

/// EmitReturnStmt - Note that due to GCC extensions, this can have an operand
/// if the function returns void, or may be missing one if the function returns
/// non-void.  Fun stuff :).
void CodeGenFunction::EmitReturnStmt(const ReturnStmt &S) {
  // Emit the result value, even if unused, to evalute the side effects.
  const Expr *RV = S.getRetValue();

  QualType FnRetTy = CurFuncDecl->getType().getCanonicalType();
  FnRetTy = cast<FunctionType>(FnRetTy)->getResultType();
  
  if (FnRetTy->isVoidType()) {
    // If the function returns void, emit ret void.
    Builder.CreateRetVoid();
  } else if (RV == 0) {
    // Handle "return;" in a function that returns a value.
    const llvm::Type *RetTy = CurFn->getFunctionType()->getReturnType();
    if (RetTy == llvm::Type::VoidTy)
      Builder.CreateRetVoid();   // struct return etc.
    else
      Builder.CreateRet(llvm::UndefValue::get(RetTy));
  } else if (!hasAggregateLLVMType(RV->getType())) {
    Builder.CreateRet(EmitScalarExpr(RV));
  } else if (RV->getType()->isComplexType()) {
    llvm::Value *SRetPtr = CurFn->arg_begin();
    EmitComplexExprIntoAddr(RV, SRetPtr, false);
  } else {
    llvm::Value *SRetPtr = CurFn->arg_begin();
    EmitAggExpr(RV, SRetPtr, false);
  }
  
  // Emit a block after the branch so that dead code after a return has some
  // place to go.
  EmitBlock(new llvm::BasicBlock());
}

void CodeGenFunction::EmitDeclStmt(const DeclStmt &S) {
  for (const ScopedDecl *Decl = S.getDecl(); Decl; 
       Decl = Decl->getNextDeclarator())
    EmitDecl(*Decl);
}

void CodeGenFunction::EmitBreakStmt() {
  assert(!BreakContinueStack.empty() && "break stmt not in a loop or switch!");

  llvm::BasicBlock *Block = BreakContinueStack.back().BreakBlock;
  Builder.CreateBr(Block);
  EmitBlock(new llvm::BasicBlock());
}

void CodeGenFunction::EmitContinueStmt() {
  assert(!BreakContinueStack.empty() && "continue stmt not in a loop!");

  llvm::BasicBlock *Block = BreakContinueStack.back().ContinueBlock;
  Builder.CreateBr(Block);
  EmitBlock(new llvm::BasicBlock());
}

/// EmitCaseStmtRange - If case statement range is not too big then
/// add multiple cases to switch instruction, one for each value within
/// the range. If range is too big then emit "if" condition check.
void CodeGenFunction::EmitCaseStmtRange(const CaseStmt &S) {
  assert (S.getRHS() && "Unexpected RHS value in CaseStmt");

  const Expr *L = S.getLHS();
  const Expr *R = S.getRHS();
  llvm::ConstantInt *LV = cast<llvm::ConstantInt>(EmitScalarExpr(L));
  llvm::ConstantInt *RV = cast<llvm::ConstantInt>(EmitScalarExpr(R));
  llvm::APInt LHS = LV->getValue();
  const llvm::APInt &RHS = RV->getValue();

  llvm::APInt Range = RHS - LHS;
  if (Range.ult(llvm::APInt(Range.getBitWidth(), 64))) {
    // Range is small enough to add multiple switch instruction cases.
    StartBlock("sw.bb");
    llvm::BasicBlock *CaseDest = Builder.GetInsertBlock();
    SwitchInsn->addCase(LV, CaseDest);
    LHS++;
    while (LHS != RHS) {
      SwitchInsn->addCase(llvm::ConstantInt::get(LHS), CaseDest);
      LHS++;
    }
    SwitchInsn->addCase(RV, CaseDest);
    EmitStmt(S.getSubStmt());
    return;
  } 
    
  // The range is too big. Emit "if" condition.
  llvm::BasicBlock *FalseDest = NULL;
  llvm::BasicBlock *CaseDest = new llvm::BasicBlock("sw.bb");

  // If we have already seen one case statement range for this switch
  // instruction then piggy-back otherwise use default block as false
  // destination.
  if (CaseRangeBlock)
    FalseDest = CaseRangeBlock;
  else 
    FalseDest = SwitchInsn->getDefaultDest();

  // Start new block to hold case statement range check instructions.
  StartBlock("case.range");
  CaseRangeBlock = Builder.GetInsertBlock();

  // Emit range check.
  llvm::Value *Diff = 
    Builder.CreateSub(SwitchInsn->getCondition(), LV, "tmp");
  llvm::Value *Cond = 
    Builder.CreateICmpULE(Diff, llvm::ConstantInt::get(Range), "tmp");
  Builder.CreateCondBr(Cond, CaseDest, FalseDest);

  // Now emit case statement body.
  EmitBlock(CaseDest);
  EmitStmt(S.getSubStmt());
}

void CodeGenFunction::EmitCaseStmt(const CaseStmt &S) {
  if (S.getRHS()) {
    EmitCaseStmtRange(S);
    return;
  }
    
  StartBlock("sw.bb");
  llvm::BasicBlock *CaseDest = Builder.GetInsertBlock();
  llvm::APSInt CaseVal(32);
  S.getLHS()->isIntegerConstantExpr(CaseVal, getContext());
  llvm::ConstantInt *LV = llvm::ConstantInt::get(CaseVal);
  SwitchInsn->addCase(LV, CaseDest);
  EmitStmt(S.getSubStmt());
}

void CodeGenFunction::EmitDefaultStmt(const DefaultStmt &S) {
  StartBlock("sw.default");
  // Current insert block is the default destination.
  SwitchInsn->setSuccessor(0, Builder.GetInsertBlock());
  EmitStmt(S.getSubStmt());
}

void CodeGenFunction::EmitSwitchStmt(const SwitchStmt &S) {
  llvm::Value *CondV = EmitScalarExpr(S.getCond());

  // Handle nested switch statements.
  llvm::SwitchInst *SavedSwitchInsn = SwitchInsn;
  llvm::BasicBlock *SavedCRBlock = CaseRangeBlock;
  CaseRangeBlock = NULL;

  // Create basic block to hold stuff that comes after switch statement.
  // Initially use it to hold DefaultStmt.
  llvm::BasicBlock *NextBlock = new llvm::BasicBlock("after.sw");
  SwitchInsn = Builder.CreateSwitch(CondV, NextBlock);

  // All break statements jump to NextBlock. If BreakContinueStack is non empty
  // then reuse last ContinueBlock.
  llvm::BasicBlock *ContinueBlock = NULL;
  if (!BreakContinueStack.empty())
    ContinueBlock = BreakContinueStack.back().ContinueBlock;
  BreakContinueStack.push_back(BreakContinue(NextBlock, ContinueBlock));

  // Emit switch body.
  EmitStmt(S.getBody());
  BreakContinueStack.pop_back();

  // If one or more case statement range is seen then use CaseRangeBlock
  // as the default block. False edge of CaseRangeBlock will lead to 
  // original default block.
  if (CaseRangeBlock)
    SwitchInsn->setSuccessor(0, CaseRangeBlock);
  
  // Prune insert block if it is dummy.
  llvm::BasicBlock *BB = Builder.GetInsertBlock();
  if (isDummyBlock(BB))
    BB->eraseFromParent();
  else  // Otherwise, branch to continuation.
    Builder.CreateBr(NextBlock);

  // Place NextBlock as the new insert point.
  CurFn->getBasicBlockList().push_back(NextBlock);
  Builder.SetInsertPoint(NextBlock);
  SwitchInsn = SavedSwitchInsn;
  CaseRangeBlock = SavedCRBlock;
}
