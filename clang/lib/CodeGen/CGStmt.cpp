//===--- CGStmt.cpp - Emit LLVM Code from Statements ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit Stmt nodes as LLVM code.
//
//===----------------------------------------------------------------------===//

#include "CGDebugInfo.h"
#include "CodeGenModule.h"
#include "CodeGenFunction.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Basic/PrettyStackTrace.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/InlineAsm.h"
#include "llvm/Intrinsics.h"
#include "llvm/Target/TargetData.h"
using namespace clang;
using namespace CodeGen;

//===----------------------------------------------------------------------===//
//                              Statement Emission
//===----------------------------------------------------------------------===//

void CodeGenFunction::EmitStopPoint(const Stmt *S) {
  if (CGDebugInfo *DI = getDebugInfo()) {
    if (isa<DeclStmt>(S))
      DI->setLocation(S->getLocEnd());
    else
      DI->setLocation(S->getLocStart());
    DI->UpdateLineDirectiveRegion(Builder);
    DI->EmitStopPoint(Builder);
  }
}

void CodeGenFunction::EmitStmt(const Stmt *S) {
  assert(S && "Null statement?");

  // Check if we can handle this without bothering to generate an
  // insert point or debug info.
  if (EmitSimpleStmt(S))
    return;

  // Check if we are generating unreachable code.
  if (!HaveInsertPoint()) {
    // If so, and the statement doesn't contain a label, then we do not need to
    // generate actual code. This is safe because (1) the current point is
    // unreachable, so we don't need to execute the code, and (2) we've already
    // handled the statements which update internal data structures (like the
    // local variable map) which could be used by subsequent statements.
    if (!ContainsLabel(S)) {
      // Verify that any decl statements were handled as simple, they may be in
      // scope of subsequent reachable statements.
      assert(!isa<DeclStmt>(*S) && "Unexpected DeclStmt!");
      return;
    }

    // Otherwise, make a new block to hold the code.
    EnsureInsertPoint();
  }

  // Generate a stoppoint if we are emitting debug info.
  EmitStopPoint(S);

  switch (S->getStmtClass()) {
  case Stmt::NoStmtClass:
  case Stmt::CXXCatchStmtClass:
  case Stmt::SwitchCaseClass:
    llvm_unreachable("invalid statement class to emit generically");
  case Stmt::NullStmtClass:
  case Stmt::CompoundStmtClass:
  case Stmt::DeclStmtClass:
  case Stmt::LabelStmtClass:
  case Stmt::GotoStmtClass:
  case Stmt::BreakStmtClass:
  case Stmt::ContinueStmtClass:
  case Stmt::DefaultStmtClass:
  case Stmt::CaseStmtClass:
    llvm_unreachable("should have emitted these statements as simple");

#define STMT(Type, Base)
#define ABSTRACT_STMT(Op)
#define EXPR(Type, Base) \
  case Stmt::Type##Class:
#include "clang/AST/StmtNodes.inc"
    EmitIgnoredExpr(cast<Expr>(S));

    // Expression emitters don't handle unreachable blocks yet, so look for one
    // explicitly here. This handles the common case of a call to a noreturn
    // function.
    if (llvm::BasicBlock *CurBB = Builder.GetInsertBlock()) {
      if (CurBB->empty() && CurBB->use_empty()) {
        CurBB->eraseFromParent();
        Builder.ClearInsertionPoint();
      }
    }
    break;

  case Stmt::IndirectGotoStmtClass:
    EmitIndirectGotoStmt(cast<IndirectGotoStmt>(*S)); break;

  case Stmt::IfStmtClass:       EmitIfStmt(cast<IfStmt>(*S));             break;
  case Stmt::WhileStmtClass:    EmitWhileStmt(cast<WhileStmt>(*S));       break;
  case Stmt::DoStmtClass:       EmitDoStmt(cast<DoStmt>(*S));             break;
  case Stmt::ForStmtClass:      EmitForStmt(cast<ForStmt>(*S));           break;

  case Stmt::ReturnStmtClass:   EmitReturnStmt(cast<ReturnStmt>(*S));     break;

  case Stmt::SwitchStmtClass:   EmitSwitchStmt(cast<SwitchStmt>(*S));     break;
  case Stmt::AsmStmtClass:      EmitAsmStmt(cast<AsmStmt>(*S));           break;

  case Stmt::ObjCAtTryStmtClass:
    EmitObjCAtTryStmt(cast<ObjCAtTryStmt>(*S));
    break;
  case Stmt::ObjCAtCatchStmtClass:
    assert(0 && "@catch statements should be handled by EmitObjCAtTryStmt");
    break;
  case Stmt::ObjCAtFinallyStmtClass:
    assert(0 && "@finally statements should be handled by EmitObjCAtTryStmt");
    break;
  case Stmt::ObjCAtThrowStmtClass:
    EmitObjCAtThrowStmt(cast<ObjCAtThrowStmt>(*S));
    break;
  case Stmt::ObjCAtSynchronizedStmtClass:
    EmitObjCAtSynchronizedStmt(cast<ObjCAtSynchronizedStmt>(*S));
    break;
  case Stmt::ObjCForCollectionStmtClass:
    EmitObjCForCollectionStmt(cast<ObjCForCollectionStmt>(*S));
    break;
      
  case Stmt::CXXTryStmtClass:
    EmitCXXTryStmt(cast<CXXTryStmt>(*S));
    break;
  }
}

bool CodeGenFunction::EmitSimpleStmt(const Stmt *S) {
  switch (S->getStmtClass()) {
  default: return false;
  case Stmt::NullStmtClass: break;
  case Stmt::CompoundStmtClass: EmitCompoundStmt(cast<CompoundStmt>(*S)); break;
  case Stmt::DeclStmtClass:     EmitDeclStmt(cast<DeclStmt>(*S));         break;
  case Stmt::LabelStmtClass:    EmitLabelStmt(cast<LabelStmt>(*S));       break;
  case Stmt::GotoStmtClass:     EmitGotoStmt(cast<GotoStmt>(*S));         break;
  case Stmt::BreakStmtClass:    EmitBreakStmt(cast<BreakStmt>(*S));       break;
  case Stmt::ContinueStmtClass: EmitContinueStmt(cast<ContinueStmt>(*S)); break;
  case Stmt::DefaultStmtClass:  EmitDefaultStmt(cast<DefaultStmt>(*S));   break;
  case Stmt::CaseStmtClass:     EmitCaseStmt(cast<CaseStmt>(*S));         break;
  }

  return true;
}

/// EmitCompoundStmt - Emit a compound statement {..} node.  If GetLast is true,
/// this captures the expression result of the last sub-statement and returns it
/// (for use by the statement expression extension).
RValue CodeGenFunction::EmitCompoundStmt(const CompoundStmt &S, bool GetLast,
                                         AggValueSlot AggSlot) {
  PrettyStackTraceLoc CrashInfo(getContext().getSourceManager(),S.getLBracLoc(),
                             "LLVM IR generation of compound statement ('{}')");

  CGDebugInfo *DI = getDebugInfo();
  if (DI) {
    DI->setLocation(S.getLBracLoc());
    DI->EmitRegionStart(Builder);
  }

  // Keep track of the current cleanup stack depth.
  RunCleanupsScope Scope(*this);

  for (CompoundStmt::const_body_iterator I = S.body_begin(),
       E = S.body_end()-GetLast; I != E; ++I)
    EmitStmt(*I);

  if (DI) {
    DI->setLocation(S.getRBracLoc());
    DI->EmitRegionEnd(Builder);
  }

  RValue RV;
  if (!GetLast)
    RV = RValue::get(0);
  else {
    // We have to special case labels here.  They are statements, but when put
    // at the end of a statement expression, they yield the value of their
    // subexpression.  Handle this by walking through all labels we encounter,
    // emitting them before we evaluate the subexpr.
    const Stmt *LastStmt = S.body_back();
    while (const LabelStmt *LS = dyn_cast<LabelStmt>(LastStmt)) {
      EmitLabel(*LS);
      LastStmt = LS->getSubStmt();
    }

    EnsureInsertPoint();

    RV = EmitAnyExpr(cast<Expr>(LastStmt), AggSlot);
  }

  return RV;
}

void CodeGenFunction::SimplifyForwardingBlocks(llvm::BasicBlock *BB) {
  llvm::BranchInst *BI = dyn_cast<llvm::BranchInst>(BB->getTerminator());

  // If there is a cleanup stack, then we it isn't worth trying to
  // simplify this block (we would need to remove it from the scope map
  // and cleanup entry).
  if (!EHStack.empty())
    return;

  // Can only simplify direct branches.
  if (!BI || !BI->isUnconditional())
    return;

  BB->replaceAllUsesWith(BI->getSuccessor(0));
  BI->eraseFromParent();
  BB->eraseFromParent();
}

void CodeGenFunction::EmitBlock(llvm::BasicBlock *BB, bool IsFinished) {
  llvm::BasicBlock *CurBB = Builder.GetInsertBlock();

  // Fall out of the current block (if necessary).
  EmitBranch(BB);

  if (IsFinished && BB->use_empty()) {
    delete BB;
    return;
  }

  // Place the block after the current block, if possible, or else at
  // the end of the function.
  if (CurBB && CurBB->getParent())
    CurFn->getBasicBlockList().insertAfter(CurBB, BB);
  else
    CurFn->getBasicBlockList().push_back(BB);
  Builder.SetInsertPoint(BB);
}

void CodeGenFunction::EmitBranch(llvm::BasicBlock *Target) {
  // Emit a branch from the current block to the target one if this
  // was a real block.  If this was just a fall-through block after a
  // terminator, don't emit it.
  llvm::BasicBlock *CurBB = Builder.GetInsertBlock();

  if (!CurBB || CurBB->getTerminator()) {
    // If there is no insert point or the previous block is already
    // terminated, don't touch it.
  } else {
    // Otherwise, create a fall-through branch.
    Builder.CreateBr(Target);
  }

  Builder.ClearInsertionPoint();
}

CodeGenFunction::JumpDest
CodeGenFunction::getJumpDestForLabel(const LabelStmt *S) {
  JumpDest &Dest = LabelMap[S];
  if (Dest.isValid()) return Dest;

  // Create, but don't insert, the new block.
  Dest = JumpDest(createBasicBlock(S->getName()),
                  EHScopeStack::stable_iterator::invalid(),
                  NextCleanupDestIndex++);
  return Dest;
}

void CodeGenFunction::EmitLabel(const LabelStmt &S) {
  JumpDest &Dest = LabelMap[&S];

  // If we didn't need a forward reference to this label, just go
  // ahead and create a destination at the current scope.
  if (!Dest.isValid()) {
    Dest = getJumpDestInCurrentScope(S.getName());

  // Otherwise, we need to give this label a target depth and remove
  // it from the branch-fixups list.
  } else {
    assert(!Dest.getScopeDepth().isValid() && "already emitted label!");
    Dest = JumpDest(Dest.getBlock(),
                    EHStack.stable_begin(),
                    Dest.getDestIndex());

    ResolveBranchFixups(Dest.getBlock());
  }

  EmitBlock(Dest.getBlock());
}


void CodeGenFunction::EmitLabelStmt(const LabelStmt &S) {
  EmitLabel(S);
  EmitStmt(S.getSubStmt());
}

void CodeGenFunction::EmitGotoStmt(const GotoStmt &S) {
  // If this code is reachable then emit a stop point (if generating
  // debug info). We have to do this ourselves because we are on the
  // "simple" statement path.
  if (HaveInsertPoint())
    EmitStopPoint(&S);

  EmitBranchThroughCleanup(getJumpDestForLabel(S.getLabel()));
}


void CodeGenFunction::EmitIndirectGotoStmt(const IndirectGotoStmt &S) {
  if (const LabelStmt *Target = S.getConstantTarget()) {
    EmitBranchThroughCleanup(getJumpDestForLabel(Target));
    return;
  }

  // Ensure that we have an i8* for our PHI node.
  llvm::Value *V = Builder.CreateBitCast(EmitScalarExpr(S.getTarget()),
                                         llvm::Type::getInt8PtrTy(VMContext),
                                          "addr");
  llvm::BasicBlock *CurBB = Builder.GetInsertBlock();
  

  // Get the basic block for the indirect goto.
  llvm::BasicBlock *IndGotoBB = GetIndirectGotoBlock();
  
  // The first instruction in the block has to be the PHI for the switch dest,
  // add an entry for this branch.
  cast<llvm::PHINode>(IndGotoBB->begin())->addIncoming(V, CurBB);
  
  EmitBranch(IndGotoBB);
}

void CodeGenFunction::EmitIfStmt(const IfStmt &S) {
  // C99 6.8.4.1: The first substatement is executed if the expression compares
  // unequal to 0.  The condition must be a scalar type.
  RunCleanupsScope ConditionScope(*this);

  if (S.getConditionVariable())
    EmitAutoVarDecl(*S.getConditionVariable());

  // If the condition constant folds and can be elided, try to avoid emitting
  // the condition and the dead arm of the if/else.
  if (int Cond = ConstantFoldsToSimpleInteger(S.getCond())) {
    // Figure out which block (then or else) is executed.
    const Stmt *Executed = S.getThen(), *Skipped  = S.getElse();
    if (Cond == -1)  // Condition false?
      std::swap(Executed, Skipped);

    // If the skipped block has no labels in it, just emit the executed block.
    // This avoids emitting dead code and simplifies the CFG substantially.
    if (!ContainsLabel(Skipped)) {
      if (Executed) {
        RunCleanupsScope ExecutedScope(*this);
        EmitStmt(Executed);
      }
      return;
    }
  }

  // Otherwise, the condition did not fold, or we couldn't elide it.  Just emit
  // the conditional branch.
  llvm::BasicBlock *ThenBlock = createBasicBlock("if.then");
  llvm::BasicBlock *ContBlock = createBasicBlock("if.end");
  llvm::BasicBlock *ElseBlock = ContBlock;
  if (S.getElse())
    ElseBlock = createBasicBlock("if.else");
  EmitBranchOnBoolExpr(S.getCond(), ThenBlock, ElseBlock);

  // Emit the 'then' code.
  EmitBlock(ThenBlock); 
  {
    RunCleanupsScope ThenScope(*this);
    EmitStmt(S.getThen());
  }
  EmitBranch(ContBlock);

  // Emit the 'else' code if present.
  if (const Stmt *Else = S.getElse()) {
    EmitBlock(ElseBlock);
    {
      RunCleanupsScope ElseScope(*this);
      EmitStmt(Else);
    }
    EmitBranch(ContBlock);
  }

  // Emit the continuation block for code after the if.
  EmitBlock(ContBlock, true);
}

void CodeGenFunction::EmitWhileStmt(const WhileStmt &S) {
  // Emit the header for the loop, which will also become
  // the continue target.
  JumpDest LoopHeader = getJumpDestInCurrentScope("while.cond");
  EmitBlock(LoopHeader.getBlock());

  // Create an exit block for when the condition fails, which will
  // also become the break target.
  JumpDest LoopExit = getJumpDestInCurrentScope("while.end");

  // Store the blocks to use for break and continue.
  BreakContinueStack.push_back(BreakContinue(LoopExit, LoopHeader));

  // C++ [stmt.while]p2:
  //   When the condition of a while statement is a declaration, the
  //   scope of the variable that is declared extends from its point
  //   of declaration (3.3.2) to the end of the while statement.
  //   [...]
  //   The object created in a condition is destroyed and created
  //   with each iteration of the loop.
  RunCleanupsScope ConditionScope(*this);

  if (S.getConditionVariable())
    EmitAutoVarDecl(*S.getConditionVariable());
  
  // Evaluate the conditional in the while header.  C99 6.8.5.1: The
  // evaluation of the controlling expression takes place before each
  // execution of the loop body.
  llvm::Value *BoolCondVal = EvaluateExprAsBool(S.getCond());
   
  // while(1) is common, avoid extra exit blocks.  Be sure
  // to correctly handle break/continue though.
  bool EmitBoolCondBranch = true;
  if (llvm::ConstantInt *C = dyn_cast<llvm::ConstantInt>(BoolCondVal))
    if (C->isOne())
      EmitBoolCondBranch = false;

  // As long as the condition is true, go to the loop body.
  llvm::BasicBlock *LoopBody = createBasicBlock("while.body");
  if (EmitBoolCondBranch) {
    llvm::BasicBlock *ExitBlock = LoopExit.getBlock();
    if (ConditionScope.requiresCleanups())
      ExitBlock = createBasicBlock("while.exit");

    Builder.CreateCondBr(BoolCondVal, LoopBody, ExitBlock);

    if (ExitBlock != LoopExit.getBlock()) {
      EmitBlock(ExitBlock);
      EmitBranchThroughCleanup(LoopExit);
    }
  }
 
  // Emit the loop body.  We have to emit this in a cleanup scope
  // because it might be a singleton DeclStmt.
  {
    RunCleanupsScope BodyScope(*this);
    EmitBlock(LoopBody);
    EmitStmt(S.getBody());
  }

  BreakContinueStack.pop_back();

  // Immediately force cleanup.
  ConditionScope.ForceCleanup();

  // Branch to the loop header again.
  EmitBranch(LoopHeader.getBlock());

  // Emit the exit block.
  EmitBlock(LoopExit.getBlock(), true);

  // The LoopHeader typically is just a branch if we skipped emitting
  // a branch, try to erase it.
  if (!EmitBoolCondBranch)
    SimplifyForwardingBlocks(LoopHeader.getBlock());
}

void CodeGenFunction::EmitDoStmt(const DoStmt &S) {
  JumpDest LoopExit = getJumpDestInCurrentScope("do.end");
  JumpDest LoopCond = getJumpDestInCurrentScope("do.cond");

  // Store the blocks to use for break and continue.
  BreakContinueStack.push_back(BreakContinue(LoopExit, LoopCond));

  // Emit the body of the loop.
  llvm::BasicBlock *LoopBody = createBasicBlock("do.body");
  EmitBlock(LoopBody);
  {
    RunCleanupsScope BodyScope(*this);
    EmitStmt(S.getBody());
  }

  BreakContinueStack.pop_back();

  EmitBlock(LoopCond.getBlock());

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
    Builder.CreateCondBr(BoolCondVal, LoopBody, LoopExit.getBlock());

  // Emit the exit block.
  EmitBlock(LoopExit.getBlock());

  // The DoCond block typically is just a branch if we skipped
  // emitting a branch, try to erase it.
  if (!EmitBoolCondBranch)
    SimplifyForwardingBlocks(LoopCond.getBlock());
}

void CodeGenFunction::EmitForStmt(const ForStmt &S) {
  JumpDest LoopExit = getJumpDestInCurrentScope("for.end");

  RunCleanupsScope ForScope(*this);

  CGDebugInfo *DI = getDebugInfo();
  if (DI) {
    DI->setLocation(S.getSourceRange().getBegin());
    DI->EmitRegionStart(Builder);
  }

  // Evaluate the first part before the loop.
  if (S.getInit())
    EmitStmt(S.getInit());

  // Start the loop with a block that tests the condition.
  // If there's an increment, the continue scope will be overwritten
  // later.
  JumpDest Continue = getJumpDestInCurrentScope("for.cond");
  llvm::BasicBlock *CondBlock = Continue.getBlock();
  EmitBlock(CondBlock);

  // Create a cleanup scope for the condition variable cleanups.
  RunCleanupsScope ConditionScope(*this);
  
  llvm::Value *BoolCondVal = 0;
  if (S.getCond()) {
    // If the for statement has a condition scope, emit the local variable
    // declaration.
    llvm::BasicBlock *ExitBlock = LoopExit.getBlock();
    if (S.getConditionVariable()) {
      EmitAutoVarDecl(*S.getConditionVariable());
    }

    // If there are any cleanups between here and the loop-exit scope,
    // create a block to stage a loop exit along.
    if (ForScope.requiresCleanups())
      ExitBlock = createBasicBlock("for.cond.cleanup");
    
    // As long as the condition is true, iterate the loop.
    llvm::BasicBlock *ForBody = createBasicBlock("for.body");

    // C99 6.8.5p2/p4: The first substatement is executed if the expression
    // compares unequal to 0.  The condition must be a scalar type.
    BoolCondVal = EvaluateExprAsBool(S.getCond());
    Builder.CreateCondBr(BoolCondVal, ForBody, ExitBlock);

    if (ExitBlock != LoopExit.getBlock()) {
      EmitBlock(ExitBlock);
      EmitBranchThroughCleanup(LoopExit);
    }

    EmitBlock(ForBody);
  } else {
    // Treat it as a non-zero constant.  Don't even create a new block for the
    // body, just fall into it.
  }

  // If the for loop doesn't have an increment we can just use the
  // condition as the continue block.  Otherwise we'll need to create
  // a block for it (in the current scope, i.e. in the scope of the
  // condition), and that we will become our continue block.
  if (S.getInc())
    Continue = getJumpDestInCurrentScope("for.inc");

  // Store the blocks to use for break and continue.
  BreakContinueStack.push_back(BreakContinue(LoopExit, Continue));

  {
    // Create a separate cleanup scope for the body, in case it is not
    // a compound statement.
    RunCleanupsScope BodyScope(*this);
    EmitStmt(S.getBody());
  }

  // If there is an increment, emit it next.
  if (S.getInc()) {
    EmitBlock(Continue.getBlock());
    EmitStmt(S.getInc());
  }

  BreakContinueStack.pop_back();

  ConditionScope.ForceCleanup();
  EmitBranch(CondBlock);

  ForScope.ForceCleanup();

  if (DI) {
    DI->setLocation(S.getSourceRange().getEnd());
    DI->EmitRegionEnd(Builder);
  }

  // Emit the fall-through block.
  EmitBlock(LoopExit.getBlock(), true);
}

void CodeGenFunction::EmitReturnOfRValue(RValue RV, QualType Ty) {
  if (RV.isScalar()) {
    Builder.CreateStore(RV.getScalarVal(), ReturnValue);
  } else if (RV.isAggregate()) {
    EmitAggregateCopy(ReturnValue, RV.getAggregateAddr(), Ty);
  } else {
    StoreComplexToAddr(RV.getComplexVal(), ReturnValue, false);
  }
  EmitBranchThroughCleanup(ReturnBlock);
}

/// EmitReturnStmt - Note that due to GCC extensions, this can have an operand
/// if the function returns void, or may be missing one if the function returns
/// non-void.  Fun stuff :).
void CodeGenFunction::EmitReturnStmt(const ReturnStmt &S) {
  // Emit the result value, even if unused, to evalute the side effects.
  const Expr *RV = S.getRetValue();

  // FIXME: Clean this up by using an LValue for ReturnTemp,
  // EmitStoreThroughLValue, and EmitAnyExpr.
  if (S.getNRVOCandidate() && S.getNRVOCandidate()->isNRVOVariable() &&
      !Target.useGlobalsForAutomaticVariables()) {
    // Apply the named return value optimization for this return statement,
    // which means doing nothing: the appropriate result has already been
    // constructed into the NRVO variable.
    
    // If there is an NRVO flag for this variable, set it to 1 into indicate
    // that the cleanup code should not destroy the variable.
    if (llvm::Value *NRVOFlag = NRVOFlags[S.getNRVOCandidate()]) {
      const llvm::Type *BoolTy = llvm::Type::getInt1Ty(VMContext);
      llvm::Value *One = llvm::ConstantInt::get(BoolTy, 1);
      Builder.CreateStore(One, NRVOFlag);
    }
  } else if (!ReturnValue) {
    // Make sure not to return anything, but evaluate the expression
    // for side effects.
    if (RV)
      EmitAnyExpr(RV);
  } else if (RV == 0) {
    // Do nothing (return value is left uninitialized)
  } else if (FnRetTy->isReferenceType()) {
    // If this function returns a reference, take the address of the expression
    // rather than the value.
    RValue Result = EmitReferenceBindingToExpr(RV, /*InitializedDecl=*/0);
    Builder.CreateStore(Result.getScalarVal(), ReturnValue);
  } else if (!hasAggregateLLVMType(RV->getType())) {
    Builder.CreateStore(EmitScalarExpr(RV), ReturnValue);
  } else if (RV->getType()->isAnyComplexType()) {
    EmitComplexExprIntoAddr(RV, ReturnValue, false);
  } else {
    EmitAggExpr(RV, AggValueSlot::forAddr(ReturnValue, false, true));
  }

  EmitBranchThroughCleanup(ReturnBlock);
}

void CodeGenFunction::EmitDeclStmt(const DeclStmt &S) {
  // As long as debug info is modeled with instructions, we have to ensure we
  // have a place to insert here and write the stop point here.
  if (getDebugInfo()) {
    EnsureInsertPoint();
    EmitStopPoint(&S);
  }

  for (DeclStmt::const_decl_iterator I = S.decl_begin(), E = S.decl_end();
       I != E; ++I)
    EmitDecl(**I);
}

void CodeGenFunction::EmitBreakStmt(const BreakStmt &S) {
  assert(!BreakContinueStack.empty() && "break stmt not in a loop or switch!");

  // If this code is reachable then emit a stop point (if generating
  // debug info). We have to do this ourselves because we are on the
  // "simple" statement path.
  if (HaveInsertPoint())
    EmitStopPoint(&S);

  JumpDest Block = BreakContinueStack.back().BreakBlock;
  EmitBranchThroughCleanup(Block);
}

void CodeGenFunction::EmitContinueStmt(const ContinueStmt &S) {
  assert(!BreakContinueStack.empty() && "continue stmt not in a loop!");

  // If this code is reachable then emit a stop point (if generating
  // debug info). We have to do this ourselves because we are on the
  // "simple" statement path.
  if (HaveInsertPoint())
    EmitStopPoint(&S);

  JumpDest Block = BreakContinueStack.back().ContinueBlock;
  EmitBranchThroughCleanup(Block);
}

/// EmitCaseStmtRange - If case statement range is not too big then
/// add multiple cases to switch instruction, one for each value within
/// the range. If range is too big then emit "if" condition check.
void CodeGenFunction::EmitCaseStmtRange(const CaseStmt &S) {
  assert(S.getRHS() && "Expected RHS value in CaseStmt");

  llvm::APSInt LHS = S.getLHS()->EvaluateAsInt(getContext());
  llvm::APSInt RHS = S.getRHS()->EvaluateAsInt(getContext());

  // Emit the code for this case. We do this first to make sure it is
  // properly chained from our predecessor before generating the
  // switch machinery to enter this block.
  EmitBlock(createBasicBlock("sw.bb"));
  llvm::BasicBlock *CaseDest = Builder.GetInsertBlock();
  EmitStmt(S.getSubStmt());

  // If range is empty, do nothing.
  if (LHS.isSigned() ? RHS.slt(LHS) : RHS.ult(LHS))
    return;

  llvm::APInt Range = RHS - LHS;
  // FIXME: parameters such as this should not be hardcoded.
  if (Range.ult(llvm::APInt(Range.getBitWidth(), 64))) {
    // Range is small enough to add multiple switch instruction cases.
    for (unsigned i = 0, e = Range.getZExtValue() + 1; i != e; ++i) {
      SwitchInsn->addCase(llvm::ConstantInt::get(VMContext, LHS), CaseDest);
      LHS++;
    }
    return;
  }

  // The range is too big. Emit "if" condition into a new block,
  // making sure to save and restore the current insertion point.
  llvm::BasicBlock *RestoreBB = Builder.GetInsertBlock();

  // Push this test onto the chain of range checks (which terminates
  // in the default basic block). The switch's default will be changed
  // to the top of this chain after switch emission is complete.
  llvm::BasicBlock *FalseDest = CaseRangeBlock;
  CaseRangeBlock = createBasicBlock("sw.caserange");

  CurFn->getBasicBlockList().push_back(CaseRangeBlock);
  Builder.SetInsertPoint(CaseRangeBlock);

  // Emit range check.
  llvm::Value *Diff =
    Builder.CreateSub(SwitchInsn->getCondition(),
                      llvm::ConstantInt::get(VMContext, LHS),  "tmp");
  llvm::Value *Cond =
    Builder.CreateICmpULE(Diff,
                          llvm::ConstantInt::get(VMContext, Range), "tmp");
  Builder.CreateCondBr(Cond, CaseDest, FalseDest);

  // Restore the appropriate insertion point.
  if (RestoreBB)
    Builder.SetInsertPoint(RestoreBB);
  else
    Builder.ClearInsertionPoint();
}

void CodeGenFunction::EmitCaseStmt(const CaseStmt &S) {
  if (S.getRHS()) {
    EmitCaseStmtRange(S);
    return;
  }

  EmitBlock(createBasicBlock("sw.bb"));
  llvm::BasicBlock *CaseDest = Builder.GetInsertBlock();
  llvm::APSInt CaseVal = S.getLHS()->EvaluateAsInt(getContext());
  SwitchInsn->addCase(llvm::ConstantInt::get(VMContext, CaseVal), CaseDest);

  // Recursively emitting the statement is acceptable, but is not wonderful for
  // code where we have many case statements nested together, i.e.:
  //  case 1:
  //    case 2:
  //      case 3: etc.
  // Handling this recursively will create a new block for each case statement
  // that falls through to the next case which is IR intensive.  It also causes
  // deep recursion which can run into stack depth limitations.  Handle
  // sequential non-range case statements specially.
  const CaseStmt *CurCase = &S;
  const CaseStmt *NextCase = dyn_cast<CaseStmt>(S.getSubStmt());

  // Otherwise, iteratively add consequtive cases to this switch stmt.
  while (NextCase && NextCase->getRHS() == 0) {
    CurCase = NextCase;
    CaseVal = CurCase->getLHS()->EvaluateAsInt(getContext());
    SwitchInsn->addCase(llvm::ConstantInt::get(VMContext, CaseVal), CaseDest);

    NextCase = dyn_cast<CaseStmt>(CurCase->getSubStmt());
  }

  // Normal default recursion for non-cases.
  EmitStmt(CurCase->getSubStmt());
}

void CodeGenFunction::EmitDefaultStmt(const DefaultStmt &S) {
  llvm::BasicBlock *DefaultBlock = SwitchInsn->getDefaultDest();
  assert(DefaultBlock->empty() &&
         "EmitDefaultStmt: Default block already defined?");
  EmitBlock(DefaultBlock);
  EmitStmt(S.getSubStmt());
}

void CodeGenFunction::EmitSwitchStmt(const SwitchStmt &S) {
  JumpDest SwitchExit = getJumpDestInCurrentScope("sw.epilog");

  RunCleanupsScope ConditionScope(*this);

  if (S.getConditionVariable())
    EmitAutoVarDecl(*S.getConditionVariable());

  llvm::Value *CondV = EmitScalarExpr(S.getCond());

  // Handle nested switch statements.
  llvm::SwitchInst *SavedSwitchInsn = SwitchInsn;
  llvm::BasicBlock *SavedCRBlock = CaseRangeBlock;

  // Create basic block to hold stuff that comes after switch
  // statement. We also need to create a default block now so that
  // explicit case ranges tests can have a place to jump to on
  // failure.
  llvm::BasicBlock *DefaultBlock = createBasicBlock("sw.default");
  SwitchInsn = Builder.CreateSwitch(CondV, DefaultBlock);
  CaseRangeBlock = DefaultBlock;

  // Clear the insertion point to indicate we are in unreachable code.
  Builder.ClearInsertionPoint();

  // All break statements jump to NextBlock. If BreakContinueStack is non empty
  // then reuse last ContinueBlock.
  JumpDest OuterContinue;
  if (!BreakContinueStack.empty())
    OuterContinue = BreakContinueStack.back().ContinueBlock;

  BreakContinueStack.push_back(BreakContinue(SwitchExit, OuterContinue));

  // Emit switch body.
  EmitStmt(S.getBody());

  BreakContinueStack.pop_back();

  // Update the default block in case explicit case range tests have
  // been chained on top.
  SwitchInsn->setSuccessor(0, CaseRangeBlock);

  // If a default was never emitted:
  if (!DefaultBlock->getParent()) {
    // If we have cleanups, emit the default block so that there's a
    // place to jump through the cleanups from.
    if (ConditionScope.requiresCleanups()) {
      EmitBlock(DefaultBlock);

    // Otherwise, just forward the default block to the switch end.
    } else {
      DefaultBlock->replaceAllUsesWith(SwitchExit.getBlock());
      delete DefaultBlock;
    }
  }

  ConditionScope.ForceCleanup();

  // Emit continuation.
  EmitBlock(SwitchExit.getBlock(), true);

  SwitchInsn = SavedSwitchInsn;
  CaseRangeBlock = SavedCRBlock;
}

static std::string
SimplifyConstraint(const char *Constraint, const TargetInfo &Target,
                 llvm::SmallVectorImpl<TargetInfo::ConstraintInfo> *OutCons=0) {
  std::string Result;

  while (*Constraint) {
    switch (*Constraint) {
    default:
      Result += Target.convertConstraint(*Constraint);
      break;
    // Ignore these
    case '*':
    case '?':
    case '!':
    case '=': // Will see this and the following in mult-alt constraints.
    case '+':
      break;
    case ',':
      Result += "|";
      break;
    case 'g':
      Result += "imr";
      break;
    case '[': {
      assert(OutCons &&
             "Must pass output names to constraints with a symbolic name");
      unsigned Index;
      bool result = Target.resolveSymbolicName(Constraint,
                                               &(*OutCons)[0],
                                               OutCons->size(), Index);
      assert(result && "Could not resolve symbolic name"); result=result;
      Result += llvm::utostr(Index);
      break;
    }
    }

    Constraint++;
  }

  return Result;
}

static std::string
AddVariableConstraits(const std::string &Constraint, const Expr &AsmExpr,
                      const TargetInfo &Target, CodeGenModule &CGM,
                      const AsmStmt &Stmt) {
  const DeclRefExpr *AsmDeclRef = dyn_cast<DeclRefExpr>(&AsmExpr);
  if (!AsmDeclRef)
    return Constraint;
  const ValueDecl &Value = *AsmDeclRef->getDecl();
  const VarDecl *Variable = dyn_cast<VarDecl>(&Value);
  if (!Variable)
    return Constraint;
  AsmLabelAttr *Attr = Variable->getAttr<AsmLabelAttr>();
  if (!Attr)
    return Constraint;
  llvm::StringRef Register = Attr->getLabel();
  if (!Target.isValidGCCRegisterName(Register)) {
    CGM.ErrorUnsupported(Variable, "__asm__");
    return Constraint;
  }
  if (Constraint != "r") {
    CGM.ErrorUnsupported(&Stmt, "__asm__");
    return Constraint;
  }
  return "{" + Register.str() + "}";
}

llvm::Value*
CodeGenFunction::EmitAsmInputLValue(const AsmStmt &S,
                                    const TargetInfo::ConstraintInfo &Info,
                                    LValue InputValue, QualType InputType,
                                    std::string &ConstraintStr) {
  llvm::Value *Arg;
  if (Info.allowsRegister() || !Info.allowsMemory()) {
    if (!CodeGenFunction::hasAggregateLLVMType(InputType)) {
      Arg = EmitLoadOfLValue(InputValue, InputType).getScalarVal();
    } else {
      const llvm::Type *Ty = ConvertType(InputType);
      uint64_t Size = CGM.getTargetData().getTypeSizeInBits(Ty);
      if (Size <= 64 && llvm::isPowerOf2_64(Size)) {
        Ty = llvm::IntegerType::get(VMContext, Size);
        Ty = llvm::PointerType::getUnqual(Ty);

        Arg = Builder.CreateLoad(Builder.CreateBitCast(InputValue.getAddress(),
                                                       Ty));
      } else {
        Arg = InputValue.getAddress();
        ConstraintStr += '*';
      }
    }
  } else {
    Arg = InputValue.getAddress();
    ConstraintStr += '*';
  }

  return Arg;
}

llvm::Value* CodeGenFunction::EmitAsmInput(const AsmStmt &S,
                                         const TargetInfo::ConstraintInfo &Info,
                                           const Expr *InputExpr,
                                           std::string &ConstraintStr) {
  if (Info.allowsRegister() || !Info.allowsMemory())
    if (!CodeGenFunction::hasAggregateLLVMType(InputExpr->getType()))
      return EmitScalarExpr(InputExpr);

  InputExpr = InputExpr->IgnoreParenNoopCasts(getContext());
  LValue Dest = EmitLValue(InputExpr);
  return EmitAsmInputLValue(S, Info, Dest, InputExpr->getType(), ConstraintStr);
}

/// getAsmSrcLocInfo - Return the !srcloc metadata node to attach to an inline
/// asm call instruction.  The !srcloc MDNode contains a list of constant
/// integers which are the source locations of the start of each line in the
/// asm.
static llvm::MDNode *getAsmSrcLocInfo(const StringLiteral *Str,
                                      CodeGenFunction &CGF) {
  llvm::SmallVector<llvm::Value *, 8> Locs;
  // Add the location of the first line to the MDNode.
  Locs.push_back(llvm::ConstantInt::get(CGF.Int32Ty,
                                        Str->getLocStart().getRawEncoding()));
  llvm::StringRef StrVal = Str->getString();
  if (!StrVal.empty()) {
    const SourceManager &SM = CGF.CGM.getContext().getSourceManager();
    const LangOptions &LangOpts = CGF.CGM.getLangOptions();
    
    // Add the location of the start of each subsequent line of the asm to the
    // MDNode.
    for (unsigned i = 0, e = StrVal.size()-1; i != e; ++i) {
      if (StrVal[i] != '\n') continue;
      SourceLocation LineLoc = Str->getLocationOfByte(i+1, SM, LangOpts,
                                                      CGF.Target);
      Locs.push_back(llvm::ConstantInt::get(CGF.Int32Ty,
                                            LineLoc.getRawEncoding()));
    }
  }    
  
  return llvm::MDNode::get(CGF.getLLVMContext(), Locs.data(), Locs.size());
}

void CodeGenFunction::EmitAsmStmt(const AsmStmt &S) {
  // Analyze the asm string to decompose it into its pieces.  We know that Sema
  // has already done this, so it is guaranteed to be successful.
  llvm::SmallVector<AsmStmt::AsmStringPiece, 4> Pieces;
  unsigned DiagOffs;
  S.AnalyzeAsmString(Pieces, getContext(), DiagOffs);

  // Assemble the pieces into the final asm string.
  std::string AsmString;
  for (unsigned i = 0, e = Pieces.size(); i != e; ++i) {
    if (Pieces[i].isString())
      AsmString += Pieces[i].getString();
    else if (Pieces[i].getModifier() == '\0')
      AsmString += '$' + llvm::utostr(Pieces[i].getOperandNo());
    else
      AsmString += "${" + llvm::utostr(Pieces[i].getOperandNo()) + ':' +
                   Pieces[i].getModifier() + '}';
  }

  // Get all the output and input constraints together.
  llvm::SmallVector<TargetInfo::ConstraintInfo, 4> OutputConstraintInfos;
  llvm::SmallVector<TargetInfo::ConstraintInfo, 4> InputConstraintInfos;

  for (unsigned i = 0, e = S.getNumOutputs(); i != e; i++) {
    TargetInfo::ConstraintInfo Info(S.getOutputConstraint(i),
                                    S.getOutputName(i));
    bool IsValid = Target.validateOutputConstraint(Info); (void)IsValid;
    assert(IsValid && "Failed to parse output constraint"); 
    OutputConstraintInfos.push_back(Info);
  }

  for (unsigned i = 0, e = S.getNumInputs(); i != e; i++) {
    TargetInfo::ConstraintInfo Info(S.getInputConstraint(i),
                                    S.getInputName(i));
    bool IsValid = Target.validateInputConstraint(OutputConstraintInfos.data(),
                                                  S.getNumOutputs(), Info);
    assert(IsValid && "Failed to parse input constraint"); (void)IsValid;
    InputConstraintInfos.push_back(Info);
  }

  std::string Constraints;

  std::vector<LValue> ResultRegDests;
  std::vector<QualType> ResultRegQualTys;
  std::vector<const llvm::Type *> ResultRegTypes;
  std::vector<const llvm::Type *> ResultTruncRegTypes;
  std::vector<const llvm::Type*> ArgTypes;
  std::vector<llvm::Value*> Args;

  // Keep track of inout constraints.
  std::string InOutConstraints;
  std::vector<llvm::Value*> InOutArgs;
  std::vector<const llvm::Type*> InOutArgTypes;

  for (unsigned i = 0, e = S.getNumOutputs(); i != e; i++) {
    TargetInfo::ConstraintInfo &Info = OutputConstraintInfos[i];

    // Simplify the output constraint.
    std::string OutputConstraint(S.getOutputConstraint(i));
    OutputConstraint = SimplifyConstraint(OutputConstraint.c_str() + 1, Target);

    const Expr *OutExpr = S.getOutputExpr(i);
    OutExpr = OutExpr->IgnoreParenNoopCasts(getContext());

    OutputConstraint = AddVariableConstraits(OutputConstraint, *OutExpr, Target,
                                             CGM, S);

    LValue Dest = EmitLValue(OutExpr);
    if (!Constraints.empty())
      Constraints += ',';

    // If this is a register output, then make the inline asm return it
    // by-value.  If this is a memory result, return the value by-reference.
    if (!Info.allowsMemory() && !hasAggregateLLVMType(OutExpr->getType())) {
      Constraints += "=" + OutputConstraint;
      ResultRegQualTys.push_back(OutExpr->getType());
      ResultRegDests.push_back(Dest);
      ResultRegTypes.push_back(ConvertTypeForMem(OutExpr->getType()));
      ResultTruncRegTypes.push_back(ResultRegTypes.back());

      // If this output is tied to an input, and if the input is larger, then
      // we need to set the actual result type of the inline asm node to be the
      // same as the input type.
      if (Info.hasMatchingInput()) {
        unsigned InputNo;
        for (InputNo = 0; InputNo != S.getNumInputs(); ++InputNo) {
          TargetInfo::ConstraintInfo &Input = InputConstraintInfos[InputNo];
          if (Input.hasTiedOperand() && Input.getTiedOperand() == i)
            break;
        }
        assert(InputNo != S.getNumInputs() && "Didn't find matching input!");

        QualType InputTy = S.getInputExpr(InputNo)->getType();
        QualType OutputType = OutExpr->getType();

        uint64_t InputSize = getContext().getTypeSize(InputTy);
        if (getContext().getTypeSize(OutputType) < InputSize) {
          // Form the asm to return the value as a larger integer or fp type.
          ResultRegTypes.back() = ConvertType(InputTy);
        }
      }
      if (const llvm::Type* AdjTy = 
            Target.adjustInlineAsmType(OutputConstraint, ResultRegTypes.back(),
                                       VMContext))
        ResultRegTypes.back() = AdjTy;
    } else {
      ArgTypes.push_back(Dest.getAddress()->getType());
      Args.push_back(Dest.getAddress());
      Constraints += "=*";
      Constraints += OutputConstraint;
    }

    if (Info.isReadWrite()) {
      InOutConstraints += ',';

      const Expr *InputExpr = S.getOutputExpr(i);
      llvm::Value *Arg = EmitAsmInputLValue(S, Info, Dest, InputExpr->getType(),
                                            InOutConstraints);

      if (Info.allowsRegister())
        InOutConstraints += llvm::utostr(i);
      else
        InOutConstraints += OutputConstraint;

      InOutArgTypes.push_back(Arg->getType());
      InOutArgs.push_back(Arg);
    }
  }

  unsigned NumConstraints = S.getNumOutputs() + S.getNumInputs();

  for (unsigned i = 0, e = S.getNumInputs(); i != e; i++) {
    const Expr *InputExpr = S.getInputExpr(i);

    TargetInfo::ConstraintInfo &Info = InputConstraintInfos[i];

    if (!Constraints.empty())
      Constraints += ',';

    // Simplify the input constraint.
    std::string InputConstraint(S.getInputConstraint(i));
    InputConstraint = SimplifyConstraint(InputConstraint.c_str(), Target,
                                         &OutputConstraintInfos);

    InputConstraint =
      AddVariableConstraits(InputConstraint,
                            *InputExpr->IgnoreParenNoopCasts(getContext()),
                            Target, CGM, S);

    llvm::Value *Arg = EmitAsmInput(S, Info, InputExpr, Constraints);

    // If this input argument is tied to a larger output result, extend the
    // input to be the same size as the output.  The LLVM backend wants to see
    // the input and output of a matching constraint be the same size.  Note
    // that GCC does not define what the top bits are here.  We use zext because
    // that is usually cheaper, but LLVM IR should really get an anyext someday.
    if (Info.hasTiedOperand()) {
      unsigned Output = Info.getTiedOperand();
      QualType OutputType = S.getOutputExpr(Output)->getType();
      QualType InputTy = InputExpr->getType();

      if (getContext().getTypeSize(OutputType) >
          getContext().getTypeSize(InputTy)) {
        // Use ptrtoint as appropriate so that we can do our extension.
        if (isa<llvm::PointerType>(Arg->getType()))
          Arg = Builder.CreatePtrToInt(Arg, IntPtrTy);
        const llvm::Type *OutputTy = ConvertType(OutputType);
        if (isa<llvm::IntegerType>(OutputTy))
          Arg = Builder.CreateZExt(Arg, OutputTy);
        else
          Arg = Builder.CreateFPExt(Arg, OutputTy);
      }
    }
    if (const llvm::Type* AdjTy = 
              Target.adjustInlineAsmType(InputConstraint, Arg->getType(),
                                         VMContext))
      Arg = Builder.CreateBitCast(Arg, AdjTy);

    ArgTypes.push_back(Arg->getType());
    Args.push_back(Arg);
    Constraints += InputConstraint;
  }

  // Append the "input" part of inout constraints last.
  for (unsigned i = 0, e = InOutArgs.size(); i != e; i++) {
    ArgTypes.push_back(InOutArgTypes[i]);
    Args.push_back(InOutArgs[i]);
  }
  Constraints += InOutConstraints;

  // Clobbers
  for (unsigned i = 0, e = S.getNumClobbers(); i != e; i++) {
    llvm::StringRef Clobber = S.getClobber(i)->getString();

    Clobber = Target.getNormalizedGCCRegisterName(Clobber);

    if (i != 0 || NumConstraints != 0)
      Constraints += ',';

    Constraints += "~{";
    Constraints += Clobber;
    Constraints += '}';
  }

  // Add machine specific clobbers
  std::string MachineClobbers = Target.getClobbers();
  if (!MachineClobbers.empty()) {
    if (!Constraints.empty())
      Constraints += ',';
    Constraints += MachineClobbers;
  }

  const llvm::Type *ResultType;
  if (ResultRegTypes.empty())
    ResultType = llvm::Type::getVoidTy(VMContext);
  else if (ResultRegTypes.size() == 1)
    ResultType = ResultRegTypes[0];
  else
    ResultType = llvm::StructType::get(VMContext, ResultRegTypes);

  const llvm::FunctionType *FTy =
    llvm::FunctionType::get(ResultType, ArgTypes, false);

  llvm::InlineAsm *IA =
    llvm::InlineAsm::get(FTy, AsmString, Constraints,
                         S.isVolatile() || S.getNumOutputs() == 0);
  llvm::CallInst *Result = Builder.CreateCall(IA, Args.begin(), Args.end());
  Result->addAttribute(~0, llvm::Attribute::NoUnwind);

  // Slap the source location of the inline asm into a !srcloc metadata on the
  // call.
  Result->setMetadata("srcloc", getAsmSrcLocInfo(S.getAsmString(), *this));

  // Extract all of the register value results from the asm.
  std::vector<llvm::Value*> RegResults;
  if (ResultRegTypes.size() == 1) {
    RegResults.push_back(Result);
  } else {
    for (unsigned i = 0, e = ResultRegTypes.size(); i != e; ++i) {
      llvm::Value *Tmp = Builder.CreateExtractValue(Result, i, "asmresult");
      RegResults.push_back(Tmp);
    }
  }

  for (unsigned i = 0, e = RegResults.size(); i != e; ++i) {
    llvm::Value *Tmp = RegResults[i];

    // If the result type of the LLVM IR asm doesn't match the result type of
    // the expression, do the conversion.
    if (ResultRegTypes[i] != ResultTruncRegTypes[i]) {
      const llvm::Type *TruncTy = ResultTruncRegTypes[i];
      
      // Truncate the integer result to the right size, note that TruncTy can be
      // a pointer.
      if (TruncTy->isFloatingPointTy())
        Tmp = Builder.CreateFPTrunc(Tmp, TruncTy);
      else if (TruncTy->isPointerTy() && Tmp->getType()->isIntegerTy()) {
        uint64_t ResSize = CGM.getTargetData().getTypeSizeInBits(TruncTy);
        Tmp = Builder.CreateTrunc(Tmp, llvm::IntegerType::get(VMContext,
                                                            (unsigned)ResSize));
        Tmp = Builder.CreateIntToPtr(Tmp, TruncTy);
      } else if (Tmp->getType()->isPointerTy() && TruncTy->isIntegerTy()) {
        uint64_t TmpSize =CGM.getTargetData().getTypeSizeInBits(Tmp->getType());
        Tmp = Builder.CreatePtrToInt(Tmp, llvm::IntegerType::get(VMContext,
                                                            (unsigned)TmpSize));
        Tmp = Builder.CreateTrunc(Tmp, TruncTy);
      } else if (TruncTy->isIntegerTy()) {
        Tmp = Builder.CreateTrunc(Tmp, TruncTy);
      } else if (TruncTy->isVectorTy()) {
        Tmp = Builder.CreateBitCast(Tmp, TruncTy);
      }
    }

    EmitStoreThroughLValue(RValue::get(Tmp), ResultRegDests[i],
                           ResultRegQualTys[i]);
  }
}
