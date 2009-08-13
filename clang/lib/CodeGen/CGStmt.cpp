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
    DI->setLocation(S->getLocStart());
    DI->EmitStopPoint(CurFn, Builder);
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
  default:
    // Must be an expression in a stmt context.  Emit the value (to get
    // side-effects) and ignore the result.
    if (!isa<Expr>(S))
      ErrorUnsupported(S, "statement");

    EmitAnyExpr(cast<Expr>(S), 0, false, true);
    
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
                                         llvm::Value *AggLoc, bool isAggVol) {
  PrettyStackTraceLoc CrashInfo(getContext().getSourceManager(),S.getLBracLoc(),
                             "LLVM IR generation of compound statement ('{}')");
  
  CGDebugInfo *DI = getDebugInfo();
  if (DI) {
    EnsureInsertPoint();
    DI->setLocation(S.getLBracLoc());
    // FIXME: The llvm backend is currently not ready to deal with region_end
    // for block scoping.  In the presence of always_inline functions it gets so
    // confused that it doesn't emit any debug info.  Just disable this for now.
    //DI->EmitRegionStart(CurFn, Builder);
  }

  // Keep track of the current cleanup stack depth.
  size_t CleanupStackDepth = CleanupEntries.size();
  bool OldDidCallStackSave = DidCallStackSave;
  DidCallStackSave = false;
  
  for (CompoundStmt::const_body_iterator I = S.body_begin(),
       E = S.body_end()-GetLast; I != E; ++I)
    EmitStmt(*I);

  if (DI) {
    EnsureInsertPoint();
    DI->setLocation(S.getRBracLoc());
    
    // FIXME: The llvm backend is currently not ready to deal with region_end
    // for block scoping.  In the presence of always_inline functions it gets so
    // confused that it doesn't emit any debug info.  Just disable this for now.
    //DI->EmitRegionEnd(CurFn, Builder);
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
    
    RV = EmitAnyExpr(cast<Expr>(LastStmt), AggLoc);
  }

  DidCallStackSave = OldDidCallStackSave;
  
  EmitCleanupBlocks(CleanupStackDepth);
  
  return RV;
}

void CodeGenFunction::SimplifyForwardingBlocks(llvm::BasicBlock *BB) {
  llvm::BranchInst *BI = dyn_cast<llvm::BranchInst>(BB->getTerminator());
  
  // If there is a cleanup stack, then we it isn't worth trying to
  // simplify this block (we would need to remove it from the scope map
  // and cleanup entry).
  if (!CleanupEntries.empty())
    return;

  // Can only simplify direct branches.
  if (!BI || !BI->isUnconditional())
    return;

  BB->replaceAllUsesWith(BI->getSuccessor(0));
  BI->eraseFromParent();
  BB->eraseFromParent();
}

void CodeGenFunction::EmitBlock(llvm::BasicBlock *BB, bool IsFinished) {
  // Fall out of the current block (if necessary).
  EmitBranch(BB);

  if (IsFinished && BB->use_empty()) {
    delete BB;
    return;
  }

  // If necessary, associate the block with the cleanup stack size.
  if (!CleanupEntries.empty()) {
    // Check if the basic block has already been inserted.
    BlockScopeMap::iterator I = BlockScopes.find(BB);
    if (I != BlockScopes.end()) {
      assert(I->second == CleanupEntries.size() - 1);
    } else {
      BlockScopes[BB] = CleanupEntries.size() - 1;
      CleanupEntries.back().Blocks.push_back(BB);
    }
  }
  
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

void CodeGenFunction::EmitLabel(const LabelStmt &S) {
  EmitBlock(getBasicBlockForLabel(&S));
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

  EmitBranchThroughCleanup(getBasicBlockForLabel(S.getLabel()));
}

void CodeGenFunction::EmitIndirectGotoStmt(const IndirectGotoStmt &S) {
  // Emit initial switch which will be patched up later by
  // EmitIndirectSwitches(). We need a default dest, so we use the
  // current BB, but this is overwritten.
  llvm::Value *V = Builder.CreatePtrToInt(EmitScalarExpr(S.getTarget()),
                                          llvm::Type::getInt32Ty(VMContext), 
                                          "addr");
  llvm::SwitchInst *I = Builder.CreateSwitch(V, Builder.GetInsertBlock());
  IndirectSwitches.push_back(I);

  // Clear the insertion point to indicate we are in unreachable code.
  Builder.ClearInsertionPoint();
}

void CodeGenFunction::EmitIfStmt(const IfStmt &S) {
  // C99 6.8.4.1: The first substatement is executed if the expression compares
  // unequal to 0.  The condition must be a scalar type.
  
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
      if (Executed)
        EmitStmt(Executed);
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
  EmitStmt(S.getThen());
  EmitBranch(ContBlock);
  
  // Emit the 'else' code if present.
  if (const Stmt *Else = S.getElse()) {
    EmitBlock(ElseBlock);
    EmitStmt(Else);
    EmitBranch(ContBlock);
  }
  
  // Emit the continuation block for code after the if.
  EmitBlock(ContBlock, true);
}

void CodeGenFunction::EmitWhileStmt(const WhileStmt &S) {
  // Emit the header for the loop, insert it, which will create an uncond br to
  // it.
  llvm::BasicBlock *LoopHeader = createBasicBlock("while.cond");
  EmitBlock(LoopHeader);

  // Create an exit block for when the condition fails, create a block for the
  // body of the loop.
  llvm::BasicBlock *ExitBlock = createBasicBlock("while.end");
  llvm::BasicBlock *LoopBody  = createBasicBlock("while.body");

  // Store the blocks to use for break and continue.
  BreakContinueStack.push_back(BreakContinue(ExitBlock, LoopHeader));
  
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
  if (EmitBoolCondBranch)
    Builder.CreateCondBr(BoolCondVal, LoopBody, ExitBlock);
  
  // Emit the loop body.
  EmitBlock(LoopBody);
  EmitStmt(S.getBody());

  BreakContinueStack.pop_back();  
  
  // Cycle to the condition.
  EmitBranch(LoopHeader);
  
  // Emit the exit block.
  EmitBlock(ExitBlock, true);

  // The LoopHeader typically is just a branch if we skipped emitting
  // a branch, try to erase it.
  if (!EmitBoolCondBranch)
    SimplifyForwardingBlocks(LoopHeader);
}

void CodeGenFunction::EmitDoStmt(const DoStmt &S) {
  // Emit the body for the loop, insert it, which will create an uncond br to
  // it.
  llvm::BasicBlock *LoopBody = createBasicBlock("do.body");
  llvm::BasicBlock *AfterDo = createBasicBlock("do.end");
  EmitBlock(LoopBody);

  llvm::BasicBlock *DoCond = createBasicBlock("do.cond");
  
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

  // The DoCond block typically is just a branch if we skipped
  // emitting a branch, try to erase it.
  if (!EmitBoolCondBranch)
    SimplifyForwardingBlocks(DoCond);
}

void CodeGenFunction::EmitForStmt(const ForStmt &S) {
  // FIXME: What do we do if the increment (f.e.) contains a stmt expression,
  // which contains a continue/break?

  // Evaluate the first part before the loop.
  if (S.getInit())
    EmitStmt(S.getInit());

  // Start the loop with a block that tests the condition.
  llvm::BasicBlock *CondBlock = createBasicBlock("for.cond");
  llvm::BasicBlock *AfterFor = createBasicBlock("for.end");

  EmitBlock(CondBlock);

  // Evaluate the condition if present.  If not, treat it as a
  // non-zero-constant according to 6.8.5.3p2, aka, true.
  if (S.getCond()) {
    // As long as the condition is true, iterate the loop.
    llvm::BasicBlock *ForBody = createBasicBlock("for.body");
    
    // C99 6.8.5p2/p4: The first substatement is executed if the expression
    // compares unequal to 0.  The condition must be a scalar type.
    EmitBranchOnBoolExpr(S.getCond(), ForBody, AfterFor);
    
    EmitBlock(ForBody);    
  } else {
    // Treat it as a non-zero constant.  Don't even create a new block for the
    // body, just fall into it.
  }

  // If the for loop doesn't have an increment we can just use the 
  // condition as the continue block.
  llvm::BasicBlock *ContinueBlock;
  if (S.getInc())
    ContinueBlock = createBasicBlock("for.inc");
  else
    ContinueBlock = CondBlock;  
  
  // Store the blocks to use for break and continue.
  BreakContinueStack.push_back(BreakContinue(AfterFor, ContinueBlock));

  // If the condition is true, execute the body of the for stmt.
  EmitStmt(S.getBody());

  BreakContinueStack.pop_back();
  
  // If there is an increment, emit it next.
  if (S.getInc()) {
    EmitBlock(ContinueBlock);
    EmitStmt(S.getInc());
  }
      
  // Finally, branch back up to the condition for the next iteration.
  EmitBranch(CondBlock);

  // Emit the fall-through block.
  EmitBlock(AfterFor, true);
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
  if (!ReturnValue) {
    // Make sure not to return anything, but evaluate the expression
    // for side effects.
    if (RV)
      EmitAnyExpr(RV);
  } else if (RV == 0) {
    // Do nothing (return value is left uninitialized)
  } else if (FnRetTy->isReferenceType()) {
    // If this function returns a reference, take the address of the expression
    // rather than the value.
    Builder.CreateStore(EmitLValue(RV).getAddress(), ReturnValue);
  } else if (!hasAggregateLLVMType(RV->getType())) {
    Builder.CreateStore(EmitScalarExpr(RV), ReturnValue);
  } else if (RV->getType()->isAnyComplexType()) {
    EmitComplexExprIntoAddr(RV, ReturnValue, false);
  } else {
    EmitAggExpr(RV, ReturnValue, false);
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

  llvm::BasicBlock *Block = BreakContinueStack.back().BreakBlock;
  EmitBranchThroughCleanup(Block);
}

void CodeGenFunction::EmitContinueStmt(const ContinueStmt &S) {
  assert(!BreakContinueStack.empty() && "continue stmt not in a loop!");

  // If this code is reachable then emit a stop point (if generating
  // debug info). We have to do this ourselves because we are on the
  // "simple" statement path.
  if (HaveInsertPoint())
    EmitStopPoint(&S);

  llvm::BasicBlock *Block = BreakContinueStack.back().ContinueBlock;
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
  llvm::Value *CondV = EmitScalarExpr(S.getCond());

  // Handle nested switch statements.
  llvm::SwitchInst *SavedSwitchInsn = SwitchInsn;
  llvm::BasicBlock *SavedCRBlock = CaseRangeBlock;

  // Create basic block to hold stuff that comes after switch
  // statement. We also need to create a default block now so that
  // explicit case ranges tests can have a place to jump to on
  // failure.
  llvm::BasicBlock *NextBlock = createBasicBlock("sw.epilog");
  llvm::BasicBlock *DefaultBlock = createBasicBlock("sw.default");
  SwitchInsn = Builder.CreateSwitch(CondV, DefaultBlock);
  CaseRangeBlock = DefaultBlock;

  // Clear the insertion point to indicate we are in unreachable code.
  Builder.ClearInsertionPoint();

  // All break statements jump to NextBlock. If BreakContinueStack is non empty
  // then reuse last ContinueBlock.
  llvm::BasicBlock *ContinueBlock = 0;
  if (!BreakContinueStack.empty())
    ContinueBlock = BreakContinueStack.back().ContinueBlock;

  // Ensure any vlas created between there and here, are undone
  BreakContinueStack.push_back(BreakContinue(NextBlock, ContinueBlock));

  // Emit switch body.
  EmitStmt(S.getBody());
  
  BreakContinueStack.pop_back();

  // Update the default block in case explicit case range tests have
  // been chained on top.
  SwitchInsn->setSuccessor(0, CaseRangeBlock);
  
  // If a default was never emitted then reroute any jumps to it and
  // discard.
  if (!DefaultBlock->getParent()) {
    DefaultBlock->replaceAllUsesWith(NextBlock);
    delete DefaultBlock;
  }

  // Emit continuation.
  EmitBlock(NextBlock, true);

  SwitchInsn = SavedSwitchInsn;
  CaseRangeBlock = SavedCRBlock;
}

static std::string
SimplifyConstraint(const char *Constraint, TargetInfo &Target,
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

llvm::Value* CodeGenFunction::EmitAsmInput(const AsmStmt &S,
                                         const TargetInfo::ConstraintInfo &Info,
                                           const Expr *InputExpr,
                                           std::string &ConstraintStr) {
  llvm::Value *Arg;
  if (Info.allowsRegister() || !Info.allowsMemory()) { 
    const llvm::Type *Ty = ConvertType(InputExpr->getType());
    
    if (Ty->isSingleValueType()) {
      Arg = EmitScalarExpr(InputExpr);
    } else {
      InputExpr = InputExpr->IgnoreParenNoopCasts(getContext());
      LValue Dest = EmitLValue(InputExpr);

      uint64_t Size = CGM.getTargetData().getTypeSizeInBits(Ty);
      if (Size <= 64 && llvm::isPowerOf2_64(Size)) {
        Ty = llvm::IntegerType::get(VMContext, Size);
        Ty = llvm::PointerType::getUnqual(Ty);
        
        Arg = Builder.CreateLoad(Builder.CreateBitCast(Dest.getAddress(), Ty));
      } else {
        Arg = Dest.getAddress();
        ConstraintStr += '*';
      }
    }
  } else {
    InputExpr = InputExpr->IgnoreParenNoopCasts(getContext());
    LValue Dest = EmitLValue(InputExpr);
    Arg = Dest.getAddress();
    ConstraintStr += '*';
  }
  
  return Arg;
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
    bool result = Target.validateOutputConstraint(Info);
    assert(result && "Failed to parse output constraint"); result=result;
    OutputConstraintInfos.push_back(Info);
  }    
  
  for (unsigned i = 0, e = S.getNumInputs(); i != e; i++) {
    TargetInfo::ConstraintInfo Info(S.getInputConstraint(i),
                                    S.getInputName(i));
    bool result = Target.validateInputConstraint(OutputConstraintInfos.data(),
                                                 S.getNumOutputs(),
                                                 Info); result=result;
    assert(result && "Failed to parse input constraint");
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
          if (Input.hasTiedOperand() &&
              Input.getTiedOperand() == i)
            break;
        }
        assert(InputNo != S.getNumInputs() && "Didn't find matching input!");
        
        QualType InputTy = S.getInputExpr(InputNo)->getType();
        QualType OutputTy = OutExpr->getType();
        
        uint64_t InputSize = getContext().getTypeSize(InputTy);
        if (getContext().getTypeSize(OutputTy) < InputSize) {
          // Form the asm to return the value as a larger integer type.
          ResultRegTypes.back() = llvm::IntegerType::get(VMContext, (unsigned)InputSize);
        }
      }
    } else {
      ArgTypes.push_back(Dest.getAddress()->getType());
      Args.push_back(Dest.getAddress());
      Constraints += "=*";
      Constraints += OutputConstraint;
    }
    
    if (Info.isReadWrite()) {
      InOutConstraints += ',';

      const Expr *InputExpr = S.getOutputExpr(i);
      llvm::Value *Arg = EmitAsmInput(S, Info, InputExpr, InOutConstraints);
      
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

    llvm::Value *Arg = EmitAsmInput(S, Info, InputExpr, Constraints);
    
    // If this input argument is tied to a larger output result, extend the
    // input to be the same size as the output.  The LLVM backend wants to see
    // the input and output of a matching constraint be the same size.  Note
    // that GCC does not define what the top bits are here.  We use zext because
    // that is usually cheaper, but LLVM IR should really get an anyext someday.
    if (Info.hasTiedOperand()) {
      unsigned Output = Info.getTiedOperand();
      QualType OutputTy = S.getOutputExpr(Output)->getType();
      QualType InputTy = InputExpr->getType();
      
      if (getContext().getTypeSize(OutputTy) >
          getContext().getTypeSize(InputTy)) {
        // Use ptrtoint as appropriate so that we can do our extension.
        if (isa<llvm::PointerType>(Arg->getType()))
          Arg = Builder.CreatePtrToInt(Arg,
                                      llvm::IntegerType::get(VMContext, LLVMPointerWidth));
        unsigned OutputSize = (unsigned)getContext().getTypeSize(OutputTy);
        Arg = Builder.CreateZExt(Arg, llvm::IntegerType::get(VMContext, OutputSize));
      }
    }
    
    
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
    std::string Clobber(S.getClobber(i)->getStrData(),
                        S.getClobber(i)->getByteLength());

    Clobber = Target.getNormalizedGCCRegisterName(Clobber.c_str());
    
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
      // Truncate the integer result to the right size, note that
      // ResultTruncRegTypes can be a pointer.
      uint64_t ResSize = CGM.getTargetData().getTypeSizeInBits(TruncTy);
      Tmp = Builder.CreateTrunc(Tmp, llvm::IntegerType::get(VMContext, (unsigned)ResSize));
      
      if (Tmp->getType() != TruncTy) {
        assert(isa<llvm::PointerType>(TruncTy));
        Tmp = Builder.CreateIntToPtr(Tmp, TruncTy);
      }
    }
    
    EmitStoreThroughLValue(RValue::get(Tmp), ResultRegDests[i],
                           ResultRegQualTys[i]);
  }
}
