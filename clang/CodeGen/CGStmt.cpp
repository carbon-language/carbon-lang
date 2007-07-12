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
    // Must be an expression in a stmt context.  Emit the value and ignore the
    // result.
    if (const Expr *E = dyn_cast<Expr>(S)) {
      EmitExpr(E);
    } else {
      printf("Unimplemented stmt!\n");
      S->dump();
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
  }
}

void CodeGenFunction::EmitCompoundStmt(const CompoundStmt &S) {
  // FIXME: handle vla's etc.
  
  for (CompoundStmt::const_body_iterator I = S.body_begin(), E = S.body_end();
       I != E; ++I)
    EmitStmt(*I);
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
  Builder.CreateBr(ContBlock);
  
  // Emit the 'else' code if present.
  if (const Stmt *Else = S.getElse()) {
    EmitBlock(ElseBlock);
    EmitStmt(Else);
    Builder.CreateBr(ContBlock);
  }
  
  // Emit the continuation block for code after the if.
  EmitBlock(ContBlock);
}

void CodeGenFunction::EmitWhileStmt(const WhileStmt &S) {
  // FIXME: Handle continue/break.
  
  // Emit the header for the loop, insert it, which will create an uncond br to
  // it.
  llvm::BasicBlock *LoopHeader = new llvm::BasicBlock("whilecond");
  EmitBlock(LoopHeader);
  
  // Evaluate the conditional in the while header.  C99 6.8.5.1: The evaluation
  // of the controlling expression takes place before each execution of the loop
  // body. 
  llvm::Value *BoolCondVal = EvaluateExprAsBool(S.getCond());
  
  // TODO: while(1) is common, avoid extra exit blocks, etc.  Be sure
  // to correctly handle break/continue though.
  
  // Create an exit block for when the condition fails, create a block for the
  // body of the loop.
  llvm::BasicBlock *ExitBlock = new llvm::BasicBlock("whileexit");
  llvm::BasicBlock *LoopBody  = new llvm::BasicBlock("whilebody");
  
  // As long as the condition is true, go to the loop body.
  Builder.CreateCondBr(BoolCondVal, LoopBody, ExitBlock);
  
  // Emit the loop body.
  EmitBlock(LoopBody);
  EmitStmt(S.getBody());
  
  // Cycle to the condition.
  Builder.CreateBr(LoopHeader);
  
  // Emit the exit block.
  EmitBlock(ExitBlock);
}

void CodeGenFunction::EmitDoStmt(const DoStmt &S) {
  // FIXME: Handle continue/break.
  // TODO: "do {} while (0)" is common in macros, avoid extra blocks.  Be sure
  // to correctly handle break/continue though.

  // Emit the body for the loop, insert it, which will create an uncond br to
  // it.
  llvm::BasicBlock *LoopBody = new llvm::BasicBlock("dobody");
  llvm::BasicBlock *AfterDo = new llvm::BasicBlock("afterdo");
  EmitBlock(LoopBody);
  
  // Emit the body of the loop into the block.
  EmitStmt(S.getBody());
  
  // C99 6.8.5.2: "The evaluation of the controlling expression takes place
  // after each execution of the loop body."
  
  // Evaluate the conditional in the while header.
  // C99 6.8.5p2/p4: The first substatement is executed if the expression
  // compares unequal to 0.  The condition must be a scalar type.
  llvm::Value *BoolCondVal = EvaluateExprAsBool(S.getCond());
  
  // As long as the condition is true, iterate the loop.
  Builder.CreateCondBr(BoolCondVal, LoopBody, AfterDo);
  
  // Emit the exit block.
  EmitBlock(AfterDo);
}

void CodeGenFunction::EmitForStmt(const ForStmt &S) {
  // FIXME: Handle continue/break.
  // FIXME: What do we do if the increment (f.e.) contains a stmt expression,
  // which contains a continue/break?
  
  // Evaluate the first part before the loop.
  if (S.getInit())
    EmitStmt(S.getInit());

  // Start the loop with a block that tests the condition.
  llvm::BasicBlock *CondBlock = new llvm::BasicBlock("forcond");
  llvm::BasicBlock *AfterFor = 0;
  EmitBlock(CondBlock);

  // Evaluate the condition if present.  If not, treat it as a non-zero-constant
  // according to 6.8.5.3p2, aka, true.
  if (S.getCond()) {
    // C99 6.8.5p2/p4: The first substatement is executed if the expression
    // compares unequal to 0.  The condition must be a scalar type.
    llvm::Value *BoolCondVal = EvaluateExprAsBool(S.getCond());
    
    // As long as the condition is true, iterate the loop.
    llvm::BasicBlock *ForBody = new llvm::BasicBlock("forbody");
    AfterFor = new llvm::BasicBlock("afterfor");
    Builder.CreateCondBr(BoolCondVal, ForBody, AfterFor);
    EmitBlock(ForBody);    
  } else {
    // Treat it as a non-zero constant.  Don't even create a new block for the
    // body, just fall into it.
  }

  // If the condition is true, execute the body of the for stmt.
  EmitStmt(S.getBody());
  
  // If there is an increment, emit it next.
  if (S.getInc())
    EmitExpr(S.getInc());
      
  // Finally, branch back up to the condition for the next iteration.
  Builder.CreateBr(CondBlock);

  // Emit the fall-through block if there is any.
  if (AfterFor) 
    EmitBlock(AfterFor);
  else
    EmitBlock(new llvm::BasicBlock());
}

/// EmitReturnStmt - Note that due to GCC extensions, this can have an operand
/// if the function returns void, or may be missing one if the function returns
/// non-void.  Fun stuff :).
void CodeGenFunction::EmitReturnStmt(const ReturnStmt &S) {
  RValue RetVal;
  
  // Emit the result value, even if unused, to evalute the side effects.
  const Expr *RV = S.getRetValue();
  if (RV)
    RetVal = EmitExpr(RV);
  
  QualType FnRetTy = CurFuncDecl->getType().getCanonicalType();
  FnRetTy = cast<FunctionType>(FnRetTy)->getResultType();
  
  if (FnRetTy->isVoidType()) {
    // If the function returns void, emit ret void, and ignore the retval.
    Builder.CreateRetVoid();
  } else if (RV == 0) {
    // "return;" in a function that returns a value.
    const llvm::Type *RetTy = CurFn->getFunctionType()->getReturnType();
    if (RetTy == llvm::Type::VoidTy)
      Builder.CreateRetVoid();   // struct return etc.
    else
      Builder.CreateRet(llvm::UndefValue::get(RetTy));
  } else {
    // Do implicit conversions to the returned type.
    RetVal = EmitConversion(RetVal, RV->getType(), FnRetTy);
    
    if (RetVal.isScalar()) {
      Builder.CreateRet(RetVal.getVal());
    } else {
      llvm::Value *SRetPtr = CurFn->arg_begin();
      EmitStoreThroughLValue(RetVal, LValue::MakeAddr(SRetPtr), FnRetTy);
    }
  }
  
  // Emit a block after the branch so that dead code after a return has some
  // place to go.
  EmitBlock(new llvm::BasicBlock());
}

void CodeGenFunction::EmitDeclStmt(const DeclStmt &S) {
  for (const Decl *Decl = S.getDecl(); Decl; Decl = Decl->getNextDeclarator())
    EmitDecl(*Decl);
}
