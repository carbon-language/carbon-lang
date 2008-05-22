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
#include "clang/AST/AST.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/InlineAsm.h"
#include "llvm/ADT/StringExtras.h"
using namespace clang;
using namespace CodeGen;

//===----------------------------------------------------------------------===//
//                              Statement Emission
//===----------------------------------------------------------------------===//

void CodeGenFunction::EmitStmt(const Stmt *S) {
  assert(S && "Null statement?");
  
  // Generate stoppoints if we are emitting debug info.
  // Beginning of a Compound Statement (e.g. an opening '{') does not produce 
  // executable code. So do not generate a stoppoint for that.
  CGDebugInfo *DI = CGM.getDebugInfo();
  if (DI && S->getStmtClass() != Stmt::CompoundStmtClass) {
    if (S->getLocStart().isValid()) {
        DI->setLocation(S->getLocStart());
    }

    DI->EmitStopPoint(CurFn, Builder);
  }

  switch (S->getStmtClass()) {
  default:
    // Must be an expression in a stmt context.  Emit the value (to get
    // side-effects) and ignore the result.
    if (const Expr *E = dyn_cast<Expr>(S)) {
      if (!hasAggregateLLVMType(E->getType()))
        EmitScalarExpr(E);
      else if (E->getType()->isAnyComplexType())
        EmitComplexExpr(E);
      else
        EmitAggExpr(E, 0, false);
    } else {
      WarnUnsupported(S, "statement");
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
  case Stmt::AsmStmtClass:      EmitAsmStmt(cast<AsmStmt>(*S));           break;
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
  Builder.SetInsertPoint(llvm::BasicBlock::Create("", CurFn));
}

void CodeGenFunction::EmitIfStmt(const IfStmt &S) {
  // C99 6.8.4.1: The first substatement is executed if the expression compares
  // unequal to 0.  The condition must be a scalar type.
  llvm::Value *BoolCondVal = EvaluateExprAsBool(S.getCond());
  
  llvm::BasicBlock *ContBlock = llvm::BasicBlock::Create("ifend");
  llvm::BasicBlock *ThenBlock = llvm::BasicBlock::Create("ifthen");
  llvm::BasicBlock *ElseBlock = ContBlock;
  
  if (S.getElse())
    ElseBlock = llvm::BasicBlock::Create("ifelse");
  
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
  llvm::BasicBlock *LoopHeader = llvm::BasicBlock::Create("whilecond");
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
  llvm::BasicBlock *ExitBlock = llvm::BasicBlock::Create("whileexit");
  llvm::BasicBlock *LoopBody  = llvm::BasicBlock::Create("whilebody");
  
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
  llvm::BasicBlock *LoopBody = llvm::BasicBlock::Create("dobody");
  llvm::BasicBlock *AfterDo = llvm::BasicBlock::Create("afterdo");
  EmitBlock(LoopBody);

  llvm::BasicBlock *DoCond = llvm::BasicBlock::Create("docond");
  
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
  llvm::BasicBlock *CondBlock = llvm::BasicBlock::Create("forcond");
  llvm::BasicBlock *AfterFor = llvm::BasicBlock::Create("afterfor");

  EmitBlock(CondBlock);

  // Evaluate the condition if present.  If not, treat it as a non-zero-constant
  // according to 6.8.5.3p2, aka, true.
  if (S.getCond()) {
    // C99 6.8.5p2/p4: The first substatement is executed if the expression
    // compares unequal to 0.  The condition must be a scalar type.
    llvm::Value *BoolCondVal = EvaluateExprAsBool(S.getCond());
    
    // As long as the condition is true, iterate the loop.
    llvm::BasicBlock *ForBody = llvm::BasicBlock::Create("forbody");
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
    ContinueBlock = llvm::BasicBlock::Create("forinc");
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

  llvm::Value* RetValue = 0;
  if (FnRetTy->isVoidType()) {
    // Make sure not to return anything
    if (RV) {
      // Evaluate the expression for side effects
      EmitAnyExpr(RV);
    }
  } else if (RV == 0) {
    const llvm::Type *RetTy = CurFn->getFunctionType()->getReturnType();
    if (RetTy != llvm::Type::VoidTy) {
      // Handle "return;" in a function that returns a value.
      RetValue = llvm::UndefValue::get(RetTy);
    }
  } else if (!hasAggregateLLVMType(RV->getType())) {
    RetValue = EmitScalarExpr(RV);
  } else if (RV->getType()->isAnyComplexType()) {
    llvm::Value *SRetPtr = CurFn->arg_begin();
    EmitComplexExprIntoAddr(RV, SRetPtr, false);
  } else {
    llvm::Value *SRetPtr = CurFn->arg_begin();
    EmitAggExpr(RV, SRetPtr, false);
  }

  CGDebugInfo *DI = CGM.getDebugInfo();
  if (DI) {
    CompoundStmt* body = cast<CompoundStmt>(CurFuncDecl->getBody());
    if (body->getRBracLoc().isValid()) {
      DI->setLocation(body->getRBracLoc());
    }
    DI->EmitFunctionEnd(CurFn, Builder);
  }

  if (RetValue) {
    Builder.CreateRet(RetValue);
  } else {
    Builder.CreateRetVoid();
  }
  
  // Emit a block after the branch so that dead code after a return has some
  // place to go.
  EmitBlock(llvm::BasicBlock::Create());
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
  EmitBlock(llvm::BasicBlock::Create());
}

void CodeGenFunction::EmitContinueStmt() {
  assert(!BreakContinueStack.empty() && "continue stmt not in a loop!");

  llvm::BasicBlock *Block = BreakContinueStack.back().ContinueBlock;
  Builder.CreateBr(Block);
  EmitBlock(llvm::BasicBlock::Create());
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
  llvm::BasicBlock *CaseDest = llvm::BasicBlock::Create("sw.bb");

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
  llvm::BasicBlock *NextBlock = llvm::BasicBlock::Create("after.sw");
  SwitchInsn = Builder.CreateSwitch(CondV, NextBlock);

  // Create basic block for body of switch
  StartBlock("body.sw");

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

static inline std::string ConvertAsmString(const char *Start,
                                           unsigned NumOperands,
                                           bool IsSimple)
{
  static unsigned AsmCounter = 0;
  
  AsmCounter++;
  
  std::string Result;
  if (IsSimple) {
    while (*Start) {
      switch (*Start) {
      default:
        Result += *Start;
        break;
      case '$':
        Result += "$$";
        break;
      }
      
      Start++;
    }
    
    return Result;
  }
  
  while (*Start) {
    switch (*Start) {
    default:
      Result += *Start;
      break;
    case '$':
      Result += "$$";
      break;
    case '%':
      // Escaped character
      Start++;
      if (!*Start) {
        // FIXME: This should be caught during Sema.
        assert(0 && "Trailing '%' in asm string.");
      }
      
      char EscapedChar = *Start;
      if (EscapedChar == '%') {
        // Escaped percentage sign.
        Result += '%';
      }
      else if (EscapedChar == '=') {
        // Generate an unique ID.
        Result += llvm::utostr(AsmCounter);
      } else if (isdigit(EscapedChar)) {
        // %n - Assembler operand n
        char *End;
        
        unsigned long n = strtoul(Start, &End, 10);
        if (Start == End) {
          // FIXME: This should be caught during Sema.
          assert(0 && "Missing operand!");
        } else if (n >= NumOperands) {
          // FIXME: This should be caught during Sema.
          assert(0 && "Operand number out of range!");
        }
        
        Result += '$' + llvm::utostr(n);
        Start = End - 1;
      } else if (isalpha(EscapedChar)) {
        char *End;
        
        unsigned long n = strtoul(Start + 1, &End, 10);
        if (Start == End) {
          // FIXME: This should be caught during Sema.
          assert(0 && "Missing operand!");
        } else if (n >= NumOperands) {
          // FIXME: This should be caught during Sema.
          assert(0 && "Operand number out of range!");
        }
        
        Result += "${" + llvm::utostr(n) + ':' + EscapedChar + '}';
        Start = End - 1;
      } else {
        assert(0 && "Unhandled asm escaped character!");
      }
    }
    Start++;
  }
  
  return Result;
}

static std::string SimplifyConstraint(const char* Constraint,
                                      TargetInfo &Target) {
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
    }
    
    Constraint++;
  }
  
  return Result;
}

void CodeGenFunction::EmitAsmStmt(const AsmStmt &S) {
  std::string AsmString = 
    ConvertAsmString(std::string(S.getAsmString()->getStrData(),
                                 S.getAsmString()->getByteLength()).c_str(),
                     S.getNumOutputs() + S.getNumInputs(), S.isSimple());
  
  std::string Constraints;
  
  llvm::Value *ResultAddr = 0;
  const llvm::Type *ResultType = llvm::Type::VoidTy;
  
  std::vector<const llvm::Type*> ArgTypes;
  std::vector<llvm::Value*> Args;

  // Keep track of inout constraints.
  std::string InOutConstraints;
  std::vector<llvm::Value*> InOutArgs;
  std::vector<const llvm::Type*> InOutArgTypes;
  
  for (unsigned i = 0, e = S.getNumOutputs(); i != e; i++) {    
    std::string OutputConstraint(S.getOutputConstraint(i)->getStrData(),
                                 S.getOutputConstraint(i)->getByteLength());
    
    TargetInfo::ConstraintInfo Info;
    bool result = Target.validateOutputConstraint(OutputConstraint.c_str(), 
                                                  Info);
    assert(result && "Failed to parse output constraint");
    
    // Simplify the output constraint.
    OutputConstraint = SimplifyConstraint(OutputConstraint.c_str() + 1, Target);
    
    LValue Dest = EmitLValue(S.getOutputExpr(i));
    const llvm::Type *DestValueType = 
      cast<llvm::PointerType>(Dest.getAddress()->getType())->getElementType();
    
    // If the first output operand is not a memory dest, we'll
    // make it the return value.
    if (i == 0 && !(Info & TargetInfo::CI_AllowsMemory) &&
        DestValueType->isSingleValueType()) {
      ResultAddr = Dest.getAddress();
      ResultType = DestValueType;
      Constraints += "=" + OutputConstraint;
    } else {
      ArgTypes.push_back(Dest.getAddress()->getType());
      Args.push_back(Dest.getAddress());
      if (i != 0)
        Constraints += ',';
      Constraints += "=*";
      Constraints += OutputConstraint;
    }
    
    if (Info & TargetInfo::CI_ReadWrite) {
      // FIXME: This code should be shared with the code that handles inputs.
      InOutConstraints += ',';
      
      const Expr *InputExpr = S.getOutputExpr(i);
      llvm::Value *Arg;
      if ((Info & TargetInfo::CI_AllowsRegister) ||
          !(Info & TargetInfo::CI_AllowsMemory)) {      
        if (ConvertType(InputExpr->getType())->isSingleValueType()) {
          Arg = EmitScalarExpr(InputExpr);
        } else {
          assert(0 && "FIXME: Implement passing multiple-value types as inputs");
        }
      } else {
        LValue Dest = EmitLValue(InputExpr);
        Arg = Dest.getAddress();
        InOutConstraints += '*';
      }
      
      InOutArgTypes.push_back(Arg->getType());
      InOutArgs.push_back(Arg);
      InOutConstraints += OutputConstraint;
    }
  }
  
  unsigned NumConstraints = S.getNumOutputs() + S.getNumInputs();
  
  for (unsigned i = 0, e = S.getNumInputs(); i != e; i++) {
    const Expr *InputExpr = S.getInputExpr(i);

    std::string InputConstraint(S.getInputConstraint(i)->getStrData(),
                                S.getInputConstraint(i)->getByteLength());
    
    TargetInfo::ConstraintInfo Info;
    bool result = Target.validateInputConstraint(InputConstraint.c_str(),
                                                 NumConstraints, 
                                                 Info);
    assert(result && "Failed to parse input constraint");
    
    if (i != 0 || S.getNumOutputs() > 0)
      Constraints += ',';
    
    // Simplify the input constraint.
    InputConstraint = SimplifyConstraint(InputConstraint.c_str(), Target);

    llvm::Value *Arg;
    
    if ((Info & TargetInfo::CI_AllowsRegister) ||
        !(Info & TargetInfo::CI_AllowsMemory)) {      
      if (ConvertType(InputExpr->getType())->isSingleValueType()) {
        Arg = EmitScalarExpr(InputExpr);
      } else {
        assert(0 && "FIXME: Implement passing multiple-value types as inputs");
      }
    } else {
      LValue Dest = EmitLValue(InputExpr);
      Arg = Dest.getAddress();
      Constraints += '*';
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
  if (const char *C = Target.getClobbers()) {
    if (!Constraints.empty())
      Constraints += ',';
    Constraints += C;
  }
    
  const llvm::FunctionType *FTy = 
    llvm::FunctionType::get(ResultType, ArgTypes, false);
  
  llvm::InlineAsm *IA = 
    llvm::InlineAsm::get(FTy, AsmString, Constraints, 
                         S.isVolatile() || S.getNumOutputs() == 0);
  llvm::Value *Result = Builder.CreateCall(IA, Args.begin(), Args.end(), "");
  if (ResultAddr)
    Builder.CreateStore(Result, ResultAddr);
}
