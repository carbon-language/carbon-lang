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
using namespace llvm;
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
  }
}

void CodeGenFunction::EmitCompoundStmt(const CompoundStmt &S) {
  // FIXME: handle vla's etc.
  
  for (CompoundStmt::const_body_iterator I = S.body_begin(), E = S.body_end();
       I != E; ++I)
    EmitStmt(*I);
}

void CodeGenFunction::EmitBlock(BasicBlock *BB) {
  // Emit a branch from this block to the next one if this was a real block.  If
  // this was just a fall-through block after a terminator, don't emit it.
  BasicBlock *LastBB = Builder.GetInsertBlock();
  
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
  Builder.SetInsertPoint(new BasicBlock("", CurFn));
}

void CodeGenFunction::EmitIfStmt(const IfStmt &S) {
  // Emit the if condition.
  ExprResult CondVal = EmitExpr(S.getCond());
  QualType CondTy = S.getCond()->getType().getCanonicalType();
  
  // C99 6.8.4.1: The first substatement is executed if the expression compares
  // unequal to 0.  The condition must be a scalar type.
  llvm::Value *BoolCondVal;
  
  // MOVE this to a helper method, to share with for/while, assign to bool, etc.
  if (const BuiltinType *BT = dyn_cast<BuiltinType>(CondTy)) {
    switch (BT->getKind()) {
    default: assert(0 && "Unknown scalar value");
    case BuiltinType::Bool:
      BoolCondVal = CondVal.getVal();
      // Bool is already evaluated right.
      assert(BoolCondVal->getType() == llvm::Type::Int1Ty &&
             "Unexpected bool value type!");
      break;
    case BuiltinType::Char:
    case BuiltinType::SChar:
    case BuiltinType::UChar:
    case BuiltinType::Int:
    case BuiltinType::UInt:
    case BuiltinType::Long:
    case BuiltinType::ULong:
    case BuiltinType::LongLong:
    case BuiltinType::ULongLong: {
      // Compare against zero for integers.
      BoolCondVal = CondVal.getVal();
      llvm::Value *Zero = Constant::getNullValue(BoolCondVal->getType());
      BoolCondVal = Builder.CreateICmpNE(BoolCondVal, Zero);
      break;
    }
    case BuiltinType::Float:
    case BuiltinType::Double:
    case BuiltinType::LongDouble: {
      // Compare against 0.0 for fp scalars.
      BoolCondVal = CondVal.getVal();
      llvm::Value *Zero = Constant::getNullValue(BoolCondVal->getType());
      // FIXME: llvm-gcc produces a une comparison: validate this is right.
      BoolCondVal = Builder.CreateFCmpUNE(BoolCondVal, Zero);
      break;
    }
      
    case BuiltinType::FloatComplex:
    case BuiltinType::DoubleComplex:
    case BuiltinType::LongDoubleComplex:
      assert(0 && "comparisons against complex not implemented yet");
    }
  } else if (isa<PointerType>(CondTy)) {
    BoolCondVal = CondVal.getVal();
    llvm::Value *NullPtr = Constant::getNullValue(BoolCondVal->getType());
    BoolCondVal = Builder.CreateICmpNE(BoolCondVal, NullPtr);
    
  } else {
    const TagType *TT = cast<TagType>(CondTy);
    assert(TT->getDecl()->getKind() == Decl::Enum && "Unknown scalar type");
    // Compare against zero.
    BoolCondVal = CondVal.getVal();
    llvm::Value *Zero = Constant::getNullValue(BoolCondVal->getType());
    BoolCondVal = Builder.CreateICmpNE(BoolCondVal, Zero);
  }
  
  BasicBlock *ContBlock = new BasicBlock("ifend");
  BasicBlock *ThenBlock = new BasicBlock("ifthen");
  BasicBlock *ElseBlock = ContBlock;
  
  if (S.getElse())
    ElseBlock = new BasicBlock("ifelse");
  
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

