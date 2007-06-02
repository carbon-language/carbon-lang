//===--- CodeGenFunction.h - Per-Function state for LLVM CodeGen ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is the internal per-function state used for llvm translation. 
//
//===----------------------------------------------------------------------===//

#ifndef CODEGEN_CODEGENFUNCTION_H
#define CODEGEN_CODEGENFUNCTION_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/LLVMBuilder.h"

namespace llvm {
  class Module;
namespace clang {
  class ASTContext;
  class Decl;
  class FunctionDecl;
  class QualType;
  class SourceLocation;
  class TargetInfo;
  
  class Stmt;
  class CompoundStmt;
  class LabelStmt;
  class GotoStmt;
  class IfStmt;
  class ReturnStmt;
  class DeclStmt;
  
  class Expr;
  class DeclRefExpr;
  class IntegerLiteral;
  class UnaryOperator;
  class BinaryOperator;
  
  class BlockVarDecl;
  class EnumConstantDecl;
namespace CodeGen {
  class CodeGenModule;
  

/// ExprResult - This trivial value class is used to represent the result of an
/// expression that is evaluated.  It can be one of two things: either a simple
/// LLVM SSA value, or the address of an aggregate value in memory.  These two
/// possibilities are discriminated by isAggregate/isScalar.
class ExprResult {
  Value *V;
  // TODO: Encode this into the low bit of pointer for more efficient
  // return-by-value.
  bool IsAggregate;
public:
  
  bool isAggregate() const { return IsAggregate; }
  bool isScalar() const { return !IsAggregate; }
  
  /// getVal() - Return the Value* of this scalar value.
  Value *getVal() const {
    assert(!isAggregate() && "Not a scalar!");
    return V;
  }

  /// getAggregateVal() - Return the Value* of the address of the aggregate.
  Value *getAggregateVal() const {
    assert(isAggregate() && "Not an aggregate!");
    return V;
  }
  
  static ExprResult get(Value *V) {
    ExprResult ER;
    ER.V = V;
    ER.IsAggregate = false;
    return ER;
  }
  static ExprResult getAggregate(Value *V) {
    ExprResult ER;
    ER.V = V;
    ER.IsAggregate = true;
    return ER;
  }
};


/// LValue - This represents an lvalue references.  Because C/C++ allow
/// bitfields, this is not a simple LLVM pointer, it may be a pointer plus a
/// bitrange.
class LValue {
  // FIXME: Volatility.  Restrict?
  llvm::Value *V;
public:
  bool isBitfield() const { return false; }
  
  llvm::Value *getAddress() const { assert(!isBitfield()); return V; }
  
  static LValue getAddr(Value *V) {
    LValue R;
    R.V = V;
    return R;
  }
};

/// CodeGenFunction - This class organizes the per-function state that is used
/// while generating LLVM code.
class CodeGenFunction {
  CodeGenModule &CGM;  // Per-module state.
  TargetInfo &Target;
  LLVMBuilder Builder;
  
  const FunctionDecl *CurFuncDecl;
  llvm::Function *CurFn;

  /// AllocaInsertPoint - This is an instruction in the entry block before which
  /// we prefer to insert allocas.
  llvm::Instruction *AllocaInsertPt;
  
  /// LocalDeclMap - This keeps track of the LLVM allocas or globals for local C
  /// decls.
  DenseMap<const Decl*, llvm::Value*> LocalDeclMap;

  /// LabelMap - This keeps track of the LLVM basic block for each C label.
  DenseMap<const LabelStmt*, llvm::BasicBlock*> LabelMap;
  
  const llvm::Type *LLVMIntTy;
public:
  CodeGenFunction(CodeGenModule &cgm);
  
  const llvm::Type *ConvertType(QualType T, SourceLocation Loc);
  
  void GenerateCode(const FunctionDecl *FD);
  
  
  /// getBasicBlockForLabel - Return the LLVM basicblock that the specified
  /// label maps to.
  llvm::BasicBlock *getBasicBlockForLabel(const LabelStmt *S);
  
  
  void EmitBlock(BasicBlock *BB);

  
  /// EvaluateScalarValueToBool - Evaluate the specified expression value to a
  /// boolean (i1) truth value.  This is equivalent to "Val == 0".
  Value *EvaluateScalarValueToBool(ExprResult Val, QualType Ty);
  
  //===--------------------------------------------------------------------===//
  //                        Local Declaration Emission
  //===--------------------------------------------------------------------===//
  
  void EmitDeclStmt(const DeclStmt &S);
  void EmitEnumConstantDecl(const EnumConstantDecl &D);
  void EmitBlockVarDecl(const BlockVarDecl &D);
  void EmitLocalBlockVarDecl(const BlockVarDecl &D);
  
  //===--------------------------------------------------------------------===//
  //                             Statement Emission
  //===--------------------------------------------------------------------===//

  void EmitStmt(const Stmt *S);
  void EmitCompoundStmt(const CompoundStmt &S);
  void EmitLabelStmt(const LabelStmt &S);
  void EmitGotoStmt(const GotoStmt &S);
  void EmitIfStmt(const IfStmt &S);
  void EmitReturnStmt(const ReturnStmt &S);
  
  //===--------------------------------------------------------------------===//
  //                         LValue Expression Emission
  //===--------------------------------------------------------------------===//
  
  LValue EmitLValue(const Expr *E);
  LValue EmitDeclRefLValue(const DeclRefExpr *E);
    
  //===--------------------------------------------------------------------===//
  //                             Expression Emission
  //===--------------------------------------------------------------------===//

  ExprResult EmitExpr(const Expr *E);
  ExprResult EmitIntegerLiteral(const IntegerLiteral *E);
  

  void EmitUsualArithmeticConversions(const BinaryOperator *E,
                                      ExprResult &LHS, ExprResult &RHS);
  
  // Unary Operators.
  ExprResult EmitUnaryOperator(const UnaryOperator *E);
  ExprResult EmitUnaryLNot(const UnaryOperator *E);
  
  // Binary Operators.
  ExprResult EmitBinaryOperator(const BinaryOperator *E);
  ExprResult EmitBinaryAdd(const BinaryOperator *E);
};
}  // end namespace CodeGen
}  // end namespace clang
}  // end namespace llvm

#endif
