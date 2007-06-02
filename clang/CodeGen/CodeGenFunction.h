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
  
  class Expr;
  class IntegerLiteral;
  class BinaryOperator;
  
namespace CodeGen {
  class CodeGenModule;
  
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
  
/// CodeGenFunction - This class organizes the per-function state that is used
/// while generating LLVM code.
class CodeGenFunction {
  CodeGenModule &CGM;  // Per-module state.
  TargetInfo &Target;
  LLVMBuilder Builder;
  
  const FunctionDecl *CurFuncDecl;
  llvm::Function *CurFn;

  /// LabelMap - This keeps track of the LLVM basic block for each C label.
  DenseMap<const LabelStmt*, llvm::BasicBlock*> LabelMap;
public:
  CodeGenFunction(CodeGenModule &cgm);
  
  const llvm::Type *ConvertType(QualType T, SourceLocation Loc);
  
  void GenerateCode(const FunctionDecl *FD);
  
  
  /// getBasicBlockForLabel - Return the LLVM basicblock that the specified
  /// label maps to.
  llvm::BasicBlock *getBasicBlockForLabel(const LabelStmt *S);
  
  
  void EmitBlock(BasicBlock *BB);
  
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
  //                             Expression Emission
  //===--------------------------------------------------------------------===//

  ExprResult EmitExpr(const Expr *E);
  ExprResult EmitIntegerLiteral(const IntegerLiteral *E);
  ExprResult EmitBinaryOperator(const BinaryOperator *E);
  

  void EmitUsualArithmeticConversions(const BinaryOperator *E,
                                      ExprResult &LHS, ExprResult &RHS);
  
  // Binary Operators.
  ExprResult EmitBinaryAdd(const BinaryOperator *E);
};
}  // end namespace CodeGen
}  // end namespace clang
}  // end namespace llvm

#endif
