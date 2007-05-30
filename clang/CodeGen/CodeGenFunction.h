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
  
  class Expr;
  class IntegerLiteral;
  
namespace CodeGen {
  class CodeGenModule;
  
struct ExprResult {
  Value *V;
  bool isAggregate;
  
  static ExprResult get(Value *V) {
    ExprResult ER;
    ER.V = V;
    ER.isAggregate = false;
    return ER;
  }
  static ExprResult getAggregate(Value *V) {
    ExprResult ER;
    ER.V = V;
    ER.isAggregate = true;
    return ER;
  }
};
  
/// CodeGenFunction - This class organizes the per-function state that is used
/// while generating LLVM code.
class CodeGenFunction {
  CodeGenModule &CGM;  // Per-module state.
  TargetInfo &Target;
  LLVMBuilder Builder;
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
  
  
  //===--------------------------------------------------------------------===//
  //                             Expression Emission
  //===--------------------------------------------------------------------===//

  ExprResult EmitExpr(const Expr *E);
  ExprResult EmitIntegerLiteral(const IntegerLiteral *E);
};
}  // end namespace CodeGen
}  // end namespace clang
}  // end namespace llvm

#endif
