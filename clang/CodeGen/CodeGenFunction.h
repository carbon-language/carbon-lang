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
#include <vector>

namespace llvm {
  class Module;
namespace clang {
  class SourceLocation;
  class TargetInfo;
  class ASTContext;
  class Decl;
  class FunctionDecl;
  class QualType;
  class FunctionTypeProto;
  
  class Stmt;
  class CompoundStmt;
  class LabelStmt;
  class GotoStmt;
  class IfStmt;
  class WhileStmt;
  class DoStmt;
  class ForStmt;
  class ReturnStmt;
  class DeclStmt;
  
  class Expr;
  class DeclRefExpr;
  class StringLiteral;
  class IntegerLiteral;
  class CastExpr;
  class UnaryOperator;
  class BinaryOperator;
  class ArraySubscriptExpr;
  
  class BlockVarDecl;
  class EnumConstantDecl;
namespace CodeGen {
  class CodeGenModule;
  

/// RValue - This trivial value class is used to represent the result of an
/// expression that is evaluated.  It can be one of two things: either a simple
/// LLVM SSA value, or the address of an aggregate value in memory.  These two
/// possibilities are discriminated by isAggregate/isScalar.
class RValue {
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
  
  static RValue get(Value *V) {
    RValue ER;
    ER.V = V;
    ER.IsAggregate = false;
    return ER;
  }
  static RValue getAggregate(Value *V) {
    RValue ER;
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
  // alignment?
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
  
  const llvm::Type *LLVMIntTy;
  unsigned LLVMPointerWidth;
  
  /// LocalDeclMap - This keeps track of the LLVM allocas or globals for local C
  /// decls.
  DenseMap<const Decl*, llvm::Value*> LocalDeclMap;

  /// LabelMap - This keeps track of the LLVM basic block for each C label.
  DenseMap<const LabelStmt*, llvm::BasicBlock*> LabelMap;
public:
  CodeGenFunction(CodeGenModule &cgm);
  
  ASTContext &getContext() const;

  const llvm::Type *ConvertType(QualType T, SourceLocation Loc);
  void DecodeArgumentTypes(const FunctionTypeProto &FTP, 
                           std::vector<const llvm::Type*> &ArgTys,
                           SourceLocation Loc);
  
  void GenerateCode(const FunctionDecl *FD);
  
  
  /// getBasicBlockForLabel - Return the LLVM basicblock that the specified
  /// label maps to.
  llvm::BasicBlock *getBasicBlockForLabel(const LabelStmt *S);
  
  
  void EmitBlock(BasicBlock *BB);


  /// EvaluateExprAsBool - Perform the usual unary conversions on the specified
  /// expression and compare the result against zero, returning an Int1Ty value.
  Value *EvaluateExprAsBool(const Expr *E);
  
  //===--------------------------------------------------------------------===//
  //                                Conversions
  //===--------------------------------------------------------------------===//
  
  /// EmitConversion - Convert the value specied by Val, whose type is ValTy, to
  /// the type specified by DstTy, following the rules of C99 6.3.
  RValue EmitConversion(RValue Val, QualType ValTy, QualType DstTy,
                        SourceLocation Loc);
  
  /// ConvertScalarValueToBool - Convert the specified expression value to a
  /// boolean (i1) truth value.  This is equivalent to "Val == 0".
  Value *ConvertScalarValueToBool(RValue Val, QualType Ty);
  
  //===--------------------------------------------------------------------===//
  //                        Local Declaration Emission
  //===--------------------------------------------------------------------===//
  
  void EmitDecl(const Decl &D);
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
  void EmitWhileStmt(const WhileStmt &S);
  void EmitDoStmt(const DoStmt &S);
  void EmitForStmt(const ForStmt &S);
  void EmitReturnStmt(const ReturnStmt &S);
  void EmitDeclStmt(const DeclStmt &S);

  //===--------------------------------------------------------------------===//
  //                         LValue Expression Emission
  //===--------------------------------------------------------------------===//

  /// EmitLValue - Emit code to compute a designator that specifies the location
  /// of the expression.
  ///
  /// This can return one of two things: a simple address or a bitfield
  /// reference.  In either case, the LLVM Value* in the LValue structure is
  /// guaranteed to be an LLVM pointer type.
  ///
  /// If this returns a bitfield reference, nothing about the pointee type of
  /// the LLVM value is known: For example, it may not be a pointer to an
  /// integer.
  ///
  /// If this returns a normal address, and if the lvalue's C type is fixed
  /// size, this method guarantees that the returned pointer type will point to
  /// an LLVM type of the same size of the lvalue's type.  If the lvalue has a
  /// variable length type, this is not possible.
  ///
  LValue EmitLValue(const Expr *E);
  
  /// EmitLoadOfLValue - Given an expression that represents a value lvalue,
  /// this method emits the address of the lvalue, then loads the result as an
  /// rvalue, returning the rvalue.
  RValue EmitLoadOfLValue(const Expr *E);
  
  /// EmitStoreThroughLValue - Store the specified rvalue into the specified
  /// lvalue, where both are guaranteed to the have the same type, and that type
  /// is 'Ty'.
  void EmitStoreThroughLValue(RValue Src, LValue Dst, QualType Ty);
  
  LValue EmitDeclRefLValue(const DeclRefExpr *E);
  LValue EmitStringLiteralLValue(const StringLiteral *E);
  LValue EmitUnaryOpLValue(const UnaryOperator *E);
  LValue EmitArraySubscriptExpr(const ArraySubscriptExpr *E);
    
  //===--------------------------------------------------------------------===//
  //                             Expression Emission
  //===--------------------------------------------------------------------===//

  RValue EmitExprWithUsualUnaryConversions(const Expr *E, QualType &ResTy);
  QualType EmitUsualArithmeticConversions(const BinaryOperator *E,
                                          RValue &LHS, RValue &RHS);
  
  RValue EmitExpr(const Expr *E);
  RValue EmitIntegerLiteral(const IntegerLiteral *E);
  
  RValue EmitCastExpr(const CastExpr *E);

  // Unary Operators.
  RValue EmitUnaryOperator(const UnaryOperator *E);
  // FIXME: pre/post inc/dec
  RValue EmitUnaryAddrOf  (const UnaryOperator *E);
  RValue EmitUnaryPlus    (const UnaryOperator *E);
  RValue EmitUnaryMinus   (const UnaryOperator *E);
  RValue EmitUnaryNot     (const UnaryOperator *E);
  RValue EmitUnaryLNot    (const UnaryOperator *E);
  // FIXME: SIZEOF/ALIGNOF(expr).
  // FIXME: real/imag
  
  // Binary Operators.
  RValue EmitBinaryOperator(const BinaryOperator *E);
  RValue EmitBinaryMul(const BinaryOperator *E);
  RValue EmitBinaryDiv(const BinaryOperator *E);
  RValue EmitBinaryRem(const BinaryOperator *E);
  RValue EmitBinaryAdd(const BinaryOperator *E);
  RValue EmitBinarySub(const BinaryOperator *E);
  RValue EmitBinaryShl(const BinaryOperator *E);
  RValue EmitBinaryShr(const BinaryOperator *E);
  
  // FIXME: relational
  
  RValue EmitBinaryAnd(const BinaryOperator *E);
  RValue EmitBinaryXor(const BinaryOperator *E);
  RValue EmitBinaryOr (const BinaryOperator *E);
  RValue EmitBinaryLAnd(const BinaryOperator *E);
  RValue EmitBinaryLOr(const BinaryOperator *E);
  
  RValue EmitBinaryAssign(const BinaryOperator *E);
  // FIXME: Assignment.
  
  RValue EmitBinaryComma(const BinaryOperator *E);
};
}  // end namespace CodeGen
}  // end namespace clang
}  // end namespace llvm

#endif
