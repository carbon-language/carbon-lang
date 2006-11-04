//===--- Expr.h - Classes for representing expressions ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the Expr interface and subclasses.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_EXPR_H
#define LLVM_CLANG_AST_EXPR_H

#include "clang/AST/Stmt.h"

namespace llvm {
namespace clang {
  class IdentifierInfo;
  class Decl;
  class Type;
  
/// Expr - This represents one expression.  Note that Expr's are subclasses of
/// Stmt.  This allows an expression to be transparently used any place a Stmt
/// is required.
///
class Expr : public Stmt {
  /// TODO: Type.
public:
  Expr() {}
  ~Expr() {}
  
  
  // FIXME: move to isa/dyncast etc.
  virtual bool isExpr() const { return true; }
  virtual void visit(StmtVisitor *Visitor);
};

//===----------------------------------------------------------------------===//
// Primary Expressions.
//===----------------------------------------------------------------------===//

/// DeclRefExpr - [C99 6.5.1p2] - A reference to a declared variable, function,
/// enum, etc.
class DeclRefExpr : public Expr {
  // TODO: Union with the decl when resolved.
  Decl &D;
public:
  DeclRefExpr(Decl &d) : D(d) {}
  virtual void dump_impl() const;
  virtual void visit(StmtVisitor *Visitor);
};

class IntegerConstant : public Expr {
public:
  IntegerConstant() {}
  virtual void dump_impl() const;
  virtual void visit(StmtVisitor *Visitor);
};

class FloatingConstant : public Expr {
public:
  FloatingConstant() {}
  virtual void dump_impl() const;
  virtual void visit(StmtVisitor *Visitor);
};

class StringExpr : public Expr {
  const char *StrData;
  unsigned ByteLength;
  bool isWide;
public:
  StringExpr(const char *strData, unsigned byteLength, bool Wide);
  virtual ~StringExpr();
  virtual void dump_impl() const;
  virtual void visit(StmtVisitor *Visitor);
};

class StringExprLOC : public StringExpr {
  // Locations for the string tokens before string concatenation.
  SmallVector<SourceLocation, 4> Locs;
public:
  StringExprLOC(const char *StrData, unsigned ByteLength, bool isWide,
                SourceLocation *L, unsigned NumLocs)
    : StringExpr(StrData, ByteLength, isWide), Locs(L, L+NumLocs) {
  }
};

/// ParenExpr - This represents a parethesized expression, e.g. "(1)".  This
/// AST node is only formed if full location information is requested.
class ParenExpr : public Expr {
  SourceLocation L, R;
  Expr *Val;
public:
  ParenExpr(SourceLocation l, SourceLocation r, Expr *val)
    : L(l), R(r), Val(val) {}
  virtual void dump_impl() const;
  virtual void visit(StmtVisitor *Visitor);
};


/// UnaryOperator - This represents the unary-expression's (except sizeof of
/// types), the postinc/postdec operators from postfix-expression, and various
/// extensions.
class UnaryOperator : public Expr {
public:
  enum Opcode {
    PostInc, PostDec, // [C99 6.5.2.4] Postfix increment and decrement operators
    PreInc, PreDec,   // [C99 6.5.3.1] Prefix increment and decrement operators.
    AddrOf, Deref,    // [C99 6.5.3.2] Address and indirection operators.
    Plus, Minus,      // [C99 6.5.3.3] Unary arithmetic operators.
    Not, LNot,        // [C99 6.5.3.3] Unary arithmetic operators.
    SizeOf, AlignOf,  // [C99 6.5.3.4] Sizeof (expr, not type) operator.
    Real, Imag,       // "__real expr"/"__imag expr" Extension.
    AddrLabel,        // && label Extension.
    Extension         // __extension__ marker.
  };

  UnaryOperator(Expr *input, Opcode opc)
    : Input(input), Opc(opc) {}
  
  /// getOpcodeStr - Turn an Opcode enum value into the punctuation char it
  /// corresponds to, e.g. "sizeof" or "[pre]++"
  static const char *getOpcodeStr(Opcode Op);
  
  virtual void dump_impl() const;
  virtual void visit(StmtVisitor *Visitor);

private:
  Expr *Input;
  Opcode Opc;
};

class UnaryOperatorLOC : public UnaryOperator {
  SourceLocation Loc;
public:
  UnaryOperatorLOC(SourceLocation loc, Expr *Input, Opcode Opc)
   : UnaryOperator(Input, Opc), Loc(loc) {}
};

/// SizeOfAlignOfTypeExpr - [C99 6.5.3.4] - This is only for sizeof/alignof of
/// *types*.  sizeof(expr) is handled by UnaryOperator.
class SizeOfAlignOfTypeExpr : public Expr {
  bool isSizeof;  // true if sizeof, false if alignof.
  Type *Ty;
public:
  SizeOfAlignOfTypeExpr(bool issizeof, Type *ty) : isSizeof(issizeof), Ty(ty) {
  }

  virtual void dump_impl() const;
  virtual void visit(StmtVisitor *Visitor);
};

class SizeOfAlignOfTypeExprLOC : public SizeOfAlignOfTypeExpr {
  SourceLocation OpLoc, LParenLoc, RParenLoc;
public:
  SizeOfAlignOfTypeExprLOC(SourceLocation oploc, bool isSizeof, 
                           SourceLocation lparenloc, Type *Ty,
                           SourceLocation rparenloc)
    : SizeOfAlignOfTypeExpr(isSizeof, Ty), OpLoc(oploc), LParenLoc(lparenloc),
      RParenLoc(rparenloc) {
  }
};

//===----------------------------------------------------------------------===//
// Postfix Operators.
//===----------------------------------------------------------------------===//

/// ArraySubscriptExpr - [C99 6.5.2.1] Array Subscripting.
class ArraySubscriptExpr : public Expr {
  Expr *Base, *Idx;
public:
  ArraySubscriptExpr(Expr *base, Expr *idx) : Base(base), Idx(idx) {}
  
  virtual void dump_impl() const;
  virtual void visit(StmtVisitor *Visitor);
};


class ArraySubscriptExprLOC : public ArraySubscriptExpr {
  SourceLocation LLoc, RLoc;
public:
  ArraySubscriptExprLOC(Expr *Base, SourceLocation lloc, Expr *Idx, 
                        SourceLocation rloc)
    : ArraySubscriptExpr(Base, Idx), LLoc(lloc), RLoc(rloc) {}
};

/// CallExpr - [C99 6.5.2.2] Function Calls.
///
class CallExpr : public Expr {
  Expr *Fn;
  Expr **Args;
  unsigned NumArgs;
public:
  CallExpr(Expr *fn, Expr **args, unsigned numargs);
  ~CallExpr() {
    delete [] Args;
  }
  
  /// getNumArgs - Return the number of actual arguments to this call.
  ///
  unsigned getNumArgs() const { return NumArgs; }
  
  /// getArg - Return the specified argument.
  Expr *getArg(unsigned Arg) const {
    assert(Arg < NumArgs && "Arg access out of range!");
    return Args[Arg];
  }
  
  /// getNumCommas - Return the number of commas that must have been present in
  /// this function call.
  unsigned getNumCommas() const { return NumArgs ? NumArgs - 1 : 0; }
  
  virtual void dump_impl() const;
  virtual void visit(StmtVisitor *Visitor);
};

class CallExprLOC : public CallExpr {
  SourceLocation LParenLoc, RParenLoc;
  SourceLocation *CommaLocs;
public:
  CallExprLOC(Expr *Fn, SourceLocation lparenloc, Expr **Args, unsigned NumArgs,
              SourceLocation *commalocs, SourceLocation rparenloc);
  ~CallExprLOC() {
    delete [] CommaLocs;
  }
};

/// MemberExpr - [C99 6.5.2.3] Structure and Union Members.
///
class MemberExpr : public Expr {
  Expr *Base;
  Decl *MemberDecl;
  bool isArrow;      // True if this is "X->F", false if this is "X.F".
public:
  MemberExpr(Expr *base, bool isarrow, Decl *memberdecl) 
    : Base(base), MemberDecl(memberdecl), isArrow(isarrow) {
  }
  virtual void dump_impl() const;
  virtual void visit(StmtVisitor *Visitor);
};

class MemberExprLOC : public MemberExpr {
  SourceLocation OpLoc, MemberLoc;
public:
  MemberExprLOC(Expr *Base, SourceLocation oploc, bool isArrow,
                SourceLocation memberLoc, Decl *MemberDecl) 
    : MemberExpr(Base, isArrow, MemberDecl), OpLoc(oploc), MemberLoc(memberLoc){
  }

};

/// CastExpr - [C99 6.5.4] Cast Operators.
///
class CastExpr : public Expr {
  Type *Ty;
  Expr *Op;
public:
  CastExpr(Type *ty, Expr *op) : Ty(ty), Op(op) {}
  
  virtual void dump_impl() const;
  virtual void visit(StmtVisitor *Visitor);
};

class CastExprLOC : public CastExpr {
  SourceLocation LParenLoc, RParenLoc;
public:
  CastExprLOC(SourceLocation lparenloc, Type *Ty, SourceLocation rparenloc,
              Expr *Op)
  : CastExpr(Ty, Op), LParenLoc(lparenloc), RParenLoc(rparenloc) {
  }
};


class BinaryOperator : public Expr {
public:
  enum Opcode {
    // Operators listed in order of precedence.
    Mul, Div, Rem,    // [C99 6.5.5] Multiplicative operators.
    Add, Sub,         // [C99 6.5.6] Additive operators.
    Shl, Shr,         // [C99 6.5.7] Bitwise shift operators.
    LT, GT, LE, GE,   // [C99 6.5.8] Relational operators.
    EQ, NE,           // [C99 6.5.9] Equality operators.
    And,              // [C99 6.5.10] Bitwise AND operator.
    Xor,              // [C99 6.5.11] Bitwise XOR operator.
    Or,               // [C99 6.5.12] Bitwise OR operator.
    LAnd,             // [C99 6.5.13] Logical AND operator.
    LOr,              // [C99 6.5.14] Logical OR operator.
    Assign, MulAssign,// [C99 6.5.16] Assignment operators.
    DivAssign, RemAssign,
    AddAssign, SubAssign,
    ShlAssign, ShrAssign,
    AndAssign, XorAssign,
    OrAssign,
    Comma             // [C99 6.5.17] Comma operator.
  };
  
  BinaryOperator(Expr *lhs, Expr *rhs, Opcode opc)
    : LHS(lhs), RHS(rhs), Opc(opc) {}

  /// getOpcodeStr - Turn an Opcode enum value into the punctuation char it
  /// corresponds to, e.g. "<<=".
  static const char *getOpcodeStr(Opcode Op);
  
  virtual void dump_impl() const;
  virtual void visit(StmtVisitor *Visitor);

private:
  Expr *LHS, *RHS;
  Opcode Opc;
};

class BinaryOperatorLOC : public BinaryOperator {
  SourceLocation OperatorLoc;
public:
  BinaryOperatorLOC(Expr *LHS, SourceLocation OpLoc, Expr *RHS, Opcode Opc)
    : BinaryOperator(LHS, RHS, Opc), OperatorLoc(OpLoc) {
  }
};

/// ConditionalOperator - The ?: operator.  Note that LHS may be null when the
/// GNU "missing LHS" extension is in use.
///
class ConditionalOperator : public Expr {
  Expr *Cond, *LHS, *RHS;  // Left/Middle/Right hand sides.
public:
  ConditionalOperator(Expr *cond, Expr *lhs, Expr *rhs)
    : Cond(cond), LHS(lhs), RHS(rhs) {}
  virtual void dump_impl() const;
  virtual void visit(StmtVisitor *Visitor);
};

/// ConditionalOperatorLOC - ConditionalOperator with full location info.
///
class ConditionalOperatorLOC : public ConditionalOperator {
  SourceLocation QuestionLoc, ColonLoc;
public:
  ConditionalOperatorLOC(Expr *Cond, SourceLocation QLoc, Expr *LHS,
                         SourceLocation CLoc, Expr *RHS)
    : ConditionalOperator(Cond, LHS, RHS), QuestionLoc(QLoc), ColonLoc(CLoc) {}
};

  
}  // end namespace clang
}  // end namespace llvm

#endif
