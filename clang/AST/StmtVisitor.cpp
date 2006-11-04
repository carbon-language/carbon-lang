//===--- StmtVisitor.cpp - Visitor for Stmt subclasses --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the StmtVisitor class.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/StmtVisitor.h"
#include "clang/AST/Expr.h"
using namespace llvm;
using namespace clang;

StmtVisitor::~StmtVisitor() {
  // Out-of-line virtual dtor.
}

#define DELEGATE_VISITOR(FROM, TO) \
  void StmtVisitor::Visit##FROM(FROM *Node) { Visit##TO(Node); }

DELEGATE_VISITOR(Expr, Stmt)

// Stmt subclasses to Stmt.
DELEGATE_VISITOR(CompoundStmt, Stmt)
DELEGATE_VISITOR(IfStmt      , Stmt)
DELEGATE_VISITOR(ForStmt     , Stmt)
DELEGATE_VISITOR(ReturnStmt  , Stmt)

// Expr subclasses to Expr.
DELEGATE_VISITOR(DeclRefExpr          , Expr)
DELEGATE_VISITOR(IntegerConstant      , Expr)
DELEGATE_VISITOR(FloatingConstant     , Expr)
DELEGATE_VISITOR(StringExpr           , Expr)
DELEGATE_VISITOR(ParenExpr            , Expr)
DELEGATE_VISITOR(UnaryOperator        , Expr)
DELEGATE_VISITOR(SizeOfAlignOfTypeExpr, Expr)
DELEGATE_VISITOR(ArraySubscriptExpr   , Expr)
DELEGATE_VISITOR(CallExpr             , Expr)
DELEGATE_VISITOR(MemberExpr           , Expr)
DELEGATE_VISITOR(CastExpr             , Expr)
DELEGATE_VISITOR(BinaryOperator       , Expr)
DELEGATE_VISITOR(ConditionalOperator  , Expr)
