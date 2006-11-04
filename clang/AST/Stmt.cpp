//===--- Stmt.cpp - Statement AST Node Implementation ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Stmt class and statement subclasses.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/Stmt.h"
#include "clang/AST/Expr.h"
#include "clang/AST/StmtVisitor.h"
using namespace llvm;
using namespace clang;

#define MAKE_VISITOR(CLASS) \
void CLASS::visit(StmtVisitor &V) { return V.Visit##CLASS(this); }

MAKE_VISITOR(Stmt)
MAKE_VISITOR(CompoundStmt)
MAKE_VISITOR(IfStmt)
MAKE_VISITOR(ReturnStmt)

#undef MAKE_VISITOR
