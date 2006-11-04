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

// Implement all the AST node visit methods using the StmtNodes.def database.
#define STMT(CLASS, PARENT) \
void CLASS::visit(StmtVisitor &V) { return V.Visit##CLASS(this); }

STMT(Stmt, )
#include "clang/AST/StmtNodes.def"

