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

// Implement all of the delegation visitor methods.
#define STMT(FROM, TO) \
  void StmtVisitor::Visit##FROM(FROM *Node) { Visit##TO(Node); }
#include "clang/AST/StmtNodes.def"

