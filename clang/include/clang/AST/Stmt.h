//===--- Stmt.h - Classes for representing statements -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the Stmt interface and subclasses.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_STMT_H
#define LLVM_CLANG_AST_STMT_H

#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/SmallVector.h"

namespace llvm {
namespace clang {
  
/// Stmt - This represents one statement.  Note that statements are modelled as
/// subclasses of exprs so that 
///
class Stmt {
  /// Type.
public:
  Stmt() {}
  virtual ~Stmt() {}
  
  // FIXME: Change to non-virtual method that uses visitor pattern to do this.
  void dump() const;
  
private:
  virtual void dump_impl() const = 0;
};

//===----------------------------------------------------------------------===//
// Primary Expressions.
//===----------------------------------------------------------------------===//

#if 0
/// DeclRefExpr - [C99 6.5.1p2] - A reference to a declared variable, function,
/// enum, etc.
class DeclRefExpr : public Stmt {
  // TODO: Union with the decl when resolved.
  Decl &D;
public:
  DeclRef(Decl &d) : D(d) {}
  virtual void dump_impl() const;
};
#endif
  
}  // end namespace clang
}  // end namespace llvm

#endif
