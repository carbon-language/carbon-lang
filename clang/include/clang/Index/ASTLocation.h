//===--- ASTLocation.h - A <Decl, Stmt> pair --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  ASTLocation is Decl or a Stmt and its immediate Decl parent.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_INDEX_ASTLOCATION_H
#define LLVM_CLANG_INDEX_ASTLOCATION_H

#include <cassert>

namespace llvm {
  class raw_ostream;
}

namespace clang {
  class Decl;
  class Stmt;

namespace idx {

/// \brief Represents a Decl or a Stmt and its immediate Decl parent. It's
/// immutable.
///
/// ASTLocation is intended to be used as a "pointer" into the AST. It is either
/// just a Decl, or a Stmt and its Decl parent. Since a single Stmt is devoid
/// of context, its parent Decl provides all the additional missing information
/// like the declaration context, ASTContext, etc.
///
class ASTLocation {
  Decl *D;
  Stmt *Stm;

public:
  ASTLocation() : D(0), Stm(0) {}

  explicit ASTLocation(const Decl *d, const Stmt *stm = 0)
    : D(const_cast<Decl*>(d)), Stm(const_cast<Stmt*>(stm)) {
    assert((Stm == 0 || isImmediateParent(D, Stm)) &&
           "The Decl is not the immediate parent of the Stmt.");
  }

  const Decl *getDecl() const { return D; }
  const Stmt *getStmt() const { return Stm; }
  Decl *getDecl() { return D; }
  Stmt *getStmt() { return Stm; }

  bool isValid() const { return D != 0; }
  bool isInvalid() const { return !isValid(); }
  bool hasStmt() const { return Stm != 0; }

  /// \brief Checks that D is the immediate Decl parent of Node.
  static bool isImmediateParent(Decl *D, Stmt *Node);

  friend bool operator==(const ASTLocation &L, const ASTLocation &R) { 
    return L.D == R.D && L.Stm == R.Stm;
  }
  friend bool operator!=(const ASTLocation &L, const ASTLocation &R) { 
    return !(L == R);
  }
  
  void print(llvm::raw_ostream &OS);
};

} // namespace idx

} // namespace clang

#endif
