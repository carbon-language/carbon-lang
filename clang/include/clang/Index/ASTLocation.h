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
  class SourceRange;

namespace idx {
  class TranslationUnit;

/// \brief Represents a Decl or a Stmt and its immediate Decl parent. It's
/// immutable.
///
/// ASTLocation is intended to be used as a "pointer" into the AST. It is either
/// just a Decl, or a Stmt and its Decl parent. Since a single Stmt is devoid
/// of context, its parent Decl provides all the additional missing information
/// like the declaration context, ASTContext, etc.
///
class ASTLocation {
  const Decl *D;
  const Stmt *Stm;

public:
  ASTLocation() : D(0), Stm(0) {}

  explicit ASTLocation(const Decl *d, const Stmt *stm = 0) : D(d), Stm(stm) {
    assert((Stm == 0 || isImmediateParent(D, Stm)) &&
           "The Decl is not the immediate parent of the Stmt.");
  }

  const Decl *getDecl() const { return D; }
  const Stmt *getStmt() const { return Stm; }
  Decl *getDecl() { return const_cast<Decl*>(D); }
  Stmt *getStmt() { return const_cast<Stmt*>(Stm); }

  bool isValid() const { return D != 0; }
  bool isInvalid() const { return !isValid(); }
  bool isDecl() const { return isValid() && Stm == 0; }
  bool isStmt() const { return isValid() && Stm != 0; }

  /// \brief Returns the declaration that this ASTLocation references.
  ///
  /// If this points to a Decl, that Decl is returned.
  /// If this points to an Expr that references a Decl, that Decl is returned,
  /// otherwise it returns NULL.
  Decl *getReferencedDecl();
  const Decl *getReferencedDecl() const {
    return const_cast<ASTLocation*>(this)->getReferencedDecl();
  }

  SourceRange getSourceRange() const;

  /// \brief Checks that D is the immediate Decl parent of Node.
  static bool isImmediateParent(const Decl *D, const Stmt *Node);
  static const Decl *FindImmediateParent(const Decl *D, const Stmt *Node);

  friend bool operator==(const ASTLocation &L, const ASTLocation &R) {
    return L.D == R.D && L.Stm == R.Stm;
  }
  friend bool operator!=(const ASTLocation &L, const ASTLocation &R) {
    return !(L == R);
  }

  void print(llvm::raw_ostream &OS) const;
};

/// \brief Like ASTLocation but also contains the TranslationUnit that the
/// ASTLocation originated from.
class TULocation : public ASTLocation {
  TranslationUnit *TU;

public:
  TULocation(TranslationUnit *tu, ASTLocation astLoc)
    : ASTLocation(astLoc), TU(tu) {
    assert(tu && "Passed null translation unit");
  }

  TranslationUnit *getTU() const { return TU; }
};

} // namespace idx

} // namespace clang

#endif
