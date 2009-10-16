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

#include "clang/AST/TypeLoc.h"
#include "llvm/ADT/PointerIntPair.h"

namespace llvm {
  class raw_ostream;
}

namespace clang {
  class Decl;
  class Stmt;
  class NamedDecl;

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
public:
  enum NodeKind {
    N_Decl, N_NamedRef, N_Stmt, N_Type
  };

  struct NamedRef {
    NamedDecl *ND;
    SourceLocation Loc;
    
    NamedRef() : ND(0) { }
    NamedRef(NamedDecl *nd, SourceLocation loc) : ND(nd), Loc(loc) { }
  };

private:
  llvm::PointerIntPair<Decl *, 2, NodeKind> ParentDecl;

  union {
    Decl *D;
    Stmt *Stm;
    struct {
      NamedDecl *ND;
      unsigned RawLoc;
    } NDRef;
    struct {
      void *TyPtr;
      void *Data;
    } Ty;
  };

public:
  ASTLocation() { }

  explicit ASTLocation(const Decl *d)
    : ParentDecl(const_cast<Decl*>(d), N_Decl), D(const_cast<Decl*>(d)) { }

  ASTLocation(const Decl *parentDecl, const Stmt *stm)
    : ParentDecl(const_cast<Decl*>(parentDecl), N_Stmt),
      Stm(const_cast<Stmt*>(stm)) {
    if (!stm) ParentDecl.setPointer(0);
  }

  ASTLocation(const Decl *parentDecl, NamedDecl *ndRef, SourceLocation loc)
    : ParentDecl(const_cast<Decl*>(parentDecl), N_NamedRef) {
    if (ndRef) {
      NDRef.ND = ndRef;
      NDRef.RawLoc = loc.getRawEncoding();
    } else
      ParentDecl.setPointer(0);
  }

  ASTLocation(const Decl *parentDecl, TypeLoc tyLoc)
    : ParentDecl(const_cast<Decl*>(parentDecl), N_Type) {
    if (tyLoc) {
      Ty.TyPtr = tyLoc.getSourceType().getAsOpaquePtr();
      Ty.Data = tyLoc.getOpaqueData();
    } else
      ParentDecl.setPointer(0);
  }

  bool isValid() const { return ParentDecl.getPointer() != 0; }
  bool isInvalid() const { return !isValid(); }
  
  NodeKind getKind() const {
    assert(isValid());
    return (NodeKind)ParentDecl.getInt();
  }
  
  Decl *getParentDecl() const { return ParentDecl.getPointer(); }
  
  Decl *AsDecl() const {
    assert(getKind() == N_Decl);
    return D;
  }
  Stmt *AsStmt() const {
    assert(getKind() == N_Stmt);
    return Stm;
  }
  NamedRef AsNamedRef() const {
    assert(getKind() == N_NamedRef);
    return NamedRef(NDRef.ND, SourceLocation::getFromRawEncoding(NDRef.RawLoc));
  }
  TypeLoc AsTypeLoc() const {
    assert(getKind() == N_Type);
    return TypeLoc(QualType::getFromOpaquePtr(Ty.TyPtr), Ty.Data);
  }

  Decl *dyn_AsDecl() const { return isValid() && getKind() == N_Decl ? D : 0; }
  Stmt *dyn_AsStmt() const { return isValid() && getKind() == N_Stmt ? Stm : 0; }
  NamedRef dyn_AsNamedRef() const {
    return getKind() == N_Type ? AsNamedRef() : NamedRef();
  }
  TypeLoc dyn_AsTypeLoc() const {
    return getKind() == N_Type ? AsTypeLoc() : TypeLoc();
  }
  
  bool isDecl() const { return isValid() && getKind() == N_Decl; }
  bool isStmt() const { return isValid() && getKind() == N_Stmt; }
  bool isNamedRef() const { return isValid() && getKind() == N_NamedRef; }
  bool isType() const { return isValid() && getKind() == N_Type; }

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
