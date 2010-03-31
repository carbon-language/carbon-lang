//===--- StmtObjC.h - Classes for representing ObjC statements --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the Objective-C statement AST node classes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_STMTOBJC_H
#define LLVM_CLANG_AST_STMTOBJC_H

#include "clang/AST/Stmt.h"

namespace clang {

/// ObjCForCollectionStmt - This represents Objective-c's collection statement;
/// represented as 'for (element 'in' collection-expression)' stmt.
///
class ObjCForCollectionStmt : public Stmt {
  enum { ELEM, COLLECTION, BODY, END_EXPR };
  Stmt* SubExprs[END_EXPR]; // SubExprs[ELEM] is an expression or declstmt.
  SourceLocation ForLoc;
  SourceLocation RParenLoc;
public:
  ObjCForCollectionStmt(Stmt *Elem, Expr *Collect, Stmt *Body,
                        SourceLocation FCL, SourceLocation RPL);
  explicit ObjCForCollectionStmt(EmptyShell Empty) :
    Stmt(ObjCForCollectionStmtClass, Empty) { }

  Stmt *getElement() { return SubExprs[ELEM]; }
  Expr *getCollection() {
    return reinterpret_cast<Expr*>(SubExprs[COLLECTION]);
  }
  Stmt *getBody() { return SubExprs[BODY]; }

  const Stmt *getElement() const { return SubExprs[ELEM]; }
  const Expr *getCollection() const {
    return reinterpret_cast<Expr*>(SubExprs[COLLECTION]);
  }
  const Stmt *getBody() const { return SubExprs[BODY]; }

  void setElement(Stmt *S) { SubExprs[ELEM] = S; }
  void setCollection(Expr *E) {
    SubExprs[COLLECTION] = reinterpret_cast<Stmt*>(E);
  }
  void setBody(Stmt *S) { SubExprs[BODY] = S; }

  SourceLocation getForLoc() const { return ForLoc; }
  void setForLoc(SourceLocation Loc) { ForLoc = Loc; }
  SourceLocation getRParenLoc() const { return RParenLoc; }
  void setRParenLoc(SourceLocation Loc) { RParenLoc = Loc; }

  virtual SourceRange getSourceRange() const {
    return SourceRange(ForLoc, SubExprs[BODY]->getLocEnd());
  }
  static bool classof(const Stmt *T) {
    return T->getStmtClass() == ObjCForCollectionStmtClass;
  }
  static bool classof(const ObjCForCollectionStmt *) { return true; }

  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
};

/// ObjCAtCatchStmt - This represents objective-c's @catch statement.
class ObjCAtCatchStmt : public Stmt {
private:
  enum { BODY, NEXT_CATCH, END_EXPR };
  ParmVarDecl *ExceptionDecl;
  Stmt *SubExprs[END_EXPR];
  SourceLocation AtCatchLoc, RParenLoc;

public:
  ObjCAtCatchStmt(SourceLocation atCatchLoc, SourceLocation rparenloc,
                  ParmVarDecl *catchVarDecl,
                  Stmt *atCatchStmt, Stmt *atCatchList);

  explicit ObjCAtCatchStmt(EmptyShell Empty) :
    Stmt(ObjCAtCatchStmtClass, Empty) { }

  const Stmt *getCatchBody() const { return SubExprs[BODY]; }
  Stmt *getCatchBody() { return SubExprs[BODY]; }
  void setCatchBody(Stmt *S) { SubExprs[BODY] = S; }

  const ObjCAtCatchStmt *getNextCatchStmt() const {
    return static_cast<const ObjCAtCatchStmt*>(SubExprs[NEXT_CATCH]);
  }
  ObjCAtCatchStmt *getNextCatchStmt() {
    return static_cast<ObjCAtCatchStmt*>(SubExprs[NEXT_CATCH]);
  }
  void setNextCatchStmt(Stmt *S) { SubExprs[NEXT_CATCH] = S; }

  const ParmVarDecl *getCatchParamDecl() const {
    return ExceptionDecl;
  }
  ParmVarDecl *getCatchParamDecl() {
    return ExceptionDecl;
  }
  void setCatchParamDecl(ParmVarDecl *D) { ExceptionDecl = D; }

  SourceLocation getAtCatchLoc() const { return AtCatchLoc; }
  void setAtCatchLoc(SourceLocation Loc) { AtCatchLoc = Loc; }
  SourceLocation getRParenLoc() const { return RParenLoc; }
  void setRParenLoc(SourceLocation Loc) { RParenLoc = Loc; }

  virtual SourceRange getSourceRange() const {
    return SourceRange(AtCatchLoc, SubExprs[BODY]->getLocEnd());
  }

  bool hasEllipsis() const { return getCatchParamDecl() == 0; }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == ObjCAtCatchStmtClass;
  }
  static bool classof(const ObjCAtCatchStmt *) { return true; }

  virtual child_iterator child_begin();
  virtual child_iterator child_end();
};

/// ObjCAtFinallyStmt - This represent objective-c's @finally Statement
class ObjCAtFinallyStmt : public Stmt {
  Stmt *AtFinallyStmt;
  SourceLocation AtFinallyLoc;
public:
  ObjCAtFinallyStmt(SourceLocation atFinallyLoc, Stmt *atFinallyStmt)
  : Stmt(ObjCAtFinallyStmtClass),
    AtFinallyStmt(atFinallyStmt), AtFinallyLoc(atFinallyLoc) {}

  explicit ObjCAtFinallyStmt(EmptyShell Empty) :
    Stmt(ObjCAtFinallyStmtClass, Empty) { }

  const Stmt *getFinallyBody() const { return AtFinallyStmt; }
  Stmt *getFinallyBody() { return AtFinallyStmt; }
  void setFinallyBody(Stmt *S) { AtFinallyStmt = S; }

  virtual SourceRange getSourceRange() const {
    return SourceRange(AtFinallyLoc, AtFinallyStmt->getLocEnd());
  }

  SourceLocation getAtFinallyLoc() const { return AtFinallyLoc; }
  void setAtFinallyLoc(SourceLocation Loc) { AtFinallyLoc = Loc; }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == ObjCAtFinallyStmtClass;
  }
  static bool classof(const ObjCAtFinallyStmt *) { return true; }

  virtual child_iterator child_begin();
  virtual child_iterator child_end();
};

/// ObjCAtTryStmt - This represent objective-c's over-all
/// @try ... @catch ... @finally statement.
class ObjCAtTryStmt : public Stmt {
private:
  enum { TRY, CATCH, FINALLY, END_EXPR };
  Stmt* SubStmts[END_EXPR];

  SourceLocation AtTryLoc;
public:
  ObjCAtTryStmt(SourceLocation atTryLoc, Stmt *atTryStmt,
                Stmt *atCatchStmt,
                Stmt *atFinallyStmt)
  : Stmt(ObjCAtTryStmtClass) {
      SubStmts[TRY] = atTryStmt;
      SubStmts[CATCH] = atCatchStmt;
      SubStmts[FINALLY] = atFinallyStmt;
      AtTryLoc = atTryLoc;
    }
  explicit ObjCAtTryStmt(EmptyShell Empty) :
    Stmt(ObjCAtTryStmtClass, Empty) { }

  SourceLocation getAtTryLoc() const { return AtTryLoc; }
  void setAtTryLoc(SourceLocation Loc) { AtTryLoc = Loc; }

  const Stmt *getTryBody() const { return SubStmts[TRY]; }
  Stmt *getTryBody() { return SubStmts[TRY]; }
  void setTryBody(Stmt *S) { SubStmts[TRY] = S; }

  const ObjCAtCatchStmt *getCatchStmts() const {
    return dyn_cast_or_null<ObjCAtCatchStmt>(SubStmts[CATCH]);
  }
  ObjCAtCatchStmt *getCatchStmts() {
    return dyn_cast_or_null<ObjCAtCatchStmt>(SubStmts[CATCH]);
  }
  void setCatchStmts(Stmt *S) { SubStmts[CATCH] = S; }

  const ObjCAtFinallyStmt *getFinallyStmt() const {
    return dyn_cast_or_null<ObjCAtFinallyStmt>(SubStmts[FINALLY]);
  }
  ObjCAtFinallyStmt *getFinallyStmt() {
    return dyn_cast_or_null<ObjCAtFinallyStmt>(SubStmts[FINALLY]);
  }
  void setFinallyStmt(Stmt *S) { SubStmts[FINALLY] = S; }

  virtual SourceRange getSourceRange() const {
    return SourceRange(AtTryLoc, SubStmts[TRY]->getLocEnd());
  }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == ObjCAtTryStmtClass;
  }
  static bool classof(const ObjCAtTryStmt *) { return true; }

  virtual child_iterator child_begin();
  virtual child_iterator child_end();
};

/// ObjCAtSynchronizedStmt - This is for objective-c's @synchronized statement.
/// Example: @synchronized (sem) {
///             do-something;
///          }
///
class ObjCAtSynchronizedStmt : public Stmt {
private:
  enum { SYNC_EXPR, SYNC_BODY, END_EXPR };
  Stmt* SubStmts[END_EXPR];
  SourceLocation AtSynchronizedLoc;

public:
  ObjCAtSynchronizedStmt(SourceLocation atSynchronizedLoc, Stmt *synchExpr,
                         Stmt *synchBody)
  : Stmt(ObjCAtSynchronizedStmtClass) {
    SubStmts[SYNC_EXPR] = synchExpr;
    SubStmts[SYNC_BODY] = synchBody;
    AtSynchronizedLoc = atSynchronizedLoc;
  }
  explicit ObjCAtSynchronizedStmt(EmptyShell Empty) :
    Stmt(ObjCAtSynchronizedStmtClass, Empty) { }

  SourceLocation getAtSynchronizedLoc() const { return AtSynchronizedLoc; }
  void setAtSynchronizedLoc(SourceLocation Loc) { AtSynchronizedLoc = Loc; }

  const CompoundStmt *getSynchBody() const {
    return reinterpret_cast<CompoundStmt*>(SubStmts[SYNC_BODY]);
  }
  CompoundStmt *getSynchBody() {
    return reinterpret_cast<CompoundStmt*>(SubStmts[SYNC_BODY]);
  }
  void setSynchBody(Stmt *S) { SubStmts[SYNC_BODY] = S; }

  const Expr *getSynchExpr() const {
    return reinterpret_cast<Expr*>(SubStmts[SYNC_EXPR]);
  }
  Expr *getSynchExpr() {
    return reinterpret_cast<Expr*>(SubStmts[SYNC_EXPR]);
  }
  void setSynchExpr(Stmt *S) { SubStmts[SYNC_EXPR] = S; }

  virtual SourceRange getSourceRange() const {
    return SourceRange(AtSynchronizedLoc, getSynchBody()->getLocEnd());
  }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == ObjCAtSynchronizedStmtClass;
  }
  static bool classof(const ObjCAtSynchronizedStmt *) { return true; }

  virtual child_iterator child_begin();
  virtual child_iterator child_end();
};

/// ObjCAtThrowStmt - This represents objective-c's @throw statement.
class ObjCAtThrowStmt : public Stmt {
  Stmt *Throw;
  SourceLocation AtThrowLoc;
public:
  ObjCAtThrowStmt(SourceLocation atThrowLoc, Stmt *throwExpr)
  : Stmt(ObjCAtThrowStmtClass), Throw(throwExpr) {
    AtThrowLoc = atThrowLoc;
  }
  explicit ObjCAtThrowStmt(EmptyShell Empty) :
    Stmt(ObjCAtThrowStmtClass, Empty) { }

  const Expr *getThrowExpr() const { return reinterpret_cast<Expr*>(Throw); }
  Expr *getThrowExpr() { return reinterpret_cast<Expr*>(Throw); }
  void setThrowExpr(Stmt *S) { Throw = S; }

  SourceLocation getThrowLoc() { return AtThrowLoc; }
  void setThrowLoc(SourceLocation Loc) { AtThrowLoc = Loc; }

  virtual SourceRange getSourceRange() const {
    if (Throw)
      return SourceRange(AtThrowLoc, Throw->getLocEnd());
    else
      return SourceRange(AtThrowLoc);
  }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == ObjCAtThrowStmtClass;
  }
  static bool classof(const ObjCAtThrowStmt *) { return true; }

  virtual child_iterator child_begin();
  virtual child_iterator child_end();
};

}  // end namespace clang

#endif
