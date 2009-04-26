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
  
  SourceLocation getRParenLoc() const { return RParenLoc; }
  
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

  // Used by deserialization.
  ObjCAtCatchStmt(SourceLocation atCatchLoc, SourceLocation rparenloc)
  : Stmt(ObjCAtCatchStmtClass), AtCatchLoc(atCatchLoc), RParenLoc(rparenloc) {}

public:
  ObjCAtCatchStmt(SourceLocation atCatchLoc, SourceLocation rparenloc,
                  ParmVarDecl *catchVarDecl, 
                  Stmt *atCatchStmt, Stmt *atCatchList);
  
  const Stmt *getCatchBody() const { return SubExprs[BODY]; }
  Stmt *getCatchBody() { return SubExprs[BODY]; }

  const ObjCAtCatchStmt *getNextCatchStmt() const {
    return static_cast<const ObjCAtCatchStmt*>(SubExprs[NEXT_CATCH]);
  }
  ObjCAtCatchStmt *getNextCatchStmt() { 
    return static_cast<ObjCAtCatchStmt*>(SubExprs[NEXT_CATCH]);
  }

  const ParmVarDecl *getCatchParamDecl() const { 
    return ExceptionDecl; 
  }
  ParmVarDecl *getCatchParamDecl() { 
    return ExceptionDecl; 
  }
  
  SourceLocation getAtCatchLoc() const { return AtCatchLoc; }
  SourceLocation getRParenLoc() const { return RParenLoc; }
  
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
  
  const Stmt *getFinallyBody() const { return AtFinallyStmt; }
  Stmt *getFinallyBody() { return AtFinallyStmt; }

  virtual SourceRange getSourceRange() const { 
    return SourceRange(AtFinallyLoc, AtFinallyStmt->getLocEnd()); 
  }
  
  SourceLocation getAtFinallyLoc() const { return AtFinallyLoc; }
  
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
    
  SourceLocation getAtTryLoc() const { return AtTryLoc; }
  const Stmt *getTryBody() const { return SubStmts[TRY]; }
  Stmt *getTryBody() { return SubStmts[TRY]; }
  const ObjCAtCatchStmt *getCatchStmts() const { 
    return dyn_cast_or_null<ObjCAtCatchStmt>(SubStmts[CATCH]); 
  }
  ObjCAtCatchStmt *getCatchStmts() { 
    return dyn_cast_or_null<ObjCAtCatchStmt>(SubStmts[CATCH]); 
  }
  const ObjCAtFinallyStmt *getFinallyStmt() const { 
    return dyn_cast_or_null<ObjCAtFinallyStmt>(SubStmts[FINALLY]); 
  }
  ObjCAtFinallyStmt *getFinallyStmt() { 
    return dyn_cast_or_null<ObjCAtFinallyStmt>(SubStmts[FINALLY]); 
  }
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
  
  SourceLocation getAtSynchronizedLoc() const { return AtSynchronizedLoc; }
  
  const CompoundStmt *getSynchBody() const {
    return reinterpret_cast<CompoundStmt*>(SubStmts[SYNC_BODY]);
  }
  CompoundStmt *getSynchBody() { 
    return reinterpret_cast<CompoundStmt*>(SubStmts[SYNC_BODY]); 
  }
  
  const Expr *getSynchExpr() const { 
    return reinterpret_cast<Expr*>(SubStmts[SYNC_EXPR]); 
  }
  Expr *getSynchExpr() { 
    return reinterpret_cast<Expr*>(SubStmts[SYNC_EXPR]); 
  }
  
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
  
  const Expr *getThrowExpr() const { return reinterpret_cast<Expr*>(Throw); }
  Expr *getThrowExpr() { return reinterpret_cast<Expr*>(Throw); }
  
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
