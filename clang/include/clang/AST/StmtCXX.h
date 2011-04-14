//===--- StmtCXX.h - Classes for representing C++ statements ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the C++ statement AST node classes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_STMTCXX_H
#define LLVM_CLANG_AST_STMTCXX_H

#include "clang/AST/Stmt.h"

namespace clang {

class VarDecl;

/// CXXCatchStmt - This represents a C++ catch block.
///
class CXXCatchStmt : public Stmt {
  SourceLocation CatchLoc;
  /// The exception-declaration of the type.
  VarDecl *ExceptionDecl;
  /// The handler block.
  Stmt *HandlerBlock;

public:
  CXXCatchStmt(SourceLocation catchLoc, VarDecl *exDecl, Stmt *handlerBlock)
  : Stmt(CXXCatchStmtClass), CatchLoc(catchLoc), ExceptionDecl(exDecl),
    HandlerBlock(handlerBlock) {}

  CXXCatchStmt(EmptyShell Empty)
  : Stmt(CXXCatchStmtClass), ExceptionDecl(0), HandlerBlock(0) {}

  SourceRange getSourceRange() const {
    return SourceRange(CatchLoc, HandlerBlock->getLocEnd());
  }

  SourceLocation getCatchLoc() const { return CatchLoc; }
  VarDecl *getExceptionDecl() const { return ExceptionDecl; }
  QualType getCaughtType() const;
  Stmt *getHandlerBlock() const { return HandlerBlock; }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == CXXCatchStmtClass;
  }
  static bool classof(const CXXCatchStmt *) { return true; }

  child_range children() { return child_range(&HandlerBlock, &HandlerBlock+1); }

  friend class ASTStmtReader;
};

/// CXXTryStmt - A C++ try block, including all handlers.
///
class CXXTryStmt : public Stmt {
  SourceLocation TryLoc;
  unsigned NumHandlers;

  CXXTryStmt(SourceLocation tryLoc, Stmt *tryBlock, Stmt **handlers,
             unsigned numHandlers);

  CXXTryStmt(EmptyShell Empty, unsigned numHandlers)
    : Stmt(CXXTryStmtClass), NumHandlers(numHandlers) { }

  Stmt const * const *getStmts() const {
    return reinterpret_cast<Stmt const * const*>(this + 1);
  }
  Stmt **getStmts() {
    return reinterpret_cast<Stmt **>(this + 1);
  }

public:
  static CXXTryStmt *Create(ASTContext &C, SourceLocation tryLoc,
                            Stmt *tryBlock, Stmt **handlers,
                            unsigned numHandlers);

  static CXXTryStmt *Create(ASTContext &C, EmptyShell Empty,
                            unsigned numHandlers);

  SourceRange getSourceRange() const {
    return SourceRange(getTryLoc(), getEndLoc());
  }

  SourceLocation getTryLoc() const { return TryLoc; }
  SourceLocation getEndLoc() const {
    return getStmts()[NumHandlers]->getLocEnd();
  }

  CompoundStmt *getTryBlock() {
    return llvm::cast<CompoundStmt>(getStmts()[0]);
  }
  const CompoundStmt *getTryBlock() const {
    return llvm::cast<CompoundStmt>(getStmts()[0]);
  }

  unsigned getNumHandlers() const { return NumHandlers; }
  CXXCatchStmt *getHandler(unsigned i) {
    return llvm::cast<CXXCatchStmt>(getStmts()[i + 1]);
  }
  const CXXCatchStmt *getHandler(unsigned i) const {
    return llvm::cast<CXXCatchStmt>(getStmts()[i + 1]);
  }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == CXXTryStmtClass;
  }
  static bool classof(const CXXTryStmt *) { return true; }

  child_range children() {
    return child_range(getStmts(), getStmts() + getNumHandlers() + 1);
  }

  friend class ASTStmtReader;
};

/// CXXForRangeStmt - This represents C++0x [stmt.ranged]'s ranged for
/// statement, represented as 'for (range-declarator : range-expression)'.
///
/// This is stored in a partially-desugared form to allow full semantic
/// analysis of the constituent components. The original syntactic components
/// can be extracted using getLoopVariable and getRangeInit.
class CXXForRangeStmt : public Stmt {
  enum { RANGE, BEGINEND, COND, INC, LOOPVAR, BODY, END };
  // SubExprs[RANGE] is an expression or declstmt.
  // SubExprs[COND] and SubExprs[INC] are expressions.
  Stmt *SubExprs[END];
  SourceLocation ForLoc;
  SourceLocation ColonLoc;
  SourceLocation RParenLoc;
public:
  CXXForRangeStmt(DeclStmt *Range, DeclStmt *BeginEnd,
                  Expr *Cond, Expr *Inc, DeclStmt *LoopVar, Stmt *Body,
                  SourceLocation FL, SourceLocation CL, SourceLocation RPL);
  CXXForRangeStmt(EmptyShell Empty) : Stmt(CXXForRangeStmtClass, Empty) { }


  VarDecl *getLoopVariable();
  Expr *getRangeInit();

  const VarDecl *getLoopVariable() const;
  const Expr *getRangeInit() const;


  DeclStmt *getRangeStmt() { return cast<DeclStmt>(SubExprs[RANGE]); }
  DeclStmt *getBeginEndStmt() { return cast_or_null<DeclStmt>(SubExprs[BEGINEND]); }
  Expr *getCond() { return cast_or_null<Expr>(SubExprs[COND]); }
  Expr *getInc() { return cast_or_null<Expr>(SubExprs[INC]); }
  DeclStmt *getLoopVarStmt() { return cast<DeclStmt>(SubExprs[LOOPVAR]); }
  Stmt *getBody() { return SubExprs[BODY]; }

  const DeclStmt *getRangeStmt() const {
    return cast<DeclStmt>(SubExprs[RANGE]);
  }
  const DeclStmt *getBeginEndStmt() const {
    return cast_or_null<DeclStmt>(SubExprs[BEGINEND]);
  }
  const Expr *getCond() const {
    return cast_or_null<Expr>(SubExprs[COND]);
  }
  const Expr *getInc() const {
    return cast_or_null<Expr>(SubExprs[INC]);
  }
  const DeclStmt *getLoopVarStmt() const {
    return cast<DeclStmt>(SubExprs[LOOPVAR]);
  }
  const Stmt *getBody() const { return SubExprs[BODY]; }

  void setRangeInit(Expr *E) { SubExprs[RANGE] = reinterpret_cast<Stmt*>(E); }
  void setRangeStmt(Stmt *S) { SubExprs[RANGE] = S; }
  void setBeginEndStmt(Stmt *S) { SubExprs[BEGINEND] = S; }
  void setCond(Expr *E) { SubExprs[COND] = reinterpret_cast<Stmt*>(E); }
  void setInc(Expr *E) { SubExprs[INC] = reinterpret_cast<Stmt*>(E); }
  void setLoopVarStmt(Stmt *S) { SubExprs[LOOPVAR] = S; }
  void setBody(Stmt *S) { SubExprs[BODY] = S; }


  SourceLocation getForLoc() const { return ForLoc; }
  void setForLoc(SourceLocation Loc) { ForLoc = Loc; }
  SourceLocation getColonLoc() const { return ColonLoc; }
  void setColonLoc(SourceLocation Loc) { ColonLoc = Loc; }
  SourceLocation getRParenLoc() const { return RParenLoc; }
  void setRParenLoc(SourceLocation Loc) { RParenLoc = Loc; }

  SourceRange getSourceRange() const {
    return SourceRange(ForLoc, SubExprs[BODY]->getLocEnd());
  }
  static bool classof(const Stmt *T) {
    return T->getStmtClass() == CXXForRangeStmtClass;
  }
  static bool classof(const CXXForRangeStmt *) { return true; }

  // Iterators
  child_range children() {
    return child_range(&SubExprs[0], &SubExprs[END]);
  }
};


}  // end namespace clang

#endif
