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

protected:
  virtual void DoDestroy(ASTContext& Ctx);

public:
  CXXCatchStmt(SourceLocation catchLoc, VarDecl *exDecl, Stmt *handlerBlock)
  : Stmt(CXXCatchStmtClass), CatchLoc(catchLoc), ExceptionDecl(exDecl),
    HandlerBlock(handlerBlock) {}

  virtual SourceRange getSourceRange() const {
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

  virtual child_iterator child_begin();
  virtual child_iterator child_end();
};

/// CXXTryStmt - A C++ try block, including all handlers.
///
class CXXTryStmt : public Stmt {
  SourceLocation TryLoc;
  unsigned NumHandlers;

  CXXTryStmt(SourceLocation tryLoc, Stmt *tryBlock, Stmt **handlers,
             unsigned numHandlers);

public:
  static CXXTryStmt *Create(ASTContext &C, SourceLocation tryLoc,
                            Stmt *tryBlock, Stmt **handlers,
                            unsigned numHandlers);

  virtual SourceRange getSourceRange() const {
    return SourceRange(getTryLoc(), getEndLoc());
  }

  SourceLocation getTryLoc() const { return TryLoc; }
  SourceLocation getEndLoc() const {
    Stmt const * const*Stmts = reinterpret_cast<Stmt const * const*>(this + 1);
    return Stmts[NumHandlers]->getLocEnd();
  }

  CompoundStmt *getTryBlock() {
    Stmt **Stmts = reinterpret_cast<Stmt **>(this + 1);
    return llvm::cast<CompoundStmt>(Stmts[0]);
  }
  const CompoundStmt *getTryBlock() const {
    Stmt const * const*Stmts = reinterpret_cast<Stmt const * const*>(this + 1);
    return llvm::cast<CompoundStmt>(Stmts[0]);
  }

  unsigned getNumHandlers() const { return NumHandlers; }
  CXXCatchStmt *getHandler(unsigned i) {
    Stmt **Stmts = reinterpret_cast<Stmt **>(this + 1);
    return llvm::cast<CXXCatchStmt>(Stmts[i + 1]);
  }
  const CXXCatchStmt *getHandler(unsigned i) const {
    Stmt const * const*Stmts = reinterpret_cast<Stmt const * const*>(this + 1);
    return llvm::cast<CXXCatchStmt>(Stmts[i + 1]);
  }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == CXXTryStmtClass;
  }
  static bool classof(const CXXTryStmt *) { return true; }

  virtual child_iterator child_begin();
  virtual child_iterator child_end();
};


}  // end namespace clang

#endif
