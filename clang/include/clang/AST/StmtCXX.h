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

/// CXXCatchStmt - This represents a C++ catch block.
///
class CXXCatchStmt : public Stmt {
  SourceLocation CatchLoc;
  /// The exception-declaration of the type.
  Decl *ExceptionDecl;
  /// The handler block.
  Stmt *HandlerBlock;

public:
  CXXCatchStmt(SourceLocation catchLoc, Decl *exDecl, Stmt *handlerBlock)
  : Stmt(CXXCatchStmtClass), CatchLoc(catchLoc), ExceptionDecl(exDecl),
    HandlerBlock(handlerBlock) {}

  virtual void Destroy(ASTContext& Ctx);

  virtual SourceRange getSourceRange() const {
    return SourceRange(CatchLoc, HandlerBlock->getLocEnd());
  }

  Decl *getExceptionDecl() { return ExceptionDecl; }
  QualType getCaughtType();
  Stmt *getHandlerBlock() { return HandlerBlock; }

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
  // First place is the guarded CompoundStatement. Subsequent are the handlers.
  // More than three handlers should be rare.
  llvm::SmallVector<Stmt*, 4> Stmts;

public:
  CXXTryStmt(SourceLocation tryLoc, Stmt *tryBlock,
             Stmt **handlers, unsigned numHandlers);

  virtual SourceRange getSourceRange() const {
    return SourceRange(TryLoc, Stmts.back()->getLocEnd());
  }

  CompoundStmt *getTryBlock() { return llvm::cast<CompoundStmt>(Stmts[0]); }
  const CompoundStmt *getTryBlock() const {
    return llvm::cast<CompoundStmt>(Stmts[0]);
  }

  unsigned getNumHandlers() const { return Stmts.size() - 1; }
  CXXCatchStmt *getHandler(unsigned i) {
    return llvm::cast<CXXCatchStmt>(Stmts[i + 1]);
  }
  const CXXCatchStmt *getHandler(unsigned i) const {
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
