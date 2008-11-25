//===--- AstGuard.h - Parser Ownership Tracking Utilities -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines RAII objects for managing ExprTy* and StmtTy*.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_PARSE_ASTGUARD_H
#define LLVM_CLANG_PARSE_ASTGUARD_H

#include "clang/Parse/Action.h"
#include "llvm/ADT/SmallVector.h"

namespace clang
{
  /// RAII guard for freeing StmtTys and ExprTys on early exit in the parser.
  /// Instantiated for statements and expressions (Action::DeleteStmt and
  /// Action::DeleteExpr).
  template <void (Action::*Destroyer)(void*)>
  class ASTGuard {
    Action &Actions;
    void *Node;

    void destroy() {
      if (Node)
        (Actions.*Destroyer)(Node);
    }

    ASTGuard(const ASTGuard&); // DO NOT IMPLEMENT
    // Reference member prevents copy assignment.

  public:
    explicit ASTGuard(Action &actions) : Actions(actions), Node(0) {}
    ASTGuard(Action &actions, void *node)
      : Actions(actions), Node(node) {}
    template <unsigned N>
    ASTGuard(Action &actions, const Action::ActionResult<N> &res)
      : Actions(actions), Node(res.Val) {}
    ~ASTGuard() { destroy(); }

    void reset(void *element) {
      destroy();
      Node = element;
    }
    template <unsigned N>
    void reset(const Action::ActionResult<N> &res) {
      reset(res.Val);
    }
    void *take() {
      void *Temp = Node;
      Node = 0;
      return Temp;
    }
    void *get() const { return Node; }
  };

  typedef ASTGuard<&Action::DeleteStmt> StmtGuard;
  typedef ASTGuard<&Action::DeleteExpr> ExprGuard;

  /// RAII SmallVector wrapper that holds Action::ExprTy* and similar,
  /// automatically freeing them on destruction unless it's been disowned.
  /// Instantiated for statements and expressions (Action::DeleteStmt and
  /// Action::DeleteExpr).
  template <void (Action::*Destroyer)(void*), unsigned N>
  class ASTVector : public llvm::SmallVector<void*, N> {
  private:
    Action &Actions;
    bool Owns;

    void destroy() {
      if (Owns) {
        while (!this->empty()) {
          (Actions.*Destroyer)(this->back());
          this->pop_back();
        }
      }
    }

    ASTVector(const ASTVector&); // DO NOT IMPLEMENT
    // Reference member prevents copy assignment.

  public:
    ASTVector(Action &actions) : Actions(actions), Owns(true) {}

    ~ASTVector() { destroy(); }

    void **take() {
      Owns = false;
      return &(*this)[0];
    }
  };

  /// A SmallVector of statements, with stack size 32 (as that is the only one
  /// used.)
  typedef ASTVector<&Action::DeleteStmt, 32> StmtVector;
  /// A SmallVector of expressions, with stack size 12 (the maximum used.)
  typedef ASTVector<&Action::DeleteExpr, 12> ExprVector;
}

#endif
