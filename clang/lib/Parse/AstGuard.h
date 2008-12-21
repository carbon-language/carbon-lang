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
  /// RAII SmallVector wrapper that holds Action::ExprTy* and similar,
  /// automatically freeing them on destruction unless it's been disowned.
  /// Instantiated for statements and expressions (Action::DeleteStmt and
  /// Action::DeleteExpr).
  template <ASTDestroyer Destroyer, unsigned N>
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

    Action &getActions() const { return Actions; }
  };

  /// A SmallVector of statements, with stack size 32 (as that is the only one
  /// used.)
  typedef ASTVector<&Action::DeleteStmt, 32> StmtVector;
  /// A SmallVector of expressions, with stack size 12 (the maximum used.)
  typedef ASTVector<&Action::DeleteExpr, 12> ExprVector;

  template <ASTDestroyer Destroyer, unsigned N> inline
  ASTMultiPtr<Destroyer> move_convert(ASTVector<Destroyer, N> &vec) {
    return ASTMultiPtr<Destroyer>(vec.getActions(), vec.take(), vec.size());
  }
}

#endif
