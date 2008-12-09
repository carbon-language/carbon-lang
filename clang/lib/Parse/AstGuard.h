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
  template <void (Action::*Destroyer)(void*)>
  class ASTOwner;

  typedef ASTOwner<&Action::DeleteStmt> StmtOwner;
  typedef ASTOwner<&Action::DeleteExpr> ExprOwner;

  /// Some trickery to switch between an ActionResult and an ASTOwner
  template <typename Owner> struct ResultOfOwner;
  template <> struct ResultOfOwner<StmtOwner> {
    typedef Action::StmtResult type;
  };
  template <> struct ResultOfOwner<ExprOwner> {
    typedef Action::ExprResult type;
  };

  /// Move emulation helper for ASTOwner. Implicitly convertible to ActionResult
  /// and void*, which means ASTOwner::move() can be used universally.
  template <void (Action::*Destroyer)(void*)>
  class ASTMove {
    ASTOwner<Destroyer> &Moved;

  public:
    explicit ASTMove(ASTOwner<Destroyer> &moved) : Moved(moved) {}
    ASTOwner<Destroyer> * operator ->() {
      return &Moved;
    }

    /// Allow moving from ASTOwner to ActionResult
    operator typename ResultOfOwner< ASTOwner<Destroyer> >::type() {
      if (Moved.isInvalid())
        return true;
      return Moved.take();
    }

    /// Allow moving from ASTOwner to void*
    operator void*() {
      if (Moved.isInvalid())
        return 0;
      return Moved.take();
    }
  };

  /// RAII owning pointer for StmtTys and ExprTys. Simple move emulation.
  template <void (Action::*Destroyer)(void*)>
  class ASTOwner {
    typedef typename ResultOfOwner<ASTOwner>::type Result;

    Action &Actions;
    void *Node;
    bool Invalid;

    void destroy() {
      if (Node)
        (Actions.*Destroyer)(Node);
    }

    ASTOwner(const ASTOwner&); // DO NOT IMPLEMENT
    // Reference member prevents copy assignment.

  public:
    explicit ASTOwner(Action &actions, bool invalid = false)
      : Actions(actions), Node(0), Invalid(invalid) {}
    ASTOwner(Action &actions, void *node)
      : Actions(actions), Node(node), Invalid(false) {}
    ASTOwner(Action &actions, const Result &res)
      : Actions(actions), Node(res.Val), Invalid(res.isInvalid) {}
    /// Move constructor
    ASTOwner(ASTMove<Destroyer> mover)
      : Actions(mover->Actions), Node(mover->take()), Invalid(mover->Invalid) {}
    /// Move assignment
    ASTOwner & operator =(ASTMove<Destroyer> mover) {
      assert(&Actions == &mover->Actions &&
             "AST Owners from different actions.");
      destroy();
      Node = mover->take();
      Invalid = mover->Invalid;
      return *this;
    }
    /// Convenience, for better syntax. reset() is so ugly. Just remember that
    /// this takes ownership.
    ASTOwner & operator =(const Result &res) {
      reset(res);
      return *this;
    }

    void reset(void *node = 0) {
      destroy();
      Node = node;
      Invalid = false;
    }
    void reset(const Result &res) {
      destroy();
      Node = res.Val;
      Invalid = res.isInvalid;
    }
    /// Take ownership from this pointer and return the node. Calling move() is
    /// better.
    void *take() {
      void *Temp = Node;
      Node = 0;
      return Temp;
    }
    void *get() const { return Node; }
    bool isInvalid() const { return Invalid; }
    /// Does this point to a usable AST node? To be usable, the node must be
    /// valid and non-null.
    bool isUsable() const { return !Invalid && Node; }

    ASTMove<Destroyer> move() {
      return ASTMove<Destroyer>(*this);
    }
  };

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
