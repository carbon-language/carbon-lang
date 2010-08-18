//== PsuedoConstantAnalysis.cpp - Find Psuedoconstants in the AST-*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tracks the usage of variables in a Decl body to see if they are
// never written to, implying that they constant. This is useful in static
// analysis to see if a developer might have intended a variable to be const.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Analyses/PsuedoConstantAnalysis.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"
#include <deque>

using namespace clang;

// Returns true if the given ValueDecl is never written to in the given DeclBody
bool PsuedoConstantAnalysis::isPsuedoConstant(const ValueDecl *VD) {
  if (!Analyzed) {
    RunAnalysis();
    Analyzed = true;
  }

  return !NonConstants.count(VD);
}

void PsuedoConstantAnalysis::RunAnalysis() {
  std::deque<const Stmt *> WorkList;

  // Start with the top level statement of the function
  WorkList.push_back(DeclBody);

  while (!WorkList.empty()) {
    const Stmt* Head = WorkList.front();
    WorkList.pop_front();

    switch (Head->getStmtClass()) {
    // Case 1: Assignment operators modifying ValueDecl
    case Stmt::BinaryOperatorClass: {
      const BinaryOperator *BO = cast<BinaryOperator>(Head);
      const Expr *LHS = BO->getLHS()->IgnoreParenImpCasts();
      const DeclRefExpr *DR = dyn_cast<DeclRefExpr>(LHS);

      // We only care about DeclRefExprs on the LHS
      if (!DR)
        break;

      // We found a binary operator with a DeclRefExpr on the LHS. We now check
      // for any of the assignment operators, implying that this Decl is being
      // written to.
      switch (BO->getOpcode()) {
      case BinaryOperator::Assign:
      case BinaryOperator::AddAssign:
      case BinaryOperator::SubAssign:
      case BinaryOperator::MulAssign:
      case BinaryOperator::DivAssign:
      case BinaryOperator::AndAssign:
      case BinaryOperator::OrAssign:
      case BinaryOperator::XorAssign:
      case BinaryOperator::ShlAssign:
      case BinaryOperator::ShrAssign:
        // The DeclRefExpr is being assigned to - mark it as non-constant
        NonConstants.insert(DR->getDecl());
        continue; // Continue without looking at children

      default:
        break;
      }
      break;
    }

    // Case 2: Pre/post increment/decrement and address of
    case Stmt::UnaryOperatorClass: {
      const UnaryOperator *UO = cast<UnaryOperator>(Head);
      const Expr *SubExpr = UO->getSubExpr()->IgnoreParenImpCasts();
      const DeclRefExpr *DR = dyn_cast<DeclRefExpr>(SubExpr);

      // We only care about DeclRefExprs in the subexpression
      if (!DR)
        break;

      // We found a unary operator with a DeclRefExpr as a subexpression. We now
      // check for any of the increment/decrement operators, as well as
      // addressOf.
      switch (UO->getOpcode()) {
      case UnaryOperator::PostDec:
      case UnaryOperator::PostInc:
      case UnaryOperator::PreDec:
      case UnaryOperator::PreInc:
        // The DeclRefExpr is being changed - mark it as non-constant
      case UnaryOperator::AddrOf:
        // If we are taking the address of the DeclRefExpr, assume it is
        // non-constant.
        NonConstants.insert(DR->getDecl());

      default:
        break;
      }
      break;
    }

      default:
        break;
    } // switch (head->getStmtClass())

    // Add all substatements to the worklist
    for (Stmt::const_child_iterator I = Head->child_begin(),
        E = Head->child_end(); I != E; ++I)
      if (*I)
        WorkList.push_back(*I);
  } // while (!WorkList.empty())
}
