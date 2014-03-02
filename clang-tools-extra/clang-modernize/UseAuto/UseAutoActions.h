//===-- UseAuto/Actions.h - Matcher callback --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
///  \file
///  \brief This file contains the declarations for callbacks used by the
///  UseAuto transform.
///
//===----------------------------------------------------------------------===//

#ifndef CLANG_MODERNIZE_USE_AUTO_ACTIONS_H
#define CLANG_MODERNIZE_USE_AUTO_ACTIONS_H

#include "Core/Transform.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/Refactoring.h"

/// \brief The callback to be used when replacing type specifiers of variable
/// declarations that are iterators.
class IteratorReplacer
    : public clang::ast_matchers::MatchFinder::MatchCallback {
public:
  IteratorReplacer(unsigned &AcceptedChanges, RiskLevel, Transform &Owner)
      : AcceptedChanges(AcceptedChanges), Owner(Owner) {}

  /// \brief Entry point to the callback called when matches are made.
  virtual void run(const clang::ast_matchers::MatchFinder::MatchResult &Result)
      override;

private:
  unsigned &AcceptedChanges;
  Transform &Owner;
};

/// \brief The callback used when replacing type specifiers of variable
/// declarations initialized by a C++ new expression.
class NewReplacer : public clang::ast_matchers::MatchFinder::MatchCallback {
public:
  NewReplacer(unsigned &AcceptedChanges, RiskLevel, Transform &Owner)
      : AcceptedChanges(AcceptedChanges), Owner(Owner) {}

  /// \brief Entry point to the callback called when matches are made.
  virtual void run(const clang::ast_matchers::MatchFinder::MatchResult &Result)
      override;

private:
  unsigned &AcceptedChanges;
  Transform &Owner;
};

#endif // CLANG_MODERNIZE_USE_AUTO_ACTIONS_H
