//===-- PassByValueActions.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file contains the declaration of the ASTMatcher callback for the
/// PassByValue transform.
///
//===----------------------------------------------------------------------===//

#ifndef CLANG_MODERNIZE_PASS_BY_VALUE_ACTIONS_H
#define CLANG_MODERNIZE_PASS_BY_VALUE_ACTIONS_H

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/Refactoring.h"

class Transform;
class IncludeDirectives;

/// \brief Callback that replaces const-ref parameters in constructors to use
/// pass-by-value semantic where applicable.
///
/// Modifications done by the callback:
/// - \#include \<utility\> is added if necessary for the definition of
///   \c std::move() to be available.
/// - The parameter type is changed from const-ref to value-type.
/// - In the init-list the parameter is moved.
///
/// Example:
/// \code
/// + #include <utility>
///
/// class Foo(const std::string &S) {
/// public:
///   - Foo(const std::string &S) : S(S) {}
///   + Foo(std::string S) : S(std::move(S)) {}
///
/// private:
///   std::string S;
/// };
/// \endcode
///
/// \note Since an include may be added by this matcher it's necessary to call
/// \c setIncludeDirectives() with an up-to-date \c IncludeDirectives. This is
/// typically done by overloading \c Transform::handleBeginSource().
class ConstructorParamReplacer
    : public clang::ast_matchers::MatchFinder::MatchCallback {
public:
  ConstructorParamReplacer(unsigned &AcceptedChanges, unsigned &RejectedChanges,
                           Transform &Owner)
      : AcceptedChanges(AcceptedChanges), RejectedChanges(RejectedChanges),
        Owner(Owner), IncludeManager(nullptr) {}

  void setIncludeDirectives(IncludeDirectives *Includes) {
    IncludeManager = Includes;
  }

private:
  /// \brief Entry point to the callback called when matches are made.
  virtual void run(const clang::ast_matchers::MatchFinder::MatchResult &Result)
      override;

  unsigned &AcceptedChanges;
  unsigned &RejectedChanges;
  Transform &Owner;
  IncludeDirectives *IncludeManager;
};

#endif // CLANG_MODERNIZE_PASS_BY_VALUE_ACTIONS_H
