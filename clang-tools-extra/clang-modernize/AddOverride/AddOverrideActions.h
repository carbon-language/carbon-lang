//===-- AddOverride/AddOverrideActions.h - add C++11 override ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file contains the declaration of the AddOverrideFixer class
/// which is used as a ASTMatcher callback.
///
//===----------------------------------------------------------------------===//

#ifndef CLANG_MODERNIZE_ADD_OVERRIDE_ACTIONS_H
#define CLANG_MODERNIZE_ADD_OVERRIDE_ACTIONS_H

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/Refactoring.h"

class Transform;

/// \brief The callback to be used for add-override migration matchers.
///
class AddOverrideFixer : public clang::ast_matchers::MatchFinder::MatchCallback {
public:
  AddOverrideFixer(unsigned &AcceptedChanges, bool DetectMacros,
                   Transform &Owner)
      : AcceptedChanges(AcceptedChanges), DetectMacros(DetectMacros),
        Owner(Owner) {}

  /// \brief Entry point to the callback called when matches are made.
  virtual void run(const clang::ast_matchers::MatchFinder::MatchResult &Result);

  void setPreprocessor(clang::Preprocessor &PP) { this->PP = &PP; }

private:
  clang::Preprocessor *PP;
  unsigned &AcceptedChanges;
  bool DetectMacros;
  Transform &Owner;
};

#endif // CLANG_MODERNIZE_ADD_OVERRIDE_ACTIONS_H
