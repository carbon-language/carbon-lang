//===-- ReplaceAutoPtrActions.h ----- std::auto_ptr replacement -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file contains the declaration of the ASTMatcher callback
/// for the ReplaceAutoPtr transform.
///
//===----------------------------------------------------------------------===//
#ifndef CPP11_MIGRATE_REPLACE_AUTO_PTR_ACTIONS_H
#define CPP11_MIGRATE_REPLACE_AUTO_PTR_ACTIONS_H

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/Refactoring.h"

class Transform;

/// \brief The callback to be used when replacing the \c std::auto_ptr types and
/// using declarations.
class AutoPtrReplacer : public clang::ast_matchers::MatchFinder::MatchCallback {
public:
  AutoPtrReplacer(clang::tooling::Replacements &Replace,
                  unsigned &AcceptedChanges, const Transform &Owner)
      : Replace(Replace), AcceptedChanges(AcceptedChanges), Owner(Owner) {}

  /// \brief Entry point to the callback called when matches are made.
  virtual void run(const clang::ast_matchers::MatchFinder::MatchResult &Result)
      LLVM_OVERRIDE;

private:
  /// \brief Locates the \c auto_ptr token when it is referred by a \c TypeLoc.
  ///
  /// \code
  ///   std::auto_ptr<int> i;
  ///        ^~~~~~~~~~~~~
  /// \endcode
  /// The caret represents the location returned and the tildes cover the
  /// parameter \p AutoPtrTypeLoc.
  ///
  /// \return An invalid \c SourceLocation if not found, otherwise the location
  /// of the beginning of the \c auto_ptr token.
  clang::SourceLocation locateFromTypeLoc(clang::TypeLoc AutoPtrTypeLoc,
                                          const clang::SourceManager &SM);

  /// \brief Locates the \c auto_ptr token in using declarations.
  ///
  /// \code
  ///   using std::auto_ptr;
  ///              ^
  /// \endcode
  /// The caret represents the location returned.
  ///
  /// \return An invalid \c SourceLocation if not found, otherwise the
  /// location of the beginning of the \c auto_ptr token.
  clang::SourceLocation
  locateFromUsingDecl(const clang::UsingDecl *UsingAutoPtrDecl,
                      const clang::SourceManager &SM);

private:
  clang::tooling::Replacements &Replace;
  unsigned &AcceptedChanges;
  const Transform &Owner;
};

/// \brief The callback to be used to fix the ownership transfers of
/// \c auto_ptr,
///
/// \c unique_ptr requires to use \c std::move() explicitly in order to transfer
/// the ownership.
///
/// Given:
/// \code
///   std::auto_ptr<int> a, b;
///   a = b;
/// \endcode
/// The last statement is transformed to:
/// \code
///   a = std::move(b);
/// \endcode
class OwnershipTransferFixer
    : public clang::ast_matchers::MatchFinder::MatchCallback {
public:
  OwnershipTransferFixer(clang::tooling::Replacements &Replace,
                         unsigned &AcceptedChanges, const Transform &Owner)
      : Replace(Replace), AcceptedChanges(AcceptedChanges), Owner(Owner) {}

  /// \brief Entry point to the callback called when matches are made.
  virtual void run(const clang::ast_matchers::MatchFinder::MatchResult &Result)
      LLVM_OVERRIDE;

private:
  clang::tooling::Replacements &Replace;
  unsigned &AcceptedChanges;
  const Transform &Owner;
};

#endif // CPP11_MIGRATE_REPLACE_AUTO_PTR_ACTIONS_H
