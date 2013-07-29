//===--- ClangTidy.h - clang-tidy -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CLANG_TIDY_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CLANG_TIDY_H

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Tooling/Refactoring.h"

namespace clang {

class CompilerInstance;
namespace ast_matchers {
class MatchFinder;
}
namespace tooling {
class CompilationDatabase;
}

namespace tidy {

/// \brief A detected error complete with information to display diagnostic and
/// automatic fix.
///
/// This is used as an intermediate format to transport Diagnostics without a
/// dependency on a SourceManager.
///
/// FIXME: Make Diagnostics flexible enough to support this directly.
struct ClangTidyError {
  ClangTidyError(const SourceManager &Sources, SourceLocation Loc,
                 StringRef Message, const tooling::Replacements &Fix);

  std::string Message;
  unsigned FileOffset;
  std::string FilePath;
  tooling::Replacements Fix;
};

/// \brief Every \c ClangTidyCheck reports errors through a \c DiagnosticEngine
/// provided by this context.
///
/// A \c ClangTidyCheck always has access to the active context to report
/// warnings like:
/// \code
/// Context->Diag(Loc, "Single-argument constructors must be explicit")
///     << FixItHint::CreateInsertion(Loc, "explicit ");
/// \endcode
class ClangTidyContext {
public:
  ClangTidyContext(SmallVectorImpl<ClangTidyError> *Errors) : Errors(Errors) {}

  /// \brief Report any errors detected using this method.
  ///
  /// This is still under heavy development and will likely change towards using
  /// tablegen'd diagnostic IDs.
  /// FIXME: Figure out a way to manage ID spaces.
  DiagnosticBuilder Diag(SourceLocation Loc, StringRef Message);

  /// \brief Sets the \c DiagnosticsEngine so that Diagnostics can be generated
  /// correctly.
  ///
  /// This is called from the \c ClangTidyCheck base class.
  void setDiagnosticsEngine(DiagnosticsEngine *Engine);

  /// \brief Sets the \c SourceManager of the used \c DiagnosticsEngine.
  ///
  /// This is called from the \c ClangTidyCheck base class.
  void setSourceManager(SourceManager *SourceMgr);

private:
  friend class ClangTidyDiagnosticConsumer; // Calls storeError().

  /// \brief Store a \c ClangTidyError.
  void storeError(const ClangTidyError &Error);

  SmallVectorImpl<ClangTidyError> *Errors;
  DiagnosticsEngine *DiagEngine;
};

/// \brief Base class for all clang-tidy checks.
///
/// To implement a \c ClangTidyCheck, write a subclass and overwrite some of the
/// base class's methods. E.g. to implement a check that validates namespace
/// declarations, overwrite \c registerMatchers:
///
/// \code
/// registerMatchers(ast_matchers::MatchFinder *Finder) {
///   Finder->addMatcher(namespaceDecl().bind("namespace"), this);
/// }
/// \endcode
///
/// and then overwrite \c check(const MatchResult &Result) to do the actual
/// check for each match.
///
/// A new \c ClangTidyCheck instance is created per translation unit.
///
/// FIXME: Figure out whether carrying information from one TU to another is
/// useful/necessary.
class ClangTidyCheck : public ast_matchers::MatchFinder::MatchCallback {
public:
  virtual ~ClangTidyCheck() {}

  /// \brief Overwrite this to register \c PPCallbacks with \c Compiler.
  ///
  /// This should be used for clang-tidy checks that analyze preprocessor-
  /// dependent properties, e.g. the order of include directives.
  virtual void registerPPCallbacks(CompilerInstance &Compiler) {}

  /// \brief Overwrite this to register ASTMatchers with \p Finder.
  ///
  /// This should be used by clang-tidy checks that analyze code properties that
  /// dependent on AST knowledge.
  ///
  /// You can register as many matchers as necessary with \p Finder. Usually,
  /// "this" will be used as callback, but you can also specify other callback
  /// classes. Thereby, different matchers can trigger different callbacks.
  ///
  /// If you need to merge information between the different matchers, you can
  /// store these as members of the derived class. However, note that all
  /// matches occur in the order of the AST traversal.
  virtual void registerMatchers(ast_matchers::MatchFinder *Finder) {}

  /// \brief \c ClangTidyChecks that register ASTMatchers should do the actual
  /// work in here.
  virtual void check(const ast_matchers::MatchFinder::MatchResult &Result) {}

  /// \brief The infrastructure sets the context to \p Ctx with this function.
  void setContext(ClangTidyContext *Ctx) { Context = Ctx; }

protected:
  ClangTidyContext *Context;

private:
  virtual void run(const ast_matchers::MatchFinder::MatchResult &Result);
};

/// \brief Run a set of clang-tidy checks on a set of files.
void runClangTidy(StringRef CheckRegexString,
                  const tooling::CompilationDatabase &Compilations,
                  ArrayRef<std::string> Ranges,
                  SmallVectorImpl<ClangTidyError> *Errors);

// FIXME: This interface will need to be significantly extended to be useful.
// FIXME: Implement confidence levels for displaying/fixing errors.
//
/// \brief Displays the found \p Errors to the users. If \p Fix is true, \p
/// Errors containing fixes are automatically applied.
void handleErrors(SmallVectorImpl<ClangTidyError> &Errors, bool Fix);

} // end namespace tidy
} // end namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CLANG_TIDY_H
