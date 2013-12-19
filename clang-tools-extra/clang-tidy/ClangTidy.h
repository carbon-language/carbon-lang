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
#include "ClangTidyDiagnosticConsumer.h"

namespace clang {

class CompilerInstance;
namespace ast_matchers {
class MatchFinder;
}
namespace tooling {
class CompilationDatabase;
}

namespace tidy {

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

/// \brief Filters checks by name.
class ChecksFilter {
public:
  ChecksFilter(StringRef EnableChecksRegex, StringRef DisableChecksRegex);
  bool IsCheckEnabled(StringRef Name);

private:
  llvm::Regex EnableChecks;
  llvm::Regex DisableChecks;
};

/// \brief Fills the list of check names that are enabled when the provided
/// filters are applied.
std::vector<std::string> getCheckNames(StringRef EnableChecksRegex,
                                       StringRef DisableChecksRegex);

/// \brief Returns an action factory that runs the specified clang-tidy checks.
tooling::FrontendActionFactory *
createClangTidyActionFactory(StringRef EnableChecksRegex,
                             StringRef DisableChecksRegex,
                             ClangTidyContext &Context);

/// \brief Run a set of clang-tidy checks on a set of files.
void runClangTidy(StringRef EnableChecksRegex, StringRef DisableChecksRegex,
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
