//===--- ClangTidyDiagnosticConsumer.h - clang-tidy -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CLANG_TIDY_DIAGNOSTIC_CONSUMER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CLANG_TIDY_DIAGNOSTIC_CONSUMER_H

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Tooling/Refactoring.h"
#include "llvm/ADT/SmallString.h"

namespace clang {

class CompilerInstance;
namespace ast_matchers {
class MatchFinder;
}
namespace tooling {
class CompilationDatabase;
}

namespace tidy {

/// \brief A message from a clang-tidy check.
///
/// Note that this is independent of a \c SourceManager.
struct ClangTidyMessage {
  ClangTidyMessage(StringRef Message = "");
  ClangTidyMessage(StringRef Message, const SourceManager &Sources,
                   SourceLocation Loc);
  std::string Message;
  std::string FilePath;
  unsigned FileOffset;
};

/// \brief A detected error complete with information to display diagnostic and
/// automatic fix.
///
/// This is used as an intermediate format to transport Diagnostics without a
/// dependency on a SourceManager.
///
/// FIXME: Make Diagnostics flexible enough to support this directly.
struct ClangTidyError {
  ClangTidyError(const ClangTidyMessage &Message);

  ClangTidyMessage Message;
  tooling::Replacements Fix;
  SmallVector<ClangTidyMessage, 1> Notes;
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

/// \brief A diagnostic consumer that turns each \c Diagnostic into a
/// \c SourceManager-independent \c ClangTidyError.
//
// FIXME: If we move away from unit-tests, this can be moved to a private
// implementation file.
class ClangTidyDiagnosticConsumer : public DiagnosticConsumer {
public:
  ClangTidyDiagnosticConsumer(ClangTidyContext &Ctx) : Context(Ctx) {
    IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
    Diags.reset(new DiagnosticsEngine(
        IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs), &*DiagOpts, this,
        /*ShouldOwnClient=*/false));
    Context.setDiagnosticsEngine(Diags.get());
  }

  // FIXME: The concept of converting between FixItHints and Replacements is
  // more generic and should be pulled out into a more useful Diagnostics
  // library.
  virtual void HandleDiagnostic(DiagnosticsEngine::Level DiagLevel,
                                const Diagnostic &Info) {
    if (DiagLevel != DiagnosticsEngine::Note) {
      Errors.push_back(ClangTidyError(getMessage(Info)));
    } else {
      Errors.back().Notes.push_back(getMessage(Info));
    }
    addFixes(Info, Errors.back());
  }

  virtual void finish() {
    for (unsigned i = 0, e = Errors.size(); i != e; ++i) {
      Context.storeError(Errors[i]);
    }
  }

private:
  void addFixes(const Diagnostic &Info, ClangTidyError &Error) {
    if (!Info.hasSourceManager())
      return;
    SourceManager &SourceMgr = Info.getSourceManager();
    tooling::Replacements Replacements;
    for (unsigned i = 0, e = Info.getNumFixItHints(); i != e; ++i) {
      Error.Fix.insert(tooling::Replacement(
          SourceMgr, Info.getFixItHint(i).RemoveRange.getBegin(), 0,
          Info.getFixItHint(i).CodeToInsert));
    }
  }

  ClangTidyMessage getMessage(const Diagnostic &Info) const {
    SmallString<100> Buf;
    Info.FormatDiagnostic(Buf);
    if (!Info.hasSourceManager()) {
      return ClangTidyMessage(Buf.str());
    }
    return ClangTidyMessage(Buf.str(), Info.getSourceManager(),
                            Info.getLocation());
  }

  ClangTidyContext &Context;
  OwningPtr<DiagnosticsEngine> Diags;
  SmallVector<ClangTidyError, 8> Errors;
};

} // end namespace tidy
} // end namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CLANG_TIDY_DIAGNOSTIC_CONSUMER_H
