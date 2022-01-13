//===--- Diagnostics.h -------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_DIAGNOSTICS_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_DIAGNOSTICS_H

#include "Protocol.h"
#include "support/Path.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/SourceMgr.h"
#include <cassert>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace clang {
namespace tidy {
class ClangTidyContext;
} // namespace tidy
namespace clangd {

struct ClangdDiagnosticOptions {
  /// If true, Clangd uses an LSP extension to embed the fixes with the
  /// diagnostics that are sent to the client.
  bool EmbedFixesInDiagnostics = false;

  /// If true, Clangd uses the relatedInformation field to include other
  /// locations (in particular attached notes).
  /// Otherwise, these are flattened into the diagnostic message.
  bool EmitRelatedLocations = false;

  /// If true, Clangd uses an LSP extension to send the diagnostic's
  /// category to the client. The category typically describes the compilation
  /// stage during which the issue was produced, e.g. "Semantic Issue" or "Parse
  /// Issue".
  bool SendDiagnosticCategory = false;

  /// If true, Clangd will add a number of available fixes to the diagnostic's
  /// message.
  bool DisplayFixesCount = true;
};

/// Contains basic information about a diagnostic.
struct DiagBase {
  std::string Message;
  // Intended to be used only in error messages.
  // May be relative, absolute or even artificially constructed.
  std::string File;
  // Absolute path to containing file, if available.
  llvm::Optional<std::string> AbsFile;

  clangd::Range Range;
  DiagnosticsEngine::Level Severity = DiagnosticsEngine::Note;
  std::string Category;
  // Since File is only descriptive, we store a separate flag to distinguish
  // diags from the main file.
  bool InsideMainFile = false;
  unsigned ID; // e.g. member of clang::diag, or clang-tidy assigned ID.
  // Feature modules can make use of this field to propagate data from a
  // diagnostic to a CodeAction request. Each module should only append to the
  // list.
  llvm::json::Object OpaqueData;
};
llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const DiagBase &D);

/// Represents a single fix-it that editor can apply to fix the error.
struct Fix {
  /// Message for the fix-it.
  std::string Message;
  /// TextEdits from clang's fix-its. Must be non-empty.
  llvm::SmallVector<TextEdit, 1> Edits;
};
llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const Fix &F);

/// Represents a note for the diagnostic. Severity of notes can only be 'note'
/// or 'remark'.
struct Note : DiagBase {};

/// A top-level diagnostic that may have Notes and Fixes.
struct Diag : DiagBase {
  std::string Name; // if ID was recognized.
  // The source of this diagnostic.
  enum DiagSource {
    Unknown,
    Clang,
    ClangTidy,
    Clangd,
    ClangdConfig,
  } Source = Unknown;
  /// Elaborate on the problem, usually pointing to a related piece of code.
  std::vector<Note> Notes;
  /// *Alternative* fixes for this diagnostic, one should be chosen.
  std::vector<Fix> Fixes;
  llvm::SmallVector<DiagnosticTag, 1> Tags;
};
llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const Diag &D);

Diag toDiag(const llvm::SMDiagnostic &, Diag::DiagSource Source);

/// Conversion to LSP diagnostics. Formats the error message of each diagnostic
/// to include all its notes. Notes inside main file are also provided as
/// separate diagnostics with their corresponding fixits. Notes outside main
/// file do not have a corresponding LSP diagnostic, but can still be included
/// as part of their main diagnostic's message.
void toLSPDiags(
    const Diag &D, const URIForFile &File, const ClangdDiagnosticOptions &Opts,
    llvm::function_ref<void(clangd::Diagnostic, llvm::ArrayRef<Fix>)> OutFn);

/// Convert from Fix to LSP CodeAction.
CodeAction toCodeAction(const Fix &D, const URIForFile &File);

/// Convert from clang diagnostic level to LSP severity.
int getSeverity(DiagnosticsEngine::Level L);

/// StoreDiags collects the diagnostics that can later be reported by
/// clangd. It groups all notes for a diagnostic into a single Diag
/// and filters out diagnostics that don't mention the main file (i.e. neither
/// the diag itself nor its notes are in the main file).
class StoreDiags : public DiagnosticConsumer {
public:
  // The ClangTidyContext populates Source and Name for clang-tidy diagnostics.
  std::vector<Diag> take(const clang::tidy::ClangTidyContext *Tidy = nullptr);

  void BeginSourceFile(const LangOptions &Opts,
                       const Preprocessor *PP) override;
  void EndSourceFile() override;
  void HandleDiagnostic(DiagnosticsEngine::Level DiagLevel,
                        const clang::Diagnostic &Info) override;

  /// When passed a main diagnostic, returns fixes to add to it.
  /// When passed a note diagnostic, returns fixes to replace it with.
  using DiagFixer = std::function<std::vector<Fix>(DiagnosticsEngine::Level,
                                                   const clang::Diagnostic &)>;
  using LevelAdjuster = std::function<DiagnosticsEngine::Level(
      DiagnosticsEngine::Level, const clang::Diagnostic &)>;
  using DiagCallback =
      std::function<void(const clang::Diagnostic &, clangd::Diag &)>;
  /// If set, possibly adds fixes for diagnostics using \p Fixer.
  void contributeFixes(DiagFixer Fixer) { this->Fixer = Fixer; }
  /// If set, this allows the client of this class to adjust the level of
  /// diagnostics, such as promoting warnings to errors, or ignoring
  /// diagnostics.
  void setLevelAdjuster(LevelAdjuster Adjuster) { this->Adjuster = Adjuster; }
  /// Invokes a callback every time a diagnostics is completely formed. Handler
  /// of the callback can also mutate the diagnostic.
  void setDiagCallback(DiagCallback CB) { DiagCB = std::move(CB); }

private:
  void flushLastDiag();

  DiagFixer Fixer = nullptr;
  LevelAdjuster Adjuster = nullptr;
  DiagCallback DiagCB = nullptr;
  std::vector<Diag> Output;
  llvm::Optional<LangOptions> LangOpts;
  llvm::Optional<Diag> LastDiag;
  llvm::Optional<FullSourceLoc> LastDiagLoc; // Valid only when LastDiag is set.
  bool LastDiagOriginallyError = false;      // Valid only when LastDiag is set.
  SourceManager *OrigSrcMgr = nullptr;

  llvm::DenseSet<std::pair<unsigned, unsigned>> IncludedErrorLocations;
};

/// Determine whether a (non-clang-tidy) diagnostic is suppressed by config.
bool isBuiltinDiagnosticSuppressed(unsigned ID,
                                   const llvm::StringSet<> &Suppressed);
/// Take a user-specified diagnostic code, and convert it to a normalized form
/// stored in the config and consumed by isBuiltinDiagnosticsSuppressed.
///
/// (This strips err_ and -W prefix so we can match with or without them.)
llvm::StringRef normalizeSuppressedCode(llvm::StringRef);

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_DIAGNOSTICS_H
