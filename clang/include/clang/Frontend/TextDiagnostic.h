//===--- TextDiagnostic.h - Text Diagnostic Pretty-Printing -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is a utility class that provides support for textual pretty-printing of
// diagnostics. It is used to implement the different code paths which require
// such functionality in a consistent way.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FRONTEND_TEXT_DIAGNOSTIC_H_
#define LLVM_CLANG_FRONTEND_TEXT_DIAGNOSTIC_H_

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/SourceLocation.h"

namespace clang {
class DiagnosticOptions;
class LangOptions;
class SourceManager;

/// \brief Class to encapsulate the logic for formatting and printing a textual
/// diagnostic message.
///
/// This class provides an interface for building and emitting a textual
/// diagnostic, including all of the macro backtraces, caret diagnostics, FixIt
/// Hints, and code snippets. In the presence of macros this involves
/// a recursive process, synthesizing notes for each macro expansion.
///
/// The purpose of this class is to isolate the implementation of printing
/// beautiful text diagnostics from any particular interfaces. The Clang
/// DiagnosticClient is implemented through this class as is diagnostic
/// printing coming out of libclang.
///
/// A brief worklist:
/// FIXME: Sink the recursive printing of template instantiations into this
/// class.
class TextDiagnostic {
  raw_ostream &OS;
  const SourceManager &SM;
  const LangOptions &LangOpts;
  const DiagnosticOptions &DiagOpts;

  /// \brief The location of the previous diagnostic if known.
  ///
  /// This will be invalid in cases where there is no (known) previous
  /// diagnostic location, or that location itself is invalid or comes from
  /// a different source manager than SM.
  SourceLocation LastLoc;

  /// \brief The location of the last include whose stack was printed if known.
  ///
  /// Same restriction as \see LastLoc essentially, but tracking include stack
  /// root locations rather than diagnostic locations.
  SourceLocation LastIncludeLoc;

  /// \brief The level of the last diagnostic emitted.
  ///
  /// The level of the last diagnostic emitted. Used to detect level changes
  /// which change the amount of information displayed.
  DiagnosticsEngine::Level LastLevel;

public:
  TextDiagnostic(raw_ostream &OS,
                 const SourceManager &SM,
                 const LangOptions &LangOpts,
                 const DiagnosticOptions &DiagOpts);

  /// \brief Emit a textual diagnostic.
  ///
  /// This is the primary entry point for emitting textual diagnostic messages.
  /// It handles formatting and printing the message as well as any ancillary
  /// information needed based on macros whose expansions impact the
  /// diagnostic.
  ///
  /// \param Loc The location for this caret.
  /// \param Level The level of the diagnostic to be emitted.
  /// \param Message The diagnostic message to emit.
  /// \param Ranges The underlined ranges for this code snippet.
  /// \param FixItHints The FixIt hints active for this diagnostic.
  void emitDiagnostic(SourceLocation Loc, DiagnosticsEngine::Level Level,
                      StringRef Message, ArrayRef<CharSourceRange> Ranges,
                      ArrayRef<FixItHint> FixItHints);

  /// \brief Print the diagonstic level to a raw_ostream.
  ///
  /// This is a static helper that handles colorizing the level and formatting
  /// it into an arbitrary output stream. This is used internally by the
  /// TextDiagnostic emission code, but it can also be used directly by
  /// consumers that don't have a source manager or other state that the full
  /// TextDiagnostic logic requires.
  static void printDiagnosticLevel(raw_ostream &OS,
                                   DiagnosticsEngine::Level Level,
                                   bool ShowColors);

  /// \brief Pretty-print a diagnostic message to a raw_ostream.
  ///
  /// This is a static helper to handle the line wrapping, colorizing, and
  /// rendering of a diagnostic message to a particular ostream. It is
  /// publically visible so that clients which do not have sufficient state to
  /// build a complete TextDiagnostic object can still get consistent
  /// formatting of their diagnostic messages.
  ///
  /// \param OS Where the message is printed
  /// \param Level Used to colorizing the message
  /// \param Message The text actually printed
  /// \param CurrentColumn The starting column of the first line, accounting
  ///                      for any prefix.
  /// \param Columns The number of columns to use in line-wrapping, 0 disables
  ///                all line-wrapping.
  /// \param ShowColors Enable colorizing of the message.
  static void printDiagnosticMessage(raw_ostream &OS,
                                     DiagnosticsEngine::Level Level,
                                     StringRef Message,
                                     unsigned CurrentColumn, unsigned Columns,
                                     bool ShowColors);

private:
  void emitIncludeStack(SourceLocation Loc, DiagnosticsEngine::Level Level);
  void emitIncludeStackRecursively(SourceLocation Loc);
  void emitDiagnosticLoc(SourceLocation Loc, PresumedLoc PLoc,
                         DiagnosticsEngine::Level Level,
                         ArrayRef<CharSourceRange> Ranges);
  void emitMacroExpansionsAndCarets(SourceLocation Loc,
                                    DiagnosticsEngine::Level Level,
                                    SmallVectorImpl<CharSourceRange>& Ranges,
                                    ArrayRef<FixItHint> Hints,
                                    unsigned &MacroDepth,
                                    unsigned OnMacroInst = 0);
  void emitSnippetAndCaret(SourceLocation Loc, DiagnosticsEngine::Level Level,
                           SmallVectorImpl<CharSourceRange>& Ranges,
                           ArrayRef<FixItHint> Hints);

  void highlightRange(const CharSourceRange &R,
                      unsigned LineNo, FileID FID,
                      const std::string &SourceLine,
                      std::string &CaretLine);
  std::string buildFixItInsertionLine(unsigned LineNo,
                                      const char *LineStart,
                                      const char *LineEnd,
                                      ArrayRef<FixItHint> Hints);
  void expandTabs(std::string &SourceLine, std::string &CaretLine);
  void emitParseableFixits(ArrayRef<FixItHint> Hints);
};

} // end namespace clang

#endif
