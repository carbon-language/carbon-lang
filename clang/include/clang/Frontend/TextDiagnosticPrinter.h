//===--- TextDiagnosticPrinter.h - Text Diagnostic Client -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is a concrete diagnostic client, which prints the diagnostics to
// standard error.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FRONTEND_TEXT_DIAGNOSTIC_PRINTER_H_
#define LLVM_CLANG_FRONTEND_TEXT_DIAGNOSTIC_PRINTER_H_

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceLocation.h"

namespace clang {
class DiagnosticOptions;
class LangOptions;

class TextDiagnosticPrinter : public DiagnosticClient {
  llvm::raw_ostream &OS;
  const LangOptions *LangOpts;
  const DiagnosticOptions *DiagOpts;

  SourceLocation LastWarningLoc;
  FullSourceLoc LastLoc;
  unsigned LastCaretDiagnosticWasNote : 1;
  unsigned OwnsOutputStream : 1;

  /// A string to prefix to error messages.
  std::string Prefix;

public:
  TextDiagnosticPrinter(llvm::raw_ostream &os, const DiagnosticOptions &diags,
                        bool OwnsOutputStream = false);
  virtual ~TextDiagnosticPrinter();

  /// setPrefix - Set the diagnostic printer prefix string, which will be
  /// printed at the start of any diagnostics. If empty, no prefix string is
  /// used.
  void setPrefix(std::string Value) { Prefix = Value; }

  void BeginSourceFile(const LangOptions &LO, const Preprocessor *PP) {
    LangOpts = &LO;
  }

  void EndSourceFile() {
    LangOpts = 0;
  }

  void PrintIncludeStack(Diagnostic::Level Level, SourceLocation Loc,
                         const SourceManager &SM);

  void HighlightRange(const CharSourceRange &R,
                      const SourceManager &SrcMgr,
                      unsigned LineNo, FileID FID,
                      std::string &CaretLine,
                      const std::string &SourceLine);

  virtual void HandleDiagnostic(Diagnostic::Level Level,
                                const DiagnosticInfo &Info);

private:
  void EmitCaretDiagnostic(SourceLocation Loc, CharSourceRange *Ranges,
                           unsigned NumRanges, const SourceManager &SM,
                           const FixItHint *Hints,
                           unsigned NumHints, unsigned Columns,  
                           unsigned OnMacroInst, unsigned MacroSkipStart,
                           unsigned MacroSkipEnd);
  
};

} // end namespace clang

#endif
