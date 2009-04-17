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

namespace llvm {
  class raw_ostream;
}

namespace clang {
class SourceManager;
class LangOptions;

class TextDiagnosticPrinter : public DiagnosticClient {
  llvm::raw_ostream &OS;
  const LangOptions *LangOpts;
  SourceLocation LastWarningLoc;
  FullSourceLoc LastLoc;
  bool LastCaretDiagnosticWasNote;

  bool ShowColumn;
  bool CaretDiagnostics;
  bool ShowLocation;
  bool PrintRangeInfo;
  bool PrintDiagnosticOption;
public:
  TextDiagnosticPrinter(llvm::raw_ostream &os,
                        bool showColumn = true,
                        bool caretDiagnistics = true, bool showLocation = true,
                        bool printRangeInfo = true,
                        bool printDiagnosticOption = true)
    : OS(os), LangOpts(0),
      LastCaretDiagnosticWasNote(false), ShowColumn(showColumn), 
      CaretDiagnostics(caretDiagnistics), ShowLocation(showLocation),
      PrintRangeInfo(printRangeInfo),
      PrintDiagnosticOption(printDiagnosticOption) {}

  void setLangOptions(const LangOptions *LO) {
    LangOpts = LO;
  }
  
  void PrintIncludeStack(SourceLocation Loc, const SourceManager &SM);

  void HighlightRange(const SourceRange &R,
                      const SourceManager &SrcMgr,
                      unsigned LineNo, FileID FID,
                      std::string &CaretLine,
                      const std::string &SourceLine);

  void EmitCaretDiagnostic(SourceLocation Loc, 
                           SourceRange *Ranges, unsigned NumRanges,
                           SourceManager &SM,
                           const CodeModificationHint *Hints = 0,
                           unsigned NumHints = 0);
  
  virtual void HandleDiagnostic(Diagnostic::Level DiagLevel,
                                const DiagnosticInfo &Info);
};

} // end namspace clang

#endif
