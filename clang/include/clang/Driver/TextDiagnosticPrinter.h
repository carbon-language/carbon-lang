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

#ifndef TEXT_DIAGNOSTIC_PRINTER_H_
#define TEXT_DIAGNOSTIC_PRINTER_H_

#include "clang/Driver/TextDiagnostics.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/Support/Streams.h"

namespace clang {
class SourceManager;

class TextDiagnosticPrinter : public TextDiagnostics {
  FullSourceLoc LastWarningLoc;
  FullSourceLoc LastLoc;
  llvm::OStream OS;
  bool ShowColumn;
  bool CaretDiagnostics;
public:
  TextDiagnosticPrinter(bool showColumn = true, bool caretDiagnistics = true,
      llvm::OStream &os = llvm::cerr)
    : OS(os), ShowColumn(showColumn), CaretDiagnostics(caretDiagnistics) {}

  void PrintIncludeStack(FullSourceLoc Pos);

  void HighlightRange(const SourceRange &R,
                      SourceManager& SrcMgr,
                      unsigned LineNo, unsigned FileID,
                      std::string &CaretLine,
                      const std::string &SourceLine);

  virtual void HandleDiagnostic(Diagnostic &Diags,
                                Diagnostic::Level DiagLevel,
                                FullSourceLoc Pos,
                                diag::kind ID,
                                const std::string *Strs,
                                unsigned NumStrs,
                                const SourceRange *Ranges, 
                                unsigned NumRanges);
};

} // end namspace clang

#endif
