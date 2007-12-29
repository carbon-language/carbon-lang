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

#include "TextDiagnostics.h"
#include "clang/Basic/SourceLocation.h"

namespace clang {
class SourceManager;

class TextDiagnosticPrinter : public TextDiagnostics {
  FullSourceLoc LastWarningLoc;
public:
  TextDiagnosticPrinter() {}

  void PrintIncludeStack(FullSourceLoc Pos);

  void HighlightRange(const SourceRange &R,
                      SourceManager& SrcMgr,
                      unsigned LineNo,
                      std::string &CaratLine,
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
