//===--- TextDiagnosticPrinter.h - Text Diagnostic Client -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Bill Wendling and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
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

namespace llvm {
  namespace clang {
    class SourceManager;

    class TextDiagnosticPrinter : public TextDiagnostics {
      SourceLocation LastWarningLoc;
    public:
      TextDiagnosticPrinter(SourceManager &sourceMgr)
        : TextDiagnostics(sourceMgr) {}

      void PrintIncludeStack(SourceLocation Pos);
      void HighlightRange(const SourceRange &R, unsigned LineNo,
                          std::string &CaratLine,
                          const std::string &SourceLine);
      unsigned GetTokenLength(SourceLocation Loc);

      virtual void HandleDiagnostic(Diagnostic::Level DiagLevel,
                                    SourceLocation Pos,
                                    diag::kind ID, const std::string *Strs,
                                    unsigned NumStrs,
                                    const SourceRange *Ranges, 
                                    unsigned NumRanges);
    };

  } // end namspace clang
} // end namespace llvm

#endif
