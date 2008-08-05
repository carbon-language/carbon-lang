//===--- TextDiagnostics.h - Text Diagnostics Checkers ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is the parent class for all text diagnostics.
//
//===----------------------------------------------------------------------===//

#ifndef TEXT_DIAGNOSTICS_H_
#define TEXT_DIAGNOSTICS_H_

#include "clang/Basic/Diagnostic.h"

namespace clang {
class SourceManager;
class HeaderSearch;
class Preprocessor;

class TextDiagnostics : public DiagnosticClient {
  HeaderSearch *TheHeaderSearch;
protected:
  std::string FormatDiagnostic(Diagnostic &Diags, Diagnostic::Level Level,
                               diag::kind ID,
                               const std::string *Strs,
                               unsigned NumStrs);
public:
  TextDiagnostics() {}
  virtual ~TextDiagnostics();

  void setHeaderSearch(HeaderSearch &HS) { TheHeaderSearch = &HS; }

  virtual bool isInSystemHeader(FullSourceLoc Pos) const;

  virtual void HandleDiagnostic(Diagnostic &Diags, Diagnostic::Level DiagLevel,
                                FullSourceLoc Pos,
                                diag::kind ID,
                                const std::string *Strs,
                                unsigned NumStrs,
                                const SourceRange *Ranges, 
                                unsigned NumRanges) = 0;
};

} // end namspace clang

#endif
