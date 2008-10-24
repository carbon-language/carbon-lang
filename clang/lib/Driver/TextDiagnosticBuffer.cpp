//===--- TextDiagnosticBuffer.cpp - Buffer Text Diagnostics ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is a concrete diagnostic client, which buffers the diagnostic messages.
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/TextDiagnosticBuffer.h"
using namespace clang;

/// HandleDiagnostic - Store the errors, warnings, and notes that are
/// reported.
/// 
void TextDiagnosticBuffer::HandleDiagnostic(Diagnostic &Diags,
                                            Diagnostic::Level Level,
                                            FullSourceLoc Pos,
                                            diag::kind ID,
                                            const std::string *Strs,
                                            unsigned NumStrs,
                                            const SourceRange *,
                                            unsigned) {
  switch (Level) {
  default: assert(0 && "Diagnostic not handled during diagnostic buffering!");
  case Diagnostic::Note:
    Notes.push_back(std::make_pair(Pos.getLocation(),
                                   FormatDiagnostic(Diags, Level, ID, 
                                                    Strs, NumStrs)));
    break;
  case Diagnostic::Warning:
    Warnings.push_back(std::make_pair(Pos.getLocation(),
                                      FormatDiagnostic(Diags, Level, ID, 
                                                       Strs, NumStrs)));
    break;
  case Diagnostic::Error:
    Errors.push_back(std::make_pair(Pos.getLocation(),
                                    FormatDiagnostic(Diags, Level, ID,
                                                     Strs, NumStrs)));
    break;
  }
}
