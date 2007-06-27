//===--- TextDiagnosticBuffer.cpp - Buffer Text Diagnostics ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Bill Wendling and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is a concrete diagnostic client, which buffers the diagnostic messages.
//
//===----------------------------------------------------------------------===//

#include "TextDiagnosticBuffer.h"
#include "clang/Basic/SourceManager.h"
using namespace clang;

/// HandleDiagnostic - Store the errors & warnings that are reported.
/// 
void TextDiagnosticBuffer::HandleDiagnostic(Diagnostic::Level Level,
                                            SourceLocation Pos,
                                            diag::kind ID,
                                            const std::string *Strs,
                                            unsigned NumStrs,
                                            const SourceRange *,
                                            unsigned) {
  switch (Level) {
  default: assert(0 && "Diagnostic not handled during diagnostic buffering!");
  case Diagnostic::Warning:
    Warnings.push_back(std::make_pair(Pos, FormatDiagnostic(Level, ID, Strs,
                                                            NumStrs)));
    break;
  case Diagnostic::Error:
    Errors.push_back(std::make_pair(Pos, FormatDiagnostic(Level, ID, Strs,
                                                          NumStrs)));
    break;
  }
}
