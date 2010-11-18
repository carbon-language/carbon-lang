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

#include "clang/Frontend/TextDiagnosticBuffer.h"
#include "llvm/ADT/SmallString.h"
using namespace clang;

/// HandleDiagnostic - Store the errors, warnings, and notes that are
/// reported.
///
void TextDiagnosticBuffer::HandleDiagnostic(Diagnostic::Level Level,
                                            const DiagnosticInfo &Info) {
  // Default implementation (Warnings/errors count).
  DiagnosticClient::HandleDiagnostic(Level, Info);

  llvm::SmallString<100> Buf;
  Info.FormatDiagnostic(Buf);
  switch (Level) {
  default: assert(0 && "Diagnostic not handled during diagnostic buffering!");
  case Diagnostic::Note:
    Notes.push_back(std::make_pair(Info.getLocation(), Buf.str()));
    break;
  case Diagnostic::Warning:
    Warnings.push_back(std::make_pair(Info.getLocation(), Buf.str()));
    break;
  case Diagnostic::Error:
  case Diagnostic::Fatal:
    Errors.push_back(std::make_pair(Info.getLocation(), Buf.str()));
    break;
  }
}

void TextDiagnosticBuffer::FlushDiagnostics(Diagnostic &Diags) const {
  // FIXME: Flush the diagnostics in order.
  for (const_iterator it = err_begin(), ie = err_end(); it != ie; ++it)
    Diags.Report(Diags.getCustomDiagID(Diagnostic::Error, it->second.c_str()));
  for (const_iterator it = warn_begin(), ie = warn_end(); it != ie; ++it)
    Diags.Report(Diags.getCustomDiagID(Diagnostic::Warning,it->second.c_str()));
  for (const_iterator it = note_begin(), ie = note_end(); it != ie; ++it)
    Diags.Report(Diags.getCustomDiagID(Diagnostic::Note, it->second.c_str()));
}
