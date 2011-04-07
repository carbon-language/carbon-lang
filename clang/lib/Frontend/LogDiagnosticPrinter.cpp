//===--- LogDiagnosticPrinter.cpp - Log Diagnostic Printer ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/LogDiagnosticPrinter.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"
using namespace clang;

LogDiagnosticPrinter::LogDiagnosticPrinter(llvm::raw_ostream &os,
                                           const DiagnosticOptions &diags,
                                           bool _OwnsOutputStream)
  : OS(os), LangOpts(0), DiagOpts(&diags),
    OwnsOutputStream(_OwnsOutputStream) {
}

LogDiagnosticPrinter::~LogDiagnosticPrinter() {
  if (OwnsOutputStream)
    delete &OS;
}

void LogDiagnosticPrinter::HandleDiagnostic(Diagnostic::Level Level,
                                             const DiagnosticInfo &Info) {
  // Default implementation (Warnings/errors count).
  DiagnosticClient::HandleDiagnostic(Level, Info);

  // Write to a temporary string to ensure atomic write of diagnostic object.
  llvm::SmallString<512> Msg;
  llvm::raw_svector_ostream OS(Msg);

  OS << "hello!\n";

  this->OS << OS.str();
}
