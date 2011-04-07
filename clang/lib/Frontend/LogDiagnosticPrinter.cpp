//===--- LogDiagnosticPrinter.cpp - Log Diagnostic Printer ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/LogDiagnosticPrinter.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
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

static llvm::StringRef getLevelName(Diagnostic::Level Level) {
  switch (Level) {
  default:
    return "<unknown>";
  case Diagnostic::Ignored: return "ignored";
  case Diagnostic::Note:    return "note";
  case Diagnostic::Warning: return "warning";
  case Diagnostic::Error:   return "error";
  case Diagnostic::Fatal:   return "fatal error";
  }
}

void LogDiagnosticPrinter::EndSourceFile() {
  // We emit all the diagnostics in EndSourceFile. However, we don't emit any
  // entry if no diagnostics were present.
  //
  // Note that DiagnosticClient has no "end-of-compilation" callback, so we will
  // miss any diagnostics which are emitted after and outside the translation
  // unit processing.
  if (Entries.empty())
    return;

  // Write to a temporary string to ensure atomic write of diagnostic object.
  llvm::SmallString<512> Msg;
  llvm::raw_svector_ostream OS(Msg);

  OS << "{\n";
  // FIXME: Output main translation unit file name.
  // FIXME: Include the invocation, if dwarf-debug-flags is available.
  OS << "  \"diagnostics\" : [\n";
  for (unsigned i = 0, e = Entries.size(); i != e; ++i) {
    DiagEntry &DE = Entries[i];

    OS << "    {\n";
    OS << "      \"filename\" : \"" << DE.Filename << "\",\n";
    OS << "      \"line\" : " << DE.Line << ",\n";
    OS << "      \"column\" : " << DE.Column << ",\n";
    OS << "      \"message\" : \"" << DE.Message << "\",\n";
    OS << "      \"level\" : \"" << getLevelName(DE.DiagnosticLevel) << "\"\n";
    OS << "    }" << ((i + 1 != e) ? "," : "") << '\n';
  }
  OS << "  ]\n";
  OS << "},\n";

  this->OS << OS.str();
}

void LogDiagnosticPrinter::HandleDiagnostic(Diagnostic::Level Level,
                                            const DiagnosticInfo &Info) {
  // Default implementation (Warnings/errors count).
  DiagnosticClient::HandleDiagnostic(Level, Info);

  // Create the diag entry.
  DiagEntry DE;
  DE.DiagnosticID = Info.getID();
  DE.DiagnosticLevel = Level;

  // Format the message.
  llvm::SmallString<100> MessageStr;
  Info.FormatDiagnostic(MessageStr);
  DE.Message = MessageStr.str();

  // Set the location information.
  DE.Filename = "";
  DE.Line = DE.Column = 0;
  if (Info.getLocation().isValid()) {
    const SourceManager &SM = Info.getSourceManager();
    PresumedLoc PLoc = SM.getPresumedLoc(Info.getLocation());

    if (PLoc.isInvalid()) {
      // At least print the file name if available:
      FileID FID = SM.getFileID(Info.getLocation());
      if (!FID.isInvalid()) {
        const FileEntry *FE = SM.getFileEntryForID(FID);
        if (FE && FE->getName())
          DE.Filename = FE->getName();
      }
    } else {
      DE.Filename = PLoc.getFilename();
      DE.Line = PLoc.getLine();
      DE.Column = PLoc.getColumn();
    }
  }

  // Record the diagnostic entry.
  Entries.push_back(DE);
}
