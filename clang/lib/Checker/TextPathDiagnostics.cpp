//===--- TextPathDiagnostics.cpp - Text Diagnostics for Paths ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the TextPathDiagnostics object.
//
//===----------------------------------------------------------------------===//

#include "clang/Checker/PathDiagnosticClients.h"
#include "clang/Checker/BugReporter/PathDiagnostic.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/Support/raw_ostream.h"
using namespace clang;
using namespace llvm;

namespace {

/// \brief Simple path diagnostic client used for outputting as text
/// the sequence of events.
class TextPathDiagnostics : public PathDiagnosticClient {
  const std::string OutputFile;
  Diagnostic &Diag;

public:
  TextPathDiagnostics(const std::string& output, Diagnostic &diag)
    : OutputFile(output), Diag(diag) {}

  void HandlePathDiagnostic(const PathDiagnostic* D);

  void FlushDiagnostics(llvm::SmallVectorImpl<std::string> *FilesMade) { }
  
  virtual llvm::StringRef getName() const {
    return "TextPathDiagnostics";
  }

  PathGenerationScheme getGenerationScheme() const { return Extensive; }
  bool supportsLogicalOpControlFlow() const { return true; }
  bool supportsAllBlockEdges() const { return true; }
  virtual bool useVerboseDescription() const { return true; }
};

} // end anonymous namespace

PathDiagnosticClient*
clang::createTextPathDiagnosticClient(const std::string& out,
                                      const Preprocessor &PP) {
  return new TextPathDiagnostics(out, PP.getDiagnostics());
}

void TextPathDiagnostics::HandlePathDiagnostic(const PathDiagnostic* D) {
  if (!D)
    return;

  if (D->empty()) {
    delete D;
    return;
  }

  // Open the file.
  std::string ErrMsg;
  llvm::raw_fd_ostream o(OutputFile.c_str(), ErrMsg);
  if (!ErrMsg.empty()) {
    llvm::errs() << "warning: could not create file: " << OutputFile << '\n';
    return;
  }

  for (PathDiagnostic::const_iterator I=D->begin(), E=D->end(); I != E; ++I) {
    if (isa<PathDiagnosticEventPiece>(*I)) {
      PathDiagnosticEventPiece &event = cast<PathDiagnosticEventPiece>(*I);
      unsigned diagID = Diag.getDiagnosticIDs()->getCustomDiagID(
                                        DiagnosticIDs::Note, event.getString());
      Diag.Report(event.getLocation().asLocation(), diagID);
    }
  }
}
