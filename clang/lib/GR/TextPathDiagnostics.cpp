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

#include "clang/GR/PathDiagnosticClients.h"
#include "clang/GR/BugReporter/PathDiagnostic.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/Support/raw_ostream.h"
using namespace clang;
using namespace GR;
using namespace llvm;

namespace {

/// \brief Simple path diagnostic client used for outputting as diagnostic notes
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

  PathGenerationScheme getGenerationScheme() const { return Minimal; }
  bool supportsLogicalOpControlFlow() const { return true; }
  bool supportsAllBlockEdges() const { return true; }
  virtual bool useVerboseDescription() const { return true; }
};

} // end anonymous namespace

PathDiagnosticClient*
GR::createTextPathDiagnosticClient(const std::string& out,
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

  for (PathDiagnostic::const_iterator I=D->begin(), E=D->end(); I != E; ++I) {
    unsigned diagID = Diag.getDiagnosticIDs()->getCustomDiagID(
                                           DiagnosticIDs::Note, I->getString());
    Diag.Report(I->getLocation().asLocation(), diagID);
  }
}
