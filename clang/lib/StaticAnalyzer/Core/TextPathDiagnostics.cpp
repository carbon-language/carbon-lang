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

#include "clang/StaticAnalyzer/Core/PathDiagnosticConsumers.h"
#include "clang/StaticAnalyzer/Core/BugReporter/PathDiagnostic.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/Support/raw_ostream.h"
using namespace clang;
using namespace ento;
using namespace llvm;

namespace {

/// \brief Simple path diagnostic client used for outputting as diagnostic notes
/// the sequence of events.
class TextPathDiagnostics : public PathDiagnosticConsumer {
  const std::string OutputFile;
  DiagnosticsEngine &Diag;

public:
  TextPathDiagnostics(const std::string& output, DiagnosticsEngine &diag)
    : OutputFile(output), Diag(diag) {}

  void FlushDiagnosticsImpl(std::vector<const PathDiagnostic *> &Diags,
                            SmallVectorImpl<std::string> *FilesMade);
  
  virtual StringRef getName() const {
    return "TextPathDiagnostics";
  }

  PathGenerationScheme getGenerationScheme() const { return Minimal; }
  bool supportsLogicalOpControlFlow() const { return true; }
  bool supportsAllBlockEdges() const { return true; }
  virtual bool useVerboseDescription() const { return true; }
};

} // end anonymous namespace

PathDiagnosticConsumer*
ento::createTextPathDiagnosticConsumer(const std::string& out,
                                     const Preprocessor &PP) {
  return new TextPathDiagnostics(out, PP.getDiagnostics());
}

void TextPathDiagnostics::FlushDiagnosticsImpl(
                              std::vector<const PathDiagnostic *> &Diags,
                              SmallVectorImpl<std::string> *FilesMade) {
  for (std::vector<const PathDiagnostic *>::iterator it = Diags.begin(),
       et = Diags.end(); it != et; ++it) {
    const PathDiagnostic *D = *it;
    for (PathPieces::const_iterator I = D->path.begin(), E = D->path.end(); 
         I != E; ++I) {
      unsigned diagID =
        Diag.getDiagnosticIDs()->getCustomDiagID(DiagnosticIDs::Note,
                                                 (*I)->getString());
      Diag.Report((*I)->getLocation().asLocation(), diagID);
    }
  }
}
