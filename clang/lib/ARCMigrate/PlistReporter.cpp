//===--- PlistReporter.cpp - ARC Migrate Tool Plist Reporter ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Internals.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/PlistSupport.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
using namespace clang;
using namespace arcmt;
using namespace markup;

void arcmt::writeARCDiagsToPlist(const std::string &outPath,
                                 ArrayRef<StoredDiagnostic> diags,
                                 SourceManager &SM,
                                 const LangOptions &LangOpts) {
  DiagnosticIDs DiagIDs;

  // Build up a set of FIDs that we use by scanning the locations and
  // ranges of the diagnostics.
  FIDMap FM;
  SmallVector<FileID, 10> Fids;

  for (ArrayRef<StoredDiagnostic>::iterator
         I = diags.begin(), E = diags.end(); I != E; ++I) {
    const StoredDiagnostic &D = *I;

    AddFID(FM, Fids, SM, D.getLocation());

    for (StoredDiagnostic::range_iterator
           RI = D.range_begin(), RE = D.range_end(); RI != RE; ++RI) {
      AddFID(FM, Fids, SM, RI->getBegin());
      AddFID(FM, Fids, SM, RI->getEnd());
    }
  }

  std::string errMsg;
  llvm::raw_fd_ostream o(outPath.c_str(), errMsg, llvm::sys::fs::F_None);
  if (!errMsg.empty()) {
    llvm::errs() << "error: could not create file: " << outPath << '\n';
    return;
  }

  // Write the plist header.
  o << PlistHeader;

  // Write the root object: a <dict> containing...
  //  - "files", an <array> mapping from FIDs to file names
  //  - "diagnostics", an <array> containing the diagnostics
  o << "<dict>\n"
       " <key>files</key>\n"
       " <array>\n";

  for (SmallVectorImpl<FileID>::iterator I=Fids.begin(), E=Fids.end();
       I!=E; ++I) {
    o << "  ";
    EmitString(o, SM.getFileEntryForID(*I)->getName()) << '\n';
  }

  o << " </array>\n"
       " <key>diagnostics</key>\n"
       " <array>\n";

  for (ArrayRef<StoredDiagnostic>::iterator
         DI = diags.begin(), DE = diags.end(); DI != DE; ++DI) {
    
    const StoredDiagnostic &D = *DI;

    if (D.getLevel() == DiagnosticsEngine::Ignored)
      continue;

    o << "  <dict>\n";

    // Output the diagnostic.
    o << "   <key>description</key>";
    EmitString(o, D.getMessage()) << '\n';
    o << "   <key>category</key>";
    EmitString(o, DiagIDs.getCategoryNameFromID(
                          DiagIDs.getCategoryNumberForDiag(D.getID()))) << '\n';
    o << "   <key>type</key>";
    if (D.getLevel() >= DiagnosticsEngine::Error)
      EmitString(o, "error") << '\n';
    else if (D.getLevel() == DiagnosticsEngine::Warning)
      EmitString(o, "warning") << '\n';
    else
      EmitString(o, "note") << '\n';

    // Output the location of the bug.
    o << "  <key>location</key>\n";
    EmitLocation(o, SM, LangOpts, D.getLocation(), FM, 2);

    // Output the ranges (if any).
    StoredDiagnostic::range_iterator RI = D.range_begin(), RE = D.range_end();

    if (RI != RE) {
      o << "   <key>ranges</key>\n";
      o << "   <array>\n";
      for (; RI != RE; ++RI)
        EmitRange(o, SM, LangOpts, *RI, FM, 4);
      o << "   </array>\n";
    }

    // Close up the entry.
    o << "  </dict>\n";
  }

  o << " </array>\n";

  // Finish.
  o << "</dict>\n</plist>";
}
