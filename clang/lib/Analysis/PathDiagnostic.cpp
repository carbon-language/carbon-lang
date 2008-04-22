//===--- PathDiagnostic.cpp - Path-Specific Diagnostic Handling -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the PathDiagnostic-related interfaces.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/PathDiagnostic.h"
#include <sstream>

using namespace clang;
  
PathDiagnostic::~PathDiagnostic() {
  for (iterator I = begin(), E = end(); I != E; ++I) delete &*I;
}

void PathDiagnosticClient::HandleDiagnostic(Diagnostic &Diags, 
                                            Diagnostic::Level DiagLevel,
                                            FullSourceLoc Pos,
                                            diag::kind ID,
                                            const std::string *Strs,
                                            unsigned NumStrs,
                                            const SourceRange *Ranges, 
                                            unsigned NumRanges) {
  
  // Create a PathDiagnostic with a single piece.
  
  PathDiagnostic* D = new PathDiagnostic();
  
  // Ripped from TextDiagnostics::FormatDiagnostic.  Perhaps we should
  // centralize it somewhere?
  
  std::ostringstream os;
  
  switch (DiagLevel) {
    default: assert(0 && "Unknown diagnostic type!");
    case Diagnostic::Note:    os << "note: "; break;
    case Diagnostic::Warning: os << "warning: "; break;
    case Diagnostic::Error:   os << "error: "; break;
    case Diagnostic::Fatal:   os << "fatal error: "; break;
      break;
  }
  
  std::string Msg = Diags.getDescription(ID);

  for (unsigned i = 0; i < Msg.size() - 1; ++i) {
    if (Msg[i] == '%' && isdigit(Msg[i + 1])) {
      unsigned StrNo = Msg[i + 1] - '0';
      Msg = std::string(Msg.begin(), Msg.begin() + i) +
      (StrNo < NumStrs ? Strs[StrNo] : "<<<INTERNAL ERROR>>>") +
      std::string(Msg.begin() + i + 2, Msg.end());
    }
  }
  
  os << Msg;
  
  PathDiagnosticPiece* P = new PathDiagnosticPiece(Pos, os.str());
  
  while (NumRanges) {
    P->addRange(*Ranges);
    --NumRanges;
    ++Ranges;
  }
  
  D->push_front(P);

  HandlePathDiagnostic(D);  
}
