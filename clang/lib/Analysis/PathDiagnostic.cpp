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
  
  PathDiagnostic D(DiagLevel, ID);
  
  PathDiagnosticPiece* P = new PathDiagnosticPiece(Pos);
  
  while (NumStrs) {
    P->addString(*Strs);
    --NumStrs;
    ++Strs;
  }
  
  while (NumRanges) {
    P->addRange(*Ranges);
    --NumRanges;
    ++Ranges;
  }
  
  D.push_front(P);

  HandlePathDiagnostic(Diags, D);  
}
