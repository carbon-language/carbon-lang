//===--- TargetInfo.cpp - Information about Target machine ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the TargetInfo and TargetInfoImpl interfaces.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/Diagnostic.h"
using namespace llvm;
using namespace clang;

/// DiagnoseNonPortability - When a use of a non-portable target feature is
/// used, this method emits the diagnostic and marks the translation unit as
/// non-portable.
void TargetInfo::DiagnoseNonPortability(SourceLocation Loc, unsigned DiagKind) {
  NonPortable = true;
  if (Diag) Diag->Report(Loc, DiagKind);
}


/// ComputeWCharWidth - Determine the width of the wchar_t type for the primary
/// target, diagnosing whether this is non-portable across the secondary
/// targets.
void TargetInfo::ComputeWCharWidth(SourceLocation Loc) {
  WCharWidth = PrimaryTarget->getWCharWidth();
  
  // Check whether this is portable across the secondary targets if the T-U is
  // portable so far.
  for (unsigned i = 0, e = SecondaryTargets.size(); i != e; ++i)
    if (SecondaryTargets[i]->getWCharWidth() != WCharWidth)
      return DiagnoseNonPortability(Loc, diag::port_wchar_t);
}

