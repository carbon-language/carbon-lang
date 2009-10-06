//== AnalysisManager.cpp - Path sensitive analysis data manager ----*- C++ -*-//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the AnalysisManager class.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/PathSensitive/AnalysisManager.h"
#include "clang/Basic/SourceManager.h"

using namespace clang;

void AnalysisManager::DisplayFunction(Decl *D) {

  if (DisplayedFunction)
    return;

  DisplayedFunction = true;

  // FIXME: Is getCodeDecl() always a named decl?
  if (isa<FunctionDecl>(D) || isa<ObjCMethodDecl>(D)) {
    const NamedDecl *ND = cast<NamedDecl>(D);
    SourceManager &SM = getASTContext().getSourceManager();
    (llvm::errs() << "ANALYZE: "
                  << SM.getPresumedLoc(ND->getLocation()).getFilename()
                  << ' ' << ND->getNameAsString() << '\n').flush();
  }
}

