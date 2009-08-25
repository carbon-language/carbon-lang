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

void AnalysisManager::DisplayFunction() {
      
  if (DisplayedFunction)
    return;
  
  DisplayedFunction = true;
  
  // FIXME: Is getCodeDecl() always a named decl?
  if (isa<FunctionDecl>(getCodeDecl()) ||
      isa<ObjCMethodDecl>(getCodeDecl())) {
    const NamedDecl *ND = cast<NamedDecl>(getCodeDecl());
    SourceManager &SM = getASTContext().getSourceManager();
    llvm::errs() << "ANALYZE: "
                 << SM.getPresumedLoc(ND->getLocation()).getFilename()
                 << ' ' << ND->getNameAsString() << '\n';
  }
}
