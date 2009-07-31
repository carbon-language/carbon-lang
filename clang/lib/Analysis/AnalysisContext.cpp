//== AnalysisContext.cpp - Analysis context for Path Sens analysis -*- C++ -*-//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines AnalysisContext, a class that manages the analysis context
// data for path sensitive analysis.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/PathSensitive/AnalysisContext.h"
#include "clang/Analysis/Analyses/LiveVariables.h"
#include "clang/Analysis/CFG.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/ParentMap.h"
#include "llvm/Support/ErrorHandling.h"

using namespace clang;

AnalysisContext::~AnalysisContext() {
  delete cfg;
  delete liveness;
  delete PM;
}

Stmt *AnalysisContext::getBody() {
  if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D))
    return FD->getBody();
  else if (ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(D))
    return MD->getBody();

  llvm::llvm_unreachable("unknown code decl");
}

CFG *AnalysisContext::getCFG() {
  if (!cfg) 
    cfg = CFG::buildCFG(getBody(), &D->getASTContext());
  return cfg;
}

ParentMap &AnalysisContext::getParentMap() {
  if (!PM) 
    PM = new ParentMap(getBody());
  return *PM;
}

LiveVariables *AnalysisContext::getLiveVariables() {
  if (!liveness) {
    CFG *c = getCFG();
    if (!c)
      return 0;
    
    liveness = new LiveVariables(D->getASTContext(), *c);
    liveness->runOnCFG(*c);
    liveness->runOnAllBlocks(*c, 0, true);
  }
  
  return liveness;
}

AnalysisContext *AnalysisContextManager::getContext(Decl *D) {
  iterator I = Contexts.find(D);
  if (I != Contexts.end())
    return &(I->second);

  AnalysisContext &Ctx = Contexts[D];
  Ctx.setDecl(D);
  return &Ctx;
}
