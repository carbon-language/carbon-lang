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

AnalysisContextManager::~AnalysisContextManager() {
  for (ContextMap::iterator I = Contexts.begin(), E = Contexts.end(); I!=E; ++I)
    delete I->second;
}

Stmt *AnalysisContext::getBody() {
  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D))
    return FD->getBody();
  else if (const ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(D))
    return MD->getBody();

  llvm::llvm_unreachable("unknown code decl");
}

const ImplicitParamDecl *AnalysisContext::getSelfDecl() const {
  if (const ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(D))
    return MD->getSelfDecl();

  return NULL;
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

AnalysisContext *AnalysisContextManager::getContext(const Decl *D) {
  AnalysisContext *&AC = Contexts[D];
  if (!AC)
    AC = new AnalysisContext(D);

  return AC;
}

void LocationContext::Profile(llvm::FoldingSetNodeID &ID, ContextKind k,
                              AnalysisContext *ctx,
                              const LocationContext *parent) {
  ID.AddInteger(k);
  ID.AddPointer(ctx);
  ID.AddPointer(parent);
}

void StackFrameContext::Profile(llvm::FoldingSetNodeID &ID,AnalysisContext *ctx,
                                const LocationContext *parent, const Stmt *s) {
  LocationContext::Profile(ID, StackFrame, ctx, parent);
  ID.AddPointer(s);
}

void ScopeContext::Profile(llvm::FoldingSetNodeID &ID, AnalysisContext *ctx,
                           const LocationContext *parent, const Stmt *s) {
  LocationContext::Profile(ID, Scope, ctx, parent);
  ID.AddPointer(s);
}

StackFrameContext*
LocationContextManager::getStackFrame(AnalysisContext *ctx,
                                      const LocationContext *parent,
                                      const Stmt *s) {
  llvm::FoldingSetNodeID ID;
  StackFrameContext::Profile(ID, ctx, parent, s);
  void *InsertPos;

  StackFrameContext *f =
   cast_or_null<StackFrameContext>(Contexts.FindNodeOrInsertPos(ID, InsertPos));
  if (!f) {
    f = new StackFrameContext(ctx, parent, s);
    Contexts.InsertNode(f, InsertPos);
  }
  return f;
}

ScopeContext *LocationContextManager::getScope(AnalysisContext *ctx,
                                               const LocationContext *parent,
                                               const Stmt *s) {
  llvm::FoldingSetNodeID ID;
  ScopeContext::Profile(ID, ctx, parent, s);
  void *InsertPos;

  ScopeContext *scope =
    cast_or_null<ScopeContext>(Contexts.FindNodeOrInsertPos(ID, InsertPos));

  if (!scope) {
    scope = new ScopeContext(ctx, parent, s);
    Contexts.InsertNode(scope, InsertPos);
  }
  return scope;
}
