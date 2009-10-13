//=== AnalysisContext.h - Analysis context for Path Sens analysis --*- C++ -*-//
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

#ifndef LLVM_CLANG_ANALYSIS_ANALYSISCONTEXT_H
#define LLVM_CLANG_ANALYSIS_ANALYSISCONTEXT_H

#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/DenseMap.h"

namespace clang {

class Decl;
class Stmt;
class CFG;
class LiveVariables;
class ParentMap;
class ImplicitParamDecl;

/// AnalysisContext contains the context data for the function or method under
/// analysis.
class AnalysisContext {
  const Decl *D;

  // AnalysisContext owns the following data.
  CFG *cfg;
  LiveVariables *liveness;
  ParentMap *PM;

public:
  AnalysisContext(const Decl *d) : D(d), cfg(0), liveness(0), PM(0) {}
  ~AnalysisContext();

  const Decl *getDecl() { return D; }
  Stmt *getBody();
  CFG *getCFG();
  ParentMap &getParentMap();
  LiveVariables *getLiveVariables();

  /// Return the ImplicitParamDecl* associated with 'self' if this
  /// AnalysisContext wraps an ObjCMethodDecl.  Returns NULL otherwise.
  const ImplicitParamDecl *getSelfDecl() const;
};

class AnalysisContextManager {
  typedef llvm::DenseMap<const Decl*, AnalysisContext*> ContextMap;
  ContextMap Contexts;
public:
  ~AnalysisContextManager();

  AnalysisContext *getContext(const Decl *D);
};

class LocationContext : public llvm::FoldingSetNode {
public:
  enum ContextKind { StackFrame, Scope };

private:
  ContextKind Kind;
  AnalysisContext *Ctx;
  const LocationContext *Parent;

protected:
  LocationContext(ContextKind k, AnalysisContext *ctx,
                  const LocationContext *parent)
    : Kind(k), Ctx(ctx), Parent(parent) {}

public:
  ContextKind getKind() const { return Kind; }

  AnalysisContext *getAnalysisContext() const { return Ctx; }

  const LocationContext *getParent() const { return Parent; }

  const Decl *getDecl() const { return getAnalysisContext()->getDecl(); }

  CFG *getCFG() const { return getAnalysisContext()->getCFG(); }

  LiveVariables *getLiveVariables() const {
    return getAnalysisContext()->getLiveVariables();
  }

  ParentMap &getParentMap() const { 
    return getAnalysisContext()->getParentMap();
  }

  const ImplicitParamDecl *getSelfDecl() const {
    return Ctx->getSelfDecl();
  }

  void Profile(llvm::FoldingSetNodeID &ID) {
    Profile(ID, Kind, Ctx, Parent);
  }

  static void Profile(llvm::FoldingSetNodeID &ID, ContextKind k,
                      AnalysisContext *ctx, const LocationContext *parent);

  static bool classof(const LocationContext*) { return true; }
};

class StackFrameContext : public LocationContext {
  const Stmt *CallSite;

public:
  StackFrameContext(AnalysisContext *ctx, const LocationContext *parent,
                    const Stmt *s)
    : LocationContext(StackFrame, ctx, parent), CallSite(s) {}

  Stmt const *getCallSite() const { return CallSite; }

  void Profile(llvm::FoldingSetNodeID &ID) {
    Profile(ID, getAnalysisContext(), getParent(), CallSite);
  }

  static void Profile(llvm::FoldingSetNodeID &ID, AnalysisContext *ctx,
                      const LocationContext *parent, const Stmt *s);

  static bool classof(const LocationContext* Ctx) {
    return Ctx->getKind() == StackFrame;
  }
};

class ScopeContext : public LocationContext {
  const Stmt *Enter;

public:
  ScopeContext(AnalysisContext *ctx, const LocationContext *parent,
               const Stmt *s)
    : LocationContext(Scope, ctx, parent), Enter(s) {}

  void Profile(llvm::FoldingSetNodeID &ID) {
    Profile(ID, getAnalysisContext(), getParent(), Enter);
  }

  static void Profile(llvm::FoldingSetNodeID &ID, AnalysisContext *ctx,
                      const LocationContext *parent, const Stmt *s);

  static bool classof(const LocationContext* Ctx) {
    return Ctx->getKind() == Scope;
  }
};

class LocationContextManager {
  llvm::FoldingSet<LocationContext> Contexts;

public:
  StackFrameContext *getStackFrame(AnalysisContext *ctx,
                                   const LocationContext *parent,
                                   const Stmt *s);

  ScopeContext *getScope(AnalysisContext *ctx, const LocationContext *parent,
                         const Stmt *s);
};

} // end clang namespace
#endif
