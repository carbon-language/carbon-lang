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

#include "clang/AST/Decl.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Allocator.h"

namespace clang {

class Decl;
class Stmt;
class CFG;
class LiveVariables;
class ParentMap;
class ImplicitParamDecl;
class LocationContextManager;
class BlockDataRegion;
class StackFrameContext;
  
/// AnalysisContext contains the context data for the function or method under
/// analysis.
class AnalysisContext {
  const Decl *D;

  // AnalysisContext owns the following data.
  CFG *cfg;
  LiveVariables *liveness;
  ParentMap *PM;
  llvm::DenseMap<const BlockDecl*,void*> *ReferencedBlockVars;
  llvm::BumpPtrAllocator A;
public:
  AnalysisContext(const Decl *d) : D(d), cfg(0), liveness(0), PM(0),
    ReferencedBlockVars(0) {}

  ~AnalysisContext();

  ASTContext &getASTContext() { return D->getASTContext(); }
  const Decl *getDecl() { return D; }
  Stmt *getBody();
  CFG *getCFG();
  ParentMap &getParentMap();
  LiveVariables *getLiveVariables();

  typedef const VarDecl * const * referenced_decls_iterator;

  std::pair<referenced_decls_iterator, referenced_decls_iterator>
    getReferencedBlockVars(const BlockDecl *BD);
  
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
  
  // Discard all previously created AnalysisContexts.
  void clear();
};

class LocationContext : public llvm::FoldingSetNode {
public:
  enum ContextKind { StackFrame, Scope, Block };

private:
  ContextKind Kind;
  AnalysisContext *Ctx;
  const LocationContext *Parent;

protected:
  LocationContext(ContextKind k, AnalysisContext *ctx,
                  const LocationContext *parent)
    : Kind(k), Ctx(ctx), Parent(parent) {}

public:
  virtual ~LocationContext();
  
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
  
  const StackFrameContext *getCurrentStackFrame() const;
  const StackFrameContext *
    getStackFrameForDeclContext(const DeclContext *DC) const;

  virtual void Profile(llvm::FoldingSetNodeID &ID) = 0;

  static bool classof(const LocationContext*) { return true; }

public:
  static void ProfileCommon(llvm::FoldingSetNodeID &ID,
                            ContextKind ck,
                            AnalysisContext *ctx,
                            const LocationContext *parent,
                            const void* data);
};

class StackFrameContext : public LocationContext {
  const Stmt *CallSite;

  friend class LocationContextManager;
  StackFrameContext(AnalysisContext *ctx, const LocationContext *parent,
                    const Stmt *s)
    : LocationContext(StackFrame, ctx, parent), CallSite(s) {}

public:
  ~StackFrameContext() {}

  const Stmt *getCallSite() const { return CallSite; }

  void Profile(llvm::FoldingSetNodeID &ID);
  
  static void Profile(llvm::FoldingSetNodeID &ID, AnalysisContext *ctx,
                      const LocationContext *parent, const Stmt *s) {
    ProfileCommon(ID, StackFrame, ctx, parent, s);
  }

  static bool classof(const LocationContext* Ctx) {
    return Ctx->getKind() == StackFrame;
  }
};

class ScopeContext : public LocationContext {
  const Stmt *Enter;
  
  friend class LocationContextManager;
  ScopeContext(AnalysisContext *ctx, const LocationContext *parent,
               const Stmt *s)
    : LocationContext(Scope, ctx, parent), Enter(s) {}

public:
  ~ScopeContext() {}

  void Profile(llvm::FoldingSetNodeID &ID);

  static void Profile(llvm::FoldingSetNodeID &ID, AnalysisContext *ctx,
                      const LocationContext *parent, const Stmt *s) {
    ProfileCommon(ID, Scope, ctx, parent, s);
  }

  static bool classof(const LocationContext* Ctx) {
    return Ctx->getKind() == Scope;
  }
};

class BlockInvocationContext : public LocationContext {
  llvm::PointerUnion<const BlockDataRegion *, const BlockDecl *> Data;

  friend class LocationContextManager;

  BlockInvocationContext(AnalysisContext *ctx, const LocationContext *parent,
                         const BlockDataRegion *br)
    : LocationContext(Block, ctx, parent), Data(br) {}
  
  BlockInvocationContext(AnalysisContext *ctx, const LocationContext *parent,
                         const BlockDecl *bd)
    : LocationContext(Block, ctx, parent), Data(bd) {}

public:
  ~BlockInvocationContext() {}
  
  const BlockDataRegion *getBlockRegion() const {
    return Data.is<const BlockDataRegion*>() ? 
      Data.get<const BlockDataRegion*>() : 0;
  }
  
  const BlockDecl *getBlockDecl() const;
  
  void Profile(llvm::FoldingSetNodeID &ID);
  
  static void Profile(llvm::FoldingSetNodeID &ID, AnalysisContext *ctx,
                      const LocationContext *parent, const BlockDataRegion *br){
    ProfileCommon(ID, Block, ctx, parent, br);
  }
  
  static void Profile(llvm::FoldingSetNodeID &ID, AnalysisContext *ctx,
                      const LocationContext *parent, const BlockDecl *bd) {
    ProfileCommon(ID, Block, ctx, parent, bd);
  }
  
  static bool classof(const LocationContext* Ctx) {
    return Ctx->getKind() == Block;
  }
};

class LocationContextManager {
  llvm::FoldingSet<LocationContext> Contexts;
public:
  ~LocationContextManager();
  
  const StackFrameContext *getStackFrame(AnalysisContext *ctx,
                                         const LocationContext *parent,
                                         const Stmt *s);

  const ScopeContext *getScope(AnalysisContext *ctx,
                               const LocationContext *parent,
                               const Stmt *s);
  
  const BlockInvocationContext *
  getBlockInvocation(AnalysisContext *ctx, const LocationContext *parent,
                     const BlockDataRegion *BR);
  
  /// Discard all previously created LocationContext objects.
  void clear();
private:
  template <typename LOC, typename DATA>
  const LOC *getLocationContext(AnalysisContext *ctx,
                                const LocationContext *parent,
                                const DATA *d);
};

} // end clang namespace
#endif
