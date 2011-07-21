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
#include "clang/AST/Expr.h"
#include "clang/Analysis/CFG.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Allocator.h"

namespace clang {

class Decl;
class Stmt;
class CFGReverseBlockReachabilityAnalysis;
class CFGStmtMap;
class LiveVariables;
class ParentMap;
class PseudoConstantAnalysis;
class ImplicitParamDecl;
class LocationContextManager;
class StackFrameContext;

namespace idx { class TranslationUnit; }

/// AnalysisContext contains the context data for the function or method under
/// analysis.
class AnalysisContext {
  const Decl *D;

  // TranslationUnit is NULL if we don't have multiple translation units.
  idx::TranslationUnit *TU;

  llvm::OwningPtr<CFG> cfg, completeCFG;
  llvm::OwningPtr<CFGStmtMap> cfgStmtMap;

  CFG::BuildOptions cfgBuildOptions;
  CFG::BuildOptions::ForcedBlkExprs *forcedBlkExprs;
  
  bool builtCFG, builtCompleteCFG;

  llvm::OwningPtr<LiveVariables> liveness;
  llvm::OwningPtr<LiveVariables> relaxedLiveness;
  llvm::OwningPtr<ParentMap> PM;
  llvm::OwningPtr<PseudoConstantAnalysis> PCA;
  llvm::OwningPtr<CFGReverseBlockReachabilityAnalysis> CFA;

  llvm::BumpPtrAllocator A;

  // FIXME: remove.
  llvm::DenseMap<const BlockDecl*,void*> *ReferencedBlockVars;

public:
  AnalysisContext(const Decl *d, idx::TranslationUnit *tu);

  AnalysisContext(const Decl *d, idx::TranslationUnit *tu,
                  const CFG::BuildOptions &buildOptions);

  ~AnalysisContext();

  ASTContext &getASTContext() { return D->getASTContext(); }
  const Decl *getDecl() const { return D; }

  idx::TranslationUnit *getTranslationUnit() const { return TU; }

  /// Return the build options used to construct the CFG.
  CFG::BuildOptions &getCFGBuildOptions() {
    return cfgBuildOptions;
  }

  const CFG::BuildOptions &getCFGBuildOptions() const {
    return cfgBuildOptions;
  }
  
  /// getAddEHEdges - Return true iff we are adding exceptional edges from
  /// callExprs.  If this is false, then try/catch statements and blocks
  /// reachable from them can appear to be dead in the CFG, analysis passes must
  /// cope with that.
  bool getAddEHEdges() const { return cfgBuildOptions.AddEHEdges; }  
  bool getUseUnoptimizedCFG() const {
      return !cfgBuildOptions.PruneTriviallyFalseEdges;
  }
  bool getAddImplicitDtors() const { return cfgBuildOptions.AddImplicitDtors; }
  bool getAddInitializers() const { return cfgBuildOptions.AddInitializers; }

  void registerForcedBlockExpression(const Stmt *stmt);
  const CFGBlock *getBlockForRegisteredExpression(const Stmt *stmt);
  
  Stmt *getBody();
  CFG *getCFG();
  
  CFGStmtMap *getCFGStmtMap();

  CFGReverseBlockReachabilityAnalysis *getCFGReachablityAnalysis();
  
  /// Return a version of the CFG without any edges pruned.
  CFG *getUnoptimizedCFG();

  void dumpCFG();

  /// \brief Returns true if we have built a CFG for this analysis context.
  /// Note that this doesn't correspond to whether or not a valid CFG exists, it
  /// corresponds to whether we *attempted* to build one.
  bool isCFGBuilt() const { return builtCFG; }

  ParentMap &getParentMap();
  PseudoConstantAnalysis *getPseudoConstantAnalysis();
  LiveVariables *getLiveVariables();
  LiveVariables *getRelaxedLiveVariables();

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
  CFG::BuildOptions cfgBuildOptions;
public:
  AnalysisContextManager(bool useUnoptimizedCFG = false,
                         bool addImplicitDtors = false,
                         bool addInitializers = false);
  
  ~AnalysisContextManager();

  AnalysisContext *getContext(const Decl *D, idx::TranslationUnit *TU = 0);

  bool getUseUnoptimizedCFG() const {
    return !cfgBuildOptions.PruneTriviallyFalseEdges;
  }
  
  CFG::BuildOptions &getCFGBuildOptions() {
    return cfgBuildOptions;
  }

  /// Discard all previously created AnalysisContexts.
  void clear();
};

class LocationContext : public llvm::FoldingSetNode {
public:
  enum ContextKind { StackFrame, Scope, Block };

private:
  ContextKind Kind;

  // AnalysisContext can't be const since some methods may modify its member.
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

  idx::TranslationUnit *getTranslationUnit() const { 
    return Ctx->getTranslationUnit(); 
  }

  const LocationContext *getParent() const { return Parent; }

  bool isParentOf(const LocationContext *LC) const;

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
  // The callsite where this stack frame is established.
  const Stmt *CallSite;

  // The parent block of the callsite.
  const CFGBlock *Block;

  // The index of the callsite in the CFGBlock.
  unsigned Index;

  friend class LocationContextManager;
  StackFrameContext(AnalysisContext *ctx, const LocationContext *parent,
                    const Stmt *s, const CFGBlock *blk, 
                    unsigned idx)
    : LocationContext(StackFrame, ctx, parent), CallSite(s),
      Block(blk), Index(idx) {}

public:
  ~StackFrameContext() {}

  const Stmt *getCallSite() const { return CallSite; }

  const CFGBlock *getCallSiteBlock() const { return Block; }

  unsigned getIndex() const { return Index; }

  void Profile(llvm::FoldingSetNodeID &ID);

  static void Profile(llvm::FoldingSetNodeID &ID, AnalysisContext *ctx,
                      const LocationContext *parent, const Stmt *s,
                      const CFGBlock *blk, unsigned idx) {
    ProfileCommon(ID, StackFrame, ctx, parent, s);
    ID.AddPointer(blk);
    ID.AddInteger(idx);
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
  // FIXME: Add back context-sensivity (we don't want libAnalysis to know
  //  about MemRegion).
  const BlockDecl *BD;

  friend class LocationContextManager;

  BlockInvocationContext(AnalysisContext *ctx, const LocationContext *parent,
                         const BlockDecl *bd)
    : LocationContext(Block, ctx, parent), BD(bd) {}

public:
  ~BlockInvocationContext() {}

  const BlockDecl *getBlockDecl() const { return BD; }

  void Profile(llvm::FoldingSetNodeID &ID);

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
                                         const Stmt *s,
                                         const CFGBlock *blk, unsigned idx);

  const ScopeContext *getScope(AnalysisContext *ctx,
                               const LocationContext *parent,
                               const Stmt *s);

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
