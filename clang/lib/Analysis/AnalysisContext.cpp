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

#include "clang/AST/Decl.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ParentMap.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Analysis/Analyses/LiveVariables.h"
#include "clang/Analysis/Analyses/PseudoConstantAnalysis.h"
#include "clang/Analysis/Analyses/CFGReachabilityAnalysis.h"
#include "clang/Analysis/AnalysisContext.h"
#include "clang/Analysis/CFG.h"
#include "clang/Analysis/CFGStmtMap.h"
#include "clang/Analysis/Support/BumpVector.h"
#include "clang/Analysis/Support/SaveAndRestore.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/ErrorHandling.h"

using namespace clang;

AnalysisContext::AnalysisContext(const Decl *d,
                                 idx::TranslationUnit *tu,
                                 const CFG::BuildOptions &buildOptions)
  : D(d), TU(tu),
    cfgBuildOptions(buildOptions),
    forcedBlkExprs(0),
    builtCFG(false),
    builtCompleteCFG(false),
    ReferencedBlockVars(0)
{  
  cfgBuildOptions.forcedBlkExprs = &forcedBlkExprs;
}

AnalysisContext::AnalysisContext(const Decl *d,
                                 idx::TranslationUnit *tu)
: D(d), TU(tu),
  forcedBlkExprs(0),
  builtCFG(false),
  builtCompleteCFG(false),
  ReferencedBlockVars(0)
{  
  cfgBuildOptions.forcedBlkExprs = &forcedBlkExprs;
}

AnalysisContextManager::AnalysisContextManager(bool useUnoptimizedCFG,
                                               bool addImplicitDtors,
                                               bool addInitializers) {
  cfgBuildOptions.PruneTriviallyFalseEdges = !useUnoptimizedCFG;
  cfgBuildOptions.AddImplicitDtors = addImplicitDtors;
  cfgBuildOptions.AddInitializers = addInitializers;
}

void AnalysisContextManager::clear() {
  for (ContextMap::iterator I = Contexts.begin(), E = Contexts.end(); I!=E; ++I)
    delete I->second;
  Contexts.clear();
}

Stmt *AnalysisContext::getBody() const {
  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D))
    return FD->getBody();
  else if (const ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(D))
    return MD->getBody();
  else if (const BlockDecl *BD = dyn_cast<BlockDecl>(D))
    return BD->getBody();
  else if (const FunctionTemplateDecl *FunTmpl
           = dyn_cast_or_null<FunctionTemplateDecl>(D))
    return FunTmpl->getTemplatedDecl()->getBody();

  llvm_unreachable("unknown code decl");
}

const ImplicitParamDecl *AnalysisContext::getSelfDecl() const {
  if (const ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(D))
    return MD->getSelfDecl();

  return NULL;
}

void AnalysisContext::registerForcedBlockExpression(const Stmt *stmt) {
  if (!forcedBlkExprs)
    forcedBlkExprs = new CFG::BuildOptions::ForcedBlkExprs();
  // Default construct an entry for 'stmt'.
  if (const Expr *e = dyn_cast<Expr>(stmt))
    stmt = e->IgnoreParens();
  (void) (*forcedBlkExprs)[stmt];
}

const CFGBlock *
AnalysisContext::getBlockForRegisteredExpression(const Stmt *stmt) {
  assert(forcedBlkExprs);
  if (const Expr *e = dyn_cast<Expr>(stmt))
    stmt = e->IgnoreParens();
  CFG::BuildOptions::ForcedBlkExprs::const_iterator itr = 
    forcedBlkExprs->find(stmt);
  assert(itr != forcedBlkExprs->end());
  return itr->second;
}

CFG *AnalysisContext::getCFG() {
  if (!cfgBuildOptions.PruneTriviallyFalseEdges)
    return getUnoptimizedCFG();

  if (!builtCFG) {
    cfg.reset(CFG::buildCFG(D, getBody(),
                            &D->getASTContext(), cfgBuildOptions));
    // Even when the cfg is not successfully built, we don't
    // want to try building it again.
    builtCFG = true;
  }
  return cfg.get();
}

CFG *AnalysisContext::getUnoptimizedCFG() {
  if (!builtCompleteCFG) {
    SaveAndRestore<bool> NotPrune(cfgBuildOptions.PruneTriviallyFalseEdges,
                                  false);
    completeCFG.reset(CFG::buildCFG(D, getBody(), &D->getASTContext(),
                                    cfgBuildOptions));
    // Even when the cfg is not successfully built, we don't
    // want to try building it again.
    builtCompleteCFG = true;
  }
  return completeCFG.get();
}

CFGStmtMap *AnalysisContext::getCFGStmtMap() {
  if (cfgStmtMap)
    return cfgStmtMap.get();
  
  if (CFG *c = getCFG()) {
    cfgStmtMap.reset(CFGStmtMap::Build(c, &getParentMap()));
    return cfgStmtMap.get();
  }
    
  return 0;
}

CFGReverseBlockReachabilityAnalysis *AnalysisContext::getCFGReachablityAnalysis() {
  if (CFA)
    return CFA.get();
  
  if (CFG *c = getCFG()) {
    CFA.reset(new CFGReverseBlockReachabilityAnalysis(*c));
    return CFA.get();
  }
  
  return 0;
}

void AnalysisContext::dumpCFG() {
    getCFG()->dump(getASTContext().getLangOptions());
}

ParentMap &AnalysisContext::getParentMap() {
  if (!PM)
    PM.reset(new ParentMap(getBody()));
  return *PM;
}

PseudoConstantAnalysis *AnalysisContext::getPseudoConstantAnalysis() {
  if (!PCA)
    PCA.reset(new PseudoConstantAnalysis(getBody()));
  return PCA.get();
}

LiveVariables *AnalysisContext::getLiveVariables() {
  if (!liveness)
    liveness.reset(LiveVariables::computeLiveness(*this));
  return liveness.get();
}

LiveVariables *AnalysisContext::getRelaxedLiveVariables() {
  if (!relaxedLiveness)
    relaxedLiveness.reset(LiveVariables::computeLiveness(*this, false));
  return relaxedLiveness.get();
}

AnalysisContext *AnalysisContextManager::getContext(const Decl *D,
                                                    idx::TranslationUnit *TU) {
  AnalysisContext *&AC = Contexts[D];
  if (!AC)
    AC = new AnalysisContext(D, TU, cfgBuildOptions);
  return AC;
}

//===----------------------------------------------------------------------===//
// FoldingSet profiling.
//===----------------------------------------------------------------------===//

void LocationContext::ProfileCommon(llvm::FoldingSetNodeID &ID,
                                    ContextKind ck,
                                    AnalysisContext *ctx,
                                    const LocationContext *parent,
                                    const void *data) {
  ID.AddInteger(ck);
  ID.AddPointer(ctx);
  ID.AddPointer(parent);
  ID.AddPointer(data);
}

void StackFrameContext::Profile(llvm::FoldingSetNodeID &ID) {
  Profile(ID, getAnalysisContext(), getParent(), CallSite, Block, Index);
}

void ScopeContext::Profile(llvm::FoldingSetNodeID &ID) {
  Profile(ID, getAnalysisContext(), getParent(), Enter);
}

void BlockInvocationContext::Profile(llvm::FoldingSetNodeID &ID) {
  Profile(ID, getAnalysisContext(), getParent(), BD);
}

//===----------------------------------------------------------------------===//
// LocationContext creation.
//===----------------------------------------------------------------------===//

template <typename LOC, typename DATA>
const LOC*
LocationContextManager::getLocationContext(AnalysisContext *ctx,
                                           const LocationContext *parent,
                                           const DATA *d) {
  llvm::FoldingSetNodeID ID;
  LOC::Profile(ID, ctx, parent, d);
  void *InsertPos;

  LOC *L = cast_or_null<LOC>(Contexts.FindNodeOrInsertPos(ID, InsertPos));

  if (!L) {
    L = new LOC(ctx, parent, d);
    Contexts.InsertNode(L, InsertPos);
  }
  return L;
}

const StackFrameContext*
LocationContextManager::getStackFrame(AnalysisContext *ctx,
                                      const LocationContext *parent,
                                      const Stmt *s,
                                      const CFGBlock *blk, unsigned idx) {
  llvm::FoldingSetNodeID ID;
  StackFrameContext::Profile(ID, ctx, parent, s, blk, idx);
  void *InsertPos;
  StackFrameContext *L =
   cast_or_null<StackFrameContext>(Contexts.FindNodeOrInsertPos(ID, InsertPos));
  if (!L) {
    L = new StackFrameContext(ctx, parent, s, blk, idx);
    Contexts.InsertNode(L, InsertPos);
  }
  return L;
}

const ScopeContext *
LocationContextManager::getScope(AnalysisContext *ctx,
                                 const LocationContext *parent,
                                 const Stmt *s) {
  return getLocationContext<ScopeContext, Stmt>(ctx, parent, s);
}

//===----------------------------------------------------------------------===//
// LocationContext methods.
//===----------------------------------------------------------------------===//

const StackFrameContext *LocationContext::getCurrentStackFrame() const {
  const LocationContext *LC = this;
  while (LC) {
    if (const StackFrameContext *SFC = dyn_cast<StackFrameContext>(LC))
      return SFC;
    LC = LC->getParent();
  }
  return NULL;
}

const StackFrameContext *
LocationContext::getStackFrameForDeclContext(const DeclContext *DC) const {
  const LocationContext *LC = this;
  while (LC) {
    if (const StackFrameContext *SFC = dyn_cast<StackFrameContext>(LC)) {
      if (cast<DeclContext>(SFC->getDecl()) == DC)
        return SFC;
    }
    LC = LC->getParent();
  }
  return NULL;
}

bool LocationContext::isParentOf(const LocationContext *LC) const {
  do {
    const LocationContext *Parent = LC->getParent();
    if (Parent == this)
      return true;
    else
      LC = Parent;
  } while (LC);

  return false;
}

//===----------------------------------------------------------------------===//
// Lazily generated map to query the external variables referenced by a Block.
//===----------------------------------------------------------------------===//

namespace {
class FindBlockDeclRefExprsVals : public StmtVisitor<FindBlockDeclRefExprsVals>{
  BumpVector<const VarDecl*> &BEVals;
  BumpVectorContext &BC;
  llvm::DenseMap<const VarDecl*, unsigned> Visited;
  llvm::SmallSet<const DeclContext*, 4> IgnoredContexts;
public:
  FindBlockDeclRefExprsVals(BumpVector<const VarDecl*> &bevals,
                            BumpVectorContext &bc)
  : BEVals(bevals), BC(bc) {}

  bool IsTrackedDecl(const VarDecl *VD) {
    const DeclContext *DC = VD->getDeclContext();
    return IgnoredContexts.count(DC) == 0;
  }

  void VisitStmt(Stmt *S) {
    for (Stmt::child_range I = S->children(); I; ++I)
      if (Stmt *child = *I)
        Visit(child);
  }

  void VisitDeclRefExpr(const DeclRefExpr *DR) {
    // Non-local variables are also directly modified.
    if (const VarDecl *VD = dyn_cast<VarDecl>(DR->getDecl()))
      if (!VD->hasLocalStorage()) {
        unsigned &flag = Visited[VD];
        if (!flag) {
          flag = 1;
          BEVals.push_back(VD, BC);
        }
      }
  }

  void VisitBlockDeclRefExpr(BlockDeclRefExpr *DR) {
    if (const VarDecl *VD = dyn_cast<VarDecl>(DR->getDecl())) {
      unsigned &flag = Visited[VD];
      if (!flag) {
        flag = 1;
        if (IsTrackedDecl(VD))
          BEVals.push_back(VD, BC);
      }
    }
  }

  void VisitBlockExpr(BlockExpr *BR) {
    // Blocks containing blocks can transitively capture more variables.
    IgnoredContexts.insert(BR->getBlockDecl());
    Visit(BR->getBlockDecl()->getBody());
  }
};
} // end anonymous namespace

typedef BumpVector<const VarDecl*> DeclVec;

static DeclVec* LazyInitializeReferencedDecls(const BlockDecl *BD,
                                              void *&Vec,
                                              llvm::BumpPtrAllocator &A) {
  if (Vec)
    return (DeclVec*) Vec;

  BumpVectorContext BC(A);
  DeclVec *BV = (DeclVec*) A.Allocate<DeclVec>();
  new (BV) DeclVec(BC, 10);

  // Find the referenced variables.
  FindBlockDeclRefExprsVals F(*BV, BC);
  F.Visit(BD->getBody());

  Vec = BV;
  return BV;
}

std::pair<AnalysisContext::referenced_decls_iterator,
          AnalysisContext::referenced_decls_iterator>
AnalysisContext::getReferencedBlockVars(const BlockDecl *BD) {
  if (!ReferencedBlockVars)
    ReferencedBlockVars = new llvm::DenseMap<const BlockDecl*,void*>();

  DeclVec *V = LazyInitializeReferencedDecls(BD, (*ReferencedBlockVars)[BD], A);
  return std::make_pair(V->begin(), V->end());
}

//===----------------------------------------------------------------------===//
// Cleanup.
//===----------------------------------------------------------------------===//

AnalysisContext::~AnalysisContext() {
  delete forcedBlkExprs;
  delete ReferencedBlockVars;
}

AnalysisContextManager::~AnalysisContextManager() {
  for (ContextMap::iterator I = Contexts.begin(), E = Contexts.end(); I!=E; ++I)
    delete I->second;
}

LocationContext::~LocationContext() {}

LocationContextManager::~LocationContextManager() {
  clear();
}

void LocationContextManager::clear() {
  for (llvm::FoldingSet<LocationContext>::iterator I = Contexts.begin(),
       E = Contexts.end(); I != E; ) {
    LocationContext *LC = &*I;
    ++I;
    delete LC;
  }

  Contexts.clear();
}

