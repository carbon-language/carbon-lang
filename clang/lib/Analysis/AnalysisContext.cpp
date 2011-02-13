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
#include "clang/Analysis/AnalysisContext.h"
#include "clang/Analysis/CFG.h"
#include "clang/Analysis/Support/BumpVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/ErrorHandling.h"

using namespace clang;

void AnalysisContextManager::clear() {
  for (ContextMap::iterator I = Contexts.begin(), E = Contexts.end(); I!=E; ++I)
    delete I->second;
  Contexts.clear();
}

Stmt *AnalysisContext::getBody() {
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

CFG *AnalysisContext::getCFG() {
  if (UseUnoptimizedCFG)
    return getUnoptimizedCFG();

  if (!builtCFG) {
    CFG::BuildOptions B;
    B.AddEHEdges = AddEHEdges;
    B.AddImplicitDtors = AddImplicitDtors;
    B.AddInitializers = AddInitializers;
    cfg = CFG::buildCFG(D, getBody(), &D->getASTContext(), B);
    // Even when the cfg is not successfully built, we don't
    // want to try building it again.
    builtCFG = true;
  }
  return cfg;
}

CFG *AnalysisContext::getUnoptimizedCFG() {
  if (!builtCompleteCFG) {
    CFG::BuildOptions B;
    B.PruneTriviallyFalseEdges = false;
    B.AddEHEdges = AddEHEdges;
    B.AddImplicitDtors = AddImplicitDtors;
    B.AddInitializers = AddInitializers;
    completeCFG = CFG::buildCFG(D, getBody(), &D->getASTContext(), B);
    // Even when the cfg is not successfully built, we don't
    // want to try building it again.
    builtCompleteCFG = true;
  }
  return completeCFG;
}

void AnalysisContext::dumpCFG() {
    getCFG()->dump(getASTContext().getLangOptions());
}

ParentMap &AnalysisContext::getParentMap() {
  if (!PM)
    PM = new ParentMap(getBody());
  return *PM;
}

PseudoConstantAnalysis *AnalysisContext::getPseudoConstantAnalysis() {
  if (!PCA)
    PCA = new PseudoConstantAnalysis(getBody());
  return PCA;
}

LiveVariables *AnalysisContext::getLiveVariables() {
  if (!liveness) {
    CFG *c = getCFG();
    if (!c)
      return 0;

    liveness = new LiveVariables(*this);
    liveness->runOnCFG(*c);
    liveness->runOnAllBlocks(*c, 0, true);
  }

  return liveness;
}

LiveVariables *AnalysisContext::getRelaxedLiveVariables() {
  if (!relaxedLiveness) {
    CFG *c = getCFG();
    if (!c)
      return 0;

    relaxedLiveness = new LiveVariables(*this, false);
    relaxedLiveness->runOnCFG(*c);
    relaxedLiveness->runOnAllBlocks(*c, 0, true);
  }

  return relaxedLiveness;
}

AnalysisContext *AnalysisContextManager::getContext(const Decl *D,
                                                    idx::TranslationUnit *TU) {
  AnalysisContext *&AC = Contexts[D];
  if (!AC)
    AC = new AnalysisContext(D, TU, UseUnoptimizedCFG, false,
        AddImplicitDtors, AddInitializers);

  return AC;
}

//===----------------------------------------------------------------------===//
// FoldingSet profiling.
//===----------------------------------------------------------------------===//

void LocationContext::ProfileCommon(llvm::FoldingSetNodeID &ID,
                                    ContextKind ck,
                                    AnalysisContext *ctx,
                                    const LocationContext *parent,
                                    const void* data) {
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
  delete cfg;
  delete completeCFG;
  delete liveness;
  delete relaxedLiveness;
  delete PM;
  delete PCA;
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

