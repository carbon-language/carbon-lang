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
#include "clang/Analysis/PathSensitive/MemRegion.h"
#include "clang/Analysis/Analyses/LiveVariables.h"
#include "clang/Analysis/CFG.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ParentMap.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Analysis/Support/BumpVector.h"
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

    liveness = new LiveVariables(*this);
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

const BlockDecl *BlockInvocationContext::getBlockDecl() const {
  return Data.is<const BlockDataRegion*>() ?
    Data.get<const BlockDataRegion*>()->getDecl()
  : Data.get<const BlockDecl*>();
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
  if (const BlockDataRegion *BR = getBlockRegion())
    Profile(ID, getAnalysisContext(), getParent(), BR);
  else
    Profile(ID, getAnalysisContext(), getParent(),
            Data.get<const BlockDecl*>());    
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
                                      const Stmt *s, const CFGBlock *blk,
                                      unsigned idx) {
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

const BlockInvocationContext *
LocationContextManager::getBlockInvocation(AnalysisContext *ctx,
                                 const LocationContext *parent,
                                 const BlockDataRegion *BR) {
  return getLocationContext<BlockInvocationContext, BlockDataRegion>(ctx,
                                                                     parent,
                                                                     BR);
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

//===----------------------------------------------------------------------===//
// Lazily generated map to query the external variables referenced by a Block.
//===----------------------------------------------------------------------===//

namespace {
class FindBlockDeclRefExprsVals : public StmtVisitor<FindBlockDeclRefExprsVals>{
  BumpVector<const VarDecl*> &BEVals;
  BumpVectorContext &BC;
public:
  FindBlockDeclRefExprsVals(BumpVector<const VarDecl*> &bevals,
                            BumpVectorContext &bc)
  : BEVals(bevals), BC(bc) {}
  
  void VisitStmt(Stmt *S) {
    for (Stmt::child_iterator I = S->child_begin(), E = S->child_end();I!=E;++I)
      if (Stmt *child = *I)
        Visit(child);
  }
  
  void VisitBlockDeclRefExpr(BlockDeclRefExpr *DR) {
    if (const VarDecl *VD = dyn_cast<VarDecl>(DR->getDecl()))
      BEVals.push_back(VD, BC);
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
  delete liveness;
  delete PM;
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

