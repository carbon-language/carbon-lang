//==- GREngine.cpp - Path-Sensitive Dataflow Engine ----------------*- C++ -*-//
//             
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines a generic engine for intraprocedural, path-sensitive,
//  dataflow analysis via graph reachability engine.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/PathSensitive/GREngine.h"
#include "clang/AST/Expr.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Casting.h"
#include "llvm/ADT/DenseMap.h"
#include <vector>

using llvm::cast;
using llvm::isa;
using namespace clang;

namespace {
  class VISIBILITY_HIDDEN DFS : public GRWorkList {
  llvm::SmallVector<GRWorkListUnit,20> Stack;
public:
  virtual bool hasWork() const {
    return !Stack.empty();
  }

  virtual void Enqueue(const GRWorkListUnit& U) {
    Stack.push_back(U);
  }

  virtual GRWorkListUnit Dequeue() {
    assert (!Stack.empty());
    const GRWorkListUnit& U = Stack.back();
    Stack.pop_back(); // This technically "invalidates" U, but we are fine.
    return U;
  }
};
} // end anonymous namespace

GRWorkList* GRWorkList::MakeDFS() { return new DFS(); }

/// ExecuteWorkList - Run the worklist algorithm for a maximum number of steps.
bool GREngineImpl::ExecuteWorkList(unsigned Steps) {
  
  if (G->num_roots() == 0) { // Initialize the analysis by constructing
    // the root if none exists.
    
    CFGBlock* Entry = &cfg.getEntry();
    
    assert (Entry->empty() && 
            "Entry block must be empty.");
    
    assert (Entry->succ_size() == 1 &&
            "Entry block must have 1 successor.");
    
    // Get the solitary successor.
    CFGBlock* Succ = *(Entry->succ_begin());   
    
    // Construct an edge representing the
    // starting location in the function.
    BlockEdge StartLoc(cfg, Entry, Succ);
    
    // Generate the root.
    GenerateNode(StartLoc, getInitialState());
  }
  
  while (Steps && WList->hasWork()) {
    --Steps;
    const GRWorkListUnit& WU = WList->Dequeue();
    ExplodedNodeImpl* Node = WU.getNode();
    
    // Dispatch on the location type.
    switch (Node->getLocation().getKind()) {
      default:
        assert (isa<BlockEdge>(Node->getLocation()));
        HandleBlockEdge(cast<BlockEdge>(Node->getLocation()), Node);
        break;
        
      case ProgramPoint::BlockEntranceKind:
        HandleBlockEntrance(cast<BlockEntrance>(Node->getLocation()), Node);
        break;
        
      case ProgramPoint::BlockExitKind:
        HandleBlockExit(cast<BlockExit>(Node->getLocation()), Node);
        break;
        
      case ProgramPoint::PostStmtKind:
        HandlePostStmt(cast<PostStmt>(Node->getLocation()), WU.getBlock(),
                       WU.getIndex(), Node);
        break;        
    }
  }
  
  return WList->hasWork();
}

void GREngineImpl::HandleBlockEdge(const BlockEdge& L, ExplodedNodeImpl* Pred) {
  
  CFGBlock* Blk = L.getDst();
  
  // Check if we are entering the EXIT block. 
  if (Blk == &cfg.getExit()) {
    
    assert (cfg.getExit().size() == 0 && "EXIT block cannot contain Stmts.");

    // Process the final state transition.    
    void* State = ProcessEOP(Blk, Pred->State);

    bool IsNew;
    ExplodedNodeImpl* Node = G->getNodeImpl(BlockEntrance(Blk), State, &IsNew);
    Node->addPredecessor(Pred);
    
    // If the node was freshly created, mark it as an "End-Of-Path" node.
    if (IsNew) G->addEndOfPath(Node); 
    
    // This path is done. Don't enqueue any more nodes.
    return;
  }
  
  // FIXME: we will dispatch to a function that
  //  manipulates the state at the entrance to a block.
  
  if (!Blk->empty())                            
    GenerateNode(BlockEntrance(Blk), Pred->State, Pred);
  else
    GenerateNode(BlockExit(Blk), Pred->State, Pred);
}

void GREngineImpl::HandleBlockEntrance(const BlockEntrance& L,
                                       ExplodedNodeImpl* Pred) {
  
  if (Stmt* S = L.getFirstStmt()) {
    GRNodeBuilderImpl Builder(L.getBlock(), 0, Pred, this);
    ProcessStmt(S, Builder);
  }
  else
    GenerateNode(BlockExit(L.getBlock()), Pred->State, Pred);
}


void GREngineImpl::HandleBlockExit(const BlockExit& L, ExplodedNodeImpl* Pred) {
  
  CFGBlock* B = L.getBlock();
  
  if (Stmt* Terminator = B->getTerminator())
    ProcessTerminator(Terminator, B, Pred);
  else {
    assert (B->succ_size() == 1 &&
            "Blocks with no terminator should have at most 1 successor.");
    
    GenerateNode(BlockEdge(cfg,B,*(B->succ_begin())), Pred->State, Pred);    
  }
}

void GREngineImpl::HandlePostStmt(const PostStmt& L, CFGBlock* B,
                                  unsigned StmtIdx, ExplodedNodeImpl* Pred) {
  
  assert (!B->empty());

  if (StmtIdx == B->size()) {
    // FIXME: This is essentially an epsilon-transition.  Do we need it?
    //  It does simplify the logic, and it is also another point
    //  were we could introduce a dispatch to the client.
    GenerateNode(BlockExit(B), Pred->State, Pred);
  }
  else {
    GRNodeBuilderImpl Builder(B, StmtIdx, Pred, this);
    ProcessStmt(L.getStmt(), Builder);
  }
}

typedef llvm::DenseMap<Stmt*,Stmt*> ParentMapTy;
/// PopulateParentMap - Recurse the AST starting at 'Parent' and add the
///  mappings between child and parent to ParentMap.
static void PopulateParentMap(Stmt* Parent, ParentMapTy& M) {
  for (Stmt::child_iterator I=Parent->child_begin(), 
       E=Parent->child_end(); I!=E; ++I) {
    
    assert (M.find(*I) == M.end());
    M[*I] = Parent;
    PopulateParentMap(*I, M);
  }
}

/// GenerateNode - Utility method to generate nodes, hook up successors,
///  and add nodes to the worklist.
void GREngineImpl::GenerateNode(const ProgramPoint& Loc, void* State,
                                ExplodedNodeImpl* Pred) {
  
  bool IsNew;
  ExplodedNodeImpl* Node = G->getNodeImpl(Loc, State, &IsNew);
  
  if (Pred) 
    Node->addPredecessor(Pred);  // Link 'Node' with its predecessor.
  else {
    assert (IsNew);
    G->addRoot(Node);  // 'Node' has no predecessor.  Make it a root.
  }
  
  // Only add 'Node' to the worklist if it was freshly generated.
  if (IsNew) WList->Enqueue(GRWorkListUnit(Node));
}

GRNodeBuilderImpl::GRNodeBuilderImpl(CFGBlock* b, unsigned idx,
                                     ExplodedNodeImpl* N, GREngineImpl* e)
  : Eng(*e), B(*b), Idx(idx), LastNode(N), Populated(false) {
  Deferred.insert(N);
}

GRNodeBuilderImpl::~GRNodeBuilderImpl() {
  for (DeferredTy::iterator I=Deferred.begin(), E=Deferred.end(); I!=E; ++I)
    if (!(*I)->isInfeasible())
      GenerateAutoTransition(*I);
}

void GRNodeBuilderImpl::GenerateAutoTransition(ExplodedNodeImpl* N) {
  assert (!N->isInfeasible());
  
  PostStmt Loc(getStmt());
  
  if (Loc == N->getLocation()) {
    // Note: 'N' should be a fresh node because otherwise it shouldn't be
    // a member of Deferred.
    Eng.WList->Enqueue(N, B, Idx+1);
    return;
  }
  
  bool IsNew;
  ExplodedNodeImpl* Succ = Eng.G->getNodeImpl(Loc, N->State, &IsNew);
  Succ->addPredecessor(N);

  if (IsNew)
    Eng.WList->Enqueue(Succ, B, Idx+1);
}

ExplodedNodeImpl* GRNodeBuilderImpl::generateNodeImpl(Stmt* S, void* State,
                                                      ExplodedNodeImpl* Pred) {
  
  bool IsNew;
  ExplodedNodeImpl* N = Eng.G->getNodeImpl(PostStmt(S), State, &IsNew);
  N->addPredecessor(Pred);
  Deferred.erase(Pred);
  
  HasGeneratedNode = true;
  
  if (IsNew) {
    Deferred.insert(N);
    LastNode = N;
    return N;
  }
  
  LastNode = NULL;
  return NULL;  
}
