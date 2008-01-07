//= ReachabilityEngine.cpp - Path-Sens. Dataflow Engine ------------*- C++ -*-//
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

#include "clang/Analysis/PathSensitive/ReachabilityEngine.h"
#include "clang/AST/Stmt.h"
#include "llvm/Support/Casting.h"

using namespace clang;
using clang::reng::WorkList;
using llvm::isa;
using llvm::cast;

// Place dstor here so that all of the virtual functions in DFS have their
// code placed in the object file of this translation unit.
clang::reng::DFS::~DFS() {}

ReachabilityEngineImpl::ReachabilityEngineImpl(CFG& c,
                                               clang::reng::WorkList* wlist)
  : cfg(c), WList(wlist) {
    
  // Get the entry block.  Make sure that it has 1 (and only 1) successor.
  CFGBlock* Entry = &c.getEntry();  
  assert (Entry->empty() && "Entry block must be empty.");
  assert (Entry->succ_size() == 1 && "Entry block must have 1 successor.");
  
  // Get the first (and only) successor of Entry.
  CFGBlock* Succ = *(Entry->succ_begin());
  
  // Construct an edge representing the starting location in the function.
  BlkBlkEdge StartLoc(Entry,Succ);
  
  // Create the root node.
  assert (false && "FIXME soon.");
//  WList->Enqueue(G->addRoot(getNode(StartLoc));
}

void ReachabilityEngineImpl::getNode(const ProgramEdge& Loc, void* State, 
                                     ExplodedNodeImpl* Pred) {
  
  bool IsNew;
  ExplodedNodeImpl* V = G->getNodeImpl(Loc,State,IsNew);

  // Link the node with its predecessor.
  V->addUntypedPredecessor(Pred);
  
  if (IsNew) {
    // Only add the node to the worklist if it was freshly generated.    
    WList->Enqueue(V);
  
    // Check if the node's edge is a StmtStmtEdge where the destination
    // statement is not a BlockLevelExpr.  In this case, we must lazily
    // populate ParentMap.
    if (isa<StmtStmtEdge>(Loc)) {
      Stmt* S = cast<StmtStmtEdge>(Loc).Dst();
      assert (CurrentBlkExpr != NULL);

      if (S != CurrentBlkExpr && ParentMap.find(S) == ParentMap.end()) {
        // Populate ParentMap starting from CurrentBlkExpr.
        PopulateParentMap(CurrentBlkExpr);
        assert (ParentMap.find(S) != ParentMap.end());
      }
    }
  }
}

void ReachabilityEngineImpl::PopulateParentMap(Stmt* Parent) {
  for (Stmt::child_iterator I=Parent->child_begin(), 
                            E=Parent->child_end(); I!=E; ++I) {
    
    assert (ParentMap.find(*I) == ParentMap.end());
    ParentMap[*I] = Parent;
    PopulateParentMap(*I);
  }       
}
  
bool ReachabilityEngineImpl::ExecuteWorkList(unsigned Steps) {
  
  while (Steps && WList->hasWork()) {
    --Steps;
    ExplodedNodeImpl* V = WList->Dequeue();
    
    // Dispatch on the location type.
    switch (V->getLocation().getKind()) {
      case ProgramEdge::BlkBlk:
        ProcessBlkBlk(cast<BlkBlkEdge>(V->getLocation()),V);
        break;
        
      case ProgramEdge::BlkStmt:
        ProcessBlkStmt(cast<BlkStmtEdge>(V->getLocation()),V);
        break;
        
      case ProgramEdge::StmtBlk:
        ProcessStmtBlk(cast<StmtBlkEdge>(V->getLocation()),V);
        break;
        
      case ProgramEdge::StmtStmt:
        ProcessStmt(cast<StmtStmtEdge>(V->getLocation()).Dst(),V);
        break;
        
      default:
        assert (false && "Unsupported edge type.");
    }
  }
  
  return WList->hasWork();
}


void ReachabilityEngineImpl::ProcessBlkBlk(const BlkBlkEdge& E,
                                           ExplodedNodeImpl* Pred) {
  
  CFGBlock* Blk = E.Dst();
  
  // Check if we are entering the EXIT block.
  if (Blk == &cfg.getExit()) {
    assert (cfg.getExit().size() == 0 && "EXIT block cannot contain Stmts.");
    // Process the End-Of-Path.
    ProcessEOP(Blk, Pred);
    return;
  }
  
  // FIXME: we will dispatch to a function that manipulates the state
  //  at the entrance to a block.
  
  if (!Blk->empty()) {
    // If 'Blk' has at least one statement, create a BlkStmtEdge and create
    // the appropriate node.  This is the common case.
    getNode(BlkStmtEdge(Blk,Blk->front()), Pred->State, Pred);
  }
  else {
    // Otherwise, create a node at the BlkStmtEdge right before the terminator
    // (if any) is evaluated.  
    getNode(StmtBlkEdge(NULL,Blk),Pred->State, Pred);
  }
}

void ReachabilityEngineImpl::ProcessBlkStmt(const BlkStmtEdge& E,
                                            ExplodedNodeImpl* Pred) {  
  if (Stmt* S = E.Dst())
    ProcessStmt(S,Pred);
  else {
    // No statement.  Create an edge right before the terminator is evaluated.
    getNode(StmtBlkEdge(NULL,E.Src()), Pred->State, Pred);
  }
}

void ReachabilityEngineImpl::ProcessStmtBlk(const StmtBlkEdge& E,
                                            ExplodedNodeImpl* Pred) {
  CFGBlock* Blk = E.Dst();
  
  if (Stmt* Terminator = Blk->getTerminator())
    ProcessTerminator(Terminator,Pred);
  else {
    // No terminator.  We should have only 1 successor.
    assert (Blk->succ_size() == 1);    
    getNode(BlkBlkEdge(Blk,*(Blk->succ_begin())), Pred);
  }
}
