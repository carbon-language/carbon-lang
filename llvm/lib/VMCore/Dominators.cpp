//===- DominatorSet.cpp - Dominator Set Calculation --------------*- C++ -*--=//
//
// This file provides a simple class to calculate the dominator set of a
// function.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/Dominators.h"
#include "llvm/Transforms/Utils/UnifyFunctionExitNodes.h"
#include "llvm/Support/CFG.h"
#include "Support/DepthFirstIterator.h"
#include "Support/STLExtras.h"
#include "Support/SetOperations.h"
#include <algorithm>
using std::set;

//===----------------------------------------------------------------------===//
//  DominatorSet Implementation
//===----------------------------------------------------------------------===//

AnalysisID DominatorSet::ID(AnalysisID::create<DominatorSet>(), true);
AnalysisID DominatorSet::PostDomID(AnalysisID::create<DominatorSet>(), true);

bool DominatorSet::runOnFunction(Function &F) {
  Doms.clear();   // Reset from the last time we were run...

  if (isPostDominator())
    calcPostDominatorSet(F);
  else
    calcForwardDominatorSet(F);
  return false;
}

// dominates - Return true if A dominates B.  This performs the special checks
// neccesary if A and B are in the same basic block.
//
bool DominatorSet::dominates(Instruction *A, Instruction *B) const {
  BasicBlock *BBA = A->getParent(), *BBB = B->getParent();
  if (BBA != BBB) return dominates(BBA, BBB);
  
  // Loop through the basic block until we find A or B.
  BasicBlock::iterator I = BBA->begin();
  for (; &*I != A && &*I != B; ++I) /*empty*/;
  
  // A dominates B if it is found first in the basic block...
  return &*I == A;
}

// calcForwardDominatorSet - This method calculates the forward dominator sets
// for the specified function.
//
void DominatorSet::calcForwardDominatorSet(Function &F) {
  Root = &F.getEntryNode();
  assert(pred_begin(Root) == pred_end(Root) &&
	 "Root node has predecessors in function!");

  bool Changed;
  do {
    Changed = false;

    DomSetType WorkingSet;
    df_iterator<Function*> It = df_begin(&F), End = df_end(&F);
    for ( ; It != End; ++It) {
      BasicBlock *BB = *It;
      pred_iterator PI = pred_begin(BB), PEnd = pred_end(BB);
      if (PI != PEnd) {                // Is there SOME predecessor?
	// Loop until we get to a predecessor that has had it's dom set filled
	// in at least once.  We are guaranteed to have this because we are
	// traversing the graph in DFO and have handled start nodes specially.
	//
	while (Doms[*PI].size() == 0) ++PI;
	WorkingSet = Doms[*PI];

	for (++PI; PI != PEnd; ++PI) { // Intersect all of the predecessor sets
	  DomSetType &PredSet = Doms[*PI];
	  if (PredSet.size())
	    set_intersect(WorkingSet, PredSet);
	}
      }
	
      WorkingSet.insert(BB);           // A block always dominates itself
      DomSetType &BBSet = Doms[BB];
      if (BBSet != WorkingSet) {
	BBSet.swap(WorkingSet);        // Constant time operation!
	Changed = true;                // The sets changed.
      }
      WorkingSet.clear();              // Clear out the set for next iteration
    }
  } while (Changed);
}

// Postdominator set constructor.  This ctor converts the specified function to
// only have a single exit node (return stmt), then calculates the post
// dominance sets for the function.
//
void DominatorSet::calcPostDominatorSet(Function &F) {
  // Since we require that the unify all exit nodes pass has been run, we know
  // that there can be at most one return instruction in the function left.
  // Get it.
  //
  Root = getAnalysis<UnifyFunctionExitNodes>().getExitNode();

  if (Root == 0) {  // No exit node for the function?  Postdomsets are all empty
    for (Function::iterator FI = F.begin(), FE = F.end(); FI != FE; ++FI)
      Doms[FI] = DomSetType();
    return;
  }

  bool Changed;
  do {
    Changed = false;

    set<const BasicBlock*> Visited;
    DomSetType WorkingSet;
    idf_iterator<BasicBlock*> It = idf_begin(Root), End = idf_end(Root);
    for ( ; It != End; ++It) {
      BasicBlock *BB = *It;
      succ_iterator PI = succ_begin(BB), PEnd = succ_end(BB);
      if (PI != PEnd) {                // Is there SOME predecessor?
	// Loop until we get to a successor that has had it's dom set filled
	// in at least once.  We are guaranteed to have this because we are
	// traversing the graph in DFO and have handled start nodes specially.
	//
	while (Doms[*PI].size() == 0) ++PI;
	WorkingSet = Doms[*PI];

	for (++PI; PI != PEnd; ++PI) { // Intersect all of the successor sets
	  DomSetType &PredSet = Doms[*PI];
	  if (PredSet.size())
	    set_intersect(WorkingSet, PredSet);
	}
      }
	
      WorkingSet.insert(BB);           // A block always dominates itself
      DomSetType &BBSet = Doms[BB];
      if (BBSet != WorkingSet) {
	BBSet.swap(WorkingSet);        // Constant time operation!
	Changed = true;                // The sets changed.
      }
      WorkingSet.clear();              // Clear out the set for next iteration
    }
  } while (Changed);
}

// getAnalysisUsage - This obviously provides a dominator set, but it also
// uses the UnifyFunctionExitNodes pass if building post-dominators
//
void DominatorSet::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  if (isPostDominator()) {
    AU.addProvided(PostDomID);
    AU.addRequired(UnifyFunctionExitNodes::ID);
  } else {
    AU.addProvided(ID);
  }
}


//===----------------------------------------------------------------------===//
//  ImmediateDominators Implementation
//===----------------------------------------------------------------------===//

AnalysisID ImmediateDominators::ID(AnalysisID::create<ImmediateDominators>(), true);
AnalysisID ImmediateDominators::PostDomID(AnalysisID::create<ImmediateDominators>(), true);

// calcIDoms - Calculate the immediate dominator mapping, given a set of
// dominators for every basic block.
void ImmediateDominators::calcIDoms(const DominatorSet &DS) {
  // Loop over all of the nodes that have dominators... figuring out the IDOM
  // for each node...
  //
  for (DominatorSet::const_iterator DI = DS.begin(), DEnd = DS.end(); 
       DI != DEnd; ++DI) {
    BasicBlock *BB = DI->first;
    const DominatorSet::DomSetType &Dominators = DI->second;
    unsigned DomSetSize = Dominators.size();
    if (DomSetSize == 1) continue;  // Root node... IDom = null

    // Loop over all dominators of this node.  This corresponds to looping over
    // nodes in the dominator chain, looking for a node whose dominator set is
    // equal to the current nodes, except that the current node does not exist
    // in it.  This means that it is one level higher in the dom chain than the
    // current node, and it is our idom!
    //
    DominatorSet::DomSetType::const_iterator I = Dominators.begin();
    DominatorSet::DomSetType::const_iterator End = Dominators.end();
    for (; I != End; ++I) {   // Iterate over dominators...
      // All of our dominators should form a chain, where the number of elements
      // in the dominator set indicates what level the node is at in the chain.
      // We want the node immediately above us, so it will have an identical 
      // dominator set, except that BB will not dominate it... therefore it's
      // dominator set size will be one less than BB's...
      //
      if (DS.getDominators(*I).size() == DomSetSize - 1) {
	IDoms[BB] = *I;
	break;
      }
    }
  }
}


//===----------------------------------------------------------------------===//
//  DominatorTree Implementation
//===----------------------------------------------------------------------===//

AnalysisID DominatorTree::ID(AnalysisID::create<DominatorTree>(), true);
AnalysisID DominatorTree::PostDomID(AnalysisID::create<DominatorTree>(), true);

// DominatorTree::reset - Free all of the tree node memory.
//
void DominatorTree::reset() { 
  for (NodeMapType::iterator I = Nodes.begin(), E = Nodes.end(); I != E; ++I)
    delete I->second;
  Nodes.clear();
}


#if 0
// Given immediate dominators, we can also calculate the dominator tree
DominatorTree::DominatorTree(const ImmediateDominators &IDoms) 
  : DominatorBase(IDoms.getRoot()) {
  const Function *M = Root->getParent();

  Nodes[Root] = new Node(Root, 0);   // Add a node for the root...

  // Iterate over all nodes in depth first order...
  for (df_iterator<const Function*> I = df_begin(M), E = df_end(M); I!=E; ++I) {
    const BasicBlock *BB = *I, *IDom = IDoms[*I];

    if (IDom != 0) {   // Ignore the root node and other nasty nodes
      // We know that the immediate dominator should already have a node, 
      // because we are traversing the CFG in depth first order!
      //
      assert(Nodes[IDom] && "No node for IDOM?");
      Node *IDomNode = Nodes[IDom];

      // Add a new tree node for this BasicBlock, and link it as a child of
      // IDomNode
      Nodes[BB] = IDomNode->addChild(new Node(BB, IDomNode));
    }
  }
}
#endif

void DominatorTree::calculate(const DominatorSet &DS) {
  Nodes[Root] = new Node(Root, 0);   // Add a node for the root...

  if (!isPostDominator()) {
    // Iterate over all nodes in depth first order...
    for (df_iterator<BasicBlock*> I = df_begin(Root), E = df_end(Root);
         I != E; ++I) {
      BasicBlock *BB = *I;
      const DominatorSet::DomSetType &Dominators = DS.getDominators(BB);
      unsigned DomSetSize = Dominators.size();
      if (DomSetSize == 1) continue;  // Root node... IDom = null
      
      // Loop over all dominators of this node. This corresponds to looping over
      // nodes in the dominator chain, looking for a node whose dominator set is
      // equal to the current nodes, except that the current node does not exist
      // in it. This means that it is one level higher in the dom chain than the
      // current node, and it is our idom!  We know that we have already added
      // a DominatorTree node for our idom, because the idom must be a
      // predecessor in the depth first order that we are iterating through the
      // function.
      //
      DominatorSet::DomSetType::const_iterator I = Dominators.begin();
      DominatorSet::DomSetType::const_iterator End = Dominators.end();
      for (; I != End; ++I) {   // Iterate over dominators...
	// All of our dominators should form a chain, where the number of
	// elements in the dominator set indicates what level the node is at in
	// the chain.  We want the node immediately above us, so it will have
	// an identical dominator set, except that BB will not dominate it...
	// therefore it's dominator set size will be one less than BB's...
	//
	if (DS.getDominators(*I).size() == DomSetSize - 1) {
	  // We know that the immediate dominator should already have a node, 
	  // because we are traversing the CFG in depth first order!
	  //
	  Node *IDomNode = Nodes[*I];
	  assert(IDomNode && "No node for IDOM?");
	  
	  // Add a new tree node for this BasicBlock, and link it as a child of
	  // IDomNode
	  Nodes[BB] = IDomNode->addChild(new Node(BB, IDomNode));
	  break;
	}
      }
    }
  } else if (Root) {
    // Iterate over all nodes in depth first order...
    for (idf_iterator<BasicBlock*> I = idf_begin(Root), E = idf_end(Root);
         I != E; ++I) {
      BasicBlock *BB = *I;
      const DominatorSet::DomSetType &Dominators = DS.getDominators(BB);
      unsigned DomSetSize = Dominators.size();
      if (DomSetSize == 1) continue;  // Root node... IDom = null
      
      // Loop over all dominators of this node.  This corresponds to looping
      // over nodes in the dominator chain, looking for a node whose dominator
      // set is equal to the current nodes, except that the current node does
      // not exist in it.  This means that it is one level higher in the dom
      // chain than the current node, and it is our idom!  We know that we have
      // already added a DominatorTree node for our idom, because the idom must
      // be a predecessor in the depth first order that we are iterating through
      // the function.
      //
      DominatorSet::DomSetType::const_iterator I = Dominators.begin();
      DominatorSet::DomSetType::const_iterator End = Dominators.end();
      for (; I != End; ++I) {   // Iterate over dominators...
	// All of our dominators should form a chain, where the number
	// of elements in the dominator set indicates what level the
	// node is at in the chain.  We want the node immediately
	// above us, so it will have an identical dominator set,
	// except that BB will not dominate it... therefore it's
	// dominator set size will be one less than BB's...
	//
	if (DS.getDominators(*I).size() == DomSetSize - 1) {
	  // We know that the immediate dominator should already have a node, 
	  // because we are traversing the CFG in depth first order!
	  //
	  Node *IDomNode = Nodes[*I];
	  assert(IDomNode && "No node for IDOM?");
	  
	  // Add a new tree node for this BasicBlock, and link it as a child of
	  // IDomNode
	  Nodes[BB] = IDomNode->addChild(new Node(BB, IDomNode));
	  break;
	}
      }
    }
  }
}



//===----------------------------------------------------------------------===//
//  DominanceFrontier Implementation
//===----------------------------------------------------------------------===//

AnalysisID DominanceFrontier::ID(AnalysisID::create<DominanceFrontier>(), true);
AnalysisID DominanceFrontier::PostDomID(AnalysisID::create<DominanceFrontier>(), true);

const DominanceFrontier::DomSetType &
DominanceFrontier::calcDomFrontier(const DominatorTree &DT, 
                                   const DominatorTree::Node *Node) {
  // Loop over CFG successors to calculate DFlocal[Node]
  BasicBlock *BB = Node->getNode();
  DomSetType &S = Frontiers[BB];       // The new set to fill in...

  for (succ_iterator SI = succ_begin(BB), SE = succ_end(BB);
       SI != SE; ++SI) {
    // Does Node immediately dominate this successor?
    if (DT[*SI]->getIDom() != Node)
      S.insert(*SI);
  }

  // At this point, S is DFlocal.  Now we union in DFup's of our children...
  // Loop through and visit the nodes that Node immediately dominates (Node's
  // children in the IDomTree)
  //
  for (DominatorTree::Node::const_iterator NI = Node->begin(), NE = Node->end();
       NI != NE; ++NI) {
    DominatorTree::Node *IDominee = *NI;
    const DomSetType &ChildDF = calcDomFrontier(DT, IDominee);

    DomSetType::const_iterator CDFI = ChildDF.begin(), CDFE = ChildDF.end();
    for (; CDFI != CDFE; ++CDFI) {
      if (!Node->dominates(DT[*CDFI]))
	S.insert(*CDFI);
    }
  }

  return S;
}

const DominanceFrontier::DomSetType &
DominanceFrontier::calcPostDomFrontier(const DominatorTree &DT, 
                                       const DominatorTree::Node *Node) {
  // Loop over CFG successors to calculate DFlocal[Node]
  BasicBlock *BB = Node->getNode();
  DomSetType &S = Frontiers[BB];       // The new set to fill in...
  if (!Root) return S;

  for (pred_iterator SI = pred_begin(BB), SE = pred_end(BB);
       SI != SE; ++SI) {
    // Does Node immediately dominate this predeccessor?
    if (DT[*SI]->getIDom() != Node)
      S.insert(*SI);
  }

  // At this point, S is DFlocal.  Now we union in DFup's of our children...
  // Loop through and visit the nodes that Node immediately dominates (Node's
  // children in the IDomTree)
  //
  for (DominatorTree::Node::const_iterator NI = Node->begin(), NE = Node->end();
       NI != NE; ++NI) {
    DominatorTree::Node *IDominee = *NI;
    const DomSetType &ChildDF = calcPostDomFrontier(DT, IDominee);

    DomSetType::const_iterator CDFI = ChildDF.begin(), CDFE = ChildDF.end();
    for (; CDFI != CDFE; ++CDFI) {
      if (!Node->dominates(DT[*CDFI]))
	S.insert(*CDFI);
    }
  }

  return S;
}
