//===- PostDominators.cpp - Post-Dominator Calculation --------------------===//
//
// This file implements the post-dominator construction algorithms.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/PostDominators.h"
#include "llvm/Transforms/Utils/UnifyFunctionExitNodes.h"
#include "llvm/Support/CFG.h"
#include "Support/DepthFirstIterator.h"
#include "Support/SetOperations.h"

//===----------------------------------------------------------------------===//
//  PostDominatorSet Implementation
//===----------------------------------------------------------------------===//

static RegisterAnalysis<PostDominatorSet>
B("postdomset", "Post-Dominator Set Construction", true);

// Postdominator set construction.  This converts the specified function to only
// have a single exit node (return stmt), then calculates the post dominance
// sets for the function.
//
bool PostDominatorSet::runOnFunction(Function &F) {
  Doms.clear();   // Reset from the last time we were run...
  // Since we require that the unify all exit nodes pass has been run, we know
  // that there can be at most one return instruction in the function left.
  // Get it.
  //
  Root = getAnalysis<UnifyFunctionExitNodes>().getExitNode();

  if (Root == 0) {  // No exit node for the function?  Postdomsets are all empty
    for (Function::iterator FI = F.begin(), FE = F.end(); FI != FE; ++FI)
      Doms[FI] = DomSetType();
    return false;
  }

  bool Changed;
  do {
    Changed = false;

    std::set<const BasicBlock*> Visited;
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
      } else if (BB != Root) {
        // If this isn't the root basic block and it has no successors, it must
        // be an non-returning block.  Fib a bit by saying that the root node
        // postdominates this unreachable node.  This isn't exactly true,
        // because there is no path from this node to the root node, but it is
        // sorta true because any paths to the exit node would have to go
        // through this node.
        //
        // This allows for postdominator properties to be built for code that
        // doesn't return in a reasonable manner.
        //
        WorkingSet = Doms[Root];
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
  return false;
}

// getAnalysisUsage - This obviously provides a post-dominator set, but it also
// requires the UnifyFunctionExitNodes pass.
//
void PostDominatorSet::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<UnifyFunctionExitNodes>();
}

//===----------------------------------------------------------------------===//
//  ImmediatePostDominators Implementation
//===----------------------------------------------------------------------===//

static RegisterAnalysis<ImmediatePostDominators>
D("postidom", "Immediate Post-Dominators Construction", true);

//===----------------------------------------------------------------------===//
//  PostDominatorTree Implementation
//===----------------------------------------------------------------------===//

static RegisterAnalysis<PostDominatorTree>
F("postdomtree", "Post-Dominator Tree Construction", true);

void PostDominatorTree::calculate(const PostDominatorSet &DS) {
  Nodes[Root] = new Node(Root, 0);   // Add a node for the root...

  if (Root) {
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
//  PostDominanceFrontier Implementation
//===----------------------------------------------------------------------===//

static RegisterAnalysis<PostDominanceFrontier>
H("postdomfrontier", "Post-Dominance Frontier Construction", true);

const DominanceFrontier::DomSetType &
PostDominanceFrontier::calculate(const PostDominatorTree &DT, 
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
  for (PostDominatorTree::Node::const_iterator
         NI = Node->begin(), NE = Node->end(); NI != NE; ++NI) {
    DominatorTree::Node *IDominee = *NI;
    const DomSetType &ChildDF = calculate(DT, IDominee);

    DomSetType::const_iterator CDFI = ChildDF.begin(), CDFE = ChildDF.end();
    for (; CDFI != CDFE; ++CDFI) {
      if (!Node->dominates(DT[*CDFI]))
	S.insert(*CDFI);
    }
  }

  return S;
}

// stub - a dummy function to make linking work ok.
void PostDominanceFrontier::stub() {
}
