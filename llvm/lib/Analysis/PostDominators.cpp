//===- PostDominators.cpp - Post-Dominator Calculation --------------------===//
//
// This file implements the post-dominator construction algorithms.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/PostDominators.h"
#include "llvm/iTerminators.h"
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

  // Scan the function looking for the root nodes of the post-dominance
  // relationships.  These blocks end with return and unwind instructions.
  // While we are iterating over the function, we also initialize all of the
  // domsets to empty.
  Roots.clear();
  for (Function::iterator I = F.begin(), E = F.end(); I != E; ++I) {
    Doms[I];  // Initialize to empty

    if (isa<ReturnInst>(I->getTerminator()) ||
        isa<UnwindInst>(I->getTerminator()))
      Roots.push_back(I);
  }

  // If there are no exit nodes for the function, postdomsets are all empty.
  // This can happen if the function just contains an infinite loop, for
  // example.
  if (Roots.empty()) return false;

  // If we have more than one root, we insert an artificial "null" exit, which
  // has "virtual edges" to each of the real exit nodes.
  if (Roots.size() > 1)
    Doms[0].insert(0);

  bool Changed;
  do {
    Changed = false;

    std::set<const BasicBlock*> Visited;
    DomSetType WorkingSet;

    for (unsigned i = 0, e = Roots.size(); i != e; ++i)
      for (idf_iterator<BasicBlock*> It = idf_begin(Roots[i]),
             E = idf_end(Roots[i]); It != E; ++It) {
        BasicBlock *BB = *It;
        succ_iterator SI = succ_begin(BB), SE = succ_end(BB);
        if (SI != SE) {                // Is there SOME successor?
          // Loop until we get to a successor that has had it's dom set filled
          // in at least once.  We are guaranteed to have this because we are
          // traversing the graph in DFO and have handled start nodes specially.
          //
          while (Doms[*SI].size() == 0) ++SI;
          WorkingSet = Doms[*SI];
          
          for (++SI; SI != SE; ++SI) { // Intersect all of the successor sets
            DomSetType &SuccSet = Doms[*SI];
            if (SuccSet.size())
              set_intersect(WorkingSet, SuccSet);
          }
        } else {
          // If this node has no successors, it must be one of the root nodes.
          // We will already take care of the notion that the node
          // post-dominates itself.  The only thing we have to add is that if
          // there are multiple root nodes, we want to insert a special "null"
          // exit node which dominates the roots as well.
          if (Roots.size() > 1)
            WorkingSet.insert(0);
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
  if (Roots.empty()) return;
  BasicBlock *Root = Roots.size() == 1 ? Roots[0] : 0;

  Nodes[Root] = RootNode = new Node(Root, 0);   // Add a node for the root...

  // Iterate over all nodes in depth first order...
  for (unsigned i = 0, e = Roots.size(); i != e; ++i)
    for (idf_iterator<BasicBlock*> I = idf_begin(Roots[i]),
           E = idf_end(Roots[i]); I != E; ++I) {
      BasicBlock *BB = *I;
      const DominatorSet::DomSetType &Dominators = DS.getDominators(BB);
      unsigned DomSetSize = Dominators.size();
      if (DomSetSize == 1) continue;  // Root node... IDom = null

      // If we have already computed the immediate dominator for this node,
      // don't revisit.  This can happen due to nodes reachable from multiple
      // roots, but which the idf_iterator doesn't know about.
      if (Nodes.find(BB) != Nodes.end()) continue;

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

//===----------------------------------------------------------------------===//
//  PostDominanceFrontier Implementation
//===----------------------------------------------------------------------===//

static RegisterAnalysis<PostDominanceFrontier>
H("postdomfrontier", "Post-Dominance Frontier Construction", true);

const DominanceFrontier::DomSetType &
PostDominanceFrontier::calculate(const PostDominatorTree &DT, 
                                 const DominatorTree::Node *Node) {
  // Loop over CFG successors to calculate DFlocal[Node]
  BasicBlock *BB = Node->getBlock();
  DomSetType &S = Frontiers[BB];       // The new set to fill in...
  if (getRoots().empty()) return S;

  if (BB)
    for (pred_iterator SI = pred_begin(BB), SE = pred_end(BB);
         SI != SE; ++SI)
      // Does Node immediately dominate this predeccessor?
      if (DT[*SI]->getIDom() != Node)
        S.insert(*SI);

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
