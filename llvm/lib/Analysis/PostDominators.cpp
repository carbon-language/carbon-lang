//===- PostDominators.cpp - Post-Dominator Calculation --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the post-dominator construction algorithms.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/PostDominators.h"
#include "llvm/Instructions.h"
#include "llvm/Support/CFG.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SetOperations.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
//  PostDominatorTree Implementation
//===----------------------------------------------------------------------===//

char PostDominatorTree::ID = 0;
char PostDominanceFrontier::ID = 0;
static RegisterPass<PostDominatorTree>
F("postdomtree", "Post-Dominator Tree Construction", true);

unsigned PostDominatorTree::DFSPass(BasicBlock *V, InfoRec &VInfo,
                                          unsigned N) {
  std::vector<std::pair<BasicBlock *, InfoRec *> > workStack;
  std::set<BasicBlock *> visited;
  workStack.push_back(std::make_pair(V, &VInfo));

  do {
    BasicBlock *currentBB = workStack.back().first; 
    InfoRec *currentVInfo = workStack.back().second;

    // Visit each block only once.
    if (visited.count(currentBB) == 0) {

      visited.insert(currentBB);
      currentVInfo->Semi = ++N;
      currentVInfo->Label = currentBB;
      
      Vertex.push_back(currentBB);  // Vertex[n] = current;
      // Info[currentBB].Ancestor = 0;     
      // Ancestor[n] = 0
      // Child[currentBB] = 0;
      currentVInfo->Size = 1;       // Size[currentBB] = 1
    }

    // Visit children
    bool visitChild = false;
    for (pred_iterator PI = pred_begin(currentBB), PE = pred_end(currentBB); 
         PI != PE && !visitChild; ++PI) {
      InfoRec &SuccVInfo = Info[*PI];
      if (SuccVInfo.Semi == 0) {
        SuccVInfo.Parent = currentBB;
        if (visited.count (*PI) == 0) {
          workStack.push_back(std::make_pair(*PI, &SuccVInfo));   
          visitChild = true;
        }
      }
    }

    // If all children are visited or if this block has no child then pop this
    // block out of workStack.
    if (!visitChild)
      workStack.pop_back();

  } while (!workStack.empty());

  return N;
}

void PostDominatorTree::Compress(BasicBlock *V, InfoRec &VInfo) {
  BasicBlock *VAncestor = VInfo.Ancestor;
  InfoRec &VAInfo = Info[VAncestor];
  if (VAInfo.Ancestor == 0)
    return;
  
  Compress(VAncestor, VAInfo);
  
  BasicBlock *VAncestorLabel = VAInfo.Label;
  BasicBlock *VLabel = VInfo.Label;
  if (Info[VAncestorLabel].Semi < Info[VLabel].Semi)
    VInfo.Label = VAncestorLabel;
  
  VInfo.Ancestor = VAInfo.Ancestor;
}

BasicBlock *PostDominatorTree::Eval(BasicBlock *V) {
  InfoRec &VInfo = Info[V];

  // Higher-complexity but faster implementation
  if (VInfo.Ancestor == 0)
    return V;
  Compress(V, VInfo);
  return VInfo.Label;
}

void PostDominatorTree::Link(BasicBlock *V, BasicBlock *W, 
                                   InfoRec &WInfo) {
  // Higher-complexity but faster implementation
  WInfo.Ancestor = V;
}

void PostDominatorTree::calculate(Function &F) {
  // Step #0: Scan the function looking for the root nodes of the post-dominance
  // relationships.  These blocks, which have no successors, end with return and
  // unwind instructions.
  for (Function::iterator I = F.begin(), E = F.end(); I != E; ++I)
    if (succ_begin(I) == succ_end(I)) {
      Instruction *Insn = I->getTerminator();
      // Unreachable block is not a root node.
      if (!isa<UnreachableInst>(Insn))
        Roots.push_back(I);
    }
  
  Vertex.push_back(0);
  
  // Step #1: Number blocks in depth-first order and initialize variables used
  // in later stages of the algorithm.
  unsigned N = 0;
  for (unsigned i = 0, e = Roots.size(); i != e; ++i)
    N = DFSPass(Roots[i], Info[Roots[i]], N);
  
  for (unsigned i = N; i >= 2; --i) {
    BasicBlock *W = Vertex[i];
    InfoRec &WInfo = Info[W];
    
    // Step #2: Calculate the semidominators of all vertices
    for (succ_iterator SI = succ_begin(W), SE = succ_end(W); SI != SE; ++SI)
      if (Info.count(*SI)) {  // Only if this predecessor is reachable!
        unsigned SemiU = Info[Eval(*SI)].Semi;
        if (SemiU < WInfo.Semi)
          WInfo.Semi = SemiU;
      }
        
    Info[Vertex[WInfo.Semi]].Bucket.push_back(W);
    
    BasicBlock *WParent = WInfo.Parent;
    Link(WParent, W, WInfo);
    
    // Step #3: Implicitly define the immediate dominator of vertices
    std::vector<BasicBlock*> &WParentBucket = Info[WParent].Bucket;
    while (!WParentBucket.empty()) {
      BasicBlock *V = WParentBucket.back();
      WParentBucket.pop_back();
      BasicBlock *U = Eval(V);
      IDoms[V] = Info[U].Semi < Info[V].Semi ? U : WParent;
    }
  }
  
  // Step #4: Explicitly define the immediate dominator of each vertex
  for (unsigned i = 2; i <= N; ++i) {
    BasicBlock *W = Vertex[i];
    BasicBlock *&WIDom = IDoms[W];
    if (WIDom != Vertex[Info[W].Semi])
      WIDom = IDoms[WIDom];
  }
  
  if (Roots.empty()) return;

  // Add a node for the root.  This node might be the actual root, if there is
  // one exit block, or it may be the virtual exit (denoted by (BasicBlock *)0)
  // which postdominates all real exits if there are multiple exit blocks.
  BasicBlock *Root = Roots.size() == 1 ? Roots[0] : 0;
  DomTreeNodes[Root] = RootNode = new DomTreeNode(Root, 0);
  
  // Loop over all of the reachable blocks in the function...
  for (Function::iterator I = F.begin(), E = F.end(); I != E; ++I)
    if (BasicBlock *ImmPostDom = getIDom(I)) {  // Reachable block.
      DomTreeNode *&BBNode = DomTreeNodes[I];
      if (!BBNode) {  // Haven't calculated this node yet?
                      // Get or calculate the node for the immediate dominator
        DomTreeNode *IPDomNode = getNodeForBlock(ImmPostDom);
        
        // Add a new tree node for this BasicBlock, and link it as a child of
        // IDomNode
        DomTreeNode *C = new DomTreeNode(I, IPDomNode);
        DomTreeNodes[I] = C;
        BBNode = IPDomNode->addChild(C);
      }
    }

  // Free temporary memory used to construct idom's
  IDoms.clear();
  Info.clear();
  std::vector<BasicBlock*>().swap(Vertex);

  int dfsnum = 0;
  // Iterate over all nodes in depth first order...
  for (unsigned i = 0, e = Roots.size(); i != e; ++i)
    for (idf_iterator<BasicBlock*> I = idf_begin(Roots[i]),
           E = idf_end(Roots[i]); I != E; ++I) {
      if (!getNodeForBlock(*I)->getIDom())
        getNodeForBlock(*I)->assignDFSNumber(dfsnum);
    }
  DFSInfoValid = true;
}


DomTreeNode *PostDominatorTree::getNodeForBlock(BasicBlock *BB) {
  DomTreeNode *&BBNode = DomTreeNodes[BB];
  if (BBNode) return BBNode;
  
  // Haven't calculated this node yet?  Get or calculate the node for the
  // immediate postdominator.
  BasicBlock *IPDom = getIDom(BB);
  DomTreeNode *IPDomNode = getNodeForBlock(IPDom);
  
  // Add a new tree node for this BasicBlock, and link it as a child of
  // IDomNode
  DomTreeNode *C = new DomTreeNode(BB, IPDomNode);
  DomTreeNodes[BB] = C;
  return BBNode = IPDomNode->addChild(C);
}

//===----------------------------------------------------------------------===//
//  PostDominanceFrontier Implementation
//===----------------------------------------------------------------------===//

static RegisterPass<PostDominanceFrontier>
H("postdomfrontier", "Post-Dominance Frontier Construction", true);

const DominanceFrontier::DomSetType &
PostDominanceFrontier::calculate(const PostDominatorTree &DT,
                                 const DomTreeNode *Node) {
  // Loop over CFG successors to calculate DFlocal[Node]
  BasicBlock *BB = Node->getBlock();
  DomSetType &S = Frontiers[BB];       // The new set to fill in...
  if (getRoots().empty()) return S;

  if (BB)
    for (pred_iterator SI = pred_begin(BB), SE = pred_end(BB);
         SI != SE; ++SI) {
      // Does Node immediately dominate this predecessor?
      DomTreeNode *SINode = DT[*SI];
      if (SINode && SINode->getIDom() != Node)
        S.insert(*SI);
    }

  // At this point, S is DFlocal.  Now we union in DFup's of our children...
  // Loop through and visit the nodes that Node immediately dominates (Node's
  // children in the IDomTree)
  //
  for (DomTreeNode::const_iterator
         NI = Node->begin(), NE = Node->end(); NI != NE; ++NI) {
    DomTreeNode *IDominee = *NI;
    const DomSetType &ChildDF = calculate(DT, IDominee);

    DomSetType::const_iterator CDFI = ChildDF.begin(), CDFE = ChildDF.end();
    for (; CDFI != CDFE; ++CDFI) {
      if (!DT.properlyDominates(Node, DT[*CDFI]))
        S.insert(*CDFI);
    }
  }

  return S;
}

// Ensure that this .cpp file gets linked when PostDominators.h is used.
DEFINING_FILE_FOR(PostDominanceFrontier)
