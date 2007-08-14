//===- Dominators.cpp - Dominator Calculation -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements simple dominator construction algorithms for finding
// forward dominators.  Postdominators are available in libanalysis, but are not
// included in libvmcore, because it's not needed.  Forward dominators are
// needed to support the Verifier pass.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/Dominators.h"
#include "llvm/Support/CFG.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Instructions.h"
#include "llvm/Support/Streams.h"
#include <algorithm>
using namespace llvm;

namespace llvm {
static std::ostream &operator<<(std::ostream &o,
                                const std::set<BasicBlock*> &BBs) {
  for (std::set<BasicBlock*>::const_iterator I = BBs.begin(), E = BBs.end();
       I != E; ++I)
    if (*I)
      WriteAsOperand(o, *I, false);
    else
      o << " <<exit node>>";
  return o;
}
}

//===----------------------------------------------------------------------===//
//  DominatorTree Implementation
//===----------------------------------------------------------------------===//
//
// DominatorTree construction - This pass constructs immediate dominator
// information for a flow-graph based on the algorithm described in this
// document:
//
//   A Fast Algorithm for Finding Dominators in a Flowgraph
//   T. Lengauer & R. Tarjan, ACM TOPLAS July 1979, pgs 121-141.
//
// This implements both the O(n*ack(n)) and the O(n*log(n)) versions of EVAL and
// LINK, but it turns out that the theoretically slower O(n*log(n))
// implementation is actually faster than the "efficient" algorithm (even for
// large CFGs) because the constant overheads are substantially smaller.  The
// lower-complexity version can be enabled with the following #define:
//
#define BALANCE_IDOM_TREE 0
//
//===----------------------------------------------------------------------===//

char DominatorTree::ID = 0;
static RegisterPass<DominatorTree>
E("domtree", "Dominator Tree Construction", true);

// NewBB is split and now it has one successor. Update dominator tree to
// reflect this change.
void DominatorTree::splitBlock(BasicBlock *NewBB) {
  assert(NewBB->getTerminator()->getNumSuccessors() == 1
         && "NewBB should have a single successor!");
  BasicBlock *NewBBSucc = NewBB->getTerminator()->getSuccessor(0);

  std::vector<BasicBlock*> PredBlocks;
  for (pred_iterator PI = pred_begin(NewBB), PE = pred_end(NewBB);
       PI != PE; ++PI)
      PredBlocks.push_back(*PI);  

  assert(!PredBlocks.empty() && "No predblocks??");

  // The newly inserted basic block will dominate existing basic blocks iff the
  // PredBlocks dominate all of the non-pred blocks.  If all predblocks dominate
  // the non-pred blocks, then they all must be the same block!
  //
  bool NewBBDominatesNewBBSucc = true;
  {
    BasicBlock *OnePred = PredBlocks[0];
    unsigned i = 1, e = PredBlocks.size();
    for (i = 1; !isReachableFromEntry(OnePred); ++i) {
      assert(i != e && "Didn't find reachable pred?");
      OnePred = PredBlocks[i];
    }
    
    for (; i != e; ++i)
      if (PredBlocks[i] != OnePred && isReachableFromEntry(OnePred)) {
        NewBBDominatesNewBBSucc = false;
        break;
      }

    if (NewBBDominatesNewBBSucc)
      for (pred_iterator PI = pred_begin(NewBBSucc), E = pred_end(NewBBSucc);
           PI != E; ++PI)
        if (*PI != NewBB && !dominates(NewBBSucc, *PI)) {
          NewBBDominatesNewBBSucc = false;
          break;
        }
  }

  // The other scenario where the new block can dominate its successors are when
  // all predecessors of NewBBSucc that are not NewBB are dominated by NewBBSucc
  // already.
  if (!NewBBDominatesNewBBSucc) {
    NewBBDominatesNewBBSucc = true;
    for (pred_iterator PI = pred_begin(NewBBSucc), E = pred_end(NewBBSucc);
         PI != E; ++PI)
      if (*PI != NewBB && !dominates(NewBBSucc, *PI)) {
        NewBBDominatesNewBBSucc = false;
        break;
      }
  }

  // Find NewBB's immediate dominator and create new dominator tree node for
  // NewBB.
  BasicBlock *NewBBIDom = 0;
  unsigned i = 0;
  for (i = 0; i < PredBlocks.size(); ++i)
    if (isReachableFromEntry(PredBlocks[i])) {
      NewBBIDom = PredBlocks[i];
      break;
    }
  assert(i != PredBlocks.size() && "No reachable preds?");
  for (i = i + 1; i < PredBlocks.size(); ++i) {
    if (isReachableFromEntry(PredBlocks[i]))
      NewBBIDom = findNearestCommonDominator(NewBBIDom, PredBlocks[i]);
  }
  assert(NewBBIDom && "No immediate dominator found??");
  
  // Create the new dominator tree node... and set the idom of NewBB.
  DomTreeNode *NewBBNode = addNewBlock(NewBB, NewBBIDom);
  
  // If NewBB strictly dominates other blocks, then it is now the immediate
  // dominator of NewBBSucc.  Update the dominator tree as appropriate.
  if (NewBBDominatesNewBBSucc) {
    DomTreeNode *NewBBSuccNode = getNode(NewBBSucc);
    changeImmediateDominator(NewBBSuccNode, NewBBNode);
  }
}

unsigned DominatorTree::DFSPass(BasicBlock *V, unsigned N) {
  // This is more understandable as a recursive algorithm, but we can't use the
  // recursive algorithm due to stack depth issues.  Keep it here for
  // documentation purposes.
#if 0
  InfoRec &VInfo = Info[Roots[i]];
  VInfo.Semi = ++N;
  VInfo.Label = V;

  Vertex.push_back(V);        // Vertex[n] = V;
  //Info[V].Ancestor = 0;     // Ancestor[n] = 0
  //Info[V].Child = 0;        // Child[v] = 0
  VInfo.Size = 1;             // Size[v] = 1

  for (succ_iterator SI = succ_begin(V), E = succ_end(V); SI != E; ++SI) {
    InfoRec &SuccVInfo = Info[*SI];
    if (SuccVInfo.Semi == 0) {
      SuccVInfo.Parent = V;
      N = DFSPass(*SI, N);
    }
  }
#else
  std::vector<std::pair<BasicBlock*, unsigned> > Worklist;
  Worklist.push_back(std::make_pair(V, 0U));
  while (!Worklist.empty()) {
    BasicBlock *BB = Worklist.back().first;
    unsigned NextSucc = Worklist.back().second;
    
    // First time we visited this BB?
    if (NextSucc == 0) {
      InfoRec &BBInfo = Info[BB];
      BBInfo.Semi = ++N;
      BBInfo.Label = BB;
      
      Vertex.push_back(BB);       // Vertex[n] = V;
      //BBInfo[V].Ancestor = 0;   // Ancestor[n] = 0
      //BBInfo[V].Child = 0;      // Child[v] = 0
      BBInfo.Size = 1;            // Size[v] = 1
    }
    
    // If we are done with this block, remove it from the worklist.
    if (NextSucc == BB->getTerminator()->getNumSuccessors()) {
      Worklist.pop_back();
      continue;
    }
    
    // Otherwise, increment the successor number for the next time we get to it.
    ++Worklist.back().second;
    
    // Visit the successor next, if it isn't already visited.
    BasicBlock *Succ = BB->getTerminator()->getSuccessor(NextSucc);
    
    InfoRec &SuccVInfo = Info[Succ];
    if (SuccVInfo.Semi == 0) {
      SuccVInfo.Parent = BB;
      Worklist.push_back(std::make_pair(Succ, 0U));
    }
  }
#endif
  return N;
}

void DominatorTree::Compress(BasicBlock *VIn) {

  std::vector<BasicBlock *> Work;
  SmallPtrSet<BasicBlock *, 32> Visited;
  BasicBlock *VInAncestor = Info[VIn].Ancestor;
  InfoRec &VInVAInfo = Info[VInAncestor];

  if (VInVAInfo.Ancestor != 0)
    Work.push_back(VIn);
  
  while (!Work.empty()) {
    BasicBlock *V = Work.back();
    InfoRec &VInfo = Info[V];
    BasicBlock *VAncestor = VInfo.Ancestor;
    InfoRec &VAInfo = Info[VAncestor];

    // Process Ancestor first
    if (Visited.insert(VAncestor) &&
        VAInfo.Ancestor != 0) {
      Work.push_back(VAncestor);
      continue;
    } 
    Work.pop_back(); 

    // Update VInfo based on Ancestor info
    if (VAInfo.Ancestor == 0)
      continue;
    BasicBlock *VAncestorLabel = VAInfo.Label;
    BasicBlock *VLabel = VInfo.Label;
    if (Info[VAncestorLabel].Semi < Info[VLabel].Semi)
      VInfo.Label = VAncestorLabel;
    VInfo.Ancestor = VAInfo.Ancestor;
  }
}

BasicBlock *DominatorTree::Eval(BasicBlock *V) {
  InfoRec &VInfo = Info[V];
#if !BALANCE_IDOM_TREE
  // Higher-complexity but faster implementation
  if (VInfo.Ancestor == 0)
    return V;
  Compress(V);
  return VInfo.Label;
#else
  // Lower-complexity but slower implementation
  if (VInfo.Ancestor == 0)
    return VInfo.Label;
  Compress(V);
  BasicBlock *VLabel = VInfo.Label;

  BasicBlock *VAncestorLabel = Info[VInfo.Ancestor].Label;
  if (Info[VAncestorLabel].Semi >= Info[VLabel].Semi)
    return VLabel;
  else
    return VAncestorLabel;
#endif
}

void DominatorTree::Link(BasicBlock *V, BasicBlock *W, InfoRec &WInfo){
#if !BALANCE_IDOM_TREE
  // Higher-complexity but faster implementation
  WInfo.Ancestor = V;
#else
  // Lower-complexity but slower implementation
  BasicBlock *WLabel = WInfo.Label;
  unsigned WLabelSemi = Info[WLabel].Semi;
  BasicBlock *S = W;
  InfoRec *SInfo = &Info[S];

  BasicBlock *SChild = SInfo->Child;
  InfoRec *SChildInfo = &Info[SChild];

  while (WLabelSemi < Info[SChildInfo->Label].Semi) {
    BasicBlock *SChildChild = SChildInfo->Child;
    if (SInfo->Size+Info[SChildChild].Size >= 2*SChildInfo->Size) {
      SChildInfo->Ancestor = S;
      SInfo->Child = SChild = SChildChild;
      SChildInfo = &Info[SChild];
    } else {
      SChildInfo->Size = SInfo->Size;
      S = SInfo->Ancestor = SChild;
      SInfo = SChildInfo;
      SChild = SChildChild;
      SChildInfo = &Info[SChild];
    }
  }

  InfoRec &VInfo = Info[V];
  SInfo->Label = WLabel;

  assert(V != W && "The optimization here will not work in this case!");
  unsigned WSize = WInfo.Size;
  unsigned VSize = (VInfo.Size += WSize);

  if (VSize < 2*WSize)
    std::swap(S, VInfo.Child);

  while (S) {
    SInfo = &Info[S];
    SInfo->Ancestor = V;
    S = SInfo->Child;
  }
#endif
}

void DominatorTree::calculate(Function &F) {
  BasicBlock* Root = Roots[0];

  // Add a node for the root...
  DomTreeNodes[Root] = RootNode = new DomTreeNode(Root, 0);

  Vertex.push_back(0);

  // Step #1: Number blocks in depth-first order and initialize variables used
  // in later stages of the algorithm.
  unsigned N = DFSPass(Root, 0);

  for (unsigned i = N; i >= 2; --i) {
    BasicBlock *W = Vertex[i];
    InfoRec &WInfo = Info[W];

    // Step #2: Calculate the semidominators of all vertices
    for (pred_iterator PI = pred_begin(W), E = pred_end(W); PI != E; ++PI)
      if (Info.count(*PI)) {  // Only if this predecessor is reachable!
        unsigned SemiU = Info[Eval(*PI)].Semi;
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

  // Loop over all of the reachable blocks in the function...
  for (Function::iterator I = F.begin(), E = F.end(); I != E; ++I)
    if (BasicBlock *ImmDom = getIDom(I)) {  // Reachable block.
      DomTreeNode *BBNode = DomTreeNodes[I];
      if (BBNode) continue;  // Haven't calculated this node yet?

      // Get or calculate the node for the immediate dominator
      DomTreeNode *IDomNode = getNodeForBlock(ImmDom);

      // Add a new tree node for this BasicBlock, and link it as a child of
      // IDomNode
      DomTreeNode *C = new DomTreeNode(I, IDomNode);
      DomTreeNodes[I] = IDomNode->addChild(C);
    }

  // Free temporary memory used to construct idom's
  Info.clear();
  IDoms.clear();
  std::vector<BasicBlock*>().swap(Vertex);

  updateDFSNumbers();
}

void DominatorTreeBase::updateDFSNumbers() {
  unsigned DFSNum = 0;

  SmallVector<std::pair<DomTreeNode*, DomTreeNode::iterator>, 32> WorkStack;
  
  for (unsigned i = 0, e = Roots.size(); i != e; ++i) {
    DomTreeNode *ThisRoot = getNode(Roots[i]);
    WorkStack.push_back(std::make_pair(ThisRoot, ThisRoot->begin()));
    ThisRoot->DFSNumIn = DFSNum++;
    
    while (!WorkStack.empty()) {
      DomTreeNode *Node = WorkStack.back().first;
      DomTreeNode::iterator ChildIt = WorkStack.back().second;

      // If we visited all of the children of this node, "recurse" back up the
      // stack setting the DFOutNum.
      if (ChildIt == Node->end()) {
        Node->DFSNumOut = DFSNum++;
        WorkStack.pop_back();
      } else {
        // Otherwise, recursively visit this child.
        DomTreeNode *Child = *ChildIt;
        ++WorkStack.back().second;
        
        WorkStack.push_back(std::make_pair(Child, Child->begin()));
        Child->DFSNumIn = DFSNum++;
      }
    }
  }
  
  SlowQueries = 0;
  DFSInfoValid = true;
}

/// isReachableFromEntry - Return true if A is dominated by the entry
/// block of the function containing it.
const bool DominatorTreeBase::isReachableFromEntry(BasicBlock* A) {
  assert (!isPostDominator() 
          && "This is not implemented for post dominators");
  return dominates(&A->getParent()->getEntryBlock(), A);
}

// dominates - Return true if A dominates B. THis performs the
// special checks necessary if A and B are in the same basic block.
bool DominatorTreeBase::dominates(Instruction *A, Instruction *B) {
  BasicBlock *BBA = A->getParent(), *BBB = B->getParent();
  if (BBA != BBB) return dominates(BBA, BBB);
  
  // It is not possible to determine dominance between two PHI nodes 
  // based on their ordering.
  if (isa<PHINode>(A) && isa<PHINode>(B)) 
    return false;

  // Loop through the basic block until we find A or B.
  BasicBlock::iterator I = BBA->begin();
  for (; &*I != A && &*I != B; ++I) /*empty*/;
  
  if(!IsPostDominators) {
    // A dominates B if it is found first in the basic block.
    return &*I == A;
  } else {
    // A post-dominates B if B is found first in the basic block.
    return &*I == B;
  }
}

// DominatorTreeBase::reset - Free all of the tree node memory.
//
void DominatorTreeBase::reset() {
  for (DomTreeNodeMapType::iterator I = DomTreeNodes.begin(), 
         E = DomTreeNodes.end(); I != E; ++I)
    delete I->second;
  DomTreeNodes.clear();
  IDoms.clear();
  Roots.clear();
  Vertex.clear();
  RootNode = 0;
}

/// findNearestCommonDominator - Find nearest common dominator basic block
/// for basic block A and B. If there is no such block then return NULL.
BasicBlock *DominatorTreeBase::findNearestCommonDominator(BasicBlock *A, 
                                                          BasicBlock *B) {

  assert (!isPostDominator() 
          && "This is not implemented for post dominators");
  assert (A->getParent() == B->getParent() 
          && "Two blocks are not in same function");

  // If either A or B is a entry block then it is nearest common dominator.
  BasicBlock &Entry  = A->getParent()->getEntryBlock();
  if (A == &Entry || B == &Entry)
    return &Entry;

  // If B dominates A then B is nearest common dominator.
  if (dominates(B, A))
    return B;

  // If A dominates B then A is nearest common dominator.
  if (dominates(A, B))
    return A;

  DomTreeNode *NodeA = getNode(A);
  DomTreeNode *NodeB = getNode(B);

  // Collect NodeA dominators set.
  SmallPtrSet<DomTreeNode*, 16> NodeADoms;
  NodeADoms.insert(NodeA);
  DomTreeNode *IDomA = NodeA->getIDom();
  while (IDomA) {
    NodeADoms.insert(IDomA);
    IDomA = IDomA->getIDom();
  }

  // Walk NodeB immediate dominators chain and find common dominator node.
  DomTreeNode *IDomB = NodeB->getIDom();
  while(IDomB) {
    if (NodeADoms.count(IDomB) != 0)
      return IDomB->getBlock();

    IDomB = IDomB->getIDom();
  }

  return NULL;
}

void DomTreeNode::setIDom(DomTreeNode *NewIDom) {
  assert(IDom && "No immediate dominator?");
  if (IDom != NewIDom) {
    std::vector<DomTreeNode*>::iterator I =
      std::find(IDom->Children.begin(), IDom->Children.end(), this);
    assert(I != IDom->Children.end() &&
           "Not in immediate dominator children set!");
    // I am no longer your child...
    IDom->Children.erase(I);

    // Switch to new dominator
    IDom = NewIDom;
    IDom->Children.push_back(this);
  }
}

DomTreeNode *DominatorTree::getNodeForBlock(BasicBlock *BB) {
  if (DomTreeNode *BBNode = DomTreeNodes[BB])
    return BBNode;

  // Haven't calculated this node yet?  Get or calculate the node for the
  // immediate dominator.
  BasicBlock *IDom = getIDom(BB);
  DomTreeNode *IDomNode = getNodeForBlock(IDom);

  // Add a new tree node for this BasicBlock, and link it as a child of
  // IDomNode
  DomTreeNode *C = new DomTreeNode(BB, IDomNode);
  return DomTreeNodes[BB] = IDomNode->addChild(C);
}

static std::ostream &operator<<(std::ostream &o, const DomTreeNode *Node) {
  if (Node->getBlock())
    WriteAsOperand(o, Node->getBlock(), false);
  else
    o << " <<exit node>>";
  
  o << " {" << Node->getDFSNumIn() << "," << Node->getDFSNumOut() << "}";
  
  return o << "\n";
}

static void PrintDomTree(const DomTreeNode *N, std::ostream &o,
                         unsigned Lev) {
  o << std::string(2*Lev, ' ') << "[" << Lev << "] " << N;
  for (DomTreeNode::const_iterator I = N->begin(), E = N->end();
       I != E; ++I)
    PrintDomTree(*I, o, Lev+1);
}

/// eraseNode - Removes a node from  the domiantor tree. Block must not
/// domiante any other blocks. Removes node from its immediate dominator's
/// children list. Deletes dominator node associated with basic block BB.
void DominatorTreeBase::eraseNode(BasicBlock *BB) {
  DomTreeNode *Node = getNode(BB);
  assert (Node && "Removing node that isn't in dominator tree.");
  assert (Node->getChildren().empty() && "Node is not a leaf node.");

    // Remove node from immediate dominator's children list.
  DomTreeNode *IDom = Node->getIDom();
  if (IDom) {
    std::vector<DomTreeNode*>::iterator I =
      std::find(IDom->Children.begin(), IDom->Children.end(), Node);
    assert(I != IDom->Children.end() &&
           "Not in immediate dominator children set!");
    // I am no longer your child...
    IDom->Children.erase(I);
  }
  
  DomTreeNodes.erase(BB);
  delete Node;
}

void DominatorTreeBase::print(std::ostream &o, const Module* ) const {
  o << "=============================--------------------------------\n";
  o << "Inorder Dominator Tree: ";
  if (DFSInfoValid)
    o << "DFSNumbers invalid: " << SlowQueries << " slow queries.";
  o << "\n";
  
  PrintDomTree(getRootNode(), o, 1);
}

void DominatorTreeBase::dump() {
  print(llvm::cerr);
}

bool DominatorTree::runOnFunction(Function &F) {
  reset();     // Reset from the last time we were run...
  Roots.push_back(&F.getEntryBlock());
  calculate(F);
  return false;
}

//===----------------------------------------------------------------------===//
//  DominanceFrontier Implementation
//===----------------------------------------------------------------------===//

char DominanceFrontier::ID = 0;
static RegisterPass<DominanceFrontier>
G("domfrontier", "Dominance Frontier Construction", true);

// NewBB is split and now it has one successor. Update dominace frontier to
// reflect this change.
void DominanceFrontier::splitBlock(BasicBlock *NewBB) {
  assert(NewBB->getTerminator()->getNumSuccessors() == 1
         && "NewBB should have a single successor!");
  BasicBlock *NewBBSucc = NewBB->getTerminator()->getSuccessor(0);

  std::vector<BasicBlock*> PredBlocks;
  for (pred_iterator PI = pred_begin(NewBB), PE = pred_end(NewBB);
       PI != PE; ++PI)
      PredBlocks.push_back(*PI);  

  if (PredBlocks.empty())
    // If NewBB does not have any predecessors then it is a entry block.
    // In this case, NewBB and its successor NewBBSucc dominates all
    // other blocks.
    return;

  // NewBBSucc inherits original NewBB frontier.
  DominanceFrontier::iterator NewBBI = find(NewBB);
  if (NewBBI != end()) {
    DominanceFrontier::DomSetType NewBBSet = NewBBI->second;
    DominanceFrontier::DomSetType NewBBSuccSet;
    NewBBSuccSet.insert(NewBBSet.begin(), NewBBSet.end());
    addBasicBlock(NewBBSucc, NewBBSuccSet);
  }

  // If NewBB dominates NewBBSucc, then DF(NewBB) is now going to be the
  // DF(PredBlocks[0]) without the stuff that the new block does not dominate
  // a predecessor of.
  DominatorTree &DT = getAnalysis<DominatorTree>();
  if (DT.dominates(NewBB, NewBBSucc)) {
    DominanceFrontier::iterator DFI = find(PredBlocks[0]);
    if (DFI != end()) {
      DominanceFrontier::DomSetType Set = DFI->second;
      // Filter out stuff in Set that we do not dominate a predecessor of.
      for (DominanceFrontier::DomSetType::iterator SetI = Set.begin(),
             E = Set.end(); SetI != E;) {
        bool DominatesPred = false;
        for (pred_iterator PI = pred_begin(*SetI), E = pred_end(*SetI);
             PI != E; ++PI)
          if (DT.dominates(NewBB, *PI))
            DominatesPred = true;
        if (!DominatesPred)
          Set.erase(SetI++);
        else
          ++SetI;
      }

      if (NewBBI != end()) {
        DominanceFrontier::DomSetType NewBBSet = NewBBI->second;
        NewBBSet.insert(Set.begin(), Set.end());
      } else 
        addBasicBlock(NewBB, Set);
    }
    
  } else {
    // DF(NewBB) is {NewBBSucc} because NewBB does not strictly dominate
    // NewBBSucc, but it does dominate itself (and there is an edge (NewBB ->
    // NewBBSucc)).  NewBBSucc is the single successor of NewBB.
    DominanceFrontier::DomSetType NewDFSet;
    NewDFSet.insert(NewBBSucc);
    addBasicBlock(NewBB, NewDFSet);
  }
  
  // Now we must loop over all of the dominance frontiers in the function,
  // replacing occurrences of NewBBSucc with NewBB in some cases.  All
  // blocks that dominate a block in PredBlocks and contained NewBBSucc in
  // their dominance frontier must be updated to contain NewBB instead.
  //
  for (Function::iterator FI = NewBB->getParent()->begin(),
         FE = NewBB->getParent()->end(); FI != FE; ++FI) {
    DominanceFrontier::iterator DFI = find(FI);
    if (DFI == end()) continue;  // unreachable block.
    
    // Only consider nodes that have NewBBSucc in their dominator frontier.
    if (!DFI->second.count(NewBBSucc)) continue;

    // Verify whether this block dominates a block in predblocks.  If not, do
    // not update it.
    bool BlockDominatesAny = false;
    for (std::vector<BasicBlock*>::const_iterator BI = PredBlocks.begin(), 
           BE = PredBlocks.end(); BI != BE; ++BI) {
      if (DT.dominates(FI, *BI)) {
        BlockDominatesAny = true;
        break;
      }
    }
    
    if (!BlockDominatesAny)
      continue;
    
    // If NewBBSucc should not stay in our dominator frontier, remove it.
    // We remove it unless there is a predecessor of NewBBSucc that we
    // dominate, but we don't strictly dominate NewBBSucc.
    bool ShouldRemove = true;
    if ((BasicBlock*)FI == NewBBSucc || !DT.dominates(FI, NewBBSucc)) {
      // Okay, we know that PredDom does not strictly dominate NewBBSucc.
      // Check to see if it dominates any predecessors of NewBBSucc.
      for (pred_iterator PI = pred_begin(NewBBSucc),
           E = pred_end(NewBBSucc); PI != E; ++PI)
        if (DT.dominates(FI, *PI)) {
          ShouldRemove = false;
          break;
        }
    }
    
    if (ShouldRemove)
      removeFromFrontier(DFI, NewBBSucc);
    addToFrontier(DFI, NewBB);
  }
}

namespace {
  class DFCalculateWorkObject {
  public:
    DFCalculateWorkObject(BasicBlock *B, BasicBlock *P, 
                          const DomTreeNode *N,
                          const DomTreeNode *PN)
    : currentBB(B), parentBB(P), Node(N), parentNode(PN) {}
    BasicBlock *currentBB;
    BasicBlock *parentBB;
    const DomTreeNode *Node;
    const DomTreeNode *parentNode;
  };
}

const DominanceFrontier::DomSetType &
DominanceFrontier::calculate(const DominatorTree &DT,
                             const DomTreeNode *Node) {
  BasicBlock *BB = Node->getBlock();
  DomSetType *Result = NULL;

  std::vector<DFCalculateWorkObject> workList;
  SmallPtrSet<BasicBlock *, 32> visited;

  workList.push_back(DFCalculateWorkObject(BB, NULL, Node, NULL));
  do {
    DFCalculateWorkObject *currentW = &workList.back();
    assert (currentW && "Missing work object.");

    BasicBlock *currentBB = currentW->currentBB;
    BasicBlock *parentBB = currentW->parentBB;
    const DomTreeNode *currentNode = currentW->Node;
    const DomTreeNode *parentNode = currentW->parentNode;
    assert (currentBB && "Invalid work object. Missing current Basic Block");
    assert (currentNode && "Invalid work object. Missing current Node");
    DomSetType &S = Frontiers[currentBB];

    // Visit each block only once.
    if (visited.count(currentBB) == 0) {
      visited.insert(currentBB);

      // Loop over CFG successors to calculate DFlocal[currentNode]
      for (succ_iterator SI = succ_begin(currentBB), SE = succ_end(currentBB);
           SI != SE; ++SI) {
        // Does Node immediately dominate this successor?
        if (DT[*SI]->getIDom() != currentNode)
          S.insert(*SI);
      }
    }

    // At this point, S is DFlocal.  Now we union in DFup's of our children...
    // Loop through and visit the nodes that Node immediately dominates (Node's
    // children in the IDomTree)
    bool visitChild = false;
    for (DomTreeNode::const_iterator NI = currentNode->begin(), 
           NE = currentNode->end(); NI != NE; ++NI) {
      DomTreeNode *IDominee = *NI;
      BasicBlock *childBB = IDominee->getBlock();
      if (visited.count(childBB) == 0) {
        workList.push_back(DFCalculateWorkObject(childBB, currentBB,
                                                 IDominee, currentNode));
        visitChild = true;
      }
    }

    // If all children are visited or there is any child then pop this block
    // from the workList.
    if (!visitChild) {

      if (!parentBB) {
        Result = &S;
        break;
      }

      DomSetType::const_iterator CDFI = S.begin(), CDFE = S.end();
      DomSetType &parentSet = Frontiers[parentBB];
      for (; CDFI != CDFE; ++CDFI) {
        if (!DT.properlyDominates(parentNode, DT[*CDFI]))
          parentSet.insert(*CDFI);
      }
      workList.pop_back();
    }

  } while (!workList.empty());

  return *Result;
}

void DominanceFrontierBase::print(std::ostream &o, const Module* ) const {
  for (const_iterator I = begin(), E = end(); I != E; ++I) {
    o << "  DomFrontier for BB";
    if (I->first)
      WriteAsOperand(o, I->first, false);
    else
      o << " <<exit node>>";
    o << " is:\t" << I->second << "\n";
  }
}

void DominanceFrontierBase::dump() {
  print (llvm::cerr);
}
