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
#include "Support/DepthFirstIterator.h"
#include "Support/SetOperations.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
//  ImmediateDominators Implementation
//===----------------------------------------------------------------------===//
//
// Immediate Dominators construction - This pass constructs immediate dominator
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

static RegisterAnalysis<ImmediateDominators>
C("idom", "Immediate Dominators Construction", true);

unsigned ImmediateDominators::DFSPass(BasicBlock *V, InfoRec &VInfo,
                                      unsigned N) {
  VInfo.Semi = ++N;
  VInfo.Label = V;

  Vertex.push_back(V);        // Vertex[n] = V;
  //Info[V].Ancestor = 0;     // Ancestor[n] = 0
  //Child[V] = 0;             // Child[v] = 0
  VInfo.Size = 1;             // Size[v] = 1

  for (succ_iterator SI = succ_begin(V), E = succ_end(V); SI != E; ++SI) {
    InfoRec &SuccVInfo = Info[*SI];
    if (SuccVInfo.Semi == 0) {
      SuccVInfo.Parent = V;
      N = DFSPass(*SI, SuccVInfo, N);
    }
  }
  return N;
}

void ImmediateDominators::Compress(BasicBlock *V, InfoRec &VInfo) {
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

BasicBlock *ImmediateDominators::Eval(BasicBlock *V) {
  InfoRec &VInfo = Info[V];
#if !BALANCE_IDOM_TREE
  // Higher-complexity but faster implementation
  if (VInfo.Ancestor == 0)
    return V;
  Compress(V, VInfo);
  return VInfo.Label;
#else
  // Lower-complexity but slower implementation
  if (VInfo.Ancestor == 0)
    return VInfo.Label;
  Compress(V, VInfo);
  BasicBlock *VLabel = VInfo.Label;

  BasicBlock *VAncestorLabel = Info[VInfo.Ancestor].Label;
  if (Info[VAncestorLabel].Semi >= Info[VLabel].Semi)
    return VLabel;
  else
    return VAncestorLabel;
#endif
}

void ImmediateDominators::Link(BasicBlock *V, BasicBlock *W, InfoRec &WInfo){
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



bool ImmediateDominators::runOnFunction(Function &F) {
  IDoms.clear();     // Reset from the last time we were run...
  BasicBlock *Root = &F.getEntryBlock();
  Roots.clear();
  Roots.push_back(Root);

  Vertex.push_back(0);
  
  // Step #1: Number blocks in depth-first order and initialize variables used
  // in later stages of the algorithm.
  unsigned N = 0;
  for (unsigned i = 0, e = Roots.size(); i != e; ++i)
    N = DFSPass(Roots[i], Info[Roots[i]], 0);

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

  // Free temporary memory used to construct idom's
  Info.clear();
  std::vector<BasicBlock*>().swap(Vertex);

  return false;
}

void ImmediateDominatorsBase::print(std::ostream &o) const {
  for (const_iterator I = begin(), E = end(); I != E; ++I) {
    o << "  Immediate Dominator For Basic Block:";
    if (I->first)
      WriteAsOperand(o, I->first, false);
    else
      o << " <<exit node>>";
    o << " is:";
    if (I->second)
      WriteAsOperand(o, I->second, false);
    else
      o << " <<exit node>>";
    o << "\n";
  }
  o << "\n";
}



//===----------------------------------------------------------------------===//
//  DominatorSet Implementation
//===----------------------------------------------------------------------===//

static RegisterAnalysis<DominatorSet>
B("domset", "Dominator Set Construction", true);

// dominates - Return true if A dominates B.  This performs the special checks
// necessary if A and B are in the same basic block.
//
bool DominatorSetBase::dominates(Instruction *A, Instruction *B) const {
  BasicBlock *BBA = A->getParent(), *BBB = B->getParent();
  if (BBA != BBB) return dominates(BBA, BBB);
  
  // Loop through the basic block until we find A or B.
  BasicBlock::iterator I = BBA->begin();
  for (; &*I != A && &*I != B; ++I) /*empty*/;
  
  // A dominates B if it is found first in the basic block...
  return &*I == A;
}


void DominatorSet::recalculate() {
  ImmediateDominators &ID = getAnalysis<ImmediateDominators>();
  Doms.clear();
  if (Roots.empty()) return;

  // Root nodes only dominate themselves.
  for (unsigned i = 0, e = Roots.size(); i != e; ++i)
    Doms[Roots[i]].insert(Roots[i]);

  Function *F = Roots.back()->getParent();

  // Loop over all of the blocks in the function, calculating dominator sets for
  // each function.
  for (Function::iterator I = F->begin(), E = F->end(); I != E; ++I)
    if (BasicBlock *IDom = ID[I]) {   // Get idom if block is reachable
      DomSetType &DS = Doms[I];
      assert(DS.empty() && "Domset already filled in for this block?");
      DS.insert(I);  // Blocks always dominate themselves
      
      // Insert all dominators into the set... 
      while (IDom) {
        // If we have already computed the dominator sets for our immediate
        // dominator, just use it instead of walking all the way up to the root.
        DomSetType &IDS = Doms[IDom];
        if (!IDS.empty()) {
          DS.insert(IDS.begin(), IDS.end());
          break;
        } else {
          DS.insert(IDom);
          IDom = ID[IDom];
        }
      }
    } else {
      // Ensure that every basic block has at least an empty set of nodes.  This
      // is important for the case when there is unreachable blocks.
      Doms[I];
    }
}

// runOnFunction - This method calculates the forward dominator sets for the
// specified function.
//
bool DominatorSet::runOnFunction(Function &F) {
  BasicBlock *Root = &F.getEntryBlock();
  Roots.clear();
  Roots.push_back(Root);
  assert(pred_begin(Root) == pred_end(Root) &&
	 "Root node has predecessors in function!");
  recalculate();
  return false;
}

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

void DominatorSetBase::print(std::ostream &o) const {
  for (const_iterator I = begin(), E = end(); I != E; ++I) {
    o << "  DomSet For BB: ";
    if (I->first)
      WriteAsOperand(o, I->first, false);
    else
      o << " <<exit node>>";
    o << " is:\t" << I->second << "\n";
  }
}

//===----------------------------------------------------------------------===//
//  DominatorTree Implementation
//===----------------------------------------------------------------------===//

static RegisterAnalysis<DominatorTree>
E("domtree", "Dominator Tree Construction", true);

// DominatorTreeBase::reset - Free all of the tree node memory.
//
void DominatorTreeBase::reset() { 
  for (NodeMapType::iterator I = Nodes.begin(), E = Nodes.end(); I != E; ++I)
    delete I->second;
  Nodes.clear();
  RootNode = 0;
}

void DominatorTreeBase::Node::setIDom(Node *NewIDom) {
  assert(IDom && "No immediate dominator?");
  if (IDom != NewIDom) {
    std::vector<Node*>::iterator I =
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

DominatorTreeBase::Node *DominatorTree::getNodeForBlock(BasicBlock *BB) {
  Node *&BBNode = Nodes[BB];
  if (BBNode) return BBNode;

  // Haven't calculated this node yet?  Get or calculate the node for the
  // immediate dominator.
  BasicBlock *IDom = getAnalysis<ImmediateDominators>()[BB];
  Node *IDomNode = getNodeForBlock(IDom);
    
  // Add a new tree node for this BasicBlock, and link it as a child of
  // IDomNode
  return BBNode = IDomNode->addChild(new Node(BB, IDomNode));
}

void DominatorTree::calculate(const ImmediateDominators &ID) {
  assert(Roots.size() == 1 && "DominatorTree should have 1 root block!");
  BasicBlock *Root = Roots[0];
  Nodes[Root] = RootNode = new Node(Root, 0); // Add a node for the root...

  // Loop over all of the reachable blocks in the function...
  for (ImmediateDominators::const_iterator I = ID.begin(), E = ID.end();
       I != E; ++I) {
    Node *&BBNode = Nodes[I->first];
    if (!BBNode) {  // Haven't calculated this node yet?
      // Get or calculate the node for the immediate dominator
      Node *IDomNode = getNodeForBlock(I->second);

      // Add a new tree node for this BasicBlock, and link it as a child of
      // IDomNode
      BBNode = IDomNode->addChild(new Node(I->first, IDomNode));
    }
  }
}

static std::ostream &operator<<(std::ostream &o,
                                const DominatorTreeBase::Node *Node) {
  if (Node->getBlock())
    WriteAsOperand(o, Node->getBlock(), false);
  else
    o << " <<exit node>>";
  return o << "\n";
}

static void PrintDomTree(const DominatorTreeBase::Node *N, std::ostream &o,
                         unsigned Lev) {
  o << std::string(2*Lev, ' ') << "[" << Lev << "] " << N;
  for (DominatorTreeBase::Node::const_iterator I = N->begin(), E = N->end(); 
       I != E; ++I)
    PrintDomTree(*I, o, Lev+1);
}

void DominatorTreeBase::print(std::ostream &o) const {
  o << "=============================--------------------------------\n"
    << "Inorder Dominator Tree:\n";
  PrintDomTree(getRootNode(), o, 1);
}


//===----------------------------------------------------------------------===//
//  DominanceFrontier Implementation
//===----------------------------------------------------------------------===//

static RegisterAnalysis<DominanceFrontier>
G("domfrontier", "Dominance Frontier Construction", true);

const DominanceFrontier::DomSetType &
DominanceFrontier::calculate(const DominatorTree &DT, 
                             const DominatorTree::Node *Node) {
  // Loop over CFG successors to calculate DFlocal[Node]
  BasicBlock *BB = Node->getBlock();
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
    const DomSetType &ChildDF = calculate(DT, IDominee);

    DomSetType::const_iterator CDFI = ChildDF.begin(), CDFE = ChildDF.end();
    for (; CDFI != CDFE; ++CDFI) {
      if (!Node->dominates(DT[*CDFI]))
	S.insert(*CDFI);
    }
  }

  return S;
}

void DominanceFrontierBase::print(std::ostream &o) const {
  for (const_iterator I = begin(), E = end(); I != E; ++I) {
    o << "  DomFrontier for BB";
    if (I->first)
      WriteAsOperand(o, I->first, false);
    else
      o << " <<exit node>>";
    o << " is:\t" << I->second << "\n";
  }
}

