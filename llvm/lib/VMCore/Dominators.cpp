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
#include <algorithm>
#include <iostream>
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

/// dominates - Return true if A dominates B.
///
bool ImmediateDominatorsBase::dominates(BasicBlock *A, BasicBlock *B) const {
  assert(A && B && "Null pointers?");
  
  // Walk up the dominator tree from B to determine if A dom B.
  while (A != B && B)
    B = get(B);
  return A == B;
}

void ImmediateDominatorsBase::print(std::ostream &o, const Module* ) const {
  Function *F = getRoots()[0]->getParent();
  for (Function::iterator I = F->begin(), E = F->end(); I != E; ++I) {
    o << "  Immediate Dominator For Basic Block:";
    WriteAsOperand(o, I, false);
    o << " is:";
    if (BasicBlock *ID = get(I))
      WriteAsOperand(o, ID, false);
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

  if(!IsPostDominators) {
    // A dominates B if it is found first in the basic block.
    return &*I == A;
  } else {
    // A post-dominates B if B is found first in the basic block.
    return &*I == B;
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

  ImmediateDominators &ID = getAnalysis<ImmediateDominators>();
  Doms.clear();
  if (Roots.empty()) return false;

  // Root nodes only dominate themselves.
  for (unsigned i = 0, e = Roots.size(); i != e; ++i)
    Doms[Roots[i]].insert(Roots[i]);

  // Loop over all of the blocks in the function, calculating dominator sets for
  // each function.
  for (Function::iterator I = F.begin(), E = F.end(); I != E; ++I)
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

void DominatorSetBase::print(std::ostream &o, const Module* ) const {
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

  Function *F = Root->getParent();
  // Loop over all of the reachable blocks in the function...
  for (Function::iterator I = F->begin(), E = F->end(); I != E; ++I)
    if (BasicBlock *ImmDom = ID.get(I)) {  // Reachable block.
      Node *&BBNode = Nodes[I];
      if (!BBNode) {  // Haven't calculated this node yet?
        // Get or calculate the node for the immediate dominator
        Node *IDomNode = getNodeForBlock(ImmDom);

        // Add a new tree node for this BasicBlock, and link it as a child of
        // IDomNode
        BBNode = IDomNode->addChild(new Node(I, IDomNode));
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

void DominatorTreeBase::print(std::ostream &o, const Module* ) const {
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
      if (!Node->properlyDominates(DT[*CDFI]))
        S.insert(*CDFI);
    }
  }

  return S;
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

//===----------------------------------------------------------------------===//
// ETOccurrence Implementation
//===----------------------------------------------------------------------===//

void ETOccurrence::Splay() {
  ETOccurrence *father;
  ETOccurrence *grandfather;
  int occdepth;
  int fatherdepth;
  
  while (Parent) {
    occdepth = Depth;
    
    father = Parent;
    fatherdepth = Parent->Depth;
    grandfather = father->Parent;
    
    // If we have no grandparent, a single zig or zag will do.
    if (!grandfather) {
      setDepthAdd(fatherdepth);
      MinOccurrence = father->MinOccurrence;
      Min = father->Min;
      
      // See what we have to rotate
      if (father->Left == this) {
        // Zig
        father->setLeft(Right);
        setRight(father);
        if (father->Left)
          father->Left->setDepthAdd(occdepth);
      } else {
        // Zag
        father->setRight(Left);
        setLeft(father);
        if (father->Right)
          father->Right->setDepthAdd(occdepth);
      }
      father->setDepth(-occdepth);
      Parent = NULL;
      
      father->recomputeMin();
      return;
    }
    
    // If we have a grandfather, we need to do some
    // combination of zig and zag.
    int grandfatherdepth = grandfather->Depth;
    
    setDepthAdd(fatherdepth + grandfatherdepth);
    MinOccurrence = grandfather->MinOccurrence;
    Min = grandfather->Min;
    
    ETOccurrence *greatgrandfather = grandfather->Parent;
    
    if (grandfather->Left == father) {
      if (father->Left == this) {
        // Zig zig
        grandfather->setLeft(father->Right);
        father->setLeft(Right);
        setRight(father);
        father->setRight(grandfather);
        
        father->setDepth(-occdepth);
        
        if (father->Left)
          father->Left->setDepthAdd(occdepth);
        
        grandfather->setDepth(-fatherdepth);
        if (grandfather->Left)
          grandfather->Left->setDepthAdd(fatherdepth);
      } else {
        // Zag zig
        grandfather->setLeft(Right);
        father->setRight(Left);
        setLeft(father);
        setRight(grandfather);
        
        father->setDepth(-occdepth);
        if (father->Right)
          father->Right->setDepthAdd(occdepth);
        grandfather->setDepth(-occdepth - fatherdepth);
        if (grandfather->Left)
          grandfather->Left->setDepthAdd(occdepth + fatherdepth);
      }
    } else {
      if (father->Left == this) {
        // Zig zag
        grandfather->setRight(Left);
        father->setLeft(Right);
        setLeft(grandfather);
        setRight(father);
        
        father->setDepth(-occdepth);
        if (father->Left)
          father->Left->setDepthAdd(occdepth);
        grandfather->setDepth(-occdepth - fatherdepth);
        if (grandfather->Right)
          grandfather->Right->setDepthAdd(occdepth + fatherdepth);
      } else {              // Zag Zag
        grandfather->setRight(father->Left);
        father->setRight(Left);
        setLeft(father);
        father->setLeft(grandfather);
        
        father->setDepth(-occdepth);
        if (father->Right)
          father->Right->setDepthAdd(occdepth);
        grandfather->setDepth(-fatherdepth);
        if (grandfather->Right)
          grandfather->Right->setDepthAdd(fatherdepth);
      }
    }
    
    // Might need one more rotate depending on greatgrandfather.
    setParent(greatgrandfather);
    if (greatgrandfather) {
      if (greatgrandfather->Left == grandfather)
        greatgrandfather->Left = this;
      else
        greatgrandfather->Right = this;
      
    }
    grandfather->recomputeMin();
    father->recomputeMin();
  }
}

//===----------------------------------------------------------------------===//
// ETNode implementation
//===----------------------------------------------------------------------===//

void ETNode::Split() {
  ETOccurrence *right, *left;
  ETOccurrence *rightmost = RightmostOcc;
  ETOccurrence *parent;

  // Update the occurrence tree first.
  RightmostOcc->Splay();

  // Find the leftmost occurrence in the rightmost subtree, then splay
  // around it.
  for (right = rightmost->Right; right->Left; right = right->Left);

  right->Splay();

  // Start splitting
  right->Left->Parent = NULL;
  parent = ParentOcc;
  parent->Splay();
  ParentOcc = NULL;

  left = parent->Left;
  parent->Right->Parent = NULL;

  right->setLeft(left);

  right->recomputeMin();

  rightmost->Splay();
  rightmost->Depth = 0;
  rightmost->Min = 0;

  delete parent;

  // Now update *our* tree

  if (Father->Son == this)
    Father->Son = Right;

  if (Father->Son == this)
    Father->Son = NULL;
  else {
    Left->Right = Right;
    Right->Left = Left;
  }
  Left = Right = NULL;
  Father = NULL;
}

void ETNode::setFather(ETNode *NewFather) {
  ETOccurrence *rightmost;
  ETOccurrence *leftpart;
  ETOccurrence *NewFatherOcc;
  ETOccurrence *temp;

  // First update the path in the splay tree
  NewFatherOcc = new ETOccurrence(NewFather);

  rightmost = NewFather->RightmostOcc;
  rightmost->Splay();

  leftpart = rightmost->Left;

  temp = RightmostOcc;
  temp->Splay();

  NewFatherOcc->setLeft(leftpart);
  NewFatherOcc->setRight(temp);

  temp->Depth++;
  temp->Min++;
  NewFatherOcc->recomputeMin();

  rightmost->setLeft(NewFatherOcc);

  if (NewFatherOcc->Min + rightmost->Depth < rightmost->Min) {
    rightmost->Min = NewFatherOcc->Min + rightmost->Depth;
    rightmost->MinOccurrence = NewFatherOcc->MinOccurrence;
  }

  delete ParentOcc;
  ParentOcc = NewFatherOcc;

  // Update *our* tree
  ETNode *left;
  ETNode *right;

  Father = NewFather;
  right = Father->Son;

  if (right)
    left = right->Left;
  else
    left = right = this;

  left->Right = this;
  right->Left = this;
  Left = left;
  Right = right;

  Father->Son = this;
}

bool ETNode::Below(ETNode *other) {
  ETOccurrence *up = other->RightmostOcc;
  ETOccurrence *down = RightmostOcc;

  if (this == other)
    return true;

  up->Splay();

  ETOccurrence *left, *right;
  left = up->Left;
  right = up->Right;

  if (!left)
    return false;

  left->Parent = NULL;

  if (right)
    right->Parent = NULL;

  down->Splay();

  if (left == down || left->Parent != NULL) {
    if (right)
      right->Parent = up;
    up->setLeft(down);
  } else {
    left->Parent = up;

    // If the two occurrences are in different trees, put things
    // back the way they were.
    if (right && right->Parent != NULL)
      up->setRight(down);
    else
      up->setRight(right);
    return false;
  }

  if (down->Depth <= 0)
    return false;

  return !down->Right || down->Right->Min + down->Depth >= 0;
}

ETNode *ETNode::NCA(ETNode *other) {
  ETOccurrence *occ1 = RightmostOcc;
  ETOccurrence *occ2 = other->RightmostOcc;
  
  ETOccurrence *left, *right, *ret;
  ETOccurrence *occmin;
  int mindepth;
  
  if (this == other)
    return this;
  
  occ1->Splay();
  left = occ1->Left;
  right = occ1->Right;
  
  if (left)
    left->Parent = NULL;
  
  if (right)
    right->Parent = NULL;
  occ2->Splay();

  if (left == occ2 || (left && left->Parent != NULL)) {
    ret = occ2->Right;
    
    occ1->setLeft(occ2);
    if (right)
      right->Parent = occ1;
  } else {
    ret = occ2->Left;
    
    occ1->setRight(occ2);
    if (left)
      left->Parent = occ1;
  }

  if (occ2->Depth > 0) {
    occmin = occ1;
    mindepth = occ1->Depth;
  } else {
    occmin = occ2;
    mindepth = occ2->Depth + occ1->Depth;
  }
  
  if (ret && ret->Min + occ1->Depth + occ2->Depth < mindepth)
    return ret->MinOccurrence->OccFor;
  else
    return occmin->OccFor;
}

//===----------------------------------------------------------------------===//
// ETForest implementation
//===----------------------------------------------------------------------===//

static RegisterAnalysis<ETForest>
D("etforest", "ET Forest Construction", true);

void ETForestBase::reset() {
  for (ETMapType::iterator I = Nodes.begin(), E = Nodes.end(); I != E; ++I)
    delete I->second;
  Nodes.clear();
}

void ETForestBase::updateDFSNumbers()
{
  int dfsnum = 0;
  // Iterate over all nodes in depth first order.
  for (unsigned i = 0, e = Roots.size(); i != e; ++i)
    for (df_iterator<BasicBlock*> I = df_begin(Roots[i]),
           E = df_end(Roots[i]); I != E; ++I) {
      BasicBlock *BB = *I;
      if (!getNode(BB)->hasFather())
        getNode(BB)->assignDFSNumber(dfsnum);    
  }
  SlowQueries = 0;
  DFSInfoValid = true;
}

ETNode *ETForest::getNodeForBlock(BasicBlock *BB) {
  ETNode *&BBNode = Nodes[BB];
  if (BBNode) return BBNode;

  // Haven't calculated this node yet?  Get or calculate the node for the
  // immediate dominator.
  BasicBlock *IDom = getAnalysis<ImmediateDominators>()[BB];

  // If we are unreachable, we may not have an immediate dominator.
  if (!IDom)
    return BBNode = new ETNode(BB);
  else {
    ETNode *IDomNode = getNodeForBlock(IDom);
    
    // Add a new tree node for this BasicBlock, and link it as a child of
    // IDomNode
    BBNode = new ETNode(BB);
    BBNode->setFather(IDomNode);
    return BBNode;
  }
}

void ETForest::calculate(const ImmediateDominators &ID) {
  assert(Roots.size() == 1 && "ETForest should have 1 root block!");
  BasicBlock *Root = Roots[0];
  Nodes[Root] = new ETNode(Root); // Add a node for the root

  Function *F = Root->getParent();
  // Loop over all of the reachable blocks in the function...
  for (Function::iterator I = F->begin(), E = F->end(); I != E; ++I)
    if (BasicBlock *ImmDom = ID.get(I)) {  // Reachable block.
      ETNode *&BBNode = Nodes[I];
      if (!BBNode) {  // Haven't calculated this node yet?
        // Get or calculate the node for the immediate dominator
        ETNode *IDomNode =  getNodeForBlock(ImmDom);

        // Add a new ETNode for this BasicBlock, and set it's parent
        // to it's immediate dominator.
        BBNode = new ETNode(I);
        BBNode->setFather(IDomNode);
      }
    }

  // Make sure we've got nodes around for every block
  for (Function::iterator I = F->begin(), E = F->end(); I != E; ++I) {
    ETNode *&BBNode = Nodes[I];
    if (!BBNode)
      BBNode = new ETNode(I);
  }

  updateDFSNumbers ();
}

//===----------------------------------------------------------------------===//
// ETForestBase Implementation
//===----------------------------------------------------------------------===//

void ETForestBase::addNewBlock(BasicBlock *BB, BasicBlock *IDom) {
  ETNode *&BBNode = Nodes[BB];
  assert(!BBNode && "BasicBlock already in ET-Forest");

  BBNode = new ETNode(BB);
  BBNode->setFather(getNode(IDom));
  DFSInfoValid = false;
}

void ETForestBase::setImmediateDominator(BasicBlock *BB, BasicBlock *newIDom) {
  assert(getNode(BB) && "BasicBlock not in ET-Forest");
  assert(getNode(newIDom) && "IDom not in ET-Forest");
  
  ETNode *Node = getNode(BB);
  if (Node->hasFather()) {
    if (Node->getFather()->getData<BasicBlock>() == newIDom)
      return;
    Node->Split();
  }
  Node->setFather(getNode(newIDom));
  DFSInfoValid= false;
}

void ETForestBase::print(std::ostream &o, const Module *) const {
  o << "=============================--------------------------------\n";
  o << "ET Forest:\n";
  o << "DFS Info ";
  if (DFSInfoValid)
    o << "is";
  else
    o << "is not";
  o << " up to date\n";

  Function *F = getRoots()[0]->getParent();
  for (Function::iterator I = F->begin(), E = F->end(); I != E; ++I) {
    o << "  DFS Numbers For Basic Block:";
    WriteAsOperand(o, I, false);
    o << " are:";
    if (ETNode *EN = getNode(I)) {
      o << "In: " << EN->getDFSNumIn();
      o << " Out: " << EN->getDFSNumOut() << "\n";
    } else {
      o << "No associated ETNode";
    }
    o << "\n";
  }
  o << "\n";
}

DEFINING_FILE_FOR(DominatorSet)
