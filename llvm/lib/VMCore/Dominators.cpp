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
#include "llvm/Instructions.h"
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

unsigned DominatorTree::DFSPass(BasicBlock *V, InfoRec &VInfo,
                                      unsigned N) {
  // This is more understandable as a recursive algorithm, but we can't use the
  // recursive algorithm due to stack depth issues.  Keep it here for
  // documentation purposes.
#if 0
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
      N = DFSPass(*SI, SuccVInfo, N);
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
  std::set<BasicBlock *> Visited;
  InfoRec &VInInfo = Info[VIn];
  BasicBlock *VInAncestor = VInInfo.Ancestor;
  InfoRec &VInVAInfo = Info[VInAncestor];

  if (VInVAInfo.Ancestor != 0)
    Work.push_back(VIn);
  
  while (!Work.empty()) {
    BasicBlock *V = Work.back();
    InfoRec &VInfo = Info[V];
    BasicBlock *VAncestor = VInfo.Ancestor;
    InfoRec &VAInfo = Info[VAncestor];

    // Process Ancestor first
    if (Visited.count(VAncestor) == 0 && VAInfo.Ancestor != 0) {
      Work.push_back(VAncestor);
      Visited.insert(VAncestor);
      continue;
    } 
    Work.pop_back(); 

    // Update VINfo based on Ancestor info
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

void DominatorTree::calculate(Function& F) {
  BasicBlock* Root = Roots[0];
  
  Nodes[Root] = RootNode = new Node(Root, 0); // Add a node for the root...

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

  // Loop over all of the reachable blocks in the function...
  for (Function::iterator I = F.begin(), E = F.end(); I != E; ++I)
    if (BasicBlock *ImmDom = getIDom(I)) {  // Reachable block.
      Node *&BBNode = Nodes[I];
      if (!BBNode) {  // Haven't calculated this node yet?
        // Get or calculate the node for the immediate dominator
        Node *IDomNode = getNodeForBlock(ImmDom);

        // Add a new tree node for this BasicBlock, and link it as a child of
        // IDomNode
        BBNode = IDomNode->addChild(new Node(I, IDomNode));
      }
    }

  // Free temporary memory used to construct idom's
  Info.clear();
  IDoms.clear();
  std::vector<BasicBlock*>().swap(Vertex);
}

// DominatorTreeBase::reset - Free all of the tree node memory.
//
void DominatorTreeBase::reset() {
  for (NodeMapType::iterator I = Nodes.begin(), E = Nodes.end(); I != E; ++I)
    delete I->second;
  Nodes.clear();
  IDoms.clear();
  Roots.clear();
  Vertex.clear();
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
  BasicBlock *IDom = getIDom(BB);
  Node *IDomNode = getNodeForBlock(IDom);

  // Add a new tree node for this BasicBlock, and link it as a child of
  // IDomNode
  return BBNode = IDomNode->addChild(new Node(BB, IDomNode));
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

namespace {
  class DFCalculateWorkObject {
  public:
    DFCalculateWorkObject(BasicBlock *B, BasicBlock *P, 
                          const DominatorTree::Node *N,
                          const DominatorTree::Node *PN)
    : currentBB(B), parentBB(P), Node(N), parentNode(PN) {}
    BasicBlock *currentBB;
    BasicBlock *parentBB;
    const DominatorTree::Node *Node;
    const DominatorTree::Node *parentNode;
  };
}

const DominanceFrontier::DomSetType &
DominanceFrontier::calculate(const DominatorTree &DT,
                             const DominatorTree::Node *Node) {
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
    const DominatorTree::Node *currentNode = currentW->Node;
    const DominatorTree::Node *parentNode = currentW->parentNode;
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
    for (DominatorTree::Node::const_iterator NI = currentNode->begin(), 
           NE = currentNode->end(); NI != NE; ++NI) {
      DominatorTree::Node *IDominee = *NI;
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
        if (!parentNode->properlyDominates(DT[*CDFI]))
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

void ETNode::assignDFSNumber(int num) {
  std::vector<ETNode *>  workStack;
  std::set<ETNode *> visitedNodes;
  
  workStack.push_back(this);
  visitedNodes.insert(this);
  this->DFSNumIn = num++;

  while (!workStack.empty()) {
    ETNode  *Node = workStack.back();
    
    // If this is leaf node then set DFSNumOut and pop the stack
    if (!Node->Son) {
      Node->DFSNumOut = num++;
      workStack.pop_back();
      continue;
    }
    
    ETNode *son = Node->Son;
    
    // Visit Node->Son first
    if (visitedNodes.count(son) == 0) {
      son->DFSNumIn = num++;
      workStack.push_back(son);
      visitedNodes.insert(son);
      continue;
    }
    
    bool visitChild = false;
    // Visit remaining children
    for (ETNode *s = son->Right;  s != son && !visitChild; s = s->Right) {
      if (visitedNodes.count(s) == 0) {
        visitChild = true;
        s->DFSNumIn = num++;
        workStack.push_back(s);
        visitedNodes.insert(s);
      }
    }
    
    if (!visitChild) {
      // If we reach here means all children are visited
      Node->DFSNumOut = num++;
      workStack.pop_back();
    }
  }
}

//===----------------------------------------------------------------------===//
// ETForest implementation
//===----------------------------------------------------------------------===//

char ETForest::ID = 0;
static RegisterPass<ETForest>
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
      ETNode *ETN = getNode(BB);
      if (ETN && !ETN->hasFather())
        ETN->assignDFSNumber(dfsnum);    
  }
  SlowQueries = 0;
  DFSInfoValid = true;
}

// dominates - Return true if A dominates B. THis performs the
// special checks necessary if A and B are in the same basic block.
bool ETForestBase::dominates(Instruction *A, Instruction *B) {
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

/// isReachableFromEntry - Return true if A is dominated by the entry
/// block of the function containing it.
const bool ETForestBase::isReachableFromEntry(BasicBlock* A) {
  return dominates(&A->getParent()->getEntryBlock(), A);
}

ETNode *ETForest::getNodeForBlock(BasicBlock *BB) {
  ETNode *&BBNode = Nodes[BB];
  if (BBNode) return BBNode;

  // Haven't calculated this node yet?  Get or calculate the node for the
  // immediate dominator.
  DominatorTree::Node *node= getAnalysis<DominatorTree>().getNode(BB);

  // If we are unreachable, we may not have an immediate dominator.
  if (!node || !node->getIDom())
    return BBNode = new ETNode(BB);
  else {
    ETNode *IDomNode = getNodeForBlock(node->getIDom()->getBlock());
    
    // Add a new tree node for this BasicBlock, and link it as a child of
    // IDomNode
    BBNode = new ETNode(BB);
    BBNode->setFather(IDomNode);
    return BBNode;
  }
}

void ETForest::calculate(const DominatorTree &DT) {
  assert(Roots.size() == 1 && "ETForest should have 1 root block!");
  BasicBlock *Root = Roots[0];
  Nodes[Root] = new ETNode(Root); // Add a node for the root

  Function *F = Root->getParent();
  // Loop over all of the reachable blocks in the function...
  for (Function::iterator I = F->begin(), E = F->end(); I != E; ++I) {
    DominatorTree::Node* node = DT.getNode(I);
    if (node && node->getIDom()) {  // Reachable block.
      BasicBlock* ImmDom = node->getIDom()->getBlock();
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
