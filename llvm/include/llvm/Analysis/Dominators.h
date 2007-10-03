//===- llvm/Analysis/Dominators.h - Dominator Info Calculation --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the following classes:
//  1. DominatorTree: Represent dominators as an explicit tree structure.
//  2. DominanceFrontier: Calculate and hold the dominance frontier for a
//     function.
//
//  These data structures are listed in increasing order of complexity.  It
//  takes longer to calculate the dominator frontier, for example, than the
//  DominatorTree mapping.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_DOMINATORS_H
#define LLVM_ANALYSIS_DOMINATORS_H

#include "llvm/Pass.h"
#include <set>
#include "llvm/ADT/DenseMap.h"

namespace llvm {

class Instruction;

template <typename GraphType> struct GraphTraits;

//===----------------------------------------------------------------------===//
/// DominatorBase - Base class that other, more interesting dominator analyses
/// inherit from.
///
class DominatorBase : public FunctionPass {
protected:
  std::vector<BasicBlock*> Roots;
  const bool IsPostDominators;
  inline DominatorBase(intptr_t ID, bool isPostDom) : 
    FunctionPass(ID), Roots(), IsPostDominators(isPostDom) {}
public:

  /// getRoots -  Return the root blocks of the current CFG.  This may include
  /// multiple blocks if we are computing post dominators.  For forward
  /// dominators, this will always be a single block (the entry node).
  ///
  inline const std::vector<BasicBlock*> &getRoots() const { return Roots; }

  /// isPostDominator - Returns true if analysis based of postdoms
  ///
  bool isPostDominator() const { return IsPostDominators; }
};


//===----------------------------------------------------------------------===//
// DomTreeNode - Dominator Tree Node
class DominatorTreeBase;
class PostDominatorTree;
class DomTreeNode {
  BasicBlock *TheBB;
  DomTreeNode *IDom;
  std::vector<DomTreeNode*> Children;
  int DFSNumIn, DFSNumOut;

  friend class DominatorTreeBase;
  friend class PostDominatorTree;
public:
  typedef std::vector<DomTreeNode*>::iterator iterator;
  typedef std::vector<DomTreeNode*>::const_iterator const_iterator;
  
  iterator begin()             { return Children.begin(); }
  iterator end()               { return Children.end(); }
  const_iterator begin() const { return Children.begin(); }
  const_iterator end()   const { return Children.end(); }
  
  BasicBlock *getBlock() const { return TheBB; }
  DomTreeNode *getIDom() const { return IDom; }
  const std::vector<DomTreeNode*> &getChildren() const { return Children; }
  
  DomTreeNode(BasicBlock *BB, DomTreeNode *iDom)
    : TheBB(BB), IDom(iDom), DFSNumIn(-1), DFSNumOut(-1) { }
  DomTreeNode *addChild(DomTreeNode *C) { Children.push_back(C); return C; }
  void setIDom(DomTreeNode *NewIDom);

  
  /// getDFSNumIn/getDFSNumOut - These are an internal implementation detail, do
  /// not call them.
  unsigned getDFSNumIn() const { return DFSNumIn; }
  unsigned getDFSNumOut() const { return DFSNumOut; }
private:
  // Return true if this node is dominated by other. Use this only if DFS info
  // is valid.
  bool DominatedBy(const DomTreeNode *other) const {
    return this->DFSNumIn >= other->DFSNumIn &&
      this->DFSNumOut <= other->DFSNumOut;
  }
};

//===----------------------------------------------------------------------===//
/// DominatorTree - Calculate the immediate dominator tree for a function.
///
class DominatorTreeBase : public DominatorBase {
protected:
  void reset();
  typedef DenseMap<BasicBlock*, DomTreeNode*> DomTreeNodeMapType;
  DomTreeNodeMapType DomTreeNodes;
  DomTreeNode *RootNode;

  bool DFSInfoValid;
  unsigned int SlowQueries;
  // Information record used during immediate dominators computation.
  struct InfoRec {
    unsigned Semi;
    unsigned Size;
    BasicBlock *Label, *Parent, *Child, *Ancestor;

    std::vector<BasicBlock*> Bucket;

    InfoRec() : Semi(0), Size(0), Label(0), Parent(0), Child(0), Ancestor(0) {}
  };

  DenseMap<BasicBlock*, BasicBlock*> IDoms;

  // Vertex - Map the DFS number to the BasicBlock*
  std::vector<BasicBlock*> Vertex;

  // Info - Collection of information used during the computation of idoms.
  DenseMap<BasicBlock*, InfoRec> Info;

public:
  DominatorTreeBase(intptr_t ID, bool isPostDom) 
    : DominatorBase(ID, isPostDom), DFSInfoValid(false), SlowQueries(0) {}
  ~DominatorTreeBase() { reset(); }

  virtual void releaseMemory() { reset(); }

  /// getNode - return the (Post)DominatorTree node for the specified basic
  /// block.  This is the same as using operator[] on this class.
  ///
  inline DomTreeNode *getNode(BasicBlock *BB) const {
    DomTreeNodeMapType::const_iterator I = DomTreeNodes.find(BB);
    return I != DomTreeNodes.end() ? I->second : 0;
  }

  inline DomTreeNode *operator[](BasicBlock *BB) const {
    return getNode(BB);
  }

  /// getRootNode - This returns the entry node for the CFG of the function.  If
  /// this tree represents the post-dominance relations for a function, however,
  /// this root may be a node with the block == NULL.  This is the case when
  /// there are multiple exit nodes from a particular function.  Consumers of
  /// post-dominance information must be capable of dealing with this
  /// possibility.
  ///
  DomTreeNode *getRootNode() { return RootNode; }
  const DomTreeNode *getRootNode() const { return RootNode; }

  /// properlyDominates - Returns true iff this dominates N and this != N.
  /// Note that this is not a constant time operation!
  ///
  bool properlyDominates(const DomTreeNode *A, DomTreeNode *B) const {
    if (A == 0 || B == 0) return false;
    return dominatedBySlowTreeWalk(A, B);
  }

  inline bool properlyDominates(BasicBlock *A, BasicBlock *B) {
    return properlyDominates(getNode(A), getNode(B));
  }

  bool dominatedBySlowTreeWalk(const DomTreeNode *A, 
                               const DomTreeNode *B) const {
    const DomTreeNode *IDom;
    if (A == 0 || B == 0) return false;
    while ((IDom = B->getIDom()) != 0 && IDom != A && IDom != B)
      B = IDom;   // Walk up the tree
    return IDom != 0;
  }


  /// isReachableFromEntry - Return true if A is dominated by the entry
  /// block of the function containing it.
  const bool isReachableFromEntry(BasicBlock* A);
  
  /// dominates - Returns true iff A dominates B.  Note that this is not a
  /// constant time operation!
  ///
  inline bool dominates(const DomTreeNode *A, DomTreeNode *B) {
    if (B == A) 
      return true;  // A node trivially dominates itself.

    if (A == 0 || B == 0)
      return false;

    if (DFSInfoValid)
      return B->DominatedBy(A);

    // If we end up with too many slow queries, just update the
    // DFS numbers on the theory that we are going to keep querying.
    SlowQueries++;
    if (SlowQueries > 32) {
      updateDFSNumbers();
      return B->DominatedBy(A);
    }

    return dominatedBySlowTreeWalk(A, B);
  }

  inline bool dominates(BasicBlock *A, BasicBlock *B) {
    if (A == B) 
      return true;
    
    return dominates(getNode(A), getNode(B));
  }

  /// findNearestCommonDominator - Find nearest common dominator basic block
  /// for basic block A and B. If there is no such block then return NULL.
  BasicBlock *findNearestCommonDominator(BasicBlock *A, BasicBlock *B);

  // dominates - Return true if A dominates B. This performs the
  // special checks necessary if A and B are in the same basic block.
  bool dominates(Instruction *A, Instruction *B);

  //===--------------------------------------------------------------------===//
  // API to update (Post)DominatorTree information based on modifications to
  // the CFG...

  /// addNewBlock - Add a new node to the dominator tree information.  This
  /// creates a new node as a child of DomBB dominator node,linking it into 
  /// the children list of the immediate dominator.
  DomTreeNode *addNewBlock(BasicBlock *BB, BasicBlock *DomBB) {
    assert(getNode(BB) == 0 && "Block already in dominator tree!");
    DomTreeNode *IDomNode = getNode(DomBB);
    assert(IDomNode && "Not immediate dominator specified for block!");
    DFSInfoValid = false;
    return DomTreeNodes[BB] = 
      IDomNode->addChild(new DomTreeNode(BB, IDomNode));
  }

  /// changeImmediateDominator - This method is used to update the dominator
  /// tree information when a node's immediate dominator changes.
  ///
  void changeImmediateDominator(DomTreeNode *N, DomTreeNode *NewIDom) {
    assert(N && NewIDom && "Cannot change null node pointers!");
    DFSInfoValid = false;
    N->setIDom(NewIDom);
  }

  void changeImmediateDominator(BasicBlock *BB, BasicBlock *NewBB) {
    changeImmediateDominator(getNode(BB), getNode(NewBB));
  }

  /// eraseNode - Removes a node from  the dominator tree. Block must not
  /// domiante any other blocks. Removes node from its immediate dominator's
  /// children list. Deletes dominator node associated with basic block BB.
  void eraseNode(BasicBlock *BB);

  /// removeNode - Removes a node from the dominator tree.  Block must not
  /// dominate any other blocks.  Invalidates any node pointing to removed
  /// block.
  void removeNode(BasicBlock *BB) {
    assert(getNode(BB) && "Removing node that isn't in dominator tree.");
    DomTreeNodes.erase(BB);
  }

  /// print - Convert to human readable form
  ///
  virtual void print(std::ostream &OS, const Module* = 0) const;
  void print(std::ostream *OS, const Module* M = 0) const {
    if (OS) print(*OS, M);
  }
  virtual void dump();
  
protected:
  template<class GraphT> friend void Compress(DominatorTreeBase& DT,
                                              typename GraphT::NodeType* VIn);
  template<class GraphT> friend typename GraphT::NodeType* Eval(
                                                  DominatorTreeBase& DT,
                                                  typename GraphT::NodeType* V);
  template<class GraphT> friend void Link(DominatorTreeBase& DT,
                                          typename GraphT::NodeType* V,
                                          typename GraphT::NodeType* W,
                                          InfoRec &WInfo);
  
  template<class GraphT> friend unsigned DFSPass(DominatorTreeBase& DT,
                                                 typename GraphT::NodeType* V,
                                                 unsigned N);
  
  template<class NodeT> friend void Calculate(DominatorTreeBase& DT,
                                              Function& F);
  
  /// updateDFSNumbers - Assign In and Out numbers to the nodes while walking
  /// dominator tree in dfs order.
  void updateDFSNumbers();
  
  DomTreeNode *getNodeForBlock(BasicBlock *BB);
  
  inline BasicBlock *getIDom(BasicBlock *BB) const {
    DenseMap<BasicBlock*, BasicBlock*>::const_iterator I = IDoms.find(BB);
    return I != IDoms.end() ? I->second : 0;
  }
};

//===-------------------------------------
/// DominatorTree Class - Concrete subclass of DominatorTreeBase that is used to
/// compute a normal dominator tree.
///
class DominatorTree : public DominatorTreeBase {
public:
  static char ID; // Pass ID, replacement for typeid
  DominatorTree() : DominatorTreeBase(intptr_t(&ID), false) {}
  
  BasicBlock *getRoot() const {
    assert(Roots.size() == 1 && "Should always have entry node!");
    return Roots[0];
  }
  
  virtual bool runOnFunction(Function &F);
  
  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
  }

  /// splitBlock
  /// BB is split and now it has one successor. Update dominator tree to
  /// reflect this change.
  void splitBlock(BasicBlock *BB);
};

//===-------------------------------------
/// DominatorTree GraphTraits specialization so the DominatorTree can be
/// iterable by generic graph iterators.
///
template <> struct GraphTraits<DomTreeNode*> {
  typedef DomTreeNode NodeType;
  typedef NodeType::iterator  ChildIteratorType;
  
  static NodeType *getEntryNode(NodeType *N) {
    return N;
  }
  static inline ChildIteratorType child_begin(NodeType* N) {
    return N->begin();
  }
  static inline ChildIteratorType child_end(NodeType* N) {
    return N->end();
  }
};

template <> struct GraphTraits<DominatorTree*>
  : public GraphTraits<DomTreeNode*> {
  static NodeType *getEntryNode(DominatorTree *DT) {
    return DT->getRootNode();
  }
};


//===----------------------------------------------------------------------===//
/// DominanceFrontierBase - Common base class for computing forward and inverse
/// dominance frontiers for a function.
///
class DominanceFrontierBase : public DominatorBase {
public:
  typedef std::set<BasicBlock*>             DomSetType;    // Dom set for a bb
  typedef std::map<BasicBlock*, DomSetType> DomSetMapType; // Dom set map
protected:
  DomSetMapType Frontiers;
public:
  DominanceFrontierBase(intptr_t ID, bool isPostDom) 
    : DominatorBase(ID, isPostDom) {}

  virtual void releaseMemory() { Frontiers.clear(); }

  // Accessor interface:
  typedef DomSetMapType::iterator iterator;
  typedef DomSetMapType::const_iterator const_iterator;
  iterator       begin()       { return Frontiers.begin(); }
  const_iterator begin() const { return Frontiers.begin(); }
  iterator       end()         { return Frontiers.end(); }
  const_iterator end()   const { return Frontiers.end(); }
  iterator       find(BasicBlock *B)       { return Frontiers.find(B); }
  const_iterator find(BasicBlock *B) const { return Frontiers.find(B); }

  void addBasicBlock(BasicBlock *BB, const DomSetType &frontier) {
    assert(find(BB) == end() && "Block already in DominanceFrontier!");
    Frontiers.insert(std::make_pair(BB, frontier));
  }

  /// removeBlock - Remove basic block BB's frontier.
  void removeBlock(BasicBlock *BB) {
    assert(find(BB) != end() && "Block is not in DominanceFrontier!");
    for (iterator I = begin(), E = end(); I != E; ++I)
      I->second.erase(BB);
    Frontiers.erase(BB);
  }

  void addToFrontier(iterator I, BasicBlock *Node) {
    assert(I != end() && "BB is not in DominanceFrontier!");
    I->second.insert(Node);
  }

  void removeFromFrontier(iterator I, BasicBlock *Node) {
    assert(I != end() && "BB is not in DominanceFrontier!");
    assert(I->second.count(Node) && "Node is not in DominanceFrontier of BB");
    I->second.erase(Node);
  }

  /// print - Convert to human readable form
  ///
  virtual void print(std::ostream &OS, const Module* = 0) const;
  void print(std::ostream *OS, const Module* M = 0) const {
    if (OS) print(*OS, M);
  }
  virtual void dump();
};


//===-------------------------------------
/// DominanceFrontier Class - Concrete subclass of DominanceFrontierBase that is
/// used to compute a forward dominator frontiers.
///
class DominanceFrontier : public DominanceFrontierBase {
public:
  static char ID; // Pass ID, replacement for typeid
  DominanceFrontier() : 
    DominanceFrontierBase(intptr_t(&ID), false) {}

  BasicBlock *getRoot() const {
    assert(Roots.size() == 1 && "Should always have entry node!");
    return Roots[0];
  }

  virtual bool runOnFunction(Function &) {
    Frontiers.clear();
    DominatorTree &DT = getAnalysis<DominatorTree>();
    Roots = DT.getRoots();
    assert(Roots.size() == 1 && "Only one entry block for forward domfronts!");
    calculate(DT, DT[Roots[0]]);
    return false;
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
    AU.addRequired<DominatorTree>();
  }

  /// splitBlock - BB is split and now it has one successor. Update dominance
  /// frontier to reflect this change.
  void splitBlock(BasicBlock *BB);

  /// BasicBlock BB's new dominator is NewBB. Update BB's dominance frontier
  /// to reflect this change.
  void changeImmediateDominator(BasicBlock *BB, BasicBlock *NewBB,
                                DominatorTree *DT) {
    // NewBB is now  dominating BB. Which means BB's dominance
    // frontier is now part of NewBB's dominance frontier. However, BB
    // itself is not member of NewBB's dominance frontier.
    DominanceFrontier::iterator NewDFI = find(NewBB);
    DominanceFrontier::iterator DFI = find(BB);
    DominanceFrontier::DomSetType BBSet = DFI->second;
    for (DominanceFrontier::DomSetType::iterator BBSetI = BBSet.begin(),
           BBSetE = BBSet.end(); BBSetI != BBSetE; ++BBSetI) {
      BasicBlock *DFMember = *BBSetI;
      // Insert only if NewBB dominates DFMember.
      if (!DT->dominates(NewBB, DFMember))
        NewDFI->second.insert(DFMember);
    }
    NewDFI->second.erase(BB);
  }

private:
  const DomSetType &calculate(const DominatorTree &DT,
                              const DomTreeNode *Node);
};


} // End llvm namespace

#endif
