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
//  2. ETForest: Efficient data structure for dominance comparisons and 
//     nearest-common-ancestor queries.
//  3. DominanceFrontier: Calculate and hold the dominance frontier for a
//     function.
//
//  These data structures are listed in increasing order of complexity.  It
//  takes longer to calculate the dominator frontier, for example, than the
//  DominatorTree mapping.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_DOMINATORS_H
#define LLVM_ANALYSIS_DOMINATORS_H

#include "llvm/Analysis/ET-Forest.h"
#include "llvm/Pass.h"
#include <set>

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

class DomTreeNode {
  BasicBlock *TheBB;
  DomTreeNode *IDom;
  ETNode *ETN;
  std::vector<DomTreeNode*> Children;
public:
  typedef std::vector<DomTreeNode*>::iterator iterator;
  typedef std::vector<DomTreeNode*>::const_iterator const_iterator;
  
  iterator begin()             { return Children.begin(); }
  iterator end()               { return Children.end(); }
  const_iterator begin() const { return Children.begin(); }
  const_iterator end()   const { return Children.end(); }
  
  inline BasicBlock *getBlock() const { return TheBB; }
  inline DomTreeNode *getIDom() const { return IDom; }
  inline ETNode *getETNode() const { return ETN; }
  inline const std::vector<DomTreeNode*> &getChildren() const { return Children; }
  
  inline DomTreeNode(BasicBlock *BB, DomTreeNode *iDom, ETNode *E) 
    : TheBB(BB), IDom(iDom), ETN(E) {
    if (IDom)
      ETN->setFather(IDom->getETNode());
  }
  inline DomTreeNode *addChild(DomTreeNode *C) { Children.push_back(C); return C; }
  void setIDom(DomTreeNode *NewIDom);
};

//===----------------------------------------------------------------------===//
/// DominatorTree - Calculate the immediate dominator tree for a function.
///
class DominatorTreeBase : public DominatorBase {

protected:
  void reset();
  typedef std::map<BasicBlock*, DomTreeNode*> DomTreeNodeMapType;
  DomTreeNodeMapType DomTreeNodes;
  DomTreeNode *RootNode;

  typedef std::map<BasicBlock*, ETNode*> ETMapType;
  ETMapType ETNodes;

  bool DFSInfoValid;
  unsigned int SlowQueries;
  // Information record used during immediate dominators computation.
  struct InfoRec {
    unsigned Semi;
    unsigned Size;
    BasicBlock *Label, *Parent, *Child, *Ancestor;

    std::vector<BasicBlock*> Bucket;

    InfoRec() : Semi(0), Size(0), Label(0), Parent(0), Child(0), Ancestor(0){}
  };

  std::map<BasicBlock*, BasicBlock*> IDoms;

  // Vertex - Map the DFS number to the BasicBlock*
  std::vector<BasicBlock*> Vertex;

  // Info - Collection of information used during the computation of idoms.
  std::map<BasicBlock*, InfoRec> Info;

  public:
  DominatorTreeBase(intptr_t ID, bool isPostDom) 
    : DominatorBase(ID, isPostDom), DFSInfoValid(false), SlowQueries(0) {}
  ~DominatorTreeBase() { reset(); }

  virtual void releaseMemory() { reset(); }

  /// getNode - return the (Post)DominatorTree node for the specified basic
  /// block.  This is the same as using operator[] on this class.
  ///
  inline DomTreeNode *getNode(BasicBlock *BB) const {
    DomTreeNodeMapType::const_iterator i = DomTreeNodes.find(BB);
    return (i != DomTreeNodes.end()) ? i->second : 0;
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
    while ((IDom = B->getIDom()) != 0 && IDom != A)
      B = IDom;   // Walk up the tree
    return IDom != 0;
  }

  void updateDFSNumbers();  

  /// dominates - Returns true iff this dominates N.  Note that this is not a
  /// constant time operation!
  ///
  inline bool dominates(const DomTreeNode *A, DomTreeNode *B) {
    if (B == A) 
      return true;  // A node trivially dominates itself.

    if (A == 0 || B == 0)
      return false;

    ETNode *NodeA = A->getETNode();
    ETNode *NodeB = B->getETNode();
    
    if (DFSInfoValid)
      return NodeB->DominatedBy(NodeA);

    // If we end up with too many slow queries, just update the
    // DFS numbers on the theory that we are going to keep querying.
    SlowQueries++;
    if (SlowQueries > 32) {
      updateDFSNumbers();
      return NodeB->DominatedBy(NodeA);
    }
    //return NodeB->DominatedBySlow(NodeA);
    return dominatedBySlowTreeWalk(A, B);
  }

  inline bool dominates(BasicBlock *A, BasicBlock *B) {
    if (A == B) 
      return true;
    
    return dominates(getNode(A), getNode(B));
  }

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
    ETNode *E = new ETNode(BB);
    ETNodes[BB] = E;
    return DomTreeNodes[BB] = 
      IDomNode->addChild(new DomTreeNode(BB, IDomNode, E));
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
};

//===-------------------------------------
/// DominatorTree Class - Concrete subclass of DominatorTreeBase that is used to
/// compute a normal dominator tree.
///
class DominatorTree : public DominatorTreeBase {
public:
  static char ID; // Pass ID, replacement for typeid
  DominatorTree() : DominatorTreeBase((intptr_t)&ID, false) {}
  
  BasicBlock *getRoot() const {
    assert(Roots.size() == 1 && "Should always have entry node!");
    return Roots[0];
  }
  
  virtual bool runOnFunction(Function &F);
  
  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
  }
private:
  void calculate(Function& F);
  DomTreeNode *getNodeForBlock(BasicBlock *BB);
  unsigned DFSPass(BasicBlock *V, InfoRec &VInfo, unsigned N);
  void Compress(BasicBlock *V);
  BasicBlock *Eval(BasicBlock *v);
  void Link(BasicBlock *V, BasicBlock *W, InfoRec &WInfo);
  inline BasicBlock *getIDom(BasicBlock *BB) const {
      std::map<BasicBlock*, BasicBlock*>::const_iterator I = IDoms.find(BB);
      return I != IDoms.end() ? I->second : 0;
    }
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


//===-------------------------------------
/// ET-Forest Class - Class used to construct forwards and backwards 
/// ET-Forests
///
class ETForestBase : public DominatorBase {
public:
  ETForestBase(intptr_t ID, bool isPostDom) 
    : DominatorBase(ID, isPostDom), Nodes(), 
      DFSInfoValid(false), SlowQueries(0) {}
  
  virtual void releaseMemory() { reset(); }

  typedef std::map<BasicBlock*, ETNode*> ETMapType;

  // FIXME : There is no need to make this interface public. 
  // Fix predicate simplifier.
  void updateDFSNumbers();
    
  /// dominates - Return true if A dominates B.
  ///
  inline bool dominates(BasicBlock *A, BasicBlock *B) {
    if (A == B)
      return true;
    
    ETNode *NodeA = getNode(A);
    ETNode *NodeB = getNode(B);
    
    if (DFSInfoValid)
      return NodeB->DominatedBy(NodeA);
    else {
      // If we end up with too many slow queries, just update the
      // DFS numbers on the theory that we are going to keep querying.
      SlowQueries++;
      if (SlowQueries > 32) {
        updateDFSNumbers();
        return NodeB->DominatedBy(NodeA);
      }
      return NodeB->DominatedBySlow(NodeA);
    }
  }

  // dominates - Return true if A dominates B. This performs the
  // special checks necessary if A and B are in the same basic block.
  bool dominates(Instruction *A, Instruction *B);

  /// properlyDominates - Return true if A dominates B and A != B.
  ///
  bool properlyDominates(BasicBlock *A, BasicBlock *B) {
    return dominates(A, B) && A != B;
  }

  /// isReachableFromEntry - Return true if A is dominated by the entry
  /// block of the function containing it.
  const bool isReachableFromEntry(BasicBlock* A);
  
  /// Return the nearest common dominator of A and B.
  BasicBlock *nearestCommonDominator(BasicBlock *A, BasicBlock *B) const  {
    ETNode *NodeA = getNode(A);
    ETNode *NodeB = getNode(B);
    
    ETNode *Common = NodeA->NCA(NodeB);
    if (!Common)
      return NULL;
    return Common->getData<BasicBlock>();
  }
  
  /// Return the immediate dominator of A.
  BasicBlock *getIDom(BasicBlock *A) const {
    ETNode *NodeA = getNode(A);
    if (!NodeA) return 0;
    const ETNode *idom = NodeA->getFather();
    return idom ? idom->getData<BasicBlock>() : 0;
  }
  
  void getETNodeChildren(BasicBlock *A, std::vector<BasicBlock*>& children) const {
    ETNode *NodeA = getNode(A);
    if (!NodeA) return;
    const ETNode* son = NodeA->getSon();
    
    if (!son) return;
    children.push_back(son->getData<BasicBlock>());
        
    const ETNode* brother = son->getBrother();
    while (brother != son) {
      children.push_back(brother->getData<BasicBlock>());
      brother = brother->getBrother();
    }
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
    AU.addRequired<DominatorTree>();
  }
  //===--------------------------------------------------------------------===//
  // API to update Forest information based on modifications
  // to the CFG...

  /// addNewBlock - Add a new block to the CFG, with the specified immediate
  /// dominator.
  ///
  void addNewBlock(BasicBlock *BB, BasicBlock *IDom);

  /// setImmediateDominator - Update the immediate dominator information to
  /// change the current immediate dominator for the specified block
  /// to another block.  This method requires that BB for NewIDom
  /// already have an ETNode, otherwise just use addNewBlock.
  ///
  void setImmediateDominator(BasicBlock *BB, BasicBlock *NewIDom);
  /// print - Convert to human readable form
  ///
  virtual void print(std::ostream &OS, const Module* = 0) const;
  void print(std::ostream *OS, const Module* M = 0) const {
    if (OS) print(*OS, M);
  }
  virtual void dump();
protected:
  /// getNode - return the (Post)DominatorTree node for the specified basic
  /// block.  This is the same as using operator[] on this class.
  ///
  inline ETNode *getNode(BasicBlock *BB) const {
    ETMapType::const_iterator i = Nodes.find(BB);
    return (i != Nodes.end()) ? i->second : 0;
  }

  inline ETNode *operator[](BasicBlock *BB) const {
    return getNode(BB);
  }

  void reset();
  ETMapType Nodes;
  bool DFSInfoValid;
  unsigned int SlowQueries;

};

//==-------------------------------------
/// ETForest Class - Concrete subclass of ETForestBase that is used to
/// compute a forwards ET-Forest.

class ETForest : public ETForestBase {
public:
  static char ID; // Pass identification, replacement for typeid

  ETForest() : ETForestBase((intptr_t)&ID, false) {}

  BasicBlock *getRoot() const {
    assert(Roots.size() == 1 && "Should always have entry node!");
    return Roots[0];
  }

  virtual bool runOnFunction(Function &F) {
    reset();     // Reset from the last time we were run...
    DominatorTree &DT = getAnalysis<DominatorTree>();
    Roots = DT.getRoots();
    calculate(DT);
    return false;
  }

  void calculate(const DominatorTree &DT);
  // FIXME : There is no need to make getNodeForBlock public. Fix
  // predicate simplifier.
  ETNode *getNodeForBlock(BasicBlock *BB);
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
    DominanceFrontierBase((intptr_t)& ID, false) {}

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

private:
  const DomSetType &calculate(const DominatorTree &DT,
                              const DomTreeNode *Node);
};


} // End llvm namespace

#endif
