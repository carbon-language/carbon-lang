//===- llvm/Analysis/Dominators.h - Dominator Info Calculation ---*- C++ -*--=//
//
// This file defines the following classes:
//  1. DominatorSet: Calculates the [reverse] dominator set for a function
//  2. ImmediateDominators: Calculates and holds a mapping between BasicBlocks
//     and their immediate dominator.
//  3. DominatorTree: Represent the ImmediateDominator as an explicit tree
//     structure.
//  4. DominanceFrontier: Calculate and hold the dominance frontier for a 
//     function.
//
//  These data structures are listed in increasing order of complexity.  It
//  takes longer to calculate the dominator frontier, for example, than the 
//  ImmediateDominator mapping.
// 
//===----------------------------------------------------------------------===//

#ifndef LLVM_DOMINATORS_H
#define LLVM_DOMINATORS_H

#include "llvm/Pass.h"
#include <set>
class Instruction;

//===----------------------------------------------------------------------===//
//
// DominatorBase - Base class that other, more interesting dominator analyses
// inherit from.
//
class DominatorBase : public FunctionPass {
protected:
  BasicBlock *Root;
  const bool IsPostDominators;

  inline DominatorBase(bool isPostDom) : Root(0), IsPostDominators(isPostDom) {}
public:
  inline BasicBlock *getRoot() const { return Root; }

  // Returns true if analysis based of postdoms
  bool isPostDominator() const { return IsPostDominators; }
};

//===----------------------------------------------------------------------===//
//
// DominatorSet - Maintain a set<BasicBlock*> for every basic block in a
// function, that represents the blocks that dominate the block.
//
class DominatorSet : public DominatorBase {
public:
  typedef std::set<BasicBlock*> DomSetType;    // Dom set for a bb
  // Map of dom sets
  typedef std::map<BasicBlock*, DomSetType> DomSetMapType;
private:
  DomSetMapType Doms;

  void calcForwardDominatorSet(Function &F);
  void calcPostDominatorSet(Function &F);
public:
  // DominatorSet ctor - Build either the dominator set or the post-dominator
  // set for a function... 
  //
  static AnalysisID ID;            // Build dominator set
  static AnalysisID PostDomID;     // Build postdominator set

  DominatorSet(AnalysisID id) : DominatorBase(id == PostDomID) {}

  virtual const char *getPassName() const {
    if (isPostDominator()) return "Post-Dominator Set Construction";
    else return "Dominator Set Construction";
  }

  virtual bool runOnFunction(Function &F);

  // Accessor interface:
  typedef DomSetMapType::const_iterator const_iterator;
  typedef DomSetMapType::iterator iterator;
  inline const_iterator begin() const { return Doms.begin(); }
  inline       iterator begin()       { return Doms.begin(); }
  inline const_iterator end()   const { return Doms.end(); }
  inline       iterator end()         { return Doms.end(); }
  inline const_iterator find(BasicBlock* B) const { return Doms.find(B); }
  inline       iterator find(BasicBlock* B)       { return Doms.find(B); }

  // getDominators - Return the set of basic blocks that dominate the specified
  // block.
  //
  inline const DomSetType &getDominators(BasicBlock *BB) const {
    const_iterator I = find(BB);
    assert(I != end() && "BB not in function!");
    return I->second;
  }

  // dominates - Return true if A dominates B.
  //
  inline bool dominates(BasicBlock *A, BasicBlock *B) const {
    return getDominators(B).count(A) != 0;
  }

  // dominates - Return true if A dominates B.  This performs the special checks
  // neccesary if A and B are in the same basic block.
  //
  bool dominates(Instruction *A, Instruction *B) const;


  // getAnalysisUsage - This obviously provides a dominator set, but it also
  // uses the UnifyFunctionExitNode pass if building post-dominators
  //
  virtual void getAnalysisUsage(AnalysisUsage &AU) const;
};


//===----------------------------------------------------------------------===//
//
// ImmediateDominators - Calculate the immediate dominator for each node in a
// function.
//
class ImmediateDominators : public DominatorBase {
  std::map<BasicBlock*, BasicBlock*> IDoms;
  void calcIDoms(const DominatorSet &DS);
public:

  // ImmediateDominators ctor - Calculate the idom or post-idom mapping,
  // for a function...
  //
  static AnalysisID ID;         // Build immediate dominators
  static AnalysisID PostDomID;  // Build immediate postdominators

  ImmediateDominators(AnalysisID id) : DominatorBase(id == PostDomID) {}

  virtual const char *getPassName() const {
    if (isPostDominator()) return "Immediate Post-Dominators Construction";
    else return "Immediate Dominators Construction";
  }

  virtual bool runOnFunction(Function &F) {
    IDoms.clear();     // Reset from the last time we were run...
    DominatorSet *DS;
    if (isPostDominator())
      DS = &getAnalysis<DominatorSet>(DominatorSet::PostDomID);
    else
      DS = &getAnalysis<DominatorSet>();

    Root = DS->getRoot();
    calcIDoms(*DS);                         // Can be used to make rev-idoms
    return false;
  }

  // Accessor interface:
  typedef std::map<BasicBlock*, BasicBlock*> IDomMapType;
  typedef IDomMapType::const_iterator const_iterator;
  inline const_iterator begin() const { return IDoms.begin(); }
  inline const_iterator end()   const { return IDoms.end(); }
  inline const_iterator find(BasicBlock* B) const { return IDoms.find(B);}

  // operator[] - Return the idom for the specified basic block.  The start
  // node returns null, because it does not have an immediate dominator.
  //
  inline BasicBlock *operator[](BasicBlock *BB) const {
    std::map<BasicBlock*, BasicBlock*>::const_iterator I = IDoms.find(BB);
    return I != IDoms.end() ? I->second : 0;
  }

  // getAnalysisUsage - This obviously provides a dominator tree, but it
  // can only do so with the input of dominator sets
  //
  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
    if (isPostDominator()) {
      AU.addRequired(DominatorSet::PostDomID);
      AU.addProvided(PostDomID);
    } else {
      AU.addRequired(DominatorSet::ID);
      AU.addProvided(ID);
    }
  }
};


//===----------------------------------------------------------------------===//
//
// DominatorTree - Calculate the immediate dominator tree for a function.
//
class DominatorTree : public DominatorBase {
  class Node2;
public:
  typedef Node2 Node;
private:
  std::map<BasicBlock*, Node*> Nodes;
  void calculate(const DominatorSet &DS);
  void reset();
  typedef std::map<BasicBlock*, Node*> NodeMapType;
public:
  class Node2 : public std::vector<Node*> {
    friend class DominatorTree;
    BasicBlock *TheNode;
    Node2 *IDom;
  public:
    inline BasicBlock *getNode() const { return TheNode; }
    inline Node2 *getIDom() const { return IDom; }
    inline const std::vector<Node*> &getChildren() const { return *this; }

    // dominates - Returns true iff this dominates N.  Note that this is not a 
    // constant time operation!
    inline bool dominates(const Node2 *N) const {
      const Node2 *IDom;
      while ((IDom = N->getIDom()) != 0 && IDom != this)
	N = IDom;   // Walk up the tree
      return IDom != 0;
    }

  private:
    inline Node2(BasicBlock *node, Node *iDom) 
      : TheNode(node), IDom(iDom) {}
    inline Node2 *addChild(Node *C) { push_back(C); return C; }
  };

public:
  // DominatorTree ctor - Compute a dominator tree, given various amounts of
  // previous knowledge...
  static AnalysisID ID;         // Build dominator tree
  static AnalysisID PostDomID;  // Build postdominator tree

  DominatorTree(AnalysisID id) : DominatorBase(id == PostDomID) {}
  ~DominatorTree() { reset(); }

  virtual const char *getPassName() const {
    if (isPostDominator()) return "Post-Dominator Tree Construction";
    else return "Dominator Tree Construction";
  }

  virtual bool runOnFunction(Function &F) {
    reset();
    DominatorSet *DS;
    if (isPostDominator())
      DS = &getAnalysis<DominatorSet>(DominatorSet::PostDomID);
    else
      DS = &getAnalysis<DominatorSet>();
    Root = DS->getRoot();
    calculate(*DS);                         // Can be used to make rev-idoms
    return false;
  }

  inline Node *operator[](BasicBlock *BB) const {
    NodeMapType::const_iterator i = Nodes.find(BB);
    return (i != Nodes.end()) ? i->second : 0;
  }

  // getAnalysisUsage - This obviously provides a dominator tree, but it
  // uses dominator sets
  //
  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
    if (isPostDominator()) {
      AU.addRequired(DominatorSet::PostDomID);
      AU.addProvided(PostDomID);
    } else {
      AU.addRequired(DominatorSet::ID);
      AU.addProvided(ID);
    }
  }
};


//===----------------------------------------------------------------------===//
//
// DominanceFrontier - Calculate the dominance frontiers for a function.
//
class DominanceFrontier : public DominatorBase {
public:
  typedef std::set<BasicBlock*>             DomSetType;    // Dom set for a bb
  typedef std::map<BasicBlock*, DomSetType> DomSetMapType; // Dom set map
private:
  DomSetMapType Frontiers;
  const DomSetType &calcDomFrontier(const DominatorTree &DT,
				    const DominatorTree::Node *Node);
  const DomSetType &calcPostDomFrontier(const DominatorTree &DT,
					const DominatorTree::Node *Node);
public:

  // DominatorFrontier ctor - Compute dominator frontiers for a function
  //
  static AnalysisID ID;         // Build dominator frontier
  static AnalysisID PostDomID;  // Build postdominator frontier

  DominanceFrontier(AnalysisID id) : DominatorBase(id == PostDomID) {}

  virtual const char *getPassName() const {
    if (isPostDominator()) return "Post-Dominance Frontier Construction";
    else return "Dominance Frontier Construction";
  }

  virtual bool runOnFunction(Function &) {
    Frontiers.clear();
    DominatorTree *DT;
    if (isPostDominator())
      DT = &getAnalysis<DominatorTree>(DominatorTree::PostDomID);
    else
      DT = &getAnalysis<DominatorTree>();
    Root = DT->getRoot();

    if (isPostDominator())
      calcPostDomFrontier(*DT, (*DT)[Root]);
    else
      calcDomFrontier(*DT, (*DT)[Root]);
    return false;
  }

  // Accessor interface:
  typedef DomSetMapType::const_iterator const_iterator;
  inline const_iterator begin() const { return Frontiers.begin(); }
  inline const_iterator end()   const { return Frontiers.end(); }
  inline const_iterator find(BasicBlock* B) const { return Frontiers.find(B); }

  // getAnalysisUsage - This obviously provides the dominance frontier, but it
  // uses dominator sets
  //
  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
    if (isPostDominator()) {
      AU.addRequired(DominatorTree::PostDomID);
      AU.addProvided(PostDomID);
    } else {
      AU.addRequired(DominatorTree::ID);
      AU.addProvided(ID);
    }
  }
};

#endif
