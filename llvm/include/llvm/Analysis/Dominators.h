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
class DominatorSetBase : public DominatorBase {
public:
  typedef std::set<BasicBlock*> DomSetType;    // Dom set for a bb
  // Map of dom sets
  typedef std::map<BasicBlock*, DomSetType> DomSetMapType;
protected:
  DomSetMapType Doms;
public:
  DominatorSetBase(bool isPostDom) : DominatorBase(isPostDom) {}

  virtual void releaseMemory() { Doms.clear(); }

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
};


//===-------------------------------------
// DominatorSet Class - Concrete subclass of DominatorSetBase that is used to
// compute a normal dominator set.
//
struct DominatorSet : public DominatorSetBase {
  static AnalysisID ID;            // Build dominator set

  DominatorSet(AnalysisID id) : DominatorSetBase(false) { assert(id == ID); }

  virtual const char *getPassName() const {
    return "Dominator Set Construction";
  }

  virtual bool runOnFunction(Function &F);

  // getAnalysisUsage - This simply provides a dominator set
  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
    AU.addProvided(ID);
  }
};


//===-------------------------------------
// DominatorSet Class - Concrete subclass of DominatorSetBase that is used to
// compute the post-dominator set.
//
struct PostDominatorSet : public DominatorSetBase {
  static AnalysisID ID;            // Build post-dominator set

  PostDominatorSet(AnalysisID id) : DominatorSetBase(true) { assert(id == ID); }

  virtual const char *getPassName() const {
    return "Post-Dominator Set Construction";
  }

  virtual bool runOnFunction(Function &F);

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
class ImmediateDominatorsBase : public DominatorBase {
protected:
  std::map<BasicBlock*, BasicBlock*> IDoms;
  void calcIDoms(const DominatorSetBase &DS);
public:
  ImmediateDominatorsBase(bool isPostDom) : DominatorBase(isPostDom) {}

  virtual void releaseMemory() { IDoms.clear(); }

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
};

//===-------------------------------------
// ImmediateDominators Class - Concrete subclass of ImmediateDominatorsBase that
// is used to compute a normal immediate dominator set.
//
struct ImmediateDominators : public ImmediateDominatorsBase {
  static AnalysisID ID;         // Build immediate dominators

  ImmediateDominators(AnalysisID id) : ImmediateDominatorsBase(false) {
    assert(id == ID);
  }

  virtual const char *getPassName() const {
    return "Immediate Dominators Construction";
  }

  virtual bool runOnFunction(Function &F) {
    IDoms.clear();     // Reset from the last time we were run...
    DominatorSet &DS = getAnalysis<DominatorSet>();
    Root = DS.getRoot();
    calcIDoms(DS);
    return false;
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
    AU.addProvided(ID);
    AU.addRequired(DominatorSet::ID);
  }
};


//===-------------------------------------
// ImmediatePostDominators Class - Concrete subclass of ImmediateDominatorsBase
// that is used to compute the immediate post-dominators.
//
struct ImmediatePostDominators : public ImmediateDominatorsBase {
  static AnalysisID ID;         // Build immediate postdominators

  ImmediatePostDominators(AnalysisID id) : ImmediateDominatorsBase(true) {
    assert(id == ID);
  }

  virtual const char *getPassName() const {
    return "Immediate Post-Dominators Construction";
  }

  virtual bool runOnFunction(Function &F) {
    IDoms.clear();     // Reset from the last time we were run...
    PostDominatorSet &DS = getAnalysis<PostDominatorSet>();
    Root = DS.getRoot();
    calcIDoms(DS);
    return false;
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
    AU.addRequired(PostDominatorSet::ID);
    AU.addProvided(ID);
  }
};



//===----------------------------------------------------------------------===//
//
// DominatorTree - Calculate the immediate dominator tree for a function.
//
class DominatorTreeBase : public DominatorBase {
protected:
  class Node2;
public:
  typedef Node2 Node;
protected:
  std::map<BasicBlock*, Node*> Nodes;
  void reset();
  typedef std::map<BasicBlock*, Node*> NodeMapType;
public:
  class Node2 : public std::vector<Node*> {
    friend class DominatorTree;
    friend class PostDominatorTree;
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
  DominatorTreeBase(bool isPostDom) : DominatorBase(isPostDom) {}
  ~DominatorTreeBase() { reset(); }

  virtual void releaseMemory() { reset(); }

  inline Node *operator[](BasicBlock *BB) const {
    NodeMapType::const_iterator i = Nodes.find(BB);
    return (i != Nodes.end()) ? i->second : 0;
  }
};


//===-------------------------------------
// DominatorTree Class - Concrete subclass of DominatorTreeBase that is used to
// compute a normal dominator tree.
//
struct DominatorTree : public DominatorTreeBase {
  static AnalysisID ID;         // Build dominator tree

  DominatorTree(AnalysisID id) : DominatorTreeBase(false) {
    assert(id == ID);
  }

  virtual const char *getPassName() const {
    return "Dominator Tree Construction";
  }

  virtual bool runOnFunction(Function &F) {
    reset();     // Reset from the last time we were run...
    DominatorSet &DS = getAnalysis<DominatorSet>();
    Root = DS.getRoot();
    calculate(DS);
    return false;
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
    AU.addProvided(ID);
    AU.addRequired(DominatorSet::ID);
  }
private:
  void calculate(const DominatorSet &DS);
};


//===-------------------------------------
// PostDominatorTree Class - Concrete subclass of DominatorTree that is used to
// compute the a post-dominator tree.
//
struct PostDominatorTree : public DominatorTreeBase {
  static AnalysisID ID;         // Build immediate postdominators

  PostDominatorTree(AnalysisID id) : DominatorTreeBase(true) {
    assert(id == ID);
  }

  virtual const char *getPassName() const {
    return "Post-Dominator Tree Construction";
  }

  virtual bool runOnFunction(Function &F) {
    reset();     // Reset from the last time we were run...
    PostDominatorSet &DS = getAnalysis<PostDominatorSet>();
    Root = DS.getRoot();
    calculate(DS);
    return false;
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
    AU.addRequired(PostDominatorSet::ID);
    AU.addProvided(ID);
  }
private:
  void calculate(const PostDominatorSet &DS);
};


//===----------------------------------------------------------------------===//
//
// DominanceFrontier - Calculate the dominance frontiers for a function.
//
class DominanceFrontierBase : public DominatorBase {
public:
  typedef std::set<BasicBlock*>             DomSetType;    // Dom set for a bb
  typedef std::map<BasicBlock*, DomSetType> DomSetMapType; // Dom set map
protected:
  DomSetMapType Frontiers;
public:
  DominanceFrontierBase(bool isPostDom) : DominatorBase(isPostDom) {}

  virtual void releaseMemory() { Frontiers.clear(); }

  // Accessor interface:
  typedef DomSetMapType::const_iterator const_iterator;
  inline const_iterator begin() const { return Frontiers.begin(); }
  inline const_iterator end()   const { return Frontiers.end(); }
  inline const_iterator find(BasicBlock* B) const { return Frontiers.find(B); }
};


//===-------------------------------------
// DominatorTree Class - Concrete subclass of DominatorTreeBase that is used to
// compute a normal dominator tree.
//
struct DominanceFrontier : public DominanceFrontierBase {
  static AnalysisID ID;         // Build dominance frontier

  DominanceFrontier(AnalysisID id) : DominanceFrontierBase(false) {
    assert(id == ID);
  }

  virtual const char *getPassName() const {
    return "Dominance Frontier Construction";
  }

  virtual bool runOnFunction(Function &) {
    Frontiers.clear();
    DominatorTree &DT = getAnalysis<DominatorTree>();
    Root = DT.getRoot();
    calculate(DT, DT[Root]);
    return false;
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
    AU.addProvided(ID);
    AU.addRequired(DominatorTree::ID);
  }
private:
  const DomSetType &calculate(const DominatorTree &DT,
                              const DominatorTree::Node *Node);
};


//===-------------------------------------

// PostDominanceFrontier Class - Concrete subclass of DominanceFrontier that is
// used to compute the a post-dominance frontier.
//
struct PostDominanceFrontier : public DominanceFrontierBase {
  static AnalysisID ID;         // Build post dominance frontier

  PostDominanceFrontier(AnalysisID id) : DominanceFrontierBase(true) {
    assert(id == ID);
  }

  virtual const char *getPassName() const {
    return "Post-Dominance Frontier Construction";
  }

  virtual bool runOnFunction(Function &) {
    Frontiers.clear();
    PostDominatorTree &DT = getAnalysis<PostDominatorTree>();
    Root = DT.getRoot();
    calculate(DT, DT[Root]);
    return false;
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
    AU.addRequired(PostDominatorTree::ID);
    AU.addProvided(ID);
  }
private:
  const DomSetType &calculate(const PostDominatorTree &DT,
                              const DominatorTree::Node *Node);
};

#endif
