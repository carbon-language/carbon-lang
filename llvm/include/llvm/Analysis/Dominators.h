//===- llvm/Analysis/DominatorSet.h - Dominator Set Calculation --*- C++ -*--=//
//
// This file defines the following classes:
//  1. DominatorSet: Calculates the [reverse] dominator set for a method
//  2. ImmediateDominators: Calculates and holds the immediate dominator tree
//     for a method.
// 
//===----------------------------------------------------------------------===//

#ifndef LLVM_DOMINATOR_SET_H
#define LLVM_DOMINATOR_SET_H

#include <set>
#include <map>
#include <vector>
class Method;
class BasicBlock;

namespace cfg {

//===----------------------------------------------------------------------===//
//
// DominatorSet - Maintain a set<const BasicBlock*> for every basic block in a
// method, that represents the blocks that dominate the block.
//
class DominatorSet {
public:
  typedef set<const BasicBlock*>              DomSetType;    // Dom set for a bb
  typedef map<const BasicBlock *, DomSetType> DomSetMapType; // Map of dom sets
private:
  DomSetMapType Doms;
  const BasicBlock *Root;
public:
  // DominatorSet ctor - Build either the dominator set or the post-dominator
  // set for a method...
  //
  DominatorSet(const Method *M, bool PostDomSet = false);

  // Accessor interface:
  typedef DomSetMapType::const_iterator const_iterator;
  inline const_iterator begin() const { return Doms.begin(); }
  inline const_iterator end()   const { return Doms.end(); }
  inline const_iterator find(const BasicBlock* B) const { return Doms.find(B); }
  inline const BasicBlock *getRoot() const { return Root; }

  // getDominators - Return the set of basic blocks that dominate the specified
  // block.
  //
  inline const DomSetType &getDominators(const BasicBlock *BB) const {
    const_iterator I = find(BB);
    assert(I != end() && "BB not in method!");
    return I->second;
  }

  // dominates - Return true if A dominates B.
  //
  inline bool dominates(const BasicBlock *A, const BasicBlock *B) const {
    return getDominators(B).count(A) != 0;
  }
};


//===----------------------------------------------------------------------===//
//
// ImmediateDominators - Calculate the immediate dominator for each node in a
// method.
//
class ImmediateDominators {
  map<const BasicBlock*, const BasicBlock*> IDoms;
  const BasicBlock *Root;
  void calcIDoms(const DominatorSet &DS);
public:

  // ImmediateDominators ctor - Calculate the idom mapping, for a method, or
  // from a dominator set calculated for something else...
  //
  inline ImmediateDominators(const DominatorSet &DS) : Root(DS.getRoot()) {
    calcIDoms(DS);                         // Can be used to make rev-idoms
  }

  // Accessor interface:
  typedef map<const BasicBlock*, const BasicBlock*> IDomMapType;
  typedef IDomMapType::const_iterator const_iterator;
  inline const_iterator begin() const { return IDoms.begin(); }
  inline const_iterator end()   const { return IDoms.end(); }
  inline const_iterator find(const BasicBlock* B) const { return IDoms.find(B);}
  inline const BasicBlock *getRoot() const { return Root; }

  // operator[] - Return the idom for the specified basic block.  The start
  // node returns null, because it does not have an immediate dominator.
  //
  inline const BasicBlock *operator[](const BasicBlock *BB) const {
    map<const BasicBlock*, const BasicBlock*>::const_iterator I = 
      IDoms.find(BB);
    return I != IDoms.end() ? I->second : 0;
  }
};


//===----------------------------------------------------------------------===//
//
// DominatorTree - Calculate the immediate dominator tree for a method.
//
class DominatorTree {
  class Node;
  const BasicBlock *Root;
  map<const BasicBlock*, Node*> Nodes;
  void calculate(const DominatorSet &DS);
  typedef map<const BasicBlock*, Node*> NodeMapType;
public:
  class Node : public vector<Node*> {
    friend class DominatorTree;
    const BasicBlock *TheNode;
    Node * const IDom;
  public:
    inline const BasicBlock *getNode() const { return TheNode; }
    inline Node *getIDom() const { return IDom; }
    inline const vector<Node*> &getChildren() const { return *this; }

    // dominates - Returns true iff this dominates N.  Note that this is not a 
    // constant time operation!
    inline bool dominates(const Node *N) const {
      const Node *IDom;
      while ((IDom = N->getIDom()) != 0 && IDom != this)
	N = IDom;   // Walk up the tree
      return IDom != 0;
    }

  private:
    inline Node(const BasicBlock *node, Node *iDom) 
      : TheNode(node), IDom(iDom) {}
    inline Node *addChild(Node *C) { push_back(C); return C; }
  };

public:
  // DominatorTree ctors - Compute a dominator tree, given various amounts of
  // previous knowledge...
  //inline DominatorTree(const Method *M) { calculate(DominatorSet(M)); }
  inline DominatorTree(const DominatorSet &DS) : Root(DS.getRoot()) { 
    calculate(DS); 
  }

  DominatorTree(const ImmediateDominators &IDoms);
  ~DominatorTree();

  inline const BasicBlock *getRoot() const { return Root; }
  inline const Node *operator[](const BasicBlock *BB) const {
    NodeMapType::const_iterator i = Nodes.find(BB);
    return (i != Nodes.end()) ? i->second : 0;
  }
};


//===----------------------------------------------------------------------===//
//
// DominanceFrontier - Calculate the dominance frontiers for a method.
//
class DominanceFrontier {
  typedef set<const BasicBlock*>              DomSetType;    // Dom set for a bb
  typedef map<const BasicBlock *, DomSetType> DomSetMapType; // Map of dom sets
private:
  DomSetMapType Frontiers;
  const BasicBlock *Root;
  const DomSetType &calcDomFrontier(const DominatorTree &DT,
				    const DominatorTree::Node *Node);
public:
  DominanceFrontier(const DominatorSet &DS) : Root(DS.getRoot()) {
    const DominatorTree DT(DS);
    calcDomFrontier(DT, DT[Root]);
  }    
  DominanceFrontier(const ImmediateDominators &ID) : Root(ID.getRoot()) {
    const DominatorTree DT(ID);
    calcDomFrontier(DT, DT[Root]);
  }
  DominanceFrontier(const DominatorTree &DT) : Root(DT.getRoot()) {
    calcDomFrontier(DT, DT[Root]);
  }

  // Accessor interface:
  typedef DomSetMapType::const_iterator const_iterator;
  inline const_iterator begin() const { return Frontiers.begin(); }
  inline const_iterator end()   const { return Frontiers.end(); }
  inline const_iterator find(const BasicBlock* B) const { return Frontiers.find(B);}
  inline const BasicBlock *getRoot() const { return Root; }

};

} // End namespace cfg

#endif
