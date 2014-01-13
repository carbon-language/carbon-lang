//===- Dominators.h - Dominator Info Calculation ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the DominatorTree class, which provides fast and efficient
// dominance queries.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_DOMINATORS_H
#define LLVM_IR_DOMINATORS_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/GenericDomTree.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

namespace llvm {

EXTERN_TEMPLATE_INSTANTIATION(class DomTreeNodeBase<BasicBlock>);
EXTERN_TEMPLATE_INSTANTIATION(class DominatorTreeBase<BasicBlock>);

typedef DomTreeNodeBase<BasicBlock> DomTreeNode;

class BasicBlockEdge {
  const BasicBlock *Start;
  const BasicBlock *End;
public:
  BasicBlockEdge(const BasicBlock *Start_, const BasicBlock *End_) :
    Start(Start_), End(End_) { }
  const BasicBlock *getStart() const {
    return Start;
  }
  const BasicBlock *getEnd() const {
    return End;
  }
  bool isSingleEdge() const;
};

/// \brief Concrete subclass of DominatorTreeBase that is used to compute a
/// normal dominator tree.
class DominatorTree : public FunctionPass {
public:
  static char ID; // Pass ID, replacement for typeid
  DominatorTreeBase<BasicBlock>* DT;

  DominatorTree() : FunctionPass(ID) {
    initializeDominatorTreePass(*PassRegistry::getPassRegistry());
    DT = new DominatorTreeBase<BasicBlock>(false);
  }

  ~DominatorTree() {
    delete DT;
  }

  DominatorTreeBase<BasicBlock>& getBase() { return *DT; }

  /// \brief Returns the root blocks of the current CFG.
  ///
  /// This may include multiple blocks if we are computing post dominators.
  /// For forward dominators, this will always be a single block (the entry
  /// node).
  inline const std::vector<BasicBlock*> &getRoots() const {
    return DT->getRoots();
  }

  inline BasicBlock *getRoot() const {
    return DT->getRoot();
  }

  inline DomTreeNode *getRootNode() const {
    return DT->getRootNode();
  }

  /// Get all nodes dominated by R, including R itself.
  void getDescendants(BasicBlock *R,
                     SmallVectorImpl<BasicBlock *> &Result) const {
    DT->getDescendants(R, Result);
  }

  /// \brief Returns *false* if the other dominator tree matches this dominator
  /// tree.
  inline bool compare(DominatorTree &Other) const {
    DomTreeNode *R = getRootNode();
    DomTreeNode *OtherR = Other.getRootNode();

    if (!R || !OtherR || R->getBlock() != OtherR->getBlock())
      return true;

    if (DT->compare(Other.getBase()))
      return true;

    return false;
  }

  virtual bool runOnFunction(Function &F);

  virtual void verifyAnalysis() const;

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
  }

  inline bool dominates(const DomTreeNode* A, const DomTreeNode* B) const {
    return DT->dominates(A, B);
  }

  inline bool dominates(const BasicBlock* A, const BasicBlock* B) const {
    return DT->dominates(A, B);
  }

  // \brief Return true if Def dominates a use in User.
  //
  // This performs the special checks necessary if Def and User are in the same
  // basic block. Note that Def doesn't dominate a use in Def itself!
  bool dominates(const Instruction *Def, const Use &U) const;
  bool dominates(const Instruction *Def, const Instruction *User) const;
  bool dominates(const Instruction *Def, const BasicBlock *BB) const;
  bool dominates(const BasicBlockEdge &BBE, const Use &U) const;
  bool dominates(const BasicBlockEdge &BBE, const BasicBlock *BB) const;

  bool properlyDominates(const DomTreeNode *A, const DomTreeNode *B) const {
    return DT->properlyDominates(A, B);
  }

  bool properlyDominates(const BasicBlock *A, const BasicBlock *B) const {
    return DT->properlyDominates(A, B);
  }

  /// \brief Find nearest common dominator basic block for basic block A and B.
  ///
  /// If there is no such block then return NULL.
  inline BasicBlock *findNearestCommonDominator(BasicBlock *A, BasicBlock *B) {
    return DT->findNearestCommonDominator(A, B);
  }

  inline const BasicBlock *findNearestCommonDominator(const BasicBlock *A,
                                                      const BasicBlock *B) {
    return DT->findNearestCommonDominator(A, B);
  }

  inline DomTreeNode *operator[](BasicBlock *BB) const {
    return DT->getNode(BB);
  }

  /// \brief Returns the DominatorTree node for the specified basic block.
  ///
  /// This is the same as using operator[] on this class.
  inline DomTreeNode *getNode(BasicBlock *BB) const {
    return DT->getNode(BB);
  }

  /// \brief Add a new node to the dominator tree information.
  ///
  /// This creates a new node as a child of DomBB dominator node, linking it
  /// into the children list of the immediate dominator.
  inline DomTreeNode *addNewBlock(BasicBlock *BB, BasicBlock *DomBB) {
    return DT->addNewBlock(BB, DomBB);
  }

  /// \brief Updates the dominator tree information when a node's immediate
  /// dominator changes.
  inline void changeImmediateDominator(BasicBlock *N, BasicBlock* NewIDom) {
    DT->changeImmediateDominator(N, NewIDom);
  }

  inline void changeImmediateDominator(DomTreeNode *N, DomTreeNode* NewIDom) {
    DT->changeImmediateDominator(N, NewIDom);
  }

  /// \brief Removes a node from the dominator tree.
  ///
  /// The block must not dominate any other blocks. Removes node from its
  /// immediate dominator's children list. Deletes dominator node associated
  /// with basic block BB.
  inline void eraseNode(BasicBlock *BB) {
    DT->eraseNode(BB);
  }

  /// \brief BB is split and now it has one successor; update dominator tree to
  /// reflect this change.
  inline void splitBlock(BasicBlock* NewBB) {
    DT->splitBlock(NewBB);
  }

  bool isReachableFromEntry(const BasicBlock* A) const {
    return DT->isReachableFromEntry(A);
  }

  bool isReachableFromEntry(const Use &U) const;


  virtual void releaseMemory() {
    DT->releaseMemory();
  }

  virtual void print(raw_ostream &OS, const Module* M= 0) const;
};

//===-------------------------------------
// DominatorTree GraphTraits specializations so the DominatorTree can be
// iterable by generic graph iterators.

template <> struct GraphTraits<DomTreeNode*> {
  typedef DomTreeNode NodeType;
  typedef NodeType::iterator  ChildIteratorType;

  static NodeType *getEntryNode(NodeType *N) {
    return N;
  }
  static inline ChildIteratorType child_begin(NodeType *N) {
    return N->begin();
  }
  static inline ChildIteratorType child_end(NodeType *N) {
    return N->end();
  }

  typedef df_iterator<DomTreeNode*> nodes_iterator;

  static nodes_iterator nodes_begin(DomTreeNode *N) {
    return df_begin(getEntryNode(N));
  }

  static nodes_iterator nodes_end(DomTreeNode *N) {
    return df_end(getEntryNode(N));
  }
};

template <> struct GraphTraits<DominatorTree*>
  : public GraphTraits<DomTreeNode*> {
  static NodeType *getEntryNode(DominatorTree *DT) {
    return DT->getRootNode();
  }

  static nodes_iterator nodes_begin(DominatorTree *N) {
    return df_begin(getEntryNode(N));
  }

  static nodes_iterator nodes_end(DominatorTree *N) {
    return df_end(getEntryNode(N));
  }
};

} // End llvm namespace

#endif
