//===--- BinaryLoop.h - Interface for machine-level loop ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the BinaryLoop class, which represents a loop in the
// CFG of a binary function, and the BinaryLoopInfo class, which stores
// information about all the loops of a binary function.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_BOLT_BINARY_LOOP_H
#define LLVM_TOOLS_LLVM_BOLT_BINARY_LOOP_H

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/Analysis/LoopInfoImpl.h"
#include "llvm/Support/GenericDomTreeConstruction.h"

namespace llvm {
namespace bolt {

class BinaryBasicBlock;

typedef DomTreeNodeBase<BinaryBasicBlock> BinaryDomTreeNode;
typedef DominatorTreeBase<BinaryBasicBlock> BinaryDominatorTree;

class BinaryLoop : public LoopBase<BinaryBasicBlock, BinaryLoop> {
public:
  BinaryLoop() : LoopBase<BinaryBasicBlock, BinaryLoop>() { }

  // The total count of all the back edges of this loop.
  uint64_t TotalBackEdgeCount{0};

  // The times the loop is entered from outside.
  uint64_t EntryCount{0};

  // The times the loop is exited.
  uint64_t ExitCount{0};

  // Most of the public interface is provided by LoopBase.

protected:
  friend class LoopInfoBase<BinaryBasicBlock, BinaryLoop>;
  explicit BinaryLoop(BinaryBasicBlock *BB) :
    LoopBase<BinaryBasicBlock, BinaryLoop>(BB) { }
};

class BinaryLoopInfo : public LoopInfoBase<BinaryBasicBlock, BinaryLoop> {
public:
  BinaryLoopInfo() { }

  unsigned OuterLoops{0};
  unsigned TotalLoops{0};
  unsigned MaximumDepth{0};

  // Most of the public interface is provided by LoopInfoBase.
};

} // namespace bolt
} // namespace llvm

namespace llvm {

// BinaryDominatorTree GraphTraits specializations.
template <> struct GraphTraits<bolt::BinaryDomTreeNode *>
  : public DomTreeGraphTraitsBase<bolt::BinaryDomTreeNode,
                                  bolt::BinaryDomTreeNode::iterator> {};

template <> struct GraphTraits<const bolt::BinaryDomTreeNode *>
  : public DomTreeGraphTraitsBase<const bolt::BinaryDomTreeNode,
                                  bolt::BinaryDomTreeNode::const_iterator> {};

template <> struct GraphTraits<bolt::BinaryDominatorTree *>
  : public GraphTraits<bolt::BinaryDomTreeNode *> {
  static NodeType *getEntryNode(bolt::BinaryDominatorTree *DT) {
    return DT->getRootNode();
  }

  static nodes_iterator nodes_begin(bolt::BinaryDominatorTree *N) {
    return df_begin(getEntryNode(N));
  }

  static nodes_iterator nodes_end(bolt::BinaryDominatorTree *N) {
    return df_end(getEntryNode(N));
  }
};

} // namescpae llvm

#endif
