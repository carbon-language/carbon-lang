//==- GRWorkList.h - Worklist class used by GRCoreEngine -----------*- C++ -*-//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines GRWorkList, a pure virtual class that represents an opaque
//  worklist used by GRCoreEngine to explore the reachability state space.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_GRWORKLIST
#define LLVM_CLANG_ANALYSIS_GRWORKLIST

#include "clang/Checker/PathSensitive/GRBlockCounter.h"
#include <cstddef>

namespace clang {
  
class CFGBlock;
class ExplodedNode;
class ExplodedNodeImpl;

class GRWorkListUnit {
  ExplodedNode* Node;
  GRBlockCounter Counter;
  const CFGBlock* Block;
  unsigned BlockIdx; // This is the index of the next statement.

public:
  GRWorkListUnit(ExplodedNode* N, GRBlockCounter C,
                 const CFGBlock* B, unsigned idx)
  : Node(N),
    Counter(C),
    Block(B),
    BlockIdx(idx) {}

  explicit GRWorkListUnit(ExplodedNode* N, GRBlockCounter C)
  : Node(N),
    Counter(C),
    Block(NULL),
    BlockIdx(0) {}

  ExplodedNode* getNode()         const { return Node; }
  GRBlockCounter    getBlockCounter() const { return Counter; }
  const CFGBlock*   getBlock()        const { return Block; }
  unsigned          getIndex()        const { return BlockIdx; }
};

class GRWorkList {
  GRBlockCounter CurrentCounter;
public:
  virtual ~GRWorkList();
  virtual bool hasWork() const = 0;

  virtual void Enqueue(const GRWorkListUnit& U) = 0;

  void Enqueue(ExplodedNode* N, const CFGBlock* B, unsigned idx) {
    Enqueue(GRWorkListUnit(N, CurrentCounter, B, idx));
  }

  void Enqueue(ExplodedNode* N) {
    Enqueue(GRWorkListUnit(N, CurrentCounter));
  }

  virtual GRWorkListUnit Dequeue() = 0;

  void setBlockCounter(GRBlockCounter C) { CurrentCounter = C; }
  GRBlockCounter getBlockCounter() const { return CurrentCounter; }

  static GRWorkList *MakeDFS();
  static GRWorkList *MakeBFS();
  static GRWorkList *MakeBFSBlockDFSContents();
};
} // end clang namespace
#endif
