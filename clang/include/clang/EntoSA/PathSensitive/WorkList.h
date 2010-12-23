//==- WorkList.h - Worklist class used by CoreEngine ---------------*- C++ -*-//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines WorkList, a pure virtual class that represents an opaque
//  worklist used by CoreEngine to explore the reachability state space.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_GR_WORKLIST
#define LLVM_CLANG_GR_WORKLIST

#include "clang/EntoSA/PathSensitive/BlockCounter.h"
#include <cstddef>

namespace clang {
  
class CFGBlock;

namespace ento {

class ExplodedNode;
class ExplodedNodeImpl;

class WorkListUnit {
  ExplodedNode* Node;
  BlockCounter Counter;
  const CFGBlock* Block;
  unsigned BlockIdx; // This is the index of the next statement.

public:
  WorkListUnit(ExplodedNode* N, BlockCounter C,
                 const CFGBlock* B, unsigned idx)
  : Node(N),
    Counter(C),
    Block(B),
    BlockIdx(idx) {}

  explicit WorkListUnit(ExplodedNode* N, BlockCounter C)
  : Node(N),
    Counter(C),
    Block(NULL),
    BlockIdx(0) {}

  ExplodedNode* getNode()         const { return Node; }
  BlockCounter    getBlockCounter() const { return Counter; }
  const CFGBlock*   getBlock()        const { return Block; }
  unsigned          getIndex()        const { return BlockIdx; }
};

class WorkList {
  BlockCounter CurrentCounter;
public:
  virtual ~WorkList();
  virtual bool hasWork() const = 0;

  virtual void Enqueue(const WorkListUnit& U) = 0;

  void Enqueue(ExplodedNode* N, const CFGBlock* B, unsigned idx) {
    Enqueue(WorkListUnit(N, CurrentCounter, B, idx));
  }

  void Enqueue(ExplodedNode* N) {
    Enqueue(WorkListUnit(N, CurrentCounter));
  }

  virtual WorkListUnit Dequeue() = 0;

  void setBlockCounter(BlockCounter C) { CurrentCounter = C; }
  BlockCounter getBlockCounter() const { return CurrentCounter; }

  class Visitor {
  public:
    Visitor() {}
    virtual ~Visitor();
    virtual bool Visit(const WorkListUnit &U) = 0;
  };
  virtual bool VisitItemsInWorkList(Visitor &V) = 0;
  
  static WorkList *MakeDFS();
  static WorkList *MakeBFS();
  static WorkList *MakeBFSBlockDFSContents();
};

} // end GR namespace

} // end clang namespace

#endif
