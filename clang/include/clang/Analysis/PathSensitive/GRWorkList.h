//==- GRWorkList.h - Worklist class used by GREngine ---------------*- C++ -*-//
//             
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines GRWorkList, a pure virtual class that represents an
//  opaque worklist used by GREngine to explore the reachability state space.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_GRWORKLIST
#define LLVM_CLANG_ANALYSIS_GRWORKLIST

namespace clang {  

class ExplodedNodeImpl;
  
class GRWorkListUnit {
  ExplodedNodeImpl* Node;
  CFGBlock* Block;
  unsigned BlockIdx;
  
public:
  GRWorkListUnit(ExplodedNodeImpl* N, CFGBlock* B, unsigned idx)
  : Node(N), Block(B), BlockIdx(idx) {}
  
  explicit GRWorkListUnit(ExplodedNodeImpl* N)
  : Node(N), Block(NULL), BlockIdx(0) {}
  
  ExplodedNodeImpl* getNode()  const { return Node; }    
  CFGBlock*         getBlock() const { return Block; }
  unsigned          getIndex() const { return BlockIdx; }
};

class GRWorkList {
public:
  virtual ~GRWorkList();
  virtual bool hasWork() const = 0;
  virtual void Enqueue(const GRWorkListUnit& U) = 0;

  void Enqueue(ExplodedNodeImpl* N, CFGBlock& B, unsigned idx) {
    Enqueue(GRWorkListUnit(N,&B,idx));
  }
  
  virtual GRWorkListUnit Dequeue() = 0;
  
  static GRWorkList* MakeDFS(); 
};
} // end clang namespace  
#endif
