//===- llvm/Analysis/LoopInfo.h - Natural Loop Calculator --------*- C++ -*--=//
//
// This file defines the LoopInfo class that is used to identify natural loops
// and determine the loop depth of various nodes of the CFG.  Note that the
// loops identified may actually be several natural loops that share the same
// header node... not just a single natural loop.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_LOOP_INFO_H
#define LLVM_ANALYSIS_LOOP_INFO_H

#include "llvm/Pass.h"
#include <set>

class DominatorSet;
class LoopInfo;

//===----------------------------------------------------------------------===//
// Loop class - Instances of this class are used to represent loops that are 
// detected in the flow graph 
//
class Loop {
  Loop *ParentLoop;
  std::vector<BasicBlock *> Blocks;  // First entry is the header node
  std::vector<Loop*> SubLoops;       // Loops contained entirely within this one
  unsigned LoopDepth;                // Nesting depth of this loop

  Loop(const Loop &);                  // DO NOT IMPLEMENT
  const Loop &operator=(const Loop &); // DO NOT IMPLEMENT
public:

  inline unsigned getLoopDepth() const { return LoopDepth; }
  inline BasicBlock *getHeader() const { return Blocks.front(); }

  // contains - Return true of the specified basic block is in this loop
  bool contains(const BasicBlock *BB) const;

  // getSubLoops - Return the loops contained entirely within this loop
  inline const std::vector<Loop*> &getSubLoops() const { return SubLoops; }
  inline const std::vector<BasicBlock*> &getBlocks() const { return Blocks; }

private:
  friend class LoopInfo;
  inline Loop(BasicBlock *BB) { Blocks.push_back(BB); LoopDepth = 0; }
  ~Loop() {
    for (unsigned i = 0, e = SubLoops.size(); i != e; ++i)
      delete SubLoops[i];
  }
  
  void setLoopDepth(unsigned Level) {
    LoopDepth = Level;
    for (unsigned i = 0, e = SubLoops.size(); i != e; ++i)
      SubLoops[i]->setLoopDepth(Level+1);
  }
};



//===----------------------------------------------------------------------===//
// LoopInfo - This class builds and contains all of the top level loop
// structures in the specified function.
//
class LoopInfo : public FunctionPass {
  // BBMap - Mapping of basic blocks to the inner most loop they occur in
  std::map<BasicBlock*, Loop*> BBMap;
  std::vector<Loop*> TopLevelLoops;
public:
  static AnalysisID ID;            // LoopInfo Analysis ID 

  // LoopInfo ctor - Calculate the natural loop information for a CFG
  LoopInfo(AnalysisID id) { assert(id == ID); }
  ~LoopInfo() { releaseMemory(); }

  const char *getPassName() const { return "Natural Loop Analysis"; }

  const std::vector<Loop*> &getTopLevelLoops() const { return TopLevelLoops; }

  // getLoopFor - Return the inner most loop that BB lives in.  If a basic block
  // is in no loop (for example the entry node), null is returned.
  //
  const Loop *getLoopFor(const BasicBlock *BB) const {
    std::map<BasicBlock *, Loop*>::const_iterator I=BBMap.find((BasicBlock*)BB);
    return I != BBMap.end() ? I->second : 0;
  }
  inline const Loop *operator[](const BasicBlock *BB) const {
    return getLoopFor(BB);
  }

  // getLoopDepth - Return the loop nesting level of the specified block...
  unsigned getLoopDepth(const BasicBlock *BB) const {
    const Loop *L = getLoopFor(BB);
    return L ? L->getLoopDepth() : 0;
  }

#if 0
  // isLoopHeader - True if the block is a loop header node
  bool isLoopHeader(BasicBlock *BB) const {
    return getLoopFor(BB)->getHeader() == BB;
  }
  // isLoopEnd - True if block jumps to loop entry
  bool isLoopEnd(BasicBlock *BB) const;
  // isLoopExit - True if block is the loop exit
  bool isLoopExit(BasicBlock *BB) const;
#endif

  // runOnFunction - Pass framework implementation
  virtual bool runOnFunction(Function &F);

  virtual void releaseMemory();

  // getAnalysisUsage - Provide loop info, require dominator set
  //
  virtual void getAnalysisUsage(AnalysisUsage &AU) const;

private:
  void Calculate(const DominatorSet &DS);
  Loop *ConsiderForLoop(BasicBlock *BB, const DominatorSet &DS);
};

#endif
