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
/// Loop class - Instances of this class are used to represent loops that are 
/// detected in the flow graph 
///
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
  inline Loop *getParentLoop() const { return ParentLoop; }

  /// contains - Return true of the specified basic block is in this loop
  bool contains(const BasicBlock *BB) const;

  /// getSubLoops - Return the loops contained entirely within this loop
  ///
  inline const std::vector<Loop*> &getSubLoops() const { return SubLoops; }
  inline const std::vector<BasicBlock*> &getBlocks() const { return Blocks; }

  /// isLoopExit - True if terminator in the block can branch to another block
  /// that is outside of the current loop.
  bool isLoopExit(BasicBlock *BB) const;

  /// getLoopPreheader - If there is a preheader for this loop, return it.  A
  /// loop has a preheader if there is only one edge to the header of the loop
  /// from outside of the loop.  If this is the case, the block branching to the
  /// header of the loop is the preheader node.  The "preheaders" pass can be
  /// "Required" to ensure that there is always a preheader node for every loop.
  ///
  /// This method returns null if there is no preheader for the loop (either
  /// because the loop is dead or because multiple blocks branch to the header
  /// node of this loop).
  ///
  BasicBlock *getLoopPreheader() const;

  /// addBasicBlockToLoop - This function is used by other analyses to update
  /// loop information.  NewBB is set to be a new member of the current loop.
  /// Because of this, it is added as a member of all parent loops, and is added
  /// to the specified LoopInfo object as being in the current basic block.  It
  /// is not valid to replace the loop header with this method.
  ///
  void addBasicBlockToLoop(BasicBlock *NewBB, LoopInfo &LI);

  void print(std::ostream &O) const;
private:
  friend class LoopInfo;
  inline Loop(BasicBlock *BB) : ParentLoop(0) {
    Blocks.push_back(BB); LoopDepth = 0;
  }
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
/// LoopInfo - This class builds and contains all of the top level loop
/// structures in the specified function.
///
class LoopInfo : public FunctionPass {
  // BBMap - Mapping of basic blocks to the inner most loop they occur in
  std::map<BasicBlock*, Loop*> BBMap;
  std::vector<Loop*> TopLevelLoops;
  friend class Loop;
public:
  ~LoopInfo() { releaseMemory(); }

  const std::vector<Loop*> &getTopLevelLoops() const { return TopLevelLoops; }

  /// getLoopFor - Return the inner most loop that BB lives in.  If a basic
  /// block is in no loop (for example the entry node), null is returned.
  ///
  const Loop *getLoopFor(const BasicBlock *BB) const {
    std::map<BasicBlock *, Loop*>::const_iterator I=BBMap.find((BasicBlock*)BB);
    return I != BBMap.end() ? I->second : 0;
  }

  /// operator[] - same as getLoopFor...
  ///
  inline const Loop *operator[](const BasicBlock *BB) const {
    return getLoopFor(BB);
  }

  /// getLoopDepth - Return the loop nesting level of the specified block...
  ///
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
#endif

  /// runOnFunction - Calculate the natural loop information.
  ///
  virtual bool runOnFunction(Function &F);

  virtual void releaseMemory();
  void print(std::ostream &O) const;

  /// getAnalysisUsage - Requires dominator sets
  ///
  virtual void getAnalysisUsage(AnalysisUsage &AU) const;

  static void stub();  // Noop
private:
  void Calculate(const DominatorSet &DS);
  Loop *ConsiderForLoop(BasicBlock *BB, const DominatorSet &DS);
};


// Make sure that any clients of this file link in LoopInfo.cpp
static IncludeFile
LOOP_INFO_INCLUDE_FILE((void*)&LoopInfo::stub);

#endif
