//===- llvm/Transforms/Utils/LoopUtils.h - Loop utilities -*- C++ -*-=========//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines some loop transformation utilities.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_LOOPUTILS_H
#define LLVM_TRANSFORMS_UTILS_LOOPUTILS_H

namespace llvm {
class AliasAnalysis;
class AssumptionCache;
class BasicBlock;
class DataLayout;
class DominatorTree;
class Loop;
class LoopInfo;
class Pass;
class ScalarEvolution;
class AliasSetTracker;
class AliasSet;
class PredIteratorCache;

/// \brief Captures loop safety information.
/// It keep information for loop & its header may throw exception.
struct LICMSafetyInfo {
  bool MayThrow;           // The current loop contains an instruction which
                           // may throw.
  bool HeaderMayThrow;     // Same as previous, but specific to loop header
  LICMSafetyInfo() : MayThrow(false), HeaderMayThrow(false)
  {}
};

BasicBlock *InsertPreheaderForLoop(Loop *L, Pass *P);

/// \brief Simplify each loop in a loop nest recursively.
///
/// This takes a potentially un-simplified loop L (and its children) and turns
/// it into a simplified loop nest with preheaders and single backedges. It
/// will optionally update \c AliasAnalysis and \c ScalarEvolution analyses if
/// passed into it.
bool simplifyLoop(Loop *L, DominatorTree *DT, LoopInfo *LI, Pass *PP,
                  AliasAnalysis *AA = nullptr, ScalarEvolution *SE = nullptr,
                  const DataLayout *DL = nullptr,
                  AssumptionCache *AC = nullptr);

/// \brief Put loop into LCSSA form.
///
/// Looks at all instructions in the loop which have uses outside of the
/// current loop. For each, an LCSSA PHI node is inserted and the uses outside
/// the loop are rewritten to use this node.
///
/// LoopInfo and DominatorTree are required and preserved.
///
/// If ScalarEvolution is passed in, it will be preserved.
///
/// Returns true if any modifications are made to the loop.
bool formLCSSA(Loop &L, DominatorTree &DT, LoopInfo *LI,
               ScalarEvolution *SE = nullptr);

/// \brief Put a loop nest into LCSSA form.
///
/// This recursively forms LCSSA for a loop nest.
///
/// LoopInfo and DominatorTree are required and preserved.
///
/// If ScalarEvolution is passed in, it will be preserved.
///
/// Returns true if any modifications are made to the loop.
bool formLCSSARecursively(Loop &L, DominatorTree &DT, LoopInfo *LI,
                          ScalarEvolution *SE = nullptr);

/// \brief Walk the specified region of the CFG (defined by all blocks
/// dominated by the specified block, and that are in the current loop) in
/// reverse depth first order w.r.t the DominatorTree. This allows us to visit
/// uses before definitions, allowing us to sink a loop body in one pass without
/// iteration. Takes DomTreeNode, AliasAnalysis, LoopInfo, DominatorTree, 
/// DataLayout, TargetLibraryInfo, Loop, AliasSet information for all 
/// instructions of the loop and loop safety information as arguments. 
/// It returns changed status. 
bool sinkRegion(DomTreeNode *, AliasAnalysis *, LoopInfo *, DominatorTree *,
                const DataLayout *, TargetLibraryInfo *, Loop *,
                AliasSetTracker *, LICMSafetyInfo *);

/// \brief Walk the specified region of the CFG (defined by all blocks
/// dominated by the specified block, and that are in the current loop) in depth
/// first order w.r.t the DominatorTree.  This allows us to visit definitions
/// before uses, allowing us to hoist a loop body in one pass without iteration.
/// Takes DomTreeNode, AliasAnalysis, LoopInfo, DominatorTree, DataLayout,
/// TargetLibraryInfo, Loop, AliasSet information for all instructions of the 
/// loop and loop safety information as arguments. It returns changed status.
bool hoistRegion(DomTreeNode *, AliasAnalysis *, LoopInfo *, DominatorTree *,
                 const DataLayout *, TargetLibraryInfo *, Loop *,
                 AliasSetTracker *, LICMSafetyInfo *);

/// \brief Try to promote memory values to scalars by sinking stores out of 
/// the loop and moving loads to before the loop.  We do this by looping over
/// the stores in the loop, looking for stores to Must pointers which are 
/// loop invariant. It takes AliasSet, Loop exit blocks vector, loop exit blocks
/// insertion point vector, PredIteratorCache, LoopInfo, DominatorTree, Loop,
/// AliasSet information for all instructions of the loop and loop safety 
/// information as arguments. It returns changed status.
bool promoteLoopAccessesToScalars(AliasSet &, SmallVectorImpl<BasicBlock*> &,
                                  SmallVectorImpl<Instruction*> &,
                                  PredIteratorCache &, LoopInfo *,
                                  DominatorTree *, Loop *, AliasSetTracker *,
                                  LICMSafetyInfo *);

/// \brief Computes safety information for a loop
/// checks loop body & header for the possiblity of may throw
/// exception, it takes LICMSafetyInfo and loop as argument.
/// Updates safety information in LICMSafetyInfo argument.
void computeLICMSafetyInfo(LICMSafetyInfo *, Loop *);

}

#endif
