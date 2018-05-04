//===--- Passes/DataflowInfoManager.h -------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_BOLT_PASSES_DATAFLOWINFOMANAGER_H
#define LLVM_TOOLS_LLVM_BOLT_PASSES_DATAFLOWINFOMANAGER_H

#include "DominatorAnalysis.h"
#include "FrameAnalysis.h"
#include "LivenessAnalysis.h"
#include "ReachingDefOrUse.h"
#include "ReachingInsns.h"
#include "RegAnalysis.h"
#include "StackAllocationAnalysis.h"
#include "StackPointerTracking.h"
#include "StackReachingUses.h"

namespace llvm {
namespace bolt {

/// Manages instances for dataflow analyses and try to preserve the data
/// calculated by each analysis as much as possible, saving the need to
/// recompute it. Also provide an interface for data invalidation when the
/// analysis is outdated after a transform pass modified the function.
class DataflowInfoManager {
  const RegAnalysis *RA;
  const FrameAnalysis *FA;
  const BinaryContext &BC;
  BinaryFunction &BF;
  std::unique_ptr<ReachingDefOrUse</*Def=*/true>> RD;
  std::unique_ptr<ReachingDefOrUse</*Def=*/false>> RU;
  std::unique_ptr<LivenessAnalysis> LA;
  std::unique_ptr<StackReachingUses> SRU;
  std::unique_ptr<DominatorAnalysis</*Bwd=*/false>> DA;
  std::unique_ptr<DominatorAnalysis</*Bwd=*/true>> PDA;
  std::unique_ptr<StackPointerTracking> SPT;
  std::unique_ptr<ReachingInsns<false>> RI;
  std::unique_ptr<ReachingInsns<true>> RIB;
  std::unique_ptr<StackAllocationAnalysis> SAA;
  std::unique_ptr<std::unordered_map<const MCInst *, BinaryBasicBlock *>>
      InsnToBB;

public:
  DataflowInfoManager(const BinaryContext &BC, BinaryFunction &BF,
                      const RegAnalysis *RA, const FrameAnalysis *FA)
      : RA(RA), FA(FA), BC(BC), BF(BF){};

  /// Helper function to fetch the parent BB associated with a program point
  /// If PP is a BB itself, then return itself (cast to a BinaryBasicBlock)
  BinaryBasicBlock *getParentBB(ProgramPoint PP) {
    return PP.isBB() ? PP.getBB() : getInsnToBBMap()[PP.getInst()];
  }

  ReachingDefOrUse</*Def=*/true> &getReachingDefs();
  void invalidateReachingDefs();
  ReachingDefOrUse</*Def=*/false> &getReachingUses();
  void invalidateReachingUses();
  LivenessAnalysis &getLivenessAnalysis();
  void invalidateLivenessAnalysis();
  StackReachingUses &getStackReachingUses();
  void invalidateStackReachingUses();
  DominatorAnalysis<false> &getDominatorAnalysis();
  void invalidateDominatorAnalysis();
  DominatorAnalysis<true> &getPostDominatorAnalysis();
  void invalidatePostDominatorAnalysis();
  StackPointerTracking &getStackPointerTracking();
  void invalidateStackPointerTracking();
  ReachingInsns<false> &getReachingInsns();
  void invalidateReachingInsns();
  ReachingInsns<true> &getReachingInsnsBackwards();
  void invalidateReachingInsnsBackwards();
  StackAllocationAnalysis &getStackAllocationAnalysis();
  void invalidateStackAllocationAnalysis();
  std::unordered_map<const MCInst *, BinaryBasicBlock *> &getInsnToBBMap();
  void invalidateInsnToBBMap();
  void invalidateAll();
};

} // end namespace bolt
} // end namespace llvm

#endif
