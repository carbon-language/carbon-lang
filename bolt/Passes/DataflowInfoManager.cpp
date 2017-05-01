//===--- Passes/DataflowInfoManager.cpp -----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "DataflowInfoManager.h"


namespace llvm {
namespace bolt {

ReachingDefOrUse</*Def=*/true> &DataflowInfoManager::getReachingDefs() {
  if (RD)
    return *RD;
  assert(FA && "FrameAnalysis required");
  RD.reset(new ReachingDefOrUse<true>(*FA, BC, BF));
  {
    NamedRegionTimer T1("RD", "Dataflow", true);
    RD->run();
  }
  return *RD;
}

void DataflowInfoManager::invalidateReachingDefs() {
  RD.reset(nullptr);
}

ReachingDefOrUse</*Def=*/false> &DataflowInfoManager::getReachingUses() {
  if (RU)
    return *RU;
  assert(FA && "FrameAnalysis required");
  RU.reset(new ReachingDefOrUse<false>(*FA, BC, BF));
  {
    NamedRegionTimer T1("RU", "Dataflow", true);
    RU->run();
  }
  return *RU;
}

void DataflowInfoManager::invalidateReachingUses() {
  RU.reset(nullptr);
}

LivenessAnalysis &DataflowInfoManager::getLivenessAnalysis() {
  if (LA)
    return *LA;
  assert(FA && "FrameAnalysis required");
  LA.reset(new LivenessAnalysis(*FA, BC, BF));
  {
    NamedRegionTimer T1("LA", "Dataflow", true);
    LA->run();
  }
  return *LA;
}

void DataflowInfoManager::invalidateLivenessAnalysis() {
  LA.reset(nullptr);
}

DominatorAnalysis<false> &DataflowInfoManager::getDominatorAnalysis() {
  if (DA)
    return *DA;
  DA.reset(new DominatorAnalysis<false>(BC, BF));
  {
    NamedRegionTimer T1("DA", "Dataflow", true);
    DA->run();
  }
  return *DA;
}

void DataflowInfoManager::invalidateDominatorAnalysis() {
  DA.reset(nullptr);
}

DominatorAnalysis<true> &DataflowInfoManager::getPostDominatorAnalysis() {
  if (PDA)
    return *PDA;
  PDA.reset(new DominatorAnalysis<true>(BC, BF));
  {
    NamedRegionTimer T1("PDA", "Dataflow", true);
    PDA->run();
  }
  return *PDA;
}

void DataflowInfoManager::invalidatePostDominatorAnalysis() {
  PDA.reset(nullptr);
}

StackPointerTracking &DataflowInfoManager::getStackPointerTracking() {
  if (SPT)
    return *SPT;
  SPT.reset(new StackPointerTracking(BC, BF));
  {
    NamedRegionTimer T1("SPT", "Dataflow", true);
    SPT->run();
  }
  return *SPT;
}

void DataflowInfoManager::invalidateStackPointerTracking() {
  SPT.reset(nullptr);
}

ReachingInsns<false> &DataflowInfoManager::getReachingInsns() {
  if (RI)
    return *RI;
  RI.reset(new ReachingInsns<false>(BC, BF));
  {
    NamedRegionTimer T1("RI", "Dataflow", true);
    RI->run();
  }
  return *RI;
}

void DataflowInfoManager::invalidateReachingInsns() {
  RI.reset(nullptr);
}

ReachingInsns<true> &DataflowInfoManager::getReachingInsnsBackwards() {
  if (RIB)
    return *RIB;
  RIB.reset(new ReachingInsns<true>(BC, BF));
  {
    NamedRegionTimer T1("RIB", "Dataflow", true);
    RIB->run();
  }
  return *RIB;
}

void DataflowInfoManager::invalidateReachingInsnsBackwards() {
  RIB.reset(nullptr);
}

std::unordered_map<const MCInst *, BinaryBasicBlock *> &
DataflowInfoManager::getInsnToBBMap() {
  if (InsnToBB)
    return *InsnToBB;
  InsnToBB.reset(new std::unordered_map<const MCInst *, BinaryBasicBlock *>());
  for (auto &BB : BF) {
    for (auto &Inst : BB)
      (*InsnToBB)[&Inst] = &BB;
  }
  return *InsnToBB;
}

void DataflowInfoManager::invalidateInsnToBBMap() {
  InsnToBB.reset(nullptr);
}

void DataflowInfoManager::invalidateAll() {
  invalidateReachingDefs();
  invalidateReachingUses();
  invalidateLivenessAnalysis();
  invalidateDominatorAnalysis();
  invalidatePostDominatorAnalysis();
  invalidateStackPointerTracking();
  invalidateReachingInsns();
  invalidateReachingInsnsBackwards();
  invalidateInsnToBBMap();
}

} // end namespace bolt
} // end namespace llvm
