//===-- llvm/CodeGen/Splitter.h - Splitter -*- C++ -*----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_SPLITTER_H
#define LLVM_CODEGEN_SPLITTER_H

#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/SlotIndexes.h"

#include <deque>
#include <map>
#include <string>
#include <vector>

namespace llvm {

  class LiveInterval;
  class LiveIntervals;
  struct LiveRange;
  class LoopSplit;
  class MachineDominatorTree;
  class MachineRegisterInfo;
  class SlotIndexes;
  class TargetInstrInfo;
  class VNInfo;

  class LoopSplitter : public MachineFunctionPass {
    friend class LoopSplit;
  public:
    static char ID;

    LoopSplitter() : MachineFunctionPass(ID) {}

    virtual void getAnalysisUsage(AnalysisUsage &au) const;

    virtual bool runOnMachineFunction(MachineFunction &fn);

    virtual void releaseMemory();


  private:

    MachineFunction *mf;
    LiveIntervals *lis;
    MachineLoopInfo *mli;
    MachineRegisterInfo *mri;
    MachineDominatorTree *mdt;
    SlotIndexes *sis;
    const TargetInstrInfo *tii;
    const TargetRegisterInfo *tri;

    std::string fqn;
    std::deque<LiveInterval*> intervals;

    typedef std::pair<SlotIndex, SlotIndex> SlotPair;
    typedef std::vector<SlotPair> LoopRanges;
    typedef std::map<MachineLoop*, LoopRanges> LoopRangeMap;
    LoopRangeMap loopRangeMap;

    void dumpLoopInfo(MachineLoop &loop);

    void dumpOddTerminators();

    void updateTerminators(MachineBasicBlock &mbb);

    bool canInsertPreHeader(MachineLoop &loop);
    MachineBasicBlock& insertPreHeader(MachineLoop &loop);

    bool isCriticalEdge(MachineLoop::Edge &edge);
    bool canSplitEdge(MachineLoop::Edge &edge);
    MachineBasicBlock& splitEdge(MachineLoop::Edge &edge, MachineLoop &loop);

    LoopRanges& getLoopRanges(MachineLoop &loop);
    std::pair<bool, SlotPair> getLoopSubRange(const LiveRange &lr,
                                              MachineLoop &loop);

    void dumpLoopRanges(MachineLoop &loop);

    void processHeader(LoopSplit &split);
    void processLoopExits(LoopSplit &split);
    void processLoopUses(LoopSplit &split);

    bool splitOverLoop(LiveInterval &li, MachineLoop &loop);

    void processInterval(LiveInterval &li);

    void processIntervals();
  };

}

#endif
