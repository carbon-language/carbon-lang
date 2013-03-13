//===-- MipsSEISelLowering.h - MipsSE DAG Lowering Interface ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Subclass of MipsTargetLowering specialized for mips32/64.
//
//===----------------------------------------------------------------------===//

#ifndef MipsSEISELLOWERING_H
#define MipsSEISELLOWERING_H

#include "MipsISelLowering.h"

namespace llvm {
  class MipsSETargetLowering : public MipsTargetLowering  {
  public:
    explicit MipsSETargetLowering(MipsTargetMachine &TM);

    virtual bool allowsUnalignedMemoryAccesses(EVT VT, bool *Fast) const;

    virtual MachineBasicBlock *
    EmitInstrWithCustomInserter(MachineInstr *MI, MachineBasicBlock *MBB) const;

  private:
    virtual bool
    isEligibleForTailCallOptimization(const MipsCC &MipsCCInfo,
                                      unsigned NextStackOffset,
                                      const MipsFunctionInfo& FI) const;

    virtual void
    getOpndList(SmallVectorImpl<SDValue> &Ops,
                std::deque< std::pair<unsigned, SDValue> > &RegsToPass,
                bool IsPICCall, bool GlobalOrExternal, bool InternalLinkage,
                CallLoweringInfo &CLI, SDValue Callee, SDValue Chain) const;

    MachineBasicBlock *emitBPOSGE32(MachineInstr *MI,
                                    MachineBasicBlock *BB) const;
  };
}

#endif // MipsSEISELLOWERING_H
