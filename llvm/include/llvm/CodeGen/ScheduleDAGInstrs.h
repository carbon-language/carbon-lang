//==- llvm/CodeGen/ScheduleDAGInstrs.h - MachineInstr Scheduling -*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the ScheduleDAGInstrs class, which implements
// scheduling for a MachineInstr-based dependency graph.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_SCHEDULEDAGINSTRS_H
#define LLVM_CODEGEN_SCHEDULEDAGINSTRS_H

#include "llvm/CodeGen/ScheduleDAG.h"

namespace llvm {
  struct SUnit;
  class MachineConstantPool;
  class MachineFunction;
  class MachineModuleInfo;
  class MachineRegisterInfo;
  class MachineInstr;
  class TargetRegisterInfo;
  class ScheduleDAG;
  class SelectionDAG;
  class SelectionDAGISel;
  class TargetInstrInfo;
  class TargetInstrDesc;
  class TargetLowering;
  class TargetMachine;
  class TargetRegisterClass;

  class ScheduleDAGInstrs : public ScheduleDAG {
  public:
    ScheduleDAGInstrs(MachineBasicBlock *bb,
                      const TargetMachine &tm);

    virtual ~ScheduleDAGInstrs() {}

    /// NewSUnit - Creates a new SUnit and return a ptr to it.
    ///
    SUnit *NewSUnit(MachineInstr *MI) {
      SUnits.push_back(SUnit(MI, (unsigned)SUnits.size()));
      SUnits.back().OrigNode = &SUnits.back();
      return &SUnits.back();
    }

    /// BuildSchedUnits - Build SUnits from the MachineBasicBlock that we are
    /// input.
    virtual void BuildSchedUnits();

    virtual MachineBasicBlock *EmitSchedule();

    /// Schedule - Order nodes according to selected style, filling
    /// in the Sequence member.
    ///
    virtual void Schedule() = 0;

    virtual void dumpNode(const SUnit *SU) const;

    virtual std::string getGraphNodeLabel(const SUnit *SU) const;
  };
}

#endif
