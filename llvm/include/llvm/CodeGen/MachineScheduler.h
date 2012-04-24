//==- MachineScheduler.h - MachineInstr Scheduling Pass ----------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides a MachineSchedRegistry for registering alternative machine
// schedulers. A Target may provide an alternative scheduler implementation by
// implementing the following boilerplate:
//
// static ScheduleDAGInstrs *createCustomMachineSched(MachineSchedContext *C) {
//  return new CustomMachineScheduler(C);
// }
// static MachineSchedRegistry
// SchedCustomRegistry("custom", "Run my target's custom scheduler",
//                     createCustomMachineSched);
//
// Inside <Target>PassConfig:
//   enablePass(MachineSchedulerID);
//   MachineSchedRegistry::setDefault(createCustomMachineSched);
//
//===----------------------------------------------------------------------===//

#ifndef MACHINESCHEDULER_H
#define MACHINESCHEDULER_H

#include "RegisterClassInfo.h"
#include "llvm/CodeGen/MachinePassRegistry.h"

namespace llvm {

class AliasAnalysis;
class LiveIntervals;
class MachineDominatorTree;
class MachineLoopInfo;
class ScheduleDAGInstrs;

/// MachineSchedContext provides enough context from the MachineScheduler pass
/// for the target to instantiate a scheduler.
struct MachineSchedContext {
  MachineFunction *MF;
  const MachineLoopInfo *MLI;
  const MachineDominatorTree *MDT;
  const TargetPassConfig *PassConfig;
  AliasAnalysis *AA;
  LiveIntervals *LIS;

  RegisterClassInfo RegClassInfo;

  MachineSchedContext():
    MF(0), MLI(0), MDT(0), PassConfig(0), AA(0), LIS(0) {}
};

/// MachineSchedRegistry provides a selection of available machine instruction
/// schedulers.
class MachineSchedRegistry : public MachinePassRegistryNode {
public:
  typedef ScheduleDAGInstrs *(*ScheduleDAGCtor)(MachineSchedContext *);

  // RegisterPassParser requires a (misnamed) FunctionPassCtor type.
  typedef ScheduleDAGCtor FunctionPassCtor;

  static MachinePassRegistry Registry;

  MachineSchedRegistry(const char *N, const char *D, ScheduleDAGCtor C)
    : MachinePassRegistryNode(N, D, (MachinePassCtor)C) {
    Registry.Add(this);
  }
  ~MachineSchedRegistry() { Registry.Remove(this); }

  // Accessors.
  //
  MachineSchedRegistry *getNext() const {
    return (MachineSchedRegistry *)MachinePassRegistryNode::getNext();
  }
  static MachineSchedRegistry *getList() {
    return (MachineSchedRegistry *)Registry.getList();
  }
  static ScheduleDAGCtor getDefault() {
    return (ScheduleDAGCtor)Registry.getDefault();
  }
  static void setDefault(ScheduleDAGCtor C) {
    Registry.setDefault((MachinePassCtor)C);
  }
  static void setDefault(StringRef Name) {
    Registry.setDefault(Name);
  }
  static void setListener(MachinePassRegistryListener *L) {
    Registry.setListener(L);
  }
};

} // namespace llvm

#endif
