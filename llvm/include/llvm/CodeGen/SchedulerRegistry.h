//===-- llvm/CodeGen/SchedulerRegistry.h ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation for instruction scheduler function
// pass registry (RegisterScheduler).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGENSCHEDULERREGISTRY_H
#define LLVM_CODEGENSCHEDULERREGISTRY_H

#include "llvm/CodeGen/MachinePassRegistry.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

//===----------------------------------------------------------------------===//
///
/// RegisterScheduler class - Track the registration of instruction schedulers.
///
//===----------------------------------------------------------------------===//

class SelectionDAGISel;
class ScheduleDAGSDNodes;
class SelectionDAG;
class MachineBasicBlock;

class RegisterScheduler : public MachinePassRegistryNode {
public:
  typedef ScheduleDAGSDNodes *(*FunctionPassCtor)(SelectionDAGISel*,
                                                  CodeGenOpt::Level);

  static MachinePassRegistry Registry;

  RegisterScheduler(const char *N, const char *D, FunctionPassCtor C)
  : MachinePassRegistryNode(N, D, (MachinePassCtor)C)
  { Registry.Add(this); }
  ~RegisterScheduler() { Registry.Remove(this); }
  

  // Accessors.
  //
  RegisterScheduler *getNext() const {
    return (RegisterScheduler *)MachinePassRegistryNode::getNext();
  }
  static RegisterScheduler *getList() {
    return (RegisterScheduler *)Registry.getList();
  }
  static FunctionPassCtor getDefault() {
    return (FunctionPassCtor)Registry.getDefault();
  }
  static void setDefault(FunctionPassCtor C) {
    Registry.setDefault((MachinePassCtor)C);
  }
  static void setListener(MachinePassRegistryListener *L) {
    Registry.setListener(L);
  }
};

/// createBURRListDAGScheduler - This creates a bottom up register usage
/// reduction list scheduler.
ScheduleDAGSDNodes *createBURRListDAGScheduler(SelectionDAGISel *IS,
                                               CodeGenOpt::Level OptLevel);

/// createTDRRListDAGScheduler - This creates a top down register usage
/// reduction list scheduler.
ScheduleDAGSDNodes *createTDRRListDAGScheduler(SelectionDAGISel *IS,
                                               CodeGenOpt::Level OptLevel);

/// createTDListDAGScheduler - This creates a top-down list scheduler with
/// a hazard recognizer.
ScheduleDAGSDNodes *createTDListDAGScheduler(SelectionDAGISel *IS,
                                             CodeGenOpt::Level OptLevel);

/// createFastDAGScheduler - This creates a "fast" scheduler.
///
ScheduleDAGSDNodes *createFastDAGScheduler(SelectionDAGISel *IS,
                                           CodeGenOpt::Level OptLevel);

/// createDefaultScheduler - This creates an instruction scheduler appropriate
/// for the target.
ScheduleDAGSDNodes *createDefaultScheduler(SelectionDAGISel *IS,
                                           CodeGenOpt::Level OptLevel);

} // end namespace llvm

#endif
