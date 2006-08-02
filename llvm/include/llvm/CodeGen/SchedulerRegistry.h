//===-- llvm/CodeGen/SchedulerRegistry.h ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the James M. Laskey and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
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

namespace llvm {

//===----------------------------------------------------------------------===//
///
/// RegisterScheduler class - Track the registration of instruction schedulers.
///
//===----------------------------------------------------------------------===//

class SelectionDAGISel;
class ScheduleDAG;
class SelectionDAG;
class MachineBasicBlock;

class RegisterScheduler : public MachinePassRegistryNode {

public:

  typedef ScheduleDAG *(*FunctionPassCtor)(SelectionDAGISel*, SelectionDAG*,
                                           MachineBasicBlock*);

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

} // end namespace llvm


#endif
