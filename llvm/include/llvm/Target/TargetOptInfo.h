//===-- llvm/Target/MachineOptInfo.h -----------------------------*- C++ -*-==//
//
//  Describes properties of the target cache architecture.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_MACHINEOPTINFO_H
#define LLVM_TARGET_MACHINEOPTINFO_H

#include "Support/DataTypes.h"
class TargetMachine;

struct MachineOptInfo : public NonCopyableV {
  const TargetMachine &target;
  
public:
  MachineOptInfo(const TargetMachine& tgt): target(tgt) { }

  virtual bool IsUselessCopy    (const MachineInstr* MI) const = 0;
};

#endif
