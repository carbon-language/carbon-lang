//===-- llvm/Target/TargetOptInfo.h ------------------------------*- C++ -*-==//
//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETOPTINFO_H
#define LLVM_TARGET_TARGETOPTINFO_H

#include "Support/DataTypes.h"
class TargetMachine;

struct TargetOptInfo : public NonCopyableV {
  const TargetMachine &target;
  
public:
  TargetOptInfo(const TargetMachine& tgt): target(tgt) { }

  virtual bool IsUselessCopy    (const MachineInstr* MI) const = 0;
};

#endif
