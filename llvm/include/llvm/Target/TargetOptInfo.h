//===-- llvm/Target/TargetOptInfo.h ------------------------------*- C++ -*-==//
//
//  FIXME: ADD A COMMENT DESCRIBING THIS FILE!
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETOPTINFO_H
#define LLVM_TARGET_TARGETOPTINFO_H

class MachineInstr;
class TargetMachine;

struct TargetOptInfo {
  const TargetMachine &target;
  
  TargetOptInfo(const TargetOptInfo &);   // DO NOT IMPLEMENT
  void operator=(const TargetOptInfo &);  // DO NOT IMPLEMENT
public:
  TargetOptInfo(const TargetMachine &TM) : target(TM) { }

  virtual bool IsUselessCopy(const MachineInstr* MI) const = 0;
};

#endif
