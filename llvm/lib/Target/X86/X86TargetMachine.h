//===-- X86TargetMachine.h - Define TargetMachine for the X86 ---*- C++ -*-===//
// 
// This file declares the X86 specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef X86TARGETMACHINE_H
#define X86TARGETMACHINE_H

#include "llvm/Target/TargetMachine.h"
#include "X86InstrInfo.h"

class X86TargetMachine : public TargetMachine {
  X86InstrInfo instrInfo;
public:
  X86TargetMachine();

  virtual const MachineInstrInfo &getInstrInfo() const { return instrInfo; }
  virtual const MachineSchedInfo &getSchedInfo() const { abort(); }
  virtual const MachineRegInfo   &getRegInfo()   const { abort(); }
  virtual const MachineFrameInfo &getFrameInfo() const { abort(); }
  virtual const MachineCacheInfo &getCacheInfo() const { abort(); }
  virtual const MachineOptInfo   &getOptInfo()   const { abort(); }

  /// addPassesToJITCompile - Add passes to the specified pass manager to
  /// implement a fast dynamic compiler for this target.  Return true if this is
  /// not supported for this target.
  ///
  virtual bool addPassesToJITCompile(PassManager &PM);
};

#endif
