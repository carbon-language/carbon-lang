//===-- X86TargetMachine.h - Define TargetMachine for the X86 ---*- C++ -*-===//
// 
// This file declares the X86 specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef X86TARGETMACHINE_H
#define X86TARGETMACHINE_H

#include "llvm/Target/TargetMachine.h"
#include "X86InstrInfo.h"
#include "llvm/Target/MachineFrameInfo.h"

class X86TargetMachine : public TargetMachine {
  X86InstrInfo InstrInfo;
  TargetFrameInfo FrameInfo;
public:
  X86TargetMachine(unsigned Configuration);

  virtual const X86InstrInfo     &getInstrInfo() const { return InstrInfo; }
  virtual const TargetFrameInfo  &getFrameInfo() const { return FrameInfo; }
  virtual const MRegisterInfo *getRegisterInfo() const {
    return &InstrInfo.getRegisterInfo();
  }

  virtual const MachineSchedInfo &getSchedInfo() const { abort(); }
  virtual const MachineRegInfo   &getRegInfo()   const { abort(); }
  virtual const MachineCacheInfo &getCacheInfo() const { abort(); }
  virtual const MachineOptInfo   &getOptInfo()   const { abort(); }

  /// addPassesToJITCompile - Add passes to the specified pass manager to
  /// implement a fast dynamic compiler for this target.  Return true if this is
  /// not supported for this target.
  ///
  virtual bool addPassesToJITCompile(PassManager &PM);

  /// addPassesToEmitMachineCode - Add passes to the specified pass manager to
  /// get machine code emitted.  This uses a MAchineCodeEmitter object to handle
  /// actually outputting the machine code and resolving things like the address
  /// of functions.  This method should returns true if machine code emission is
  /// not supported.
  ///
  virtual bool addPassesToEmitMachineCode(PassManager &PM,
                                          MachineCodeEmitter &MCE);
};

#endif
