//===-- X86TargetMachine.h - Define TargetMachine for the X86 ---*- C++ -*-===//
// 
// This file declares the X86 specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef X86TARGETMACHINE_H
#define X86TARGETMACHINE_H

#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "X86InstrInfo.h"

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

  virtual const TargetSchedInfo &getSchedInfo()  const { abort(); }
  virtual const TargetRegInfo   &getRegInfo()    const { abort(); }
  virtual const TargetCacheInfo  &getCacheInfo() const { abort(); }
  virtual const TargetOptInfo    &getOptInfo()   const { abort(); }

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
  
  virtual bool addPassesToEmitAssembly(PassManager &PM, std::ostream &Out);
};

#endif
