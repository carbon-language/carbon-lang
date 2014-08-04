//===-- NVPTXTargetMachine.h - Define TargetMachine for NVPTX ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the NVPTX specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef NVPTX_TARGETMACHINE_H
#define NVPTX_TARGETMACHINE_H

#include "NVPTXSubtarget.h"
#include "ManagedStringPool.h"
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetSelectionDAGInfo.h"

namespace llvm {

/// NVPTXTargetMachine
///
class NVPTXTargetMachine : public LLVMTargetMachine {
  NVPTXSubtarget Subtarget;

  // Hold Strings that can be free'd all together with NVPTXTargetMachine
  ManagedStringPool ManagedStrPool;

public:
  NVPTXTargetMachine(const Target &T, StringRef TT, StringRef CPU, StringRef FS,
                     const TargetOptions &Options, Reloc::Model RM,
                     CodeModel::Model CM, CodeGenOpt::Level OP, bool is64bit);

  const NVPTXSubtarget *getSubtargetImpl() const override { return &Subtarget; }

  ManagedStringPool *getManagedStrPool() const {
    return const_cast<ManagedStringPool *>(&ManagedStrPool);
  }

  TargetPassConfig *createPassConfig(PassManagerBase &PM) override;

  // Emission of machine code through JITCodeEmitter is not supported.
  bool addPassesToEmitMachineCode(PassManagerBase &, JITCodeEmitter &,
                                  bool = true) override {
    return true;
  }

  // Emission of machine code through MCJIT is not supported.
  bool addPassesToEmitMC(PassManagerBase &, MCContext *&, raw_ostream &,
                         bool = true) override {
    return true;
  }

}; // NVPTXTargetMachine.

class NVPTXTargetMachine32 : public NVPTXTargetMachine {
  virtual void anchor();
public:
  NVPTXTargetMachine32(const Target &T, StringRef TT, StringRef CPU,
                       StringRef FS, const TargetOptions &Options,
                       Reloc::Model RM, CodeModel::Model CM,
                       CodeGenOpt::Level OL);
};

class NVPTXTargetMachine64 : public NVPTXTargetMachine {
  virtual void anchor();
public:
  NVPTXTargetMachine64(const Target &T, StringRef TT, StringRef CPU,
                       StringRef FS, const TargetOptions &Options,
                       Reloc::Model RM, CodeModel::Model CM,
                       CodeGenOpt::Level OL);
};

} // end namespace llvm

#endif
