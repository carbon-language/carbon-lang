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

#include "ManagedStringPool.h"
#include "NVPTXFrameLowering.h"
#include "NVPTXISelLowering.h"
#include "NVPTXInstrInfo.h"
#include "NVPTXRegisterInfo.h"
#include "NVPTXSubtarget.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetSelectionDAGInfo.h"

namespace llvm {

/// NVPTXTargetMachine
///
class NVPTXTargetMachine : public LLVMTargetMachine {
  NVPTXSubtarget Subtarget;
  const DataLayout DL; // Calculates type size & alignment
  NVPTXInstrInfo InstrInfo;
  NVPTXTargetLowering TLInfo;
  TargetSelectionDAGInfo TSInfo;

  // NVPTX does not have any call stack frame, but need a NVPTX specific
  // FrameLowering class because TargetFrameLowering is abstract.
  NVPTXFrameLowering FrameLowering;

  // Hold Strings that can be free'd all together with NVPTXTargetMachine
  ManagedStringPool ManagedStrPool;

  //bool addCommonCodeGenPasses(PassManagerBase &, CodeGenOpt::Level,
  //                            bool DisableVerify, MCContext *&OutCtx);

public:
  NVPTXTargetMachine(const Target &T, StringRef TT, StringRef CPU, StringRef FS,
                     const TargetOptions &Options, Reloc::Model RM,
                     CodeModel::Model CM, CodeGenOpt::Level OP, bool is64bit);

  const TargetFrameLowering *getFrameLowering() const override {
    return &FrameLowering;
  }
  const NVPTXInstrInfo *getInstrInfo() const override { return &InstrInfo; }
  const DataLayout *getDataLayout() const override { return &DL; }
  const NVPTXSubtarget *getSubtargetImpl() const override { return &Subtarget; }

  const NVPTXRegisterInfo *getRegisterInfo() const override {
    return &(InstrInfo.getRegisterInfo());
  }

  NVPTXTargetLowering *getTargetLowering() const override {
    return const_cast<NVPTXTargetLowering *>(&TLInfo);
  }

  const TargetSelectionDAGInfo *getSelectionDAGInfo() const override {
    return &TSInfo;
  }

  //virtual bool addInstSelector(PassManagerBase &PM,
  //                             CodeGenOpt::Level OptLevel);

  //virtual bool addPreRegAlloc(PassManagerBase &, CodeGenOpt::Level);

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
