//===-- AMDGPUTargetMachine.cpp - TargetMachine for hw codegen targets-----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// The AMDGPU target machine contains all of the hardware specific information
// needed to emit code for R600 and SI GPUs.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUTargetMachine.h"
#include "AMDGPU.h"
#include "R600ISelLowering.h"
#include "R600InstrInfo.h"
#include "SIISelLowering.h"
#include "SIInstrInfo.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/CodeGen/MachineFunctionAnalysis.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/PassManager.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Scalar.h"

using namespace llvm;

extern "C" void LLVMInitializeAMDGPUTarget() {
  // Register the target
  RegisterTargetMachine<AMDGPUTargetMachine> X(TheAMDGPUTarget);
}

AMDGPUTargetMachine::AMDGPUTargetMachine(const Target &T, StringRef TT,
    StringRef CPU, StringRef FS,
  TargetOptions Options,
  Reloc::Model RM, CodeModel::Model CM,
  CodeGenOpt::Level OptLevel
)
:
  LLVMTargetMachine(T, TT, CPU, FS, Options, RM, CM, OptLevel),
  Subtarget(TT, CPU, FS),
  DataLayout(Subtarget.getDataLayout()),
  FrameLowering(TargetFrameLowering::StackGrowsUp,
      Subtarget.device()->getStackAlignment(), 0),
  IntrinsicInfo(this),
  InstrItins(&Subtarget.getInstrItineraryData()),
  mDump(false)

{
  // TLInfo uses InstrInfo so it must be initialized after.
  if (Subtarget.device()->getGeneration() <= AMDILDeviceInfo::HD6XXX) {
    InstrInfo = new R600InstrInfo(*this);
    TLInfo = new R600TargetLowering(*this);
  } else {
    InstrInfo = new SIInstrInfo(*this);
    TLInfo = new SITargetLowering(*this);
  }
}

AMDGPUTargetMachine::~AMDGPUTargetMachine()
{
}

bool AMDGPUTargetMachine::addPassesToEmitFile(PassManagerBase &PM,
                                              formatted_raw_ostream &Out,
                                              CodeGenFileType FileType,
                                              bool DisableVerify,
                                              AnalysisID StartAfter,
                                              AnalysisID StopAfter) {
  // XXX: Hack here addPassesToEmitFile will fail, but this is Ok since we are
  // only using it to access addPassesToGenerateCode()
  bool fail = LLVMTargetMachine::addPassesToEmitFile(PM, Out, FileType,
                                                     DisableVerify);
  assert(fail);

  const AMDILSubtarget &STM = getSubtarget<AMDILSubtarget>();
  std::string gpu = STM.getDeviceName();
  if (gpu == "SI") {
    PM.add(createSICodeEmitterPass(Out));
  } else if (Subtarget.device()->getGeneration() <= AMDILDeviceInfo::HD6XXX) {
    PM.add(createR600CodeEmitterPass(Out));
  } else {
    abort();
    return true;
  }
  PM.add(createGCInfoDeleter());

  return false;
}

namespace {
class AMDGPUPassConfig : public TargetPassConfig {
public:
  AMDGPUPassConfig(AMDGPUTargetMachine *TM, PassManagerBase &PM)
    : TargetPassConfig(TM, PM) {}

  AMDGPUTargetMachine &getAMDGPUTargetMachine() const {
    return getTM<AMDGPUTargetMachine>();
  }

  virtual bool addPreISel();
  virtual bool addInstSelector();
  virtual bool addPreRegAlloc();
  virtual bool addPostRegAlloc();
  virtual bool addPreSched2();
  virtual bool addPreEmitPass();
};
} // End of anonymous namespace

TargetPassConfig *AMDGPUTargetMachine::createPassConfig(PassManagerBase &PM) {
  return new AMDGPUPassConfig(this, PM);
}

bool
AMDGPUPassConfig::addPreISel()
{
  const AMDILSubtarget &ST = TM->getSubtarget<AMDILSubtarget>();
  if (ST.device()->getGeneration() <= AMDILDeviceInfo::HD6XXX) {
    addPass(createR600KernelParametersPass(
                     getAMDGPUTargetMachine().getTargetData()));
  }
  return false;
}

bool AMDGPUPassConfig::addInstSelector() {
  addPass(createAMDILPeepholeOpt(*TM));
  addPass(createAMDILISelDag(getAMDGPUTargetMachine()));
  return false;
}

bool AMDGPUPassConfig::addPreRegAlloc() {
  const AMDILSubtarget &ST = TM->getSubtarget<AMDILSubtarget>();

  if (ST.device()->getGeneration() > AMDILDeviceInfo::HD6XXX) {
    addPass(createSIAssignInterpRegsPass(*TM));
  }
  addPass(createAMDGPUConvertToISAPass(*TM));
  return false;
}

bool AMDGPUPassConfig::addPostRegAlloc() {
  return false;
}

bool AMDGPUPassConfig::addPreSched2() {
  return false;
}

bool AMDGPUPassConfig::addPreEmitPass() {
  addPass(createAMDILCFGPreparationPass(*TM));
  addPass(createAMDILCFGStructurizerPass(*TM));

  return false;
}

