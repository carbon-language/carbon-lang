#include "AMDGPUMCTargetDesc.h"
#include "AMDGPUMCAsmInfo.h"
#include "llvm/MC/MachineLocation.h"
#include "llvm/MC/MCCodeGenInfo.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetRegistry.h"

#define GET_INSTRINFO_MC_DESC
#include "AMDGPUGenInstrInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "AMDGPUGenSubtargetInfo.inc"

#define GET_REGINFO_MC_DESC
#include "AMDGPUGenRegisterInfo.inc"

using namespace llvm;

static MCInstrInfo *createAMDGPUMCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  InitAMDILMCInstrInfo(X);
  return X;
}

static MCRegisterInfo *createAMDGPUMCRegisterInfo(StringRef TT) {
  MCRegisterInfo *X = new MCRegisterInfo();
  InitAMDILMCRegisterInfo(X, 0);
  return X;
}

static MCSubtargetInfo *createAMDGPUMCSubtargetInfo(StringRef TT, StringRef CPU,
                                                   StringRef FS) {
  MCSubtargetInfo * X = new MCSubtargetInfo();
  InitAMDILMCSubtargetInfo(X, TT, CPU, FS);
  return X;
}

static MCCodeGenInfo *createAMDGPUMCCodeGenInfo(StringRef TT, Reloc::Model RM,
                                               CodeModel::Model CM,
                                               CodeGenOpt::Level OL) {
  MCCodeGenInfo *X = new MCCodeGenInfo();
  X->InitMCCodeGenInfo(RM, CM, OL);
  return X;
}

extern "C" void LLVMInitializeAMDGPUTargetMC() {

  RegisterMCAsmInfo<AMDGPUMCAsmInfo> Y(TheAMDGPUTarget);

  TargetRegistry::RegisterMCCodeGenInfo(TheAMDGPUTarget, createAMDGPUMCCodeGenInfo);

  TargetRegistry::RegisterMCInstrInfo(TheAMDGPUTarget, createAMDGPUMCInstrInfo);

  TargetRegistry::RegisterMCRegInfo(TheAMDGPUTarget, createAMDGPUMCRegisterInfo);

  TargetRegistry::RegisterMCSubtargetInfo(TheAMDGPUTarget, createAMDGPUMCSubtargetInfo);

}
