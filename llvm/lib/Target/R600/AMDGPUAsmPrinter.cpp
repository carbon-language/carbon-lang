//===-- AMDGPUAsmPrinter.cpp - AMDGPU Assebly printer  --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
///
/// The AMDGPUAsmPrinter is used to print both assembly string and also binary
/// code.  When passed an MCAsmStreamer it prints assembly and when passed
/// an MCObjectStreamer it outputs binary code.
//
//===----------------------------------------------------------------------===//
//


#include "AMDGPUAsmPrinter.h"
#include "AMDGPU.h"
#include "SIDefines.h"
#include "SIMachineFunctionInfo.h"
#include "SIRegisterInfo.h"
#include "R600MachineFunctionInfo.h"
#include "R600RegisterInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Target/TargetLoweringObjectFile.h"

using namespace llvm;


static AsmPrinter *createAMDGPUAsmPrinterPass(TargetMachine &tm,
                                              MCStreamer &Streamer) {
  return new AMDGPUAsmPrinter(tm, Streamer);
}

extern "C" void LLVMInitializeR600AsmPrinter() {
  TargetRegistry::RegisterAsmPrinter(TheAMDGPUTarget, createAMDGPUAsmPrinterPass);
}

/// We need to override this function so we can avoid
/// the call to EmitFunctionHeader(), which the MCPureStreamer can't handle.
bool AMDGPUAsmPrinter::runOnMachineFunction(MachineFunction &MF) {
  const AMDGPUSubtarget &STM = TM.getSubtarget<AMDGPUSubtarget>();
  if (STM.dumpCode()) {
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
    MF.dump();
#endif
  }
  SetupMachineFunction(MF);
  if (OutStreamer.hasRawTextSupport()) {
    OutStreamer.EmitRawText("@" + MF.getName() + ":");
  }

  const MCSectionELF *ConfigSection = getObjFileLowering().getContext()
                                              .getELFSection(".AMDGPU.config",
                                              ELF::SHT_PROGBITS, 0,
                                              SectionKind::getReadOnly());
  OutStreamer.SwitchSection(ConfigSection);
  if (STM.device()->getGeneration() > AMDGPUDeviceInfo::HD6XXX) {
    EmitProgramInfoSI(MF);
  } else {
    EmitProgramInfoR600(MF);
  }
  OutStreamer.SwitchSection(getObjFileLowering().getTextSection());
  EmitFunctionBody();
  return false;
}

void AMDGPUAsmPrinter::EmitProgramInfoR600(MachineFunction &MF) {
  unsigned MaxGPR = 0;
  bool killPixel = false;
  const R600RegisterInfo * RI =
                static_cast<const R600RegisterInfo*>(TM.getRegisterInfo());
  R600MachineFunctionInfo *MFI = MF.getInfo<R600MachineFunctionInfo>();

  for (MachineFunction::iterator BB = MF.begin(), BB_E = MF.end();
                                                  BB != BB_E; ++BB) {
    MachineBasicBlock &MBB = *BB;
    for (MachineBasicBlock::iterator I = MBB.begin(), E = MBB.end();
                                                    I != E; ++I) {
      MachineInstr &MI = *I;
      if (MI.getOpcode() == AMDGPU::KILLGT)
        killPixel = true;
      unsigned numOperands = MI.getNumOperands();
      for (unsigned op_idx = 0; op_idx < numOperands; op_idx++) {
        MachineOperand & MO = MI.getOperand(op_idx);
        if (!MO.isReg())
          continue;
        unsigned HWReg = RI->getEncodingValue(MO.getReg()) & 0xff;

        // Register with value > 127 aren't GPR
        if (HWReg > 127)
          continue;
        MaxGPR = std::max(MaxGPR, HWReg);
      }
    }
  }
  OutStreamer.EmitIntValue(MaxGPR + 1, 4);
  OutStreamer.EmitIntValue(MFI->StackSize, 4);
  OutStreamer.EmitIntValue(killPixel, 4);
}

void AMDGPUAsmPrinter::EmitProgramInfoSI(MachineFunction &MF) {
  unsigned MaxSGPR = 0;
  unsigned MaxVGPR = 0;
  bool VCCUsed = false;
  const SIRegisterInfo * RI =
                static_cast<const SIRegisterInfo*>(TM.getRegisterInfo());

  for (MachineFunction::iterator BB = MF.begin(), BB_E = MF.end();
                                                  BB != BB_E; ++BB) {
    MachineBasicBlock &MBB = *BB;
    for (MachineBasicBlock::iterator I = MBB.begin(), E = MBB.end();
                                                    I != E; ++I) {
      MachineInstr &MI = *I;

      unsigned numOperands = MI.getNumOperands();
      for (unsigned op_idx = 0; op_idx < numOperands; op_idx++) {
        MachineOperand & MO = MI.getOperand(op_idx);
        unsigned maxUsed;
        unsigned width = 0;
        bool isSGPR = false;
        unsigned reg;
        unsigned hwReg;
        if (!MO.isReg()) {
          continue;
        }
        reg = MO.getReg();
        if (reg == AMDGPU::VCC) {
          VCCUsed = true;
          continue;
        }
        switch (reg) {
        default: break;
        case AMDGPU::EXEC:
        case AMDGPU::M0:
          continue;
        }

        if (AMDGPU::SReg_32RegClass.contains(reg)) {
          isSGPR = true;
          width = 1;
        } else if (AMDGPU::VReg_32RegClass.contains(reg)) {
          isSGPR = false;
          width = 1;
        } else if (AMDGPU::SReg_64RegClass.contains(reg)) {
          isSGPR = true;
          width = 2;
        } else if (AMDGPU::VReg_64RegClass.contains(reg)) {
          isSGPR = false;
          width = 2;
        } else if (AMDGPU::VReg_96RegClass.contains(reg)) {
          isSGPR = false;
          width = 3;
        } else if (AMDGPU::SReg_128RegClass.contains(reg)) {
          isSGPR = true;
          width = 4;
        } else if (AMDGPU::VReg_128RegClass.contains(reg)) {
          isSGPR = false;
          width = 4;
        } else if (AMDGPU::SReg_256RegClass.contains(reg)) {
          isSGPR = true;
          width = 8;
        } else if (AMDGPU::VReg_256RegClass.contains(reg)) {
          isSGPR = false;
          width = 8;
        } else if (AMDGPU::VReg_512RegClass.contains(reg)) {
          isSGPR = false;
          width = 16;
        } else {
          assert(!"Unknown register class");
        }
        hwReg = RI->getEncodingValue(reg) & 0xff;
        maxUsed = hwReg + width - 1;
        if (isSGPR) {
          MaxSGPR = maxUsed > MaxSGPR ? maxUsed : MaxSGPR;
        } else {
          MaxVGPR = maxUsed > MaxVGPR ? maxUsed : MaxVGPR;
        }
      }
    }
  }
  if (VCCUsed) {
    MaxSGPR += 2;
  }
  SIMachineFunctionInfo * MFI = MF.getInfo<SIMachineFunctionInfo>();
  unsigned RsrcReg;
  switch (MFI->ShaderType) {
  default: // Fall through
  case ShaderType::COMPUTE:  RsrcReg = R_00B848_COMPUTE_PGM_RSRC1; break;
  case ShaderType::GEOMETRY: RsrcReg = R_00B228_SPI_SHADER_PGM_RSRC1_GS; break;
  case ShaderType::PIXEL:    RsrcReg = R_00B028_SPI_SHADER_PGM_RSRC1_PS; break;
  case ShaderType::VERTEX:   RsrcReg = R_00B128_SPI_SHADER_PGM_RSRC1_VS; break;
  }

  OutStreamer.EmitIntValue(RsrcReg, 4);
  OutStreamer.EmitIntValue(S_00B028_VGPRS(MaxVGPR / 4) | S_00B028_SGPRS(MaxSGPR / 8), 4);
  if (MFI->ShaderType == ShaderType::PIXEL) {
    OutStreamer.EmitIntValue(R_0286CC_SPI_PS_INPUT_ENA, 4);
    OutStreamer.EmitIntValue(MFI->PSInputAddr, 4);
  }
}
