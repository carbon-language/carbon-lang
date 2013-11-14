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
#include "R600Defines.h"
#include "R600MachineFunctionInfo.h"
#include "R600RegisterInfo.h"
#include "SIDefines.h"
#include "SIMachineFunctionInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/MathExtras.h"
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

AMDGPUAsmPrinter::AMDGPUAsmPrinter(TargetMachine &TM, MCStreamer &Streamer)
    : AsmPrinter(TM, Streamer)
{
  DisasmEnabled = TM.getSubtarget<AMDGPUSubtarget>().dumpCode() &&
                  ! Streamer.hasRawTextSupport();
}

/// We need to override this function so we can avoid
/// the call to EmitFunctionHeader(), which the MCPureStreamer can't handle.
bool AMDGPUAsmPrinter::runOnMachineFunction(MachineFunction &MF) {
  SetupMachineFunction(MF);
  if (OutStreamer.hasRawTextSupport()) {
    OutStreamer.EmitRawText("@" + MF.getName() + ":");
  }

  MCContext &Context = getObjFileLowering().getContext();
  const MCSectionELF *ConfigSection = Context.getELFSection(".AMDGPU.config",
                                              ELF::SHT_PROGBITS, 0,
                                              SectionKind::getReadOnly());
  OutStreamer.SwitchSection(ConfigSection);
  const AMDGPUSubtarget &STM = TM.getSubtarget<AMDGPUSubtarget>();
  if (STM.getGeneration() > AMDGPUSubtarget::NORTHERN_ISLANDS) {
    EmitProgramInfoSI(MF);
  } else {
    EmitProgramInfoR600(MF);
  }

  DisasmLines.clear();
  HexLines.clear();
  DisasmLineMaxLen = 0;

  OutStreamer.SwitchSection(getObjFileLowering().getTextSection());
  EmitFunctionBody();

  if (STM.dumpCode()) {
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
    MF.dump();
#endif

    if (DisasmEnabled) {
      OutStreamer.SwitchSection(Context.getELFSection(".AMDGPU.disasm",
                                                  ELF::SHT_NOTE, 0,
                                                  SectionKind::getReadOnly()));

      for (size_t i = 0; i < DisasmLines.size(); ++i) {
        std::string Comment(DisasmLineMaxLen - DisasmLines[i].size(), ' ');
        Comment += " ; " + HexLines[i] + "\n";

        OutStreamer.EmitBytes(StringRef(DisasmLines[i]));
        OutStreamer.EmitBytes(StringRef(Comment));
      }
    }
  }

  return false;
}

void AMDGPUAsmPrinter::EmitProgramInfoR600(MachineFunction &MF) {
  unsigned MaxGPR = 0;
  bool killPixel = false;
  const R600RegisterInfo * RI =
                static_cast<const R600RegisterInfo*>(TM.getRegisterInfo());
  R600MachineFunctionInfo *MFI = MF.getInfo<R600MachineFunctionInfo>();
  const AMDGPUSubtarget &STM = TM.getSubtarget<AMDGPUSubtarget>();

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

  unsigned RsrcReg;
  if (STM.getGeneration() >= AMDGPUSubtarget::EVERGREEN) {
    // Evergreen / Northern Islands
    switch (MFI->ShaderType) {
    default: // Fall through
    case ShaderType::COMPUTE:  RsrcReg = R_0288D4_SQ_PGM_RESOURCES_LS; break;
    case ShaderType::GEOMETRY: RsrcReg = R_028878_SQ_PGM_RESOURCES_GS; break;
    case ShaderType::PIXEL:    RsrcReg = R_028844_SQ_PGM_RESOURCES_PS; break;
    case ShaderType::VERTEX:   RsrcReg = R_028860_SQ_PGM_RESOURCES_VS; break;
    }
  } else {
    // R600 / R700
    switch (MFI->ShaderType) {
    default: // Fall through
    case ShaderType::GEOMETRY: // Fall through
    case ShaderType::COMPUTE:  // Fall through
    case ShaderType::VERTEX:   RsrcReg = R_028868_SQ_PGM_RESOURCES_VS; break;
    case ShaderType::PIXEL:    RsrcReg = R_028850_SQ_PGM_RESOURCES_PS; break;
    }
  }

  OutStreamer.EmitIntValue(RsrcReg, 4);
  OutStreamer.EmitIntValue(S_NUM_GPRS(MaxGPR + 1) |
                           S_STACK_SIZE(MFI->StackSize), 4);
  OutStreamer.EmitIntValue(R_02880C_DB_SHADER_CONTROL, 4);
  OutStreamer.EmitIntValue(S_02880C_KILL_ENABLE(killPixel), 4);

  if (MFI->ShaderType == ShaderType::COMPUTE) {
    OutStreamer.EmitIntValue(R_0288E8_SQ_LDS_ALLOC, 4);
    OutStreamer.EmitIntValue(RoundUpToAlignment(MFI->LDSSize, 4) >> 2, 4);
  }
}

void AMDGPUAsmPrinter::EmitProgramInfoSI(MachineFunction &MF) {
  const AMDGPUSubtarget &STM = TM.getSubtarget<AMDGPUSubtarget>();
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
        MachineOperand &MO = MI.getOperand(op_idx);
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
        case AMDGPU::SCC:
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
        } else if (AMDGPU::SReg_512RegClass.contains(reg)) {
          isSGPR = true;
          width = 16;
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

  unsigned LDSAlignShift;
  if (STM.getGeneration() < AMDGPUSubtarget::SEA_ISLANDS) {
    // LDS is allocated in 64 dword blocks
    LDSAlignShift = 8;
  } else {
    // LDS is allocated in 128 dword blocks
    LDSAlignShift = 9;
  }
  unsigned LDSBlocks =
          RoundUpToAlignment(MFI->LDSSize, 1 << LDSAlignShift) >> LDSAlignShift;

  if (MFI->ShaderType == ShaderType::COMPUTE) {
    OutStreamer.EmitIntValue(R_00B84C_COMPUTE_PGM_RSRC2, 4);
    OutStreamer.EmitIntValue(S_00B84C_LDS_SIZE(LDSBlocks), 4);
  }
  if (MFI->ShaderType == ShaderType::PIXEL) {
    OutStreamer.EmitIntValue(R_00B02C_SPI_SHADER_PGM_RSRC2_PS, 4);
    OutStreamer.EmitIntValue(S_00B02C_EXTRA_LDS_SIZE(LDSBlocks), 4);
    OutStreamer.EmitIntValue(R_0286CC_SPI_PS_INPUT_ENA, 4);
    OutStreamer.EmitIntValue(MFI->PSInputAddr, 4);
  }
}
