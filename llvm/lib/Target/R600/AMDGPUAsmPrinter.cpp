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
#include "AMDGPUSubtarget.h"
#include "R600Defines.h"
#include "R600MachineFunctionInfo.h"
#include "R600RegisterInfo.h"
#include "SIDefines.h"
#include "SIMachineFunctionInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Target/TargetLoweringObjectFile.h"

using namespace llvm;

// TODO: This should get the default rounding mode from the kernel. We just set
// the default here, but this could change if the OpenCL rounding mode pragmas
// are used.
//
// The denormal mode here should match what is reported by the OpenCL runtime
// for the CL_FP_DENORM bit from CL_DEVICE_{HALF|SINGLE|DOUBLE}_FP_CONFIG, but
// can also be override to flush with the -cl-denorms-are-zero compiler flag.
//
// AMD OpenCL only sets flush none and reports CL_FP_DENORM for double
// precision, and leaves single precision to flush all and does not report
// CL_FP_DENORM for CL_DEVICE_SINGLE_FP_CONFIG. Mesa's OpenCL currently reports
// CL_FP_DENORM for both.
//
// FIXME: It seems some instructions do not support single precision denormals
// regardless of the mode (exp_*_f32, rcp_*_f32, rsq_*_f32, rsq_*f32, sqrt_f32,
// and sin_f32, cos_f32 on most parts).

// We want to use these instructions, and using fp32 denormals also causes
// instructions to run at the double precision rate for the device so it's
// probably best to just report no single precision denormals.
static uint32_t getFPMode(const MachineFunction &F) {
  const AMDGPUSubtarget& ST = F.getTarget().getSubtarget<AMDGPUSubtarget>();
  // TODO: Is there any real use for the flush in only / flush out only modes?

  uint32_t FP32Denormals =
    ST.hasFP32Denormals() ? FP_DENORM_FLUSH_NONE : FP_DENORM_FLUSH_IN_FLUSH_OUT;

  uint32_t FP64Denormals =
    ST.hasFP64Denormals() ? FP_DENORM_FLUSH_NONE : FP_DENORM_FLUSH_IN_FLUSH_OUT;

  return FP_ROUND_MODE_SP(FP_ROUND_ROUND_TO_NEAREST) |
         FP_ROUND_MODE_DP(FP_ROUND_ROUND_TO_NEAREST) |
         FP_DENORM_MODE_SP(FP32Denormals) |
         FP_DENORM_MODE_DP(FP64Denormals);
}

static AsmPrinter *createAMDGPUAsmPrinterPass(TargetMachine &tm,
                                              MCStreamer &Streamer) {
  return new AMDGPUAsmPrinter(tm, Streamer);
}

extern "C" void LLVMInitializeR600AsmPrinter() {
  TargetRegistry::RegisterAsmPrinter(TheAMDGPUTarget, createAMDGPUAsmPrinterPass);
}

AMDGPUAsmPrinter::AMDGPUAsmPrinter(TargetMachine &TM, MCStreamer &Streamer)
    : AsmPrinter(TM, Streamer) {
  DisasmEnabled = TM.getSubtarget<AMDGPUSubtarget>().dumpCode();
}

void AMDGPUAsmPrinter::EmitEndOfAsmFile(Module &M) {

  // This label is used to mark the end of the .text section.
  const TargetLoweringObjectFile &TLOF = getObjFileLowering();
  OutStreamer.SwitchSection(TLOF.getTextSection());
  MCSymbol *EndOfTextLabel =
      OutContext.GetOrCreateSymbol(StringRef(END_OF_TEXT_LABEL_NAME));
  OutStreamer.EmitLabel(EndOfTextLabel);
}

bool AMDGPUAsmPrinter::runOnMachineFunction(MachineFunction &MF) {
  SetupMachineFunction(MF);

  EmitFunctionHeader();

  MCContext &Context = getObjFileLowering().getContext();
  const MCSectionELF *ConfigSection = Context.getELFSection(".AMDGPU.config",
                                              ELF::SHT_PROGBITS, 0,
                                              SectionKind::getReadOnly());
  OutStreamer.SwitchSection(ConfigSection);

  const AMDGPUSubtarget &STM = TM.getSubtarget<AMDGPUSubtarget>();
  SIProgramInfo KernelInfo;
  if (STM.getGeneration() > AMDGPUSubtarget::NORTHERN_ISLANDS) {
    getSIProgramInfo(KernelInfo, MF);
    EmitProgramInfoSI(MF, KernelInfo);
  } else {
    EmitProgramInfoR600(MF);
  }

  DisasmLines.clear();
  HexLines.clear();
  DisasmLineMaxLen = 0;

  OutStreamer.SwitchSection(getObjFileLowering().getTextSection());
  EmitFunctionBody();

  if (isVerbose()) {
    const MCSectionELF *CommentSection
      = Context.getELFSection(".AMDGPU.csdata",
                              ELF::SHT_PROGBITS, 0,
                              SectionKind::getReadOnly());
    OutStreamer.SwitchSection(CommentSection);

    if (STM.getGeneration() >= AMDGPUSubtarget::SOUTHERN_ISLANDS) {
      OutStreamer.emitRawComment(" Kernel info:", false);
      OutStreamer.emitRawComment(" codeLenInByte = " + Twine(KernelInfo.CodeLen),
                                 false);
      OutStreamer.emitRawComment(" NumSgprs: " + Twine(KernelInfo.NumSGPR),
                                 false);
      OutStreamer.emitRawComment(" NumVgprs: " + Twine(KernelInfo.NumVGPR),
                                 false);
      OutStreamer.emitRawComment(" FloatMode: " + Twine(KernelInfo.FloatMode),
                                 false);
      OutStreamer.emitRawComment(" IeeeMode: " + Twine(KernelInfo.IEEEMode),
                                 false);
      OutStreamer.emitRawComment(" ScratchSize: " + Twine(KernelInfo.ScratchSize),
                                 false);
    } else {
      R600MachineFunctionInfo *MFI = MF.getInfo<R600MachineFunctionInfo>();
      OutStreamer.emitRawComment(
        Twine("SQ_PGM_RESOURCES:STACK_SIZE = " + Twine(MFI->StackSize)));
    }
  }

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

void AMDGPUAsmPrinter::EmitProgramInfoR600(const MachineFunction &MF) {
  unsigned MaxGPR = 0;
  bool killPixel = false;
  const R600RegisterInfo *RI = static_cast<const R600RegisterInfo *>(
      TM.getSubtargetImpl()->getRegisterInfo());
  const R600MachineFunctionInfo *MFI = MF.getInfo<R600MachineFunctionInfo>();
  const AMDGPUSubtarget &STM = TM.getSubtarget<AMDGPUSubtarget>();

  for (const MachineBasicBlock &MBB : MF) {
    for (const MachineInstr &MI : MBB) {
      if (MI.getOpcode() == AMDGPU::KILLGT)
        killPixel = true;
      unsigned numOperands = MI.getNumOperands();
      for (unsigned op_idx = 0; op_idx < numOperands; op_idx++) {
        const MachineOperand &MO = MI.getOperand(op_idx);
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
    switch (MFI->getShaderType()) {
    default: // Fall through
    case ShaderType::COMPUTE:  RsrcReg = R_0288D4_SQ_PGM_RESOURCES_LS; break;
    case ShaderType::GEOMETRY: RsrcReg = R_028878_SQ_PGM_RESOURCES_GS; break;
    case ShaderType::PIXEL:    RsrcReg = R_028844_SQ_PGM_RESOURCES_PS; break;
    case ShaderType::VERTEX:   RsrcReg = R_028860_SQ_PGM_RESOURCES_VS; break;
    }
  } else {
    // R600 / R700
    switch (MFI->getShaderType()) {
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

  if (MFI->getShaderType() == ShaderType::COMPUTE) {
    OutStreamer.EmitIntValue(R_0288E8_SQ_LDS_ALLOC, 4);
    OutStreamer.EmitIntValue(RoundUpToAlignment(MFI->LDSSize, 4) >> 2, 4);
  }
}

void AMDGPUAsmPrinter::getSIProgramInfo(SIProgramInfo &ProgInfo,
                                        const MachineFunction &MF) const {
  uint64_t CodeSize = 0;
  unsigned MaxSGPR = 0;
  unsigned MaxVGPR = 0;
  bool VCCUsed = false;
  bool FlatUsed = false;
  const SIRegisterInfo *RI = static_cast<const SIRegisterInfo *>(
      TM.getSubtargetImpl()->getRegisterInfo());

  for (const MachineBasicBlock &MBB : MF) {
    for (const MachineInstr &MI : MBB) {
      // TODO: CodeSize should account for multiple functions.
      CodeSize += MI.getDesc().Size;

      unsigned numOperands = MI.getNumOperands();
      for (unsigned op_idx = 0; op_idx < numOperands; op_idx++) {
        const MachineOperand &MO = MI.getOperand(op_idx);
        unsigned width = 0;
        bool isSGPR = false;

        if (!MO.isReg()) {
          continue;
        }
        unsigned reg = MO.getReg();
        if (reg == AMDGPU::VCC || reg == AMDGPU::VCC_LO ||
	    reg == AMDGPU::VCC_HI) {
          VCCUsed = true;
          continue;
        } else if (reg == AMDGPU::FLAT_SCR ||
                   reg == AMDGPU::FLAT_SCR_LO ||
                   reg == AMDGPU::FLAT_SCR_HI) {
          FlatUsed = true;
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
          llvm_unreachable("Unknown register class");
        }
        unsigned hwReg = RI->getEncodingValue(reg) & 0xff;
        unsigned maxUsed = hwReg + width - 1;
        if (isSGPR) {
          MaxSGPR = maxUsed > MaxSGPR ? maxUsed : MaxSGPR;
        } else {
          MaxVGPR = maxUsed > MaxVGPR ? maxUsed : MaxVGPR;
        }
      }
    }
  }

  if (VCCUsed)
    MaxSGPR += 2;

  if (FlatUsed)
    MaxSGPR += 2;

  // We found the maximum register index. They start at 0, so add one to get the
  // number of registers.
  ProgInfo.NumVGPR = MaxVGPR + 1;
  ProgInfo.NumSGPR = MaxSGPR + 1;

  // Set the value to initialize FP_ROUND and FP_DENORM parts of the mode
  // register.
  ProgInfo.FloatMode = getFPMode(MF);

  // XXX: Not quite sure what this does, but sc seems to unset this.
  ProgInfo.IEEEMode = 0;

  // Do not clamp NAN to 0.
  ProgInfo.DX10Clamp = 0;

  const MachineFrameInfo *FrameInfo = MF.getFrameInfo();
  ProgInfo.ScratchSize = FrameInfo->estimateStackSize(MF);

  ProgInfo.FlatUsed = FlatUsed;
  ProgInfo.VCCUsed = VCCUsed;
  ProgInfo.CodeLen = CodeSize;
}

void AMDGPUAsmPrinter::EmitProgramInfoSI(const MachineFunction &MF,
                                         const SIProgramInfo &KernelInfo) {
  const AMDGPUSubtarget &STM = TM.getSubtarget<AMDGPUSubtarget>();
  const SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();

  unsigned RsrcReg;
  switch (MFI->getShaderType()) {
  default: // Fall through
  case ShaderType::COMPUTE:  RsrcReg = R_00B848_COMPUTE_PGM_RSRC1; break;
  case ShaderType::GEOMETRY: RsrcReg = R_00B228_SPI_SHADER_PGM_RSRC1_GS; break;
  case ShaderType::PIXEL:    RsrcReg = R_00B028_SPI_SHADER_PGM_RSRC1_PS; break;
  case ShaderType::VERTEX:   RsrcReg = R_00B128_SPI_SHADER_PGM_RSRC1_VS; break;
  }

  unsigned LDSAlignShift;
  if (STM.getGeneration() < AMDGPUSubtarget::SEA_ISLANDS) {
    // LDS is allocated in 64 dword blocks.
    LDSAlignShift = 8;
  } else {
    // LDS is allocated in 128 dword blocks.
    LDSAlignShift = 9;
  }

  unsigned LDSSpillSize = MFI->LDSWaveSpillSize *
                          MFI->getMaximumWorkGroupSize(MF);

  unsigned LDSBlocks =
     RoundUpToAlignment(MFI->LDSSize + LDSSpillSize,
	                      1 << LDSAlignShift) >> LDSAlignShift;

  // Scratch is allocated in 256 dword blocks.
  unsigned ScratchAlignShift = 10;
  // We need to program the hardware with the amount of scratch memory that
  // is used by the entire wave.  KernelInfo.ScratchSize is the amount of
  // scratch memory used per thread.
  unsigned ScratchBlocks =
    RoundUpToAlignment(KernelInfo.ScratchSize * STM.getWavefrontSize(),
                       1 << ScratchAlignShift) >> ScratchAlignShift;

  unsigned VGPRBlocks = (KernelInfo.NumVGPR - 1) / 4;
  unsigned SGPRBlocks = (KernelInfo.NumSGPR - 1) / 8;

  if (MFI->getShaderType() == ShaderType::COMPUTE) {
    OutStreamer.EmitIntValue(R_00B848_COMPUTE_PGM_RSRC1, 4);

    const uint32_t ComputePGMRSrc1 =
      S_00B848_VGPRS(VGPRBlocks) |
      S_00B848_SGPRS(SGPRBlocks) |
      S_00B848_PRIORITY(KernelInfo.Priority) |
      S_00B848_FLOAT_MODE(KernelInfo.FloatMode) |
      S_00B848_PRIV(KernelInfo.Priv) |
      S_00B848_DX10_CLAMP(KernelInfo.DX10Clamp) |
      S_00B848_IEEE_MODE(KernelInfo.DebugMode) |
      S_00B848_IEEE_MODE(KernelInfo.IEEEMode);

    OutStreamer.EmitIntValue(ComputePGMRSrc1, 4);

    OutStreamer.EmitIntValue(R_00B84C_COMPUTE_PGM_RSRC2, 4);
    const uint32_t ComputePGMRSrc2 =
      S_00B84C_LDS_SIZE(LDSBlocks) |
      S_00B02C_SCRATCH_EN(ScratchBlocks > 0);

    OutStreamer.EmitIntValue(ComputePGMRSrc2, 4);

    OutStreamer.EmitIntValue(R_00B860_COMPUTE_TMPRING_SIZE, 4);
    OutStreamer.EmitIntValue(S_00B860_WAVESIZE(ScratchBlocks), 4);

    // TODO: Should probably note flat usage somewhere. SC emits a "FlatPtr32 =
    // 0" comment but I don't see a corresponding field in the register spec.
  } else {
    OutStreamer.EmitIntValue(RsrcReg, 4);
    OutStreamer.EmitIntValue(S_00B028_VGPRS(VGPRBlocks) |
                             S_00B028_SGPRS(SGPRBlocks), 4);
  }

  if (MFI->getShaderType() == ShaderType::PIXEL) {
    OutStreamer.EmitIntValue(R_00B02C_SPI_SHADER_PGM_RSRC2_PS, 4);
    OutStreamer.EmitIntValue(S_00B02C_EXTRA_LDS_SIZE(LDSBlocks), 4);
    OutStreamer.EmitIntValue(R_0286CC_SPI_PS_INPUT_ENA, 4);
    OutStreamer.EmitIntValue(MFI->PSInputAddr, 4);
  }
}
