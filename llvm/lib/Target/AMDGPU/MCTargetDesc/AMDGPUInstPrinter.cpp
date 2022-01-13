//===-- AMDGPUInstPrinter.cpp - AMDGPU MC Inst -> ASM ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// \file
//===----------------------------------------------------------------------===//

#include "AMDGPUInstPrinter.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIDefines.h"
#include "SIRegisterInfo.h"
#include "Utils/AMDGPUAsmUtils.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/TargetParser.h"

using namespace llvm;
using namespace llvm::AMDGPU;

static cl::opt<bool> Keep16BitSuffixes(
  "amdgpu-keep-16-bit-reg-suffixes",
  cl::desc("Keep .l and .h suffixes in asm for debugging purposes"),
  cl::init(false),
  cl::ReallyHidden);

void AMDGPUInstPrinter::printRegName(raw_ostream &OS, unsigned RegNo) const {
  // FIXME: The current implementation of
  // AsmParser::parseRegisterOrRegisterNumber in MC implies we either emit this
  // as an integer or we provide a name which represents a physical register.
  // For CFI instructions we really want to emit a name for the DWARF register
  // instead, because there may be multiple DWARF registers corresponding to a
  // single physical register. One case where this problem manifests is with
  // wave32/wave64 where using the physical register name is ambiguous: if we
  // write e.g. `.cfi_undefined v0` we lose information about the wavefront
  // size which we need to encode the register in the final DWARF. Ideally we
  // would extend MC to support parsing DWARF register names so we could do
  // something like `.cfi_undefined dwarf_wave32_v0`. For now we just live with
  // non-pretty DWARF register names in assembly text.
  OS << RegNo;
}

void AMDGPUInstPrinter::printInst(const MCInst *MI, uint64_t Address,
                                  StringRef Annot, const MCSubtargetInfo &STI,
                                  raw_ostream &OS) {
  OS.flush();
  printInstruction(MI, Address, STI, OS);
  printAnnotation(OS, Annot);
}

void AMDGPUInstPrinter::printU4ImmOperand(const MCInst *MI, unsigned OpNo,
                                          const MCSubtargetInfo &STI,
                                          raw_ostream &O) {
  O << formatHex(MI->getOperand(OpNo).getImm() & 0xf);
}

void AMDGPUInstPrinter::printU8ImmOperand(const MCInst *MI, unsigned OpNo,
                                          raw_ostream &O) {
  O << formatHex(MI->getOperand(OpNo).getImm() & 0xff);
}

void AMDGPUInstPrinter::printU16ImmOperand(const MCInst *MI, unsigned OpNo,
                                           const MCSubtargetInfo &STI,
                                           raw_ostream &O) {
  // It's possible to end up with a 32-bit literal used with a 16-bit operand
  // with ignored high bits. Print as 32-bit anyway in that case.
  int64_t Imm = MI->getOperand(OpNo).getImm();
  if (isInt<16>(Imm) || isUInt<16>(Imm))
    O << formatHex(static_cast<uint64_t>(Imm & 0xffff));
  else
    printU32ImmOperand(MI, OpNo, STI, O);
}

void AMDGPUInstPrinter::printU4ImmDecOperand(const MCInst *MI, unsigned OpNo,
                                             raw_ostream &O) {
  O << formatDec(MI->getOperand(OpNo).getImm() & 0xf);
}

void AMDGPUInstPrinter::printU8ImmDecOperand(const MCInst *MI, unsigned OpNo,
                                             raw_ostream &O) {
  O << formatDec(MI->getOperand(OpNo).getImm() & 0xff);
}

void AMDGPUInstPrinter::printU16ImmDecOperand(const MCInst *MI, unsigned OpNo,
                                              raw_ostream &O) {
  O << formatDec(MI->getOperand(OpNo).getImm() & 0xffff);
}

void AMDGPUInstPrinter::printU32ImmOperand(const MCInst *MI, unsigned OpNo,
                                           const MCSubtargetInfo &STI,
                                           raw_ostream &O) {
  O << formatHex(MI->getOperand(OpNo).getImm() & 0xffffffff);
}

void AMDGPUInstPrinter::printNamedBit(const MCInst *MI, unsigned OpNo,
                                      raw_ostream &O, StringRef BitName) {
  if (MI->getOperand(OpNo).getImm()) {
    O << ' ' << BitName;
  }
}

void AMDGPUInstPrinter::printOffen(const MCInst *MI, unsigned OpNo,
                                   raw_ostream &O) {
  printNamedBit(MI, OpNo, O, "offen");
}

void AMDGPUInstPrinter::printIdxen(const MCInst *MI, unsigned OpNo,
                                   raw_ostream &O) {
  printNamedBit(MI, OpNo, O, "idxen");
}

void AMDGPUInstPrinter::printAddr64(const MCInst *MI, unsigned OpNo,
                                    raw_ostream &O) {
  printNamedBit(MI, OpNo, O, "addr64");
}

void AMDGPUInstPrinter::printMBUFOffset(const MCInst *MI, unsigned OpNo,
                                        raw_ostream &O) {
  if (MI->getOperand(OpNo).getImm()) {
    O << " offset:";
    printU16ImmDecOperand(MI, OpNo, O);
  }
}

void AMDGPUInstPrinter::printOffset(const MCInst *MI, unsigned OpNo,
                                    const MCSubtargetInfo &STI,
                                    raw_ostream &O) {
  uint16_t Imm = MI->getOperand(OpNo).getImm();
  if (Imm != 0) {
    O << " offset:";
    printU16ImmDecOperand(MI, OpNo, O);
  }
}

void AMDGPUInstPrinter::printFlatOffset(const MCInst *MI, unsigned OpNo,
                                        const MCSubtargetInfo &STI,
                                        raw_ostream &O) {
  uint16_t Imm = MI->getOperand(OpNo).getImm();
  if (Imm != 0) {
    O << " offset:";

    const MCInstrDesc &Desc = MII.get(MI->getOpcode());
    bool IsFlatSeg = !(Desc.TSFlags &
                       (SIInstrFlags::FlatGlobal | SIInstrFlags::FlatScratch));

    if (IsFlatSeg) { // Unsigned offset
      printU16ImmDecOperand(MI, OpNo, O);
    } else {         // Signed offset
      if (AMDGPU::isGFX10Plus(STI)) {
        O << formatDec(SignExtend32<12>(MI->getOperand(OpNo).getImm()));
      } else {
        O << formatDec(SignExtend32<13>(MI->getOperand(OpNo).getImm()));
      }
    }
  }
}

void AMDGPUInstPrinter::printOffset0(const MCInst *MI, unsigned OpNo,
                                     const MCSubtargetInfo &STI,
                                     raw_ostream &O) {
  if (MI->getOperand(OpNo).getImm()) {
    O << " offset0:";
    printU8ImmDecOperand(MI, OpNo, O);
  }
}

void AMDGPUInstPrinter::printOffset1(const MCInst *MI, unsigned OpNo,
                                     const MCSubtargetInfo &STI,
                                     raw_ostream &O) {
  if (MI->getOperand(OpNo).getImm()) {
    O << " offset1:";
    printU8ImmDecOperand(MI, OpNo, O);
  }
}

void AMDGPUInstPrinter::printSMRDOffset8(const MCInst *MI, unsigned OpNo,
                                        const MCSubtargetInfo &STI,
                                        raw_ostream &O) {
  printU32ImmOperand(MI, OpNo, STI, O);
}

void AMDGPUInstPrinter::printSMEMOffset(const MCInst *MI, unsigned OpNo,
                                        const MCSubtargetInfo &STI,
                                        raw_ostream &O) {
  O << formatHex(MI->getOperand(OpNo).getImm());
}

void AMDGPUInstPrinter::printSMRDLiteralOffset(const MCInst *MI, unsigned OpNo,
                                               const MCSubtargetInfo &STI,
                                               raw_ostream &O) {
  printU32ImmOperand(MI, OpNo, STI, O);
}

void AMDGPUInstPrinter::printGDS(const MCInst *MI, unsigned OpNo,
                                 const MCSubtargetInfo &STI, raw_ostream &O) {
  printNamedBit(MI, OpNo, O, "gds");
}

void AMDGPUInstPrinter::printCPol(const MCInst *MI, unsigned OpNo,
                                  const MCSubtargetInfo &STI, raw_ostream &O) {
  auto Imm = MI->getOperand(OpNo).getImm();
  if (Imm & CPol::GLC)
    O << " glc";
  if (Imm & CPol::SLC)
    O << " slc";
  if ((Imm & CPol::DLC) && AMDGPU::isGFX10Plus(STI))
    O << " dlc";
  if ((Imm & CPol::SCC) && AMDGPU::isGFX90A(STI))
    O << " scc";
  if (Imm & ~CPol::ALL)
    O << " /* unexpected cache policy bit */";
}

void AMDGPUInstPrinter::printSWZ(const MCInst *MI, unsigned OpNo,
                                 const MCSubtargetInfo &STI, raw_ostream &O) {
}

void AMDGPUInstPrinter::printTFE(const MCInst *MI, unsigned OpNo,
                                 const MCSubtargetInfo &STI, raw_ostream &O) {
  printNamedBit(MI, OpNo, O, "tfe");
}

void AMDGPUInstPrinter::printDMask(const MCInst *MI, unsigned OpNo,
                                   const MCSubtargetInfo &STI, raw_ostream &O) {
  if (MI->getOperand(OpNo).getImm()) {
    O << " dmask:";
    printU16ImmOperand(MI, OpNo, STI, O);
  }
}

void AMDGPUInstPrinter::printDim(const MCInst *MI, unsigned OpNo,
                                 const MCSubtargetInfo &STI, raw_ostream &O) {
  unsigned Dim = MI->getOperand(OpNo).getImm();
  O << " dim:SQ_RSRC_IMG_";

  const AMDGPU::MIMGDimInfo *DimInfo = AMDGPU::getMIMGDimInfoByEncoding(Dim);
  if (DimInfo)
    O << DimInfo->AsmSuffix;
  else
    O << Dim;
}

void AMDGPUInstPrinter::printUNorm(const MCInst *MI, unsigned OpNo,
                                   const MCSubtargetInfo &STI, raw_ostream &O) {
  printNamedBit(MI, OpNo, O, "unorm");
}

void AMDGPUInstPrinter::printDA(const MCInst *MI, unsigned OpNo,
                                const MCSubtargetInfo &STI, raw_ostream &O) {
  printNamedBit(MI, OpNo, O, "da");
}

void AMDGPUInstPrinter::printR128A16(const MCInst *MI, unsigned OpNo,
                                  const MCSubtargetInfo &STI, raw_ostream &O) {
  if (STI.hasFeature(AMDGPU::FeatureR128A16))
    printNamedBit(MI, OpNo, O, "a16");
  else
    printNamedBit(MI, OpNo, O, "r128");
}

void AMDGPUInstPrinter::printGFX10A16(const MCInst *MI, unsigned OpNo,
                                  const MCSubtargetInfo &STI, raw_ostream &O) {
  printNamedBit(MI, OpNo, O, "a16");
}

void AMDGPUInstPrinter::printLWE(const MCInst *MI, unsigned OpNo,
                                 const MCSubtargetInfo &STI, raw_ostream &O) {
  printNamedBit(MI, OpNo, O, "lwe");
}

void AMDGPUInstPrinter::printD16(const MCInst *MI, unsigned OpNo,
                                 const MCSubtargetInfo &STI, raw_ostream &O) {
  printNamedBit(MI, OpNo, O, "d16");
}

void AMDGPUInstPrinter::printExpCompr(const MCInst *MI, unsigned OpNo,
                                      const MCSubtargetInfo &STI,
                                      raw_ostream &O) {
  printNamedBit(MI, OpNo, O, "compr");
}

void AMDGPUInstPrinter::printExpVM(const MCInst *MI, unsigned OpNo,
                                   const MCSubtargetInfo &STI,
                                   raw_ostream &O) {
  printNamedBit(MI, OpNo, O, "vm");
}

void AMDGPUInstPrinter::printFORMAT(const MCInst *MI, unsigned OpNo,
                                    const MCSubtargetInfo &STI,
                                    raw_ostream &O) {
}

void AMDGPUInstPrinter::printSymbolicFormat(const MCInst *MI,
                                            const MCSubtargetInfo &STI,
                                            raw_ostream &O) {
  using namespace llvm::AMDGPU::MTBUFFormat;

  int OpNo =
    AMDGPU::getNamedOperandIdx(MI->getOpcode(), AMDGPU::OpName::format);
  assert(OpNo != -1);

  unsigned Val = MI->getOperand(OpNo).getImm();
  if (AMDGPU::isGFX10Plus(STI)) {
    if (Val == UFMT_DEFAULT)
      return;
    if (isValidUnifiedFormat(Val)) {
      O << " format:[" << getUnifiedFormatName(Val) << ']';
    } else {
      O << " format:" << Val;
    }
  } else {
    if (Val == DFMT_NFMT_DEFAULT)
      return;
    if (isValidDfmtNfmt(Val, STI)) {
      unsigned Dfmt;
      unsigned Nfmt;
      decodeDfmtNfmt(Val, Dfmt, Nfmt);
      O << " format:[";
      if (Dfmt != DFMT_DEFAULT) {
        O << getDfmtName(Dfmt);
        if (Nfmt != NFMT_DEFAULT) {
          O << ',';
        }
      }
      if (Nfmt != NFMT_DEFAULT) {
        O << getNfmtName(Nfmt, STI);
      }
      O << ']';
    } else {
      O << " format:" << Val;
    }
  }
}

void AMDGPUInstPrinter::printRegOperand(unsigned RegNo, raw_ostream &O,
                                        const MCRegisterInfo &MRI) {
#if !defined(NDEBUG)
  switch (RegNo) {
  case AMDGPU::FP_REG:
  case AMDGPU::SP_REG:
  case AMDGPU::PRIVATE_RSRC_REG:
    llvm_unreachable("pseudo-register should not ever be emitted");
  case AMDGPU::SCC:
    llvm_unreachable("pseudo scc should not ever be emitted");
  default:
    break;
  }
#endif

  StringRef RegName(getRegisterName(RegNo));
  if (!Keep16BitSuffixes)
    if (!RegName.consume_back(".l"))
      RegName.consume_back(".h");

  O << RegName;
}

void AMDGPUInstPrinter::printVOPDst(const MCInst *MI, unsigned OpNo,
                                    const MCSubtargetInfo &STI,
                                    raw_ostream &O) {
  auto Opcode = MI->getOpcode();
  auto Flags = MII.get(Opcode).TSFlags;

  if (OpNo == 0) {
    if (Flags & SIInstrFlags::VOP3) {
      if (!getVOP3IsSingle(Opcode))
        O << "_e64";
    } else if (Flags & SIInstrFlags::DPP) {
      O << "_dpp";
    } else if (Flags & SIInstrFlags::SDWA) {
      O << "_sdwa";
    } else if (((Flags & SIInstrFlags::VOP1) && !getVOP1IsSingle(Opcode)) ||
               ((Flags & SIInstrFlags::VOP2) && !getVOP2IsSingle(Opcode))) {
      O << "_e32";
    }
    O << " ";
  }

  printOperand(MI, OpNo, STI, O);

  // Print default vcc/vcc_lo operand.
  switch (Opcode) {
  default: break;

  case AMDGPU::V_ADD_CO_CI_U32_e32_gfx10:
  case AMDGPU::V_SUB_CO_CI_U32_e32_gfx10:
  case AMDGPU::V_SUBREV_CO_CI_U32_e32_gfx10:
  case AMDGPU::V_ADD_CO_CI_U32_sdwa_gfx10:
  case AMDGPU::V_SUB_CO_CI_U32_sdwa_gfx10:
  case AMDGPU::V_SUBREV_CO_CI_U32_sdwa_gfx10:
  case AMDGPU::V_ADD_CO_CI_U32_dpp_gfx10:
  case AMDGPU::V_SUB_CO_CI_U32_dpp_gfx10:
  case AMDGPU::V_SUBREV_CO_CI_U32_dpp_gfx10:
  case AMDGPU::V_ADD_CO_CI_U32_dpp8_gfx10:
  case AMDGPU::V_SUB_CO_CI_U32_dpp8_gfx10:
  case AMDGPU::V_SUBREV_CO_CI_U32_dpp8_gfx10:
    printDefaultVccOperand(1, STI, O);
    break;
  }
}

void AMDGPUInstPrinter::printVINTRPDst(const MCInst *MI, unsigned OpNo,
                                       const MCSubtargetInfo &STI, raw_ostream &O) {
  if (AMDGPU::isSI(STI) || AMDGPU::isCI(STI))
    O << " ";
  else
    O << "_e32 ";

  printOperand(MI, OpNo, STI, O);
}

void AMDGPUInstPrinter::printImmediateInt16(uint32_t Imm,
                                            const MCSubtargetInfo &STI,
                                            raw_ostream &O) {
  int16_t SImm = static_cast<int16_t>(Imm);
  if (isInlinableIntLiteral(SImm)) {
    O << SImm;
  } else {
    uint64_t Imm16 = static_cast<uint16_t>(Imm);
    O << formatHex(Imm16);
  }
}

void AMDGPUInstPrinter::printImmediate16(uint32_t Imm,
                                         const MCSubtargetInfo &STI,
                                         raw_ostream &O) {
  int16_t SImm = static_cast<int16_t>(Imm);
  if (isInlinableIntLiteral(SImm)) {
    O << SImm;
    return;
  }

  if (Imm == 0x3C00)
    O<< "1.0";
  else if (Imm == 0xBC00)
    O<< "-1.0";
  else if (Imm == 0x3800)
    O<< "0.5";
  else if (Imm == 0xB800)
    O<< "-0.5";
  else if (Imm == 0x4000)
    O<< "2.0";
  else if (Imm == 0xC000)
    O<< "-2.0";
  else if (Imm == 0x4400)
    O<< "4.0";
  else if (Imm == 0xC400)
    O<< "-4.0";
  else if (Imm == 0x3118 &&
           STI.getFeatureBits()[AMDGPU::FeatureInv2PiInlineImm]) {
    O << "0.15915494";
  } else {
    uint64_t Imm16 = static_cast<uint16_t>(Imm);
    O << formatHex(Imm16);
  }
}

void AMDGPUInstPrinter::printImmediateV216(uint32_t Imm,
                                           const MCSubtargetInfo &STI,
                                           raw_ostream &O) {
  uint16_t Lo16 = static_cast<uint16_t>(Imm);
  printImmediate16(Lo16, STI, O);
}

void AMDGPUInstPrinter::printImmediate32(uint32_t Imm,
                                         const MCSubtargetInfo &STI,
                                         raw_ostream &O) {
  int32_t SImm = static_cast<int32_t>(Imm);
  if (SImm >= -16 && SImm <= 64) {
    O << SImm;
    return;
  }

  if (Imm == FloatToBits(0.0f))
    O << "0.0";
  else if (Imm == FloatToBits(1.0f))
    O << "1.0";
  else if (Imm == FloatToBits(-1.0f))
    O << "-1.0";
  else if (Imm == FloatToBits(0.5f))
    O << "0.5";
  else if (Imm == FloatToBits(-0.5f))
    O << "-0.5";
  else if (Imm == FloatToBits(2.0f))
    O << "2.0";
  else if (Imm == FloatToBits(-2.0f))
    O << "-2.0";
  else if (Imm == FloatToBits(4.0f))
    O << "4.0";
  else if (Imm == FloatToBits(-4.0f))
    O << "-4.0";
  else if (Imm == 0x3e22f983 &&
           STI.getFeatureBits()[AMDGPU::FeatureInv2PiInlineImm])
    O << "0.15915494";
  else
    O << formatHex(static_cast<uint64_t>(Imm));
}

void AMDGPUInstPrinter::printImmediate64(uint64_t Imm,
                                         const MCSubtargetInfo &STI,
                                         raw_ostream &O) {
  int64_t SImm = static_cast<int64_t>(Imm);
  if (SImm >= -16 && SImm <= 64) {
    O << SImm;
    return;
  }

  if (Imm == DoubleToBits(0.0))
    O << "0.0";
  else if (Imm == DoubleToBits(1.0))
    O << "1.0";
  else if (Imm == DoubleToBits(-1.0))
    O << "-1.0";
  else if (Imm == DoubleToBits(0.5))
    O << "0.5";
  else if (Imm == DoubleToBits(-0.5))
    O << "-0.5";
  else if (Imm == DoubleToBits(2.0))
    O << "2.0";
  else if (Imm == DoubleToBits(-2.0))
    O << "-2.0";
  else if (Imm == DoubleToBits(4.0))
    O << "4.0";
  else if (Imm == DoubleToBits(-4.0))
    O << "-4.0";
  else if (Imm == 0x3fc45f306dc9c882 &&
           STI.getFeatureBits()[AMDGPU::FeatureInv2PiInlineImm])
    O << "0.15915494309189532";
  else {
    assert(isUInt<32>(Imm) || Imm == 0x3fc45f306dc9c882);

    // In rare situations, we will have a 32-bit literal in a 64-bit
    // operand. This is technically allowed for the encoding of s_mov_b64.
    O << formatHex(static_cast<uint64_t>(Imm));
  }
}

void AMDGPUInstPrinter::printBLGP(const MCInst *MI, unsigned OpNo,
                                  const MCSubtargetInfo &STI,
                                  raw_ostream &O) {
  unsigned Imm = MI->getOperand(OpNo).getImm();
  if (!Imm)
    return;

  O << " blgp:" << Imm;
}

void AMDGPUInstPrinter::printCBSZ(const MCInst *MI, unsigned OpNo,
                                  const MCSubtargetInfo &STI,
                                  raw_ostream &O) {
  unsigned Imm = MI->getOperand(OpNo).getImm();
  if (!Imm)
    return;

  O << " cbsz:" << Imm;
}

void AMDGPUInstPrinter::printABID(const MCInst *MI, unsigned OpNo,
                                  const MCSubtargetInfo &STI,
                                  raw_ostream &O) {
  unsigned Imm = MI->getOperand(OpNo).getImm();
  if (!Imm)
    return;

  O << " abid:" << Imm;
}

void AMDGPUInstPrinter::printDefaultVccOperand(unsigned OpNo,
                                               const MCSubtargetInfo &STI,
                                               raw_ostream &O) {
  if (OpNo > 0)
    O << ", ";
  printRegOperand(STI.getFeatureBits()[AMDGPU::FeatureWavefrontSize64] ?
                  AMDGPU::VCC : AMDGPU::VCC_LO, O, MRI);
  if (OpNo == 0)
    O << ", ";
}

void AMDGPUInstPrinter::printOperand(const MCInst *MI, unsigned OpNo,
                                     const MCSubtargetInfo &STI,
                                     raw_ostream &O) {
  // Print default vcc/vcc_lo operand of VOPC.
  const MCInstrDesc &Desc = MII.get(MI->getOpcode());
  if (OpNo == 0 && (Desc.TSFlags & SIInstrFlags::VOPC) &&
      (Desc.hasImplicitDefOfPhysReg(AMDGPU::VCC) ||
       Desc.hasImplicitDefOfPhysReg(AMDGPU::VCC_LO)))
    printDefaultVccOperand(OpNo, STI, O);

  if (OpNo >= MI->getNumOperands()) {
    O << "/*Missing OP" << OpNo << "*/";
    return;
  }

  const MCOperand &Op = MI->getOperand(OpNo);
  if (Op.isReg()) {
    printRegOperand(Op.getReg(), O, MRI);
  } else if (Op.isImm()) {
    const uint8_t OpTy = Desc.OpInfo[OpNo].OperandType;
    switch (OpTy) {
    case AMDGPU::OPERAND_REG_IMM_INT32:
    case AMDGPU::OPERAND_REG_IMM_FP32:
    case AMDGPU::OPERAND_REG_IMM_FP32_DEFERRED:
    case AMDGPU::OPERAND_REG_INLINE_C_INT32:
    case AMDGPU::OPERAND_REG_INLINE_C_FP32:
    case AMDGPU::OPERAND_REG_INLINE_AC_INT32:
    case AMDGPU::OPERAND_REG_INLINE_AC_FP32:
    case AMDGPU::OPERAND_REG_IMM_V2INT32:
    case AMDGPU::OPERAND_REG_IMM_V2FP32:
    case AMDGPU::OPERAND_REG_INLINE_C_V2INT32:
    case AMDGPU::OPERAND_REG_INLINE_C_V2FP32:
    case MCOI::OPERAND_IMMEDIATE:
      printImmediate32(Op.getImm(), STI, O);
      break;
    case AMDGPU::OPERAND_REG_IMM_INT64:
    case AMDGPU::OPERAND_REG_IMM_FP64:
    case AMDGPU::OPERAND_REG_INLINE_C_INT64:
    case AMDGPU::OPERAND_REG_INLINE_C_FP64:
    case AMDGPU::OPERAND_REG_INLINE_AC_FP64:
      printImmediate64(Op.getImm(), STI, O);
      break;
    case AMDGPU::OPERAND_REG_INLINE_C_INT16:
    case AMDGPU::OPERAND_REG_INLINE_AC_INT16:
    case AMDGPU::OPERAND_REG_IMM_INT16:
      printImmediateInt16(Op.getImm(), STI, O);
      break;
    case AMDGPU::OPERAND_REG_INLINE_C_FP16:
    case AMDGPU::OPERAND_REG_INLINE_AC_FP16:
    case AMDGPU::OPERAND_REG_IMM_FP16:
    case AMDGPU::OPERAND_REG_IMM_FP16_DEFERRED:
      printImmediate16(Op.getImm(), STI, O);
      break;
    case AMDGPU::OPERAND_REG_IMM_V2INT16:
    case AMDGPU::OPERAND_REG_IMM_V2FP16:
      if (!isUInt<16>(Op.getImm()) &&
          STI.getFeatureBits()[AMDGPU::FeatureVOP3Literal]) {
        printImmediate32(Op.getImm(), STI, O);
        break;
      }

      //  Deal with 16-bit FP inline immediates not working.
      if (OpTy == AMDGPU::OPERAND_REG_IMM_V2FP16) {
        printImmediate16(static_cast<uint16_t>(Op.getImm()), STI, O);
        break;
      }
      LLVM_FALLTHROUGH;
    case AMDGPU::OPERAND_REG_INLINE_C_V2INT16:
    case AMDGPU::OPERAND_REG_INLINE_AC_V2INT16:
      printImmediateInt16(static_cast<uint16_t>(Op.getImm()), STI, O);
      break;
    case AMDGPU::OPERAND_REG_INLINE_C_V2FP16:
    case AMDGPU::OPERAND_REG_INLINE_AC_V2FP16:
      printImmediateV216(Op.getImm(), STI, O);
      break;
    case MCOI::OPERAND_UNKNOWN:
    case MCOI::OPERAND_PCREL:
      O << formatDec(Op.getImm());
      break;
    case MCOI::OPERAND_REGISTER:
      // FIXME: This should be removed and handled somewhere else. Seems to come
      // from a disassembler bug.
      O << "/*invalid immediate*/";
      break;
    default:
      // We hit this for the immediate instruction bits that don't yet have a
      // custom printer.
      llvm_unreachable("unexpected immediate operand type");
    }
  } else if (Op.isDFPImm()) {
    double Value = bit_cast<double>(Op.getDFPImm());
    // We special case 0.0 because otherwise it will be printed as an integer.
    if (Value == 0.0)
      O << "0.0";
    else {
      const MCInstrDesc &Desc = MII.get(MI->getOpcode());
      int RCID = Desc.OpInfo[OpNo].RegClass;
      unsigned RCBits = AMDGPU::getRegBitWidth(MRI.getRegClass(RCID));
      if (RCBits == 32)
        printImmediate32(FloatToBits(Value), STI, O);
      else if (RCBits == 64)
        printImmediate64(DoubleToBits(Value), STI, O);
      else
        llvm_unreachable("Invalid register class size");
    }
  } else if (Op.isExpr()) {
    const MCExpr *Exp = Op.getExpr();
    Exp->print(O, &MAI);
  } else {
    O << "/*INV_OP*/";
  }

  // Print default vcc/vcc_lo operand of v_cndmask_b32_e32.
  switch (MI->getOpcode()) {
  default: break;

  case AMDGPU::V_CNDMASK_B32_e32_gfx10:
  case AMDGPU::V_ADD_CO_CI_U32_e32_gfx10:
  case AMDGPU::V_SUB_CO_CI_U32_e32_gfx10:
  case AMDGPU::V_SUBREV_CO_CI_U32_e32_gfx10:
  case AMDGPU::V_CNDMASK_B32_dpp_gfx10:
  case AMDGPU::V_ADD_CO_CI_U32_dpp_gfx10:
  case AMDGPU::V_SUB_CO_CI_U32_dpp_gfx10:
  case AMDGPU::V_SUBREV_CO_CI_U32_dpp_gfx10:
  case AMDGPU::V_CNDMASK_B32_dpp8_gfx10:
  case AMDGPU::V_ADD_CO_CI_U32_dpp8_gfx10:
  case AMDGPU::V_SUB_CO_CI_U32_dpp8_gfx10:
  case AMDGPU::V_SUBREV_CO_CI_U32_dpp8_gfx10:

  case AMDGPU::V_CNDMASK_B32_e32_gfx6_gfx7:
  case AMDGPU::V_CNDMASK_B32_e32_vi:
    if ((int)OpNo == AMDGPU::getNamedOperandIdx(MI->getOpcode(),
                                                AMDGPU::OpName::src1))
      printDefaultVccOperand(OpNo, STI, O);
    break;
  }

  if (Desc.TSFlags & SIInstrFlags::MTBUF) {
    int SOffsetIdx =
      AMDGPU::getNamedOperandIdx(MI->getOpcode(), AMDGPU::OpName::soffset);
    assert(SOffsetIdx != -1);
    if ((int)OpNo == SOffsetIdx)
      printSymbolicFormat(MI, STI, O);
  }
}

void AMDGPUInstPrinter::printOperandAndFPInputMods(const MCInst *MI,
                                                   unsigned OpNo,
                                                   const MCSubtargetInfo &STI,
                                                   raw_ostream &O) {
  unsigned InputModifiers = MI->getOperand(OpNo).getImm();

  // Use 'neg(...)' instead of '-' to avoid ambiguity.
  // This is important for integer literals because
  // -1 is not the same value as neg(1).
  bool NegMnemo = false;

  if (InputModifiers & SISrcMods::NEG) {
    if (OpNo + 1 < MI->getNumOperands() &&
        (InputModifiers & SISrcMods::ABS) == 0) {
      const MCOperand &Op = MI->getOperand(OpNo + 1);
      NegMnemo = Op.isImm() || Op.isDFPImm();
    }
    if (NegMnemo) {
      O << "neg(";
    } else {
      O << '-';
    }
  }

  if (InputModifiers & SISrcMods::ABS)
    O << '|';
  printOperand(MI, OpNo + 1, STI, O);
  if (InputModifiers & SISrcMods::ABS)
    O << '|';

  if (NegMnemo) {
    O << ')';
  }
}

void AMDGPUInstPrinter::printOperandAndIntInputMods(const MCInst *MI,
                                                    unsigned OpNo,
                                                    const MCSubtargetInfo &STI,
                                                    raw_ostream &O) {
  unsigned InputModifiers = MI->getOperand(OpNo).getImm();
  if (InputModifiers & SISrcMods::SEXT)
    O << "sext(";
  printOperand(MI, OpNo + 1, STI, O);
  if (InputModifiers & SISrcMods::SEXT)
    O << ')';

  // Print default vcc/vcc_lo operand of VOP2b.
  switch (MI->getOpcode()) {
  default: break;

  case AMDGPU::V_CNDMASK_B32_sdwa_gfx10:
  case AMDGPU::V_ADD_CO_CI_U32_sdwa_gfx10:
  case AMDGPU::V_SUB_CO_CI_U32_sdwa_gfx10:
  case AMDGPU::V_SUBREV_CO_CI_U32_sdwa_gfx10:
    if ((int)OpNo + 1 == AMDGPU::getNamedOperandIdx(MI->getOpcode(),
                                                    AMDGPU::OpName::src1))
      printDefaultVccOperand(OpNo, STI, O);
    break;
  }
}

void AMDGPUInstPrinter::printDPP8(const MCInst *MI, unsigned OpNo,
                                  const MCSubtargetInfo &STI,
                                  raw_ostream &O) {
  if (!AMDGPU::isGFX10Plus(STI))
    llvm_unreachable("dpp8 is not supported on ASICs earlier than GFX10");

  unsigned Imm = MI->getOperand(OpNo).getImm();
  O << "dpp8:[" << formatDec(Imm & 0x7);
  for (size_t i = 1; i < 8; ++i) {
    O << ',' << formatDec((Imm >> (3 * i)) & 0x7);
  }
  O << ']';
}

void AMDGPUInstPrinter::printDPPCtrl(const MCInst *MI, unsigned OpNo,
                                     const MCSubtargetInfo &STI,
                                     raw_ostream &O) {
  using namespace AMDGPU::DPP;

  unsigned Imm = MI->getOperand(OpNo).getImm();
  const MCInstrDesc &Desc = MII.get(MI->getOpcode());
  int Src0Idx = AMDGPU::getNamedOperandIdx(MI->getOpcode(),
                                           AMDGPU::OpName::src0);

  if (Src0Idx >= 0 &&
      Desc.OpInfo[Src0Idx].RegClass == AMDGPU::VReg_64RegClassID &&
      !AMDGPU::isLegal64BitDPPControl(Imm)) {
    O << " /* 64 bit dpp only supports row_newbcast */";
    return;
  } else if (Imm <= DppCtrl::QUAD_PERM_LAST) {
    O << "quad_perm:[";
    O << formatDec(Imm & 0x3)         << ',';
    O << formatDec((Imm & 0xc)  >> 2) << ',';
    O << formatDec((Imm & 0x30) >> 4) << ',';
    O << formatDec((Imm & 0xc0) >> 6) << ']';
  } else if ((Imm >= DppCtrl::ROW_SHL_FIRST) &&
             (Imm <= DppCtrl::ROW_SHL_LAST)) {
    O << "row_shl:";
    printU4ImmDecOperand(MI, OpNo, O);
  } else if ((Imm >= DppCtrl::ROW_SHR_FIRST) &&
             (Imm <= DppCtrl::ROW_SHR_LAST)) {
    O << "row_shr:";
    printU4ImmDecOperand(MI, OpNo, O);
  } else if ((Imm >= DppCtrl::ROW_ROR_FIRST) &&
             (Imm <= DppCtrl::ROW_ROR_LAST)) {
    O << "row_ror:";
    printU4ImmDecOperand(MI, OpNo, O);
  } else if (Imm == DppCtrl::WAVE_SHL1) {
    if (AMDGPU::isGFX10Plus(STI)) {
      O << "/* wave_shl is not supported starting from GFX10 */";
      return;
    }
    O << "wave_shl:1";
  } else if (Imm == DppCtrl::WAVE_ROL1) {
    if (AMDGPU::isGFX10Plus(STI)) {
      O << "/* wave_rol is not supported starting from GFX10 */";
      return;
    }
    O << "wave_rol:1";
  } else if (Imm == DppCtrl::WAVE_SHR1) {
    if (AMDGPU::isGFX10Plus(STI)) {
      O << "/* wave_shr is not supported starting from GFX10 */";
      return;
    }
    O << "wave_shr:1";
  } else if (Imm == DppCtrl::WAVE_ROR1) {
    if (AMDGPU::isGFX10Plus(STI)) {
      O << "/* wave_ror is not supported starting from GFX10 */";
      return;
    }
    O << "wave_ror:1";
  } else if (Imm == DppCtrl::ROW_MIRROR) {
    O << "row_mirror";
  } else if (Imm == DppCtrl::ROW_HALF_MIRROR) {
    O << "row_half_mirror";
  } else if (Imm == DppCtrl::BCAST15) {
    if (AMDGPU::isGFX10Plus(STI)) {
      O << "/* row_bcast is not supported starting from GFX10 */";
      return;
    }
    O << "row_bcast:15";
  } else if (Imm == DppCtrl::BCAST31) {
    if (AMDGPU::isGFX10Plus(STI)) {
      O << "/* row_bcast is not supported starting from GFX10 */";
      return;
    }
    O << "row_bcast:31";
  } else if ((Imm >= DppCtrl::ROW_SHARE_FIRST) &&
             (Imm <= DppCtrl::ROW_SHARE_LAST)) {
    if (AMDGPU::isGFX90A(STI)) {
      O << "row_newbcast:";
    } else if (AMDGPU::isGFX10Plus(STI)) {
      O << "row_share:";
    } else {
      O << " /* row_newbcast/row_share is not supported on ASICs earlier "
           "than GFX90A/GFX10 */";
      return;
    }
    printU4ImmDecOperand(MI, OpNo, O);
  } else if ((Imm >= DppCtrl::ROW_XMASK_FIRST) &&
             (Imm <= DppCtrl::ROW_XMASK_LAST)) {
    if (!AMDGPU::isGFX10Plus(STI)) {
      O << "/* row_xmask is not supported on ASICs earlier than GFX10 */";
      return;
    }
    O << "row_xmask:";
    printU4ImmDecOperand(MI, OpNo, O);
  } else {
    O << "/* Invalid dpp_ctrl value */";
  }
}

void AMDGPUInstPrinter::printRowMask(const MCInst *MI, unsigned OpNo,
                                     const MCSubtargetInfo &STI,
                                     raw_ostream &O) {
  O << " row_mask:";
  printU4ImmOperand(MI, OpNo, STI, O);
}

void AMDGPUInstPrinter::printBankMask(const MCInst *MI, unsigned OpNo,
                                      const MCSubtargetInfo &STI,
                                      raw_ostream &O) {
  O << " bank_mask:";
  printU4ImmOperand(MI, OpNo, STI, O);
}

void AMDGPUInstPrinter::printBoundCtrl(const MCInst *MI, unsigned OpNo,
                                       const MCSubtargetInfo &STI,
                                       raw_ostream &O) {
  unsigned Imm = MI->getOperand(OpNo).getImm();
  if (Imm) {
    O << " bound_ctrl:1";
  }
}

void AMDGPUInstPrinter::printFI(const MCInst *MI, unsigned OpNo,
                                const MCSubtargetInfo &STI,
                                raw_ostream &O) {
  using namespace llvm::AMDGPU::DPP;
  unsigned Imm = MI->getOperand(OpNo).getImm();
  if (Imm == DPP_FI_1 || Imm == DPP8_FI_1) {
    O << " fi:1";
  }
}

void AMDGPUInstPrinter::printSDWASel(const MCInst *MI, unsigned OpNo,
                                     raw_ostream &O) {
  using namespace llvm::AMDGPU::SDWA;

  unsigned Imm = MI->getOperand(OpNo).getImm();
  switch (Imm) {
  case SdwaSel::BYTE_0: O << "BYTE_0"; break;
  case SdwaSel::BYTE_1: O << "BYTE_1"; break;
  case SdwaSel::BYTE_2: O << "BYTE_2"; break;
  case SdwaSel::BYTE_3: O << "BYTE_3"; break;
  case SdwaSel::WORD_0: O << "WORD_0"; break;
  case SdwaSel::WORD_1: O << "WORD_1"; break;
  case SdwaSel::DWORD: O << "DWORD"; break;
  default: llvm_unreachable("Invalid SDWA data select operand");
  }
}

void AMDGPUInstPrinter::printSDWADstSel(const MCInst *MI, unsigned OpNo,
                                        const MCSubtargetInfo &STI,
                                        raw_ostream &O) {
  O << "dst_sel:";
  printSDWASel(MI, OpNo, O);
}

void AMDGPUInstPrinter::printSDWASrc0Sel(const MCInst *MI, unsigned OpNo,
                                         const MCSubtargetInfo &STI,
                                         raw_ostream &O) {
  O << "src0_sel:";
  printSDWASel(MI, OpNo, O);
}

void AMDGPUInstPrinter::printSDWASrc1Sel(const MCInst *MI, unsigned OpNo,
                                         const MCSubtargetInfo &STI,
                                         raw_ostream &O) {
  O << "src1_sel:";
  printSDWASel(MI, OpNo, O);
}

void AMDGPUInstPrinter::printSDWADstUnused(const MCInst *MI, unsigned OpNo,
                                           const MCSubtargetInfo &STI,
                                           raw_ostream &O) {
  using namespace llvm::AMDGPU::SDWA;

  O << "dst_unused:";
  unsigned Imm = MI->getOperand(OpNo).getImm();
  switch (Imm) {
  case DstUnused::UNUSED_PAD: O << "UNUSED_PAD"; break;
  case DstUnused::UNUSED_SEXT: O << "UNUSED_SEXT"; break;
  case DstUnused::UNUSED_PRESERVE: O << "UNUSED_PRESERVE"; break;
  default: llvm_unreachable("Invalid SDWA dest_unused operand");
  }
}

void AMDGPUInstPrinter::printExpSrcN(const MCInst *MI, unsigned OpNo,
                                     const MCSubtargetInfo &STI, raw_ostream &O,
                                     unsigned N) {
  unsigned Opc = MI->getOpcode();
  int EnIdx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::en);
  unsigned En = MI->getOperand(EnIdx).getImm();

  int ComprIdx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::compr);

  // If compr is set, print as src0, src0, src1, src1
  if (MI->getOperand(ComprIdx).getImm())
    OpNo = OpNo - N + N / 2;

  if (En & (1 << N))
    printRegOperand(MI->getOperand(OpNo).getReg(), O, MRI);
  else
    O << "off";
}

void AMDGPUInstPrinter::printExpSrc0(const MCInst *MI, unsigned OpNo,
                                     const MCSubtargetInfo &STI,
                                     raw_ostream &O) {
  printExpSrcN(MI, OpNo, STI, O, 0);
}

void AMDGPUInstPrinter::printExpSrc1(const MCInst *MI, unsigned OpNo,
                                     const MCSubtargetInfo &STI,
                                     raw_ostream &O) {
  printExpSrcN(MI, OpNo, STI, O, 1);
}

void AMDGPUInstPrinter::printExpSrc2(const MCInst *MI, unsigned OpNo,
                                     const MCSubtargetInfo &STI,
                                     raw_ostream &O) {
  printExpSrcN(MI, OpNo, STI, O, 2);
}

void AMDGPUInstPrinter::printExpSrc3(const MCInst *MI, unsigned OpNo,
                                     const MCSubtargetInfo &STI,
                                     raw_ostream &O) {
  printExpSrcN(MI, OpNo, STI, O, 3);
}

void AMDGPUInstPrinter::printExpTgt(const MCInst *MI, unsigned OpNo,
                                    const MCSubtargetInfo &STI,
                                    raw_ostream &O) {
  using namespace llvm::AMDGPU::Exp;

  // This is really a 6 bit field.
  unsigned Id = MI->getOperand(OpNo).getImm() & ((1 << 6) - 1);

  int Index;
  StringRef TgtName;
  if (getTgtName(Id, TgtName, Index) && isSupportedTgtId(Id, STI)) {
    O << ' ' << TgtName;
    if (Index >= 0)
      O << Index;
  } else {
    O << " invalid_target_" << Id;
  }
}

static bool allOpsDefaultValue(const int* Ops, int NumOps, int Mod,
                               bool IsPacked, bool HasDstSel) {
  int DefaultValue = IsPacked && (Mod == SISrcMods::OP_SEL_1);

  for (int I = 0; I < NumOps; ++I) {
    if (!!(Ops[I] & Mod) != DefaultValue)
      return false;
  }

  if (HasDstSel && (Ops[0] & SISrcMods::DST_OP_SEL) != 0)
    return false;

  return true;
}

void AMDGPUInstPrinter::printPackedModifier(const MCInst *MI,
                                            StringRef Name,
                                            unsigned Mod,
                                            raw_ostream &O) {
  unsigned Opc = MI->getOpcode();
  int NumOps = 0;
  int Ops[3];

  for (int OpName : { AMDGPU::OpName::src0_modifiers,
                      AMDGPU::OpName::src1_modifiers,
                      AMDGPU::OpName::src2_modifiers }) {
    int Idx = AMDGPU::getNamedOperandIdx(Opc, OpName);
    if (Idx == -1)
      break;

    Ops[NumOps++] = MI->getOperand(Idx).getImm();
  }

  const bool HasDstSel =
    NumOps > 0 &&
    Mod == SISrcMods::OP_SEL_0 &&
    MII.get(MI->getOpcode()).TSFlags & SIInstrFlags::VOP3_OPSEL;

  const bool IsPacked =
    MII.get(MI->getOpcode()).TSFlags & SIInstrFlags::IsPacked;

  if (allOpsDefaultValue(Ops, NumOps, Mod, IsPacked, HasDstSel))
    return;

  O << Name;
  for (int I = 0; I < NumOps; ++I) {
    if (I != 0)
      O << ',';

    O << !!(Ops[I] & Mod);
  }

  if (HasDstSel) {
    O << ',' << !!(Ops[0] & SISrcMods::DST_OP_SEL);
  }

  O << ']';
}

void AMDGPUInstPrinter::printOpSel(const MCInst *MI, unsigned,
                                   const MCSubtargetInfo &STI,
                                   raw_ostream &O) {
  unsigned Opc = MI->getOpcode();
  if (Opc == AMDGPU::V_PERMLANE16_B32_gfx10 ||
      Opc == AMDGPU::V_PERMLANEX16_B32_gfx10) {
    auto FIN = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::src0_modifiers);
    auto BCN = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::src1_modifiers);
    unsigned FI = !!(MI->getOperand(FIN).getImm() & SISrcMods::OP_SEL_0);
    unsigned BC = !!(MI->getOperand(BCN).getImm() & SISrcMods::OP_SEL_0);
    if (FI || BC)
      O << " op_sel:[" << FI << ',' << BC << ']';
    return;
  }

  printPackedModifier(MI, " op_sel:[", SISrcMods::OP_SEL_0, O);
}

void AMDGPUInstPrinter::printOpSelHi(const MCInst *MI, unsigned OpNo,
                                     const MCSubtargetInfo &STI,
                                     raw_ostream &O) {
  printPackedModifier(MI, " op_sel_hi:[", SISrcMods::OP_SEL_1, O);
}

void AMDGPUInstPrinter::printNegLo(const MCInst *MI, unsigned OpNo,
                                   const MCSubtargetInfo &STI,
                                   raw_ostream &O) {
  printPackedModifier(MI, " neg_lo:[", SISrcMods::NEG, O);
}

void AMDGPUInstPrinter::printNegHi(const MCInst *MI, unsigned OpNo,
                                   const MCSubtargetInfo &STI,
                                   raw_ostream &O) {
  printPackedModifier(MI, " neg_hi:[", SISrcMods::NEG_HI, O);
}

void AMDGPUInstPrinter::printInterpSlot(const MCInst *MI, unsigned OpNum,
                                        const MCSubtargetInfo &STI,
                                        raw_ostream &O) {
  unsigned Imm = MI->getOperand(OpNum).getImm();
  switch (Imm) {
  case 0:
    O << "p10";
    break;
  case 1:
    O << "p20";
    break;
  case 2:
    O << "p0";
    break;
  default:
    O << "invalid_param_" << Imm;
  }
}

void AMDGPUInstPrinter::printInterpAttr(const MCInst *MI, unsigned OpNum,
                                        const MCSubtargetInfo &STI,
                                        raw_ostream &O) {
  unsigned Attr = MI->getOperand(OpNum).getImm();
  O << "attr" << Attr;
}

void AMDGPUInstPrinter::printInterpAttrChan(const MCInst *MI, unsigned OpNum,
                                        const MCSubtargetInfo &STI,
                                        raw_ostream &O) {
  unsigned Chan = MI->getOperand(OpNum).getImm();
  O << '.' << "xyzw"[Chan & 0x3];
}

void AMDGPUInstPrinter::printVGPRIndexMode(const MCInst *MI, unsigned OpNo,
                                           const MCSubtargetInfo &STI,
                                           raw_ostream &O) {
  using namespace llvm::AMDGPU::VGPRIndexMode;
  unsigned Val = MI->getOperand(OpNo).getImm();

  if ((Val & ~ENABLE_MASK) != 0) {
    O << formatHex(static_cast<uint64_t>(Val));
  } else {
    O << "gpr_idx(";
    bool NeedComma = false;
    for (unsigned ModeId = ID_MIN; ModeId <= ID_MAX; ++ModeId) {
      if (Val & (1 << ModeId)) {
        if (NeedComma)
          O << ',';
        O << IdSymbolic[ModeId];
        NeedComma = true;
      }
    }
    O << ')';
  }
}

void AMDGPUInstPrinter::printMemOperand(const MCInst *MI, unsigned OpNo,
                                        const MCSubtargetInfo &STI,
                                        raw_ostream &O) {
  printOperand(MI, OpNo, STI, O);
  O  << ", ";
  printOperand(MI, OpNo + 1, STI, O);
}

void AMDGPUInstPrinter::printIfSet(const MCInst *MI, unsigned OpNo,
                                   raw_ostream &O, StringRef Asm,
                                   StringRef Default) {
  const MCOperand &Op = MI->getOperand(OpNo);
  assert(Op.isImm());
  if (Op.getImm() == 1) {
    O << Asm;
  } else {
    O << Default;
  }
}

void AMDGPUInstPrinter::printIfSet(const MCInst *MI, unsigned OpNo,
                                   raw_ostream &O, char Asm) {
  const MCOperand &Op = MI->getOperand(OpNo);
  assert(Op.isImm());
  if (Op.getImm() == 1)
    O << Asm;
}

void AMDGPUInstPrinter::printHigh(const MCInst *MI, unsigned OpNo,
                                  const MCSubtargetInfo &STI,
                                  raw_ostream &O) {
  printNamedBit(MI, OpNo, O, "high");
}

void AMDGPUInstPrinter::printClampSI(const MCInst *MI, unsigned OpNo,
                                     const MCSubtargetInfo &STI,
                                     raw_ostream &O) {
  printNamedBit(MI, OpNo, O, "clamp");
}

void AMDGPUInstPrinter::printOModSI(const MCInst *MI, unsigned OpNo,
                                    const MCSubtargetInfo &STI,
                                    raw_ostream &O) {
  int Imm = MI->getOperand(OpNo).getImm();
  if (Imm == SIOutMods::MUL2)
    O << " mul:2";
  else if (Imm == SIOutMods::MUL4)
    O << " mul:4";
  else if (Imm == SIOutMods::DIV2)
    O << " div:2";
}

void AMDGPUInstPrinter::printSendMsg(const MCInst *MI, unsigned OpNo,
                                     const MCSubtargetInfo &STI,
                                     raw_ostream &O) {
  using namespace llvm::AMDGPU::SendMsg;

  const unsigned Imm16 = MI->getOperand(OpNo).getImm();

  uint16_t MsgId;
  uint16_t OpId;
  uint16_t StreamId;
  decodeMsg(Imm16, MsgId, OpId, StreamId);

  if (isValidMsgId(MsgId, STI) &&
      isValidMsgOp(MsgId, OpId, STI) &&
      isValidMsgStream(MsgId, OpId, StreamId, STI)) {
    O << "sendmsg(" << getMsgName(MsgId);
    if (msgRequiresOp(MsgId)) {
      O << ", " << getMsgOpName(MsgId, OpId);
      if (msgSupportsStream(MsgId, OpId)) {
        O << ", " << StreamId;
      }
    }
    O << ')';
  } else if (encodeMsg(MsgId, OpId, StreamId) == Imm16) {
    O << "sendmsg(" << MsgId << ", " << OpId << ", " << StreamId << ')';
  } else {
    O << Imm16; // Unknown imm16 code.
  }
}

static void printSwizzleBitmask(const uint16_t AndMask,
                                const uint16_t OrMask,
                                const uint16_t XorMask,
                                raw_ostream &O) {
  using namespace llvm::AMDGPU::Swizzle;

  uint16_t Probe0 = ((0            & AndMask) | OrMask) ^ XorMask;
  uint16_t Probe1 = ((BITMASK_MASK & AndMask) | OrMask) ^ XorMask;

  O << "\"";

  for (unsigned Mask = 1 << (BITMASK_WIDTH - 1); Mask > 0; Mask >>= 1) {
    uint16_t p0 = Probe0 & Mask;
    uint16_t p1 = Probe1 & Mask;

    if (p0 == p1) {
      if (p0 == 0) {
        O << "0";
      } else {
        O << "1";
      }
    } else {
      if (p0 == 0) {
        O << "p";
      } else {
        O << "i";
      }
    }
  }

  O << "\"";
}

void AMDGPUInstPrinter::printSwizzle(const MCInst *MI, unsigned OpNo,
                                     const MCSubtargetInfo &STI,
                                     raw_ostream &O) {
  using namespace llvm::AMDGPU::Swizzle;

  uint16_t Imm = MI->getOperand(OpNo).getImm();
  if (Imm == 0) {
    return;
  }

  O << " offset:";

  if ((Imm & QUAD_PERM_ENC_MASK) == QUAD_PERM_ENC) {

    O << "swizzle(" << IdSymbolic[ID_QUAD_PERM];
    for (unsigned I = 0; I < LANE_NUM; ++I) {
      O << ",";
      O << formatDec(Imm & LANE_MASK);
      Imm >>= LANE_SHIFT;
    }
    O << ")";

  } else if ((Imm & BITMASK_PERM_ENC_MASK) == BITMASK_PERM_ENC) {

    uint16_t AndMask = (Imm >> BITMASK_AND_SHIFT) & BITMASK_MASK;
    uint16_t OrMask  = (Imm >> BITMASK_OR_SHIFT)  & BITMASK_MASK;
    uint16_t XorMask = (Imm >> BITMASK_XOR_SHIFT) & BITMASK_MASK;

    if (AndMask == BITMASK_MAX &&
        OrMask == 0 &&
        countPopulation(XorMask) == 1) {

      O << "swizzle(" << IdSymbolic[ID_SWAP];
      O << ",";
      O << formatDec(XorMask);
      O << ")";

    } else if (AndMask == BITMASK_MAX &&
               OrMask == 0 && XorMask > 0 &&
               isPowerOf2_64(XorMask + 1)) {

      O << "swizzle(" << IdSymbolic[ID_REVERSE];
      O << ",";
      O << formatDec(XorMask + 1);
      O << ")";

    } else {

      uint16_t GroupSize = BITMASK_MAX - AndMask + 1;
      if (GroupSize > 1 &&
          isPowerOf2_64(GroupSize) &&
          OrMask < GroupSize &&
          XorMask == 0) {

        O << "swizzle(" << IdSymbolic[ID_BROADCAST];
        O << ",";
        O << formatDec(GroupSize);
        O << ",";
        O << formatDec(OrMask);
        O << ")";

      } else {
        O << "swizzle(" << IdSymbolic[ID_BITMASK_PERM];
        O << ",";
        printSwizzleBitmask(AndMask, OrMask, XorMask, O);
        O << ")";
      }
    }
  } else {
    printU16ImmDecOperand(MI, OpNo, O);
  }
}

void AMDGPUInstPrinter::printWaitFlag(const MCInst *MI, unsigned OpNo,
                                      const MCSubtargetInfo &STI,
                                      raw_ostream &O) {
  AMDGPU::IsaVersion ISA = AMDGPU::getIsaVersion(STI.getCPU());

  unsigned SImm16 = MI->getOperand(OpNo).getImm();
  unsigned Vmcnt, Expcnt, Lgkmcnt;
  decodeWaitcnt(ISA, SImm16, Vmcnt, Expcnt, Lgkmcnt);

  bool NeedSpace = false;

  if (Vmcnt != getVmcntBitMask(ISA)) {
    O << "vmcnt(" << Vmcnt << ')';
    NeedSpace = true;
  }

  if (Expcnt != getExpcntBitMask(ISA)) {
    if (NeedSpace)
      O << ' ';
    O << "expcnt(" << Expcnt << ')';
    NeedSpace = true;
  }

  if (Lgkmcnt != getLgkmcntBitMask(ISA)) {
    if (NeedSpace)
      O << ' ';
    O << "lgkmcnt(" << Lgkmcnt << ')';
  }
}

void AMDGPUInstPrinter::printHwreg(const MCInst *MI, unsigned OpNo,
                                   const MCSubtargetInfo &STI, raw_ostream &O) {
  unsigned Id;
  unsigned Offset;
  unsigned Width;

  using namespace llvm::AMDGPU::Hwreg;
  unsigned Val = MI->getOperand(OpNo).getImm();
  decodeHwreg(Val, Id, Offset, Width);
  StringRef HwRegName = getHwreg(Id, STI);

  O << "hwreg(";
  if (!HwRegName.empty()) {
    O << HwRegName;
  } else {
    O << Id;
  }
  if (Width != WIDTH_DEFAULT_ || Offset != OFFSET_DEFAULT_) {
    O << ", " << Offset << ", " << Width;
  }
  O << ')';
}

void AMDGPUInstPrinter::printEndpgm(const MCInst *MI, unsigned OpNo,
                                    const MCSubtargetInfo &STI,
                                    raw_ostream &O) {
  uint16_t Imm = MI->getOperand(OpNo).getImm();
  if (Imm == 0) {
    return;
  }

  O << ' ' << formatDec(Imm);
}

#include "AMDGPUGenAsmWriter.inc"
