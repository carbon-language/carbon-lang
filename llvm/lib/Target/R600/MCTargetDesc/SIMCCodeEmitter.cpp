//===-- SIMCCodeEmitter.cpp - SI Code Emitter -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief The SI code emitter produces machine code that can be executed
/// directly on the GPU device.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "MCTargetDesc/AMDGPUMCCodeEmitter.h"
#include "MCTargetDesc/AMDGPUFixupKinds.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

/// \brief Helper type used in encoding
typedef union {
  int32_t I;
  float F;
} IntFloatUnion;

class SIMCCodeEmitter : public  AMDGPUMCCodeEmitter {
  SIMCCodeEmitter(const SIMCCodeEmitter &) LLVM_DELETED_FUNCTION;
  void operator=(const SIMCCodeEmitter &) LLVM_DELETED_FUNCTION;
  const MCInstrInfo &MCII;
  const MCRegisterInfo &MRI;
  MCContext &Ctx;

  /// \brief Can this operand also contain immediate values?
  bool isSrcOperand(const MCInstrDesc &Desc, unsigned OpNo) const;

  /// \brief Encode an fp or int literal
  uint32_t getLitEncoding(const MCOperand &MO) const;

public:
  SIMCCodeEmitter(const MCInstrInfo &mcii, const MCRegisterInfo &mri,
                  MCContext &ctx)
    : MCII(mcii), MRI(mri), Ctx(ctx) { }

  ~SIMCCodeEmitter() { }

  /// \brief Encode the instruction and write it to the OS.
  void EncodeInstruction(const MCInst &MI, raw_ostream &OS,
                         SmallVectorImpl<MCFixup> &Fixups,
                         const MCSubtargetInfo &STI) const override;

  /// \returns the encoding for an MCOperand.
  uint64_t getMachineOpValue(const MCInst &MI, const MCOperand &MO,
                             SmallVectorImpl<MCFixup> &Fixups,
                             const MCSubtargetInfo &STI) const override;

  /// \brief Use a fixup to encode the simm16 field for SOPP branch
  ///        instructions.
  unsigned getSOPPBrEncoding(const MCInst &MI, unsigned OpNo,
                             SmallVectorImpl<MCFixup> &Fixups,
                             const MCSubtargetInfo &STI) const override;
};

} // End anonymous namespace

MCCodeEmitter *llvm::createSIMCCodeEmitter(const MCInstrInfo &MCII,
                                           const MCRegisterInfo &MRI,
                                           const MCSubtargetInfo &STI,
                                           MCContext &Ctx) {
  return new SIMCCodeEmitter(MCII, MRI, Ctx);
}

bool SIMCCodeEmitter::isSrcOperand(const MCInstrDesc &Desc,
                                   unsigned OpNo) const {

  unsigned RegClass = Desc.OpInfo[OpNo].RegClass;
  return (AMDGPU::SSrc_32RegClassID == RegClass) ||
         (AMDGPU::SSrc_64RegClassID == RegClass) ||
         (AMDGPU::VSrc_32RegClassID == RegClass) ||
         (AMDGPU::VSrc_64RegClassID == RegClass);
}

uint32_t SIMCCodeEmitter::getLitEncoding(const MCOperand &MO) const {

  IntFloatUnion Imm;
  if (MO.isImm())
    Imm.I = MO.getImm();
  else if (MO.isFPImm())
    Imm.F = MO.getFPImm();
  else if (MO.isExpr())
    return 255;
  else
    return ~0;

  if (Imm.I >= 0 && Imm.I <= 64)
    return 128 + Imm.I;

  if (Imm.I >= -16 && Imm.I <= -1)
    return 192 + abs(Imm.I);

  if (Imm.F == 0.5f)
    return 240;

  if (Imm.F == -0.5f)
    return 241;

  if (Imm.F == 1.0f)
    return 242;

  if (Imm.F == -1.0f)
    return 243;

  if (Imm.F == 2.0f)
    return 244;

  if (Imm.F == -2.0f)
    return 245;

  if (Imm.F == 4.0f)
    return 246;

  if (Imm.F == -4.0f)
    return 247;

  return 255;
}

void SIMCCodeEmitter::EncodeInstruction(const MCInst &MI, raw_ostream &OS,
                                       SmallVectorImpl<MCFixup> &Fixups,
                                       const MCSubtargetInfo &STI) const {

  uint64_t Encoding = getBinaryCodeForInstr(MI, Fixups, STI);
  const MCInstrDesc &Desc = MCII.get(MI.getOpcode());
  unsigned bytes = Desc.getSize();

  for (unsigned i = 0; i < bytes; i++) {
    OS.write((uint8_t) ((Encoding >> (8 * i)) & 0xff));
  }

  if (bytes > 4)
    return;

  // Check for additional literals in SRC0/1/2 (Op 1/2/3)
  for (unsigned i = 0, e = MI.getNumOperands(); i < e; ++i) {

    // Check if this operand should be encoded as [SV]Src
    if (!isSrcOperand(Desc, i))
      continue;

    // Is this operand a literal immediate?
    const MCOperand &Op = MI.getOperand(i);
    if (getLitEncoding(Op) != 255)
      continue;

    // Yes! Encode it
    IntFloatUnion Imm;
    if (Op.isImm())
      Imm.I = Op.getImm();
    else if (Op.isFPImm())
      Imm.F = Op.getFPImm();
    else {
      assert(Op.isExpr());
      // This will be replaced with a fixup value.
      Imm.I = 0;
    }

    for (unsigned j = 0; j < 4; j++) {
      OS.write((uint8_t) ((Imm.I >> (8 * j)) & 0xff));
    }

    // Only one literal value allowed
    break;
  }
}

unsigned SIMCCodeEmitter::getSOPPBrEncoding(const MCInst &MI, unsigned OpNo,
                                            SmallVectorImpl<MCFixup> &Fixups,
                                            const MCSubtargetInfo &STI) const {
  const MCOperand &MO = MI.getOperand(OpNo);

  if (MO.isExpr()) {
    const MCExpr *Expr = MO.getExpr();
    MCFixupKind Kind = (MCFixupKind)AMDGPU::fixup_si_sopp_br;
    Fixups.push_back(MCFixup::Create(0, Expr, Kind, MI.getLoc()));
    return 0;
  }

  return getMachineOpValue(MI, MO, Fixups, STI);
}

uint64_t SIMCCodeEmitter::getMachineOpValue(const MCInst &MI,
                                            const MCOperand &MO,
                                       SmallVectorImpl<MCFixup> &Fixups,
                                       const MCSubtargetInfo &STI) const {
  if (MO.isReg())
    return MRI.getEncodingValue(MO.getReg());

  if (MO.isExpr()) {
    const MCSymbolRefExpr *Expr = cast<MCSymbolRefExpr>(MO.getExpr());
    MCFixupKind Kind;
    const MCSymbol *Sym =
        Ctx.GetOrCreateSymbol(StringRef(END_OF_TEXT_LABEL_NAME));

    if (&Expr->getSymbol() == Sym) {
      // Add the offset to the beginning of the constant values.
      Kind = (MCFixupKind)AMDGPU::fixup_si_end_of_text;
    } else {
      // This is used for constant data stored in .rodata.
     Kind = (MCFixupKind)AMDGPU::fixup_si_rodata;
    }
    Fixups.push_back(MCFixup::Create(4, Expr, Kind, MI.getLoc()));
  }

  // Figure out the operand number, needed for isSrcOperand check
  unsigned OpNo = 0;
  for (unsigned e = MI.getNumOperands(); OpNo < e; ++OpNo) {
    if (&MO == &MI.getOperand(OpNo))
      break;
  }

  const MCInstrDesc &Desc = MCII.get(MI.getOpcode());
  if (isSrcOperand(Desc, OpNo)) {
    uint32_t Enc = getLitEncoding(MO);
    if (Enc != ~0U && (Enc != 255 || Desc.getSize() == 4))
      return Enc;

  } else if (MO.isImm())
    return MO.getImm();

  llvm_unreachable("Encoding of this operand type is not supported yet.");
  return 0;
}

