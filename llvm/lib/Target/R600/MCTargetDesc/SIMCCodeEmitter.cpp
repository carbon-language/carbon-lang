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

#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "MCTargetDesc/AMDGPUMCCodeEmitter.h"
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
class SIMCCodeEmitter : public  AMDGPUMCCodeEmitter {
  SIMCCodeEmitter(const SIMCCodeEmitter &); // DO NOT IMPLEMENT
  void operator=(const SIMCCodeEmitter &); // DO NOT IMPLEMENT
  const MCInstrInfo &MCII;
  const MCRegisterInfo &MRI;
  const MCSubtargetInfo &STI;
  MCContext &Ctx;

public:
  SIMCCodeEmitter(const MCInstrInfo &mcii, const MCRegisterInfo &mri,
                  const MCSubtargetInfo &sti, MCContext &ctx)
    : MCII(mcii), MRI(mri), STI(sti), Ctx(ctx) { }

  ~SIMCCodeEmitter() { }

  /// \breif Encode the instruction and write it to the OS.
  virtual void EncodeInstruction(const MCInst &MI, raw_ostream &OS,
                         SmallVectorImpl<MCFixup> &Fixups) const;

  /// \returns the encoding for an MCOperand.
  virtual uint64_t getMachineOpValue(const MCInst &MI, const MCOperand &MO,
                                     SmallVectorImpl<MCFixup> &Fixups) const;

public:

  /// \brief Encode a sequence of registers with the correct alignment.
  unsigned GPRAlign(const MCInst &MI, unsigned OpNo, unsigned shift) const;

  /// \brief Encoding for when 2 consecutive registers are used
  virtual unsigned GPR2AlignEncode(const MCInst &MI, unsigned OpNo,
                                   SmallVectorImpl<MCFixup> &Fixup) const;

  /// \brief Encoding for when 4 consectuive registers are used
  virtual unsigned GPR4AlignEncode(const MCInst &MI, unsigned OpNo,
                                   SmallVectorImpl<MCFixup> &Fixup) const;
};

} // End anonymous namespace

MCCodeEmitter *llvm::createSIMCCodeEmitter(const MCInstrInfo &MCII,
                                           const MCRegisterInfo &MRI,
                                           const MCSubtargetInfo &STI,
                                           MCContext &Ctx) {
  return new SIMCCodeEmitter(MCII, MRI, STI, Ctx);
}

void SIMCCodeEmitter::EncodeInstruction(const MCInst &MI, raw_ostream &OS,
                                       SmallVectorImpl<MCFixup> &Fixups) const {
  uint64_t Encoding = getBinaryCodeForInstr(MI, Fixups);
  unsigned bytes = MCII.get(MI.getOpcode()).getSize();
  for (unsigned i = 0; i < bytes; i++) {
    OS.write((uint8_t) ((Encoding >> (8 * i)) & 0xff));
  }
}

uint64_t SIMCCodeEmitter::getMachineOpValue(const MCInst &MI,
                                            const MCOperand &MO,
                                       SmallVectorImpl<MCFixup> &Fixups) const {
  if (MO.isReg()) {
    return MRI.getEncodingValue(MO.getReg());
  } else if (MO.isImm()) {
    return MO.getImm();
  } else if (MO.isFPImm()) {
    // XXX: Not all instructions can use inline literals
    // XXX: We should make sure this is a 32-bit constant
    union {
      float F;
      uint32_t I;
    } Imm;
    Imm.F = MO.getFPImm();
    return Imm.I;
  } else if (MO.isExpr()) {
    const MCExpr *Expr = MO.getExpr();
    MCFixupKind Kind = MCFixupKind(FK_PCRel_4);
    Fixups.push_back(MCFixup::Create(0, Expr, Kind, MI.getLoc()));
    return 0;
  } else{
    llvm_unreachable("Encoding of this operand type is not supported yet.");
  }
  return 0;
}

//===----------------------------------------------------------------------===//
// Custom Operand Encodings
//===----------------------------------------------------------------------===//

unsigned SIMCCodeEmitter::GPRAlign(const MCInst &MI, unsigned OpNo,
                                   unsigned shift) const {
  unsigned regCode = MRI.getEncodingValue(MI.getOperand(OpNo).getReg());
  return (regCode & 0xff) >> shift;
}
unsigned SIMCCodeEmitter::GPR2AlignEncode(const MCInst &MI,
                                          unsigned OpNo ,
                                        SmallVectorImpl<MCFixup> &Fixup) const {
  return GPRAlign(MI, OpNo, 1);
}

unsigned SIMCCodeEmitter::GPR4AlignEncode(const MCInst &MI,
                                          unsigned OpNo,
                                        SmallVectorImpl<MCFixup> &Fixup) const {
  return GPRAlign(MI, OpNo, 2);
}
