//===-- ARM/ARMMCCodeEmitter.cpp - Convert ARM code to machine code -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the ARMMCCodeEmitter class.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "arm-emitter"
#include "ARM.h"
#include "ARMAddressingModes.h"
#include "ARMInstrInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

STATISTIC(MCNumEmitted, "Number of MC instructions emitted");

namespace {
class ARMMCCodeEmitter : public MCCodeEmitter {
  ARMMCCodeEmitter(const ARMMCCodeEmitter &); // DO NOT IMPLEMENT
  void operator=(const ARMMCCodeEmitter &); // DO NOT IMPLEMENT
  const TargetMachine &TM;
  const TargetInstrInfo &TII;
  MCContext &Ctx;

public:
  ARMMCCodeEmitter(TargetMachine &tm, MCContext &ctx)
    : TM(tm), TII(*TM.getInstrInfo()), Ctx(ctx) {
  }

  ~ARMMCCodeEmitter() {}

  unsigned getMachineSoImmOpValue(unsigned SoImm) const;

  // getBinaryCodeForInstr - TableGen'erated function for getting the
  // binary encoding for an instruction.
  unsigned getBinaryCodeForInstr(const MCInst &MI) const;

  /// getMachineOpValue - Return binary encoding of operand. If the machine
  /// operand requires relocation, record the relocation and return zero.
  unsigned getMachineOpValue(const MCInst &MI,const MCOperand &MO) const;

  /// getAddrModeImm12OpValue - Return encoding info for 'reg +/- imm12'
  /// operand.
  unsigned getAddrModeImm12OpValue(const MCInst &MI, unsigned Op) const;

  /// getCCOutOpValue - Return encoding of the 's' bit.
  unsigned getCCOutOpValue(const MCInst &MI, unsigned Op) const {
    // The operand is either reg0 or CPSR. The 's' bit is encoded as '0' or
    // '1' respectively.
    return MI.getOperand(Op).getReg() == ARM::CPSR;
  }

  /// getSOImmOpValue - Return an encoded 12-bit shifted-immediate value.
  unsigned getSOImmOpValue(const MCInst &MI, unsigned Op) const {
    unsigned SoImm = MI.getOperand(Op).getImm();
    int SoImmVal = ARM_AM::getSOImmVal(SoImm);
    assert(SoImmVal != -1 && "Not a valid so_imm value!");

    // Encode rotate_imm.
    unsigned Binary = (ARM_AM::getSOImmValRot((unsigned)SoImmVal) >> 1)
      << ARMII::SoRotImmShift;

    // Encode immed_8.
    Binary |= ARM_AM::getSOImmValImm((unsigned)SoImmVal);
    return Binary;
  }

  /// getSORegOpValue - Return an encoded so_reg shifted register value.
  unsigned getSORegOpValue(const MCInst &MI, unsigned Op) const;

  unsigned getRotImmOpValue(const MCInst &MI, unsigned Op) const {
    switch (MI.getOperand(Op).getImm()) {
    default: assert (0 && "Not a valid rot_imm value!");
    case 0:  return 0;
    case 8:  return 1;
    case 16: return 2;
    case 24: return 3;
    }
  }

  unsigned getImmMinusOneOpValue(const MCInst &MI, unsigned Op) const {
    return MI.getOperand(Op).getImm() - 1;
  }

  unsigned getNEONVcvtImm32OpValue(const MCInst &MI, unsigned Op) const {
    return 64 - MI.getOperand(Op).getImm();
  }

  unsigned getBitfieldInvertedMaskOpValue(const MCInst &MI, unsigned Op) const;

  unsigned getRegisterListOpValue(const MCInst &MI, unsigned Op) const;
  unsigned getAddrMode6RegisterOperand(const MCInst &MI, unsigned Op) const;

  unsigned getNumFixupKinds() const {
    assert(0 && "ARMMCCodeEmitter::getNumFixupKinds() not yet implemented.");
    return 0;
  }

  const MCFixupKindInfo &getFixupKindInfo(MCFixupKind Kind) const {
    static MCFixupKindInfo rtn;
    assert(0 && "ARMMCCodeEmitter::getFixupKindInfo() not yet implemented.");
    return rtn;
  }

  void EmitByte(unsigned char C, unsigned &CurByte, raw_ostream &OS) const {
    OS << (char)C;
    ++CurByte;
  }

  void EmitConstant(uint64_t Val, unsigned Size, unsigned &CurByte,
                    raw_ostream &OS) const {
    // Output the constant in little endian byte order.
    for (unsigned i = 0; i != Size; ++i) {
      EmitByte(Val & 255, CurByte, OS);
      Val >>= 8;
    }
  }

  void EncodeInstruction(const MCInst &MI, raw_ostream &OS,
                         SmallVectorImpl<MCFixup> &Fixups) const;
};

} // end anonymous namespace

MCCodeEmitter *llvm::createARMMCCodeEmitter(const Target &,
                                             TargetMachine &TM,
                                             MCContext &Ctx) {
  return new ARMMCCodeEmitter(TM, Ctx);
}

/// getMachineOpValue - Return binary encoding of operand. If the machine
/// operand requires relocation, record the relocation and return zero.
unsigned ARMMCCodeEmitter::getMachineOpValue(const MCInst &MI,
                                             const MCOperand &MO) const {
  if (MO.isReg()) {
    unsigned regno = getARMRegisterNumbering(MO.getReg());

    // Q registers are encodes as 2x their register number.
    switch (MO.getReg()) {
      case ARM::Q0: case ARM::Q1: case ARM::Q2: case ARM::Q3:
      case ARM::Q4: case ARM::Q5: case ARM::Q6: case ARM::Q7:
      case ARM::Q8: case ARM::Q9: case ARM::Q10: case ARM::Q11:
      case ARM::Q12: case ARM::Q13: case ARM::Q14: case ARM::Q15:
        return 2 * regno;
      default:
        return regno;
    }
  } else if (MO.isImm()) {
    return static_cast<unsigned>(MO.getImm());
  } else if (MO.isFPImm()) {
    return static_cast<unsigned>(APFloat(MO.getFPImm())
                     .bitcastToAPInt().getHiBits(32).getLimitedValue());
  } else {
#ifndef NDEBUG
    errs() << MO;
#endif
    llvm_unreachable(0);
  }
  return 0;
}

/// getAddrModeImm12OpValue - Return encoding info for 'reg +/- imm12'
/// operand.
unsigned ARMMCCodeEmitter::getAddrModeImm12OpValue(const MCInst &MI,
                                                   unsigned OpIdx) const {
  // {17-13} = reg
  // {12}    = (U)nsigned (add == '1', sub == '0')
  // {11-0}  = imm12
  const MCOperand &MO  = MI.getOperand(OpIdx);
  const MCOperand &MO1 = MI.getOperand(OpIdx + 1);
  uint32_t Binary = 0;

  // If The first operand isn't a register, we have a label reference.
  if (!MO.isReg()) {
    Binary |= ARM::PC << 13;     // Rn is PC.
    // FIXME: Add a fixup referencing the label.
    return Binary;
  }

  unsigned Reg = getARMRegisterNumbering(MO.getReg());
  int32_t Imm12 = MO1.getImm();
  bool isAdd = Imm12 >= 0;
  // Special value for #-0
  if (Imm12 == INT32_MIN)
    Imm12 = 0;
  // Immediate is always encoded as positive. The 'U' bit controls add vs sub.
  if (Imm12 < 0)
    Imm12 = -Imm12;
  Binary = Imm12 & 0xfff;
  if (isAdd)
    Binary |= (1 << 12);
  Binary |= (Reg << 13);
  return Binary;
}

unsigned ARMMCCodeEmitter::getSORegOpValue(const MCInst &MI,
                                           unsigned OpIdx) const {
  // Sub-operands are [reg, reg, imm]. The first register is Rm, the reg
  // to be shifted. The second is either Rs, the amount to shift by, or
  // reg0 in which case the imm contains the amount to shift by.
  // {3-0} = Rm.
  // {4} = 1 if reg shift, 0 if imm shift
  // {6-5} = type
  //    If reg shift:
  //      {7} = 0
  //      {11-8} = Rs
  //    else (imm shift)
  //      {11-7} = imm

  const MCOperand &MO  = MI.getOperand(OpIdx);
  const MCOperand &MO1 = MI.getOperand(OpIdx + 1);
  const MCOperand &MO2 = MI.getOperand(OpIdx + 2);
  ARM_AM::ShiftOpc SOpc = ARM_AM::getSORegShOp(MO2.getImm());

  // Encode Rm.
  unsigned Binary = getARMRegisterNumbering(MO.getReg());

  // Encode the shift opcode.
  unsigned SBits = 0;
  unsigned Rs = MO1.getReg();
  if (Rs) {
    // Set shift operand (bit[7:4]).
    // LSL - 0001
    // LSR - 0011
    // ASR - 0101
    // ROR - 0111
    // RRX - 0110 and bit[11:8] clear.
    switch (SOpc) {
    default: llvm_unreachable("Unknown shift opc!");
    case ARM_AM::lsl: SBits = 0x1; break;
    case ARM_AM::lsr: SBits = 0x3; break;
    case ARM_AM::asr: SBits = 0x5; break;
    case ARM_AM::ror: SBits = 0x7; break;
    case ARM_AM::rrx: SBits = 0x6; break;
    }
  } else {
    // Set shift operand (bit[6:4]).
    // LSL - 000
    // LSR - 010
    // ASR - 100
    // ROR - 110
    switch (SOpc) {
    default: llvm_unreachable("Unknown shift opc!");
    case ARM_AM::lsl: SBits = 0x0; break;
    case ARM_AM::lsr: SBits = 0x2; break;
    case ARM_AM::asr: SBits = 0x4; break;
    case ARM_AM::ror: SBits = 0x6; break;
    }
  }
  Binary |= SBits << 4;
  if (SOpc == ARM_AM::rrx)
    return Binary;

  // Encode the shift operation Rs or shift_imm (except rrx).
  if (Rs) {
    // Encode Rs bit[11:8].
    assert(ARM_AM::getSORegOffset(MO2.getImm()) == 0);
    return Binary | (getARMRegisterNumbering(Rs) << ARMII::RegRsShift);
  }

  // Encode shift_imm bit[11:7].
  return Binary | ARM_AM::getSORegOffset(MO2.getImm()) << 7;
}

unsigned ARMMCCodeEmitter::getBitfieldInvertedMaskOpValue(const MCInst &MI,
                                                          unsigned Op) const {
  // 10 bits. lower 5 bits are are the lsb of the mask, high five bits are the
  // msb of the mask.
  const MCOperand &MO = MI.getOperand(Op);
  uint32_t v = ~MO.getImm();
  uint32_t lsb = CountTrailingZeros_32(v);
  uint32_t msb = (32 - CountLeadingZeros_32 (v)) - 1;
  assert (v != 0 && lsb < 32 && msb < 32 && "Illegal bitfield mask!");
  return lsb | (msb << 5);
}

unsigned ARMMCCodeEmitter::getRegisterListOpValue(const MCInst &MI,
                                                  unsigned Op) const {
  // Convert a list of GPRs into a bitfield (R0 -> bit 0). For each
  // register in the list, set the corresponding bit.
  unsigned Binary = 0;
  for (unsigned i = Op, e = MI.getNumOperands(); i < e; ++i) {
    unsigned regno = getARMRegisterNumbering(MI.getOperand(i).getReg());
    Binary |= 1 << regno;
  }
  return Binary;
}

unsigned ARMMCCodeEmitter::getAddrMode6RegisterOperand(const MCInst &MI,
                                                      unsigned Op) const {
  const MCOperand &Reg = MI.getOperand(Op);
  const MCOperand &Imm = MI.getOperand(Op+1);
  
  unsigned RegNo = getARMRegisterNumbering(Reg.getReg());
  unsigned Align = Imm.getImm();
  switch(Align) {
    case 8:  Align = 0x01; break;
    case 16: Align = 0x02; break;
    case 32: Align = 0x03; break;
    default: Align = 0x00;
  }
  return RegNo | (Align << 4);
}

void ARMMCCodeEmitter::
EncodeInstruction(const MCInst &MI, raw_ostream &OS,
                  SmallVectorImpl<MCFixup> &Fixups) const {
  unsigned Opcode = MI.getOpcode();
  const TargetInstrDesc &Desc = TII.get(Opcode);
  uint64_t TSFlags = Desc.TSFlags;
  // Keep track of the current byte being emitted.
  unsigned CurByte = 0;

  // Pseudo instructions don't get encoded.
  if ((TSFlags & ARMII::FormMask) == ARMII::Pseudo)
    return;

  ++MCNumEmitted;  // Keep track of the # of mi's emitted
  unsigned Value = getBinaryCodeForInstr(MI);
  switch (Opcode) {
  default: break;
  }
  EmitConstant(Value, 4, CurByte, OS);
}

// FIXME: These #defines shouldn't be necessary. Instead, tblgen should
// be able to generate code emitter helpers for either variant, like it
// does for the AsmWriter.
#define ARMCodeEmitter ARMMCCodeEmitter
#define MachineInstr MCInst
#include "ARMGenCodeEmitter.inc"
#undef ARMCodeEmitter
#undef MachineInstr
