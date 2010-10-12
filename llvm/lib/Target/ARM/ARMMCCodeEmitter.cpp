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

  void EmitImmediate(const MCOperand &Disp,
                     unsigned ImmSize, MCFixupKind FixupKind,
                     unsigned &CurByte, raw_ostream &OS,
                     SmallVectorImpl<MCFixup> &Fixups,
                     int ImmOffset = 0) const;

  void EncodeInstruction(const MCInst &MI, raw_ostream &OS,
                         SmallVectorImpl<MCFixup> &Fixups) const;
};

} // end anonymous namespace

unsigned ARMMCCodeEmitter::getMachineSoImmOpValue(unsigned SoImm) const {
  int SoImmVal = ARM_AM::getSOImmVal(SoImm);
  assert(SoImmVal != -1 && "Not a valid so_imm value!");

  // Encode rotate_imm.
  unsigned Binary = (ARM_AM::getSOImmValRot((unsigned)SoImmVal) >> 1)
    << ARMII::SoRotImmShift;

  // Encode immed_8.
  Binary |= ARM_AM::getSOImmValImm((unsigned)SoImmVal);
  return Binary;
}

MCCodeEmitter *llvm::createARMMCCodeEmitter(const Target &,
                                             TargetMachine &TM,
                                             MCContext &Ctx) {
  return new ARMMCCodeEmitter(TM, Ctx);
}

void ARMMCCodeEmitter::
EmitImmediate(const MCOperand &DispOp, unsigned Size, MCFixupKind FixupKind,
              unsigned &CurByte, raw_ostream &OS,
              SmallVectorImpl<MCFixup> &Fixups, int ImmOffset) const {
  assert(0 && "ARMMCCodeEmitter::EmitImmediate() not yet implemented.");
}

/// getMachineOpValue - Return binary encoding of operand. If the machine
/// operand requires relocation, record the relocation and return zero.
unsigned ARMMCCodeEmitter::getMachineOpValue(const MCInst &MI,
                                             const MCOperand &MO) const {
  if (MO.isReg())
    return getARMRegisterNumbering(MO.getReg());
  else if (MO.isImm()) {
    return static_cast<unsigned>(MO.getImm());
  } else {
#ifndef NDEBUG
    errs() << MO;
#endif
    llvm_unreachable(0);
  }
  return 0;
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
  // FIXME: TableGen doesn't deal well with operands that expand to multiple
  // machine instruction operands, so for now we'll fix those up here.
  // Similarly, operands that are encoded as other than their literal
  // values in the MI.
  unsigned Value = getBinaryCodeForInstr(MI);
  switch (Opcode) {
  default: break;
  case ARM::ADDri:
  case ARM::ANDri:
  case ARM::BICri:
  case ARM::EORri:
  case ARM::ORRri:
  case ARM::SUBri:
    // The 's' bit.
    if (MI.getOperand(5).getReg() == ARM::CPSR)
      Value |= 1 << ARMII::S_BitShift;
    // The shifted immediate value.
    Value |= getMachineSoImmOpValue((unsigned)MI.getOperand(2).getImm());
    break;
  case ARM::ADDrs:
  case ARM::ANDrs:
  case ARM::BICrs:
  case ARM::EORrs:
  case ARM::ORRrs:
  case ARM::SUBrs: {
    // The 's' bit.
    if (MI.getOperand(7).getReg() == ARM::CPSR)
      Value |= 1 << ARMII::S_BitShift;
    // The so_reg operand needs the shift ammount encoded.
    unsigned ShVal = MI.getOperand(4).getImm();
    unsigned ShType = ARM_AM::getShiftOpcEncoding(ARM_AM::getSORegShOp(ShVal));
    unsigned ShAmt = ARM_AM::getSORegOffset(ShVal);
    Value |= ShType << ARMII::ShiftTypeShift;
    Value |= ShAmt << ARMII::ShiftShift;
    break;
  }
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
