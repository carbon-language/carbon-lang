//===- AArch64Disassembler.cpp - Disassembler for AArch64 ISA -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the functions necessary to decode AArch64 instruction
// bitpatterns into MCInsts (with the help of TableGenerated information from
// the instruction definitions).
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "arm-disassembler"

#include "AArch64.h"
#include "AArch64RegisterInfo.h"
#include "AArch64Subtarget.h"
#include "Utils/AArch64BaseInfo.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler.h"
#include "llvm/MC/MCFixedLenDisassembler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MemoryObject.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

typedef MCDisassembler::DecodeStatus DecodeStatus;

namespace {
/// AArch64 disassembler for all AArch64 platforms.
class AArch64Disassembler : public MCDisassembler {
  OwningPtr<const MCRegisterInfo> RegInfo;
public:
  /// Initializes the disassembler.
  ///
  AArch64Disassembler(const MCSubtargetInfo &STI, const MCRegisterInfo *Info)
    : MCDisassembler(STI), RegInfo(Info) {
  }

  ~AArch64Disassembler() {}

  /// See MCDisassembler.
  DecodeStatus getInstruction(MCInst &instr,
                              uint64_t &size,
                              const MemoryObject &region,
                              uint64_t address,
                              raw_ostream &vStream,
                              raw_ostream &cStream) const;

  const MCRegisterInfo *getRegInfo() const { return RegInfo.get(); }
};

}

// Forward-declarations used in the auto-generated files.
static DecodeStatus DecodeGPR64RegisterClass(llvm::MCInst &Inst, unsigned RegNo,
                                         uint64_t Address, const void *Decoder);
static DecodeStatus
DecodeGPR64xspRegisterClass(llvm::MCInst &Inst, unsigned RegNo,
                            uint64_t Address, const void *Decoder);

static DecodeStatus DecodeGPR32RegisterClass(llvm::MCInst &Inst, unsigned RegNo,
                                         uint64_t Address, const void *Decoder);
static DecodeStatus
DecodeGPR32wspRegisterClass(llvm::MCInst &Inst, unsigned RegNo,
                            uint64_t Address, const void *Decoder);

static DecodeStatus DecodeFPR8RegisterClass(llvm::MCInst &Inst, unsigned RegNo,
                                         uint64_t Address, const void *Decoder);
static DecodeStatus DecodeFPR16RegisterClass(llvm::MCInst &Inst, unsigned RegNo,
                                         uint64_t Address, const void *Decoder);
static DecodeStatus DecodeFPR32RegisterClass(llvm::MCInst &Inst, unsigned RegNo,
                                         uint64_t Address, const void *Decoder);
static DecodeStatus DecodeFPR64RegisterClass(llvm::MCInst &Inst, unsigned RegNo,
                                         uint64_t Address, const void *Decoder);
static DecodeStatus DecodeFPR128RegisterClass(llvm::MCInst &Inst,
                                              unsigned RegNo, uint64_t Address,
                                              const void *Decoder);
static DecodeStatus DecodeVPR64RegisterClass(llvm::MCInst &Inst, unsigned RegNo,
                                             uint64_t Address,
                                             const void *Decoder);
static DecodeStatus DecodeVPR128RegisterClass(llvm::MCInst &Inst,
                                              unsigned RegNo, uint64_t Address,
                                              const void *Decoder);

static DecodeStatus DecodeAddrRegExtendOperand(llvm::MCInst &Inst,
                                               unsigned OptionHiS,
                                               uint64_t Address,
                                               const void *Decoder);


static DecodeStatus DecodeBitfield32ImmOperand(llvm::MCInst &Inst,
                                               unsigned Imm6Bits,
                                               uint64_t Address,
                                               const void *Decoder);

static DecodeStatus DecodeCVT32FixedPosOperand(llvm::MCInst &Inst,
                                               unsigned Imm6Bits,
                                               uint64_t Address,
                                               const void *Decoder);

static DecodeStatus DecodeFPZeroOperand(llvm::MCInst &Inst,
                                        unsigned RmBits,
                                        uint64_t Address,
                                        const void *Decoder);

template<int RegWidth>
static DecodeStatus DecodeMoveWideImmOperand(llvm::MCInst &Inst,
                                             unsigned FullImm,
                                             uint64_t Address,
                                             const void *Decoder);

template<int RegWidth>
static DecodeStatus DecodeLogicalImmOperand(llvm::MCInst &Inst,
                                            unsigned Bits,
                                            uint64_t Address,
                                            const void *Decoder);

static DecodeStatus DecodeRegExtendOperand(llvm::MCInst &Inst,
                                           unsigned ShiftAmount,
                                           uint64_t Address,
                                           const void *Decoder);
template <A64SE::ShiftExtSpecifiers Ext, bool IsHalf>
static DecodeStatus
DecodeNeonMovImmShiftOperand(llvm::MCInst &Inst, unsigned ShiftAmount,
                             uint64_t Address, const void *Decoder);

static DecodeStatus Decode32BitShiftOperand(llvm::MCInst &Inst,
                                            unsigned ShiftAmount,
                                            uint64_t Address,
                                            const void *Decoder);
static DecodeStatus DecodeBitfieldInstruction(llvm::MCInst &Inst, unsigned Insn,
                                              uint64_t Address,
                                              const void *Decoder);

static DecodeStatus DecodeFMOVLaneInstruction(llvm::MCInst &Inst, unsigned Insn,
                                              uint64_t Address,
                                              const void *Decoder);

static DecodeStatus DecodeLDSTPairInstruction(llvm::MCInst &Inst,
                                              unsigned Insn,
                                              uint64_t Address,
                                              const void *Decoder);

static DecodeStatus DecodeLoadPairExclusiveInstruction(llvm::MCInst &Inst,
                                                       unsigned Val,
                                                       uint64_t Address,
                                                       const void *Decoder);

template<typename SomeNamedImmMapper>
static DecodeStatus DecodeNamedImmOperand(llvm::MCInst &Inst,
                                          unsigned Val,
                                          uint64_t Address,
                                          const void *Decoder);

static DecodeStatus
DecodeSysRegOperand(const A64SysReg::SysRegMapper &InstMapper,
                    llvm::MCInst &Inst, unsigned Val,
                    uint64_t Address, const void *Decoder);

static DecodeStatus DecodeMRSOperand(llvm::MCInst &Inst,
                                     unsigned Val,
                                     uint64_t Address,
                                     const void *Decoder);

static DecodeStatus DecodeMSROperand(llvm::MCInst &Inst,
                                     unsigned Val,
                                     uint64_t Address,
                                     const void *Decoder);


static DecodeStatus DecodeSingleIndexedInstruction(llvm::MCInst &Inst,
                                                   unsigned Val,
                                                   uint64_t Address,
                                                   const void *Decoder);


static bool Check(DecodeStatus &Out, DecodeStatus In);

#include "AArch64GenDisassemblerTables.inc"
#include "AArch64GenInstrInfo.inc"

static bool Check(DecodeStatus &Out, DecodeStatus In) {
  switch (In) {
    case MCDisassembler::Success:
      // Out stays the same.
      return true;
    case MCDisassembler::SoftFail:
      Out = In;
      return true;
    case MCDisassembler::Fail:
      Out = In;
      return false;
  }
  llvm_unreachable("Invalid DecodeStatus!");
}

DecodeStatus AArch64Disassembler::getInstruction(MCInst &MI, uint64_t &Size,
                                                 const MemoryObject &Region,
                                                 uint64_t Address,
                                                 raw_ostream &os,
                                                 raw_ostream &cs) const {
  CommentStream = &cs;

  uint8_t bytes[4];

  // We want to read exactly 4 bytes of data.
  if (Region.readBytes(Address, 4, bytes) == -1) {
    Size = 0;
    return MCDisassembler::Fail;
  }

  // Encoded as a small-endian 32-bit word in the stream.
  uint32_t insn = (bytes[3] << 24) |
    (bytes[2] << 16) |
    (bytes[1] <<  8) |
    (bytes[0] <<  0);

  // Calling the auto-generated decoder function.
  DecodeStatus result = decodeInstruction(DecoderTableA6432, MI, insn, Address,
                                          this, STI);
  if (result != MCDisassembler::Fail) {
    Size = 4;
    return result;
  }

  MI.clear();
  Size = 0;
  return MCDisassembler::Fail;
}

static unsigned getReg(const void *D, unsigned RC, unsigned RegNo) {
  const AArch64Disassembler *Dis = static_cast<const AArch64Disassembler*>(D);
  return Dis->getRegInfo()->getRegClass(RC).getRegister(RegNo);
}

static DecodeStatus DecodeGPR64RegisterClass(llvm::MCInst &Inst, unsigned RegNo,
                                        uint64_t Address, const void *Decoder) {
  if (RegNo > 31)
    return MCDisassembler::Fail;

  uint16_t Register = getReg(Decoder, AArch64::GPR64RegClassID, RegNo);
  Inst.addOperand(MCOperand::CreateReg(Register));
  return MCDisassembler::Success;
}

static DecodeStatus
DecodeGPR64xspRegisterClass(llvm::MCInst &Inst, unsigned RegNo,
                            uint64_t Address, const void *Decoder) {
  if (RegNo > 31)
    return MCDisassembler::Fail;

  uint16_t Register = getReg(Decoder, AArch64::GPR64xspRegClassID, RegNo);
  Inst.addOperand(MCOperand::CreateReg(Register));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeGPR32RegisterClass(llvm::MCInst &Inst, unsigned RegNo,
                                             uint64_t Address,
                                             const void *Decoder) {
  if (RegNo > 31)
    return MCDisassembler::Fail;

  uint16_t Register = getReg(Decoder, AArch64::GPR32RegClassID, RegNo);
  Inst.addOperand(MCOperand::CreateReg(Register));
  return MCDisassembler::Success;
}

static DecodeStatus
DecodeGPR32wspRegisterClass(llvm::MCInst &Inst, unsigned RegNo,
                            uint64_t Address, const void *Decoder) {
  if (RegNo > 31)
    return MCDisassembler::Fail;

  uint16_t Register = getReg(Decoder, AArch64::GPR32wspRegClassID, RegNo);
  Inst.addOperand(MCOperand::CreateReg(Register));
  return MCDisassembler::Success;
}

static DecodeStatus
DecodeFPR8RegisterClass(llvm::MCInst &Inst, unsigned RegNo,
                            uint64_t Address, const void *Decoder) {
  if (RegNo > 31)
    return MCDisassembler::Fail;

  uint16_t Register = getReg(Decoder, AArch64::FPR8RegClassID, RegNo);
  Inst.addOperand(MCOperand::CreateReg(Register));
  return MCDisassembler::Success;
}

static DecodeStatus
DecodeFPR16RegisterClass(llvm::MCInst &Inst, unsigned RegNo,
                            uint64_t Address, const void *Decoder) {
  if (RegNo > 31)
    return MCDisassembler::Fail;

  uint16_t Register = getReg(Decoder, AArch64::FPR16RegClassID, RegNo);
  Inst.addOperand(MCOperand::CreateReg(Register));
  return MCDisassembler::Success;
}


static DecodeStatus
DecodeFPR32RegisterClass(llvm::MCInst &Inst, unsigned RegNo,
                            uint64_t Address, const void *Decoder) {
  if (RegNo > 31)
    return MCDisassembler::Fail;

  uint16_t Register = getReg(Decoder, AArch64::FPR32RegClassID, RegNo);
  Inst.addOperand(MCOperand::CreateReg(Register));
  return MCDisassembler::Success;
}

static DecodeStatus
DecodeFPR64RegisterClass(llvm::MCInst &Inst, unsigned RegNo,
                            uint64_t Address, const void *Decoder) {
  if (RegNo > 31)
    return MCDisassembler::Fail;

  uint16_t Register = getReg(Decoder, AArch64::FPR64RegClassID, RegNo);
  Inst.addOperand(MCOperand::CreateReg(Register));
  return MCDisassembler::Success;
}


static DecodeStatus
DecodeFPR128RegisterClass(llvm::MCInst &Inst, unsigned RegNo,
                            uint64_t Address, const void *Decoder) {
  if (RegNo > 31)
    return MCDisassembler::Fail;

  uint16_t Register = getReg(Decoder, AArch64::FPR128RegClassID, RegNo);
  Inst.addOperand(MCOperand::CreateReg(Register));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeVPR64RegisterClass(llvm::MCInst &Inst, unsigned RegNo,
                                             uint64_t Address,
                                             const void *Decoder) {
  if (RegNo > 31)
    return MCDisassembler::Fail;

  uint16_t Register = getReg(Decoder, AArch64::VPR64RegClassID, RegNo);
  Inst.addOperand(MCOperand::CreateReg(Register));
  return MCDisassembler::Success;
}

static DecodeStatus
DecodeVPR128RegisterClass(llvm::MCInst &Inst, unsigned RegNo,
						  uint64_t Address, const void *Decoder) {
  if (RegNo > 31)
    return MCDisassembler::Fail;

  uint16_t Register = getReg(Decoder, AArch64::VPR128RegClassID, RegNo);
  Inst.addOperand(MCOperand::CreateReg(Register));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeAddrRegExtendOperand(llvm::MCInst &Inst,
                                               unsigned OptionHiS,
                                               uint64_t Address,
                                               const void *Decoder) {
  // Option{1} must be 1. OptionHiS is made up of {Option{2}, Option{1},
  // S}. Hence we want to check bit 1.
  if (!(OptionHiS & 2))
    return MCDisassembler::Fail;

  Inst.addOperand(MCOperand::CreateImm(OptionHiS));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeBitfield32ImmOperand(llvm::MCInst &Inst,
                                               unsigned Imm6Bits,
                                               uint64_t Address,
                                               const void *Decoder) {
  // In the 32-bit variant, bit 6 must be zero. I.e. the immediate must be
  // between 0 and 31.
  if (Imm6Bits > 31)
    return MCDisassembler::Fail;

  Inst.addOperand(MCOperand::CreateImm(Imm6Bits));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeCVT32FixedPosOperand(llvm::MCInst &Inst,
                                               unsigned Imm6Bits,
                                               uint64_t Address,
                                               const void *Decoder) {
  // 1 <= Imm <= 32. Encoded as 64 - Imm so: 63 >= Encoded >= 32.
  if (Imm6Bits < 32)
    return MCDisassembler::Fail;

  Inst.addOperand(MCOperand::CreateImm(Imm6Bits));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeFPZeroOperand(llvm::MCInst &Inst,
                                        unsigned RmBits,
                                        uint64_t Address,
                                        const void *Decoder) {
  // Any bits are valid in the instruction (they're architecturally ignored),
  // but a code generator should insert 0.
  Inst.addOperand(MCOperand::CreateImm(0));
  return MCDisassembler::Success;
}



template<int RegWidth>
static DecodeStatus DecodeMoveWideImmOperand(llvm::MCInst &Inst,
                                             unsigned FullImm,
                                             uint64_t Address,
                                             const void *Decoder) {
  unsigned Imm16 = FullImm & 0xffff;
  unsigned Shift = FullImm >> 16;

  if (RegWidth == 32 && Shift > 1) return MCDisassembler::Fail;

  Inst.addOperand(MCOperand::CreateImm(Imm16));
  Inst.addOperand(MCOperand::CreateImm(Shift));
  return MCDisassembler::Success;
}

template<int RegWidth>
static DecodeStatus DecodeLogicalImmOperand(llvm::MCInst &Inst,
                                            unsigned Bits,
                                            uint64_t Address,
                                            const void *Decoder) {
  uint64_t Imm;
  if (!A64Imms::isLogicalImmBits(RegWidth, Bits, Imm))
    return MCDisassembler::Fail;

  Inst.addOperand(MCOperand::CreateImm(Bits));
  return MCDisassembler::Success;
}


static DecodeStatus DecodeRegExtendOperand(llvm::MCInst &Inst,
                                           unsigned ShiftAmount,
                                           uint64_t Address,
                                           const void *Decoder) {
  // Only values 0-4 are valid for this 3-bit field
  if (ShiftAmount > 4)
    return MCDisassembler::Fail;

  Inst.addOperand(MCOperand::CreateImm(ShiftAmount));
  return MCDisassembler::Success;
}

static DecodeStatus Decode32BitShiftOperand(llvm::MCInst &Inst,
                                            unsigned ShiftAmount,
                                            uint64_t Address,
                                            const void *Decoder) {
  // Only values below 32 are valid for a 32-bit register
  if (ShiftAmount > 31)
    return MCDisassembler::Fail;

  Inst.addOperand(MCOperand::CreateImm(ShiftAmount));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeBitfieldInstruction(llvm::MCInst &Inst, unsigned Insn,
                                              uint64_t Address,
                                              const void *Decoder) {
  unsigned Rd = fieldFromInstruction(Insn, 0, 5);
  unsigned Rn = fieldFromInstruction(Insn, 5, 5);
  unsigned ImmS = fieldFromInstruction(Insn, 10, 6);
  unsigned ImmR = fieldFromInstruction(Insn, 16, 6);
  unsigned SF = fieldFromInstruction(Insn, 31, 1);

  // Undef for 0b11 just in case it occurs. Don't want the compiler to optimise
  // out assertions that it thinks should never be hit.
  enum OpcTypes { SBFM = 0, BFM, UBFM, Undef } Opc;
  Opc = (OpcTypes)fieldFromInstruction(Insn, 29, 2);

  if (!SF) {
    // ImmR and ImmS must be between 0 and 31 for 32-bit instructions.
    if (ImmR > 31 || ImmS > 31)
      return MCDisassembler::Fail;
  }

  if (SF) {
    DecodeGPR64RegisterClass(Inst, Rd, Address, Decoder);
    // BFM MCInsts use Rd as a source too.
    if (Opc == BFM) DecodeGPR64RegisterClass(Inst, Rd, Address, Decoder);
    DecodeGPR64RegisterClass(Inst, Rn, Address, Decoder);
  } else {
    DecodeGPR32RegisterClass(Inst, Rd, Address, Decoder);
    // BFM MCInsts use Rd as a source too.
    if (Opc == BFM) DecodeGPR32RegisterClass(Inst, Rd, Address, Decoder);
    DecodeGPR32RegisterClass(Inst, Rn, Address, Decoder);
  }

  // ASR and LSR have more specific patterns so they won't get here:
  assert(!(ImmS == 31 && !SF && Opc != BFM)
         && "shift should have used auto decode");
  assert(!(ImmS == 63 && SF && Opc != BFM)
         && "shift should have used auto decode");

  // Extension instructions similarly:
  if (Opc == SBFM && ImmR == 0) {
    assert((ImmS != 7 && ImmS != 15) && "extension got here");
    assert((ImmS != 31 || SF == 0) && "extension got here");
  } else if (Opc == UBFM && ImmR == 0) {
    assert((SF != 0 || (ImmS != 7 && ImmS != 15)) && "extension got here");
  }

  if (Opc == UBFM) {
    // It might be a LSL instruction, which actually takes the shift amount
    // itself as an MCInst operand.
    if (SF && (ImmS + 1) % 64 == ImmR) {
      Inst.setOpcode(AArch64::LSLxxi);
      Inst.addOperand(MCOperand::CreateImm(63 - ImmS));
      return MCDisassembler::Success;
    } else if (!SF && (ImmS + 1) % 32 == ImmR) {
      Inst.setOpcode(AArch64::LSLwwi);
      Inst.addOperand(MCOperand::CreateImm(31 - ImmS));
      return MCDisassembler::Success;
    }
  }

  // Otherwise it's definitely either an extract or an insert depending on which
  // of ImmR or ImmS is larger.
  unsigned ExtractOp, InsertOp;
  switch (Opc) {
  default: llvm_unreachable("unexpected instruction trying to decode bitfield");
  case SBFM:
    ExtractOp = SF ? AArch64::SBFXxxii : AArch64::SBFXwwii;
    InsertOp = SF ? AArch64::SBFIZxxii : AArch64::SBFIZwwii;
    break;
  case BFM:
    ExtractOp = SF ? AArch64::BFXILxxii : AArch64::BFXILwwii;
    InsertOp = SF ? AArch64::BFIxxii : AArch64::BFIwwii;
    break;
  case UBFM:
    ExtractOp = SF ? AArch64::UBFXxxii : AArch64::UBFXwwii;
    InsertOp = SF ? AArch64::UBFIZxxii : AArch64::UBFIZwwii;
    break;
  }

  // Otherwise it's a boring insert or extract
  Inst.addOperand(MCOperand::CreateImm(ImmR));
  Inst.addOperand(MCOperand::CreateImm(ImmS));


  if (ImmS < ImmR)
    Inst.setOpcode(InsertOp);
  else
    Inst.setOpcode(ExtractOp);

  return MCDisassembler::Success;
}

static DecodeStatus DecodeFMOVLaneInstruction(llvm::MCInst &Inst, unsigned Insn,
                                              uint64_t Address,
                                              const void *Decoder) {
  // This decoder exists to add the dummy Lane operand to the MCInst, which must
  // be 1 in assembly but has no other real manifestation.
  unsigned Rd = fieldFromInstruction(Insn, 0, 5);
  unsigned Rn = fieldFromInstruction(Insn, 5, 5);
  unsigned IsToVec = fieldFromInstruction(Insn, 16, 1);

  if (IsToVec) {
    DecodeVPR128RegisterClass(Inst, Rd, Address, Decoder);
    DecodeGPR64RegisterClass(Inst, Rn, Address, Decoder);
  } else {
    DecodeGPR64RegisterClass(Inst, Rd, Address, Decoder);
    DecodeVPR128RegisterClass(Inst, Rn, Address, Decoder);
  }

  // Add the lane
  Inst.addOperand(MCOperand::CreateImm(1));

  return MCDisassembler::Success;
}


static DecodeStatus DecodeLDSTPairInstruction(llvm::MCInst &Inst,
                                              unsigned Insn,
                                              uint64_t Address,
                                              const void *Decoder) {
  DecodeStatus Result = MCDisassembler::Success;
  unsigned Rt = fieldFromInstruction(Insn, 0, 5);
  unsigned Rn = fieldFromInstruction(Insn, 5, 5);
  unsigned Rt2 = fieldFromInstruction(Insn, 10, 5);
  unsigned SImm7 = fieldFromInstruction(Insn, 15, 7);
  unsigned L = fieldFromInstruction(Insn, 22, 1);
  unsigned V = fieldFromInstruction(Insn, 26, 1);
  unsigned Opc = fieldFromInstruction(Insn, 30, 2);

  // Not an official name, but it turns out that bit 23 distinguishes indexed
  // from non-indexed operations.
  unsigned Indexed = fieldFromInstruction(Insn, 23, 1);

  if (Indexed && L == 0) {
    // The MCInst for an indexed store has an out operand and 4 ins:
    //    Rn_wb, Rt, Rt2, Rn, Imm
    DecodeGPR64xspRegisterClass(Inst, Rn, Address, Decoder);
  }

  // You shouldn't load to the same register twice in an instruction...
  if (L && Rt == Rt2)
    Result = MCDisassembler::SoftFail;

  // ... or do any operation that writes-back to a transfer register. But note
  // that "stp xzr, xzr, [sp], #4" is fine because xzr and sp are different.
  if (Indexed && V == 0 && Rn != 31 && (Rt == Rn || Rt2 == Rn))
    Result = MCDisassembler::SoftFail;

  // Exactly how we decode the MCInst's registers depends on the Opc and V
  // fields of the instruction. These also obviously determine the size of the
  // operation so we can fill in that information while we're at it.
  if (V) {
    // The instruction operates on the FP/SIMD registers
    switch (Opc) {
    default: return MCDisassembler::Fail;
    case 0:
      DecodeFPR32RegisterClass(Inst, Rt, Address, Decoder);
      DecodeFPR32RegisterClass(Inst, Rt2, Address, Decoder);
      break;
    case 1:
      DecodeFPR64RegisterClass(Inst, Rt, Address, Decoder);
      DecodeFPR64RegisterClass(Inst, Rt2, Address, Decoder);
      break;
    case 2:
      DecodeFPR128RegisterClass(Inst, Rt, Address, Decoder);
      DecodeFPR128RegisterClass(Inst, Rt2, Address, Decoder);
      break;
    }
  } else {
    switch (Opc) {
    default: return MCDisassembler::Fail;
    case 0:
      DecodeGPR32RegisterClass(Inst, Rt, Address, Decoder);
      DecodeGPR32RegisterClass(Inst, Rt2, Address, Decoder);
      break;
    case 1:
      assert(L && "unexpected \"store signed\" attempt");
      DecodeGPR64RegisterClass(Inst, Rt, Address, Decoder);
      DecodeGPR64RegisterClass(Inst, Rt2, Address, Decoder);
      break;
    case 2:
      DecodeGPR64RegisterClass(Inst, Rt, Address, Decoder);
      DecodeGPR64RegisterClass(Inst, Rt2, Address, Decoder);
      break;
    }
  }

  if (Indexed && L == 1) {
    // The MCInst for an indexed load has 3 out operands and an 3 ins:
    //    Rt, Rt2, Rn_wb, Rt2, Rn, Imm
    DecodeGPR64xspRegisterClass(Inst, Rn, Address, Decoder);
  }


  DecodeGPR64xspRegisterClass(Inst, Rn, Address, Decoder);
  Inst.addOperand(MCOperand::CreateImm(SImm7));

  return Result;
}

static DecodeStatus DecodeLoadPairExclusiveInstruction(llvm::MCInst &Inst,
                                                       uint32_t Val,
                                                       uint64_t Address,
                                                       const void *Decoder) {
  unsigned Rt = fieldFromInstruction(Val, 0, 5);
  unsigned Rn = fieldFromInstruction(Val, 5, 5);
  unsigned Rt2 = fieldFromInstruction(Val, 10, 5);
  unsigned MemSize = fieldFromInstruction(Val, 30, 2);

  DecodeStatus S = MCDisassembler::Success;
  if (Rt == Rt2) S = MCDisassembler::SoftFail;

  switch (MemSize) {
    case 2:
      if (!Check(S, DecodeGPR32RegisterClass(Inst, Rt, Address, Decoder)))
        return MCDisassembler::Fail;
      if (!Check(S, DecodeGPR32RegisterClass(Inst, Rt2, Address, Decoder)))
        return MCDisassembler::Fail;
      break;
    case 3:
      if (!Check(S, DecodeGPR64RegisterClass(Inst, Rt, Address, Decoder)))
        return MCDisassembler::Fail;
      if (!Check(S, DecodeGPR64RegisterClass(Inst, Rt2, Address, Decoder)))
        return MCDisassembler::Fail;
      break;
    default:
      llvm_unreachable("Invalid MemSize in DecodeLoadPairExclusiveInstruction");
  }

  if (!Check(S, DecodeGPR64xspRegisterClass(Inst, Rn, Address, Decoder)))
    return MCDisassembler::Fail;

  return S;
}

template<typename SomeNamedImmMapper>
static DecodeStatus DecodeNamedImmOperand(llvm::MCInst &Inst,
                                          unsigned Val,
                                          uint64_t Address,
                                          const void *Decoder) {
  SomeNamedImmMapper Mapper;
  bool ValidNamed;
  Mapper.toString(Val, ValidNamed);
  if (ValidNamed || Mapper.validImm(Val)) {
    Inst.addOperand(MCOperand::CreateImm(Val));
    return MCDisassembler::Success;
  }

  return MCDisassembler::Fail;
}

static DecodeStatus DecodeSysRegOperand(const A64SysReg::SysRegMapper &Mapper,
                                        llvm::MCInst &Inst,
                                        unsigned Val,
                                        uint64_t Address,
                                        const void *Decoder) {
  bool ValidNamed;
  Mapper.toString(Val, ValidNamed);

  Inst.addOperand(MCOperand::CreateImm(Val));

  return ValidNamed ? MCDisassembler::Success : MCDisassembler::Fail;
}

static DecodeStatus DecodeMRSOperand(llvm::MCInst &Inst,
                                     unsigned Val,
                                     uint64_t Address,
                                     const void *Decoder) {
  return DecodeSysRegOperand(A64SysReg::MRSMapper(), Inst, Val, Address,
                             Decoder);
}

static DecodeStatus DecodeMSROperand(llvm::MCInst &Inst,
                                     unsigned Val,
                                     uint64_t Address,
                                     const void *Decoder) {
  return DecodeSysRegOperand(A64SysReg::MSRMapper(), Inst, Val, Address,
                             Decoder);
}

static DecodeStatus DecodeSingleIndexedInstruction(llvm::MCInst &Inst,
                                                   unsigned Insn,
                                                   uint64_t Address,
                                                   const void *Decoder) {
  unsigned Rt = fieldFromInstruction(Insn, 0, 5);
  unsigned Rn = fieldFromInstruction(Insn, 5, 5);
  unsigned Imm9 = fieldFromInstruction(Insn, 12, 9);

  unsigned Opc = fieldFromInstruction(Insn, 22, 2);
  unsigned V = fieldFromInstruction(Insn, 26, 1);
  unsigned Size = fieldFromInstruction(Insn, 30, 2);

  if (Opc == 0 || (V == 1 && Opc == 2)) {
    // It's a store, the MCInst gets: Rn_wb, Rt, Rn, Imm
    DecodeGPR64xspRegisterClass(Inst, Rn, Address, Decoder);
  }

  if (V == 0 && (Opc == 2 || Size == 3)) {
    DecodeGPR64RegisterClass(Inst, Rt, Address, Decoder);
  } else if (V == 0) {
    DecodeGPR32RegisterClass(Inst, Rt, Address, Decoder);
  } else if (V == 1 && (Opc & 2)) {
    DecodeFPR128RegisterClass(Inst, Rt, Address, Decoder);
  } else {
    switch (Size) {
    case 0:
      DecodeFPR8RegisterClass(Inst, Rt, Address, Decoder);
      break;
    case 1:
      DecodeFPR16RegisterClass(Inst, Rt, Address, Decoder);
      break;
    case 2:
      DecodeFPR32RegisterClass(Inst, Rt, Address, Decoder);
      break;
    case 3:
      DecodeFPR64RegisterClass(Inst, Rt, Address, Decoder);
      break;
    }
  }

  if (Opc != 0 && (V != 1 || Opc != 2)) {
    // It's a load, the MCInst gets: Rt, Rn_wb, Rn, Imm
    DecodeGPR64xspRegisterClass(Inst, Rn, Address, Decoder);
  }

  DecodeGPR64xspRegisterClass(Inst, Rn, Address, Decoder);

  Inst.addOperand(MCOperand::CreateImm(Imm9));

  // N.b. The official documentation says undpredictable if Rt == Rn, but this
  // takes place at the architectural rather than encoding level:
  //
  // "STR xzr, [sp], #4" is perfectly valid.
  if (V == 0 && Rt == Rn && Rn != 31)
    return MCDisassembler::SoftFail;
  else
    return MCDisassembler::Success;
}

static MCDisassembler *createAArch64Disassembler(const Target &T,
                                                 const MCSubtargetInfo &STI) {
  return new AArch64Disassembler(STI, T.createMCRegInfo(""));
}

extern "C" void LLVMInitializeAArch64Disassembler() {
  TargetRegistry::RegisterMCDisassembler(TheAArch64Target,
                                         createAArch64Disassembler);
}

template <A64SE::ShiftExtSpecifiers Ext, bool IsHalf>
static DecodeStatus
DecodeNeonMovImmShiftOperand(llvm::MCInst &Inst, unsigned ShiftAmount,
                             uint64_t Address, const void *Decoder) {
  bool IsLSL = false;
  if (Ext == A64SE::LSL)
    IsLSL = true;
  else if (Ext != A64SE::MSL)
    return MCDisassembler::Fail;

  // MSL and LSLH accepts encoded shift amount 0 or 1.
  if ((!IsLSL || (IsLSL && IsHalf)) && ShiftAmount != 0 && ShiftAmount != 1)
    return MCDisassembler::Fail;

  // LSL  accepts encoded shift amount 0, 1, 2 or 3.
  if (IsLSL && ShiftAmount > 3)
    return MCDisassembler::Fail;

  Inst.addOperand(MCOperand::CreateImm(ShiftAmount));
  return MCDisassembler::Success;
}
