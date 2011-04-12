//===- ThumbDisassemblerCore.h - Thumb disassembler helpers -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is part of the ARM Disassembler.
// It contains code for disassembling a Thumb instr.  It is to be included by
// ARMDisassemblerCore.cpp because it contains the static DisassembleThumbFrm()
// function which acts as the dispatcher to disassemble a Thumb instruction.
//
//===----------------------------------------------------------------------===//

///////////////////////////////
//                           //
//     Utility Functions     //
//                           //
///////////////////////////////

// Utilities for 16-bit Thumb instructions.
/*
15 14 13 12 11 10  9  8  7  6  5  4  3  2  1  0
               [  tRt ]
                      [ tRm ]  [ tRn ]  [ tRd ]
                         D  [   Rm   ]  [  Rd ]

                      [ imm3]
               [    imm5    ]
                   i     [    imm5   ]
                            [       imm7      ]
                         [       imm8         ]
               [             imm11            ]

            [   cond  ]
*/

// Extract tRt: Inst{10-8}.
static inline unsigned getT1tRt(uint32_t insn) {
  return slice(insn, 10, 8);
}

// Extract tRm: Inst{8-6}.
static inline unsigned getT1tRm(uint32_t insn) {
  return slice(insn, 8, 6);
}

// Extract tRn: Inst{5-3}.
static inline unsigned getT1tRn(uint32_t insn) {
  return slice(insn, 5, 3);
}

// Extract tRd: Inst{2-0}.
static inline unsigned getT1tRd(uint32_t insn) {
  return slice(insn, 2, 0);
}

// Extract [D:Rd]: Inst{7:2-0}.
static inline unsigned getT1Rd(uint32_t insn) {
  return slice(insn, 7, 7) << 3 | slice(insn, 2, 0);
}

// Extract Rm: Inst{6-3}.
static inline unsigned getT1Rm(uint32_t insn) {
  return slice(insn, 6, 3);
}

// Extract imm3: Inst{8-6}.
static inline unsigned getT1Imm3(uint32_t insn) {
  return slice(insn, 8, 6);
}

// Extract imm5: Inst{10-6}.
static inline unsigned getT1Imm5(uint32_t insn) {
  return slice(insn, 10, 6);
}

// Extract i:imm5: Inst{9:7-3}.
static inline unsigned getT1Imm6(uint32_t insn) {
  return slice(insn, 9, 9) << 5 | slice(insn, 7, 3);
}

// Extract imm7: Inst{6-0}.
static inline unsigned getT1Imm7(uint32_t insn) {
  return slice(insn, 6, 0);
}

// Extract imm8: Inst{7-0}.
static inline unsigned getT1Imm8(uint32_t insn) {
  return slice(insn, 7, 0);
}

// Extract imm11: Inst{10-0}.
static inline unsigned getT1Imm11(uint32_t insn) {
  return slice(insn, 10, 0);
}

// Extract cond: Inst{11-8}.
static inline unsigned getT1Cond(uint32_t insn) {
  return slice(insn, 11, 8);
}

static inline bool IsGPR(unsigned RegClass) {
  return RegClass == ARM::GPRRegClassID || RegClass == ARM::rGPRRegClassID;
}

// Utilities for 32-bit Thumb instructions.

// Extract imm4: Inst{19-16}.
static inline unsigned getImm4(uint32_t insn) {
  return slice(insn, 19, 16);
}

// Extract imm3: Inst{14-12}.
static inline unsigned getImm3(uint32_t insn) {
  return slice(insn, 14, 12);
}

// Extract imm8: Inst{7-0}.
static inline unsigned getImm8(uint32_t insn) {
  return slice(insn, 7, 0);
}

// A8.6.61 LDRB (immediate, Thumb) and friends
// +/-: Inst{9}
// imm8: Inst{7-0}
static inline int decodeImm8(uint32_t insn) {
  int Offset = getImm8(insn);
  return slice(insn, 9, 9) ? Offset : -Offset;
}

// Extract imm12: Inst{11-0}.
static inline unsigned getImm12(uint32_t insn) {
  return slice(insn, 11, 0);
}

// A8.6.63 LDRB (literal) and friends
// +/-: Inst{23}
// imm12: Inst{11-0}
static inline int decodeImm12(uint32_t insn) {
  int Offset = getImm12(insn);
  return slice(insn, 23, 23) ? Offset : -Offset;
}

// Extract imm2: Inst{7-6}.
static inline unsigned getImm2(uint32_t insn) {
  return slice(insn, 7, 6);
}

// For BFI, BFC, t2SBFX, and t2UBFX.
// Extract lsb: Inst{14-12:7-6}.
static inline unsigned getLsb(uint32_t insn) {
  return getImm3(insn) << 2 | getImm2(insn);
}

// For BFI and BFC.
// Extract msb: Inst{4-0}.
static inline unsigned getMsb(uint32_t insn) {
  return slice(insn, 4, 0);
}

// For t2SBFX and t2UBFX.
// Extract widthminus1: Inst{4-0}.
static inline unsigned getWidthMinus1(uint32_t insn) {
  return slice(insn, 4, 0);
}

// For t2ADDri12 and t2SUBri12.
// imm12 = i:imm3:imm8;
static inline unsigned getIImm3Imm8(uint32_t insn) {
  return slice(insn, 26, 26) << 11 | getImm3(insn) << 8 | getImm8(insn);
}

// For t2MOVi16 and t2MOVTi16.
// imm16 = imm4:i:imm3:imm8;
static inline unsigned getImm16(uint32_t insn) {
  return getImm4(insn) << 12 | slice(insn, 26, 26) << 11 |
    getImm3(insn) << 8 | getImm8(insn);
}

// Inst{5-4} encodes the shift type.
static inline unsigned getShiftTypeBits(uint32_t insn) {
  return slice(insn, 5, 4);
}

// Inst{14-12}:Inst{7-6} encodes the imm5 shift amount.
static inline unsigned getShiftAmtBits(uint32_t insn) {
  return getImm3(insn) << 2 | getImm2(insn);
}

// A8.6.17 BFC
// Encoding T1 ARMv6T2, ARMv7
// LLVM-specific encoding for #<lsb> and #<width>
static inline bool getBitfieldInvMask(uint32_t insn, uint32_t &mask) {
  uint32_t lsb = getImm3(insn) << 2 | getImm2(insn);
  uint32_t msb = getMsb(insn);
  uint32_t Val = 0;
  if (msb < lsb) {
    DEBUG(errs() << "Encoding error: msb < lsb\n");
    return false;
  }
  for (uint32_t i = lsb; i <= msb; ++i)
    Val |= (1 << i);
  mask = ~Val;
  return true;
}

// A8.4 Shifts applied to a register
// A8.4.1 Constant shifts
// A8.4.3 Pseudocode details of instruction-specified shifts and rotates
//
// decodeImmShift() returns the shift amount and the the shift opcode.
// Note that, as of Jan-06-2010, LLVM does not support rrx shifted operands yet.
static inline unsigned decodeImmShift(unsigned bits2, unsigned imm5,
                                      ARM_AM::ShiftOpc &ShOp) {

  assert(imm5 < 32 && "Invalid imm5 argument");
  switch (bits2) {
  default: assert(0 && "No such value");
  case 0:
    ShOp = (imm5 == 0 ? ARM_AM::no_shift : ARM_AM::lsl);
    return imm5;
  case 1:
    ShOp = ARM_AM::lsr;
    return (imm5 == 0 ? 32 : imm5);
  case 2:
    ShOp = ARM_AM::asr;
    return (imm5 == 0 ? 32 : imm5);
  case 3:
    ShOp = (imm5 == 0 ? ARM_AM::rrx : ARM_AM::ror);
    return (imm5 == 0 ? 1 : imm5);
  }
}

// A6.3.2 Modified immediate constants in Thumb instructions
//
// ThumbExpandImm() returns the modified immediate constant given an imm12 for
// Thumb data-processing instructions with modified immediate.
// See also A6.3.1 Data-processing (modified immediate).
static inline unsigned ThumbExpandImm(unsigned imm12) {
  assert(imm12 <= 0xFFF && "Invalid imm12 argument");

  // If the leading two bits is 0b00, the modified immediate constant is
  // obtained by splatting the low 8 bits into the first byte, every other byte,
  // or every byte of a 32-bit value.
  //
  // Otherwise, a rotate right of '1':imm12<6:0> by the amount imm12<11:7> is
  // performed.

  if (slice(imm12, 11, 10) == 0) {
    unsigned short control = slice(imm12, 9, 8);
    unsigned imm8 = slice(imm12, 7, 0);
    switch (control) {
    default:
      assert(0 && "No such value");
      return 0;
    case 0:
      return imm8;
    case 1:
      return imm8 << 16 | imm8;
    case 2:
      return imm8 << 24 | imm8 << 8;
    case 3:
      return imm8 << 24 | imm8 << 16 | imm8 << 8 | imm8;
    }
  } else {
    // A rotate is required.
    unsigned Val = 1 << 7 | slice(imm12, 6, 0);
    unsigned Amt = slice(imm12, 11, 7);
    return ARM_AM::rotr32(Val, Amt);
  }
}

static inline int decodeImm32_B_EncodingT3(uint32_t insn) {
  bool S = slice(insn, 26, 26);
  bool J1 = slice(insn, 13, 13);
  bool J2 = slice(insn, 11, 11);
  unsigned Imm21 = slice(insn, 21, 16) << 12 | slice(insn, 10, 0) << 1;
  if (S) Imm21 |= 1 << 20;
  if (J2) Imm21 |= 1 << 19;
  if (J1) Imm21 |= 1 << 18;

  return SignExtend32<21>(Imm21);
}

static inline int decodeImm32_B_EncodingT4(uint32_t insn) {
  unsigned S = slice(insn, 26, 26);
  bool I1 = slice(insn, 13, 13) == S;
  bool I2 = slice(insn, 11, 11) == S;
  unsigned Imm25 = slice(insn, 25, 16) << 12 | slice(insn, 10, 0) << 1;
  if (S) Imm25 |= 1 << 24;
  if (I1) Imm25 |= 1 << 23;
  if (I2) Imm25 |= 1 << 22;

  return SignExtend32<25>(Imm25);
}

static inline int decodeImm32_BL(uint32_t insn) {
  unsigned S = slice(insn, 26, 26);
  bool I1 = slice(insn, 13, 13) == S;
  bool I2 = slice(insn, 11, 11) == S;
  unsigned Imm25 = slice(insn, 25, 16) << 12 | slice(insn, 10, 0) << 1;
  if (S) Imm25 |= 1 << 24;
  if (I1) Imm25 |= 1 << 23;
  if (I2) Imm25 |= 1 << 22;

  return SignExtend32<25>(Imm25);
}

static inline int decodeImm32_BLX(uint32_t insn) {
  unsigned S = slice(insn, 26, 26);
  bool I1 = slice(insn, 13, 13) == S;
  bool I2 = slice(insn, 11, 11) == S;
  unsigned Imm25 = slice(insn, 25, 16) << 12 | slice(insn, 10, 1) << 2;
  if (S) Imm25 |= 1 << 24;
  if (I1) Imm25 |= 1 << 23;
  if (I2) Imm25 |= 1 << 22;

  return SignExtend32<25>(Imm25);
}

// See, for example, A8.6.221 SXTAB16.
static inline unsigned decodeRotate(uint32_t insn) {
  unsigned rotate = slice(insn, 5, 4);
  return rotate << 3;
}

///////////////////////////////////////////////
//                                           //
// Thumb1 instruction disassembly functions. //
//                                           //
///////////////////////////////////////////////

// See "Utilities for 16-bit Thumb instructions" for register naming convention.

// A6.2.1 Shift (immediate), add, subtract, move, and compare
//
// shift immediate:         tRd CPSR tRn imm5
// add/sub register:        tRd CPSR tRn tRm
// add/sub 3-bit immediate: tRd CPSR tRn imm3
// add/sub 8-bit immediate: tRt CPSR tRt(TIED_TO) imm8
// mov/cmp immediate:       tRt [CPSR] imm8 (CPSR present for mov)
//
// Special case:
// tMOVSr:                  tRd tRn
static bool DisassembleThumb1General(MCInst &MI, unsigned Opcode, uint32_t insn,
    unsigned short NumOps, unsigned &NumOpsAdded, BO B) {

  const TargetOperandInfo *OpInfo = ARMInsts[Opcode].OpInfo;
  unsigned &OpIdx = NumOpsAdded;

  OpIdx = 0;

  assert(NumOps >= 2 && OpInfo[0].RegClass == ARM::tGPRRegClassID
         && "Invalid arguments");

  bool Imm3 = (Opcode == ARM::tADDi3 || Opcode == ARM::tSUBi3);

  // Use Rt implies use imm8.
  bool UseRt = (Opcode == ARM::tADDi8 || Opcode == ARM::tSUBi8 ||
                Opcode == ARM::tMOVi8 || Opcode == ARM::tCMPi8);

  // Add the destination operand.
  MI.addOperand(MCOperand::CreateReg(
                  getRegisterEnum(B, ARM::tGPRRegClassID,
                                  UseRt ? getT1tRt(insn) : getT1tRd(insn))));
  ++OpIdx;

  // Check whether the next operand to be added is a CCR Register.
  if (OpInfo[OpIdx].RegClass == ARM::CCRRegClassID) {
    assert(OpInfo[OpIdx].isOptionalDef() && "Optional def operand expected");
    MI.addOperand(MCOperand::CreateReg(B->InITBlock() ? 0 : ARM::CPSR));
    ++OpIdx;
  }

  // Check whether the next operand to be added is a Thumb1 Register.
  assert(OpIdx < NumOps && "More operands expected");
  if (OpInfo[OpIdx].RegClass == ARM::tGPRRegClassID) {
    // For UseRt, the reg operand is tied to the first reg operand.
    MI.addOperand(MCOperand::CreateReg(
                    getRegisterEnum(B, ARM::tGPRRegClassID,
                                    UseRt ? getT1tRt(insn) : getT1tRn(insn))));
    ++OpIdx;
  }

  // Special case for tMOVSr.
  if (OpIdx == NumOps)
    return true;

  // The next available operand is either a reg operand or an imm operand.
  if (OpInfo[OpIdx].RegClass == ARM::tGPRRegClassID) {
    // Three register operand instructions.
    MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::tGPRRegClassID,
                                                       getT1tRm(insn))));
  } else {
    assert(OpInfo[OpIdx].RegClass < 0 &&
           !OpInfo[OpIdx].isPredicate() && !OpInfo[OpIdx].isOptionalDef()
           && "Pure imm operand expected");
    unsigned Imm = 0;
    if (UseRt)
      Imm = getT1Imm8(insn);
    else if (Imm3)
      Imm = getT1Imm3(insn);
    else {
      Imm = getT1Imm5(insn);
      ARM_AM::ShiftOpc ShOp = getShiftOpcForBits(slice(insn, 12, 11));
      getImmShiftSE(ShOp, Imm);
    }
    MI.addOperand(MCOperand::CreateImm(Imm));
  }
  ++OpIdx;

  return true;
}

// A6.2.2 Data-processing
//
// tCMPr, tTST, tCMN: tRd tRn
// tMVN, tRSB:        tRd CPSR tRn
// Others:            tRd CPSR tRd(TIED_TO) tRn
static bool DisassembleThumb1DP(MCInst &MI, unsigned Opcode, uint32_t insn,
    unsigned short NumOps, unsigned &NumOpsAdded, BO B) {

  const TargetInstrDesc &TID = ARMInsts[Opcode];
  const TargetOperandInfo *OpInfo = TID.OpInfo;
  unsigned &OpIdx = NumOpsAdded;

  OpIdx = 0;

  assert(NumOps >= 2 && OpInfo[0].RegClass == ARM::tGPRRegClassID &&
         (OpInfo[1].RegClass == ARM::CCRRegClassID
          || OpInfo[1].RegClass == ARM::tGPRRegClassID)
         && "Invalid arguments");

  // Add the destination operand.
  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::tGPRRegClassID,
                                                     getT1tRd(insn))));
  ++OpIdx;

  // Check whether the next operand to be added is a CCR Register.
  if (OpInfo[OpIdx].RegClass == ARM::CCRRegClassID) {
    assert(OpInfo[OpIdx].isOptionalDef() && "Optional def operand expected");
    MI.addOperand(MCOperand::CreateReg(B->InITBlock() ? 0 : ARM::CPSR));
    ++OpIdx;
  }

  // We have either { tRd(TIED_TO), tRn } or { tRn } remaining.
  // Process the TIED_TO operand first.

  assert(OpIdx < NumOps && OpInfo[OpIdx].RegClass == ARM::tGPRRegClassID
         && "Thumb reg operand expected");
  int Idx;
  if ((Idx = TID.getOperandConstraint(OpIdx, TOI::TIED_TO)) != -1) {
    // The reg operand is tied to the first reg operand.
    MI.addOperand(MI.getOperand(Idx));
    ++OpIdx;
  }

  // Process possible next reg operand.
  if (OpIdx < NumOps && OpInfo[OpIdx].RegClass == ARM::tGPRRegClassID) {
    // Add tRn operand.
    MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::tGPRRegClassID,
                                                       getT1tRn(insn))));
    ++OpIdx;
  }

  return true;
}

// A6.2.3 Special data instructions and branch and exchange
//
// tADDhirr: Rd Rd(TIED_TO) Rm
// tCMPhir:  Rd Rm
// tMOVr, tMOVgpr2gpr, tMOVgpr2tgpr, tMOVtgpr2gpr: Rd|tRd Rm|tRn
// tBX_RET: 0 operand
// tBX_RET_vararg: Rm
// tBLXr_r9: Rm
static bool DisassembleThumb1Special(MCInst &MI, unsigned Opcode, uint32_t insn,
    unsigned short NumOps, unsigned &NumOpsAdded, BO B) {

  // tBX_RET has 0 operand.
  if (NumOps == 0)
    return true;

  // BX/BLX has 1 reg operand: Rm.
  if (Opcode == ARM::tBLXr_r9 || Opcode == ARM::tBX_Rm) {
    // Handling the two predicate operands before the reg operand.
    if (!B->DoPredicateOperands(MI, Opcode, insn, NumOps))
      return false;
    MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                       getT1Rm(insn))));
    NumOpsAdded = 3;
    return true;
  }

  const TargetInstrDesc &TID = ARMInsts[Opcode];
  const TargetOperandInfo *OpInfo = TID.OpInfo;
  unsigned &OpIdx = NumOpsAdded;

  OpIdx = 0;

  // Add the destination operand.
  unsigned RegClass = OpInfo[OpIdx].RegClass;
  MI.addOperand(MCOperand::CreateReg(
                  getRegisterEnum(B, RegClass,
                                  IsGPR(RegClass) ? getT1Rd(insn)
                                                  : getT1tRd(insn))));
  ++OpIdx;

  // We have either { Rd(TIED_TO), Rm } or { Rm|tRn } remaining.
  // Process the TIED_TO operand first.

  assert(OpIdx < NumOps && "More operands expected");
  int Idx;
  if ((Idx = TID.getOperandConstraint(OpIdx, TOI::TIED_TO)) != -1) {
    // The reg operand is tied to the first reg operand.
    MI.addOperand(MI.getOperand(Idx));
    ++OpIdx;
  }

  // The next reg operand is either Rm or tRn.
  assert(OpIdx < NumOps && "More operands expected");
  RegClass = OpInfo[OpIdx].RegClass;
  MI.addOperand(MCOperand::CreateReg(
                  getRegisterEnum(B, RegClass,
                                  IsGPR(RegClass) ? getT1Rm(insn)
                                                  : getT1tRn(insn))));
  ++OpIdx;

  return true;
}

// A8.6.59 LDR (literal)
//
// tLDRpci: tRt imm8*4
static bool DisassembleThumb1LdPC(MCInst &MI, unsigned Opcode, uint32_t insn,
    unsigned short NumOps, unsigned &NumOpsAdded, BO B) {

  const TargetOperandInfo *OpInfo = ARMInsts[Opcode].OpInfo;
  if (!OpInfo) return false;

  assert(NumOps >= 2 && OpInfo[0].RegClass == ARM::tGPRRegClassID &&
         (OpInfo[1].RegClass < 0 &&
          !OpInfo[1].isPredicate() &&
          !OpInfo[1].isOptionalDef())
         && "Invalid arguments");

  // Add the destination operand.
  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::tGPRRegClassID,
                                                     getT1tRt(insn))));

  // And the (imm8 << 2) operand.
  MI.addOperand(MCOperand::CreateImm(getT1Imm8(insn) << 2));

  NumOpsAdded = 2;

  return true;
}

// Thumb specific addressing modes (see ARMInstrThumb.td):
//
// t_addrmode_rr := reg + reg
//
// t_addrmode_s4 := reg + reg
//                  reg + imm5 * 4
//
// t_addrmode_s2 := reg + reg
//                  reg + imm5 * 2
//
// t_addrmode_s1 := reg + reg
//                  reg + imm5
//
// t_addrmode_sp := sp + imm8 * 4
//

// A8.6.63 LDRB (literal)
// A8.6.79 LDRSB (literal)
// A8.6.75 LDRH (literal)
// A8.6.83 LDRSH (literal)
// A8.6.59 LDR (literal)
//
// These instrs calculate an address from the PC value and an immediate offset.
// Rd Rn=PC (+/-)imm12 (+ if Inst{23} == 0b1)
static bool DisassembleThumb2Ldpci(MCInst &MI, unsigned Opcode,
    uint32_t insn, unsigned short NumOps, unsigned &NumOpsAdded, BO B) {

  const TargetOperandInfo *OpInfo = ARMInsts[Opcode].OpInfo;
  if (!OpInfo) return false;

  assert(NumOps >= 2 &&
         OpInfo[0].RegClass == ARM::GPRRegClassID &&
         OpInfo[1].RegClass < 0 &&
         "Expect >= 2 operands, first as reg, and second as imm operand");

  // Build the register operand, followed by the (+/-)imm12 immediate.

  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                     decodeRd(insn))));

  MI.addOperand(MCOperand::CreateImm(decodeImm12(insn)));

  NumOpsAdded = 2;

  return true;
}


// A6.2.4 Load/store single data item
//
// Load/Store Register (reg|imm):      tRd tRn imm5|tRm
// Load Register Signed Byte|Halfword: tRd tRn tRm
static bool DisassembleThumb1LdSt(unsigned opA, MCInst &MI, unsigned Opcode,
    uint32_t insn, unsigned short NumOps, unsigned &NumOpsAdded, BO B) {

  const TargetInstrDesc &TID = ARMInsts[Opcode];
  const TargetOperandInfo *OpInfo = TID.OpInfo;
  unsigned &OpIdx = NumOpsAdded;

  assert(NumOps >= 2
         && OpInfo[0].RegClass == ARM::tGPRRegClassID
         && OpInfo[1].RegClass == ARM::tGPRRegClassID
         && "Expect >= 2 operands and first two as thumb reg operands");

  // Add the destination reg and the base reg.
  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::tGPRRegClassID,
                                                     getT1tRd(insn))));
  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::tGPRRegClassID,
                                                     getT1tRn(insn))));
  OpIdx = 2;

  // We have either { imm5 } or { tRm } remaining.
  // Note that STR/LDR (register) should skip the imm5 offset operand for
  // t_addrmode_s[1|2|4].

  assert(OpIdx < NumOps && "More operands expected");

  if (OpInfo[OpIdx].RegClass < 0 && !OpInfo[OpIdx].isPredicate() &&
      !OpInfo[OpIdx].isOptionalDef()) {
    // Table A6-5 16-bit Thumb Load/store instructions
    // opA = 0b0101 for STR/LDR (register) and friends.
    // Otherwise, we have STR/LDR (immediate) and friends.
    assert(opA != 5 && "Immediate operand expected for this opcode");
    MI.addOperand(MCOperand::CreateImm(getT1Imm5(insn)));
    ++OpIdx;
  } else {
    // The next reg operand is tRm, the offset.
    assert(OpIdx < NumOps && OpInfo[OpIdx].RegClass == ARM::tGPRRegClassID
           && "Thumb reg operand expected");
    MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::tGPRRegClassID,
                                                       getT1tRm(insn))));
    ++OpIdx;
  }
  return true;
}

// A6.2.4 Load/store single data item
//
// Load/Store Register SP relative: tRt ARM::SP imm8
static bool DisassembleThumb1LdStSP(MCInst &MI, unsigned Opcode, uint32_t insn,
    unsigned short NumOps, unsigned &NumOpsAdded, BO B) {

  assert((Opcode == ARM::tLDRspi || Opcode == ARM::tSTRspi)
         && "Unexpected opcode");

  const TargetOperandInfo *OpInfo = ARMInsts[Opcode].OpInfo;
  if (!OpInfo) return false;

  assert(NumOps >= 3 &&
         OpInfo[0].RegClass == ARM::tGPRRegClassID &&
         OpInfo[1].RegClass == ARM::GPRRegClassID &&
         (OpInfo[2].RegClass < 0 &&
          !OpInfo[2].isPredicate() &&
          !OpInfo[2].isOptionalDef())
         && "Invalid arguments");

  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::tGPRRegClassID,
                                                     getT1tRt(insn))));
  MI.addOperand(MCOperand::CreateReg(ARM::SP));
  MI.addOperand(MCOperand::CreateImm(getT1Imm8(insn)));
  NumOpsAdded = 3;
  return true;
}

// Table A6-1 16-bit Thumb instruction encoding
// A8.6.10 ADR
//
// tADDrPCi: tRt imm8
static bool DisassembleThumb1AddPCi(MCInst &MI, unsigned Opcode, uint32_t insn,
    unsigned short NumOps, unsigned &NumOpsAdded, BO B) {

  assert(Opcode == ARM::tADDrPCi && "Unexpected opcode");

  const TargetOperandInfo *OpInfo = ARMInsts[Opcode].OpInfo;
  if (!OpInfo) return false;

  assert(NumOps >= 2 && OpInfo[0].RegClass == ARM::tGPRRegClassID &&
         (OpInfo[1].RegClass < 0 &&
          !OpInfo[1].isPredicate() &&
          !OpInfo[1].isOptionalDef())
         && "Invalid arguments");

  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::tGPRRegClassID,
                                                     getT1tRt(insn))));
  MI.addOperand(MCOperand::CreateImm(getT1Imm8(insn)));
  NumOpsAdded = 2;
  return true;
}

// Table A6-1 16-bit Thumb instruction encoding
// A8.6.8 ADD (SP plus immediate)
//
// tADDrSPi: tRt ARM::SP imm8
static bool DisassembleThumb1AddSPi(MCInst &MI, unsigned Opcode, uint32_t insn,
    unsigned short NumOps, unsigned &NumOpsAdded, BO B) {

  assert(Opcode == ARM::tADDrSPi && "Unexpected opcode");

  const TargetOperandInfo *OpInfo = ARMInsts[Opcode].OpInfo;
  if (!OpInfo) return false;

  assert(NumOps >= 3 &&
         OpInfo[0].RegClass == ARM::tGPRRegClassID &&
         OpInfo[1].RegClass == ARM::GPRRegClassID &&
         (OpInfo[2].RegClass < 0 &&
          !OpInfo[2].isPredicate() &&
          !OpInfo[2].isOptionalDef())
         && "Invalid arguments");

  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::tGPRRegClassID,
                                                     getT1tRt(insn))));
  MI.addOperand(MCOperand::CreateReg(ARM::SP));
  MI.addOperand(MCOperand::CreateImm(getT1Imm8(insn)));
  NumOpsAdded = 3;
  return true;
}

// tPUSH, tPOP: Pred-Imm Pred-CCR register_list
//
// where register_list = low registers + [lr] for PUSH or
//                       low registers + [pc] for POP
//
// "low registers" is specified by Inst{7-0}
// lr|pc is specified by Inst{8}
static bool DisassembleThumb1PushPop(MCInst &MI, unsigned Opcode, uint32_t insn,
    unsigned short NumOps, unsigned &NumOpsAdded, BO B) {

  assert((Opcode == ARM::tPUSH || Opcode == ARM::tPOP) && "Unexpected opcode");

  unsigned &OpIdx = NumOpsAdded;

  // Handling the two predicate operands before the reglist.
  if (B->DoPredicateOperands(MI, Opcode, insn, NumOps))
    OpIdx += 2;
  else {
    DEBUG(errs() << "Expected predicate operands not found.\n");
    return false;
  }

  unsigned RegListBits = slice(insn, 8, 8) << (Opcode == ARM::tPUSH ? 14 : 15)
    | slice(insn, 7, 0);

  // Fill the variadic part of reglist.
  for (unsigned i = 0; i < 16; ++i) {
    if ((RegListBits >> i) & 1) {
      MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                         i)));
      ++OpIdx;
    }
  }

  return true;
}

// A6.2.5 Miscellaneous 16-bit instructions
// Delegate to DisassembleThumb1PushPop() for tPUSH & tPOP.
//
// tADDspi, tSUBspi: ARM::SP ARM::SP(TIED_TO) imm7
// t2IT:             firstcond=Inst{7-4} mask=Inst{3-0}
// tCBNZ, tCBZ:      tRd imm6*2
// tBKPT:            imm8
// tNOP, tSEV, tYIELD, tWFE, tWFI:
//   no operand (except predicate pair)
// tSETENDBE, tSETENDLE, :
//   no operand
// Others:           tRd tRn
static bool DisassembleThumb1Misc(MCInst &MI, unsigned Opcode, uint32_t insn,
    unsigned short NumOps, unsigned &NumOpsAdded, BO B) {

  if (NumOps == 0)
    return true;

  if (Opcode == ARM::tPUSH || Opcode == ARM::tPOP)
    return DisassembleThumb1PushPop(MI, Opcode, insn, NumOps, NumOpsAdded, B);

  const TargetOperandInfo *OpInfo = ARMInsts[Opcode].OpInfo;

  // Predicate operands are handled elsewhere.
  if (NumOps == 2 &&
      OpInfo[0].isPredicate() && OpInfo[1].isPredicate() &&
      OpInfo[0].RegClass < 0 && OpInfo[1].RegClass == ARM::CCRRegClassID) {
    return true;
  }

  if (Opcode == ARM::tADDspi || Opcode == ARM::tSUBspi) {
    // Special case handling for tADDspi and tSUBspi.
    // A8.6.8 ADD (SP plus immediate) & A8.6.215 SUB (SP minus immediate)
    MI.addOperand(MCOperand::CreateReg(ARM::SP));
    MI.addOperand(MCOperand::CreateReg(ARM::SP));
    MI.addOperand(MCOperand::CreateImm(getT1Imm7(insn)));
    NumOpsAdded = 3;
    return true;
  }

  if (Opcode == ARM::t2IT) {
    // Special case handling for If-Then.
    // A8.6.50 IT
    // Tag the (firstcond[0] bit << 4) along with mask.

    // firstcond
    MI.addOperand(MCOperand::CreateImm(slice(insn, 7, 4)));

    // firstcond[0] and mask
    MI.addOperand(MCOperand::CreateImm(slice(insn, 4, 0)));
    NumOpsAdded = 2;
    return true;
  }

  if (Opcode == ARM::tBKPT) {
    MI.addOperand(MCOperand::CreateImm(getT1Imm8(insn))); // breakpoint value
    NumOpsAdded = 1;
    return true;
  }

  // CPS has a singleton $opt operand that contains the following information:
  // The first op would be 0b10 as enable and 0b11 as disable in regular ARM,
  // but in Thumb it's is 0 as enable and 1 as disable. So map it to ARM's
  // default one. The second get the AIF flags from Inst{2-0}.
  if (Opcode == ARM::tCPS) {
    MI.addOperand(MCOperand::CreateImm(2 + slice(insn, 4, 4)));
    MI.addOperand(MCOperand::CreateImm(slice(insn, 2, 0)));
    NumOpsAdded = 2;
    return true;
  }

  assert(NumOps >= 2 && OpInfo[0].RegClass == ARM::tGPRRegClassID &&
         (OpInfo[1].RegClass < 0 || OpInfo[1].RegClass==ARM::tGPRRegClassID)
         && "Expect >=2 operands");

  // Add the destination operand.
  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::tGPRRegClassID,
                                                     getT1tRd(insn))));

  if (OpInfo[1].RegClass == ARM::tGPRRegClassID) {
    // Two register instructions.
    MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::tGPRRegClassID,
                                                       getT1tRn(insn))));
  } else {
    // CBNZ, CBZ
    assert((Opcode == ARM::tCBNZ || Opcode == ARM::tCBZ) &&"Unexpected opcode");
    MI.addOperand(MCOperand::CreateImm(getT1Imm6(insn) * 2));
  }

  NumOpsAdded = 2;

  return true;
}

// A8.6.53  LDM / LDMIA
// A8.6.189 STM / STMIA
//
// tLDMIA_UPD/tSTMIA_UPD: tRt tRt AM4ModeImm Pred-Imm Pred-CCR register_list
// tLDMIA:                tRt AM4ModeImm Pred-Imm Pred-CCR register_list
static bool DisassembleThumb1LdStMul(bool Ld, MCInst &MI, unsigned Opcode,
                                     uint32_t insn, unsigned short NumOps,
                                     unsigned &NumOpsAdded, BO B) {
  assert((Opcode == ARM::tLDMIA || Opcode == ARM::tLDMIA_UPD ||
          Opcode == ARM::tSTMIA_UPD) && "Unexpected opcode");

  unsigned tRt = getT1tRt(insn);
  NumOpsAdded = 0;

  // WB register, if necessary.
  if (Opcode == ARM::tLDMIA_UPD || Opcode == ARM::tSTMIA_UPD) {
    MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                       tRt)));
    ++NumOpsAdded;
  }

  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                     tRt)));
  ++NumOpsAdded;

  // Handling the two predicate operands before the reglist.
  if (B->DoPredicateOperands(MI, Opcode, insn, NumOps)) {
    NumOpsAdded += 2;
  } else {
    DEBUG(errs() << "Expected predicate operands not found.\n");
    return false;
  }

  unsigned RegListBits = slice(insn, 7, 0);

  // Fill the variadic part of reglist.
  for (unsigned i = 0; i < 8; ++i)
    if ((RegListBits >> i) & 1) {
      MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::tGPRRegClassID,
                                                         i)));
      ++NumOpsAdded;
    }

  return true;
}

static bool DisassembleThumb1LdMul(MCInst &MI, unsigned Opcode, uint32_t insn,
    unsigned short NumOps, unsigned &NumOpsAdded, BO B) {
  return DisassembleThumb1LdStMul(true, MI, Opcode, insn, NumOps, NumOpsAdded,
                                  B);
}

static bool DisassembleThumb1StMul(MCInst &MI, unsigned Opcode, uint32_t insn,
    unsigned short NumOps, unsigned &NumOpsAdded, BO B) {
  return DisassembleThumb1LdStMul(false, MI, Opcode, insn, NumOps, NumOpsAdded,
                                  B);
}

// A8.6.16 B Encoding T1
// cond = Inst{11-8} & imm8 = Inst{7-0}
// imm32 = SignExtend(imm8:'0', 32)
//
// tBcc: offset Pred-Imm Pred-CCR
// tSVC: imm8 Pred-Imm Pred-CCR
// tTRAP: 0 operand (early return)
static bool DisassembleThumb1CondBr(MCInst &MI, unsigned Opcode, uint32_t insn,
    unsigned short NumOps, unsigned &NumOpsAdded, BO) {

  if (Opcode == ARM::tTRAP)
    return true;

  const TargetOperandInfo *OpInfo = ARMInsts[Opcode].OpInfo;
  if (!OpInfo) return false;

  assert(NumOps == 3 && OpInfo[0].RegClass < 0 &&
         OpInfo[1].isPredicate() && OpInfo[2].RegClass == ARM::CCRRegClassID
         && "Exactly 3 operands expected");

  unsigned Imm8 = getT1Imm8(insn);
  MI.addOperand(MCOperand::CreateImm(
                  Opcode == ARM::tBcc ? SignExtend32<9>(Imm8 << 1) + 4
                                      : (int)Imm8));

  // Predicate operands by ARMBasicMCBuilder::TryPredicateAndSBitModifier().
  // But note that for tBcc, if cond = '1110' then UNDEFINED.
  if (Opcode == ARM::tBcc && slice(insn, 11, 8) == 14) {
    DEBUG(errs() << "if cond = '1110' then UNDEFINED\n");
    return false;
  }
  NumOpsAdded = 1;

  return true;
}

// A8.6.16 B Encoding T2
// imm11 = Inst{10-0}
// imm32 = SignExtend(imm11:'0', 32)
//
// tB: offset
static bool DisassembleThumb1Br(MCInst &MI, unsigned Opcode, uint32_t insn,
    unsigned short NumOps, unsigned &NumOpsAdded, BO) {

  const TargetOperandInfo *OpInfo = ARMInsts[Opcode].OpInfo;
  if (!OpInfo) return false;

  assert(NumOps == 1 && OpInfo[0].RegClass < 0 && "1 imm operand expected");

  unsigned Imm11 = getT1Imm11(insn);

  MI.addOperand(MCOperand::CreateImm(SignExtend32<12>(Imm11 << 1)));

  NumOpsAdded = 1;

  return true;

}

// See A6.2 16-bit Thumb instruction encoding for instruction classes
// corresponding to op.
//
// Table A6-1 16-bit Thumb instruction encoding (abridged)
// op    Instruction or instruction class
// ------  --------------------------------------------------------------------
// 00xxxx  Shift (immediate), add, subtract, move, and compare on page A6-7
// 010000  Data-processing on page A6-8
// 010001  Special data instructions and branch and exchange on page A6-9
// 01001x  Load from Literal Pool, see LDR (literal) on page A8-122
// 0101xx  Load/store single data item on page A6-10
// 011xxx
// 100xxx
// 10100x  Generate PC-relative address, see ADR on page A8-32
// 10101x  Generate SP-relative address, see ADD (SP plus immediate) on
//         page A8-28
// 1011xx  Miscellaneous 16-bit instructions on page A6-11
// 11000x  Store multiple registers, see STM / STMIA / STMEA on page A8-374
// 11001x  Load multiple registers, see LDM / LDMIA / LDMFD on page A8-110 a
// 1101xx  Conditional branch, and Supervisor Call on page A6-13
// 11100x  Unconditional Branch, see B on page A8-44
//
static bool DisassembleThumb1(uint16_t op, MCInst &MI, unsigned Opcode,
    uint32_t insn, unsigned short NumOps, unsigned &NumOpsAdded, BO B) {

  unsigned op1 = slice(op, 5, 4);
  unsigned op2 = slice(op, 3, 2);
  unsigned op3 = slice(op, 1, 0);
  unsigned opA = slice(op, 5, 2);
  switch (op1) {
  case 0:
    // A6.2.1 Shift (immediate), add, subtract, move, and compare
    return DisassembleThumb1General(MI, Opcode, insn, NumOps, NumOpsAdded, B);
  case 1:
    switch (op2) {
    case 0:
      switch (op3) {
      case 0:
        // A6.2.2 Data-processing
        return DisassembleThumb1DP(MI, Opcode, insn, NumOps, NumOpsAdded, B);
      case 1:
        // A6.2.3 Special data instructions and branch and exchange
        return DisassembleThumb1Special(MI, Opcode, insn, NumOps, NumOpsAdded,
                                        B);
      default:
        // A8.6.59 LDR (literal)
        return DisassembleThumb1LdPC(MI, Opcode, insn, NumOps, NumOpsAdded, B);
      }
      break;
    default:
      // A6.2.4 Load/store single data item
      return DisassembleThumb1LdSt(opA, MI, Opcode, insn, NumOps, NumOpsAdded,
                                   B);
      break;
    }
    break;
  case 2:
    switch (op2) {
    case 0:
      // A6.2.4 Load/store single data item
      return DisassembleThumb1LdSt(opA, MI, Opcode, insn, NumOps, NumOpsAdded,
                                   B);
    case 1:
      // A6.2.4 Load/store single data item
      return DisassembleThumb1LdStSP(MI, Opcode, insn, NumOps, NumOpsAdded, B);
    case 2:
      if (op3 <= 1) {
        // A8.6.10 ADR
        return DisassembleThumb1AddPCi(MI, Opcode, insn, NumOps, NumOpsAdded,
                                       B);
      } else {
        // A8.6.8 ADD (SP plus immediate)
        return DisassembleThumb1AddSPi(MI, Opcode, insn, NumOps, NumOpsAdded,
                                       B);
      }
    default:
      // A6.2.5 Miscellaneous 16-bit instructions
      return DisassembleThumb1Misc(MI, Opcode, insn, NumOps, NumOpsAdded, B);
    }
    break;
  case 3:
    switch (op2) {
    case 0:
      if (op3 <= 1) {
        // A8.6.189 STM / STMIA / STMEA
        return DisassembleThumb1StMul(MI, Opcode, insn, NumOps, NumOpsAdded, B);
      } else {
        // A8.6.53 LDM / LDMIA / LDMFD
        return DisassembleThumb1LdMul(MI, Opcode, insn, NumOps, NumOpsAdded, B);
      }
    case 1:
      // A6.2.6 Conditional branch, and Supervisor Call
      return DisassembleThumb1CondBr(MI, Opcode, insn, NumOps, NumOpsAdded, B);
    case 2:
      // Unconditional Branch, see B on page A8-44
      return DisassembleThumb1Br(MI, Opcode, insn, NumOps, NumOpsAdded, B);
    default:
      assert(0 && "Unreachable code");
      break;
    }
    break;
  default:
    assert(0 && "Unreachable code");
    break;
  }

  return false;
}

///////////////////////////////////////////////
//                                           //
// Thumb2 instruction disassembly functions. //
//                                           //
///////////////////////////////////////////////

///////////////////////////////////////////////////////////
//                                                       //
// Note: the register naming follows the ARM convention! //
//                                                       //
///////////////////////////////////////////////////////////

static inline bool Thumb2SRSOpcode(unsigned Opcode) {
  switch (Opcode) {
  default:
    return false;
  case ARM::t2SRSDBW: case ARM::t2SRSDB:
  case ARM::t2SRSIAW: case ARM::t2SRSIA:
    return true;
  }
}

static inline bool Thumb2RFEOpcode(unsigned Opcode) {
  switch (Opcode) {
  default:
    return false;
  case ARM::t2RFEDBW: case ARM::t2RFEDB:
  case ARM::t2RFEIAW: case ARM::t2RFEIA:
    return true;
  }
}

// t2SRS[IA|DB]W/t2SRS[IA|DB]: mode_imm = Inst{4-0}
static bool DisassembleThumb2SRS(MCInst &MI, unsigned Opcode, uint32_t insn,
    unsigned short NumOps, unsigned &NumOpsAdded) {
  MI.addOperand(MCOperand::CreateImm(slice(insn, 4, 0)));
  NumOpsAdded = 1;
  return true;
}

// t2RFE[IA|DB]W/t2RFE[IA|DB]: Rn
static bool DisassembleThumb2RFE(MCInst &MI, unsigned Opcode, uint32_t insn,
    unsigned short NumOps, unsigned &NumOpsAdded, BO B) {
  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                     decodeRn(insn))));
  NumOpsAdded = 1;
  return true;
}

static bool DisassembleThumb2LdStMul(MCInst &MI, unsigned Opcode, uint32_t insn,
    unsigned short NumOps, unsigned &NumOpsAdded, BO B) {

  if (Thumb2SRSOpcode(Opcode))
    return DisassembleThumb2SRS(MI, Opcode, insn, NumOps, NumOpsAdded);

  if (Thumb2RFEOpcode(Opcode))
    return DisassembleThumb2RFE(MI, Opcode, insn, NumOps, NumOpsAdded, B);

  assert((Opcode == ARM::t2LDMIA || Opcode == ARM::t2LDMIA_UPD ||
          Opcode == ARM::t2LDMDB || Opcode == ARM::t2LDMDB_UPD ||
          Opcode == ARM::t2STMIA || Opcode == ARM::t2STMIA_UPD ||
          Opcode == ARM::t2STMDB || Opcode == ARM::t2STMDB_UPD)
         && "Unexpected opcode");
  assert(NumOps >= 4 && "Thumb2 LdStMul expects NumOps >= 4");

  NumOpsAdded = 0;

  unsigned Base = getRegisterEnum(B, ARM::GPRRegClassID, decodeRn(insn));

  // Writeback to base.
  if (Opcode == ARM::t2LDMIA_UPD || Opcode == ARM::t2LDMDB_UPD ||
      Opcode == ARM::t2STMIA_UPD || Opcode == ARM::t2STMDB_UPD) {
    MI.addOperand(MCOperand::CreateReg(Base));
    ++NumOpsAdded;
  }

  MI.addOperand(MCOperand::CreateReg(Base));
  ++NumOpsAdded;

  // Handling the two predicate operands before the reglist.
  if (B->DoPredicateOperands(MI, Opcode, insn, NumOps)) {
    NumOpsAdded += 2;
  } else {
    DEBUG(errs() << "Expected predicate operands not found.\n");
    return false;
  }

  unsigned RegListBits = insn & ((1 << 16) - 1);

  // Fill the variadic part of reglist.
  for (unsigned i = 0; i < 16; ++i)
    if ((RegListBits >> i) & 1) {
      MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                         i)));
      ++NumOpsAdded;
    }

  return true;
}

// t2LDREX: Rd Rn
// t2LDREXD: Rd Rs Rn
// t2LDREXB, t2LDREXH: Rd Rn
// t2STREX: Rs Rd Rn
// t2STREXD: Rm Rd Rs Rn
// t2STREXB, t2STREXH: Rm Rd Rn
static bool DisassembleThumb2LdStEx(MCInst &MI, unsigned Opcode, uint32_t insn,
    unsigned short NumOps, unsigned &NumOpsAdded, BO B) {

  const TargetOperandInfo *OpInfo = ARMInsts[Opcode].OpInfo;
  if (!OpInfo) return false;

  unsigned &OpIdx = NumOpsAdded;

  OpIdx = 0;

  assert(NumOps >= 2
         && OpInfo[0].RegClass > 0
         && OpInfo[1].RegClass > 0
         && "Expect >=2 operands and first two as reg operands");

  bool isStore = (ARM::t2STREX <= Opcode && Opcode <= ARM::t2STREXH);
  bool isSW = (Opcode == ARM::t2LDREX || Opcode == ARM::t2STREX);
  bool isDW = (Opcode == ARM::t2LDREXD || Opcode == ARM::t2STREXD);

  // Add the destination operand for store.
  if (isStore) {
    MI.addOperand(MCOperand::CreateReg(
                    getRegisterEnum(B, OpInfo[OpIdx].RegClass,
                                    isSW ? decodeRs(insn) : decodeRm(insn))));
    ++OpIdx;
  }

  // Source operand for store and destination operand for load.
  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, OpInfo[OpIdx].RegClass,
                                                     decodeRd(insn))));
  ++OpIdx;

  // Thumb2 doubleword complication: with an extra source/destination operand.
  if (isDW) {
    MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B,OpInfo[OpIdx].RegClass,
                                                       decodeRs(insn))));
    ++OpIdx;
  }

  // Finally add the pointer operand.
  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, OpInfo[OpIdx].RegClass,
                                                     decodeRn(insn))));
  ++OpIdx;

  return true;
}

// t2LDRDi8: Rd Rs Rn imm8s4 (offset mode)
// t2LDRDpci: Rd Rs imm8s4 (Not decoded, prefer the generic t2LDRDi8 version)
// t2STRDi8: Rd Rs Rn imm8s4 (offset mode)
//
// Ditto for t2LDRD_PRE, t2LDRD_POST, t2STRD_PRE, t2STRD_POST, which are for
// disassembly only and do not have a tied_to writeback base register operand.
static bool DisassembleThumb2LdStDual(MCInst &MI, unsigned Opcode,
    uint32_t insn, unsigned short NumOps, unsigned &NumOpsAdded, BO B) {

  const TargetOperandInfo *OpInfo = ARMInsts[Opcode].OpInfo;
  if (!OpInfo) return false;

  assert(NumOps >= 4
         && OpInfo[0].RegClass > 0
         && OpInfo[0].RegClass == OpInfo[1].RegClass
         && OpInfo[2].RegClass > 0
         && OpInfo[3].RegClass < 0
         && "Expect >= 4 operands and first 3 as reg operands");

  // Add the <Rt> <Rt2> operands.
  unsigned RegClassPair = OpInfo[0].RegClass;
  unsigned RegClassBase = OpInfo[2].RegClass;
  
  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, RegClassPair,
                                                     decodeRd(insn))));
  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, RegClassPair,
                                                     decodeRs(insn))));
  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, RegClassBase,
                                                     decodeRn(insn))));

  // Finally add (+/-)imm8*4, depending on the U bit.
  int Offset = getImm8(insn) * 4;
  if (getUBit(insn) == 0)
    Offset = -Offset;
  MI.addOperand(MCOperand::CreateImm(Offset));
  NumOpsAdded = 4;

  return true;
}

// t2TBB, t2TBH: Rn Rm Pred-Imm Pred-CCR
static bool DisassembleThumb2TB(MCInst &MI, unsigned Opcode,
    uint32_t insn, unsigned short NumOps, unsigned &NumOpsAdded, BO B) {

  assert(NumOps >= 2 && "Expect >= 2 operands");

  // The generic version of TBB/TBH needs a base register.
  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                     decodeRn(insn))));
  // Add the index register.
  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                     decodeRm(insn))));
  NumOpsAdded = 2;

  return true;
}

static inline bool Thumb2ShiftOpcode(unsigned Opcode) {
  switch (Opcode) {
  default:
    return false;
  case ARM::t2MOVCClsl: case ARM::t2MOVCClsr:
  case ARM::t2MOVCCasr: case ARM::t2MOVCCror:
  case ARM::t2LSLri:    case ARM::t2LSRri:
  case ARM::t2ASRri:    case ARM::t2RORri:
    return true;
  }
}

// A6.3.11 Data-processing (shifted register)
//
// Two register operands (Rn=0b1111 no 1st operand reg): Rs Rm
// Two register operands (Rs=0b1111 no dst operand reg): Rn Rm
// Three register operands: Rs Rn Rm
// Three register operands: (Rn=0b1111 Conditional Move) Rs Ro(TIED_TO) Rm
//
// Constant shifts t2_so_reg is a 2-operand unit corresponding to the Thumb2
// register with shift forms: (Rm, ConstantShiftSpecifier).
// Constant shift specifier: Imm = (ShOp | ShAmt<<3).
//
// There are special instructions, like t2MOVsra_flag and t2MOVsrl_flag, which
// only require two register operands: Rd, Rm in ARM Reference Manual terms, and
// nothing else, because the shift amount is already specified.
// Similar case holds for t2MOVrx, t2ADDrr, ..., etc.
static bool DisassembleThumb2DPSoReg(MCInst &MI, unsigned Opcode, uint32_t insn,
    unsigned short NumOps, unsigned &NumOpsAdded, BO B) {

  const TargetInstrDesc &TID = ARMInsts[Opcode];
  const TargetOperandInfo *OpInfo = TID.OpInfo;
  unsigned &OpIdx = NumOpsAdded;

  // Special case handling.
  if (Opcode == ARM::t2BR_JT) {
    assert(NumOps == 4
           && OpInfo[0].RegClass == ARM::GPRRegClassID
           && OpInfo[1].RegClass == ARM::GPRRegClassID
           && OpInfo[2].RegClass < 0
           && OpInfo[3].RegClass < 0
           && "Exactly 4 operands expect and first two as reg operands");
    // Only need to populate the src reg operand.
    MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                       decodeRm(insn))));
    MI.addOperand(MCOperand::CreateReg(0));
    MI.addOperand(MCOperand::CreateImm(0));
    MI.addOperand(MCOperand::CreateImm(0));
    NumOpsAdded = 4;
    return true;
  }

  OpIdx = 0;

  assert(NumOps >= 2
         && (OpInfo[0].RegClass == ARM::GPRRegClassID ||
             OpInfo[0].RegClass == ARM::rGPRRegClassID)
         && (OpInfo[1].RegClass == ARM::GPRRegClassID ||
             OpInfo[1].RegClass == ARM::rGPRRegClassID)
         && "Expect >= 2 operands and first two as reg operands");

  bool ThreeReg = (NumOps > 2 && (OpInfo[2].RegClass == ARM::GPRRegClassID ||
                                  OpInfo[2].RegClass == ARM::rGPRRegClassID));
  bool NoDstReg = (decodeRs(insn) == 0xF);

  // Build the register operands, followed by the constant shift specifier.

  MI.addOperand(MCOperand::CreateReg(
                  getRegisterEnum(B, OpInfo[0].RegClass,
                                  NoDstReg ? decodeRn(insn) : decodeRs(insn))));
  ++OpIdx;

  if (ThreeReg) {
    int Idx;
    if ((Idx = TID.getOperandConstraint(OpIdx, TOI::TIED_TO)) != -1) {
      // Process tied_to operand constraint.
      MI.addOperand(MI.getOperand(Idx));
      ++OpIdx;
    } else if (!NoDstReg) {
      MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, OpInfo[1].RegClass,
                                                         decodeRn(insn))));
      ++OpIdx;
    } else {
      DEBUG(errs() << "Thumb2 encoding error: d==15 for three-reg operands.\n");
      return false;
    }
  }

  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, OpInfo[OpIdx].RegClass,
                                                     decodeRm(insn))));
  ++OpIdx;

  if (NumOps == OpIdx)
    return true;

  if (OpInfo[OpIdx].RegClass < 0 && !OpInfo[OpIdx].isPredicate()
      && !OpInfo[OpIdx].isOptionalDef()) {

    if (Thumb2ShiftOpcode(Opcode)) {
      unsigned Imm = getShiftAmtBits(insn);
      ARM_AM::ShiftOpc ShOp = getShiftOpcForBits(slice(insn, 5, 4));
      getImmShiftSE(ShOp, Imm);
      MI.addOperand(MCOperand::CreateImm(Imm));
    } else {
      // Build the constant shift specifier operand.
      unsigned bits2 = getShiftTypeBits(insn);
      unsigned imm5 = getShiftAmtBits(insn);
      ARM_AM::ShiftOpc ShOp = ARM_AM::no_shift;
      unsigned ShAmt = decodeImmShift(bits2, imm5, ShOp);
      MI.addOperand(MCOperand::CreateImm(ARM_AM::getSORegOpc(ShOp, ShAmt)));
    }
    ++OpIdx;
  }

  return true;
}

// A6.3.1 Data-processing (modified immediate)
//
// Two register operands: Rs Rn ModImm
// One register operands (Rs=0b1111 no explicit dest reg): Rn ModImm
// One register operands (Rn=0b1111 no explicit src reg): Rs ModImm -
// {t2MOVi, t2MVNi}
//
// ModImm = ThumbExpandImm(i:imm3:imm8)
static bool DisassembleThumb2DPModImm(MCInst &MI, unsigned Opcode,
    uint32_t insn, unsigned short NumOps, unsigned &NumOpsAdded, BO B) {

  const TargetOperandInfo *OpInfo = ARMInsts[Opcode].OpInfo;
  unsigned &OpIdx = NumOpsAdded;

  OpIdx = 0;

  unsigned RdRegClassID = OpInfo[0].RegClass;
  assert(NumOps >= 2 && (RdRegClassID == ARM::GPRRegClassID ||
                         RdRegClassID == ARM::rGPRRegClassID)
         && "Expect >= 2 operands and first one as reg operand");

  unsigned RnRegClassID = OpInfo[1].RegClass;
  bool TwoReg = (RnRegClassID == ARM::GPRRegClassID
                 || RnRegClassID == ARM::rGPRRegClassID);
  bool NoDstReg = (decodeRs(insn) == 0xF);

  // Build the register operands, followed by the modified immediate.

  MI.addOperand(MCOperand::CreateReg(
                  getRegisterEnum(B, RdRegClassID,
                                  NoDstReg ? decodeRn(insn) : decodeRs(insn))));
  ++OpIdx;

  if (TwoReg) {
    if (NoDstReg) {
      DEBUG(errs()<<"Thumb2 encoding error: d==15 for DPModImm 2-reg instr.\n");
      return false;
    }
    MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, RnRegClassID,
                                                       decodeRn(insn))));
    ++OpIdx;
  }

  // The modified immediate operand should come next.
  assert(OpIdx < NumOps && OpInfo[OpIdx].RegClass < 0 &&
         !OpInfo[OpIdx].isPredicate() && !OpInfo[OpIdx].isOptionalDef()
         && "Pure imm operand expected");

  // i:imm3:imm8
  // A6.3.2 Modified immediate constants in Thumb instructions
  unsigned imm12 = getIImm3Imm8(insn);
  MI.addOperand(MCOperand::CreateImm(ThumbExpandImm(imm12)));
  ++OpIdx;

  return true;
}

static inline bool Thumb2SaturateOpcode(unsigned Opcode) {
  switch (Opcode) {
  case ARM::t2SSAT: case ARM::t2SSAT16:
  case ARM::t2USAT: case ARM::t2USAT16:
    return true;
  default:
    return false;
  }
}

/// DisassembleThumb2Sat - Disassemble Thumb2 saturate instructions:
/// o t2SSAT, t2USAT: Rs sat_pos Rn shamt
/// o t2SSAT16, t2USAT16: Rs sat_pos Rn
static bool DisassembleThumb2Sat(MCInst &MI, unsigned Opcode, uint32_t insn,
                                 unsigned &NumOpsAdded, BO B) {
  const TargetInstrDesc &TID = ARMInsts[Opcode];
  NumOpsAdded = TID.getNumOperands() - 2; // ignore predicate operands

  // Disassemble the register def.
  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::rGPRRegClassID,
                                                     decodeRs(insn))));

  unsigned Pos = slice(insn, 4, 0);
  if (Opcode == ARM::t2SSAT || Opcode == ARM::t2SSAT16)
    Pos += 1;
  MI.addOperand(MCOperand::CreateImm(Pos));

  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::rGPRRegClassID,
                                                     decodeRn(insn))));

  if (NumOpsAdded == 4) {
    ARM_AM::ShiftOpc Opc = (slice(insn, 21, 21) != 0 ?
                            ARM_AM::asr : ARM_AM::lsl);
    // Inst{14-12:7-6} encodes the imm5 shift amount.
    unsigned ShAmt = slice(insn, 14, 12) << 2 | slice(insn, 7, 6);
    if (ShAmt == 0) {
      if (Opc == ARM_AM::asr)
        ShAmt = 32;
      else
        Opc = ARM_AM::no_shift;
    }
    MI.addOperand(MCOperand::CreateImm(ARM_AM::getSORegOpc(Opc, ShAmt)));
  }
  return true;
}

// A6.3.3 Data-processing (plain binary immediate)
//
// o t2ADDri12, t2SUBri12: Rs Rn imm12
// o t2LEApcrel (ADR): Rs imm12
// o t2BFC (BFC): Rs Ro(TIED_TO) bf_inv_mask_imm
// o t2BFI (BFI) (Currently not defined in LLVM as of Jan-07-2010)
// o t2MOVi16: Rs imm16
// o t2MOVTi16: Rs imm16
// o t2SBFX (SBFX): Rs Rn lsb width
// o t2UBFX (UBFX): Rs Rn lsb width
// o t2BFI (BFI): Rs Rn lsb width
static bool DisassembleThumb2DPBinImm(MCInst &MI, unsigned Opcode,
    uint32_t insn, unsigned short NumOps, unsigned &NumOpsAdded, BO B) {

  const TargetInstrDesc &TID = ARMInsts[Opcode];
  const TargetOperandInfo *OpInfo = TID.OpInfo;
  unsigned &OpIdx = NumOpsAdded;

  OpIdx = 0;

  unsigned RdRegClassID = OpInfo[0].RegClass;
  assert(NumOps >= 2 && (RdRegClassID == ARM::GPRRegClassID ||
                         RdRegClassID == ARM::rGPRRegClassID)
         && "Expect >= 2 operands and first one as reg operand");

  unsigned RnRegClassID = OpInfo[1].RegClass;
  bool TwoReg = (RnRegClassID == ARM::GPRRegClassID
                 || RnRegClassID == ARM::rGPRRegClassID);

  // Build the register operand(s), followed by the immediate(s).

  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, RdRegClassID,
                                                     decodeRs(insn))));
  ++OpIdx;

  if (TwoReg) {
    assert(NumOps >= 3 && "Expect >= 3 operands");
    int Idx;
    if ((Idx = TID.getOperandConstraint(OpIdx, TOI::TIED_TO)) != -1) {
      // Process tied_to operand constraint.
      MI.addOperand(MI.getOperand(Idx));
    } else {
      // Add src reg operand.
      MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, RnRegClassID,
                                                         decodeRn(insn))));
    }
    ++OpIdx;
  }

  if (Opcode == ARM::t2BFI) {
    // Add val reg operand.
    MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, RnRegClassID,
                                                       decodeRn(insn))));
    ++OpIdx;
  }

  assert(OpInfo[OpIdx].RegClass < 0 && !OpInfo[OpIdx].isPredicate()
         && !OpInfo[OpIdx].isOptionalDef()
         && "Pure imm operand expected");

  // Pre-increment OpIdx.
  ++OpIdx;

  if (Opcode == ARM::t2ADDri12 || Opcode == ARM::t2SUBri12
      || Opcode == ARM::t2LEApcrel)
    MI.addOperand(MCOperand::CreateImm(getIImm3Imm8(insn)));
  else if (Opcode == ARM::t2MOVi16 || Opcode == ARM::t2MOVTi16) {
    if (!B->tryAddingSymbolicOperand(getImm16(insn), 4, MI))
      MI.addOperand(MCOperand::CreateImm(getImm16(insn)));
  } else if (Opcode == ARM::t2BFC || Opcode == ARM::t2BFI) {
    uint32_t mask = 0;
    if (getBitfieldInvMask(insn, mask))
      MI.addOperand(MCOperand::CreateImm(mask));
    else
      return false;
  } else {
    // Handle the case of: lsb width
    assert((Opcode == ARM::t2SBFX || Opcode == ARM::t2UBFX)
            && "Unexpected opcode");
    MI.addOperand(MCOperand::CreateImm(getLsb(insn)));
    MI.addOperand(MCOperand::CreateImm(getWidthMinus1(insn) + 1));

    ++OpIdx;
  }

  return true;
}

// A6.3.4 Table A6-15 Miscellaneous control instructions
// A8.6.41 DMB
// A8.6.42 DSB
// A8.6.49 ISB
static inline bool t2MiscCtrlInstr(uint32_t insn) {
  if (slice(insn, 31, 20) == 0xf3b && slice(insn, 15, 14) == 2 &&
      slice(insn, 12, 12) == 0)
    return true;

  return false;
}

// A6.3.4 Branches and miscellaneous control
//
// A8.6.16 B
// Branches: t2B, t2Bcc -> imm operand
//
// Branches: t2TPsoft -> no operand
//
// A8.6.23 BL, BLX (immediate)
// Branches (defined in ARMInstrThumb.td): tBLr9, tBLXi_r9 -> imm operand
//
// A8.6.26
// t2BXJ -> Rn
//
// Miscellaneous control:
//   -> no operand (except pred-imm pred-ccr for CLREX, memory barrier variants)
//
// Hint: t2NOP, t2YIELD, t2WFE, t2WFI, t2SEV
//   -> no operand (except pred-imm pred-ccr)
//
// t2DBG -> imm4 = Inst{3-0}
//
// t2MRS/t2MRSsys -> Rs
// t2MSR/t2MSRsys -> Rn mask=Inst{11-8}
// t2SMC -> imm4 = Inst{19-16}
static bool DisassembleThumb2BrMiscCtrl(MCInst &MI, unsigned Opcode,
    uint32_t insn, unsigned short NumOps, unsigned &NumOpsAdded, BO B) {

  if (NumOps == 0)
    return true;

  if (Opcode == ARM::t2DMB || Opcode == ARM::t2DSB) {
    // Inst{3-0} encodes the memory barrier option for the variants.
    unsigned opt = slice(insn, 3, 0);
    switch (opt) {
    case ARM_MB::SY:  case ARM_MB::ST:
    case ARM_MB::ISH: case ARM_MB::ISHST:
    case ARM_MB::NSH: case ARM_MB::NSHST:
    case ARM_MB::OSH: case ARM_MB::OSHST:
      MI.addOperand(MCOperand::CreateImm(opt));
      NumOpsAdded = 1;
      return true;
    default:
      return false;
    }
  }

  if (t2MiscCtrlInstr(insn))
    return true;

  switch (Opcode) {
  case ARM::t2CLREX:
  case ARM::t2NOP:
  case ARM::t2YIELD:
  case ARM::t2WFE:
  case ARM::t2WFI:
  case ARM::t2SEV:
    return true;
  default:
    break;
  }

  // FIXME: To enable correct asm parsing and disasm of CPS we need 3 different
  // opcodes which match the same real instruction. This is needed since there's
  // no current handling of optional arguments. Fix here when a better handling
  // of optional arguments is implemented.
  if (Opcode == ARM::t2CPS3p) {
    MI.addOperand(MCOperand::CreateImm(slice(insn, 10, 9))); // imod
    MI.addOperand(MCOperand::CreateImm(slice(insn, 7, 5)));  // iflags
    MI.addOperand(MCOperand::CreateImm(slice(insn, 4, 0)));  // mode
    NumOpsAdded = 3;
    return true;
  }
  if (Opcode == ARM::t2CPS2p) {
    MI.addOperand(MCOperand::CreateImm(slice(insn, 10, 9))); // imod
    MI.addOperand(MCOperand::CreateImm(slice(insn, 7, 5)));  // iflags
    NumOpsAdded = 2;
    return true;
  }
  if (Opcode == ARM::t2CPS1p) {
    MI.addOperand(MCOperand::CreateImm(slice(insn, 4, 0))); // mode
    NumOpsAdded = 1;
    return true;
  }

  // DBG has its option specified in Inst{3-0}.
  if (Opcode == ARM::t2DBG) {
    MI.addOperand(MCOperand::CreateImm(slice(insn, 3, 0)));
    NumOpsAdded = 1;
    return true;
  }

  // MRS and MRSsys take one GPR reg Rs.
  if (Opcode == ARM::t2MRS || Opcode == ARM::t2MRSsys) {
    MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                       decodeRs(insn))));
    NumOpsAdded = 1;
    return true;
  }
  // BXJ takes one GPR reg Rn.
  if (Opcode == ARM::t2BXJ) {
    MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                       decodeRn(insn))));
    NumOpsAdded = 1;
    return true;
  }
  // MSR take a mask, followed by one GPR reg Rn. The mask contains the R Bit in
  // bit 4, and the special register fields in bits 3-0.
  if (Opcode == ARM::t2MSR) {
    MI.addOperand(MCOperand::CreateImm(slice(insn, 20, 20) << 4 /* R Bit */ |
                                       slice(insn, 11, 8) /* Special Reg */));
    MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                       decodeRn(insn))));
    NumOpsAdded = 2;
    return true;
  }
  // SMC take imm4.
  if (Opcode == ARM::t2SMC) {
    MI.addOperand(MCOperand::CreateImm(slice(insn, 19, 16)));
    NumOpsAdded = 1;
    return true;
  }

  // Some instructions have predicate operands first before the immediate.
  if (Opcode == ARM::tBLXi_r9 || Opcode == ARM::tBLr9) {
    // Handling the two predicate operands before the imm operand.
    if (B->DoPredicateOperands(MI, Opcode, insn, NumOps))
      NumOpsAdded += 2;
    else {
      DEBUG(errs() << "Expected predicate operands not found.\n");
      return false;
    }
  }

  // Add the imm operand.
  int Offset = 0;

  switch (Opcode) {
  default:
    assert(0 && "Unexpected opcode");
    return false;
  case ARM::t2B:
    Offset = decodeImm32_B_EncodingT4(insn);
    break;
  case ARM::t2Bcc:
    Offset = decodeImm32_B_EncodingT3(insn);
    break;
  case ARM::tBLr9:
    Offset = decodeImm32_BL(insn);
    break;
  case ARM::tBLXi_r9:
    Offset = decodeImm32_BLX(insn);
    break;
  }

  if (!B->tryAddingSymbolicOperand(Offset + B->getBuilderAddress() + 4, 4, MI))
    MI.addOperand(MCOperand::CreateImm(Offset));

  // This is an increment as some predicate operands may have been added first.
  NumOpsAdded += 1;

  return true;
}

static inline bool Thumb2PreloadOpcode(unsigned Opcode) {
  switch (Opcode) {
  default:
    return false;
  case ARM::t2PLDi12:   case ARM::t2PLDi8:
  case ARM::t2PLDs:
  case ARM::t2PLDWi12:  case ARM::t2PLDWi8:
  case ARM::t2PLDWs:
  case ARM::t2PLIi12:   case ARM::t2PLIi8:
  case ARM::t2PLIs:
    return true;
  }
}

static bool DisassembleThumb2PreLoad(MCInst &MI, unsigned Opcode, uint32_t insn,
    unsigned short NumOps, unsigned &NumOpsAdded, BO B) {

  // Preload Data/Instruction requires either 2 or 3 operands.
  // t2PLDi12, t2PLDi8, t2PLDpci: Rn [+/-]imm12/imm8
  // t2PLDr:                      Rn Rm
  // t2PLDs:                      Rn Rm imm2=Inst{5-4}
  // Same pattern applies for t2PLDW* and t2PLI*.

  const TargetInstrDesc &TID = ARMInsts[Opcode];
  const TargetOperandInfo *OpInfo = TID.OpInfo;
  unsigned &OpIdx = NumOpsAdded;

  OpIdx = 0;

  assert(NumOps >= 2 &&
         OpInfo[0].RegClass == ARM::GPRRegClassID &&
         "Expect >= 2 operands and first one as reg operand");

  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                     decodeRn(insn))));
  ++OpIdx;

  if (OpInfo[OpIdx].RegClass == ARM::rGPRRegClassID) {
    MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                       decodeRm(insn))));
  } else {
    assert(OpInfo[OpIdx].RegClass < 0 && !OpInfo[OpIdx].isPredicate()
           && !OpInfo[OpIdx].isOptionalDef()
           && "Pure imm operand expected");
    int Offset = 0;
    if (Opcode == ARM::t2PLDi8 || Opcode == ARM::t2PLDWi8 ||
        Opcode == ARM::t2PLIi8) {
      // A8.6.117 Encoding T2: add = FALSE
      unsigned Imm8 = getImm8(insn);
      Offset = -1 * Imm8;
    } else {
      // The i12 forms.  See, for example, A8.6.117 Encoding T1.
      // Note that currently t2PLDi12 also handles the previously named t2PLDpci
      // opcode, that's why we use decodeImm12(insn) which returns +/- imm12.
      Offset = decodeImm12(insn);
    }
    MI.addOperand(MCOperand::CreateImm(Offset));
  }
  ++OpIdx;

  if (OpIdx < NumOps && OpInfo[OpIdx].RegClass < 0 &&
      !OpInfo[OpIdx].isPredicate() && !OpInfo[OpIdx].isOptionalDef()) {
    // Fills in the shift amount for t2PLDs, t2PLDWs, t2PLIs.
    MI.addOperand(MCOperand::CreateImm(slice(insn, 5, 4)));
    ++OpIdx;
  }

  return true;
}

// A6.3.10 Store single data item
// A6.3.9 Load byte, memory hints
// A6.3.8 Load halfword, memory hints
// A6.3.7 Load word
//
// For example,
//
// t2LDRi12:   Rd Rn (+)imm12
// t2LDRi8:    Rd Rn (+/-)imm8 (+ if Inst{9} == 0b1)
// t2LDRs:     Rd Rn Rm ConstantShiftSpecifier (see also
//             DisassembleThumb2DPSoReg)
// t2LDR_POST: Rd Rn Rn(TIED_TO) (+/-)imm8 (+ if Inst{9} == 0b1)
// t2LDR_PRE:  Rd Rn Rn(TIED_TO) (+/-)imm8 (+ if Inst{9} == 0b1)
//
// t2STRi12:   Rd Rn (+)imm12
// t2STRi8:    Rd Rn (+/-)imm8 (+ if Inst{9} == 0b1)
// t2STRs:     Rd Rn Rm ConstantShiftSpecifier (see also
//             DisassembleThumb2DPSoReg)
// t2STR_POST: Rn Rd Rn(TIED_TO) (+/-)imm8 (+ if Inst{9} == 0b1)
// t2STR_PRE:  Rn Rd Rn(TIED_TO) (+/-)imm8 (+ if Inst{9} == 0b1)
//
// Note that for indexed modes, the Rn(TIED_TO) operand needs to be populated
// correctly, as LLVM AsmPrinter depends on it.  For indexed stores, the first
// operand is Rn; for all the other instructions, Rd is the first operand.
//
// Delegates to DisassembleThumb2PreLoad() for preload data/instruction.
// Delegates to DisassembleThumb2Ldpci() for load * literal operations.
static bool DisassembleThumb2LdSt(bool Load, MCInst &MI, unsigned Opcode,
    uint32_t insn, unsigned short NumOps, unsigned &NumOpsAdded, BO B) {

  unsigned Rn = decodeRn(insn);

  if (Thumb2PreloadOpcode(Opcode))
    return DisassembleThumb2PreLoad(MI, Opcode, insn, NumOps, NumOpsAdded, B);

  // See, for example, A6.3.7 Load word: Table A6-18 Load word.
  if (Load && Rn == 15)
    return DisassembleThumb2Ldpci(MI, Opcode, insn, NumOps, NumOpsAdded, B);
  const TargetInstrDesc &TID = ARMInsts[Opcode];
  const TargetOperandInfo *OpInfo = TID.OpInfo;
  unsigned &OpIdx = NumOpsAdded;

  OpIdx = 0;

  assert(NumOps >= 3 &&
         OpInfo[0].RegClass == ARM::GPRRegClassID &&
         OpInfo[1].RegClass == ARM::GPRRegClassID &&
         "Expect >= 3 operands and first two as reg operands");

  bool ThreeReg = (OpInfo[2].RegClass > 0);
  bool TIED_TO = ThreeReg && TID.getOperandConstraint(2, TOI::TIED_TO) != -1;
  bool Imm12 = !ThreeReg && slice(insn, 23, 23) == 1; // ARMInstrThumb2.td

  // Build the register operands, followed by the immediate.
  unsigned R0, R1, R2 = 0;
  unsigned Rd = decodeRd(insn);
  int Imm = 0;

  if (!Load && TIED_TO) {
    R0 = Rn;
    R1 = Rd;
  } else {
    R0 = Rd;
    R1 = Rn;
  }
  if (ThreeReg) {
    if (TIED_TO) {
      R2 = Rn;
      Imm = decodeImm8(insn);
    } else {
      R2 = decodeRm(insn);
      // See, for example, A8.6.64 LDRB (register).
      // And ARMAsmPrinter::printT2AddrModeSoRegOperand().
      // LSL is the default shift opc, and LLVM does not expect it to be encoded
      // as part of the immediate operand.
      // Imm = ARM_AM::getSORegOpc(ARM_AM::lsl, slice(insn, 5, 4));
      Imm = slice(insn, 5, 4);
    }
  } else {
    if (Imm12)
      Imm = getImm12(insn);
    else
      Imm = decodeImm8(insn);
  }

  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                     R0)));
  ++OpIdx;
  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                     R1)));
  ++OpIdx;

  if (ThreeReg) {
    // This could be an offset register or a TIED_TO register.
    MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B,OpInfo[OpIdx].RegClass,
                                                       R2)));
    ++OpIdx;
  }

  assert(OpInfo[OpIdx].RegClass < 0 && !OpInfo[OpIdx].isPredicate()
         && !OpInfo[OpIdx].isOptionalDef()
         && "Pure imm operand expected");

  MI.addOperand(MCOperand::CreateImm(Imm));
  ++OpIdx;

  return true;
}

// A6.3.12 Data-processing (register)
//
// Two register operands [rotate]:   Rs Rm [rotation(= (rotate:'000'))]
// Three register operands only:     Rs Rn Rm
// Three register operands [rotate]: Rs Rn Rm [rotation(= (rotate:'000'))]
//
// Parallel addition and subtraction 32-bit Thumb instructions: Rs Rn Rm
//
// Miscellaneous operations: Rs [Rn] Rm
static bool DisassembleThumb2DPReg(MCInst &MI, unsigned Opcode, uint32_t insn,
    unsigned short NumOps, unsigned &NumOpsAdded, BO B) {

  const TargetInstrDesc &TID = ARMInsts[Opcode];
  const TargetOperandInfo *OpInfo = TID.OpInfo;
  unsigned &OpIdx = NumOpsAdded;

  OpIdx = 0;

  assert(NumOps >= 2 &&
         OpInfo[0].RegClass > 0 &&
         OpInfo[1].RegClass > 0 &&
         "Expect >= 2 operands and first two as reg operands");

  // Build the register operands, followed by the optional rotation amount.

  bool ThreeReg = NumOps > 2 && OpInfo[2].RegClass > 0;

  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, OpInfo[OpIdx].RegClass,
                                                     decodeRs(insn))));
  ++OpIdx;

  if (ThreeReg) {
    MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B,OpInfo[OpIdx].RegClass,
                                                       decodeRn(insn))));
    ++OpIdx;
  }

  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, OpInfo[OpIdx].RegClass,
                                                     decodeRm(insn))));
  ++OpIdx;

  if (OpIdx < NumOps && OpInfo[OpIdx].RegClass < 0
      && !OpInfo[OpIdx].isPredicate() && !OpInfo[OpIdx].isOptionalDef()) {
    // Add the rotation amount immediate.
    MI.addOperand(MCOperand::CreateImm(decodeRotate(insn)));
    ++OpIdx;
  }

  return true;
}

// A6.3.16 Multiply, multiply accumulate, and absolute difference
//
// t2MLA, t2MLS, t2SMMLA, t2SMMLS: Rs Rn Rm Ra=Inst{15-12}
// t2MUL, t2SMMUL:                 Rs Rn Rm
// t2SMLA[BB|BT|TB|TT|WB|WT]:      Rs Rn Rm Ra=Inst{15-12}
// t2SMUL[BB|BT|TB|TT|WB|WT]:      Rs Rn Rm
//
// Dual halfword multiply: t2SMUAD[X], t2SMUSD[X], t2SMLAD[X], t2SMLSD[X]:
//   Rs Rn Rm Ra=Inst{15-12}
//
// Unsigned Sum of Absolute Differences [and Accumulate]
//    Rs Rn Rm [Ra=Inst{15-12}]
static bool DisassembleThumb2Mul(MCInst &MI, unsigned Opcode, uint32_t insn,
    unsigned short NumOps, unsigned &NumOpsAdded, BO B) {

  const TargetOperandInfo *OpInfo = ARMInsts[Opcode].OpInfo;

  assert(NumOps >= 3 &&
         OpInfo[0].RegClass == ARM::rGPRRegClassID &&
         OpInfo[1].RegClass == ARM::rGPRRegClassID &&
         OpInfo[2].RegClass == ARM::rGPRRegClassID &&
         "Expect >= 3 operands and first three as reg operands");

  // Build the register operands.

  bool FourReg = NumOps > 3 && OpInfo[3].RegClass == ARM::rGPRRegClassID;

  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::rGPRRegClassID,
                                                     decodeRs(insn))));

  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::rGPRRegClassID,
                                                     decodeRn(insn))));

  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::rGPRRegClassID,
                                                     decodeRm(insn))));

  if (FourReg)
    MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::rGPRRegClassID,
                                                       decodeRd(insn))));

  NumOpsAdded = FourReg ? 4 : 3;

  return true;
}

// A6.3.17 Long multiply, long multiply accumulate, and divide
//
// t2SMULL, t2UMULL, t2SMLAL, t2UMLAL, t2UMAAL: RdLo RdHi Rn Rm
// where RdLo = Inst{15-12} and RdHi = Inst{11-8}
//
// Halfword multiple accumulate long: t2SMLAL<x><y>: RdLo RdHi Rn Rm
// where RdLo = Inst{15-12} and RdHi = Inst{11-8}
//
// Dual halfword multiple: t2SMLALD[X], t2SMLSLD[X]: RdLo RdHi Rn Rm
// where RdLo = Inst{15-12} and RdHi = Inst{11-8}
//
// Signed/Unsigned divide: t2SDIV, t2UDIV: Rs Rn Rm
static bool DisassembleThumb2LongMul(MCInst &MI, unsigned Opcode, uint32_t insn,
    unsigned short NumOps, unsigned &NumOpsAdded, BO B) {

  const TargetOperandInfo *OpInfo = ARMInsts[Opcode].OpInfo;

  assert(NumOps >= 3 &&
         OpInfo[0].RegClass == ARM::rGPRRegClassID &&
         OpInfo[1].RegClass == ARM::rGPRRegClassID &&
         OpInfo[2].RegClass == ARM::rGPRRegClassID &&
         "Expect >= 3 operands and first three as reg operands");

  bool FourReg = NumOps > 3 && OpInfo[3].RegClass == ARM::rGPRRegClassID;

  // Build the register operands.

  if (FourReg)
    MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::rGPRRegClassID,
                                                       decodeRd(insn))));

  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::rGPRRegClassID,
                                                     decodeRs(insn))));

  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::rGPRRegClassID,
                                                     decodeRn(insn))));

  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::rGPRRegClassID,
                                                     decodeRm(insn))));

  if (FourReg)
    NumOpsAdded = 4;
  else
    NumOpsAdded = 3;

  return true;
}

// See A6.3 32-bit Thumb instruction encoding for instruction classes
// corresponding to (op1, op2, op).
//
// Table A6-9 32-bit Thumb instruction encoding
// op1  op2    op  Instruction class, see
// ---  -------  --  -----------------------------------------------------------
// 01  00xx0xx  -  Load/store multiple on page A6-23
//     00xx1xx  -  Load/store dual, load/store exclusive, table branch on
//                 page A6-24
//     01xxxxx  -  Data-processing (shifted register) on page A6-31
//     1xxxxxx  -  Coprocessor instructions on page A6-40
// 10  x0xxxxx  0  Data-processing (modified immediate) on page A6-15
//     x1xxxxx  0  Data-processing (plain binary immediate) on page A6-19
//         -    1  Branches and miscellaneous control on page A6-20
// 11  000xxx0  -  Store single data item on page A6-30
//     001xxx0  -  Advanced SIMD element or structure load/store instructions
//                 on page A7-27
//     00xx001  - Load byte, memory hints on page A6-28
//     00xx011  -  Load halfword, memory hints on page A6-26
//     00xx101  -  Load word on page A6-25
//     00xx111  -  UNDEFINED
//     010xxxx  -  Data-processing (register) on page A6-33
//     0110xxx  -  Multiply, multiply accumulate, and absolute difference on
//                 page A6-38
//     0111xxx  -  Long multiply, long multiply accumulate, and divide on
//                 page A6-39
//     1xxxxxx  -  Coprocessor instructions on page A6-40
//
static bool DisassembleThumb2(uint16_t op1, uint16_t op2, uint16_t op,
    MCInst &MI, unsigned Opcode, uint32_t insn, unsigned short NumOps,
    unsigned &NumOpsAdded, BO B) {

  switch (op1) {
  case 1:
    if (slice(op2, 6, 5) == 0) {
      if (slice(op2, 2, 2) == 0) {
        // Load/store multiple.
        return DisassembleThumb2LdStMul(MI, Opcode, insn, NumOps, NumOpsAdded,
                                        B);
      }

      // Load/store dual, load/store exclusive, table branch, otherwise.
      assert(slice(op2, 2, 2) == 1 && "Thumb2 encoding error!");
      if ((ARM::t2LDREX <= Opcode && Opcode <= ARM::t2LDREXH) ||
          (ARM::t2STREX <= Opcode && Opcode <= ARM::t2STREXH)) {
        // Load/store exclusive.
        return DisassembleThumb2LdStEx(MI, Opcode, insn, NumOps, NumOpsAdded,
                                       B);
      }
      if (Opcode == ARM::t2LDRDi8 ||
          Opcode == ARM::t2LDRD_PRE || Opcode == ARM::t2LDRD_POST ||
          Opcode == ARM::t2STRDi8 ||
          Opcode == ARM::t2STRD_PRE || Opcode == ARM::t2STRD_POST) {
        // Load/store dual.
        return DisassembleThumb2LdStDual(MI, Opcode, insn, NumOps, NumOpsAdded,
                                         B);
      }
      if (Opcode == ARM::t2TBB || Opcode == ARM::t2TBH) {
        // Table branch.
        return DisassembleThumb2TB(MI, Opcode, insn, NumOps, NumOpsAdded, B);
      }
    } else if (slice(op2, 6, 5) == 1) {
      // Data-processing (shifted register).
      return DisassembleThumb2DPSoReg(MI, Opcode, insn, NumOps, NumOpsAdded, B);
    }

    // FIXME: A6.3.18 Coprocessor instructions
    // But see ThumbDisassembler::getInstruction().

    break;
  case 2:
    if (op == 0) {
      if (slice(op2, 5, 5) == 0)
        // Data-processing (modified immediate)
        return DisassembleThumb2DPModImm(MI, Opcode, insn, NumOps, NumOpsAdded,
                                         B);
      if (Thumb2SaturateOpcode(Opcode))
        return DisassembleThumb2Sat(MI, Opcode, insn, NumOpsAdded, B);

      // Data-processing (plain binary immediate)
      return DisassembleThumb2DPBinImm(MI, Opcode, insn, NumOps, NumOpsAdded,
                                       B);
    }
    // Branches and miscellaneous control on page A6-20.
    return DisassembleThumb2BrMiscCtrl(MI, Opcode, insn, NumOps, NumOpsAdded,
                                       B);
  case 3:
    switch (slice(op2, 6, 5)) {
    case 0:
      // Load/store instructions...
      if (slice(op2, 0, 0) == 0) {
        if (slice(op2, 4, 4) == 0) {
          // Store single data item on page A6-30
          return DisassembleThumb2LdSt(false, MI,Opcode,insn,NumOps,NumOpsAdded,
                                       B);
        } else {
          // FIXME: Advanced SIMD element or structure load/store instructions.
          // But see ThumbDisassembler::getInstruction().
          ;
        }
      } else {
        // Table A6-9 32-bit Thumb instruction encoding: Load byte|halfword|word
        return DisassembleThumb2LdSt(true, MI, Opcode, insn, NumOps,
                                     NumOpsAdded, B);
      }
      break;
    case 1:
      if (slice(op2, 4, 4) == 0) {
        // A6.3.12 Data-processing (register)
        return DisassembleThumb2DPReg(MI, Opcode, insn, NumOps, NumOpsAdded, B);
      } else if (slice(op2, 3, 3) == 0) {
        // A6.3.16 Multiply, multiply accumulate, and absolute difference
        return DisassembleThumb2Mul(MI, Opcode, insn, NumOps, NumOpsAdded, B);
      } else {
        // A6.3.17 Long multiply, long multiply accumulate, and divide
        return DisassembleThumb2LongMul(MI, Opcode, insn, NumOps, NumOpsAdded,
                                        B);
      }
      break;
    default:
      // FIXME: A6.3.18 Coprocessor instructions
      // But see ThumbDisassembler::getInstruction().
      ;
      break;
    }

    break;
  default:
    assert(0 && "Thumb2 encoding error!");
    break;
  }

  return false;
}

static bool DisassembleThumbFrm(MCInst &MI, unsigned Opcode, uint32_t insn,
    unsigned short NumOps, unsigned &NumOpsAdded, BO Builder) {

  uint16_t HalfWord = slice(insn, 31, 16);

  if (HalfWord == 0) {
    // A6.2 16-bit Thumb instruction encoding
    // op = bits[15:10]
    uint16_t op = slice(insn, 15, 10);
    return DisassembleThumb1(op, MI, Opcode, insn, NumOps, NumOpsAdded,
                             Builder);
  }

  unsigned bits15_11 = slice(HalfWord, 15, 11);

  // A6.1 Thumb instruction set encoding
  if (!(bits15_11 == 0x1D || bits15_11 == 0x1E || bits15_11 == 0x1F)) {
    assert("Bits[15:11] first halfword of Thumb2 instruction is out of range");
    return false;
  }

  // A6.3 32-bit Thumb instruction encoding

  uint16_t op1 = slice(HalfWord, 12, 11);
  uint16_t op2 = slice(HalfWord, 10, 4);
  uint16_t op = slice(insn, 15, 15);

  return DisassembleThumb2(op1, op2, op, MI, Opcode, insn, NumOps, NumOpsAdded,
                           Builder);
}
