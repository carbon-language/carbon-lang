//===- ARMDisassemblerCore.cpp - ARM disassembler helpers -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is part of the ARM Disassembler.
// It contains code to represent the core concepts of Builder and DisassembleFP
// to solve the problem of disassembling an ARM instr.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "arm-disassembler"

#include "ARMDisassemblerCore.h"
#include "MCTargetDesc/ARMAddressingModes.h"
#include "MCTargetDesc/ARMMCExpr.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

//#define DEBUG(X) do { X; } while (0)

/// ARMGenInstrInfo.inc - ARMGenInstrInfo.inc contains the static const
/// MCInstrDesc ARMInsts[] definition and the MCOperandInfo[]'s describing the
/// operand info for each ARMInsts[i].
///
/// Together with an instruction's encoding format, we can take advantage of the
/// NumOperands and the OpInfo fields of the target instruction description in
/// the quest to build out the MCOperand list for an MCInst.
///
/// The general guideline is that with a known format, the number of dst and src
/// operands are well-known.  The dst is built first, followed by the src
/// operand(s).  The operands not yet used at this point are for the Implicit
/// Uses and Defs by this instr.  For the Uses part, the pred:$p operand is
/// defined with two components:
///
/// def pred { // Operand PredicateOperand
///   ValueType Type = OtherVT;
///   string PrintMethod = "printPredicateOperand";
///   string AsmOperandLowerMethod = ?;
///   dag MIOperandInfo = (ops i32imm, CCR);
///   AsmOperandClass ParserMatchClass = ImmAsmOperand;
///   dag DefaultOps = (ops (i32 14), (i32 zero_reg));
/// }
///
/// which is manifested by the MCOperandInfo[] of:
///
/// { 0, 0|(1<<MCOI::Predicate), 0 },
/// { ARM::CCRRegClassID, 0|(1<<MCOI::Predicate), 0 }
///
/// So the first predicate MCOperand corresponds to the immediate part of the
/// ARM condition field (Inst{31-28}), and the second predicate MCOperand
/// corresponds to a register kind of ARM::CPSR.
///
/// For the Defs part, in the simple case of only cc_out:$s, we have:
///
/// def cc_out { // Operand OptionalDefOperand
///   ValueType Type = OtherVT;
///   string PrintMethod = "printSBitModifierOperand";
///   string AsmOperandLowerMethod = ?;
///   dag MIOperandInfo = (ops CCR);
///   AsmOperandClass ParserMatchClass = ImmAsmOperand;
///   dag DefaultOps = (ops (i32 zero_reg));
/// }
///
/// which is manifested by the one MCOperandInfo of:
///
/// { ARM::CCRRegClassID, 0|(1<<MCOI::OptionalDef), 0 }
///

namespace llvm {
extern MCInstrDesc ARMInsts[];
}

using namespace llvm;

const char *ARMUtils::OpcodeName(unsigned Opcode) {
  return ARMInsts[Opcode].Name;
}

// Return the register enum Based on RegClass and the raw register number.
// FIXME: Auto-gened?
static unsigned
getRegisterEnum(BO B, unsigned RegClassID, unsigned RawRegister) {
  if (RegClassID == ARM::rGPRRegClassID) {
    // Check for The register numbers 13 and 15 that are not permitted for many
    // Thumb register specifiers.
    if (RawRegister == 13 || RawRegister == 15) {
      B->SetErr(-1);
      return 0;
    }
    // For this purpose, we can treat rGPR as if it were GPR.
    RegClassID = ARM::GPRRegClassID;
  }

  // See also decodeNEONRd(), decodeNEONRn(), decodeNEONRm().
  // A7.3 register encoding
  //     Qd -> bit[12] == 0
  //     Qn -> bit[16] == 0
  //     Qm -> bit[0]  == 0
  //
  // If one of these bits is 1, the instruction is UNDEFINED.
  if (RegClassID == ARM::QPRRegClassID && slice(RawRegister, 0, 0) == 1) {
    B->SetErr(-1);
    return 0;
  }
  unsigned RegNum =
    RegClassID == ARM::QPRRegClassID ? RawRegister >> 1 : RawRegister;

  switch (RegNum) {
  default:
    break;
  case 0:
    switch (RegClassID) {
    case ARM::GPRRegClassID: case ARM::tGPRRegClassID: return ARM::R0;
    case ARM::DPRRegClassID: case ARM::DPR_8RegClassID:
    case ARM::DPR_VFP2RegClassID:
      return ARM::D0;
    case ARM::QPRRegClassID: case ARM::QPR_8RegClassID:
    case ARM::QPR_VFP2RegClassID:
      return ARM::Q0;
    case ARM::SPRRegClassID: case ARM::SPR_8RegClassID: return ARM::S0;
    }
    break;
  case 1:
    switch (RegClassID) {
    case ARM::GPRRegClassID: case ARM::tGPRRegClassID: return ARM::R1;
    case ARM::DPRRegClassID: case ARM::DPR_8RegClassID:
    case ARM::DPR_VFP2RegClassID:
      return ARM::D1;
    case ARM::QPRRegClassID: case ARM::QPR_8RegClassID:
    case ARM::QPR_VFP2RegClassID:
      return ARM::Q1;
    case ARM::SPRRegClassID: case ARM::SPR_8RegClassID: return ARM::S1;
    }
    break;
  case 2:
    switch (RegClassID) {
    case ARM::GPRRegClassID: case ARM::tGPRRegClassID: return ARM::R2;
    case ARM::DPRRegClassID: case ARM::DPR_8RegClassID:
    case ARM::DPR_VFP2RegClassID:
      return ARM::D2;
    case ARM::QPRRegClassID: case ARM::QPR_8RegClassID:
    case ARM::QPR_VFP2RegClassID:
      return ARM::Q2;
    case ARM::SPRRegClassID: case ARM::SPR_8RegClassID: return ARM::S2;
    }
    break;
  case 3:
    switch (RegClassID) {
    case ARM::GPRRegClassID: case ARM::tGPRRegClassID: return ARM::R3;
    case ARM::DPRRegClassID: case ARM::DPR_8RegClassID:
    case ARM::DPR_VFP2RegClassID:
      return ARM::D3;
    case ARM::QPRRegClassID: case ARM::QPR_8RegClassID:
    case ARM::QPR_VFP2RegClassID:
      return ARM::Q3;
    case ARM::SPRRegClassID: case ARM::SPR_8RegClassID: return ARM::S3;
    }
    break;
  case 4:
    switch (RegClassID) {
    case ARM::GPRRegClassID: case ARM::tGPRRegClassID: return ARM::R4;
    case ARM::DPRRegClassID: case ARM::DPR_8RegClassID:
    case ARM::DPR_VFP2RegClassID:
      return ARM::D4;
    case ARM::QPRRegClassID: case ARM::QPR_VFP2RegClassID: return ARM::Q4;
    case ARM::SPRRegClassID: case ARM::SPR_8RegClassID: return ARM::S4;
    }
    break;
  case 5:
    switch (RegClassID) {
    case ARM::GPRRegClassID: case ARM::tGPRRegClassID: return ARM::R5;
    case ARM::DPRRegClassID: case ARM::DPR_8RegClassID:
    case ARM::DPR_VFP2RegClassID:
      return ARM::D5;
    case ARM::QPRRegClassID: case ARM::QPR_VFP2RegClassID: return ARM::Q5;
    case ARM::SPRRegClassID: case ARM::SPR_8RegClassID: return ARM::S5;
    }
    break;
  case 6:
    switch (RegClassID) {
    case ARM::GPRRegClassID: case ARM::tGPRRegClassID: return ARM::R6;
    case ARM::DPRRegClassID: case ARM::DPR_8RegClassID:
    case ARM::DPR_VFP2RegClassID:
      return ARM::D6;
    case ARM::QPRRegClassID: case ARM::QPR_VFP2RegClassID: return ARM::Q6;
    case ARM::SPRRegClassID: case ARM::SPR_8RegClassID: return ARM::S6;
    }
    break;
  case 7:
    switch (RegClassID) {
    case ARM::GPRRegClassID: case ARM::tGPRRegClassID: return ARM::R7;
    case ARM::DPRRegClassID: case ARM::DPR_8RegClassID:
    case ARM::DPR_VFP2RegClassID:
      return ARM::D7;
    case ARM::QPRRegClassID: case ARM::QPR_VFP2RegClassID: return ARM::Q7;
    case ARM::SPRRegClassID: case ARM::SPR_8RegClassID: return ARM::S7;
    }
    break;
  case 8:
    switch (RegClassID) {
    case ARM::GPRRegClassID: return ARM::R8;
    case ARM::DPRRegClassID: case ARM::DPR_VFP2RegClassID: return ARM::D8;
    case ARM::QPRRegClassID: return ARM::Q8;
    case ARM::SPRRegClassID: case ARM::SPR_8RegClassID: return ARM::S8;
    }
    break;
  case 9:
    switch (RegClassID) {
    case ARM::GPRRegClassID: return ARM::R9;
    case ARM::DPRRegClassID: case ARM::DPR_VFP2RegClassID: return ARM::D9;
    case ARM::QPRRegClassID: return ARM::Q9;
    case ARM::SPRRegClassID: case ARM::SPR_8RegClassID: return ARM::S9;
    }
    break;
  case 10:
    switch (RegClassID) {
    case ARM::GPRRegClassID: return ARM::R10;
    case ARM::DPRRegClassID: case ARM::DPR_VFP2RegClassID: return ARM::D10;
    case ARM::QPRRegClassID: return ARM::Q10;
    case ARM::SPRRegClassID: case ARM::SPR_8RegClassID: return ARM::S10;
    }
    break;
  case 11:
    switch (RegClassID) {
    case ARM::GPRRegClassID: return ARM::R11;
    case ARM::DPRRegClassID: case ARM::DPR_VFP2RegClassID: return ARM::D11;
    case ARM::QPRRegClassID: return ARM::Q11;
    case ARM::SPRRegClassID: case ARM::SPR_8RegClassID: return ARM::S11;
    }
    break;
  case 12:
    switch (RegClassID) {
    case ARM::GPRRegClassID: return ARM::R12;
    case ARM::DPRRegClassID: case ARM::DPR_VFP2RegClassID: return ARM::D12;
    case ARM::QPRRegClassID: return ARM::Q12;
    case ARM::SPRRegClassID: case ARM::SPR_8RegClassID: return ARM::S12;
    }
    break;
  case 13:
    switch (RegClassID) {
    case ARM::GPRRegClassID: return ARM::SP;
    case ARM::DPRRegClassID: case ARM::DPR_VFP2RegClassID: return ARM::D13;
    case ARM::QPRRegClassID: return ARM::Q13;
    case ARM::SPRRegClassID: case ARM::SPR_8RegClassID: return ARM::S13;
    }
    break;
  case 14:
    switch (RegClassID) {
    case ARM::GPRRegClassID: return ARM::LR;
    case ARM::DPRRegClassID: case ARM::DPR_VFP2RegClassID: return ARM::D14;
    case ARM::QPRRegClassID: return ARM::Q14;
    case ARM::SPRRegClassID: case ARM::SPR_8RegClassID: return ARM::S14;
    }
    break;
  case 15:
    switch (RegClassID) {
    case ARM::GPRRegClassID: return ARM::PC;
    case ARM::DPRRegClassID: case ARM::DPR_VFP2RegClassID: return ARM::D15;
    case ARM::QPRRegClassID: return ARM::Q15;
    case ARM::SPRRegClassID: case ARM::SPR_8RegClassID: return ARM::S15;
    }
    break;
  case 16:
    switch (RegClassID) {
    case ARM::DPRRegClassID: return ARM::D16;
    case ARM::SPRRegClassID: return ARM::S16;
    }
    break;
  case 17:
    switch (RegClassID) {
    case ARM::DPRRegClassID: return ARM::D17;
    case ARM::SPRRegClassID: return ARM::S17;
    }
    break;
  case 18:
    switch (RegClassID) {
    case ARM::DPRRegClassID: return ARM::D18;
    case ARM::SPRRegClassID: return ARM::S18;
    }
    break;
  case 19:
    switch (RegClassID) {
    case ARM::DPRRegClassID: return ARM::D19;
    case ARM::SPRRegClassID: return ARM::S19;
    }
    break;
  case 20:
    switch (RegClassID) {
    case ARM::DPRRegClassID: return ARM::D20;
    case ARM::SPRRegClassID: return ARM::S20;
    }
    break;
  case 21:
    switch (RegClassID) {
    case ARM::DPRRegClassID: return ARM::D21;
    case ARM::SPRRegClassID: return ARM::S21;
    }
    break;
  case 22:
    switch (RegClassID) {
    case ARM::DPRRegClassID: return ARM::D22;
    case ARM::SPRRegClassID: return ARM::S22;
    }
    break;
  case 23:
    switch (RegClassID) {
    case ARM::DPRRegClassID: return ARM::D23;
    case ARM::SPRRegClassID: return ARM::S23;
    }
    break;
  case 24:
    switch (RegClassID) {
    case ARM::DPRRegClassID: return ARM::D24;
    case ARM::SPRRegClassID: return ARM::S24;
    }
    break;
  case 25:
    switch (RegClassID) {
    case ARM::DPRRegClassID: return ARM::D25;
    case ARM::SPRRegClassID: return ARM::S25;
    }
    break;
  case 26:
    switch (RegClassID) {
    case ARM::DPRRegClassID: return ARM::D26;
    case ARM::SPRRegClassID: return ARM::S26;
    }
    break;
  case 27:
    switch (RegClassID) {
    case ARM::DPRRegClassID: return ARM::D27;
    case ARM::SPRRegClassID: return ARM::S27;
    }
    break;
  case 28:
    switch (RegClassID) {
    case ARM::DPRRegClassID: return ARM::D28;
    case ARM::SPRRegClassID: return ARM::S28;
    }
    break;
  case 29:
    switch (RegClassID) {
    case ARM::DPRRegClassID: return ARM::D29;
    case ARM::SPRRegClassID: return ARM::S29;
    }
    break;
  case 30:
    switch (RegClassID) {
    case ARM::DPRRegClassID: return ARM::D30;
    case ARM::SPRRegClassID: return ARM::S30;
    }
    break;
  case 31:
    switch (RegClassID) {
    case ARM::DPRRegClassID: return ARM::D31;
    case ARM::SPRRegClassID: return ARM::S31;
    }
    break;
  }
  DEBUG(errs() << "Invalid (RegClassID, RawRegister) combination\n");
  // Encoding error.  Mark the builder with error code != 0.
  B->SetErr(-1);
  return 0;
}

///////////////////////////////
//                           //
//     Utility Functions     //
//                           //
///////////////////////////////

// Extract/Decode Rd: Inst{15-12}.
static inline unsigned decodeRd(uint32_t insn) {
  return (insn >> ARMII::RegRdShift) & ARMII::GPRRegMask;
}

// Extract/Decode Rn: Inst{19-16}.
static inline unsigned decodeRn(uint32_t insn) {
  return (insn >> ARMII::RegRnShift) & ARMII::GPRRegMask;
}

// Extract/Decode Rm: Inst{3-0}.
static inline unsigned decodeRm(uint32_t insn) {
  return (insn & ARMII::GPRRegMask);
}

// Extract/Decode Rs: Inst{11-8}.
static inline unsigned decodeRs(uint32_t insn) {
  return (insn >> ARMII::RegRsShift) & ARMII::GPRRegMask;
}

static inline unsigned getCondField(uint32_t insn) {
  return (insn >> ARMII::CondShift);
}

static inline unsigned getIBit(uint32_t insn) {
  return (insn >> ARMII::I_BitShift) & 1;
}

static inline unsigned getAM3IBit(uint32_t insn) {
  return (insn >> ARMII::AM3_I_BitShift) & 1;
}

static inline unsigned getPBit(uint32_t insn) {
  return (insn >> ARMII::P_BitShift) & 1;
}

static inline unsigned getUBit(uint32_t insn) {
  return (insn >> ARMII::U_BitShift) & 1;
}

static inline unsigned getPUBits(uint32_t insn) {
  return (insn >> ARMII::U_BitShift) & 3;
}

static inline unsigned getSBit(uint32_t insn) {
  return (insn >> ARMII::S_BitShift) & 1;
}

static inline unsigned getWBit(uint32_t insn) {
  return (insn >> ARMII::W_BitShift) & 1;
}

static inline unsigned getDBit(uint32_t insn) {
  return (insn >> ARMII::D_BitShift) & 1;
}

static inline unsigned getNBit(uint32_t insn) {
  return (insn >> ARMII::N_BitShift) & 1;
}

static inline unsigned getMBit(uint32_t insn) {
  return (insn >> ARMII::M_BitShift) & 1;
}

// See A8.4 Shifts applied to a register.
//     A8.4.2 Register controlled shifts.
//
// getShiftOpcForBits - getShiftOpcForBits translates from the ARM encoding bits
// into llvm enums for shift opcode.  The API clients should pass in the value
// encoded with two bits, so the assert stays to signal a wrong API usage.
//
// A8-12: DecodeRegShift()
static inline ARM_AM::ShiftOpc getShiftOpcForBits(unsigned bits) {
  switch (bits) {
  default: assert(0 && "No such value"); return ARM_AM::no_shift;
  case 0:  return ARM_AM::lsl;
  case 1:  return ARM_AM::lsr;
  case 2:  return ARM_AM::asr;
  case 3:  return ARM_AM::ror;
  }
}

// See A8.4 Shifts applied to a register.
//     A8.4.1 Constant shifts.
//
// getImmShiftSE - getImmShiftSE translates from the raw ShiftOpc and raw Imm5
// encodings into the intended ShiftOpc and shift amount.
//
// A8-11: DecodeImmShift()
static inline void getImmShiftSE(ARM_AM::ShiftOpc &ShOp, unsigned &ShImm) {
  if (ShImm != 0)
    return;
  switch (ShOp) {
  case ARM_AM::no_shift:
  case ARM_AM::rrx:
    break;
  case ARM_AM::lsl:
    ShOp = ARM_AM::no_shift;
    break;
  case ARM_AM::lsr:
  case ARM_AM::asr:
    ShImm = 32;
    break;
  case ARM_AM::ror:
    ShOp = ARM_AM::rrx;
    break;
  }
}

// getAMSubModeForBits - getAMSubModeForBits translates from the ARM encoding
// bits Inst{24-23} (P(24) and U(23)) into llvm enums for AMSubMode.  The API
// clients should pass in the value encoded with two bits, so the assert stays
// to signal a wrong API usage.
static inline ARM_AM::AMSubMode getAMSubModeForBits(unsigned bits) {
  switch (bits) {
  default: assert(0 && "No such value"); return ARM_AM::bad_am_submode;
  case 1:  return ARM_AM::ia;   // P=0 U=1
  case 3:  return ARM_AM::ib;   // P=1 U=1
  case 0:  return ARM_AM::da;   // P=0 U=0
  case 2:  return ARM_AM::db;   // P=1 U=0
  }
}

////////////////////////////////////////////
//                                        //
//    Disassemble function definitions    //
//                                        //
////////////////////////////////////////////

/// There is a separate Disassemble*Frm function entry for disassembly of an ARM
/// instr into a list of MCOperands in the appropriate order, with possible dst,
/// followed by possible src(s).
///
/// The processing of the predicate, and the 'S' modifier bit, if MI modifies
/// the CPSR, is factored into ARMBasicMCBuilder's method named
/// TryPredicateAndSBitModifier.

static bool DisassemblePseudo(MCInst &MI, unsigned Opcode, uint32_t insn,
    unsigned short NumOps, unsigned &NumOpsAdded, BO) {

  assert(0 && "Unexpected pseudo instruction!");
  return false;
}

// A8.6.94 MLA
// if d == 15 || n == 15 || m == 15 || a == 15 then UNPREDICTABLE;
//
// A8.6.105 MUL
// if d == 15 || n == 15 || m == 15 then UNPREDICTABLE;
//
// A8.6.246 UMULL
// if dLo == 15 || dHi == 15 || n == 15 || m == 15 then UNPREDICTABLE;
// if dHi == dLo then UNPREDICTABLE;
static bool BadRegsMulFrm(unsigned Opcode, uint32_t insn) {
  unsigned R19_16 = slice(insn, 19, 16);
  unsigned R15_12 = slice(insn, 15, 12);
  unsigned R11_8  = slice(insn, 11, 8);
  unsigned R3_0   = slice(insn, 3, 0);
  switch (Opcode) {
  default:
    // Did we miss an opcode?
    DEBUG(errs() << "BadRegsMulFrm: unexpected opcode!");
    return false;
  case ARM::MLA:     case ARM::MLS:     case ARM::SMLABB:  case ARM::SMLABT:
  case ARM::SMLATB:  case ARM::SMLATT:  case ARM::SMLAWB:  case ARM::SMLAWT:
  case ARM::SMMLA:   case ARM::SMMLAR:  case ARM::SMMLS:   case ARM::SMMLSR:
  case ARM::USADA8:
    if (R19_16 == 15 || R15_12 == 15 || R11_8 == 15 || R3_0 == 15)
      return true;
    return false;
  case ARM::MUL:     case ARM::SMMUL:   case ARM::SMMULR:
  case ARM::SMULBB:  case ARM::SMULBT:  case ARM::SMULTB:  case ARM::SMULTT:
  case ARM::SMULWB:  case ARM::SMULWT:  case ARM::SMUAD:   case ARM::SMUADX:
  // A8.6.167 SMLAD & A8.6.172 SMLSD
  case ARM::SMLAD:   case ARM::SMLADX:  case ARM::SMLSD:   case ARM::SMLSDX:
  case ARM::USAD8:
    if (R19_16 == 15 || R11_8 == 15 || R3_0 == 15)
      return true;
    return false;
  case ARM::SMLAL:   case ARM::SMULL:   case ARM::UMAAL:   case ARM::UMLAL:
  case ARM::UMULL:
  case ARM::SMLALBB: case ARM::SMLALBT: case ARM::SMLALTB: case ARM::SMLALTT:
  case ARM::SMLALD:  case ARM::SMLALDX: case ARM::SMLSLD:  case ARM::SMLSLDX:
    if (R19_16 == 15 || R15_12 == 15 || R11_8 == 15 || R3_0 == 15)
      return true;
    if (R19_16 == R15_12)
      return true;
    return false;;
  }
}

// Multiply Instructions.
// MLA, MLS, SMLABB, SMLABT, SMLATB, SMLATT, SMLAWB, SMLAWT, SMMLA, SMMLAR,
// SMMLS, SMMLAR, SMLAD, SMLADX, SMLSD, SMLSDX, and USADA8 (for convenience):
//     Rd{19-16} Rn{3-0} Rm{11-8} Ra{15-12}
// But note that register checking for {SMLAD, SMLADX, SMLSD, SMLSDX} is
// only for {d, n, m}.
//
// MUL, SMMUL, SMMULR, SMULBB, SMULBT, SMULTB, SMULTT, SMULWB, SMULWT, SMUAD,
// SMUADX, and USAD8 (for convenience):
//     Rd{19-16} Rn{3-0} Rm{11-8}
//
// SMLAL, SMULL, UMAAL, UMLAL, UMULL, SMLALBB, SMLALBT, SMLALTB, SMLALTT,
// SMLALD, SMLADLX, SMLSLD, SMLSLDX:
//     RdLo{15-12} RdHi{19-16} Rn{3-0} Rm{11-8}
//
// The mapping of the multiply registers to the "regular" ARM registers, where
// there are convenience decoder functions, is:
//
// Inst{15-12} => Rd
// Inst{19-16} => Rn
// Inst{3-0} => Rm
// Inst{11-8} => Rs
static bool DisassembleMulFrm(MCInst &MI, unsigned Opcode, uint32_t insn,
    unsigned short NumOps, unsigned &NumOpsAdded, BO B) {

  const MCInstrDesc &MCID = ARMInsts[Opcode];
  unsigned short NumDefs = MCID.getNumDefs();
  const MCOperandInfo *OpInfo = MCID.OpInfo;
  unsigned &OpIdx = NumOpsAdded;

  OpIdx = 0;

  assert(NumDefs > 0 && "NumDefs should be greater than 0 for MulFrm");
  assert(NumOps >= 3
         && OpInfo[0].RegClass == ARM::GPRRegClassID
         && OpInfo[1].RegClass == ARM::GPRRegClassID
         && OpInfo[2].RegClass == ARM::GPRRegClassID
         && "Expect three register operands");

  // Sanity check for the register encodings.
  if (BadRegsMulFrm(Opcode, insn))
    return false;

  // Instructions with two destination registers have RdLo{15-12} first.
  if (NumDefs == 2) {
    assert(NumOps >= 4 && OpInfo[3].RegClass == ARM::GPRRegClassID &&
           "Expect 4th register operand");
    MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                       decodeRd(insn))));
    ++OpIdx;
  }

  // The destination register: RdHi{19-16} or Rd{19-16}.
  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                     decodeRn(insn))));

  // The two src regsiters: Rn{3-0}, then Rm{11-8}.
  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                     decodeRm(insn))));
  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                     decodeRs(insn))));
  OpIdx += 3;

  // Many multiply instructions (e.g., MLA) have three src registers.
  // The third register operand is Ra{15-12}.
  if (OpIdx < NumOps && OpInfo[OpIdx].RegClass == ARM::GPRRegClassID) {
    MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                       decodeRd(insn))));
    ++OpIdx;
  }

  return true;
}

// Helper routines for disassembly of coprocessor instructions.

static bool LdStCopOpcode(unsigned Opcode) {
  if ((Opcode >= ARM::LDC2L_OFFSET && Opcode <= ARM::LDC_PRE) ||
      (Opcode >= ARM::STC2L_OFFSET && Opcode <= ARM::STC_PRE))
    return true;
  return false;
}
static bool CoprocessorOpcode(unsigned Opcode) {
  if (LdStCopOpcode(Opcode))
    return true;

  switch (Opcode) {
  default:
    return false;
  case ARM::CDP:  case ARM::CDP2:
  case ARM::MCR:  case ARM::MCR2:  case ARM::MRC:  case ARM::MRC2:
  case ARM::MCRR: case ARM::MCRR2: case ARM::MRRC: case ARM::MRRC2:
    return true;
  }
}
static inline unsigned GetCoprocessor(uint32_t insn) {
  return slice(insn, 11, 8);
}
static inline unsigned GetCopOpc1(uint32_t insn, bool CDP) {
  return CDP ? slice(insn, 23, 20) : slice(insn, 23, 21);
}
static inline unsigned GetCopOpc2(uint32_t insn) {
  return slice(insn, 7, 5);
}
static inline unsigned GetCopOpc(uint32_t insn) {
  return slice(insn, 7, 4);
}
// Most of the operands are in immediate forms, except Rd and Rn, which are ARM
// core registers.
//
// CDP, CDP2:                cop opc1 CRd CRn CRm opc2
//
// MCR, MCR2, MRC, MRC2:     cop opc1 Rd CRn CRm opc2
//
// MCRR, MCRR2, MRRC, MRRc2: cop opc Rd Rn CRm
//
// LDC_OFFSET, LDC_PRE, LDC_POST: cop CRd Rn R0 [+/-]imm8:00
// and friends
// STC_OFFSET, STC_PRE, STC_POST: cop CRd Rn R0 [+/-]imm8:00
// and friends
//                                        <-- addrmode2 -->
//
// LDC_OPTION:                    cop CRd Rn imm8
// and friends
// STC_OPTION:                    cop CRd Rn imm8
// and friends
//
static bool DisassembleCoprocessor(MCInst &MI, unsigned Opcode, uint32_t insn,
    unsigned short NumOps, unsigned &NumOpsAdded, BO B) {

  assert(NumOps >= 4 && "Num of operands >= 4 for coprocessor instr");

  unsigned &OpIdx = NumOpsAdded;
  // A8.6.92
  // if coproc == '101x' then SEE "Advanced SIMD and VFP"
  // But since the special instructions have more explicit encoding bits
  // specified, if coproc == 10 or 11, we should reject it as invalid.
  unsigned coproc = GetCoprocessor(insn);
  if ((Opcode == ARM::MCR || Opcode == ARM::MCRR ||
       Opcode == ARM::MRC || Opcode == ARM::MRRC) &&
      (coproc == 10 || coproc == 11)) {
    DEBUG(errs() << "Encoding error: coproc == 10 or 11 for MCR[R]/MR[R]C\n");
    return false;
  }

  bool OneCopOpc = (Opcode == ARM::MCRR || Opcode == ARM::MCRR2 ||
                    Opcode == ARM::MRRC || Opcode == ARM::MRRC2);

  // CDP/CDP2 has no GPR operand; the opc1 operand is also wider (Inst{23-20}).
  bool NoGPR = (Opcode == ARM::CDP || Opcode == ARM::CDP2);
  bool LdStCop = LdStCopOpcode(Opcode);
  bool RtOut = (Opcode == ARM::MRC || Opcode == ARM::MRC2);

  OpIdx = 0;

  if (RtOut) {
    MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                       decodeRd(insn))));
    ++OpIdx;
  }
  MI.addOperand(MCOperand::CreateImm(coproc));
  ++OpIdx;

  if (LdStCop) {
    // Unindex if P:W = 0b00 --> _OPTION variant
    unsigned PW = getPBit(insn) << 1 | getWBit(insn);

    MI.addOperand(MCOperand::CreateImm(decodeRd(insn)));

    MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                       decodeRn(insn))));
    OpIdx += 2;

    if (PW) {
      MI.addOperand(MCOperand::CreateReg(0));
      ARM_AM::AddrOpc AddrOpcode = getUBit(insn) ? ARM_AM::add : ARM_AM::sub;
      const MCInstrDesc &MCID = ARMInsts[Opcode];
      unsigned IndexMode =
                 (MCID.TSFlags & ARMII::IndexModeMask) >> ARMII::IndexModeShift;
      unsigned Offset = ARM_AM::getAM2Opc(AddrOpcode, slice(insn, 7, 0) << 2,
                                          ARM_AM::no_shift, IndexMode);
      MI.addOperand(MCOperand::CreateImm(Offset));
      OpIdx += 2;
    } else {
      MI.addOperand(MCOperand::CreateImm(slice(insn, 7, 0)));
      ++OpIdx;
    }
  } else {
    MI.addOperand(MCOperand::CreateImm(OneCopOpc ? GetCopOpc(insn)
                                                 : GetCopOpc1(insn, NoGPR)));
    ++OpIdx;

    if (!RtOut) {
      MI.addOperand(NoGPR ? MCOperand::CreateImm(decodeRd(insn))
                          : MCOperand::CreateReg(
                                getRegisterEnum(B, ARM::GPRRegClassID,
                                                decodeRd(insn))));
      ++OpIdx;
    }

    MI.addOperand(OneCopOpc ? MCOperand::CreateReg(
                                getRegisterEnum(B, ARM::GPRRegClassID,
                                                decodeRn(insn)))
                            : MCOperand::CreateImm(decodeRn(insn)));

    MI.addOperand(MCOperand::CreateImm(decodeRm(insn)));

    OpIdx += 2;

    if (!OneCopOpc) {
      MI.addOperand(MCOperand::CreateImm(GetCopOpc2(insn)));
      ++OpIdx;
    }
  }

  return true;
}

// Branch Instructions.
// BL: SignExtend(Imm24:'00', 32)
// Bcc, BL_pred: SignExtend(Imm24:'00', 32) Pred0 Pred1
// SMC: ZeroExtend(imm4, 32)
// SVC: ZeroExtend(Imm24, 32)
//
// Various coprocessor instructions are assigned BrFrm arbitrarily.
// Delegates to DisassembleCoprocessor() helper function.
//
// MRS/MRSsys: Rd
// MSR/MSRsys: Rm mask=Inst{19-16}
// BXJ:        Rm
// MSRi/MSRsysi: so_imm
// SRSW/SRS: ldstm_mode:$amode mode_imm
// RFEW/RFE: ldstm_mode:$amode Rn
static bool DisassembleBrFrm(MCInst &MI, unsigned Opcode, uint32_t insn,
    unsigned short NumOps, unsigned &NumOpsAdded, BO B) {

  if (CoprocessorOpcode(Opcode))
    return DisassembleCoprocessor(MI, Opcode, insn, NumOps, NumOpsAdded, B);

  const MCOperandInfo *OpInfo = ARMInsts[Opcode].OpInfo;
  if (!OpInfo) return false;

  // MRS and MRSsys take one GPR reg Rd.
  if (Opcode == ARM::MRS || Opcode == ARM::MRSsys) {
    assert(NumOps >= 1 && OpInfo[0].RegClass == ARM::GPRRegClassID &&
           "Reg operand expected");
    MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                       decodeRd(insn))));
    NumOpsAdded = 1;
    return true;
  }
  // BXJ takes one GPR reg Rm.
  if (Opcode == ARM::BXJ) {
    assert(NumOps >= 1 && OpInfo[0].RegClass == ARM::GPRRegClassID &&
           "Reg operand expected");
    MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                       decodeRm(insn))));
    NumOpsAdded = 1;
    return true;
  }
  // MSR take a mask, followed by one GPR reg Rm. The mask contains the R Bit in
  // bit 4, and the special register fields in bits 3-0.
  if (Opcode == ARM::MSR) {
    assert(NumOps >= 1 && OpInfo[1].RegClass == ARM::GPRRegClassID &&
           "Reg operand expected");
    MI.addOperand(MCOperand::CreateImm(slice(insn, 22, 22) << 4 /* R Bit */ |
                                       slice(insn, 19, 16) /* Special Reg */ ));
    MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                       decodeRm(insn))));
    NumOpsAdded = 2;
    return true;
  }
  // MSRi take a mask, followed by one so_imm operand. The mask contains the
  // R Bit in bit 4, and the special register fields in bits 3-0.
  if (Opcode == ARM::MSRi) {
    // A5.2.11 MSR (immediate), and hints & B6.1.6 MSR (immediate)
    // The hints instructions have more specific encodings, so if mask == 0,
    // we should reject this as an invalid instruction.
    if (slice(insn, 19, 16) == 0)
      return false;
    MI.addOperand(MCOperand::CreateImm(slice(insn, 22, 22) << 4 /* R Bit */ |
                                       slice(insn, 19, 16) /* Special Reg */ ));
    // SOImm is 4-bit rotate amount in bits 11-8 with 8-bit imm in bits 7-0.
    // A5.2.4 Rotate amount is twice the numeric value of Inst{11-8}.
    // See also ARMAddressingModes.h: getSOImmValImm() and getSOImmValRot().
    unsigned Rot = (insn >> ARMII::SoRotImmShift) & 0xF;
    unsigned Imm = insn & 0xFF;
    MI.addOperand(MCOperand::CreateImm(ARM_AM::rotr32(Imm, 2*Rot)));
    NumOpsAdded = 2;
    return true;
  }
  if (Opcode == ARM::SRSW || Opcode == ARM::SRS ||
      Opcode == ARM::RFEW || Opcode == ARM::RFE) {
    ARM_AM::AMSubMode SubMode = getAMSubModeForBits(getPUBits(insn));
    MI.addOperand(MCOperand::CreateImm(ARM_AM::getAM4ModeImm(SubMode)));

    if (Opcode == ARM::SRSW || Opcode == ARM::SRS)
      MI.addOperand(MCOperand::CreateImm(slice(insn, 4, 0)));
    else
      MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                         decodeRn(insn))));
    NumOpsAdded = 3;
    return true;
  }

  assert((Opcode == ARM::Bcc || Opcode == ARM::BL || Opcode == ARM::BL_pred
          || Opcode == ARM::SMC || Opcode == ARM::SVC) &&
         "Unexpected Opcode");

  assert(NumOps >= 1 && OpInfo[0].RegClass < 0 && "Imm operand expected");

  int Imm32 = 0;
  if (Opcode == ARM::SMC) {
    // ZeroExtend(imm4, 32) where imm24 = Inst{3-0}.
    Imm32 = slice(insn, 3, 0);
  } else if (Opcode == ARM::SVC) {
    // ZeroExtend(imm24, 32) where imm24 = Inst{23-0}.
    Imm32 = slice(insn, 23, 0);
  } else {
    // SignExtend(imm24:'00', 32) where imm24 = Inst{23-0}.
    unsigned Imm26 = slice(insn, 23, 0) << 2;
    //Imm32 = signextend<signed int, 26>(Imm26);
    Imm32 = SignExtend32<26>(Imm26);
  }

  MI.addOperand(MCOperand::CreateImm(Imm32));
  NumOpsAdded = 1;

  return true;
}

// Misc. Branch Instructions.
// BX_RET, MOVPCLR
// BLX, BLX_pred, BX, BX_pred
// BLXi
static bool DisassembleBrMiscFrm(MCInst &MI, unsigned Opcode, uint32_t insn,
    unsigned short NumOps, unsigned &NumOpsAdded, BO B) {

  const MCOperandInfo *OpInfo = ARMInsts[Opcode].OpInfo;
  if (!OpInfo) return false;

  unsigned &OpIdx = NumOpsAdded;

  OpIdx = 0;

  // BX_RET and MOVPCLR have only two predicate operands; do an early return.
  if (Opcode == ARM::BX_RET || Opcode == ARM::MOVPCLR)
    return true;

  // BLX and BX take one GPR reg.
  if (Opcode == ARM::BLX || Opcode == ARM::BLX_pred ||
      Opcode == ARM::BX || Opcode == ARM::BX_pred) {
    assert(NumOps >= 1 && OpInfo[OpIdx].RegClass == ARM::GPRRegClassID &&
           "Reg operand expected");
    MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                       decodeRm(insn))));
    OpIdx = 1;
    return true;
  }

  // BLXi takes imm32 (the PC offset).
  if (Opcode == ARM::BLXi) {
    assert(NumOps >= 1 && OpInfo[0].RegClass < 0 && "Imm operand expected");
    // SignExtend(imm24:H:'0', 32) where imm24 = Inst{23-0} and H = Inst{24}.
    unsigned Imm26 = slice(insn, 23, 0) << 2 | slice(insn, 24, 24) << 1;
    int Imm32 = SignExtend32<26>(Imm26);
    MI.addOperand(MCOperand::CreateImm(Imm32));
    OpIdx = 1;
    return true;
  }

  return false;
}

static inline bool getBFCInvMask(uint32_t insn, uint32_t &mask) {
  uint32_t lsb = slice(insn, 11, 7);
  uint32_t msb = slice(insn, 20, 16);
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

// Standard data-processing instructions allow PC as a register specifier,
// but we should reject other DPFrm instructions with PC as registers.
static bool BadRegsDPFrm(unsigned Opcode, uint32_t insn) {
  switch (Opcode) {
  default:
    // Did we miss an opcode?
    if (decodeRd(insn) == 15 || decodeRn(insn) == 15 || decodeRm(insn) == 15) {
      DEBUG(errs() << "DPFrm with bad reg specifier(s)\n");
      return true;
    }
  case ARM::ADCrr:  case ARM::ADDSrr: case ARM::ADDrr:  case ARM::ANDrr:
  case ARM::BICrr:  case ARM::CMNzrr: case ARM::CMPrr:  case ARM::EORrr:
  case ARM::ORRrr:  case ARM::RSBrr:  case ARM::RSCrr:  case ARM::SBCrr:
  case ARM::SUBSrr: case ARM::SUBrr:  case ARM::TEQrr:  case ARM::TSTrr:
    return false;
  }
}

// A major complication is the fact that some of the saturating add/subtract
// operations have Rd Rm Rn, instead of the "normal" Rd Rn Rm.
// They are QADD, QDADD, QDSUB, and QSUB.
static bool DisassembleDPFrm(MCInst &MI, unsigned Opcode, uint32_t insn,
    unsigned short NumOps, unsigned &NumOpsAdded, BO B) {

  const MCInstrDesc &MCID = ARMInsts[Opcode];
  unsigned short NumDefs = MCID.getNumDefs();
  bool isUnary = isUnaryDP(MCID.TSFlags);
  const MCOperandInfo *OpInfo = MCID.OpInfo;
  unsigned &OpIdx = NumOpsAdded;

  OpIdx = 0;

  // Disassemble register def if there is one.
  if (NumDefs && (OpInfo[OpIdx].RegClass == ARM::GPRRegClassID)) {
    MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                       decodeRd(insn))));
    ++OpIdx;
  }

  // Now disassemble the src operands.
  if (OpIdx >= NumOps)
    return false;

  // Special-case handling of BFC/BFI/SBFX/UBFX.
  if (Opcode == ARM::BFC || Opcode == ARM::BFI) {
    // A8.6.17 BFC & A8.6.18 BFI
    // Sanity check Rd.
    if (decodeRd(insn) == 15)
      return false;
    MI.addOperand(MCOperand::CreateReg(0));
    if (Opcode == ARM::BFI) {
      MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                         decodeRm(insn))));
      ++OpIdx;
    }
    uint32_t mask = 0;
    if (!getBFCInvMask(insn, mask))
      return false;

    MI.addOperand(MCOperand::CreateImm(mask));
    OpIdx += 2;
    return true;
  }
  if (Opcode == ARM::SBFX || Opcode == ARM::UBFX) {
    // Sanity check Rd and Rm.
    if (decodeRd(insn) == 15 || decodeRm(insn) == 15)
      return false;
    MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                       decodeRm(insn))));
    MI.addOperand(MCOperand::CreateImm(slice(insn, 11, 7)));
    MI.addOperand(MCOperand::CreateImm(slice(insn, 20, 16) + 1));
    OpIdx += 3;
    return true;
  }

  bool RmRn = (Opcode == ARM::QADD || Opcode == ARM::QDADD ||
               Opcode == ARM::QDSUB || Opcode == ARM::QSUB);

  // BinaryDP has an Rn operand.
  if (!isUnary) {
    assert(OpInfo[OpIdx].RegClass == ARM::GPRRegClassID &&
           "Reg operand expected");
    MI.addOperand(MCOperand::CreateReg(
                    getRegisterEnum(B, ARM::GPRRegClassID,
                                    RmRn ? decodeRm(insn) : decodeRn(insn))));
    ++OpIdx;
  }

  // If this is a two-address operand, skip it, e.g., MOVCCr operand 1.
  if (isUnary && (MCID.getOperandConstraint(OpIdx, MCOI::TIED_TO) != -1)) {
    MI.addOperand(MCOperand::CreateReg(0));
    ++OpIdx;
  }

  // Now disassemble operand 2.
  if (OpIdx >= NumOps)
    return false;

  if (OpInfo[OpIdx].RegClass == ARM::GPRRegClassID) {
    // We have a reg/reg form.
    // Assert disabled because saturating operations, e.g., A8.6.127 QASX, are
    // routed here as well.
    // assert(getIBit(insn) == 0 && "I_Bit != '0' reg/reg form");
    if (BadRegsDPFrm(Opcode, insn))
      return false;
    MI.addOperand(MCOperand::CreateReg(
                    getRegisterEnum(B, ARM::GPRRegClassID,
                                    RmRn? decodeRn(insn) : decodeRm(insn))));
    ++OpIdx;
  } else if (Opcode == ARM::MOVi16 || Opcode == ARM::MOVTi16) {
    // These two instructions don't allow d as 15.
    if (decodeRd(insn) == 15)
      return false;
    // We have an imm16 = imm4:imm12 (imm4=Inst{19:16}, imm12 = Inst{11:0}).
    assert(getIBit(insn) == 1 && "I_Bit != '1' reg/imm form");
    unsigned Imm16 = slice(insn, 19, 16) << 12 | slice(insn, 11, 0);
    if (!B->tryAddingSymbolicOperand(Imm16, 4, MI))
      MI.addOperand(MCOperand::CreateImm(Imm16));
    ++OpIdx;
  } else {
    // We have a reg/imm form.
    // SOImm is 4-bit rotate amount in bits 11-8 with 8-bit imm in bits 7-0.
    // A5.2.4 Rotate amount is twice the numeric value of Inst{11-8}.
    // See also ARMAddressingModes.h: getSOImmValImm() and getSOImmValRot().
    assert(getIBit(insn) == 1 && "I_Bit != '1' reg/imm form");
    unsigned Rot = (insn >> ARMII::SoRotImmShift) & 0xF;
    unsigned Imm = insn & 0xFF;
    MI.addOperand(MCOperand::CreateImm(ARM_AM::rotr32(Imm, 2*Rot)));
    ++OpIdx;
  }

  return true;
}

static bool DisassembleDPSoRegRegFrm(MCInst &MI, unsigned Opcode, uint32_t insn,
    unsigned short NumOps, unsigned &NumOpsAdded, BO B) {

  const MCInstrDesc &MCID = ARMInsts[Opcode];
  unsigned short NumDefs = MCID.getNumDefs();
  bool isUnary = isUnaryDP(MCID.TSFlags);
  const MCOperandInfo *OpInfo = MCID.OpInfo;
  unsigned &OpIdx = NumOpsAdded;

  OpIdx = 0;

  // Disassemble register def if there is one.
  if (NumDefs && (OpInfo[OpIdx].RegClass == ARM::GPRRegClassID)) {
    MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                       decodeRd(insn))));
    ++OpIdx;
  }

  // Disassemble the src operands.
  if (OpIdx >= NumOps)
    return false;

  // BinaryDP has an Rn operand.
  if (!isUnary) {
    assert(OpInfo[OpIdx].RegClass == ARM::GPRRegClassID &&
           "Reg operand expected");
    MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                       decodeRn(insn))));
    ++OpIdx;
  }

  // If this is a two-address operand, skip it, e.g., MOVCCs operand 1.
  if (isUnary && (MCID.getOperandConstraint(OpIdx, MCOI::TIED_TO) != -1)) {
    MI.addOperand(MCOperand::CreateReg(0));
    ++OpIdx;
  }

  // Disassemble operand 2, which consists of three components.
  if (OpIdx + 2 >= NumOps)
    return false;

  assert((OpInfo[OpIdx].RegClass == ARM::GPRRegClassID) &&
         (OpInfo[OpIdx+1].RegClass == ARM::GPRRegClassID) &&
         (OpInfo[OpIdx+2].RegClass < 0) &&
         "Expect 3 reg operands");

  // Register-controlled shifts have Inst{7} = 0 and Inst{4} = 1.
  unsigned Rs = slice(insn, 4, 4);

  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                     decodeRm(insn))));
  if (Rs) {
    // If Inst{7} != 0, we should reject this insn as an invalid encoding.
    if (slice(insn, 7, 7))
      return false;

    // A8.6.3 ADC (register-shifted register)
    // if d == 15 || n == 15 || m == 15 || s == 15 then UNPREDICTABLE;
    // 
    // This also accounts for shift instructions (register) where, fortunately,
    // Inst{19-16} = 0b0000.
    // A8.6.89 LSL (register)
    // if d == 15 || n == 15 || m == 15 then UNPREDICTABLE;
    if (decodeRd(insn) == 15 || decodeRn(insn) == 15 ||
        decodeRm(insn) == 15 || decodeRs(insn) == 15)
      return false;
    
    // Register-controlled shifts: [Rm, Rs, shift].
    MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                       decodeRs(insn))));
    // Inst{6-5} encodes the shift opcode.
    ARM_AM::ShiftOpc ShOp = getShiftOpcForBits(slice(insn, 6, 5));
    MI.addOperand(MCOperand::CreateImm(ARM_AM::getSORegOpc(ShOp, 0)));
  } else {
    // Constant shifts: [Rm, reg0, shift_imm].
    MI.addOperand(MCOperand::CreateReg(0)); // NoRegister
    // Inst{6-5} encodes the shift opcode.
    ARM_AM::ShiftOpc ShOp = getShiftOpcForBits(slice(insn, 6, 5));
    // Inst{11-7} encodes the imm5 shift amount.
    unsigned ShImm = slice(insn, 11, 7);

    // A8.4.1.  Possible rrx or shift amount of 32...
    getImmShiftSE(ShOp, ShImm);
    MI.addOperand(MCOperand::CreateImm(ARM_AM::getSORegOpc(ShOp, ShImm)));
  }
  OpIdx += 3;

  return true;
}

static bool DisassembleDPSoRegImmFrm(MCInst &MI, unsigned Opcode, uint32_t insn,
    unsigned short NumOps, unsigned &NumOpsAdded, BO B) {

  const MCInstrDesc &MCID = ARMInsts[Opcode];
  unsigned short NumDefs = MCID.getNumDefs();
  bool isUnary = isUnaryDP(MCID.TSFlags);
  const MCOperandInfo *OpInfo = MCID.OpInfo;
  unsigned &OpIdx = NumOpsAdded;

  OpIdx = 0;

  // Disassemble register def if there is one.
  if (NumDefs && (OpInfo[OpIdx].RegClass == ARM::GPRRegClassID)) {
    MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                       decodeRd(insn))));
    ++OpIdx;
  }

  // Disassemble the src operands.
  if (OpIdx >= NumOps)
    return false;

  // BinaryDP has an Rn operand.
  if (!isUnary) {
    assert(OpInfo[OpIdx].RegClass == ARM::GPRRegClassID &&
           "Reg operand expected");
    MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                       decodeRn(insn))));
    ++OpIdx;
  }

  // If this is a two-address operand, skip it, e.g., MOVCCs operand 1.
  if (isUnary && (MCID.getOperandConstraint(OpIdx, MCOI::TIED_TO) != -1)) {
    MI.addOperand(MCOperand::CreateReg(0));
    ++OpIdx;
  }

  // Disassemble operand 2, which consists of two components.
  if (OpIdx + 1 >= NumOps)
    return false;

  assert((OpInfo[OpIdx].RegClass == ARM::GPRRegClassID) &&
         (OpInfo[OpIdx+1].RegClass < 0) &&
         "Expect 2 reg operands");

  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                     decodeRm(insn))));

  // Inst{6-5} encodes the shift opcode.
  ARM_AM::ShiftOpc ShOp = getShiftOpcForBits(slice(insn, 6, 5));
  // Inst{11-7} encodes the imm5 shift amount.
  unsigned ShImm = slice(insn, 11, 7);

  // A8.4.1.  Possible rrx or shift amount of 32...
  getImmShiftSE(ShOp, ShImm);
  MI.addOperand(MCOperand::CreateImm(ARM_AM::getSORegOpc(ShOp, ShImm)));

  OpIdx += 2;

  return true;
}


static bool BadRegsLdStFrm(unsigned Opcode, uint32_t insn, bool Store, bool WBack,
                           bool Imm) {
  const StringRef Name = ARMInsts[Opcode].Name;
  unsigned Rt = decodeRd(insn);
  unsigned Rn = decodeRn(insn);
  unsigned Rm = decodeRm(insn);
  unsigned P  = getPBit(insn);
  unsigned W  = getWBit(insn);

  if (Store) {
    // Only STR (immediate, register) allows PC as the source.
    if (Name.startswith("STRB") && Rt == 15) {
      DEBUG(errs() << "if t == 15 then UNPREDICTABLE\n");
      return true;
    }
    if (WBack && (Rn == 15 || Rn == Rt)) {
      DEBUG(errs() << "if wback && (n == 15 || n == t) then UNPREDICTABLE\n");
      return true;
    }
    if (!Imm && Rm == 15) {
      DEBUG(errs() << "if m == 15 then UNPREDICTABLE\n");
      return true;
    }
  } else {
    // Only LDR (immediate, register) allows PC as the destination.
    if (Name.startswith("LDRB") && Rt == 15) {
      DEBUG(errs() << "if t == 15 then UNPREDICTABLE\n");
      return true;
    }
    if (Imm) {
      // Immediate
      if (Rn == 15) {
        // The literal form must be in offset mode; it's an encoding error
        // otherwise.
        if (!(P == 1 && W == 0)) {
          DEBUG(errs() << "Ld literal form with !(P == 1 && W == 0)\n");
          return true;
        }
        // LDRB (literal) does not allow PC as the destination.
        if (Opcode != ARM::LDRi12 && Rt == 15) {
          DEBUG(errs() << "if t == 15 then UNPREDICTABLE\n");
          return true;
        }
      } else {
        // Write back while Rn == Rt does not make sense.
        if (WBack && (Rn == Rt)) {
          DEBUG(errs() << "if wback && n == t then UNPREDICTABLE\n");
          return true;
        }
      }
    } else {
      // Register
      if (Rm == 15) {
        DEBUG(errs() << "if m == 15 then UNPREDICTABLE\n");
        return true;
      }
      if (WBack && (Rn == 15 || Rn == Rt)) {
        DEBUG(errs() << "if wback && (n == 15 || n == t) then UNPREDICTABLE\n");
        return true;
      }
    }
  }
  return false;
}

static bool DisassembleLdStFrm(MCInst &MI, unsigned Opcode, uint32_t insn,
    unsigned short NumOps, unsigned &NumOpsAdded, bool isStore, BO B) {

  const MCInstrDesc &MCID = ARMInsts[Opcode];
  bool isPrePost = isPrePostLdSt(MCID.TSFlags);
  const MCOperandInfo *OpInfo = MCID.OpInfo;
  if (!OpInfo) return false;

  unsigned &OpIdx = NumOpsAdded;

  OpIdx = 0;

  assert(((!isStore && MCID.getNumDefs() > 0) ||
          (isStore && (MCID.getNumDefs() == 0 || isPrePost)))
         && "Invalid arguments");

  // Operand 0 of a pre- and post-indexed store is the address base writeback.
  if (isPrePost && isStore) {
    assert(OpInfo[OpIdx].RegClass == ARM::GPRRegClassID &&
           "Reg operand expected");
    MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                       decodeRn(insn))));
    ++OpIdx;
  }

  // Disassemble the dst/src operand.
  if (OpIdx >= NumOps)
    return false;

  assert(OpInfo[OpIdx].RegClass == ARM::GPRRegClassID &&
         "Reg operand expected");
  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                     decodeRd(insn))));
  ++OpIdx;

  // After dst of a pre- and post-indexed load is the address base writeback.
  if (isPrePost && !isStore) {
    assert(OpInfo[OpIdx].RegClass == ARM::GPRRegClassID &&
           "Reg operand expected");
    MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                       decodeRn(insn))));
    ++OpIdx;
  }

  // Disassemble the base operand.
  if (OpIdx >= NumOps)
    return false;

  assert(OpInfo[OpIdx].RegClass == ARM::GPRRegClassID &&
         "Reg operand expected");
  assert((!isPrePost || (MCID.getOperandConstraint(OpIdx, MCOI::TIED_TO) != -1))
         && "Index mode or tied_to operand expected");
  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                     decodeRn(insn))));
  ++OpIdx;

  // For reg/reg form, base reg is followed by +/- reg shop imm.
  // For immediate form, it is followed by +/- imm12.
  // See also ARMAddressingModes.h (Addressing Mode #2).
  if (OpIdx + 1 >= NumOps)
    return false;

  if (BadRegsLdStFrm(Opcode, insn, isStore, isPrePost, getIBit(insn)==0))
    return false;

  ARM_AM::AddrOpc AddrOpcode = getUBit(insn) ? ARM_AM::add : ARM_AM::sub;
  unsigned IndexMode =
               (MCID.TSFlags & ARMII::IndexModeMask) >> ARMII::IndexModeShift;
  if (getIBit(insn) == 0) {
    // For pre- and post-indexed case, add a reg0 operand (Addressing Mode #2).
    // Otherwise, skip the reg operand since for addrmode_imm12, Rn has already
    // been populated.
    if (isPrePost) {
      MI.addOperand(MCOperand::CreateReg(0));
      OpIdx += 1;
    }

    unsigned Imm12 = slice(insn, 11, 0);
    if (Opcode == ARM::LDRBi12 || Opcode == ARM::LDRi12 ||
        Opcode == ARM::STRBi12 || Opcode == ARM::STRi12) {
      // Disassemble the 12-bit immediate offset, which is the second operand in
      // $addrmode_imm12 => (ops GPR:$base, i32imm:$offsimm).    
      int Offset = AddrOpcode == ARM_AM::add ? 1 * Imm12 : -1 * Imm12;
      MI.addOperand(MCOperand::CreateImm(Offset));
    } else {
      // Disassemble the 12-bit immediate offset, which is the second operand in
      // $am2offset => (ops GPR, i32imm).
      unsigned Offset = ARM_AM::getAM2Opc(AddrOpcode, Imm12, ARM_AM::no_shift,
                                          IndexMode);
      MI.addOperand(MCOperand::CreateImm(Offset));
    }
    OpIdx += 1;
  } else {
    // If Inst{25} = 1 and Inst{4} != 0, we should reject this as invalid.
    if (slice(insn,4,4) == 1)
      return false;

    // Disassemble the offset reg (Rm), shift type, and immediate shift length.
    MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                       decodeRm(insn))));
    // Inst{6-5} encodes the shift opcode.
    ARM_AM::ShiftOpc ShOp = getShiftOpcForBits(slice(insn, 6, 5));
    // Inst{11-7} encodes the imm5 shift amount.
    unsigned ShImm = slice(insn, 11, 7);

    // A8.4.1.  Possible rrx or shift amount of 32...
    getImmShiftSE(ShOp, ShImm);
    MI.addOperand(MCOperand::CreateImm(
                    ARM_AM::getAM2Opc(AddrOpcode, ShImm, ShOp, IndexMode)));
    OpIdx += 2;
  }

  return true;
}

static bool DisassembleLdFrm(MCInst &MI, unsigned Opcode, uint32_t insn,
    unsigned short NumOps, unsigned &NumOpsAdded, BO B) {
  return DisassembleLdStFrm(MI, Opcode, insn, NumOps, NumOpsAdded, false, B);
}

static bool DisassembleStFrm(MCInst &MI, unsigned Opcode, uint32_t insn,
    unsigned short NumOps, unsigned &NumOpsAdded, BO B) {
  return DisassembleLdStFrm(MI, Opcode, insn, NumOps, NumOpsAdded, true, B);
}

static bool HasDualReg(unsigned Opcode) {
  switch (Opcode) {
  default:
    return false;
  case ARM::LDRD: case ARM::LDRD_PRE: case ARM::LDRD_POST:
  case ARM::STRD: case ARM::STRD_PRE: case ARM::STRD_POST:
    return true;
  }
}

static bool DisassembleLdStMiscFrm(MCInst &MI, unsigned Opcode, uint32_t insn,
    unsigned short NumOps, unsigned &NumOpsAdded, bool isStore, BO B) {

  const MCInstrDesc &MCID = ARMInsts[Opcode];
  bool isPrePost = isPrePostLdSt(MCID.TSFlags);
  const MCOperandInfo *OpInfo = MCID.OpInfo;
  if (!OpInfo) return false;

  unsigned &OpIdx = NumOpsAdded;

  OpIdx = 0;

  assert(((!isStore && MCID.getNumDefs() > 0) ||
          (isStore && (MCID.getNumDefs() == 0 || isPrePost)))
         && "Invalid arguments");

  // Operand 0 of a pre- and post-indexed store is the address base writeback.
  if (isPrePost && isStore) {
    assert(OpInfo[OpIdx].RegClass == ARM::GPRRegClassID &&
           "Reg operand expected");
    MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                       decodeRn(insn))));
    ++OpIdx;
  }

  // Disassemble the dst/src operand.
  if (OpIdx >= NumOps)
    return false;

  assert(OpInfo[OpIdx].RegClass == ARM::GPRRegClassID &&
         "Reg operand expected");
  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                     decodeRd(insn))));
  ++OpIdx;

  // Fill in LDRD and STRD's second operand Rt operand.
  if (HasDualReg(Opcode)) {
    MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                       decodeRd(insn) + 1)));
    ++OpIdx;
  }

  // After dst of a pre- and post-indexed load is the address base writeback.
  if (isPrePost && !isStore) {
    assert(OpInfo[OpIdx].RegClass == ARM::GPRRegClassID &&
           "Reg operand expected");
    MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                       decodeRn(insn))));
    ++OpIdx;
  }

  // Disassemble the base operand.
  if (OpIdx >= NumOps)
    return false;

  assert(OpInfo[OpIdx].RegClass == ARM::GPRRegClassID &&
         "Reg operand expected");
  assert((!isPrePost || (MCID.getOperandConstraint(OpIdx, MCOI::TIED_TO) != -1))
         && "Offset mode or tied_to operand expected");
  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                     decodeRn(insn))));
  ++OpIdx;

  // For reg/reg form, base reg is followed by +/- reg.
  // For immediate form, it is followed by +/- imm8.
  // See also ARMAddressingModes.h (Addressing Mode #3).
  if (OpIdx + 1 >= NumOps)
    return false;

  assert((OpInfo[OpIdx].RegClass == ARM::GPRRegClassID) &&
         (OpInfo[OpIdx+1].RegClass < 0) &&
         "Expect 1 reg operand followed by 1 imm operand");

  ARM_AM::AddrOpc AddrOpcode = getUBit(insn) ? ARM_AM::add : ARM_AM::sub;
  unsigned IndexMode =
                 (MCID.TSFlags & ARMII::IndexModeMask) >> ARMII::IndexModeShift;
  if (getAM3IBit(insn) == 1) {
    MI.addOperand(MCOperand::CreateReg(0));

    // Disassemble the 8-bit immediate offset.
    unsigned Imm4H = (insn >> ARMII::ImmHiShift) & 0xF;
    unsigned Imm4L = insn & 0xF;
    unsigned Offset = ARM_AM::getAM3Opc(AddrOpcode, (Imm4H << 4) | Imm4L,
                                        IndexMode);
    MI.addOperand(MCOperand::CreateImm(Offset));
  } else {
    // Disassemble the offset reg (Rm).
    MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                       decodeRm(insn))));
    unsigned Offset = ARM_AM::getAM3Opc(AddrOpcode, 0, IndexMode);
    MI.addOperand(MCOperand::CreateImm(Offset));
  }
  OpIdx += 2;

  return true;
}

static bool DisassembleLdMiscFrm(MCInst &MI, unsigned Opcode, uint32_t insn,
    unsigned short NumOps, unsigned &NumOpsAdded, BO B) {
  return DisassembleLdStMiscFrm(MI, Opcode, insn, NumOps, NumOpsAdded, false,
                                B);
}

static bool DisassembleStMiscFrm(MCInst &MI, unsigned Opcode, uint32_t insn,
    unsigned short NumOps, unsigned &NumOpsAdded, BO B) {
  return DisassembleLdStMiscFrm(MI, Opcode, insn, NumOps, NumOpsAdded, true, B);
}

// The algorithm for disassembly of LdStMulFrm is different from others because
// it explicitly populates the two predicate operands after the base register.
// After that, we need to populate the reglist with each affected register
// encoded as an MCOperand.
static bool DisassembleLdStMulFrm(MCInst &MI, unsigned Opcode, uint32_t insn,
    unsigned short NumOps, unsigned &NumOpsAdded, BO B) {

  assert(NumOps >= 4 && "LdStMulFrm expects NumOps >= 4");
  NumOpsAdded = 0;

  unsigned Base = getRegisterEnum(B, ARM::GPRRegClassID, decodeRn(insn));

  // Writeback to base, if necessary.
  if (Opcode == ARM::LDMIA_UPD || Opcode == ARM::STMIA_UPD ||
      Opcode == ARM::LDMDA_UPD || Opcode == ARM::STMDA_UPD ||
      Opcode == ARM::LDMDB_UPD || Opcode == ARM::STMDB_UPD ||
      Opcode == ARM::LDMIB_UPD || Opcode == ARM::STMIB_UPD) {
    MI.addOperand(MCOperand::CreateReg(Base));
    ++NumOpsAdded;
  }

  // Add the base register operand.
  MI.addOperand(MCOperand::CreateReg(Base));

  // Handling the two predicate operands before the reglist.
  int64_t CondVal = getCondField(insn);
  if (CondVal == 0xF)
    return false;
  MI.addOperand(MCOperand::CreateImm(CondVal));
  MI.addOperand(MCOperand::CreateReg(ARM::CPSR));

  NumOpsAdded += 3;

  // Fill the variadic part of reglist.
  unsigned RegListBits = insn & ((1 << 16) - 1);
  for (unsigned i = 0; i < 16; ++i) {
    if ((RegListBits >> i) & 1) {
      MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                         i)));
      ++NumOpsAdded;
    }
  }

  return true;
}

// LDREX, LDREXB, LDREXH: Rd Rn
// LDREXD:                Rd Rd+1 Rn
// STREX, STREXB, STREXH: Rd Rm Rn
// STREXD:                Rd Rm Rm+1 Rn
//
// SWP, SWPB:             Rd Rm Rn
static bool DisassembleLdStExFrm(MCInst &MI, unsigned Opcode, uint32_t insn,
    unsigned short NumOps, unsigned &NumOpsAdded, BO B) {

  const MCOperandInfo *OpInfo = ARMInsts[Opcode].OpInfo;
  if (!OpInfo) return false;

  unsigned &OpIdx = NumOpsAdded;

  OpIdx = 0;

  assert(NumOps >= 2
         && OpInfo[0].RegClass == ARM::GPRRegClassID
         && OpInfo[1].RegClass == ARM::GPRRegClassID
         && "Expect 2 reg operands");

  bool isStore = slice(insn, 20, 20) == 0;
  bool isDW = (Opcode == ARM::LDREXD || Opcode == ARM::STREXD);

  // Add the destination operand.
  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                     decodeRd(insn))));
  ++OpIdx;

  // Store register Exclusive needs a source operand.
  if (isStore) {
    MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                       decodeRm(insn))));
    ++OpIdx;

    if (isDW) {
      MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                         decodeRm(insn)+1)));
      ++OpIdx;
    }
  } else if (isDW) {
    MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                       decodeRd(insn)+1)));
    ++OpIdx;
  }

  // Finally add the pointer operand.
  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                     decodeRn(insn))));
  ++OpIdx;

  return true;
}

// Misc. Arithmetic Instructions.
// CLZ: Rd Rm
// PKHBT, PKHTB: Rd Rn Rm , LSL/ASR #imm5
// RBIT, REV, REV16, REVSH: Rd Rm
static bool DisassembleArithMiscFrm(MCInst &MI, unsigned Opcode, uint32_t insn,
    unsigned short NumOps, unsigned &NumOpsAdded, BO B) {

  const MCOperandInfo *OpInfo = ARMInsts[Opcode].OpInfo;
  unsigned &OpIdx = NumOpsAdded;

  OpIdx = 0;

  assert(NumOps >= 2
         && OpInfo[0].RegClass == ARM::GPRRegClassID
         && OpInfo[1].RegClass == ARM::GPRRegClassID
         && "Expect 2 reg operands");

  bool ThreeReg = NumOps > 2 && OpInfo[2].RegClass == ARM::GPRRegClassID;

  // Sanity check the registers, which should not be 15.
  if (decodeRd(insn) == 15 || decodeRm(insn) == 15)
    return false;
  if (ThreeReg && decodeRn(insn) == 15)
    return false;

  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                     decodeRd(insn))));
  ++OpIdx;

  if (ThreeReg) {
    assert(NumOps >= 4 && "Expect >= 4 operands");
    MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                       decodeRn(insn))));
    ++OpIdx;
  }

  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                     decodeRm(insn))));
  ++OpIdx;

  // If there is still an operand info left which is an immediate operand, add
  // an additional imm5 LSL/ASR operand.
  if (ThreeReg && OpInfo[OpIdx].RegClass < 0
      && !OpInfo[OpIdx].isPredicate() && !OpInfo[OpIdx].isOptionalDef()) {
    // Extract the 5-bit immediate field Inst{11-7}.
    unsigned ShiftAmt = (insn >> ARMII::ShiftShift) & 0x1F;
    if (Opcode == ARM::PKHBT || Opcode == ARM::PKHTB)
      MI.addOperand(MCOperand::CreateImm(ShiftAmt));
     else
      MI.addOperand(MCOperand::CreateImm(ARM_AM::getSORegOpc(ARM_AM::no_shift,
                                                             ShiftAmt)));
    ++OpIdx;
  }

  return true;
}

/// DisassembleSatFrm - Disassemble saturate instructions:
/// SSAT, SSAT16, USAT, and USAT16.
static bool DisassembleSatFrm(MCInst &MI, unsigned Opcode, uint32_t insn,
    unsigned short NumOps, unsigned &NumOpsAdded, BO B) {

  // A8.6.183 SSAT
  // if d == 15 || n == 15 then UNPREDICTABLE;
  if (decodeRd(insn) == 15 || decodeRm(insn) == 15)
    return false;

  const MCInstrDesc &MCID = ARMInsts[Opcode];
  NumOpsAdded = MCID.getNumOperands() - 2; // ignore predicate operands

  // Disassemble register def.
  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                     decodeRd(insn))));

  unsigned Pos = slice(insn, 20, 16);
  if (Opcode == ARM::SSAT || Opcode == ARM::SSAT16)
    Pos += 1;
  MI.addOperand(MCOperand::CreateImm(Pos));

  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                     decodeRm(insn))));

  if (NumOpsAdded == 4) {
    ARM_AM::ShiftOpc Opc = (slice(insn, 6, 6) != 0 ? ARM_AM::asr : ARM_AM::lsl);
    // Inst{11-7} encodes the imm5 shift amount.
    unsigned ShAmt = slice(insn, 11, 7);
    if (ShAmt == 0) {
      // A8.6.183.  Possible ASR shift amount of 32...
      if (Opc == ARM_AM::asr)
        ShAmt = 32;
      else
        Opc = ARM_AM::no_shift;
    }
    MI.addOperand(MCOperand::CreateImm(ARM_AM::getSORegOpc(Opc, ShAmt)));
  }
  return true;
}

// Extend instructions.
// SXT* and UXT*: Rd [Rn] Rm [rot_imm].
// The 2nd operand register is Rn and the 3rd operand regsiter is Rm for the
// three register operand form.  Otherwise, Rn=0b1111 and only Rm is used.
static bool DisassembleExtFrm(MCInst &MI, unsigned Opcode, uint32_t insn,
    unsigned short NumOps, unsigned &NumOpsAdded, BO B) {

  // A8.6.220 SXTAB
  // if d == 15 || m == 15 then UNPREDICTABLE;
  if (decodeRd(insn) == 15 || decodeRm(insn) == 15)
    return false;

  const MCOperandInfo *OpInfo = ARMInsts[Opcode].OpInfo;
  unsigned &OpIdx = NumOpsAdded;

  OpIdx = 0;

  assert(NumOps >= 2
         && OpInfo[0].RegClass == ARM::GPRRegClassID
         && OpInfo[1].RegClass == ARM::GPRRegClassID
         && "Expect 2 reg operands");

  bool ThreeReg = NumOps > 2 && OpInfo[2].RegClass == ARM::GPRRegClassID;

  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                     decodeRd(insn))));
  ++OpIdx;

  if (ThreeReg) {
    MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                       decodeRn(insn))));
    ++OpIdx;
  }

  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                     decodeRm(insn))));
  ++OpIdx;

  // If there is still an operand info left which is an immediate operand, add
  // an additional rotate immediate operand.
  if (OpIdx < NumOps && OpInfo[OpIdx].RegClass < 0
      && !OpInfo[OpIdx].isPredicate() && !OpInfo[OpIdx].isOptionalDef()) {
    // Extract the 2-bit rotate field Inst{11-10}.
    unsigned rot = (insn >> ARMII::ExtRotImmShift) & 3;
    // Rotation by 8, 16, or 24 bits.
    MI.addOperand(MCOperand::CreateImm(rot << 3));
    ++OpIdx;
  }

  return true;
}

/////////////////////////////////////
//                                 //
//    Utility Functions For VFP    //
//                                 //
/////////////////////////////////////

// Extract/Decode Dd/Sd:
//
// SP => d = UInt(Vd:D)
// DP => d = UInt(D:Vd)
static unsigned decodeVFPRd(uint32_t insn, bool isSPVFP) {
  return isSPVFP ? (decodeRd(insn) << 1 | getDBit(insn))
                 : (decodeRd(insn) | getDBit(insn) << 4);
}

// Extract/Decode Dn/Sn:
//
// SP => n = UInt(Vn:N)
// DP => n = UInt(N:Vn)
static unsigned decodeVFPRn(uint32_t insn, bool isSPVFP) {
  return isSPVFP ? (decodeRn(insn) << 1 | getNBit(insn))
                 : (decodeRn(insn) | getNBit(insn) << 4);
}

// Extract/Decode Dm/Sm:
//
// SP => m = UInt(Vm:M)
// DP => m = UInt(M:Vm)
static unsigned decodeVFPRm(uint32_t insn, bool isSPVFP) {
  return isSPVFP ? (decodeRm(insn) << 1 | getMBit(insn))
                 : (decodeRm(insn) | getMBit(insn) << 4);
}

// A7.5.1
static APInt VFPExpandImm(unsigned char byte, unsigned N) {
  assert(N == 32 || N == 64);

  uint64_t Result;
  unsigned bit6 = slice(byte, 6, 6);
  if (N == 32) {
    Result = slice(byte, 7, 7) << 31 | slice(byte, 5, 0) << 19;
    if (bit6)
      Result |= 0x1f << 25;
    else
      Result |= 0x1 << 30;
  } else {
    Result = (uint64_t)slice(byte, 7, 7) << 63 |
             (uint64_t)slice(byte, 5, 0) << 48;
    if (bit6)
      Result |= 0xffULL << 54;
    else
      Result |= 0x1ULL << 62;
  }
  return APInt(N, Result);
}

// VFP Unary Format Instructions:
//
// VCMP[E]ZD, VCMP[E]ZS: compares one floating-point register with zero
// VCVTDS, VCVTSD: converts between double-precision and single-precision
// The rest of the instructions have homogeneous [VFP]Rd and [VFP]Rm registers.
static bool DisassembleVFPUnaryFrm(MCInst &MI, unsigned Opcode, uint32_t insn,
    unsigned short NumOps, unsigned &NumOpsAdded, BO B) {

  assert(NumOps >= 1 && "VFPUnaryFrm expects NumOps >= 1");

  const MCOperandInfo *OpInfo = ARMInsts[Opcode].OpInfo;
  unsigned &OpIdx = NumOpsAdded;

  OpIdx = 0;

  unsigned RegClass = OpInfo[OpIdx].RegClass;
  assert((RegClass == ARM::SPRRegClassID || RegClass == ARM::DPRRegClassID) &&
         "Reg operand expected");
  bool isSP = (RegClass == ARM::SPRRegClassID);

  MI.addOperand(MCOperand::CreateReg(
                  getRegisterEnum(B, RegClass, decodeVFPRd(insn, isSP))));
  ++OpIdx;

  // Early return for compare with zero instructions.
  if (Opcode == ARM::VCMPEZD || Opcode == ARM::VCMPEZS
      || Opcode == ARM::VCMPZD || Opcode == ARM::VCMPZS)
    return true;

  RegClass = OpInfo[OpIdx].RegClass;
  assert((RegClass == ARM::SPRRegClassID || RegClass == ARM::DPRRegClassID) &&
         "Reg operand expected");
  isSP = (RegClass == ARM::SPRRegClassID);

  MI.addOperand(MCOperand::CreateReg(
                  getRegisterEnum(B, RegClass, decodeVFPRm(insn, isSP))));
  ++OpIdx;

  return true;
}

// All the instructions have homogeneous [VFP]Rd, [VFP]Rn, and [VFP]Rm regs.
// Some of them have operand constraints which tie the first operand in the
// InOperandList to that of the dst.  As far as asm printing is concerned, this
// tied_to operand is simply skipped.
static bool DisassembleVFPBinaryFrm(MCInst &MI, unsigned Opcode, uint32_t insn,
    unsigned short NumOps, unsigned &NumOpsAdded, BO B) {

  assert(NumOps >= 3 && "VFPBinaryFrm expects NumOps >= 3");

  const MCInstrDesc &MCID = ARMInsts[Opcode];
  const MCOperandInfo *OpInfo = MCID.OpInfo;
  unsigned &OpIdx = NumOpsAdded;

  OpIdx = 0;

  unsigned RegClass = OpInfo[OpIdx].RegClass;
  assert((RegClass == ARM::SPRRegClassID || RegClass == ARM::DPRRegClassID) &&
         "Reg operand expected");
  bool isSP = (RegClass == ARM::SPRRegClassID);

  MI.addOperand(MCOperand::CreateReg(
                  getRegisterEnum(B, RegClass, decodeVFPRd(insn, isSP))));
  ++OpIdx;

  // Skip tied_to operand constraint.
  if (MCID.getOperandConstraint(OpIdx, MCOI::TIED_TO) != -1) {
    assert(NumOps >= 4 && "Expect >=4 operands");
    MI.addOperand(MCOperand::CreateReg(0));
    ++OpIdx;
  }

  MI.addOperand(MCOperand::CreateReg(
                  getRegisterEnum(B, RegClass, decodeVFPRn(insn, isSP))));
  ++OpIdx;

  MI.addOperand(MCOperand::CreateReg(
                  getRegisterEnum(B, RegClass, decodeVFPRm(insn, isSP))));
  ++OpIdx;

  return true;
}

// A8.6.295 vcvt (floating-point <-> integer)
// Int to FP: VSITOD, VSITOS, VUITOD, VUITOS
// FP to Int: VTOSI[Z|R]D, VTOSI[Z|R]S, VTOUI[Z|R]D, VTOUI[Z|R]S
//
// A8.6.297 vcvt (floating-point and fixed-point)
// Dd|Sd Dd|Sd(TIED_TO) #fbits(= 16|32 - UInt(imm4:i))
static bool DisassembleVFPConv1Frm(MCInst &MI, unsigned Opcode, uint32_t insn,
    unsigned short NumOps, unsigned &NumOpsAdded, BO B) {

  assert(NumOps >= 2 && "VFPConv1Frm expects NumOps >= 2");

  const MCInstrDesc &MCID = ARMInsts[Opcode];
  const MCOperandInfo *OpInfo = MCID.OpInfo;
  if (!OpInfo) return false;

  bool SP = slice(insn, 8, 8) == 0; // A8.6.295 & A8.6.297
  bool fixed_point = slice(insn, 17, 17) == 1; // A8.6.297
  unsigned RegClassID = SP ? ARM::SPRRegClassID : ARM::DPRRegClassID;

  if (fixed_point) {
    // A8.6.297
    assert(NumOps >= 3 && "Expect >= 3 operands");
    int size = slice(insn, 7, 7) == 0 ? 16 : 32;
    int fbits = size - (slice(insn,3,0) << 1 | slice(insn,5,5));
    MI.addOperand(MCOperand::CreateReg(
                    getRegisterEnum(B, RegClassID,
                                    decodeVFPRd(insn, SP))));

    assert(MCID.getOperandConstraint(1, MCOI::TIED_TO) != -1 &&
           "Tied to operand expected");
    MI.addOperand(MI.getOperand(0));

    assert(OpInfo[2].RegClass < 0 && !OpInfo[2].isPredicate() &&
           !OpInfo[2].isOptionalDef() && "Imm operand expected");
    MI.addOperand(MCOperand::CreateImm(fbits));

    NumOpsAdded = 3;
  } else {
    // A8.6.295
    // The Rd (destination) and Rm (source) bits have different interpretations
    // depending on their single-precisonness.
    unsigned d, m;
    if (slice(insn, 18, 18) == 1) { // to_integer operation
      d = decodeVFPRd(insn, true /* Is Single Precision */);
      MI.addOperand(MCOperand::CreateReg(
                      getRegisterEnum(B, ARM::SPRRegClassID, d)));
      m = decodeVFPRm(insn, SP);
      MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, RegClassID, m)));
    } else {
      d = decodeVFPRd(insn, SP);
      MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, RegClassID, d)));
      m = decodeVFPRm(insn, true /* Is Single Precision */);
      MI.addOperand(MCOperand::CreateReg(
                      getRegisterEnum(B, ARM::SPRRegClassID, m)));
    }
    NumOpsAdded = 2;
  }

  return true;
}

// VMOVRS - A8.6.330
// Rt => Rd; Sn => UInt(Vn:N)
static bool DisassembleVFPConv2Frm(MCInst &MI, unsigned Opcode, uint32_t insn,
    unsigned short NumOps, unsigned &NumOpsAdded, BO B) {

  assert(NumOps >= 2 && "VFPConv2Frm expects NumOps >= 2");

  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                     decodeRd(insn))));
  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::SPRRegClassID,
                                                     decodeVFPRn(insn, true))));
  NumOpsAdded = 2;
  return true;
}

// VMOVRRD - A8.6.332
// Rt => Rd; Rt2 => Rn; Dm => UInt(M:Vm)
//
// VMOVRRS - A8.6.331
// Rt => Rd; Rt2 => Rn; Sm => UInt(Vm:M); Sm1 = Sm+1
static bool DisassembleVFPConv3Frm(MCInst &MI, unsigned Opcode, uint32_t insn,
    unsigned short NumOps, unsigned &NumOpsAdded, BO B) {

  assert(NumOps >= 3 && "VFPConv3Frm expects NumOps >= 3");

  const MCOperandInfo *OpInfo = ARMInsts[Opcode].OpInfo;
  unsigned &OpIdx = NumOpsAdded;

  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                     decodeRd(insn))));
  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                     decodeRn(insn))));
  OpIdx = 2;

  if (OpInfo[OpIdx].RegClass == ARM::SPRRegClassID) {
    unsigned Sm = decodeVFPRm(insn, true);
    MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::SPRRegClassID,
                                                       Sm)));
    MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::SPRRegClassID,
                                                       Sm+1)));
    OpIdx += 2;
  } else {
    MI.addOperand(MCOperand::CreateReg(
                    getRegisterEnum(B, ARM::DPRRegClassID,
                                    decodeVFPRm(insn, false))));
    ++OpIdx;
  }
  return true;
}

// VMOVSR - A8.6.330
// Rt => Rd; Sn => UInt(Vn:N)
static bool DisassembleVFPConv4Frm(MCInst &MI, unsigned Opcode, uint32_t insn,
    unsigned short NumOps, unsigned &NumOpsAdded, BO B) {

  assert(NumOps >= 2 && "VFPConv4Frm expects NumOps >= 2");

  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::SPRRegClassID,
                                                     decodeVFPRn(insn, true))));
  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                     decodeRd(insn))));
  NumOpsAdded = 2;
  return true;
}

// VMOVDRR - A8.6.332
// Rt => Rd; Rt2 => Rn; Dm => UInt(M:Vm)
//
// VMOVRRS - A8.6.331
// Rt => Rd; Rt2 => Rn; Sm => UInt(Vm:M); Sm1 = Sm+1
static bool DisassembleVFPConv5Frm(MCInst &MI, unsigned Opcode, uint32_t insn,
    unsigned short NumOps, unsigned &NumOpsAdded, BO B) {

  assert(NumOps >= 3 && "VFPConv5Frm expects NumOps >= 3");

  const MCOperandInfo *OpInfo = ARMInsts[Opcode].OpInfo;
  unsigned &OpIdx = NumOpsAdded;

  OpIdx = 0;

  if (OpInfo[OpIdx].RegClass == ARM::SPRRegClassID) {
    unsigned Sm = decodeVFPRm(insn, true);
    MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::SPRRegClassID,
                                                       Sm)));
    MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::SPRRegClassID,
                                                       Sm+1)));
    OpIdx += 2;
  } else {
    MI.addOperand(MCOperand::CreateReg(
                    getRegisterEnum(B, ARM::DPRRegClassID,
                                    decodeVFPRm(insn, false))));
    ++OpIdx;
  }

  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                     decodeRd(insn))));
  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                     decodeRn(insn))));
  OpIdx += 2;
  return true;
}

// VFP Load/Store Instructions.
// VLDRD, VLDRS, VSTRD, VSTRS
static bool DisassembleVFPLdStFrm(MCInst &MI, unsigned Opcode, uint32_t insn,
    unsigned short NumOps, unsigned &NumOpsAdded, BO B) {

  assert(NumOps >= 3 && "VFPLdStFrm expects NumOps >= 3");

  bool isSPVFP = (Opcode == ARM::VLDRS || Opcode == ARM::VSTRS);
  unsigned RegClassID = isSPVFP ? ARM::SPRRegClassID : ARM::DPRRegClassID;

  // Extract Dd/Sd for operand 0.
  unsigned RegD = decodeVFPRd(insn, isSPVFP);

  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, RegClassID, RegD)));

  unsigned Base = getRegisterEnum(B, ARM::GPRRegClassID, decodeRn(insn));
  MI.addOperand(MCOperand::CreateReg(Base));

  // Next comes the AM5 Opcode.
  ARM_AM::AddrOpc AddrOpcode = getUBit(insn) ? ARM_AM::add : ARM_AM::sub;
  unsigned char Imm8 = insn & 0xFF;
  MI.addOperand(MCOperand::CreateImm(ARM_AM::getAM5Opc(AddrOpcode, Imm8)));

  NumOpsAdded = 3;

  return true;
}

// VFP Load/Store Multiple Instructions.
// We have an optional write back reg, the base, and two predicate operands.
// It is then followed by a reglist of either DPR(s) or SPR(s).
//
// VLDMD[_UPD], VLDMS[_UPD], VSTMD[_UPD], VSTMS[_UPD]
static bool DisassembleVFPLdStMulFrm(MCInst &MI, unsigned Opcode, uint32_t insn,
    unsigned short NumOps, unsigned &NumOpsAdded, BO B) {

  assert(NumOps >= 4 && "VFPLdStMulFrm expects NumOps >= 4");

  unsigned &OpIdx = NumOpsAdded;

  OpIdx = 0;

  unsigned Base = getRegisterEnum(B, ARM::GPRRegClassID, decodeRn(insn));

  // Writeback to base, if necessary.
  if (Opcode == ARM::VLDMDIA_UPD || Opcode == ARM::VLDMSIA_UPD ||
      Opcode == ARM::VLDMDDB_UPD || Opcode == ARM::VLDMSDB_UPD ||
      Opcode == ARM::VSTMDIA_UPD || Opcode == ARM::VSTMSIA_UPD ||
      Opcode == ARM::VSTMDDB_UPD || Opcode == ARM::VSTMSDB_UPD) {
    MI.addOperand(MCOperand::CreateReg(Base));
    ++OpIdx;
  }

  MI.addOperand(MCOperand::CreateReg(Base));

  // Handling the two predicate operands before the reglist.
  int64_t CondVal = getCondField(insn);
  if (CondVal == 0xF)
    return false;
  MI.addOperand(MCOperand::CreateImm(CondVal));
  MI.addOperand(MCOperand::CreateReg(ARM::CPSR));

  OpIdx += 3;

  bool isSPVFP = (Opcode == ARM::VLDMSIA     ||
                  Opcode == ARM::VLDMSIA_UPD || Opcode == ARM::VLDMSDB_UPD ||
                  Opcode == ARM::VSTMSIA     ||
                  Opcode == ARM::VSTMSIA_UPD || Opcode == ARM::VSTMSDB_UPD);
  unsigned RegClassID = isSPVFP ? ARM::SPRRegClassID : ARM::DPRRegClassID;

  // Extract Dd/Sd.
  unsigned RegD = decodeVFPRd(insn, isSPVFP);

  // Fill the variadic part of reglist.
  unsigned char Imm8 = insn & 0xFF;
  unsigned Regs = isSPVFP ? Imm8 : Imm8/2;

  // Apply some sanity checks before proceeding.
  if (Regs == 0 || (RegD + Regs) > 32 || (!isSPVFP && Regs > 16))
    return false;

  for (unsigned i = 0; i < Regs; ++i) {
    MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, RegClassID,
                                                       RegD + i)));
    ++OpIdx;
  }

  return true;
}

// Misc. VFP Instructions.
// FMSTAT (vmrs with Rt=0b1111, i.e., to apsr_nzcv and no register operand)
// FCONSTD (DPR and a VFPf64Imm operand)
// FCONSTS (SPR and a VFPf32Imm operand)
// VMRS/VMSR (GPR operand)
static bool DisassembleVFPMiscFrm(MCInst &MI, unsigned Opcode, uint32_t insn,
    unsigned short NumOps, unsigned &NumOpsAdded, BO B) {

  const MCOperandInfo *OpInfo = ARMInsts[Opcode].OpInfo;
  unsigned &OpIdx = NumOpsAdded;

  OpIdx = 0;

  if (Opcode == ARM::FMSTAT)
    return true;

  assert(NumOps >= 2 && "VFPMiscFrm expects >=2 operands");

  unsigned RegEnum = 0;
  switch (OpInfo[0].RegClass) {
  case ARM::DPRRegClassID:
    RegEnum = getRegisterEnum(B, ARM::DPRRegClassID, decodeVFPRd(insn, false));
    break;
  case ARM::SPRRegClassID:
    RegEnum = getRegisterEnum(B, ARM::SPRRegClassID, decodeVFPRd(insn, true));
    break;
  case ARM::GPRRegClassID:
    RegEnum = getRegisterEnum(B, ARM::GPRRegClassID, decodeRd(insn));
    break;
  default:
    assert(0 && "Invalid reg class id");
    return false;
  }

  MI.addOperand(MCOperand::CreateReg(RegEnum));
  ++OpIdx;

  // Extract/decode the f64/f32 immediate.
  if (OpIdx < NumOps && OpInfo[OpIdx].RegClass < 0
        && !OpInfo[OpIdx].isPredicate() && !OpInfo[OpIdx].isOptionalDef()) {
    // The asm syntax specifies the floating point value, not the 8-bit literal.
    APInt immRaw = VFPExpandImm(slice(insn,19,16) << 4 | slice(insn, 3, 0),
                             Opcode == ARM::FCONSTD ? 64 : 32);
    APFloat immFP = APFloat(immRaw, true);
    double imm = Opcode == ARM::FCONSTD ? immFP.convertToDouble() :
      immFP.convertToFloat();
    MI.addOperand(MCOperand::CreateFPImm(imm));

    ++OpIdx;
  }

  return true;
}

// DisassembleThumbFrm() is defined in ThumbDisassemblerCore.h file.
#include "ThumbDisassemblerCore.h"

/////////////////////////////////////////////////////
//                                                 //
//     Utility Functions For ARM Advanced SIMD     //
//                                                 //
/////////////////////////////////////////////////////

// The following NEON namings are based on A8.6.266 VABA, VABAL.  Notice that
// A8.6.303 VDUP (ARM core register)'s D/Vd pair is the N/Vn pair of VABA/VABAL.

// A7.3 Register encoding

// Extract/Decode NEON D/Vd:
//
// Note that for quadword, Qd = UInt(D:Vd<3:1>) = Inst{22:15-13}, whereas for
// doubleword, Dd = UInt(D:Vd).  We compensate for this difference by
// handling it in the getRegisterEnum() utility function.
// D = Inst{22}, Vd = Inst{15-12}
static unsigned decodeNEONRd(uint32_t insn) {
  return ((insn >> ARMII::NEON_D_BitShift) & 1) << 4
    | ((insn >> ARMII::NEON_RegRdShift) & ARMII::NEONRegMask);
}

// Extract/Decode NEON N/Vn:
//
// Note that for quadword, Qn = UInt(N:Vn<3:1>) = Inst{7:19-17}, whereas for
// doubleword, Dn = UInt(N:Vn).  We compensate for this difference by
// handling it in the getRegisterEnum() utility function.
// N = Inst{7}, Vn = Inst{19-16}
static unsigned decodeNEONRn(uint32_t insn) {
  return ((insn >> ARMII::NEON_N_BitShift) & 1) << 4
    | ((insn >> ARMII::NEON_RegRnShift) & ARMII::NEONRegMask);
}

// Extract/Decode NEON M/Vm:
//
// Note that for quadword, Qm = UInt(M:Vm<3:1>) = Inst{5:3-1}, whereas for
// doubleword, Dm = UInt(M:Vm).  We compensate for this difference by
// handling it in the getRegisterEnum() utility function.
// M = Inst{5}, Vm = Inst{3-0}
static unsigned decodeNEONRm(uint32_t insn) {
  return ((insn >> ARMII::NEON_M_BitShift) & 1) << 4
    | ((insn >> ARMII::NEON_RegRmShift) & ARMII::NEONRegMask);
}

namespace {
enum ElemSize {
  ESizeNA = 0,
  ESize8 = 8,
  ESize16 = 16,
  ESize32 = 32,
  ESize64 = 64
};
} // End of unnamed namespace

// size        field -> Inst{11-10}
// index_align field -> Inst{7-4}
//
// The Lane Index interpretation depends on the Data Size:
//   8  (encoded as size = 0b00) -> Index = index_align[3:1]
//   16 (encoded as size = 0b01) -> Index = index_align[3:2]
//   32 (encoded as size = 0b10) -> Index = index_align[3]
//
// Ref: A8.6.317 VLD4 (single 4-element structure to one lane).
static unsigned decodeLaneIndex(uint32_t insn) {
  unsigned size = insn >> 10 & 3;
  assert((size == 0 || size == 1 || size == 2) &&
         "Encoding error: size should be either 0, 1, or 2");

  unsigned index_align = insn >> 4 & 0xF;
  return (index_align >> 1) >> size;
}

// imm64 = AdvSIMDExpandImm(op, cmode, i:imm3:imm4)
// op = Inst{5}, cmode = Inst{11-8}
// i = Inst{24} (ARM architecture)
// imm3 = Inst{18-16}, imm4 = Inst{3-0}
// Ref: Table A7-15 Modified immediate values for Advanced SIMD instructions.
static uint64_t decodeN1VImm(uint32_t insn, ElemSize esize) {
  unsigned char op = (insn >> 5) & 1;
  unsigned char cmode = (insn >> 8) & 0xF;
  unsigned char Imm8 = ((insn >> 24) & 1) << 7 |
                       ((insn >> 16) & 7) << 4 |
                       (insn & 0xF);
  return (op << 12) | (cmode << 8) | Imm8;
}

// A8.6.339 VMUL, VMULL (by scalar)
// ESize16 => m = Inst{2-0} (Vm<2:0>) D0-D7
// ESize32 => m = Inst{3-0} (Vm<3:0>) D0-D15
static unsigned decodeRestrictedDm(uint32_t insn, ElemSize esize) {
  switch (esize) {
  case ESize16:
    return insn & 7;
  case ESize32:
    return insn & 0xF;
  default:
    assert(0 && "Unreachable code!");
    return 0;
  }
}

// A8.6.339 VMUL, VMULL (by scalar)
// ESize16 => index = Inst{5:3} (M:Vm<3>) D0-D7
// ESize32 => index = Inst{5}   (M)       D0-D15
static unsigned decodeRestrictedDmIndex(uint32_t insn, ElemSize esize) {
  switch (esize) {
  case ESize16:
    return (((insn >> 5) & 1) << 1) | ((insn >> 3) & 1);
  case ESize32:
    return (insn >> 5) & 1;
  default:
    assert(0 && "Unreachable code!");
    return 0;
  }
}

// A8.6.296 VCVT (between floating-point and fixed-point, Advanced SIMD)
// (64 - <fbits>) is encoded as imm6, i.e., Inst{21-16}.
static unsigned decodeVCVTFractionBits(uint32_t insn) {
  return 64 - ((insn >> 16) & 0x3F);
}

// A8.6.302 VDUP (scalar)
// ESize8  => index = Inst{19-17}
// ESize16 => index = Inst{19-18}
// ESize32 => index = Inst{19}
static unsigned decodeNVLaneDupIndex(uint32_t insn, ElemSize esize) {
  switch (esize) {
  case ESize8:
    return (insn >> 17) & 7;
  case ESize16:
    return (insn >> 18) & 3;
  case ESize32:
    return (insn >> 19) & 1;
  default:
    assert(0 && "Unspecified element size!");
    return 0;
  }
}

// A8.6.328 VMOV (ARM core register to scalar)
// A8.6.329 VMOV (scalar to ARM core register)
// ESize8  => index = Inst{21:6-5}
// ESize16 => index = Inst{21:6}
// ESize32 => index = Inst{21}
static unsigned decodeNVLaneOpIndex(uint32_t insn, ElemSize esize) {
  switch (esize) {
  case ESize8:
    return ((insn >> 21) & 1) << 2 | ((insn >> 5) & 3);
  case ESize16:
    return ((insn >> 21) & 1) << 1 | ((insn >> 6) & 1);
  case ESize32:
    return ((insn >> 21) & 1);
  default:
    assert(0 && "Unspecified element size!");
    return 0;
  }
}

// Imm6 = Inst{21-16}, L = Inst{7}
//
// LeftShift == true (A8.6.367 VQSHL, A8.6.387 VSLI):
// case L:imm6 of
//   '0001xxx' => esize = 8; shift_amount = imm6 - 8
//   '001xxxx' => esize = 16; shift_amount = imm6 - 16
//   '01xxxxx' => esize = 32; shift_amount = imm6 - 32
//   '1xxxxxx' => esize = 64; shift_amount = imm6
//
// LeftShift == false (A8.6.376 VRSHR, A8.6.368 VQSHRN):
// case L:imm6 of
//   '0001xxx' => esize = 8; shift_amount = 16 - imm6
//   '001xxxx' => esize = 16; shift_amount = 32 - imm6
//   '01xxxxx' => esize = 32; shift_amount = 64 - imm6
//   '1xxxxxx' => esize = 64; shift_amount = 64 - imm6
//
static unsigned decodeNVSAmt(uint32_t insn, bool LeftShift) {
  ElemSize esize = ESizeNA;
  unsigned L = (insn >> 7) & 1;
  unsigned imm6 = (insn >> 16) & 0x3F;
  if (L == 0) {
    if (imm6 >> 3 == 1)
      esize = ESize8;
    else if (imm6 >> 4 == 1)
      esize = ESize16;
    else if (imm6 >> 5 == 1)
      esize = ESize32;
    else
      assert(0 && "Wrong encoding of Inst{7:21-16}!");
  } else
    esize = ESize64;

  if (LeftShift)
    return esize == ESize64 ? imm6 : (imm6 - esize);
  else
    return esize == ESize64 ? (esize - imm6) : (2*esize - imm6);
}

// A8.6.305 VEXT
// Imm4 = Inst{11-8}
static unsigned decodeN3VImm(uint32_t insn) {
  return (insn >> 8) & 0xF;
}

// VLD*
//   D[d] D[d2] ... Rn [TIED_TO Rn] align [Rm]
// VLD*LN*
//   D[d] D[d2] ... Rn [TIED_TO Rn] align [Rm] TIED_TO ... imm(idx)
// VST*
//   Rn [TIED_TO Rn] align [Rm] D[d] D[d2] ...
// VST*LN*
//   Rn [TIED_TO Rn] align [Rm] D[d] D[d2] ... [imm(idx)]
//
// Correctly set VLD*/VST*'s TIED_TO GPR, as the asm printer needs it.
static bool DisassembleNLdSt0(MCInst &MI, unsigned Opcode, uint32_t insn,
    unsigned short NumOps, unsigned &NumOpsAdded, bool Store, bool DblSpaced,
    unsigned alignment, BO B) {

  const MCInstrDesc &MCID = ARMInsts[Opcode];
  const MCOperandInfo *OpInfo = MCID.OpInfo;

  // At least one DPR register plus addressing mode #6.
  assert(NumOps >= 3 && "Expect >= 3 operands");

  unsigned &OpIdx = NumOpsAdded;

  OpIdx = 0;

  // We have homogeneous NEON registers for Load/Store.
  unsigned RegClass = 0;

  // Double-spaced registers have increments of 2.
  unsigned Inc = DblSpaced ? 2 : 1;

  unsigned Rn = decodeRn(insn);
  unsigned Rm = decodeRm(insn);
  unsigned Rd = decodeNEONRd(insn);

  // A7.7.1 Advanced SIMD addressing mode.
  bool WB = Rm != 15;

  // LLVM Addressing Mode #6.
  unsigned RmEnum = 0;
  if (WB && Rm != 13)
    RmEnum = getRegisterEnum(B, ARM::GPRRegClassID, Rm);

  if (Store) {
    // Consume possible WB, AddrMode6, possible increment reg, the DPR/QPR's,
    // then possible lane index.
    assert(OpIdx < NumOps && OpInfo[0].RegClass == ARM::GPRRegClassID &&
           "Reg operand expected");

    if (WB) {
      MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                         Rn)));
      ++OpIdx;
    }

    assert((OpIdx+1) < NumOps && OpInfo[OpIdx].RegClass == ARM::GPRRegClassID &&
           OpInfo[OpIdx + 1].RegClass < 0 && "Addrmode #6 Operands expected");
    // addrmode6 := (ops GPR:$addr, i32imm)
    MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                       Rn)));
    MI.addOperand(MCOperand::CreateImm(alignment)); // Alignment
    OpIdx += 2;

    if (WB) {
      MI.addOperand(MCOperand::CreateReg(RmEnum));
      ++OpIdx;
    }

    assert(OpIdx < NumOps &&
           (OpInfo[OpIdx].RegClass == ARM::DPRRegClassID ||
            OpInfo[OpIdx].RegClass == ARM::QPRRegClassID) &&
           "Reg operand expected");

    RegClass = OpInfo[OpIdx].RegClass;
    while (OpIdx < NumOps && (unsigned)OpInfo[OpIdx].RegClass == RegClass) {
      MI.addOperand(MCOperand::CreateReg(
                      getRegisterEnum(B, RegClass, Rd)));
      Rd += Inc;
      ++OpIdx;
    }

    // Handle possible lane index.
    if (OpIdx < NumOps && OpInfo[OpIdx].RegClass < 0
        && !OpInfo[OpIdx].isPredicate() && !OpInfo[OpIdx].isOptionalDef()) {
      MI.addOperand(MCOperand::CreateImm(decodeLaneIndex(insn)));
      ++OpIdx;
    }

  } else {
    // Consume the DPR/QPR's, possible WB, AddrMode6, possible incrment reg,
    // possible TIED_TO DPR/QPR's (ignored), then possible lane index.
    RegClass = OpInfo[0].RegClass;

    while (OpIdx < NumOps && (unsigned)OpInfo[OpIdx].RegClass == RegClass) {
      MI.addOperand(MCOperand::CreateReg(
                      getRegisterEnum(B, RegClass, Rd)));
      Rd += Inc;
      ++OpIdx;
    }

    if (WB) {
      MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                         Rn)));
      ++OpIdx;
    }

    assert((OpIdx+1) < NumOps && OpInfo[OpIdx].RegClass == ARM::GPRRegClassID &&
           OpInfo[OpIdx + 1].RegClass < 0 && "Addrmode #6 Operands expected");
    // addrmode6 := (ops GPR:$addr, i32imm)
    MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                       Rn)));
    MI.addOperand(MCOperand::CreateImm(alignment)); // Alignment
    OpIdx += 2;

    if (WB) {
      MI.addOperand(MCOperand::CreateReg(RmEnum));
      ++OpIdx;
    }

    while (OpIdx < NumOps && (unsigned)OpInfo[OpIdx].RegClass == RegClass) {
      assert(MCID.getOperandConstraint(OpIdx, MCOI::TIED_TO) != -1 &&
             "Tied to operand expected");
      MI.addOperand(MCOperand::CreateReg(0));
      ++OpIdx;
    }

    // Handle possible lane index.
    if (OpIdx < NumOps && OpInfo[OpIdx].RegClass < 0
        && !OpInfo[OpIdx].isPredicate() && !OpInfo[OpIdx].isOptionalDef()) {
      MI.addOperand(MCOperand::CreateImm(decodeLaneIndex(insn)));
      ++OpIdx;
    }
  }

  // Accessing registers past the end of the NEON register file is not
  // defined.
  if (Rd > 32)
    return false;

  return true;
}

// A8.6.308, A8.6.311, A8.6.314, A8.6.317.
static bool Align4OneLaneInst(unsigned elem, unsigned size,
    unsigned index_align, unsigned & alignment) {
  unsigned bits = 0;
  switch (elem) {
  default:
    return false;
  case 1:
    // A8.6.308
    if (size == 0)
      return slice(index_align, 0, 0) == 0;
    else if (size == 1) {
      bits = slice(index_align, 1, 0);
      if (bits != 0 && bits != 1)
        return false;
      if (bits == 1)
        alignment = 16;
      return true;
    } else if (size == 2) {
      bits = slice(index_align, 2, 0);
      if (bits != 0 && bits != 3)
        return false;
      if (bits == 3)
        alignment = 32;
      return true;;
    }
    return true;
  case 2:
    // A8.6.311
    if (size == 0) {
      if (slice(index_align, 0, 0) == 1)
        alignment = 16;
      return true;
    } if (size == 1) {
      if (slice(index_align, 0, 0) == 1)
        alignment = 32;
      return true;
    } else if (size == 2) {
      if (slice(index_align, 1, 1) != 0)
        return false;
      if (slice(index_align, 0, 0) == 1)
        alignment = 64;
      return true;;
    }
    return true;
  case 3:
    // A8.6.314
    if (size == 0) {
      if (slice(index_align, 0, 0) != 0)
        return false;
      return true;
    } if (size == 1) {
      if (slice(index_align, 0, 0) != 0)
        return false;
      return true;
      return true;
    } else if (size == 2) {
      if (slice(index_align, 1, 0) != 0)
        return false;
      return true;;
    }
    return true;
  case 4:
    // A8.6.317
    if (size == 0) {
      if (slice(index_align, 0, 0) == 1)
        alignment = 32;
      return true;
    } if (size == 1) {
      if (slice(index_align, 0, 0) == 1)
        alignment = 64;
      return true;
    } else if (size == 2) {
      bits = slice(index_align, 1, 0);
      if (bits == 3)
        return false;
      if (bits == 1)
        alignment = 64;
      else if (bits == 2)
        alignment = 128;
      return true;;
    }
    return true;
  }
}

// A7.7
// If L (Inst{21}) == 0, store instructions.
// Find out about double-spaced-ness of the Opcode and pass it on to
// DisassembleNLdSt0().
static bool DisassembleNLdSt(MCInst &MI, unsigned Opcode, uint32_t insn,
    unsigned short NumOps, unsigned &NumOpsAdded, BO B) {

  const StringRef Name = ARMInsts[Opcode].Name;
  bool DblSpaced = false;
  // 0 represents standard alignment, i.e., unaligned data access.
  unsigned alignment = 0;

  unsigned elem = 0; // legal values: {1, 2, 3, 4}
  if (Name.startswith("VST1") || Name.startswith("VLD1"))
    elem = 1;

  if (Name.startswith("VST2") || Name.startswith("VLD2"))
    elem = 2;

  if (Name.startswith("VST3") || Name.startswith("VLD3"))
    elem = 3;

  if (Name.startswith("VST4") || Name.startswith("VLD4"))
    elem = 4;

  if (Name.find("LN") != std::string::npos) {
    // To one lane instructions.
    // See, for example, 8.6.317 VLD4 (single 4-element structure to one lane).

    // Utility function takes number of elements, size, and index_align.
    if (!Align4OneLaneInst(elem,
                           slice(insn, 11, 10),
                           slice(insn, 7, 4),
                           alignment))
      return false;

    // <size> == 16 && Inst{5} == 1 --> DblSpaced = true
    if (Name.endswith("16") || Name.endswith("16_UPD"))
      DblSpaced = slice(insn, 5, 5) == 1;

    // <size> == 32 && Inst{6} == 1 --> DblSpaced = true
    if (Name.endswith("32") || Name.endswith("32_UPD"))
      DblSpaced = slice(insn, 6, 6) == 1;
  } else if (Name.find("DUP") != std::string::npos) {
    // Single element (or structure) to all lanes.
    // Inst{9-8} encodes the number of element(s) in the structure, with:
    // 0b00 (VLD1DUP) (for this, a bit makes sense only for data size 16 and 32.
    // 0b01 (VLD2DUP)
    // 0b10 (VLD3DUP) (for this, a bit must be encoded as 0)
    // 0b11 (VLD4DUP)
    //
    // Inst{7-6} encodes the data size, with:
    // 0b00 => 8, 0b01 => 16, 0b10 => 32
    //
    // Inst{4} (the a bit) encodes the align action (0: standard alignment)
    unsigned elem = slice(insn, 9, 8) + 1;
    unsigned a = slice(insn, 4, 4);
    if (elem != 3) {
      // 0b11 is not a valid encoding for Inst{7-6}.
      if (slice(insn, 7, 6) == 3)
        return false;
      unsigned data_size = 8 << slice(insn, 7, 6);
      // For VLD1DUP, a bit makes sense only for data size of 16 and 32.
      if (a && data_size == 8)
        return false;

      // Now we can calculate the alignment!
      if (a)
        alignment = elem * data_size;
    } else {
      if (a) {
        // A8.6.315 VLD3 (single 3-element structure to all lanes)
        // The a bit must be encoded as 0.
        return false;
      }
    }
  } else {
    // Multiple n-element structures with type encoded as Inst{11-8}.
    // See, for example, A8.6.316 VLD4 (multiple 4-element structures).

    // Inst{5-4} encodes alignment.
    unsigned align = slice(insn, 5, 4);
    switch (align) {
    default:
      break;
    case 1:
      alignment = 64; break;
    case 2:
      alignment = 128; break;
    case 3:
      alignment = 256; break;
    }

    unsigned type = slice(insn, 11, 8);
    // Reject UNDEFINED instructions based on type and align.
    // Plus set DblSpaced flag where appropriate.
    switch (elem) {
    default:
      break;
    case 1:
      // n == 1
      // A8.6.307 & A8.6.391
      if ((type == 7  && slice(align, 1, 1) == 1) ||
          (type == 10 && align == 3) ||
          (type == 6  && slice(align, 1, 1) == 1))
        return false;
      break;
    case 2:
      // n == 2 && type == 0b1001 -> DblSpaced = true
      // A8.6.310 & A8.6.393
      if ((type == 8 || type == 9) && align == 3)
        return false;
      DblSpaced = (type == 9);
      break;
    case 3:
      // n == 3 && type == 0b0101 -> DblSpaced = true
      // A8.6.313 & A8.6.395
      if (slice(insn, 7, 6) == 3 || slice(align, 1, 1) == 1)
        return false;
      DblSpaced = (type == 5);
      break;
    case 4:
      // n == 4 && type == 0b0001 -> DblSpaced = true
      // A8.6.316 & A8.6.397
      if (slice(insn, 7, 6) == 3)
        return false;
      DblSpaced = (type == 1);
      break;
    }
  }
  return DisassembleNLdSt0(MI, Opcode, insn, NumOps, NumOpsAdded,
                           slice(insn, 21, 21) == 0, DblSpaced, alignment/8, B);
}

// VMOV (immediate)
//   Qd/Dd imm
// VBIC (immediate)
// VORR (immediate)
//   Qd/Dd imm src(=Qd/Dd)
static bool DisassembleN1RegModImmFrm(MCInst &MI, unsigned Opcode,
    uint32_t insn, unsigned short NumOps, unsigned &NumOpsAdded, BO B) {

  const MCInstrDesc &MCID = ARMInsts[Opcode];
  const MCOperandInfo *OpInfo = MCID.OpInfo;

  assert(NumOps >= 2 &&
         (OpInfo[0].RegClass == ARM::DPRRegClassID ||
          OpInfo[0].RegClass == ARM::QPRRegClassID) &&
         (OpInfo[1].RegClass < 0) &&
         "Expect 1 reg operand followed by 1 imm operand");

  // Qd/Dd = Inst{22:15-12} => NEON Rd
  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, OpInfo[0].RegClass,
                                                     decodeNEONRd(insn))));

  ElemSize esize = ESizeNA;
  switch (Opcode) {
  case ARM::VMOVv8i8:
  case ARM::VMOVv16i8:
    esize = ESize8;
    break;
  case ARM::VMOVv4i16:
  case ARM::VMOVv8i16:
  case ARM::VMVNv4i16:
  case ARM::VMVNv8i16:
  case ARM::VBICiv4i16:
  case ARM::VBICiv8i16:
  case ARM::VORRiv4i16:
  case ARM::VORRiv8i16:
    esize = ESize16;
    break;
  case ARM::VMOVv2i32:
  case ARM::VMOVv4i32:
  case ARM::VMVNv2i32:
  case ARM::VMVNv4i32:
  case ARM::VBICiv2i32:
  case ARM::VBICiv4i32:
  case ARM::VORRiv2i32:
  case ARM::VORRiv4i32:
    esize = ESize32;
    break;
  case ARM::VMOVv1i64:
  case ARM::VMOVv2i64:
    esize = ESize64;
    break;
  default:
    assert(0 && "Unexpected opcode!");
    return false;
  }

  // One register and a modified immediate value.
  // Add the imm operand.
  MI.addOperand(MCOperand::CreateImm(decodeN1VImm(insn, esize)));

  NumOpsAdded = 2;

  // VBIC/VORRiv*i* variants have an extra $src = $Vd to be filled in.
  if (NumOps >= 3 &&
      (OpInfo[2].RegClass == ARM::DPRRegClassID ||
       OpInfo[2].RegClass == ARM::QPRRegClassID)) {
    MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, OpInfo[0].RegClass,
                                                     decodeNEONRd(insn))));
    NumOpsAdded += 1;
  }

  return true;
}

namespace {
enum N2VFlag {
  N2V_None,
  N2V_VectorDupLane,
  N2V_VectorConvert_Between_Float_Fixed
};
} // End of unnamed namespace

// Vector Convert [between floating-point and fixed-point]
//   Qd/Dd Qm/Dm [fbits]
//
// Vector Duplicate Lane (from scalar to all elements) Instructions.
// VDUPLN16d, VDUPLN16q, VDUPLN32d, VDUPLN32q, VDUPLN8d, VDUPLN8q:
//   Qd/Dd Dm index
//
// Vector Move Long:
//   Qd Dm
//
// Vector Move Narrow:
//   Dd Qm
//
// Others
static bool DisassembleNVdVmOptImm(MCInst &MI, unsigned Opc, uint32_t insn,
    unsigned short NumOps, unsigned &NumOpsAdded, N2VFlag Flag, BO B) {

  const MCInstrDesc &MCID = ARMInsts[Opc];
  const MCOperandInfo *OpInfo = MCID.OpInfo;

  assert(NumOps >= 2 &&
         (OpInfo[0].RegClass == ARM::DPRRegClassID ||
          OpInfo[0].RegClass == ARM::QPRRegClassID) &&
         (OpInfo[1].RegClass == ARM::DPRRegClassID ||
          OpInfo[1].RegClass == ARM::QPRRegClassID) &&
         "Expect >= 2 operands and first 2 as reg operands");

  unsigned &OpIdx = NumOpsAdded;

  OpIdx = 0;

  ElemSize esize = ESizeNA;
  if (Flag == N2V_VectorDupLane) {
    // VDUPLN has its index embedded.  Its size can be inferred from the Opcode.
    assert(Opc >= ARM::VDUPLN16d && Opc <= ARM::VDUPLN8q &&
           "Unexpected Opcode");
    esize = (Opc == ARM::VDUPLN8d || Opc == ARM::VDUPLN8q) ? ESize8
       : ((Opc == ARM::VDUPLN16d || Opc == ARM::VDUPLN16q) ? ESize16
                                                           : ESize32);
  }

  // Qd/Dd = Inst{22:15-12} => NEON Rd
  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, OpInfo[OpIdx].RegClass,
                                                     decodeNEONRd(insn))));
  ++OpIdx;

  // VPADAL...
  if (MCID.getOperandConstraint(OpIdx, MCOI::TIED_TO) != -1) {
    // TIED_TO operand.
    MI.addOperand(MCOperand::CreateReg(0));
    ++OpIdx;
  }

  // Dm = Inst{5:3-0} => NEON Rm
  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, OpInfo[OpIdx].RegClass,
                                                     decodeNEONRm(insn))));
  ++OpIdx;

  // VZIP and others have two TIED_TO reg operands.
  int Idx;
  while (OpIdx < NumOps &&
         (Idx = MCID.getOperandConstraint(OpIdx, MCOI::TIED_TO)) != -1) {
    // Add TIED_TO operand.
    MI.addOperand(MI.getOperand(Idx));
    ++OpIdx;
  }

  // Add the imm operand, if required.
  if (OpIdx < NumOps && OpInfo[OpIdx].RegClass < 0
      && !OpInfo[OpIdx].isPredicate() && !OpInfo[OpIdx].isOptionalDef()) {

    unsigned imm = 0xFFFFFFFF;

    if (Flag == N2V_VectorDupLane)
      imm = decodeNVLaneDupIndex(insn, esize);
    if (Flag == N2V_VectorConvert_Between_Float_Fixed)
      imm = decodeVCVTFractionBits(insn);

    assert(imm != 0xFFFFFFFF && "Internal error");
    MI.addOperand(MCOperand::CreateImm(imm));
    ++OpIdx;
  }

  return true;
}

static bool DisassembleN2RegFrm(MCInst &MI, unsigned Opc, uint32_t insn,
    unsigned short NumOps, unsigned &NumOpsAdded, BO B) {

  return DisassembleNVdVmOptImm(MI, Opc, insn, NumOps, NumOpsAdded,
                                N2V_None, B);
}
static bool DisassembleNVCVTFrm(MCInst &MI, unsigned Opc, uint32_t insn,
    unsigned short NumOps, unsigned &NumOpsAdded, BO B) {

  return DisassembleNVdVmOptImm(MI, Opc, insn, NumOps, NumOpsAdded,
                                N2V_VectorConvert_Between_Float_Fixed, B);
}
static bool DisassembleNVecDupLnFrm(MCInst &MI, unsigned Opc, uint32_t insn,
    unsigned short NumOps, unsigned &NumOpsAdded, BO B) {

  return DisassembleNVdVmOptImm(MI, Opc, insn, NumOps, NumOpsAdded,
                                N2V_VectorDupLane, B);
}

// Vector Shift [Accumulate] Instructions.
// Qd/Dd [Qd/Dd (TIED_TO)] Qm/Dm ShiftAmt
//
// Vector Shift Left Long (with maximum shift count) Instructions.
// VSHLLi16, VSHLLi32, VSHLLi8: Qd Dm imm (== size)
//
static bool DisassembleNVectorShift(MCInst &MI, unsigned Opcode, uint32_t insn,
    unsigned short NumOps, unsigned &NumOpsAdded, bool LeftShift, BO B) {

  const MCInstrDesc &MCID = ARMInsts[Opcode];
  const MCOperandInfo *OpInfo = MCID.OpInfo;

  assert(NumOps >= 3 &&
         (OpInfo[0].RegClass == ARM::DPRRegClassID ||
          OpInfo[0].RegClass == ARM::QPRRegClassID) &&
         (OpInfo[1].RegClass == ARM::DPRRegClassID ||
          OpInfo[1].RegClass == ARM::QPRRegClassID) &&
         "Expect >= 3 operands and first 2 as reg operands");

  unsigned &OpIdx = NumOpsAdded;

  OpIdx = 0;

  // Qd/Dd = Inst{22:15-12} => NEON Rd
  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, OpInfo[OpIdx].RegClass,
                                                     decodeNEONRd(insn))));
  ++OpIdx;

  if (MCID.getOperandConstraint(OpIdx, MCOI::TIED_TO) != -1) {
    // TIED_TO operand.
    MI.addOperand(MCOperand::CreateReg(0));
    ++OpIdx;
  }

  assert((OpInfo[OpIdx].RegClass == ARM::DPRRegClassID ||
          OpInfo[OpIdx].RegClass == ARM::QPRRegClassID) &&
         "Reg operand expected");

  // Qm/Dm = Inst{5:3-0} => NEON Rm
  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, OpInfo[OpIdx].RegClass,
                                                     decodeNEONRm(insn))));
  ++OpIdx;

  assert(OpInfo[OpIdx].RegClass < 0 && "Imm operand expected");

  // Add the imm operand.

  // VSHLL has maximum shift count as the imm, inferred from its size.
  unsigned Imm;
  switch (Opcode) {
  default:
    Imm = decodeNVSAmt(insn, LeftShift);
    break;
  case ARM::VSHLLi8:
    Imm = 8;
    break;
  case ARM::VSHLLi16:
    Imm = 16;
    break;
  case ARM::VSHLLi32:
    Imm = 32;
    break;
  }
  MI.addOperand(MCOperand::CreateImm(Imm));
  ++OpIdx;

  return true;
}

// Left shift instructions.
static bool DisassembleN2RegVecShLFrm(MCInst &MI, unsigned Opcode,
    uint32_t insn, unsigned short NumOps, unsigned &NumOpsAdded, BO B) {

  return DisassembleNVectorShift(MI, Opcode, insn, NumOps, NumOpsAdded, true,
                                 B);
}
// Right shift instructions have different shift amount interpretation.
static bool DisassembleN2RegVecShRFrm(MCInst &MI, unsigned Opcode,
    uint32_t insn, unsigned short NumOps, unsigned &NumOpsAdded, BO B) {

  return DisassembleNVectorShift(MI, Opcode, insn, NumOps, NumOpsAdded, false,
                                 B);
}

namespace {
enum N3VFlag {
  N3V_None,
  N3V_VectorExtract,
  N3V_VectorShift,
  N3V_Multiply_By_Scalar
};
} // End of unnamed namespace

// NEON Three Register Instructions with Optional Immediate Operand
//
// Vector Extract Instructions.
// Qd/Dd Qn/Dn Qm/Dm imm4
//
// Vector Shift (Register) Instructions.
// Qd/Dd Qm/Dm Qn/Dn (notice the order of m, n)
//
// Vector Multiply [Accumulate/Subtract] [Long] By Scalar Instructions.
// Qd/Dd Qn/Dn RestrictedDm index
//
// Others
static bool DisassembleNVdVnVmOptImm(MCInst &MI, unsigned Opcode, uint32_t insn,
    unsigned short NumOps, unsigned &NumOpsAdded, N3VFlag Flag, BO B) {

  const MCInstrDesc &MCID = ARMInsts[Opcode];
  const MCOperandInfo *OpInfo = MCID.OpInfo;

  // No checking for OpInfo[2] because of MOVDneon/MOVQ with only two regs.
  assert(NumOps >= 3 &&
         (OpInfo[0].RegClass == ARM::DPRRegClassID ||
          OpInfo[0].RegClass == ARM::QPRRegClassID) &&
         (OpInfo[1].RegClass == ARM::DPRRegClassID ||
          OpInfo[1].RegClass == ARM::QPRRegClassID) &&
         "Expect >= 3 operands and first 2 as reg operands");

  unsigned &OpIdx = NumOpsAdded;

  OpIdx = 0;

  bool VdVnVm = Flag == N3V_VectorShift ? false : true;
  bool IsImm4 = Flag == N3V_VectorExtract ? true : false;
  bool IsDmRestricted = Flag == N3V_Multiply_By_Scalar ? true : false;
  ElemSize esize = ESizeNA;
  if (Flag == N3V_Multiply_By_Scalar) {
    unsigned size = (insn >> 20) & 3;
    if (size == 1) esize = ESize16;
    if (size == 2) esize = ESize32;
    assert (esize == ESize16 || esize == ESize32);
  }

  // Qd/Dd = Inst{22:15-12} => NEON Rd
  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, OpInfo[OpIdx].RegClass,
                                                     decodeNEONRd(insn))));
  ++OpIdx;

  // VABA, VABAL, VBSLd, VBSLq, ...
  if (MCID.getOperandConstraint(OpIdx, MCOI::TIED_TO) != -1) {
    // TIED_TO operand.
    MI.addOperand(MCOperand::CreateReg(0));
    ++OpIdx;
  }

  // Dn = Inst{7:19-16} => NEON Rn
  // or
  // Dm = Inst{5:3-0} => NEON Rm
  MI.addOperand(MCOperand::CreateReg(
                  getRegisterEnum(B, OpInfo[OpIdx].RegClass,
                                  VdVnVm ? decodeNEONRn(insn)
                                         : decodeNEONRm(insn))));
  ++OpIdx;

  // Dm = Inst{5:3-0} => NEON Rm
  // or
  // Dm is restricted to D0-D7 if size is 16, D0-D15 otherwise
  // or
  // Dn = Inst{7:19-16} => NEON Rn
  unsigned m = VdVnVm ? (IsDmRestricted ? decodeRestrictedDm(insn, esize)
                                        : decodeNEONRm(insn))
                      : decodeNEONRn(insn);

  MI.addOperand(MCOperand::CreateReg(
                  getRegisterEnum(B, OpInfo[OpIdx].RegClass, m)));
  ++OpIdx;

  if (OpIdx < NumOps && OpInfo[OpIdx].RegClass < 0
      && !OpInfo[OpIdx].isPredicate() && !OpInfo[OpIdx].isOptionalDef()) {
    // Add the imm operand.
    unsigned Imm = 0;
    if (IsImm4)
      Imm = decodeN3VImm(insn);
    else if (IsDmRestricted)
      Imm = decodeRestrictedDmIndex(insn, esize);
    else {
      assert(0 && "Internal error: unreachable code!");
      return false;
    }

    MI.addOperand(MCOperand::CreateImm(Imm));
    ++OpIdx;
  }

  return true;
}

static bool DisassembleN3RegFrm(MCInst &MI, unsigned Opcode, uint32_t insn,
    unsigned short NumOps, unsigned &NumOpsAdded, BO B) {

  return DisassembleNVdVnVmOptImm(MI, Opcode, insn, NumOps, NumOpsAdded,
                                  N3V_None, B);
}
static bool DisassembleN3RegVecShFrm(MCInst &MI, unsigned Opcode,
    uint32_t insn, unsigned short NumOps, unsigned &NumOpsAdded, BO B) {

  return DisassembleNVdVnVmOptImm(MI, Opcode, insn, NumOps, NumOpsAdded,
                                  N3V_VectorShift, B);
}
static bool DisassembleNVecExtractFrm(MCInst &MI, unsigned Opcode,
    uint32_t insn, unsigned short NumOps, unsigned &NumOpsAdded, BO B) {

  return DisassembleNVdVnVmOptImm(MI, Opcode, insn, NumOps, NumOpsAdded,
                                  N3V_VectorExtract, B);
}
static bool DisassembleNVecMulScalarFrm(MCInst &MI, unsigned Opcode,
    uint32_t insn, unsigned short NumOps, unsigned &NumOpsAdded, BO B) {

  return DisassembleNVdVnVmOptImm(MI, Opcode, insn, NumOps, NumOpsAdded,
                                  N3V_Multiply_By_Scalar, B);
}

// Vector Table Lookup
//
// VTBL1, VTBX1: Dd [Dd(TIED_TO)] Dn Dm
// VTBL2, VTBX2: Dd [Dd(TIED_TO)] Dn Dn+1 Dm
// VTBL3, VTBX3: Dd [Dd(TIED_TO)] Dn Dn+1 Dn+2 Dm
// VTBL4, VTBX4: Dd [Dd(TIED_TO)] Dn Dn+1 Dn+2 Dn+3 Dm
static bool DisassembleNVTBLFrm(MCInst &MI, unsigned Opcode, uint32_t insn,
    unsigned short NumOps, unsigned &NumOpsAdded, BO B) {

  const MCInstrDesc &MCID = ARMInsts[Opcode];
  const MCOperandInfo *OpInfo = MCID.OpInfo;
  if (!OpInfo) return false;

  assert(NumOps >= 3 &&
         OpInfo[0].RegClass == ARM::DPRRegClassID &&
         OpInfo[1].RegClass == ARM::DPRRegClassID &&
         OpInfo[2].RegClass == ARM::DPRRegClassID &&
         "Expect >= 3 operands and first 3 as reg operands");

  unsigned &OpIdx = NumOpsAdded;

  OpIdx = 0;

  unsigned Rn = decodeNEONRn(insn);

  // {Dn} encoded as len = 0b00
  // {Dn Dn+1} encoded as len = 0b01
  // {Dn Dn+1 Dn+2 } encoded as len = 0b10
  // {Dn Dn+1 Dn+2 Dn+3} encoded as len = 0b11
  unsigned Len = slice(insn, 9, 8) + 1;

  // Dd (the destination vector)
  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::DPRRegClassID,
                                                     decodeNEONRd(insn))));
  ++OpIdx;

  // Process tied_to operand constraint.
  int Idx;
  if ((Idx = MCID.getOperandConstraint(OpIdx, MCOI::TIED_TO)) != -1) {
    MI.addOperand(MI.getOperand(Idx));
    ++OpIdx;
  }

  // Do the <list> now.
  for (unsigned i = 0; i < Len; ++i) {
    assert(OpIdx < NumOps && OpInfo[OpIdx].RegClass == ARM::DPRRegClassID &&
           "Reg operand expected");
    MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::DPRRegClassID,
                                                       Rn + i)));
    ++OpIdx;
  }

  // Dm (the index vector)
  assert(OpIdx < NumOps && OpInfo[OpIdx].RegClass == ARM::DPRRegClassID &&
         "Reg operand (index vector) expected");
  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::DPRRegClassID,
                                                     decodeNEONRm(insn))));
  ++OpIdx;

  return true;
}

// Vector Get Lane (move scalar to ARM core register) Instructions.
// VGETLNi32, VGETLNs16, VGETLNs8, VGETLNu16, VGETLNu8: Rt Dn index
static bool DisassembleNGetLnFrm(MCInst &MI, unsigned Opcode, uint32_t insn,
    unsigned short NumOps, unsigned &NumOpsAdded, BO B) {

  const MCInstrDesc &MCID = ARMInsts[Opcode];
  const MCOperandInfo *OpInfo = MCID.OpInfo;
  if (!OpInfo) return false;

  assert(MCID.getNumDefs() == 1 && NumOps >= 3 &&
         OpInfo[0].RegClass == ARM::GPRRegClassID &&
         OpInfo[1].RegClass == ARM::DPRRegClassID &&
         OpInfo[2].RegClass < 0 &&
         "Expect >= 3 operands with one dst operand");

  ElemSize esize =
    Opcode == ARM::VGETLNi32 ? ESize32
      : ((Opcode == ARM::VGETLNs16 || Opcode == ARM::VGETLNu16) ? ESize16
                                                                : ESize8);

  // Rt = Inst{15-12} => ARM Rd
  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                     decodeRd(insn))));

  // Dn = Inst{7:19-16} => NEON Rn
  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::DPRRegClassID,
                                                     decodeNEONRn(insn))));

  MI.addOperand(MCOperand::CreateImm(decodeNVLaneOpIndex(insn, esize)));

  NumOpsAdded = 3;
  return true;
}

// Vector Set Lane (move ARM core register to scalar) Instructions.
// VSETLNi16, VSETLNi32, VSETLNi8: Dd Dd (TIED_TO) Rt index
static bool DisassembleNSetLnFrm(MCInst &MI, unsigned Opcode, uint32_t insn,
    unsigned short NumOps, unsigned &NumOpsAdded, BO B) {

  const MCInstrDesc &MCID = ARMInsts[Opcode];
  const MCOperandInfo *OpInfo = MCID.OpInfo;
  if (!OpInfo) return false;

  assert(MCID.getNumDefs() == 1 && NumOps >= 3 &&
         OpInfo[0].RegClass == ARM::DPRRegClassID &&
         OpInfo[1].RegClass == ARM::DPRRegClassID &&
         MCID.getOperandConstraint(1, MCOI::TIED_TO) != -1 &&
         OpInfo[2].RegClass == ARM::GPRRegClassID &&
         OpInfo[3].RegClass < 0 &&
         "Expect >= 3 operands with one dst operand");

  ElemSize esize =
    Opcode == ARM::VSETLNi8 ? ESize8
                            : (Opcode == ARM::VSETLNi16 ? ESize16
                                                        : ESize32);

  // Dd = Inst{7:19-16} => NEON Rn
  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::DPRRegClassID,
                                                     decodeNEONRn(insn))));

  // TIED_TO operand.
  MI.addOperand(MCOperand::CreateReg(0));

  // Rt = Inst{15-12} => ARM Rd
  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                     decodeRd(insn))));

  MI.addOperand(MCOperand::CreateImm(decodeNVLaneOpIndex(insn, esize)));

  NumOpsAdded = 4;
  return true;
}

// Vector Duplicate Instructions (from ARM core register to all elements).
// VDUP8d, VDUP16d, VDUP32d, VDUP8q, VDUP16q, VDUP32q: Qd/Dd Rt
static bool DisassembleNDupFrm(MCInst &MI, unsigned Opcode, uint32_t insn,
    unsigned short NumOps, unsigned &NumOpsAdded, BO B) {

  const MCOperandInfo *OpInfo = ARMInsts[Opcode].OpInfo;

  assert(NumOps >= 2 &&
         (OpInfo[0].RegClass == ARM::DPRRegClassID ||
          OpInfo[0].RegClass == ARM::QPRRegClassID) &&
         OpInfo[1].RegClass == ARM::GPRRegClassID &&
         "Expect >= 2 operands and first 2 as reg operand");

  unsigned RegClass = OpInfo[0].RegClass;

  // Qd/Dd = Inst{7:19-16} => NEON Rn
  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, RegClass,
                                                     decodeNEONRn(insn))));

  // Rt = Inst{15-12} => ARM Rd
  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                     decodeRd(insn))));

  NumOpsAdded = 2;
  return true;
}

static inline bool PreLoadOpcode(unsigned Opcode) {
  switch(Opcode) {
  case ARM::PLDi12:  case ARM::PLDrs:
  case ARM::PLDWi12: case ARM::PLDWrs:
  case ARM::PLIi12:  case ARM::PLIrs:
    return true;
  default:
    return false;
  }
}

static bool DisassemblePreLoadFrm(MCInst &MI, unsigned Opcode, uint32_t insn,
    unsigned short NumOps, unsigned &NumOpsAdded, BO B) {

  // Preload Data/Instruction requires either 2 or 3 operands.
  // PLDi12, PLDWi12, PLIi12: addrmode_imm12
  // PLDrs, PLDWrs, PLIrs:    ldst_so_reg

  MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                     decodeRn(insn))));

  if (Opcode == ARM::PLDi12 || Opcode == ARM::PLDWi12
      || Opcode == ARM::PLIi12) {
    unsigned Imm12 = slice(insn, 11, 0);
    bool Negative = getUBit(insn) == 0;

    // A8.6.118 PLD (literal) PLDWi12 with Rn=PC is transformed to PLDi12.
    if (Opcode == ARM::PLDWi12 && slice(insn, 19, 16) == 0xF) {
      DEBUG(errs() << "Rn == '1111': PLDWi12 morphed to PLDi12\n");
      MI.setOpcode(ARM::PLDi12);
    }
    
    // -0 is represented specially. All other values are as normal.
    int Offset = Negative ? -1 * Imm12 : Imm12;
    if (Imm12 == 0 && Negative)
      Offset = INT32_MIN;

    MI.addOperand(MCOperand::CreateImm(Offset));
    NumOpsAdded = 2;
  } else {
    MI.addOperand(MCOperand::CreateReg(getRegisterEnum(B, ARM::GPRRegClassID,
                                                       decodeRm(insn))));

    ARM_AM::AddrOpc AddrOpcode = getUBit(insn) ? ARM_AM::add : ARM_AM::sub;

    // Inst{6-5} encodes the shift opcode.
    ARM_AM::ShiftOpc ShOp = getShiftOpcForBits(slice(insn, 6, 5));
    // Inst{11-7} encodes the imm5 shift amount.
    unsigned ShImm = slice(insn, 11, 7);

    // A8.4.1.  Possible rrx or shift amount of 32...
    getImmShiftSE(ShOp, ShImm);
    MI.addOperand(MCOperand::CreateImm(
                    ARM_AM::getAM2Opc(AddrOpcode, ShImm, ShOp)));
    NumOpsAdded = 3;
  }

  return true;
}

static bool DisassembleMiscFrm(MCInst &MI, unsigned Opcode, uint32_t insn,
    unsigned short NumOps, unsigned &NumOpsAdded, BO B) {

  if (Opcode == ARM::DMB || Opcode == ARM::DSB || Opcode == ARM::ISB) {
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

  switch (Opcode) {
  case ARM::CLREX:
  case ARM::NOP:
  case ARM::TRAP:
  case ARM::YIELD:
  case ARM::WFE:
  case ARM::WFI:
  case ARM::SEV:
    return true;
  case ARM::SWP:
  case ARM::SWPB:
    // SWP, SWPB: Rd Rm Rn
    // Delegate to DisassembleLdStExFrm()....
    return DisassembleLdStExFrm(MI, Opcode, insn, NumOps, NumOpsAdded, B);
  default:
    break;
  }

  if (Opcode == ARM::SETEND) {
    NumOpsAdded = 1;
    MI.addOperand(MCOperand::CreateImm(slice(insn, 9, 9)));
    return true;
  }

  // FIXME: To enable correct asm parsing and disasm of CPS we need 3 different
  // opcodes which match the same real instruction. This is needed since there's
  // no current handling of optional arguments. Fix here when a better handling
  // of optional arguments is implemented.
  if (Opcode == ARM::CPS3p) {   // M = 1
    // Let's reject these impossible imod values by returning false:
    // 1. (imod=0b01)
    //
    // AsmPrinter cannot handle imod=0b00, plus (imod=0b00,M=1,iflags!=0) is an
    // invalid combination, so we just check for imod=0b00 here.
    if (slice(insn, 19, 18) == 0 || slice(insn, 19, 18) == 1)
      return false;
    MI.addOperand(MCOperand::CreateImm(slice(insn, 19, 18))); // imod
    MI.addOperand(MCOperand::CreateImm(slice(insn, 8, 6)));   // iflags
    MI.addOperand(MCOperand::CreateImm(slice(insn, 4, 0)));   // mode
    NumOpsAdded = 3;
    return true;
  }
  if (Opcode == ARM::CPS2p) { // mode = 0, M = 0
    // Let's reject these impossible imod values by returning false:
    // 1. (imod=0b00,M=0)
    // 2. (imod=0b01)
    if (slice(insn, 19, 18) == 0 || slice(insn, 19, 18) == 1)
      return false;
    MI.addOperand(MCOperand::CreateImm(slice(insn, 19, 18))); // imod
    MI.addOperand(MCOperand::CreateImm(slice(insn, 8, 6)));   // iflags
    NumOpsAdded = 2;
    return true;
  }
  if (Opcode == ARM::CPS1p) { // imod = 0, iflags = 0, M = 1
    MI.addOperand(MCOperand::CreateImm(slice(insn, 4, 0))); // mode
    NumOpsAdded = 1;
    return true;
  }

  // DBG has its option specified in Inst{3-0}.
  if (Opcode == ARM::DBG) {
    MI.addOperand(MCOperand::CreateImm(slice(insn, 3, 0)));
    NumOpsAdded = 1;
    return true;
  }

  // BKPT takes an imm32 val equal to ZeroExtend(Inst{19-8:3-0}).
  if (Opcode == ARM::BKPT) {
    MI.addOperand(MCOperand::CreateImm(slice(insn, 19, 8) << 4 |
                                       slice(insn, 3, 0)));
    NumOpsAdded = 1;
    return true;
  }

  if (PreLoadOpcode(Opcode))
    return DisassemblePreLoadFrm(MI, Opcode, insn, NumOps, NumOpsAdded, B);

  assert(0 && "Unexpected misc instruction!");
  return false;
}

/// FuncPtrs - FuncPtrs maps ARMFormat to its corresponding DisassembleFP.
/// We divide the disassembly task into different categories, with each one
/// corresponding to a specific instruction encoding format.  There could be
/// exceptions when handling a specific format, and that is why the Opcode is
/// also present in the function prototype.
static const DisassembleFP FuncPtrs[] = {
  &DisassemblePseudo,
  &DisassembleMulFrm,
  &DisassembleBrFrm,
  &DisassembleBrMiscFrm,
  &DisassembleDPFrm,
  &DisassembleDPSoRegRegFrm,
  &DisassembleLdFrm,
  &DisassembleStFrm,
  &DisassembleLdMiscFrm,
  &DisassembleStMiscFrm,
  &DisassembleLdStMulFrm,
  &DisassembleLdStExFrm,
  &DisassembleArithMiscFrm,
  &DisassembleSatFrm,
  &DisassembleExtFrm,
  &DisassembleVFPUnaryFrm,
  &DisassembleVFPBinaryFrm,
  &DisassembleVFPConv1Frm,
  &DisassembleVFPConv2Frm,
  &DisassembleVFPConv3Frm,
  &DisassembleVFPConv4Frm,
  &DisassembleVFPConv5Frm,
  &DisassembleVFPLdStFrm,
  &DisassembleVFPLdStMulFrm,
  &DisassembleVFPMiscFrm,
  &DisassembleThumbFrm,
  &DisassembleMiscFrm,
  &DisassembleNGetLnFrm,
  &DisassembleNSetLnFrm,
  &DisassembleNDupFrm,

  // VLD and VST (including one lane) Instructions.
  &DisassembleNLdSt,

  // A7.4.6 One register and a modified immediate value
  // 1-Register Instructions with imm.
  // LLVM only defines VMOVv instructions.
  &DisassembleN1RegModImmFrm,

  // 2-Register Instructions with no imm.
  &DisassembleN2RegFrm,

  // 2-Register Instructions with imm (vector convert float/fixed point).
  &DisassembleNVCVTFrm,

  // 2-Register Instructions with imm (vector dup lane).
  &DisassembleNVecDupLnFrm,

  // Vector Shift Left Instructions.
  &DisassembleN2RegVecShLFrm,

  // Vector Shift Righ Instructions, which has different interpretation of the
  // shift amount from the imm6 field.
  &DisassembleN2RegVecShRFrm,

  // 3-Register Data-Processing Instructions.
  &DisassembleN3RegFrm,

  // Vector Shift (Register) Instructions.
  // D:Vd M:Vm N:Vn (notice that M:Vm is the first operand)
  &DisassembleN3RegVecShFrm,

  // Vector Extract Instructions.
  &DisassembleNVecExtractFrm,

  // Vector [Saturating Rounding Doubling] Multiply [Accumulate/Subtract] [Long]
  // By Scalar Instructions.
  &DisassembleNVecMulScalarFrm,

  // Vector Table Lookup uses byte indexes in a control vector to look up byte
  // values in a table and generate a new vector.
  &DisassembleNVTBLFrm,

  &DisassembleDPSoRegImmFrm,


  NULL
};

/// BuildIt - BuildIt performs the build step for this ARM Basic MC Builder.
/// The general idea is to set the Opcode for the MCInst, followed by adding
/// the appropriate MCOperands to the MCInst.  ARM Basic MC Builder delegates
/// to the Format-specific disassemble function for disassembly, followed by
/// TryPredicateAndSBitModifier() to do PredicateOperand and OptionalDefOperand
/// which follow the Dst/Src Operands.
bool ARMBasicMCBuilder::BuildIt(MCInst &MI, uint32_t insn) {
  // Stage 1 sets the Opcode.
  MI.setOpcode(Opcode);
  // If the number of operands is zero, we're done!
  if (NumOps == 0)
    return true;

  // Stage 2 calls the format-specific disassemble function to build the operand
  // list.
  if (Disasm == NULL)
    return false;
  unsigned NumOpsAdded = 0;
  bool OK = (*Disasm)(MI, Opcode, insn, NumOps, NumOpsAdded, this);

  if (!OK || this->Err != 0) return false;
  if (NumOpsAdded >= NumOps)
    return true;

  // Stage 3 deals with operands unaccounted for after stage 2 is finished.
  // FIXME: Should this be done selectively?
  return TryPredicateAndSBitModifier(MI, Opcode, insn, NumOps - NumOpsAdded);
}

// A8.3 Conditional execution
// A8.3.1 Pseudocode details of conditional execution
// Condition bits '111x' indicate the instruction is always executed.
static uint32_t CondCode(uint32_t CondField) {
  if (CondField == 0xF)
    return ARMCC::AL;
  return CondField;
}

/// DoPredicateOperands - DoPredicateOperands process the predicate operands
/// of some Thumb instructions which come before the reglist operands.  It
/// returns true if the two predicate operands have been processed.
bool ARMBasicMCBuilder::DoPredicateOperands(MCInst& MI, unsigned Opcode,
    uint32_t /* insn */, unsigned short NumOpsRemaining) {

  assert(NumOpsRemaining > 0 && "Invalid argument");

  const MCOperandInfo *OpInfo = ARMInsts[Opcode].OpInfo;
  unsigned Idx = MI.getNumOperands();

  // First, we check whether this instr specifies the PredicateOperand through
  // a pair of MCOperandInfos with isPredicate() property.
  if (NumOpsRemaining >= 2 &&
      OpInfo[Idx].isPredicate() && OpInfo[Idx+1].isPredicate() &&
      OpInfo[Idx].RegClass < 0 &&
      OpInfo[Idx+1].RegClass == ARM::CCRRegClassID)
  {
    // If we are inside an IT block, get the IT condition bits maintained via
    // ARMBasicMCBuilder::ITState[7:0], through ARMBasicMCBuilder::GetITCond().
    // See also A2.5.2.
    if (InITBlock())
      MI.addOperand(MCOperand::CreateImm(GetITCond()));
    else
      MI.addOperand(MCOperand::CreateImm(ARMCC::AL));
    MI.addOperand(MCOperand::CreateReg(ARM::CPSR));
    return true;
  }

  return false;
}

/// TryPredicateAndSBitModifier - TryPredicateAndSBitModifier tries to process
/// the possible Predicate and SBitModifier, to build the remaining MCOperand
/// constituents.
bool ARMBasicMCBuilder::TryPredicateAndSBitModifier(MCInst& MI, unsigned Opcode,
    uint32_t insn, unsigned short NumOpsRemaining) {

  assert(NumOpsRemaining > 0 && "Invalid argument");

  const MCOperandInfo *OpInfo = ARMInsts[Opcode].OpInfo;
  const std::string &Name = ARMInsts[Opcode].Name;
  unsigned Idx = MI.getNumOperands();
  uint64_t TSFlags = ARMInsts[Opcode].TSFlags;

  // First, we check whether this instr specifies the PredicateOperand through
  // a pair of MCOperandInfos with isPredicate() property.
  if (NumOpsRemaining >= 2 &&
      OpInfo[Idx].isPredicate() && OpInfo[Idx+1].isPredicate() &&
      OpInfo[Idx].RegClass < 0 &&
      OpInfo[Idx+1].RegClass == ARM::CCRRegClassID)
  {
    // If we are inside an IT block, get the IT condition bits maintained via
    // ARMBasicMCBuilder::ITState[7:0], through ARMBasicMCBuilder::GetITCond().
    // See also A2.5.2.
    if (InITBlock())
      MI.addOperand(MCOperand::CreateImm(GetITCond()));
    else {
      if (Name.length() > 1 && Name[0] == 't') {
        // Thumb conditional branch instructions have their cond field embedded,
        // like ARM.
        //
        // A8.6.16 B
        // Check for undefined encodings.
        unsigned cond;
        if (Name == "t2Bcc") {
          if ((cond = slice(insn, 25, 22)) >= 14)
            return false;
          MI.addOperand(MCOperand::CreateImm(CondCode(cond)));
        } else if (Name == "tBcc") {
          if ((cond = slice(insn, 11, 8)) == 14)
            return false;
          MI.addOperand(MCOperand::CreateImm(CondCode(cond)));
        } else
          MI.addOperand(MCOperand::CreateImm(ARMCC::AL));
      } else {
        // ARM instructions get their condition field from Inst{31-28}.
        // We should reject Inst{31-28} = 0b1111 as invalid encoding.
        if (!isNEONDomain(TSFlags) && getCondField(insn) == 0xF)
          return false;
        MI.addOperand(MCOperand::CreateImm(CondCode(getCondField(insn))));
      }
    }
    MI.addOperand(MCOperand::CreateReg(ARM::CPSR));
    Idx += 2;
    NumOpsRemaining -= 2;
  }

  if (NumOpsRemaining == 0)
    return true;

  // Next, if OptionalDefOperand exists, we check whether the 'S' bit is set.
  if (OpInfo[Idx].isOptionalDef() && OpInfo[Idx].RegClass==ARM::CCRRegClassID) {
    MI.addOperand(MCOperand::CreateReg(getSBit(insn) == 1 ? ARM::CPSR : 0));
    --NumOpsRemaining;
  }

  if (NumOpsRemaining == 0)
    return true;
  else
    return false;
}

/// RunBuildAfterHook - RunBuildAfterHook performs operations deemed necessary
/// after BuildIt is finished.
bool ARMBasicMCBuilder::RunBuildAfterHook(bool Status, MCInst &MI,
    uint32_t insn) {

  if (!SP) return Status;

  if (Opcode == ARM::t2IT)
    Status = SP->InitIT(slice(insn, 7, 0)) ? Status : false;
  else if (InITBlock())
    SP->UpdateIT();

  return Status;
}

/// Opcode, Format, and NumOperands make up an ARM Basic MCBuilder.
ARMBasicMCBuilder::ARMBasicMCBuilder(unsigned opc, ARMFormat format,
                                     unsigned short num)
  : Opcode(opc), Format(format), NumOps(num), SP(0), Err(0) {
  unsigned Idx = (unsigned)format;
  assert(Idx < (array_lengthof(FuncPtrs) - 1) && "Unknown format");
  Disasm = FuncPtrs[Idx];
}

/// CreateMCBuilder - Return an ARMBasicMCBuilder that can build up the MC
/// infrastructure of an MCInst given the Opcode and Format of the instr.
/// Return NULL if it fails to create/return a proper builder.  API clients
/// are responsible for freeing up of the allocated memory.  Cacheing can be
/// performed by the API clients to improve performance.
ARMBasicMCBuilder *llvm::CreateMCBuilder(unsigned Opcode, ARMFormat Format) {
  // For "Unknown format", fail by returning a NULL pointer.
  if ((unsigned)Format >= (array_lengthof(FuncPtrs) - 1)) {
    DEBUG(errs() << "Unknown format\n");
    return 0;
  }

  return new ARMBasicMCBuilder(Opcode, Format,
                               ARMInsts[Opcode].getNumOperands());
}

/// tryAddingSymbolicOperand - tryAddingSymbolicOperand trys to add a symbolic
/// operand in place of the immediate Value in the MCInst.  The immediate
/// Value has had any PC adjustment made by the caller.  If the getOpInfo()
/// function was set as part of the setupBuilderForSymbolicDisassembly() call
/// then that function is called to get any symbolic information at the
/// builder's Address for this instrution.  If that returns non-zero then the
/// symbolic information it returns is used to create an MCExpr and that is
/// added as an operand to the MCInst.  This function returns true if it adds
/// an operand to the MCInst and false otherwise.
bool ARMBasicMCBuilder::tryAddingSymbolicOperand(uint64_t Value,
                                                 uint64_t InstSize,
                                                 MCInst &MI) {
  if (!GetOpInfo)
    return false;

  struct LLVMOpInfo1 SymbolicOp;
  SymbolicOp.Value = Value;
  if (!GetOpInfo(DisInfo, Address, 0 /* Offset */, InstSize, 1, &SymbolicOp))
    return false;

  const MCExpr *Add = NULL;
  if (SymbolicOp.AddSymbol.Present) {
    if (SymbolicOp.AddSymbol.Name) {
      StringRef Name(SymbolicOp.AddSymbol.Name);
      MCSymbol *Sym = Ctx->GetOrCreateSymbol(Name);
      Add = MCSymbolRefExpr::Create(Sym, *Ctx);
    } else {
      Add = MCConstantExpr::Create(SymbolicOp.AddSymbol.Value, *Ctx);
    }
  }

  const MCExpr *Sub = NULL;
  if (SymbolicOp.SubtractSymbol.Present) {
    if (SymbolicOp.SubtractSymbol.Name) {
      StringRef Name(SymbolicOp.SubtractSymbol.Name);
      MCSymbol *Sym = Ctx->GetOrCreateSymbol(Name);
      Sub = MCSymbolRefExpr::Create(Sym, *Ctx);
    } else {
      Sub = MCConstantExpr::Create(SymbolicOp.SubtractSymbol.Value, *Ctx);
    }
  }

  const MCExpr *Off = NULL;
  if (SymbolicOp.Value != 0)
    Off = MCConstantExpr::Create(SymbolicOp.Value, *Ctx);

  const MCExpr *Expr;
  if (Sub) {
    const MCExpr *LHS;
    if (Add)
      LHS = MCBinaryExpr::CreateSub(Add, Sub, *Ctx);
    else
      LHS = MCUnaryExpr::CreateMinus(Sub, *Ctx);
    if (Off != 0)
      Expr = MCBinaryExpr::CreateAdd(LHS, Off, *Ctx);
    else
      Expr = LHS;
  } else if (Add) {
    if (Off != 0)
      Expr = MCBinaryExpr::CreateAdd(Add, Off, *Ctx);
    else
      Expr = Add;
  } else {
    if (Off != 0)
      Expr = Off;
    else
      Expr = MCConstantExpr::Create(0, *Ctx);
  }

  if (SymbolicOp.VariantKind == LLVMDisassembler_VariantKind_ARM_HI16)
    MI.addOperand(MCOperand::CreateExpr(ARMMCExpr::CreateUpper16(Expr, *Ctx)));
  else if (SymbolicOp.VariantKind == LLVMDisassembler_VariantKind_ARM_LO16)
    MI.addOperand(MCOperand::CreateExpr(ARMMCExpr::CreateLower16(Expr, *Ctx)));
  else if (SymbolicOp.VariantKind == LLVMDisassembler_VariantKind_None)
    MI.addOperand(MCOperand::CreateExpr(Expr));
  else 
    assert("bad SymbolicOp.VariantKind");

  return true;
}
