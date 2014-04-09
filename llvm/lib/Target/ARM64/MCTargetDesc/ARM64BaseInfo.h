//===-- ARM64BaseInfo.h - Top level definitions for ARM64 -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains small standalone helper functions and enum definitions for
// the ARM64 target useful for the compiler back-end and the MC libraries.
// As such, it deliberately does not include references to LLVM core
// code gen types, passes, etc..
//
//===----------------------------------------------------------------------===//

#ifndef ARM64BASEINFO_H
#define ARM64BASEINFO_H

#include "ARM64MCTargetDesc.h"
#include "llvm/Support/ErrorHandling.h"

namespace llvm {

inline static unsigned getWRegFromXReg(unsigned Reg) {
  switch (Reg) {
  case ARM64::X0: return ARM64::W0;
  case ARM64::X1: return ARM64::W1;
  case ARM64::X2: return ARM64::W2;
  case ARM64::X3: return ARM64::W3;
  case ARM64::X4: return ARM64::W4;
  case ARM64::X5: return ARM64::W5;
  case ARM64::X6: return ARM64::W6;
  case ARM64::X7: return ARM64::W7;
  case ARM64::X8: return ARM64::W8;
  case ARM64::X9: return ARM64::W9;
  case ARM64::X10: return ARM64::W10;
  case ARM64::X11: return ARM64::W11;
  case ARM64::X12: return ARM64::W12;
  case ARM64::X13: return ARM64::W13;
  case ARM64::X14: return ARM64::W14;
  case ARM64::X15: return ARM64::W15;
  case ARM64::X16: return ARM64::W16;
  case ARM64::X17: return ARM64::W17;
  case ARM64::X18: return ARM64::W18;
  case ARM64::X19: return ARM64::W19;
  case ARM64::X20: return ARM64::W20;
  case ARM64::X21: return ARM64::W21;
  case ARM64::X22: return ARM64::W22;
  case ARM64::X23: return ARM64::W23;
  case ARM64::X24: return ARM64::W24;
  case ARM64::X25: return ARM64::W25;
  case ARM64::X26: return ARM64::W26;
  case ARM64::X27: return ARM64::W27;
  case ARM64::X28: return ARM64::W28;
  case ARM64::FP: return ARM64::W29;
  case ARM64::LR: return ARM64::W30;
  case ARM64::SP: return ARM64::WSP;
  case ARM64::XZR: return ARM64::WZR;
  }
  // For anything else, return it unchanged.
  return Reg;
}

inline static unsigned getXRegFromWReg(unsigned Reg) {
  switch (Reg) {
  case ARM64::W0: return ARM64::X0;
  case ARM64::W1: return ARM64::X1;
  case ARM64::W2: return ARM64::X2;
  case ARM64::W3: return ARM64::X3;
  case ARM64::W4: return ARM64::X4;
  case ARM64::W5: return ARM64::X5;
  case ARM64::W6: return ARM64::X6;
  case ARM64::W7: return ARM64::X7;
  case ARM64::W8: return ARM64::X8;
  case ARM64::W9: return ARM64::X9;
  case ARM64::W10: return ARM64::X10;
  case ARM64::W11: return ARM64::X11;
  case ARM64::W12: return ARM64::X12;
  case ARM64::W13: return ARM64::X13;
  case ARM64::W14: return ARM64::X14;
  case ARM64::W15: return ARM64::X15;
  case ARM64::W16: return ARM64::X16;
  case ARM64::W17: return ARM64::X17;
  case ARM64::W18: return ARM64::X18;
  case ARM64::W19: return ARM64::X19;
  case ARM64::W20: return ARM64::X20;
  case ARM64::W21: return ARM64::X21;
  case ARM64::W22: return ARM64::X22;
  case ARM64::W23: return ARM64::X23;
  case ARM64::W24: return ARM64::X24;
  case ARM64::W25: return ARM64::X25;
  case ARM64::W26: return ARM64::X26;
  case ARM64::W27: return ARM64::X27;
  case ARM64::W28: return ARM64::X28;
  case ARM64::W29: return ARM64::FP;
  case ARM64::W30: return ARM64::LR;
  case ARM64::WSP: return ARM64::SP;
  case ARM64::WZR: return ARM64::XZR;
  }
  // For anything else, return it unchanged.
  return Reg;
}

static inline unsigned getBRegFromDReg(unsigned Reg) {
  switch (Reg) {
  case ARM64::D0:  return ARM64::B0;
  case ARM64::D1:  return ARM64::B1;
  case ARM64::D2:  return ARM64::B2;
  case ARM64::D3:  return ARM64::B3;
  case ARM64::D4:  return ARM64::B4;
  case ARM64::D5:  return ARM64::B5;
  case ARM64::D6:  return ARM64::B6;
  case ARM64::D7:  return ARM64::B7;
  case ARM64::D8:  return ARM64::B8;
  case ARM64::D9:  return ARM64::B9;
  case ARM64::D10: return ARM64::B10;
  case ARM64::D11: return ARM64::B11;
  case ARM64::D12: return ARM64::B12;
  case ARM64::D13: return ARM64::B13;
  case ARM64::D14: return ARM64::B14;
  case ARM64::D15: return ARM64::B15;
  case ARM64::D16: return ARM64::B16;
  case ARM64::D17: return ARM64::B17;
  case ARM64::D18: return ARM64::B18;
  case ARM64::D19: return ARM64::B19;
  case ARM64::D20: return ARM64::B20;
  case ARM64::D21: return ARM64::B21;
  case ARM64::D22: return ARM64::B22;
  case ARM64::D23: return ARM64::B23;
  case ARM64::D24: return ARM64::B24;
  case ARM64::D25: return ARM64::B25;
  case ARM64::D26: return ARM64::B26;
  case ARM64::D27: return ARM64::B27;
  case ARM64::D28: return ARM64::B28;
  case ARM64::D29: return ARM64::B29;
  case ARM64::D30: return ARM64::B30;
  case ARM64::D31: return ARM64::B31;
  }
  // For anything else, return it unchanged.
  return Reg;
}


static inline unsigned getDRegFromBReg(unsigned Reg) {
  switch (Reg) {
  case ARM64::B0:  return ARM64::D0;
  case ARM64::B1:  return ARM64::D1;
  case ARM64::B2:  return ARM64::D2;
  case ARM64::B3:  return ARM64::D3;
  case ARM64::B4:  return ARM64::D4;
  case ARM64::B5:  return ARM64::D5;
  case ARM64::B6:  return ARM64::D6;
  case ARM64::B7:  return ARM64::D7;
  case ARM64::B8:  return ARM64::D8;
  case ARM64::B9:  return ARM64::D9;
  case ARM64::B10: return ARM64::D10;
  case ARM64::B11: return ARM64::D11;
  case ARM64::B12: return ARM64::D12;
  case ARM64::B13: return ARM64::D13;
  case ARM64::B14: return ARM64::D14;
  case ARM64::B15: return ARM64::D15;
  case ARM64::B16: return ARM64::D16;
  case ARM64::B17: return ARM64::D17;
  case ARM64::B18: return ARM64::D18;
  case ARM64::B19: return ARM64::D19;
  case ARM64::B20: return ARM64::D20;
  case ARM64::B21: return ARM64::D21;
  case ARM64::B22: return ARM64::D22;
  case ARM64::B23: return ARM64::D23;
  case ARM64::B24: return ARM64::D24;
  case ARM64::B25: return ARM64::D25;
  case ARM64::B26: return ARM64::D26;
  case ARM64::B27: return ARM64::D27;
  case ARM64::B28: return ARM64::D28;
  case ARM64::B29: return ARM64::D29;
  case ARM64::B30: return ARM64::D30;
  case ARM64::B31: return ARM64::D31;
  }
  // For anything else, return it unchanged.
  return Reg;
}

namespace ARM64CC {

// The CondCodes constants map directly to the 4-bit encoding of the condition
// field for predicated instructions.
enum CondCode {  // Meaning (integer)          Meaning (floating-point)
  EQ = 0x0,      // Equal                      Equal
  NE = 0x1,      // Not equal                  Not equal, or unordered
  CS = 0x2,      // Carry set                  >, ==, or unordered
  CC = 0x3,      // Carry clear                Less than
  MI = 0x4,      // Minus, negative            Less than
  PL = 0x5,      // Plus, positive or zero     >, ==, or unordered
  VS = 0x6,      // Overflow                   Unordered
  VC = 0x7,      // No overflow                Not unordered
  HI = 0x8,      // Unsigned higher            Greater than, or unordered
  LS = 0x9,      // Unsigned lower or same     Less than or equal
  GE = 0xa,      // Greater than or equal      Greater than or equal
  LT = 0xb,      // Less than                  Less than, or unordered
  GT = 0xc,      // Greater than               Greater than
  LE = 0xd,      // Less than or equal         <, ==, or unordered
  AL = 0xe,      // Always (unconditional)     Always (unconditional)
  NV = 0xf,      // Always (unconditional)     Always (unconditional)
  // Note the NV exists purely to disassemble 0b1111. Execution is "always".
  Invalid
};

inline static const char *getCondCodeName(CondCode Code) {
  switch (Code) {
  default: llvm_unreachable("Unknown condition code");
  case EQ:  return "eq";
  case NE:  return "ne";
  case CS:  return "cs";
  case CC:  return "cc";
  case MI:  return "mi";
  case PL:  return "pl";
  case VS:  return "vs";
  case VC:  return "vc";
  case HI:  return "hi";
  case LS:  return "ls";
  case GE:  return "ge";
  case LT:  return "lt";
  case GT:  return "gt";
  case LE:  return "le";
  case AL:  return "al";
  case NV:  return "nv";
  }
}

inline static CondCode getInvertedCondCode(CondCode Code) {
  switch (Code) {
  default: llvm_unreachable("Unknown condition code");
  case EQ:  return NE;
  case NE:  return EQ;
  case CS:  return CC;
  case CC:  return CS;
  case MI:  return PL;
  case PL:  return MI;
  case VS:  return VC;
  case VC:  return VS;
  case HI:  return LS;
  case LS:  return HI;
  case GE:  return LT;
  case LT:  return GE;
  case GT:  return LE;
  case LE:  return GT;
  }
}

/// Given a condition code, return NZCV flags that would satisfy that condition.
/// The flag bits are in the format expected by the ccmp instructions.
/// Note that many different flag settings can satisfy a given condition code,
/// this function just returns one of them.
inline static unsigned getNZCVToSatisfyCondCode(CondCode Code) {
  // NZCV flags encoded as expected by ccmp instructions, ARMv8 ISA 5.5.7.
  enum { N = 8, Z = 4, C = 2, V = 1 };
  switch (Code) {
  default: llvm_unreachable("Unknown condition code");
  case EQ: return Z; // Z == 1
  case NE: return 0; // Z == 0
  case CS: return C; // C == 1
  case CC: return 0; // C == 0
  case MI: return N; // N == 1
  case PL: return 0; // N == 0
  case VS: return V; // V == 1
  case VC: return 0; // V == 0
  case HI: return C; // C == 1 && Z == 0
  case LS: return 0; // C == 0 || Z == 1
  case GE: return 0; // N == V
  case LT: return N; // N != V
  case GT: return 0; // Z == 0 && N == V
  case LE: return Z; // Z == 1 || N != V
  }
}
} // end namespace ARM64CC

namespace ARM64SYS {
enum BarrierOption {
  InvalidBarrier = 0xff,
  OSHLD = 0x1,
  OSHST = 0x2,
  OSH =   0x3,
  NSHLD = 0x5,
  NSHST = 0x6,
  NSH =   0x7,
  ISHLD = 0x9,
  ISHST = 0xa,
  ISH =   0xb,
  LD =    0xd,
  ST =    0xe,
  SY =    0xf
};

inline static const char *getBarrierOptName(BarrierOption Opt) {
  switch (Opt) {
  default: return NULL;
  case 0x1: return "oshld";
  case 0x2: return "oshst";
  case 0x3: return "osh";
  case 0x5: return "nshld";
  case 0x6: return "nshst";
  case 0x7: return "nsh";
  case 0x9: return "ishld";
  case 0xa: return "ishst";
  case 0xb: return "ish";
  case 0xd: return "ld";
  case 0xe: return "st";
  case 0xf: return "sy";
  }
}

#define A64_SYSREG_ENC(op0,CRn,op2,CRm,op1) ((op0) << 14 | (op1) << 11 | \
                                             (CRn) << 7  | (CRm) << 3 | (op2))
enum SystemRegister {
  InvalidSystemReg = 0,
  // Table in section 3.10.3
  SPSR_EL1  = 0xc200,
  SPSR_svc  = SPSR_EL1,
  ELR_EL1   = 0xc201,
  SP_EL0    = 0xc208,
  SPSel     = 0xc210,
  CurrentEL = 0xc212,
  DAIF      = 0xda11,
  NZCV      = 0xda10,
  FPCR      = 0xda20,
  FPSR      = 0xda21,
  DSPSR     = 0xda28,
  DLR       = 0xda29,
  SPSR_EL2  = 0xe200,
  SPSR_hyp  = SPSR_EL2,
  ELR_EL2   = 0xe201,
  SP_EL1    = 0xe208,
  SPSR_irq  = 0xe218,
  SPSR_abt  = 0xe219,
  SPSR_und  = 0xe21a,
  SPSR_fiq  = 0xe21b,
  SPSR_EL3  = 0xf200,
  ELR_EL3   = 0xf201,
  SP_EL2    = 0xf208,


  // Table in section 3.10.8
  MIDR_EL1 = 0xc000,
  CTR_EL0 = 0xd801,
  MPIDR_EL1 = 0xc005,
  ECOIDR_EL1 = 0xc006,
  DCZID_EL0 = 0xd807,
  MVFR0_EL1 = 0xc018,
  MVFR1_EL1 = 0xc019,
  ID_AA64PFR0_EL1 = 0xc020,
  ID_AA64PFR1_EL1 = 0xc021,
  ID_AA64DFR0_EL1 = 0xc028,
  ID_AA64DFR1_EL1 = 0xc029,
  ID_AA64ISAR0_EL1 = 0xc030,
  ID_AA64ISAR1_EL1 = 0xc031,
  ID_AA64MMFR0_EL1 = 0xc038,
  ID_AA64MMFR1_EL1 = 0xc039,
  CCSIDR_EL1 = 0xc800,
  CLIDR_EL1 = 0xc801,
  AIDR_EL1 = 0xc807,
  CSSELR_EL1 = 0xd000,
  VPIDR_EL2 = 0xe000,
  VMPIDR_EL2 = 0xe005,
  SCTLR_EL1 = 0xc080,
  SCTLR_EL2 = 0xe080,
  SCTLR_EL3 = 0xf080,
  ACTLR_EL1 = 0xc081,
  ACTLR_EL2 = 0xe081,
  ACTLR_EL3 = 0xf081,
  CPACR_EL1 = 0xc082,
  CPTR_EL2 = 0xe08a,
  CPTR_EL3 = 0xf08a,
  SCR_EL3 = 0xf088,
  HCR_EL2 = 0xe088,
  MDCR_EL2 = 0xe089,
  MDCR_EL3 = 0xf099,
  HSTR_EL2 = 0xe08b,
  HACR_EL2 = 0xe08f,
  TTBR0_EL1 = 0xc100,
  TTBR1_EL1 = 0xc101,
  TTBR0_EL2 = 0xe100,
  TTBR0_EL3 = 0xf100,
  VTTBR_EL2 = 0xe108,
  TCR_EL1 = 0xc102,
  TCR_EL2 = 0xe102,
  TCR_EL3 = 0xf102,
  VTCR_EL2 = 0xe10a,
  ADFSR_EL1 = 0xc288,
  AIFSR_EL1 = 0xc289,
  ADFSR_EL2 = 0xe288,
  AIFSR_EL2 = 0xe289,
  ADFSR_EL3 = 0xf288,
  AIFSR_EL3 = 0xf289,
  ESR_EL1 = 0xc290,
  ESR_EL2 = 0xe290,
  ESR_EL3 = 0xf290,
  FAR_EL1 = 0xc300,
  FAR_EL2 = 0xe300,
  FAR_EL3 = 0xf300,
  HPFAR_EL2 = 0xe304,
  PAR_EL1 = 0xc3a0,
  MAIR_EL1 = 0xc510,
  MAIR_EL2 = 0xe510,
  MAIR_EL3 = 0xf510,
  AMAIR_EL1 = 0xc518,
  AMAIR_EL2 = 0xe518,
  AMAIR_EL3 = 0xf518,
  VBAR_EL1 = 0xc600,
  VBAR_EL2 = 0xe600,
  VBAR_EL3 = 0xf600,
  RVBAR_EL1 = 0xc601,
  RVBAR_EL2 = 0xe601,
  RVBAR_EL3 = 0xf601,
  ISR_EL1 = 0xc608,
  CONTEXTIDR_EL1 = 0xc681,
  TPIDR_EL0 = 0xde82,
  TPIDRRO_EL0 = 0xde83,
  TPIDR_EL1 = 0xc684,
  TPIDR_EL2 = 0xe682,
  TPIDR_EL3 = 0xf682,
  TEECR32_EL1 = 0x9000,
  CNTFRQ_EL0 = 0xdf00,
  CNTPCT_EL0 = 0xdf01,
  CNTVCT_EL0 = 0xdf02,
  CNTVOFF_EL2 = 0xe703,
  CNTKCTL_EL1 = 0xc708,
  CNTHCTL_EL2 = 0xe708,
  CNTP_TVAL_EL0 = 0xdf10,
  CNTP_CTL_EL0 = 0xdf11,
  CNTP_CVAL_EL0 = 0xdf12,
  CNTV_TVAL_EL0 = 0xdf18,
  CNTV_CTL_EL0 = 0xdf19,
  CNTV_CVAL_EL0 = 0xdf1a,
  CNTHP_TVAL_EL2 = 0xe710,
  CNTHP_CTL_EL2 = 0xe711,
  CNTHP_CVAL_EL2 = 0xe712,
  CNTPS_TVAL_EL1 = 0xff10,
  CNTPS_CTL_EL1 = 0xff11,
  CNTPS_CVAL_EL1= 0xff12,

  PMEVCNTR0_EL0  = 0xdf40,
  PMEVCNTR1_EL0  = 0xdf41,
  PMEVCNTR2_EL0  = 0xdf42,
  PMEVCNTR3_EL0  = 0xdf43,
  PMEVCNTR4_EL0  = 0xdf44,
  PMEVCNTR5_EL0  = 0xdf45,
  PMEVCNTR6_EL0  = 0xdf46,
  PMEVCNTR7_EL0  = 0xdf47,
  PMEVCNTR8_EL0  = 0xdf48,
  PMEVCNTR9_EL0  = 0xdf49,
  PMEVCNTR10_EL0 = 0xdf4a,
  PMEVCNTR11_EL0 = 0xdf4b,
  PMEVCNTR12_EL0 = 0xdf4c,
  PMEVCNTR13_EL0 = 0xdf4d,
  PMEVCNTR14_EL0 = 0xdf4e,
  PMEVCNTR15_EL0 = 0xdf4f,
  PMEVCNTR16_EL0 = 0xdf50,
  PMEVCNTR17_EL0 = 0xdf51,
  PMEVCNTR18_EL0 = 0xdf52,
  PMEVCNTR19_EL0 = 0xdf53,
  PMEVCNTR20_EL0 = 0xdf54,
  PMEVCNTR21_EL0 = 0xdf55,
  PMEVCNTR22_EL0 = 0xdf56,
  PMEVCNTR23_EL0 = 0xdf57,
  PMEVCNTR24_EL0 = 0xdf58,
  PMEVCNTR25_EL0 = 0xdf59,
  PMEVCNTR26_EL0 = 0xdf5a,
  PMEVCNTR27_EL0 = 0xdf5b,
  PMEVCNTR28_EL0 = 0xdf5c,
  PMEVCNTR29_EL0 = 0xdf5d,
  PMEVCNTR30_EL0 = 0xdf5e,

  PMEVTYPER0_EL0  = 0xdf60,
  PMEVTYPER1_EL0  = 0xdf61,
  PMEVTYPER2_EL0  = 0xdf62,
  PMEVTYPER3_EL0  = 0xdf63,
  PMEVTYPER4_EL0  = 0xdf64,
  PMEVTYPER5_EL0  = 0xdf65,
  PMEVTYPER6_EL0  = 0xdf66,
  PMEVTYPER7_EL0  = 0xdf67,
  PMEVTYPER8_EL0  = 0xdf68,
  PMEVTYPER9_EL0  = 0xdf69,
  PMEVTYPER10_EL0 = 0xdf6a,
  PMEVTYPER11_EL0 = 0xdf6b,
  PMEVTYPER12_EL0 = 0xdf6c,
  PMEVTYPER13_EL0 = 0xdf6d,
  PMEVTYPER14_EL0 = 0xdf6e,
  PMEVTYPER15_EL0 = 0xdf6f,
  PMEVTYPER16_EL0 = 0xdf70,
  PMEVTYPER17_EL0 = 0xdf71,
  PMEVTYPER18_EL0 = 0xdf72,
  PMEVTYPER19_EL0 = 0xdf73,
  PMEVTYPER20_EL0 = 0xdf74,
  PMEVTYPER21_EL0 = 0xdf75,
  PMEVTYPER22_EL0 = 0xdf76,
  PMEVTYPER23_EL0 = 0xdf77,
  PMEVTYPER24_EL0 = 0xdf78,
  PMEVTYPER25_EL0 = 0xdf79,
  PMEVTYPER26_EL0 = 0xdf7a,
  PMEVTYPER27_EL0 = 0xdf7b,
  PMEVTYPER28_EL0 = 0xdf7c,
  PMEVTYPER29_EL0 = 0xdf7d,
  PMEVTYPER30_EL0 = 0xdf7e,

  PMCCFILTR_EL0  = 0xdf7f,

  RMR_EL3 = 0xf602,
  RMR_EL2 = 0xd602,
  RMR_EL1 = 0xce02,

  // Debug Architecture 5.3, Table 17.
  MDCCSR_EL0   = A64_SYSREG_ENC(2, 0, 0, 1, 3),
  MDCCINT_EL1  = A64_SYSREG_ENC(2, 0, 0, 2, 0),
  DBGDTR_EL0   = A64_SYSREG_ENC(2, 0, 0, 4, 3),
  DBGDTRRX_EL0 = A64_SYSREG_ENC(2, 0, 0, 5, 3),
  DBGDTRTX_EL0 = DBGDTRRX_EL0,
  DBGVCR32_EL2 = A64_SYSREG_ENC(2, 0, 0, 7, 4),
  OSDTRRX_EL1  = A64_SYSREG_ENC(2, 0, 2, 0, 0),
  MDSCR_EL1    = A64_SYSREG_ENC(2, 0, 2, 2, 0),
  OSDTRTX_EL1  = A64_SYSREG_ENC(2, 0, 2, 3, 0),
  OSECCR_EL11  = A64_SYSREG_ENC(2, 0, 2, 6, 0),

  DBGBVR0_EL1  = A64_SYSREG_ENC(2, 0, 4, 0, 0),
  DBGBVR1_EL1  = A64_SYSREG_ENC(2, 0, 4, 1, 0),
  DBGBVR2_EL1  = A64_SYSREG_ENC(2, 0, 4, 2, 0),
  DBGBVR3_EL1  = A64_SYSREG_ENC(2, 0, 4, 3, 0),
  DBGBVR4_EL1  = A64_SYSREG_ENC(2, 0, 4, 4, 0),
  DBGBVR5_EL1  = A64_SYSREG_ENC(2, 0, 4, 5, 0),
  DBGBVR6_EL1  = A64_SYSREG_ENC(2, 0, 4, 6, 0),
  DBGBVR7_EL1  = A64_SYSREG_ENC(2, 0, 4, 7, 0),
  DBGBVR8_EL1  = A64_SYSREG_ENC(2, 0, 4, 8, 0),
  DBGBVR9_EL1  = A64_SYSREG_ENC(2, 0, 4, 9, 0),
  DBGBVR10_EL1 = A64_SYSREG_ENC(2, 0, 4, 10, 0),
  DBGBVR11_EL1 = A64_SYSREG_ENC(2, 0, 4, 11, 0),
  DBGBVR12_EL1 = A64_SYSREG_ENC(2, 0, 4, 12, 0),
  DBGBVR13_EL1 = A64_SYSREG_ENC(2, 0, 4, 13, 0),
  DBGBVR14_EL1 = A64_SYSREG_ENC(2, 0, 4, 14, 0),
  DBGBVR15_EL1 = A64_SYSREG_ENC(2, 0, 4, 15, 0),

  DBGBCR0_EL1  = A64_SYSREG_ENC(2, 0, 5, 0, 0),
  DBGBCR1_EL1  = A64_SYSREG_ENC(2, 0, 5, 1, 0),
  DBGBCR2_EL1  = A64_SYSREG_ENC(2, 0, 5, 2, 0),
  DBGBCR3_EL1  = A64_SYSREG_ENC(2, 0, 5, 3, 0),
  DBGBCR4_EL1  = A64_SYSREG_ENC(2, 0, 5, 4, 0),
  DBGBCR5_EL1  = A64_SYSREG_ENC(2, 0, 5, 5, 0),
  DBGBCR6_EL1  = A64_SYSREG_ENC(2, 0, 5, 6, 0),
  DBGBCR7_EL1  = A64_SYSREG_ENC(2, 0, 5, 7, 0),
  DBGBCR8_EL1  = A64_SYSREG_ENC(2, 0, 5, 8, 0),
  DBGBCR9_EL1  = A64_SYSREG_ENC(2, 0, 5, 9, 0),
  DBGBCR10_EL1 = A64_SYSREG_ENC(2, 0, 5, 10, 0),
  DBGBCR11_EL1 = A64_SYSREG_ENC(2, 0, 5, 11, 0),
  DBGBCR12_EL1 = A64_SYSREG_ENC(2, 0, 5, 12, 0),
  DBGBCR13_EL1 = A64_SYSREG_ENC(2, 0, 5, 13, 0),
  DBGBCR14_EL1 = A64_SYSREG_ENC(2, 0, 5, 14, 0),
  DBGBCR15_EL1 = A64_SYSREG_ENC(2, 0, 5, 15, 0),

  DBGWVR0_EL1  = A64_SYSREG_ENC(2, 0, 6, 0, 0),
  DBGWVR1_EL1  = A64_SYSREG_ENC(2, 0, 6, 1, 0),
  DBGWVR2_EL1  = A64_SYSREG_ENC(2, 0, 6, 2, 0),
  DBGWVR3_EL1  = A64_SYSREG_ENC(2, 0, 6, 3, 0),
  DBGWVR4_EL1  = A64_SYSREG_ENC(2, 0, 6, 4, 0),
  DBGWVR5_EL1  = A64_SYSREG_ENC(2, 0, 6, 5, 0),
  DBGWVR6_EL1  = A64_SYSREG_ENC(2, 0, 6, 6, 0),
  DBGWVR7_EL1  = A64_SYSREG_ENC(2, 0, 6, 7, 0),
  DBGWVR8_EL1  = A64_SYSREG_ENC(2, 0, 6, 8, 0),
  DBGWVR9_EL1  = A64_SYSREG_ENC(2, 0, 6, 9, 0),
  DBGWVR10_EL1 = A64_SYSREG_ENC(2, 0, 6, 10, 0),
  DBGWVR11_EL1 = A64_SYSREG_ENC(2, 0, 6, 11, 0),
  DBGWVR12_EL1 = A64_SYSREG_ENC(2, 0, 6, 12, 0),
  DBGWVR13_EL1 = A64_SYSREG_ENC(2, 0, 6, 13, 0),
  DBGWVR14_EL1 = A64_SYSREG_ENC(2, 0, 6, 14, 0),
  DBGWVR15_EL1 = A64_SYSREG_ENC(2, 0, 6, 15, 0),

  DBGWCR0_EL1  = A64_SYSREG_ENC(2, 0, 7, 0, 0),
  DBGWCR1_EL1  = A64_SYSREG_ENC(2, 0, 7, 1, 0),
  DBGWCR2_EL1  = A64_SYSREG_ENC(2, 0, 7, 2, 0),
  DBGWCR3_EL1  = A64_SYSREG_ENC(2, 0, 7, 3, 0),
  DBGWCR4_EL1  = A64_SYSREG_ENC(2, 0, 7, 4, 0),
  DBGWCR5_EL1  = A64_SYSREG_ENC(2, 0, 7, 5, 0),
  DBGWCR6_EL1  = A64_SYSREG_ENC(2, 0, 7, 6, 0),
  DBGWCR7_EL1  = A64_SYSREG_ENC(2, 0, 7, 7, 0),
  DBGWCR8_EL1  = A64_SYSREG_ENC(2, 0, 7, 8, 0),
  DBGWCR9_EL1  = A64_SYSREG_ENC(2, 0, 7, 9, 0),
  DBGWCR10_EL1 = A64_SYSREG_ENC(2, 0, 7, 10, 0),
  DBGWCR11_EL1 = A64_SYSREG_ENC(2, 0, 7, 11, 0),
  DBGWCR12_EL1 = A64_SYSREG_ENC(2, 0, 7, 12, 0),
  DBGWCR13_EL1 = A64_SYSREG_ENC(2, 0, 7, 13, 0),
  DBGWCR14_EL1 = A64_SYSREG_ENC(2, 0, 7, 14, 0),
  DBGWCR15_EL1 = A64_SYSREG_ENC(2, 0, 7, 15, 0),

  MDRAR_EL1    = A64_SYSREG_ENC(2, 1, 0, 0, 0),
  OSLAR_EL1    = A64_SYSREG_ENC(2, 1, 4, 0, 0),
  OSLSR_EL1    = A64_SYSREG_ENC(2, 1, 4, 1, 0),
  OSDLR_EL1    = A64_SYSREG_ENC(2, 1, 4, 3, 0),
  DBGPRCR_EL1  = A64_SYSREG_ENC(2, 1, 4, 4, 0),

  DBGCLAIMSET_EL1   = A64_SYSREG_ENC(2, 7, 6, 8, 0),
  DBGCLAIMCLR_EL1   = A64_SYSREG_ENC(2, 7, 6, 9, 0),
  DBGAUTHSTATUS_EL1 = A64_SYSREG_ENC(2, 7, 6, 14, 0),

  DBGDEVID2    = A64_SYSREG_ENC(2, 7, 7, 0, 0),
  DBGDEVID1    = A64_SYSREG_ENC(2, 7, 7, 1, 0),
  DBGDEVID0    = A64_SYSREG_ENC(2, 7, 7, 2, 0),

  // The following registers are defined to allow access from AArch64 to
  // registers which are only used in the AArch32 architecture.
  DACR32_EL2 = 0xe180,
  IFSR32_EL2 = 0xe281,
  TEEHBR32_EL1 = 0x9080,
  SDER32_EL3 = 0xf089,
  FPEXC32_EL2 = 0xe298,

  // Cyclone specific system registers
  CPM_IOACC_CTL_EL3 = 0xff90,

  // Architectural system registers
  ID_PFR0_EL1 = 0xc008,
  ID_PFR1_EL1 = 0xc009,
  ID_DFR0_EL1 = 0xc00a,
  ID_AFR0_EL1 = 0xc00b,
  ID_ISAR0_EL1 = 0xc010,
  ID_ISAR1_EL1 = 0xc011,
  ID_ISAR2_EL1 = 0xc012,
  ID_ISAR3_EL1 = 0xc013,
  ID_ISAR4_EL1 = 0xc014,
  ID_ISAR5_EL1 = 0xc015,
  AFSR1_EL1 = 0xc289, // note same as old AIFSR_EL1
  AFSR0_EL1 = 0xc288, // note same as old ADFSR_EL1
  REVIDR_EL1 = 0xc006 // note same as old ECOIDR_EL1

};
#undef A64_SYSREG_ENC

static inline const char *getSystemRegisterName(SystemRegister Reg) {
  switch(Reg) {
  default: return NULL; // Caller is responsible for handling invalid value.
  case SPSR_EL1: return "SPSR_EL1";
  case ELR_EL1: return "ELR_EL1";
  case SP_EL0: return "SP_EL0";
  case SPSel: return "SPSel";
  case DAIF: return "DAIF";
  case CurrentEL: return "CurrentEL";
  case NZCV: return "NZCV";
  case FPCR: return "FPCR";
  case FPSR: return "FPSR";
  case DSPSR: return "DSPSR";
  case DLR: return "DLR";
  case SPSR_EL2: return "SPSR_EL2";
  case ELR_EL2: return "ELR_EL2";
  case SP_EL1: return "SP_EL1";
  case SPSR_irq: return "SPSR_irq";
  case SPSR_abt: return "SPSR_abt";
  case SPSR_und: return "SPSR_und";
  case SPSR_fiq: return "SPSR_fiq";
  case SPSR_EL3: return "SPSR_EL3";
  case ELR_EL3: return "ELR_EL3";
  case SP_EL2: return "SP_EL2";
  case MIDR_EL1: return "MIDR_EL1";
  case CTR_EL0: return "CTR_EL0";
  case MPIDR_EL1: return "MPIDR_EL1";
  case DCZID_EL0: return "DCZID_EL0";
  case MVFR0_EL1: return "MVFR0_EL1";
  case MVFR1_EL1: return "MVFR1_EL1";
  case ID_AA64PFR0_EL1: return "ID_AA64PFR0_EL1";
  case ID_AA64PFR1_EL1: return "ID_AA64PFR1_EL1";
  case ID_AA64DFR0_EL1: return "ID_AA64DFR0_EL1";
  case ID_AA64DFR1_EL1: return "ID_AA64DFR1_EL1";
  case ID_AA64ISAR0_EL1: return "ID_AA64ISAR0_EL1";
  case ID_AA64ISAR1_EL1: return "ID_AA64ISAR1_EL1";
  case ID_AA64MMFR0_EL1: return "ID_AA64MMFR0_EL1";
  case ID_AA64MMFR1_EL1: return "ID_AA64MMFR1_EL1";
  case CCSIDR_EL1: return "CCSIDR_EL1";
  case CLIDR_EL1: return "CLIDR_EL1";
  case AIDR_EL1: return "AIDR_EL1";
  case CSSELR_EL1: return "CSSELR_EL1";
  case VPIDR_EL2: return "VPIDR_EL2";
  case VMPIDR_EL2: return "VMPIDR_EL2";
  case SCTLR_EL1: return "SCTLR_EL1";
  case SCTLR_EL2: return "SCTLR_EL2";
  case SCTLR_EL3: return "SCTLR_EL3";
  case ACTLR_EL1: return "ACTLR_EL1";
  case ACTLR_EL2: return "ACTLR_EL2";
  case ACTLR_EL3: return "ACTLR_EL3";
  case CPACR_EL1: return "CPACR_EL1";
  case CPTR_EL2: return "CPTR_EL2";
  case CPTR_EL3: return "CPTR_EL3";
  case SCR_EL3: return "SCR_EL3";
  case HCR_EL2: return "HCR_EL2";
  case MDCR_EL2: return "MDCR_EL2";
  case MDCR_EL3: return "MDCR_EL3";
  case HSTR_EL2: return "HSTR_EL2";
  case HACR_EL2: return "HACR_EL2";
  case TTBR0_EL1: return "TTBR0_EL1";
  case TTBR1_EL1: return "TTBR1_EL1";
  case TTBR0_EL2: return "TTBR0_EL2";
  case TTBR0_EL3: return "TTBR0_EL3";
  case VTTBR_EL2: return "VTTBR_EL2";
  case TCR_EL1: return "TCR_EL1";
  case TCR_EL2: return "TCR_EL2";
  case TCR_EL3: return "TCR_EL3";
  case VTCR_EL2: return "VTCR_EL2";
  case ADFSR_EL2: return "ADFSR_EL2";
  case AIFSR_EL2: return "AIFSR_EL2";
  case ADFSR_EL3: return "ADFSR_EL3";
  case AIFSR_EL3: return "AIFSR_EL3";
  case ESR_EL1: return "ESR_EL1";
  case ESR_EL2: return "ESR_EL2";
  case ESR_EL3: return "ESR_EL3";
  case FAR_EL1: return "FAR_EL1";
  case FAR_EL2: return "FAR_EL2";
  case FAR_EL3: return "FAR_EL3";
  case HPFAR_EL2: return "HPFAR_EL2";
  case PAR_EL1: return "PAR_EL1";
  case MAIR_EL1: return "MAIR_EL1";
  case MAIR_EL2: return "MAIR_EL2";
  case MAIR_EL3: return "MAIR_EL3";
  case AMAIR_EL1: return "AMAIR_EL1";
  case AMAIR_EL2: return "AMAIR_EL2";
  case AMAIR_EL3: return "AMAIR_EL3";
  case VBAR_EL1: return "VBAR_EL1";
  case VBAR_EL2: return "VBAR_EL2";
  case VBAR_EL3: return "VBAR_EL3";
  case RVBAR_EL1: return "RVBAR_EL1";
  case RVBAR_EL2: return "RVBAR_EL2";
  case RVBAR_EL3: return "RVBAR_EL3";
  case ISR_EL1: return "ISR_EL1";
  case CONTEXTIDR_EL1: return "CONTEXTIDR_EL1";
  case TPIDR_EL0: return "TPIDR_EL0";
  case TPIDRRO_EL0: return "TPIDRRO_EL0";
  case TPIDR_EL1: return "TPIDR_EL1";
  case TPIDR_EL2: return "TPIDR_EL2";
  case TPIDR_EL3: return "TPIDR_EL3";
  case TEECR32_EL1: return "TEECR32_EL1";
  case CNTFRQ_EL0: return "CNTFRQ_EL0";
  case CNTPCT_EL0: return "CNTPCT_EL0";
  case CNTVCT_EL0: return "CNTVCT_EL0";
  case CNTVOFF_EL2: return "CNTVOFF_EL2";
  case CNTKCTL_EL1: return "CNTKCTL_EL1";
  case CNTHCTL_EL2: return "CNTHCTL_EL2";
  case CNTP_TVAL_EL0: return "CNTP_TVAL_EL0";
  case CNTP_CTL_EL0: return "CNTP_CTL_EL0";
  case CNTP_CVAL_EL0: return "CNTP_CVAL_EL0";
  case CNTV_TVAL_EL0: return "CNTV_TVAL_EL0";
  case CNTV_CTL_EL0: return "CNTV_CTL_EL0";
  case CNTV_CVAL_EL0: return "CNTV_CVAL_EL0";
  case CNTHP_TVAL_EL2: return "CNTHP_TVAL_EL2";
  case CNTHP_CTL_EL2: return "CNTHP_CTL_EL2";
  case CNTHP_CVAL_EL2: return "CNTHP_CVAL_EL2";
  case CNTPS_TVAL_EL1: return "CNTPS_TVAL_EL1";
  case CNTPS_CTL_EL1: return "CNTPS_CTL_EL1";
  case CNTPS_CVAL_EL1: return "CNTPS_CVAL_EL1";
  case DACR32_EL2: return "DACR32_EL2";
  case IFSR32_EL2: return "IFSR32_EL2";
  case TEEHBR32_EL1: return "TEEHBR32_EL1";
  case SDER32_EL3: return "SDER32_EL3";
  case FPEXC32_EL2: return "FPEXC32_EL2";
  case PMEVCNTR0_EL0: return "PMEVCNTR0_EL0";
  case PMEVCNTR1_EL0: return "PMEVCNTR1_EL0";
  case PMEVCNTR2_EL0: return "PMEVCNTR2_EL0";
  case PMEVCNTR3_EL0: return "PMEVCNTR3_EL0";
  case PMEVCNTR4_EL0: return "PMEVCNTR4_EL0";
  case PMEVCNTR5_EL0: return "PMEVCNTR5_EL0";
  case PMEVCNTR6_EL0: return "PMEVCNTR6_EL0";
  case PMEVCNTR7_EL0: return "PMEVCNTR7_EL0";
  case PMEVCNTR8_EL0: return "PMEVCNTR8_EL0";
  case PMEVCNTR9_EL0: return "PMEVCNTR9_EL0";
  case PMEVCNTR10_EL0: return "PMEVCNTR10_EL0";
  case PMEVCNTR11_EL0: return "PMEVCNTR11_EL0";
  case PMEVCNTR12_EL0: return "PMEVCNTR12_EL0";
  case PMEVCNTR13_EL0: return "PMEVCNTR13_EL0";
  case PMEVCNTR14_EL0: return "PMEVCNTR14_EL0";
  case PMEVCNTR15_EL0: return "PMEVCNTR15_EL0";
  case PMEVCNTR16_EL0: return "PMEVCNTR16_EL0";
  case PMEVCNTR17_EL0: return "PMEVCNTR17_EL0";
  case PMEVCNTR18_EL0: return "PMEVCNTR18_EL0";
  case PMEVCNTR19_EL0: return "PMEVCNTR19_EL0";
  case PMEVCNTR20_EL0: return "PMEVCNTR20_EL0";
  case PMEVCNTR21_EL0: return "PMEVCNTR21_EL0";
  case PMEVCNTR22_EL0: return "PMEVCNTR22_EL0";
  case PMEVCNTR23_EL0: return "PMEVCNTR23_EL0";
  case PMEVCNTR24_EL0: return "PMEVCNTR24_EL0";
  case PMEVCNTR25_EL0: return "PMEVCNTR25_EL0";
  case PMEVCNTR26_EL0: return "PMEVCNTR26_EL0";
  case PMEVCNTR27_EL0: return "PMEVCNTR27_EL0";
  case PMEVCNTR28_EL0: return "PMEVCNTR28_EL0";
  case PMEVCNTR29_EL0: return "PMEVCNTR29_EL0";
  case PMEVCNTR30_EL0: return "PMEVCNTR30_EL0";
  case PMEVTYPER0_EL0: return "PMEVTYPER0_EL0";
  case PMEVTYPER1_EL0: return "PMEVTYPER1_EL0";
  case PMEVTYPER2_EL0: return "PMEVTYPER2_EL0";
  case PMEVTYPER3_EL0: return "PMEVTYPER3_EL0";
  case PMEVTYPER4_EL0: return "PMEVTYPER4_EL0";
  case PMEVTYPER5_EL0: return "PMEVTYPER5_EL0";
  case PMEVTYPER6_EL0: return "PMEVTYPER6_EL0";
  case PMEVTYPER7_EL0: return "PMEVTYPER7_EL0";
  case PMEVTYPER8_EL0: return "PMEVTYPER8_EL0";
  case PMEVTYPER9_EL0: return "PMEVTYPER9_EL0";
  case PMEVTYPER10_EL0: return "PMEVTYPER10_EL0";
  case PMEVTYPER11_EL0: return "PMEVTYPER11_EL0";
  case PMEVTYPER12_EL0: return "PMEVTYPER12_EL0";
  case PMEVTYPER13_EL0: return "PMEVTYPER13_EL0";
  case PMEVTYPER14_EL0: return "PMEVTYPER14_EL0";
  case PMEVTYPER15_EL0: return "PMEVTYPER15_EL0";
  case PMEVTYPER16_EL0: return "PMEVTYPER16_EL0";
  case PMEVTYPER17_EL0: return "PMEVTYPER17_EL0";
  case PMEVTYPER18_EL0: return "PMEVTYPER18_EL0";
  case PMEVTYPER19_EL0: return "PMEVTYPER19_EL0";
  case PMEVTYPER20_EL0: return "PMEVTYPER20_EL0";
  case PMEVTYPER21_EL0: return "PMEVTYPER21_EL0";
  case PMEVTYPER22_EL0: return "PMEVTYPER22_EL0";
  case PMEVTYPER23_EL0: return "PMEVTYPER23_EL0";
  case PMEVTYPER24_EL0: return "PMEVTYPER24_EL0";
  case PMEVTYPER25_EL0: return "PMEVTYPER25_EL0";
  case PMEVTYPER26_EL0: return "PMEVTYPER26_EL0";
  case PMEVTYPER27_EL0: return "PMEVTYPER27_EL0";
  case PMEVTYPER28_EL0: return "PMEVTYPER28_EL0";
  case PMEVTYPER29_EL0: return "PMEVTYPER29_EL0";
  case PMEVTYPER30_EL0: return "PMEVTYPER30_EL0";
  case PMCCFILTR_EL0: return "PMCCFILTR_EL0";
  case RMR_EL3: return "RMR_EL3";
  case RMR_EL2: return "RMR_EL2";
  case RMR_EL1: return "RMR_EL1";
  case CPM_IOACC_CTL_EL3: return "CPM_IOACC_CTL_EL3";
  case MDCCSR_EL0: return "MDCCSR_EL0";
  case MDCCINT_EL1: return "MDCCINT_EL1";
  case DBGDTR_EL0: return "DBGDTR_EL0";
  case DBGDTRRX_EL0: return "DBGDTRRX_EL0";
  case DBGVCR32_EL2: return "DBGVCR32_EL2";
  case OSDTRRX_EL1: return "OSDTRRX_EL1";
  case MDSCR_EL1: return "MDSCR_EL1";
  case OSDTRTX_EL1: return "OSDTRTX_EL1";
  case OSECCR_EL11: return "OSECCR_EL11";
  case DBGBVR0_EL1: return "DBGBVR0_EL1";
  case DBGBVR1_EL1: return "DBGBVR1_EL1";
  case DBGBVR2_EL1: return "DBGBVR2_EL1";
  case DBGBVR3_EL1: return "DBGBVR3_EL1";
  case DBGBVR4_EL1: return "DBGBVR4_EL1";
  case DBGBVR5_EL1: return "DBGBVR5_EL1";
  case DBGBVR6_EL1: return "DBGBVR6_EL1";
  case DBGBVR7_EL1: return "DBGBVR7_EL1";
  case DBGBVR8_EL1: return "DBGBVR8_EL1";
  case DBGBVR9_EL1: return "DBGBVR9_EL1";
  case DBGBVR10_EL1: return "DBGBVR10_EL1";
  case DBGBVR11_EL1: return "DBGBVR11_EL1";
  case DBGBVR12_EL1: return "DBGBVR12_EL1";
  case DBGBVR13_EL1: return "DBGBVR13_EL1";
  case DBGBVR14_EL1: return "DBGBVR14_EL1";
  case DBGBVR15_EL1: return "DBGBVR15_EL1";
  case DBGBCR0_EL1: return "DBGBCR0_EL1";
  case DBGBCR1_EL1: return "DBGBCR1_EL1";
  case DBGBCR2_EL1: return "DBGBCR2_EL1";
  case DBGBCR3_EL1: return "DBGBCR3_EL1";
  case DBGBCR4_EL1: return "DBGBCR4_EL1";
  case DBGBCR5_EL1: return "DBGBCR5_EL1";
  case DBGBCR6_EL1: return "DBGBCR6_EL1";
  case DBGBCR7_EL1: return "DBGBCR7_EL1";
  case DBGBCR8_EL1: return "DBGBCR8_EL1";
  case DBGBCR9_EL1: return "DBGBCR9_EL1";
  case DBGBCR10_EL1: return "DBGBCR10_EL1";
  case DBGBCR11_EL1: return "DBGBCR11_EL1";
  case DBGBCR12_EL1: return "DBGBCR12_EL1";
  case DBGBCR13_EL1: return "DBGBCR13_EL1";
  case DBGBCR14_EL1: return "DBGBCR14_EL1";
  case DBGBCR15_EL1: return "DBGBCR15_EL1";
  case DBGWVR0_EL1: return "DBGWVR0_EL1";
  case DBGWVR1_EL1: return "DBGWVR1_EL1";
  case DBGWVR2_EL1: return "DBGWVR2_EL1";
  case DBGWVR3_EL1: return "DBGWVR3_EL1";
  case DBGWVR4_EL1: return "DBGWVR4_EL1";
  case DBGWVR5_EL1: return "DBGWVR5_EL1";
  case DBGWVR6_EL1: return "DBGWVR6_EL1";
  case DBGWVR7_EL1: return "DBGWVR7_EL1";
  case DBGWVR8_EL1: return "DBGWVR8_EL1";
  case DBGWVR9_EL1: return "DBGWVR9_EL1";
  case DBGWVR10_EL1: return "DBGWVR10_EL1";
  case DBGWVR11_EL1: return "DBGWVR11_EL1";
  case DBGWVR12_EL1: return "DBGWVR12_EL1";
  case DBGWVR13_EL1: return "DBGWVR13_EL1";
  case DBGWVR14_EL1: return "DBGWVR14_EL1";
  case DBGWVR15_EL1: return "DBGWVR15_EL1";
  case DBGWCR0_EL1: return "DBGWCR0_EL1";
  case DBGWCR1_EL1: return "DBGWCR1_EL1";
  case DBGWCR2_EL1: return "DBGWCR2_EL1";
  case DBGWCR3_EL1: return "DBGWCR3_EL1";
  case DBGWCR4_EL1: return "DBGWCR4_EL1";
  case DBGWCR5_EL1: return "DBGWCR5_EL1";
  case DBGWCR6_EL1: return "DBGWCR6_EL1";
  case DBGWCR7_EL1: return "DBGWCR7_EL1";
  case DBGWCR8_EL1: return "DBGWCR8_EL1";
  case DBGWCR9_EL1: return "DBGWCR9_EL1";
  case DBGWCR10_EL1: return "DBGWCR10_EL1";
  case DBGWCR11_EL1: return "DBGWCR11_EL1";
  case DBGWCR12_EL1: return "DBGWCR12_EL1";
  case DBGWCR13_EL1: return "DBGWCR13_EL1";
  case DBGWCR14_EL1: return "DBGWCR14_EL1";
  case DBGWCR15_EL1: return "DBGWCR15_EL1";
  case MDRAR_EL1: return "MDRAR_EL1";
  case OSLAR_EL1: return "OSLAR_EL1";
  case OSLSR_EL1: return "OSLSR_EL1";
  case OSDLR_EL1: return "OSDLR_EL1";
  case DBGPRCR_EL1: return "DBGPRCR_EL1";
  case DBGCLAIMSET_EL1: return "DBGCLAIMSET_EL1";
  case DBGCLAIMCLR_EL1: return "DBGCLAIMCLR_EL1";
  case DBGAUTHSTATUS_EL1: return "DBGAUTHSTATUS_EL1";
  case DBGDEVID2: return "DBGDEVID2";
  case DBGDEVID1: return "DBGDEVID1";
  case DBGDEVID0: return "DBGDEVID0";
  case ID_PFR0_EL1: return "ID_PFR0_EL1";
  case ID_PFR1_EL1: return "ID_PFR1_EL1";
  case ID_DFR0_EL1: return "ID_DFR0_EL1";
  case ID_AFR0_EL1: return "ID_AFR0_EL1";
  case ID_ISAR0_EL1: return "ID_ISAR0_EL1";
  case ID_ISAR1_EL1: return "ID_ISAR1_EL1";
  case ID_ISAR2_EL1: return "ID_ISAR2_EL1";
  case ID_ISAR3_EL1: return "ID_ISAR3_EL1";
  case ID_ISAR4_EL1: return "ID_ISAR4_EL1";
  case ID_ISAR5_EL1: return "ID_ISAR5_EL1";
  case AFSR1_EL1: return "AFSR1_EL1";
  case AFSR0_EL1: return "AFSR0_EL1";
  case REVIDR_EL1: return "REVIDR_EL1";
  }
}

enum CPSRField {
  InvalidCPSRField = 0xff,
  cpsr_SPSel = 0x5,
  cpsr_DAIFSet = 0x1e,
  cpsr_DAIFClr = 0x1f
};

static inline const char *getCPSRFieldName(CPSRField Val) {
  switch(Val) {
  default: assert(0 && "Invalid system register value!");
  case cpsr_SPSel: return "SPSel";
  case cpsr_DAIFSet: return "DAIFSet";
  case cpsr_DAIFClr: return "DAIFClr";
  }
}

} // end namespace ARM64SYS

namespace ARM64II {
  /// Target Operand Flag enum.
  enum TOF {
    //===------------------------------------------------------------------===//
    // ARM64 Specific MachineOperand flags.

    MO_NO_FLAG,

    MO_FRAGMENT = 0x7,

    /// MO_PAGE - A symbol operand with this flag represents the pc-relative
    /// offset of the 4K page containing the symbol.  This is used with the
    /// ADRP instruction.
    MO_PAGE = 1,

    /// MO_PAGEOFF - A symbol operand with this flag represents the offset of
    /// that symbol within a 4K page.  This offset is added to the page address
    /// to produce the complete address.
    MO_PAGEOFF = 2,

    /// MO_G3 - A symbol operand with this flag (granule 3) represents the high
    /// 16-bits of a 64-bit address, used in a MOVZ or MOVK instruction
    MO_G3 = 3,

    /// MO_G2 - A symbol operand with this flag (granule 2) represents the bits
    /// 32-47 of a 64-bit address, used in a MOVZ or MOVK instruction
    MO_G2 = 4,

    /// MO_G1 - A symbol operand with this flag (granule 1) represents the bits
    /// 16-31 of a 64-bit address, used in a MOVZ or MOVK instruction
    MO_G1 = 5,

    /// MO_G0 - A symbol operand with this flag (granule 0) represents the bits
    /// 0-15 of a 64-bit address, used in a MOVZ or MOVK instruction
    MO_G0 = 6,

    /// MO_GOT - This flag indicates that a symbol operand represents the
    /// address of the GOT entry for the symbol, rather than the address of
    /// the symbol itself.
    MO_GOT = 8,

    /// MO_NC - Indicates whether the linker is expected to check the symbol
    /// reference for overflow. For example in an ADRP/ADD pair of relocations
    /// the ADRP usually does check, but not the ADD.
    MO_NC = 0x10,

    /// MO_TLS - Indicates that the operand being accessed is some kind of
    /// thread-local symbol. On Darwin, only one type of thread-local access
    /// exists (pre linker-relaxation), but on ELF the TLSModel used for the
    /// referee will affect interpretation.
    MO_TLS = 0x20
  };
} // end namespace ARM64II

} // end namespace llvm

#endif
