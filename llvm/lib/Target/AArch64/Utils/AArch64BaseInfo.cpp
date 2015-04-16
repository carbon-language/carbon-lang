//===-- AArch64BaseInfo.cpp - AArch64 Base encoding information------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides basic encoding and assembly information for AArch64.
//
//===----------------------------------------------------------------------===//
#include "AArch64BaseInfo.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Regex.h"

using namespace llvm;

StringRef AArch64NamedImmMapper::toString(uint32_t Value, uint64_t FeatureBits,
                                          bool &Valid) const {
  for (unsigned i = 0; i < NumMappings; ++i) {
    if (Mappings[i].isValueEqual(Value, FeatureBits)) {
      Valid = true;
      return Mappings[i].Name;
    }
  }

  Valid = false;
  return StringRef();
}

uint32_t AArch64NamedImmMapper::fromString(StringRef Name, uint64_t FeatureBits,
                                           bool &Valid) const {
  std::string LowerCaseName = Name.lower();
  for (unsigned i = 0; i < NumMappings; ++i) {
    if (Mappings[i].isNameEqual(LowerCaseName, FeatureBits)) {
      Valid = true;
      return Mappings[i].Value;
    }
  }

  Valid = false;
  return -1;
}

bool AArch64NamedImmMapper::validImm(uint32_t Value) const {
  return Value < TooBigImm;
}

const AArch64NamedImmMapper::Mapping AArch64AT::ATMapper::ATMappings[] = {
  {"s1e1r", S1E1R, 0},
  {"s1e2r", S1E2R, 0},
  {"s1e3r", S1E3R, 0},
  {"s1e1w", S1E1W, 0},
  {"s1e2w", S1E2W, 0},
  {"s1e3w", S1E3W, 0},
  {"s1e0r", S1E0R, 0},
  {"s1e0w", S1E0W, 0},
  {"s12e1r", S12E1R, 0},
  {"s12e1w", S12E1W, 0},
  {"s12e0r", S12E0R, 0},
  {"s12e0w", S12E0W, 0},
};

AArch64AT::ATMapper::ATMapper()
  : AArch64NamedImmMapper(ATMappings, 0) {}

const AArch64NamedImmMapper::Mapping AArch64DB::DBarrierMapper::DBarrierMappings[] = {
  {"oshld", OSHLD, 0},
  {"oshst", OSHST, 0},
  {"osh", OSH, 0},
  {"nshld", NSHLD, 0},
  {"nshst", NSHST, 0},
  {"nsh", NSH, 0},
  {"ishld", ISHLD, 0},
  {"ishst", ISHST, 0},
  {"ish", ISH, 0},
  {"ld", LD, 0},
  {"st", ST, 0},
  {"sy", SY, 0}
};

AArch64DB::DBarrierMapper::DBarrierMapper()
  : AArch64NamedImmMapper(DBarrierMappings, 16u) {}

const AArch64NamedImmMapper::Mapping AArch64DC::DCMapper::DCMappings[] = {
  {"zva", ZVA, 0},
  {"ivac", IVAC, 0},
  {"isw", ISW, 0},
  {"cvac", CVAC, 0},
  {"csw", CSW, 0},
  {"cvau", CVAU, 0},
  {"civac", CIVAC, 0},
  {"cisw", CISW, 0}
};

AArch64DC::DCMapper::DCMapper()
  : AArch64NamedImmMapper(DCMappings, 0) {}

const AArch64NamedImmMapper::Mapping AArch64IC::ICMapper::ICMappings[] = {
  {"ialluis",  IALLUIS, 0},
  {"iallu", IALLU, 0},
  {"ivau", IVAU, 0}
};

AArch64IC::ICMapper::ICMapper()
  : AArch64NamedImmMapper(ICMappings, 0) {}

const AArch64NamedImmMapper::Mapping AArch64ISB::ISBMapper::ISBMappings[] = {
  {"sy",  SY, 0},
};

AArch64ISB::ISBMapper::ISBMapper()
  : AArch64NamedImmMapper(ISBMappings, 16) {}

const AArch64NamedImmMapper::Mapping AArch64PRFM::PRFMMapper::PRFMMappings[] = {
  {"pldl1keep", PLDL1KEEP, 0},
  {"pldl1strm", PLDL1STRM, 0},
  {"pldl2keep", PLDL2KEEP, 0},
  {"pldl2strm", PLDL2STRM, 0},
  {"pldl3keep", PLDL3KEEP, 0},
  {"pldl3strm", PLDL3STRM, 0},
  {"plil1keep", PLIL1KEEP, 0},
  {"plil1strm", PLIL1STRM, 0},
  {"plil2keep", PLIL2KEEP, 0},
  {"plil2strm", PLIL2STRM, 0},
  {"plil3keep", PLIL3KEEP, 0},
  {"plil3strm", PLIL3STRM, 0},
  {"pstl1keep", PSTL1KEEP, 0},
  {"pstl1strm", PSTL1STRM, 0},
  {"pstl2keep", PSTL2KEEP, 0},
  {"pstl2strm", PSTL2STRM, 0},
  {"pstl3keep", PSTL3KEEP, 0},
  {"pstl3strm", PSTL3STRM, 0}
};

AArch64PRFM::PRFMMapper::PRFMMapper()
  : AArch64NamedImmMapper(PRFMMappings, 32) {}

const AArch64NamedImmMapper::Mapping AArch64PState::PStateMapper::PStateMappings[] = {
  {"spsel", SPSel, 0},
  {"daifset", DAIFSet, 0},
  {"daifclr", DAIFClr, 0},

  // v8.1a "Privileged Access Never" extension-specific PStates
  {"pan", PAN, AArch64::HasV8_1aOps},
};

AArch64PState::PStateMapper::PStateMapper()
  : AArch64NamedImmMapper(PStateMappings, 0) {}

const AArch64NamedImmMapper::Mapping AArch64SysReg::MRSMapper::MRSMappings[] = {
  {"mdccsr_el0", MDCCSR_EL0, 0},
  {"dbgdtrrx_el0", DBGDTRRX_EL0, 0},
  {"mdrar_el1", MDRAR_EL1, 0},
  {"oslsr_el1", OSLSR_EL1, 0},
  {"dbgauthstatus_el1", DBGAUTHSTATUS_EL1, 0},
  {"pmceid0_el0", PMCEID0_EL0, 0},
  {"pmceid1_el0", PMCEID1_EL0, 0},
  {"midr_el1", MIDR_EL1, 0},
  {"ccsidr_el1", CCSIDR_EL1, 0},
  {"clidr_el1", CLIDR_EL1, 0},
  {"ctr_el0", CTR_EL0, 0},
  {"mpidr_el1", MPIDR_EL1, 0},
  {"revidr_el1", REVIDR_EL1, 0},
  {"aidr_el1", AIDR_EL1, 0},
  {"dczid_el0", DCZID_EL0, 0},
  {"id_pfr0_el1", ID_PFR0_EL1, 0},
  {"id_pfr1_el1", ID_PFR1_EL1, 0},
  {"id_dfr0_el1", ID_DFR0_EL1, 0},
  {"id_afr0_el1", ID_AFR0_EL1, 0},
  {"id_mmfr0_el1", ID_MMFR0_EL1, 0},
  {"id_mmfr1_el1", ID_MMFR1_EL1, 0},
  {"id_mmfr2_el1", ID_MMFR2_EL1, 0},
  {"id_mmfr3_el1", ID_MMFR3_EL1, 0},
  {"id_isar0_el1", ID_ISAR0_EL1, 0},
  {"id_isar1_el1", ID_ISAR1_EL1, 0},
  {"id_isar2_el1", ID_ISAR2_EL1, 0},
  {"id_isar3_el1", ID_ISAR3_EL1, 0},
  {"id_isar4_el1", ID_ISAR4_EL1, 0},
  {"id_isar5_el1", ID_ISAR5_EL1, 0},
  {"id_aa64pfr0_el1", ID_A64PFR0_EL1, 0},
  {"id_aa64pfr1_el1", ID_A64PFR1_EL1, 0},
  {"id_aa64dfr0_el1", ID_A64DFR0_EL1, 0},
  {"id_aa64dfr1_el1", ID_A64DFR1_EL1, 0},
  {"id_aa64afr0_el1", ID_A64AFR0_EL1, 0},
  {"id_aa64afr1_el1", ID_A64AFR1_EL1, 0},
  {"id_aa64isar0_el1", ID_A64ISAR0_EL1, 0},
  {"id_aa64isar1_el1", ID_A64ISAR1_EL1, 0},
  {"id_aa64mmfr0_el1", ID_A64MMFR0_EL1, 0},
  {"id_aa64mmfr1_el1", ID_A64MMFR1_EL1, 0},
  {"mvfr0_el1", MVFR0_EL1, 0},
  {"mvfr1_el1", MVFR1_EL1, 0},
  {"mvfr2_el1", MVFR2_EL1, 0},
  {"rvbar_el1", RVBAR_EL1, 0},
  {"rvbar_el2", RVBAR_EL2, 0},
  {"rvbar_el3", RVBAR_EL3, 0},
  {"isr_el1", ISR_EL1, 0},
  {"cntpct_el0", CNTPCT_EL0, 0},
  {"cntvct_el0", CNTVCT_EL0, 0},

  // Trace registers
  {"trcstatr", TRCSTATR, 0},
  {"trcidr8", TRCIDR8, 0},
  {"trcidr9", TRCIDR9, 0},
  {"trcidr10", TRCIDR10, 0},
  {"trcidr11", TRCIDR11, 0},
  {"trcidr12", TRCIDR12, 0},
  {"trcidr13", TRCIDR13, 0},
  {"trcidr0", TRCIDR0, 0},
  {"trcidr1", TRCIDR1, 0},
  {"trcidr2", TRCIDR2, 0},
  {"trcidr3", TRCIDR3, 0},
  {"trcidr4", TRCIDR4, 0},
  {"trcidr5", TRCIDR5, 0},
  {"trcidr6", TRCIDR6, 0},
  {"trcidr7", TRCIDR7, 0},
  {"trcoslsr", TRCOSLSR, 0},
  {"trcpdsr", TRCPDSR, 0},
  {"trcdevaff0", TRCDEVAFF0, 0},
  {"trcdevaff1", TRCDEVAFF1, 0},
  {"trclsr", TRCLSR, 0},
  {"trcauthstatus", TRCAUTHSTATUS, 0},
  {"trcdevarch", TRCDEVARCH, 0},
  {"trcdevid", TRCDEVID, 0},
  {"trcdevtype", TRCDEVTYPE, 0},
  {"trcpidr4", TRCPIDR4, 0},
  {"trcpidr5", TRCPIDR5, 0},
  {"trcpidr6", TRCPIDR6, 0},
  {"trcpidr7", TRCPIDR7, 0},
  {"trcpidr0", TRCPIDR0, 0},
  {"trcpidr1", TRCPIDR1, 0},
  {"trcpidr2", TRCPIDR2, 0},
  {"trcpidr3", TRCPIDR3, 0},
  {"trccidr0", TRCCIDR0, 0},
  {"trccidr1", TRCCIDR1, 0},
  {"trccidr2", TRCCIDR2, 0},
  {"trccidr3", TRCCIDR3, 0},

  // GICv3 registers
  {"icc_iar1_el1", ICC_IAR1_EL1, 0},
  {"icc_iar0_el1", ICC_IAR0_EL1, 0},
  {"icc_hppir1_el1", ICC_HPPIR1_EL1, 0},
  {"icc_hppir0_el1", ICC_HPPIR0_EL1, 0},
  {"icc_rpr_el1", ICC_RPR_EL1, 0},
  {"ich_vtr_el2", ICH_VTR_EL2, 0},
  {"ich_eisr_el2", ICH_EISR_EL2, 0},
  {"ich_elsr_el2", ICH_ELSR_EL2, 0}
};

AArch64SysReg::MRSMapper::MRSMapper() {
    InstMappings = &MRSMappings[0];
    NumInstMappings = llvm::array_lengthof(MRSMappings);
}

const AArch64NamedImmMapper::Mapping AArch64SysReg::MSRMapper::MSRMappings[] = {
  {"dbgdtrtx_el0", DBGDTRTX_EL0, 0},
  {"oslar_el1", OSLAR_EL1, 0},
  {"pmswinc_el0", PMSWINC_EL0, 0},

  // Trace registers
  {"trcoslar", TRCOSLAR, 0},
  {"trclar", TRCLAR, 0},

  // GICv3 registers
  {"icc_eoir1_el1", ICC_EOIR1_EL1, 0},
  {"icc_eoir0_el1", ICC_EOIR0_EL1, 0},
  {"icc_dir_el1", ICC_DIR_EL1, 0},
  {"icc_sgi1r_el1", ICC_SGI1R_EL1, 0},
  {"icc_asgi1r_el1", ICC_ASGI1R_EL1, 0},
  {"icc_sgi0r_el1", ICC_SGI0R_EL1, 0},

  // v8.1a "Privileged Access Never" extension-specific system registers
  {"pan", PAN, AArch64::HasV8_1aOps},
};

AArch64SysReg::MSRMapper::MSRMapper() {
    InstMappings = &MSRMappings[0];
    NumInstMappings = llvm::array_lengthof(MSRMappings);
}


const AArch64NamedImmMapper::Mapping AArch64SysReg::SysRegMapper::SysRegMappings[] = {
  {"osdtrrx_el1", OSDTRRX_EL1, 0},
  {"osdtrtx_el1",  OSDTRTX_EL1, 0},
  {"teecr32_el1", TEECR32_EL1, 0},
  {"mdccint_el1", MDCCINT_EL1, 0},
  {"mdscr_el1", MDSCR_EL1, 0},
  {"dbgdtr_el0", DBGDTR_EL0, 0},
  {"oseccr_el1", OSECCR_EL1, 0},
  {"dbgvcr32_el2", DBGVCR32_EL2, 0},
  {"dbgbvr0_el1", DBGBVR0_EL1, 0},
  {"dbgbvr1_el1", DBGBVR1_EL1, 0},
  {"dbgbvr2_el1", DBGBVR2_EL1, 0},
  {"dbgbvr3_el1", DBGBVR3_EL1, 0},
  {"dbgbvr4_el1", DBGBVR4_EL1, 0},
  {"dbgbvr5_el1", DBGBVR5_EL1, 0},
  {"dbgbvr6_el1", DBGBVR6_EL1, 0},
  {"dbgbvr7_el1", DBGBVR7_EL1, 0},
  {"dbgbvr8_el1", DBGBVR8_EL1, 0},
  {"dbgbvr9_el1", DBGBVR9_EL1, 0},
  {"dbgbvr10_el1", DBGBVR10_EL1, 0},
  {"dbgbvr11_el1", DBGBVR11_EL1, 0},
  {"dbgbvr12_el1", DBGBVR12_EL1, 0},
  {"dbgbvr13_el1", DBGBVR13_EL1, 0},
  {"dbgbvr14_el1", DBGBVR14_EL1, 0},
  {"dbgbvr15_el1", DBGBVR15_EL1, 0},
  {"dbgbcr0_el1", DBGBCR0_EL1, 0},
  {"dbgbcr1_el1", DBGBCR1_EL1, 0},
  {"dbgbcr2_el1", DBGBCR2_EL1, 0},
  {"dbgbcr3_el1", DBGBCR3_EL1, 0},
  {"dbgbcr4_el1", DBGBCR4_EL1, 0},
  {"dbgbcr5_el1", DBGBCR5_EL1, 0},
  {"dbgbcr6_el1", DBGBCR6_EL1, 0},
  {"dbgbcr7_el1", DBGBCR7_EL1, 0},
  {"dbgbcr8_el1", DBGBCR8_EL1, 0},
  {"dbgbcr9_el1", DBGBCR9_EL1, 0},
  {"dbgbcr10_el1", DBGBCR10_EL1, 0},
  {"dbgbcr11_el1", DBGBCR11_EL1, 0},
  {"dbgbcr12_el1", DBGBCR12_EL1, 0},
  {"dbgbcr13_el1", DBGBCR13_EL1, 0},
  {"dbgbcr14_el1", DBGBCR14_EL1, 0},
  {"dbgbcr15_el1", DBGBCR15_EL1, 0},
  {"dbgwvr0_el1", DBGWVR0_EL1, 0},
  {"dbgwvr1_el1", DBGWVR1_EL1, 0},
  {"dbgwvr2_el1", DBGWVR2_EL1, 0},
  {"dbgwvr3_el1", DBGWVR3_EL1, 0},
  {"dbgwvr4_el1", DBGWVR4_EL1, 0},
  {"dbgwvr5_el1", DBGWVR5_EL1, 0},
  {"dbgwvr6_el1", DBGWVR6_EL1, 0},
  {"dbgwvr7_el1", DBGWVR7_EL1, 0},
  {"dbgwvr8_el1", DBGWVR8_EL1, 0},
  {"dbgwvr9_el1", DBGWVR9_EL1, 0},
  {"dbgwvr10_el1", DBGWVR10_EL1, 0},
  {"dbgwvr11_el1", DBGWVR11_EL1, 0},
  {"dbgwvr12_el1", DBGWVR12_EL1, 0},
  {"dbgwvr13_el1", DBGWVR13_EL1, 0},
  {"dbgwvr14_el1", DBGWVR14_EL1, 0},
  {"dbgwvr15_el1", DBGWVR15_EL1, 0},
  {"dbgwcr0_el1", DBGWCR0_EL1, 0},
  {"dbgwcr1_el1", DBGWCR1_EL1, 0},
  {"dbgwcr2_el1", DBGWCR2_EL1, 0},
  {"dbgwcr3_el1", DBGWCR3_EL1, 0},
  {"dbgwcr4_el1", DBGWCR4_EL1, 0},
  {"dbgwcr5_el1", DBGWCR5_EL1, 0},
  {"dbgwcr6_el1", DBGWCR6_EL1, 0},
  {"dbgwcr7_el1", DBGWCR7_EL1, 0},
  {"dbgwcr8_el1", DBGWCR8_EL1, 0},
  {"dbgwcr9_el1", DBGWCR9_EL1, 0},
  {"dbgwcr10_el1", DBGWCR10_EL1, 0},
  {"dbgwcr11_el1", DBGWCR11_EL1, 0},
  {"dbgwcr12_el1", DBGWCR12_EL1, 0},
  {"dbgwcr13_el1", DBGWCR13_EL1, 0},
  {"dbgwcr14_el1", DBGWCR14_EL1, 0},
  {"dbgwcr15_el1", DBGWCR15_EL1, 0},
  {"teehbr32_el1", TEEHBR32_EL1, 0},
  {"osdlr_el1", OSDLR_EL1, 0},
  {"dbgprcr_el1", DBGPRCR_EL1, 0},
  {"dbgclaimset_el1", DBGCLAIMSET_EL1, 0},
  {"dbgclaimclr_el1", DBGCLAIMCLR_EL1, 0},
  {"csselr_el1", CSSELR_EL1, 0},
  {"vpidr_el2", VPIDR_EL2, 0},
  {"vmpidr_el2", VMPIDR_EL2, 0},
  {"sctlr_el1", SCTLR_EL1, 0},
  {"sctlr_el2", SCTLR_EL2, 0},
  {"sctlr_el3", SCTLR_EL3, 0},
  {"actlr_el1", ACTLR_EL1, 0},
  {"actlr_el2", ACTLR_EL2, 0},
  {"actlr_el3", ACTLR_EL3, 0},
  {"cpacr_el1", CPACR_EL1, 0},
  {"hcr_el2", HCR_EL2, 0},
  {"scr_el3", SCR_EL3, 0},
  {"mdcr_el2", MDCR_EL2, 0},
  {"sder32_el3", SDER32_EL3, 0},
  {"cptr_el2", CPTR_EL2, 0},
  {"cptr_el3", CPTR_EL3, 0},
  {"hstr_el2", HSTR_EL2, 0},
  {"hacr_el2", HACR_EL2, 0},
  {"mdcr_el3", MDCR_EL3, 0},
  {"ttbr0_el1", TTBR0_EL1, 0},
  {"ttbr0_el2", TTBR0_EL2, 0},
  {"ttbr0_el3", TTBR0_EL3, 0},
  {"ttbr1_el1", TTBR1_EL1, 0},
  {"tcr_el1", TCR_EL1, 0},
  {"tcr_el2", TCR_EL2, 0},
  {"tcr_el3", TCR_EL3, 0},
  {"vttbr_el2", VTTBR_EL2, 0},
  {"vtcr_el2", VTCR_EL2, 0},
  {"dacr32_el2", DACR32_EL2, 0},
  {"spsr_el1", SPSR_EL1, 0},
  {"spsr_el2", SPSR_EL2, 0},
  {"spsr_el3", SPSR_EL3, 0},
  {"elr_el1", ELR_EL1, 0},
  {"elr_el2", ELR_EL2, 0},
  {"elr_el3", ELR_EL3, 0},
  {"sp_el0", SP_EL0, 0},
  {"sp_el1", SP_EL1, 0},
  {"sp_el2", SP_EL2, 0},
  {"spsel", SPSel, 0},
  {"nzcv", NZCV, 0},
  {"daif", DAIF, 0},
  {"currentel", CurrentEL, 0},
  {"spsr_irq", SPSR_irq, 0},
  {"spsr_abt", SPSR_abt, 0},
  {"spsr_und", SPSR_und, 0},
  {"spsr_fiq", SPSR_fiq, 0},
  {"fpcr", FPCR, 0},
  {"fpsr", FPSR, 0},
  {"dspsr_el0", DSPSR_EL0, 0},
  {"dlr_el0", DLR_EL0, 0},
  {"ifsr32_el2", IFSR32_EL2, 0},
  {"afsr0_el1", AFSR0_EL1, 0},
  {"afsr0_el2", AFSR0_EL2, 0},
  {"afsr0_el3", AFSR0_EL3, 0},
  {"afsr1_el1", AFSR1_EL1, 0},
  {"afsr1_el2", AFSR1_EL2, 0},
  {"afsr1_el3", AFSR1_EL3, 0},
  {"esr_el1", ESR_EL1, 0},
  {"esr_el2", ESR_EL2, 0},
  {"esr_el3", ESR_EL3, 0},
  {"fpexc32_el2", FPEXC32_EL2, 0},
  {"far_el1", FAR_EL1, 0},
  {"far_el2", FAR_EL2, 0},
  {"far_el3", FAR_EL3, 0},
  {"hpfar_el2", HPFAR_EL2, 0},
  {"par_el1", PAR_EL1, 0},
  {"pmcr_el0", PMCR_EL0, 0},
  {"pmcntenset_el0", PMCNTENSET_EL0, 0},
  {"pmcntenclr_el0", PMCNTENCLR_EL0, 0},
  {"pmovsclr_el0", PMOVSCLR_EL0, 0},
  {"pmselr_el0", PMSELR_EL0, 0},
  {"pmccntr_el0", PMCCNTR_EL0, 0},
  {"pmxevtyper_el0", PMXEVTYPER_EL0, 0},
  {"pmxevcntr_el0", PMXEVCNTR_EL0, 0},
  {"pmuserenr_el0", PMUSERENR_EL0, 0},
  {"pmintenset_el1", PMINTENSET_EL1, 0},
  {"pmintenclr_el1", PMINTENCLR_EL1, 0},
  {"pmovsset_el0", PMOVSSET_EL0, 0},
  {"mair_el1", MAIR_EL1, 0},
  {"mair_el2", MAIR_EL2, 0},
  {"mair_el3", MAIR_EL3, 0},
  {"amair_el1", AMAIR_EL1, 0},
  {"amair_el2", AMAIR_EL2, 0},
  {"amair_el3", AMAIR_EL3, 0},
  {"vbar_el1", VBAR_EL1, 0},
  {"vbar_el2", VBAR_EL2, 0},
  {"vbar_el3", VBAR_EL3, 0},
  {"rmr_el1", RMR_EL1, 0},
  {"rmr_el2", RMR_EL2, 0},
  {"rmr_el3", RMR_EL3, 0},
  {"contextidr_el1", CONTEXTIDR_EL1, 0},
  {"tpidr_el0", TPIDR_EL0, 0},
  {"tpidr_el2", TPIDR_EL2, 0},
  {"tpidr_el3", TPIDR_EL3, 0},
  {"tpidrro_el0", TPIDRRO_EL0, 0},
  {"tpidr_el1", TPIDR_EL1, 0},
  {"cntfrq_el0", CNTFRQ_EL0, 0},
  {"cntvoff_el2", CNTVOFF_EL2, 0},
  {"cntkctl_el1", CNTKCTL_EL1, 0},
  {"cnthctl_el2", CNTHCTL_EL2, 0},
  {"cntp_tval_el0", CNTP_TVAL_EL0, 0},
  {"cnthp_tval_el2", CNTHP_TVAL_EL2, 0},
  {"cntps_tval_el1", CNTPS_TVAL_EL1, 0},
  {"cntp_ctl_el0", CNTP_CTL_EL0, 0},
  {"cnthp_ctl_el2", CNTHP_CTL_EL2, 0},
  {"cntps_ctl_el1", CNTPS_CTL_EL1, 0},
  {"cntp_cval_el0", CNTP_CVAL_EL0, 0},
  {"cnthp_cval_el2", CNTHP_CVAL_EL2, 0},
  {"cntps_cval_el1", CNTPS_CVAL_EL1, 0},
  {"cntv_tval_el0", CNTV_TVAL_EL0, 0},
  {"cntv_ctl_el0", CNTV_CTL_EL0, 0},
  {"cntv_cval_el0", CNTV_CVAL_EL0, 0},
  {"pmevcntr0_el0", PMEVCNTR0_EL0, 0},
  {"pmevcntr1_el0", PMEVCNTR1_EL0, 0},
  {"pmevcntr2_el0", PMEVCNTR2_EL0, 0},
  {"pmevcntr3_el0", PMEVCNTR3_EL0, 0},
  {"pmevcntr4_el0", PMEVCNTR4_EL0, 0},
  {"pmevcntr5_el0", PMEVCNTR5_EL0, 0},
  {"pmevcntr6_el0", PMEVCNTR6_EL0, 0},
  {"pmevcntr7_el0", PMEVCNTR7_EL0, 0},
  {"pmevcntr8_el0", PMEVCNTR8_EL0, 0},
  {"pmevcntr9_el0", PMEVCNTR9_EL0, 0},
  {"pmevcntr10_el0", PMEVCNTR10_EL0, 0},
  {"pmevcntr11_el0", PMEVCNTR11_EL0, 0},
  {"pmevcntr12_el0", PMEVCNTR12_EL0, 0},
  {"pmevcntr13_el0", PMEVCNTR13_EL0, 0},
  {"pmevcntr14_el0", PMEVCNTR14_EL0, 0},
  {"pmevcntr15_el0", PMEVCNTR15_EL0, 0},
  {"pmevcntr16_el0", PMEVCNTR16_EL0, 0},
  {"pmevcntr17_el0", PMEVCNTR17_EL0, 0},
  {"pmevcntr18_el0", PMEVCNTR18_EL0, 0},
  {"pmevcntr19_el0", PMEVCNTR19_EL0, 0},
  {"pmevcntr20_el0", PMEVCNTR20_EL0, 0},
  {"pmevcntr21_el0", PMEVCNTR21_EL0, 0},
  {"pmevcntr22_el0", PMEVCNTR22_EL0, 0},
  {"pmevcntr23_el0", PMEVCNTR23_EL0, 0},
  {"pmevcntr24_el0", PMEVCNTR24_EL0, 0},
  {"pmevcntr25_el0", PMEVCNTR25_EL0, 0},
  {"pmevcntr26_el0", PMEVCNTR26_EL0, 0},
  {"pmevcntr27_el0", PMEVCNTR27_EL0, 0},
  {"pmevcntr28_el0", PMEVCNTR28_EL0, 0},
  {"pmevcntr29_el0", PMEVCNTR29_EL0, 0},
  {"pmevcntr30_el0", PMEVCNTR30_EL0, 0},
  {"pmccfiltr_el0", PMCCFILTR_EL0, 0},
  {"pmevtyper0_el0", PMEVTYPER0_EL0, 0},
  {"pmevtyper1_el0", PMEVTYPER1_EL0, 0},
  {"pmevtyper2_el0", PMEVTYPER2_EL0, 0},
  {"pmevtyper3_el0", PMEVTYPER3_EL0, 0},
  {"pmevtyper4_el0", PMEVTYPER4_EL0, 0},
  {"pmevtyper5_el0", PMEVTYPER5_EL0, 0},
  {"pmevtyper6_el0", PMEVTYPER6_EL0, 0},
  {"pmevtyper7_el0", PMEVTYPER7_EL0, 0},
  {"pmevtyper8_el0", PMEVTYPER8_EL0, 0},
  {"pmevtyper9_el0", PMEVTYPER9_EL0, 0},
  {"pmevtyper10_el0", PMEVTYPER10_EL0, 0},
  {"pmevtyper11_el0", PMEVTYPER11_EL0, 0},
  {"pmevtyper12_el0", PMEVTYPER12_EL0, 0},
  {"pmevtyper13_el0", PMEVTYPER13_EL0, 0},
  {"pmevtyper14_el0", PMEVTYPER14_EL0, 0},
  {"pmevtyper15_el0", PMEVTYPER15_EL0, 0},
  {"pmevtyper16_el0", PMEVTYPER16_EL0, 0},
  {"pmevtyper17_el0", PMEVTYPER17_EL0, 0},
  {"pmevtyper18_el0", PMEVTYPER18_EL0, 0},
  {"pmevtyper19_el0", PMEVTYPER19_EL0, 0},
  {"pmevtyper20_el0", PMEVTYPER20_EL0, 0},
  {"pmevtyper21_el0", PMEVTYPER21_EL0, 0},
  {"pmevtyper22_el0", PMEVTYPER22_EL0, 0},
  {"pmevtyper23_el0", PMEVTYPER23_EL0, 0},
  {"pmevtyper24_el0", PMEVTYPER24_EL0, 0},
  {"pmevtyper25_el0", PMEVTYPER25_EL0, 0},
  {"pmevtyper26_el0", PMEVTYPER26_EL0, 0},
  {"pmevtyper27_el0", PMEVTYPER27_EL0, 0},
  {"pmevtyper28_el0", PMEVTYPER28_EL0, 0},
  {"pmevtyper29_el0", PMEVTYPER29_EL0, 0},
  {"pmevtyper30_el0", PMEVTYPER30_EL0, 0},

  // Trace registers
  {"trcprgctlr", TRCPRGCTLR, 0},
  {"trcprocselr", TRCPROCSELR, 0},
  {"trcconfigr", TRCCONFIGR, 0},
  {"trcauxctlr", TRCAUXCTLR, 0},
  {"trceventctl0r", TRCEVENTCTL0R, 0},
  {"trceventctl1r", TRCEVENTCTL1R, 0},
  {"trcstallctlr", TRCSTALLCTLR, 0},
  {"trctsctlr", TRCTSCTLR, 0},
  {"trcsyncpr", TRCSYNCPR, 0},
  {"trcccctlr", TRCCCCTLR, 0},
  {"trcbbctlr", TRCBBCTLR, 0},
  {"trctraceidr", TRCTRACEIDR, 0},
  {"trcqctlr", TRCQCTLR, 0},
  {"trcvictlr", TRCVICTLR, 0},
  {"trcviiectlr", TRCVIIECTLR, 0},
  {"trcvissctlr", TRCVISSCTLR, 0},
  {"trcvipcssctlr", TRCVIPCSSCTLR, 0},
  {"trcvdctlr", TRCVDCTLR, 0},
  {"trcvdsacctlr", TRCVDSACCTLR, 0},
  {"trcvdarcctlr", TRCVDARCCTLR, 0},
  {"trcseqevr0", TRCSEQEVR0, 0},
  {"trcseqevr1", TRCSEQEVR1, 0},
  {"trcseqevr2", TRCSEQEVR2, 0},
  {"trcseqrstevr", TRCSEQRSTEVR, 0},
  {"trcseqstr", TRCSEQSTR, 0},
  {"trcextinselr", TRCEXTINSELR, 0},
  {"trccntrldvr0", TRCCNTRLDVR0, 0},
  {"trccntrldvr1", TRCCNTRLDVR1, 0},
  {"trccntrldvr2", TRCCNTRLDVR2, 0},
  {"trccntrldvr3", TRCCNTRLDVR3, 0},
  {"trccntctlr0", TRCCNTCTLR0, 0},
  {"trccntctlr1", TRCCNTCTLR1, 0},
  {"trccntctlr2", TRCCNTCTLR2, 0},
  {"trccntctlr3", TRCCNTCTLR3, 0},
  {"trccntvr0", TRCCNTVR0, 0},
  {"trccntvr1", TRCCNTVR1, 0},
  {"trccntvr2", TRCCNTVR2, 0},
  {"trccntvr3", TRCCNTVR3, 0},
  {"trcimspec0", TRCIMSPEC0, 0},
  {"trcimspec1", TRCIMSPEC1, 0},
  {"trcimspec2", TRCIMSPEC2, 0},
  {"trcimspec3", TRCIMSPEC3, 0},
  {"trcimspec4", TRCIMSPEC4, 0},
  {"trcimspec5", TRCIMSPEC5, 0},
  {"trcimspec6", TRCIMSPEC6, 0},
  {"trcimspec7", TRCIMSPEC7, 0},
  {"trcrsctlr2", TRCRSCTLR2, 0},
  {"trcrsctlr3", TRCRSCTLR3, 0},
  {"trcrsctlr4", TRCRSCTLR4, 0},
  {"trcrsctlr5", TRCRSCTLR5, 0},
  {"trcrsctlr6", TRCRSCTLR6, 0},
  {"trcrsctlr7", TRCRSCTLR7, 0},
  {"trcrsctlr8", TRCRSCTLR8, 0},
  {"trcrsctlr9", TRCRSCTLR9, 0},
  {"trcrsctlr10", TRCRSCTLR10, 0},
  {"trcrsctlr11", TRCRSCTLR11, 0},
  {"trcrsctlr12", TRCRSCTLR12, 0},
  {"trcrsctlr13", TRCRSCTLR13, 0},
  {"trcrsctlr14", TRCRSCTLR14, 0},
  {"trcrsctlr15", TRCRSCTLR15, 0},
  {"trcrsctlr16", TRCRSCTLR16, 0},
  {"trcrsctlr17", TRCRSCTLR17, 0},
  {"trcrsctlr18", TRCRSCTLR18, 0},
  {"trcrsctlr19", TRCRSCTLR19, 0},
  {"trcrsctlr20", TRCRSCTLR20, 0},
  {"trcrsctlr21", TRCRSCTLR21, 0},
  {"trcrsctlr22", TRCRSCTLR22, 0},
  {"trcrsctlr23", TRCRSCTLR23, 0},
  {"trcrsctlr24", TRCRSCTLR24, 0},
  {"trcrsctlr25", TRCRSCTLR25, 0},
  {"trcrsctlr26", TRCRSCTLR26, 0},
  {"trcrsctlr27", TRCRSCTLR27, 0},
  {"trcrsctlr28", TRCRSCTLR28, 0},
  {"trcrsctlr29", TRCRSCTLR29, 0},
  {"trcrsctlr30", TRCRSCTLR30, 0},
  {"trcrsctlr31", TRCRSCTLR31, 0},
  {"trcssccr0", TRCSSCCR0, 0},
  {"trcssccr1", TRCSSCCR1, 0},
  {"trcssccr2", TRCSSCCR2, 0},
  {"trcssccr3", TRCSSCCR3, 0},
  {"trcssccr4", TRCSSCCR4, 0},
  {"trcssccr5", TRCSSCCR5, 0},
  {"trcssccr6", TRCSSCCR6, 0},
  {"trcssccr7", TRCSSCCR7, 0},
  {"trcsscsr0", TRCSSCSR0, 0},
  {"trcsscsr1", TRCSSCSR1, 0},
  {"trcsscsr2", TRCSSCSR2, 0},
  {"trcsscsr3", TRCSSCSR3, 0},
  {"trcsscsr4", TRCSSCSR4, 0},
  {"trcsscsr5", TRCSSCSR5, 0},
  {"trcsscsr6", TRCSSCSR6, 0},
  {"trcsscsr7", TRCSSCSR7, 0},
  {"trcsspcicr0", TRCSSPCICR0, 0},
  {"trcsspcicr1", TRCSSPCICR1, 0},
  {"trcsspcicr2", TRCSSPCICR2, 0},
  {"trcsspcicr3", TRCSSPCICR3, 0},
  {"trcsspcicr4", TRCSSPCICR4, 0},
  {"trcsspcicr5", TRCSSPCICR5, 0},
  {"trcsspcicr6", TRCSSPCICR6, 0},
  {"trcsspcicr7", TRCSSPCICR7, 0},
  {"trcpdcr", TRCPDCR, 0},
  {"trcacvr0", TRCACVR0, 0},
  {"trcacvr1", TRCACVR1, 0},
  {"trcacvr2", TRCACVR2, 0},
  {"trcacvr3", TRCACVR3, 0},
  {"trcacvr4", TRCACVR4, 0},
  {"trcacvr5", TRCACVR5, 0},
  {"trcacvr6", TRCACVR6, 0},
  {"trcacvr7", TRCACVR7, 0},
  {"trcacvr8", TRCACVR8, 0},
  {"trcacvr9", TRCACVR9, 0},
  {"trcacvr10", TRCACVR10, 0},
  {"trcacvr11", TRCACVR11, 0},
  {"trcacvr12", TRCACVR12, 0},
  {"trcacvr13", TRCACVR13, 0},
  {"trcacvr14", TRCACVR14, 0},
  {"trcacvr15", TRCACVR15, 0},
  {"trcacatr0", TRCACATR0, 0},
  {"trcacatr1", TRCACATR1, 0},
  {"trcacatr2", TRCACATR2, 0},
  {"trcacatr3", TRCACATR3, 0},
  {"trcacatr4", TRCACATR4, 0},
  {"trcacatr5", TRCACATR5, 0},
  {"trcacatr6", TRCACATR6, 0},
  {"trcacatr7", TRCACATR7, 0},
  {"trcacatr8", TRCACATR8, 0},
  {"trcacatr9", TRCACATR9, 0},
  {"trcacatr10", TRCACATR10, 0},
  {"trcacatr11", TRCACATR11, 0},
  {"trcacatr12", TRCACATR12, 0},
  {"trcacatr13", TRCACATR13, 0},
  {"trcacatr14", TRCACATR14, 0},
  {"trcacatr15", TRCACATR15, 0},
  {"trcdvcvr0", TRCDVCVR0, 0},
  {"trcdvcvr1", TRCDVCVR1, 0},
  {"trcdvcvr2", TRCDVCVR2, 0},
  {"trcdvcvr3", TRCDVCVR3, 0},
  {"trcdvcvr4", TRCDVCVR4, 0},
  {"trcdvcvr5", TRCDVCVR5, 0},
  {"trcdvcvr6", TRCDVCVR6, 0},
  {"trcdvcvr7", TRCDVCVR7, 0},
  {"trcdvcmr0", TRCDVCMR0, 0},
  {"trcdvcmr1", TRCDVCMR1, 0},
  {"trcdvcmr2", TRCDVCMR2, 0},
  {"trcdvcmr3", TRCDVCMR3, 0},
  {"trcdvcmr4", TRCDVCMR4, 0},
  {"trcdvcmr5", TRCDVCMR5, 0},
  {"trcdvcmr6", TRCDVCMR6, 0},
  {"trcdvcmr7", TRCDVCMR7, 0},
  {"trccidcvr0", TRCCIDCVR0, 0},
  {"trccidcvr1", TRCCIDCVR1, 0},
  {"trccidcvr2", TRCCIDCVR2, 0},
  {"trccidcvr3", TRCCIDCVR3, 0},
  {"trccidcvr4", TRCCIDCVR4, 0},
  {"trccidcvr5", TRCCIDCVR5, 0},
  {"trccidcvr6", TRCCIDCVR6, 0},
  {"trccidcvr7", TRCCIDCVR7, 0},
  {"trcvmidcvr0", TRCVMIDCVR0, 0},
  {"trcvmidcvr1", TRCVMIDCVR1, 0},
  {"trcvmidcvr2", TRCVMIDCVR2, 0},
  {"trcvmidcvr3", TRCVMIDCVR3, 0},
  {"trcvmidcvr4", TRCVMIDCVR4, 0},
  {"trcvmidcvr5", TRCVMIDCVR5, 0},
  {"trcvmidcvr6", TRCVMIDCVR6, 0},
  {"trcvmidcvr7", TRCVMIDCVR7, 0},
  {"trccidcctlr0", TRCCIDCCTLR0, 0},
  {"trccidcctlr1", TRCCIDCCTLR1, 0},
  {"trcvmidcctlr0", TRCVMIDCCTLR0, 0},
  {"trcvmidcctlr1", TRCVMIDCCTLR1, 0},
  {"trcitctrl", TRCITCTRL, 0},
  {"trcclaimset", TRCCLAIMSET, 0},
  {"trcclaimclr", TRCCLAIMCLR, 0},

  // GICv3 registers
  {"icc_bpr1_el1", ICC_BPR1_EL1, 0},
  {"icc_bpr0_el1", ICC_BPR0_EL1, 0},
  {"icc_pmr_el1", ICC_PMR_EL1, 0},
  {"icc_ctlr_el1", ICC_CTLR_EL1, 0},
  {"icc_ctlr_el3", ICC_CTLR_EL3, 0},
  {"icc_sre_el1", ICC_SRE_EL1, 0},
  {"icc_sre_el2", ICC_SRE_EL2, 0},
  {"icc_sre_el3", ICC_SRE_EL3, 0},
  {"icc_igrpen0_el1", ICC_IGRPEN0_EL1, 0},
  {"icc_igrpen1_el1", ICC_IGRPEN1_EL1, 0},
  {"icc_igrpen1_el3", ICC_IGRPEN1_EL3, 0},
  {"icc_seien_el1", ICC_SEIEN_EL1, 0},
  {"icc_ap0r0_el1", ICC_AP0R0_EL1, 0},
  {"icc_ap0r1_el1", ICC_AP0R1_EL1, 0},
  {"icc_ap0r2_el1", ICC_AP0R2_EL1, 0},
  {"icc_ap0r3_el1", ICC_AP0R3_EL1, 0},
  {"icc_ap1r0_el1", ICC_AP1R0_EL1, 0},
  {"icc_ap1r1_el1", ICC_AP1R1_EL1, 0},
  {"icc_ap1r2_el1", ICC_AP1R2_EL1, 0},
  {"icc_ap1r3_el1", ICC_AP1R3_EL1, 0},
  {"ich_ap0r0_el2", ICH_AP0R0_EL2, 0},
  {"ich_ap0r1_el2", ICH_AP0R1_EL2, 0},
  {"ich_ap0r2_el2", ICH_AP0R2_EL2, 0},
  {"ich_ap0r3_el2", ICH_AP0R3_EL2, 0},
  {"ich_ap1r0_el2", ICH_AP1R0_EL2, 0},
  {"ich_ap1r1_el2", ICH_AP1R1_EL2, 0},
  {"ich_ap1r2_el2", ICH_AP1R2_EL2, 0},
  {"ich_ap1r3_el2", ICH_AP1R3_EL2, 0},
  {"ich_hcr_el2", ICH_HCR_EL2, 0},
  {"ich_misr_el2", ICH_MISR_EL2, 0},
  {"ich_vmcr_el2", ICH_VMCR_EL2, 0},
  {"ich_vseir_el2", ICH_VSEIR_EL2, 0},
  {"ich_lr0_el2", ICH_LR0_EL2, 0},
  {"ich_lr1_el2", ICH_LR1_EL2, 0},
  {"ich_lr2_el2", ICH_LR2_EL2, 0},
  {"ich_lr3_el2", ICH_LR3_EL2, 0},
  {"ich_lr4_el2", ICH_LR4_EL2, 0},
  {"ich_lr5_el2", ICH_LR5_EL2, 0},
  {"ich_lr6_el2", ICH_LR6_EL2, 0},
  {"ich_lr7_el2", ICH_LR7_EL2, 0},
  {"ich_lr8_el2", ICH_LR8_EL2, 0},
  {"ich_lr9_el2", ICH_LR9_EL2, 0},
  {"ich_lr10_el2", ICH_LR10_EL2, 0},
  {"ich_lr11_el2", ICH_LR11_EL2, 0},
  {"ich_lr12_el2", ICH_LR12_EL2, 0},
  {"ich_lr13_el2", ICH_LR13_EL2, 0},
  {"ich_lr14_el2", ICH_LR14_EL2, 0},
  {"ich_lr15_el2", ICH_LR15_EL2, 0},

  // Cyclone registers
  {"cpm_ioacc_ctl_el3", CPM_IOACC_CTL_EL3, AArch64::ProcCyclone},

  // v8.1a "Privileged Access Never" extension-specific system registers
  {"pan", PAN, AArch64::HasV8_1aOps},

  // v8.1a "Limited Ordering Regions" extension-specific system registers
  {"lorsa_el1", LORSA_EL1, AArch64::HasV8_1aOps},
  {"lorea_el1", LOREA_EL1, AArch64::HasV8_1aOps},
  {"lorn_el1", LORN_EL1, AArch64::HasV8_1aOps},
  {"lorc_el1", LORC_EL1, AArch64::HasV8_1aOps},
  {"lorid_el1", LORID_EL1, AArch64::HasV8_1aOps},

  // v8.1a "Virtualization host extensions" system registers
  {"ttbr1_el2", TTBR1_EL2, AArch64::HasV8_1aOps},
  {"contextidr_el2", CONTEXTIDR_EL2, AArch64::HasV8_1aOps},
  {"cnthv_tval_el2", CNTHV_TVAL_EL2, AArch64::HasV8_1aOps},
  {"cnthv_cval_el2", CNTHV_CVAL_EL2, AArch64::HasV8_1aOps},
  {"cnthv_ctl_el2", CNTHV_CTL_EL2, AArch64::HasV8_1aOps},
  {"sctlr_el12", SCTLR_EL12, AArch64::HasV8_1aOps},
  {"cpacr_el12", CPACR_EL12, AArch64::HasV8_1aOps},
  {"ttbr0_el12", TTBR0_EL12, AArch64::HasV8_1aOps},
  {"ttbr1_el12", TTBR1_EL12, AArch64::HasV8_1aOps},
  {"tcr_el12", TCR_EL12, AArch64::HasV8_1aOps},
  {"afsr0_el12", AFSR0_EL12, AArch64::HasV8_1aOps},
  {"afsr1_el12", AFSR1_EL12, AArch64::HasV8_1aOps},
  {"esr_el12", ESR_EL12, AArch64::HasV8_1aOps},
  {"far_el12", FAR_EL12, AArch64::HasV8_1aOps},
  {"mair_el12", MAIR_EL12, AArch64::HasV8_1aOps},
  {"amair_el12", AMAIR_EL12, AArch64::HasV8_1aOps},
  {"vbar_el12", VBAR_EL12, AArch64::HasV8_1aOps},
  {"contextidr_el12", CONTEXTIDR_EL12, AArch64::HasV8_1aOps},
  {"cntkctl_el12", CNTKCTL_EL12, AArch64::HasV8_1aOps},
  {"cntp_tval_el02", CNTP_TVAL_EL02, AArch64::HasV8_1aOps},
  {"cntp_ctl_el02", CNTP_CTL_EL02, AArch64::HasV8_1aOps},
  {"cntp_cval_el02", CNTP_CVAL_EL02, AArch64::HasV8_1aOps},
  {"cntv_tval_el02", CNTV_TVAL_EL02, AArch64::HasV8_1aOps},
  {"cntv_ctl_el02", CNTV_CTL_EL02, AArch64::HasV8_1aOps},
  {"cntv_cval_el02", CNTV_CVAL_EL02, AArch64::HasV8_1aOps},
  {"spsr_el12", SPSR_EL12, AArch64::HasV8_1aOps},
  {"elr_el12", ELR_EL12, AArch64::HasV8_1aOps},
};

uint32_t
AArch64SysReg::SysRegMapper::fromString(StringRef Name, uint64_t FeatureBits, 
                                        bool &Valid) const {
  std::string NameLower = Name.lower();

  // First search the registers shared by all
  for (unsigned i = 0; i < array_lengthof(SysRegMappings); ++i) {
    if (SysRegMappings[i].isNameEqual(NameLower, FeatureBits)) {
      Valid = true;
      return SysRegMappings[i].Value;
    }
  }

  // Now try the instruction-specific registers (either read-only or
  // write-only).
  for (unsigned i = 0; i < NumInstMappings; ++i) {
    if (InstMappings[i].isNameEqual(NameLower, FeatureBits)) {
      Valid = true;
      return InstMappings[i].Value;
    }
  }

  // Try to parse an S<op0>_<op1>_<Cn>_<Cm>_<op2> register name
  Regex GenericRegPattern("^s([0-3])_([0-7])_c([0-9]|1[0-5])_c([0-9]|1[0-5])_([0-7])$");

  SmallVector<StringRef, 5> Ops;
  if (!GenericRegPattern.match(NameLower, &Ops)) {
    Valid = false;
    return -1;
  }

  uint32_t Op0 = 0, Op1 = 0, CRn = 0, CRm = 0, Op2 = 0;
  uint32_t Bits;
  Ops[1].getAsInteger(10, Op0);
  Ops[2].getAsInteger(10, Op1);
  Ops[3].getAsInteger(10, CRn);
  Ops[4].getAsInteger(10, CRm);
  Ops[5].getAsInteger(10, Op2);
  Bits = (Op0 << 14) | (Op1 << 11) | (CRn << 7) | (CRm << 3) | Op2;

  Valid = true;
  return Bits;
}

std::string
AArch64SysReg::SysRegMapper::toString(uint32_t Bits, uint64_t FeatureBits) const {
  // First search the registers shared by all
  for (unsigned i = 0; i < array_lengthof(SysRegMappings); ++i) {
    if (SysRegMappings[i].isValueEqual(Bits, FeatureBits)) {
      return SysRegMappings[i].Name;
    }
  }

  // Now try the instruction-specific registers (either read-only or
  // write-only).
  for (unsigned i = 0; i < NumInstMappings; ++i) {
    if (InstMappings[i].isValueEqual(Bits, FeatureBits)) {
      return InstMappings[i].Name;
    }
  }

  assert(Bits < 0x10000);
  uint32_t Op0 = (Bits >> 14) & 0x3;
  uint32_t Op1 = (Bits >> 11) & 0x7;
  uint32_t CRn = (Bits >> 7) & 0xf;
  uint32_t CRm = (Bits >> 3) & 0xf;
  uint32_t Op2 = Bits & 0x7;

  return "s" + utostr(Op0)+ "_" + utostr(Op1) + "_c" + utostr(CRn)
               + "_c" + utostr(CRm) + "_" + utostr(Op2);
}

const AArch64NamedImmMapper::Mapping AArch64TLBI::TLBIMapper::TLBIMappings[] = {
  {"ipas2e1is", IPAS2E1IS, 0},
  {"ipas2le1is", IPAS2LE1IS, 0},
  {"vmalle1is", VMALLE1IS, 0},
  {"alle2is", ALLE2IS, 0},
  {"alle3is", ALLE3IS, 0},
  {"vae1is", VAE1IS, 0},
  {"vae2is", VAE2IS, 0},
  {"vae3is", VAE3IS, 0},
  {"aside1is", ASIDE1IS, 0},
  {"vaae1is", VAAE1IS, 0},
  {"alle1is", ALLE1IS, 0},
  {"vale1is", VALE1IS, 0},
  {"vale2is", VALE2IS, 0},
  {"vale3is", VALE3IS, 0},
  {"vmalls12e1is", VMALLS12E1IS, 0},
  {"vaale1is", VAALE1IS, 0},
  {"ipas2e1", IPAS2E1, 0},
  {"ipas2le1", IPAS2LE1, 0},
  {"vmalle1", VMALLE1, 0},
  {"alle2", ALLE2, 0},
  {"alle3", ALLE3, 0},
  {"vae1", VAE1, 0},
  {"vae2", VAE2, 0},
  {"vae3", VAE3, 0},
  {"aside1", ASIDE1, 0},
  {"vaae1", VAAE1, 0},
  {"alle1", ALLE1, 0},
  {"vale1", VALE1, 0},
  {"vale2", VALE2, 0},
  {"vale3", VALE3, 0},
  {"vmalls12e1", VMALLS12E1, 0},
  {"vaale1", VAALE1, 0}
};

AArch64TLBI::TLBIMapper::TLBIMapper()
  : AArch64NamedImmMapper(TLBIMappings, 0) {}
