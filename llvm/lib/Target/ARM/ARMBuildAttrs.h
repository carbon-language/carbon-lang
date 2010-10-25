//===-------- ARMBuildAttrs.h - ARM Build Attributes ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains enumerations and support routines for ARM build attributes
// as defined in ARM ABI addenda document (ABI release 2.08).
//
//===----------------------------------------------------------------------===//

#ifndef __TARGET_ARMBUILDATTRS_H__
#define __TARGET_ARMBUILDATTRS_H__

namespace ARMBuildAttrs {
  enum SpecialAttr {
    // This is for the .cpu asm attr. It translates into one or more
    // AttrType (below) entries in the .ARM.attributes section in the ELF.
    SEL_CPU 
  };

  enum AttrType {
    // Rest correspond to ELF/.ARM.attributes
    File                      = 1,
    Section                   = 2,
    Symbol                    = 3,
    CPU_raw_name              = 4,
    CPU_name                  = 5,
    CPU_arch                  = 6,
    CPU_arch_profile          = 7,
    ARM_ISA_use               = 8,
    THUMB_ISA_use             = 9,
    VFP_arch                  = 10,
    WMMX_arch                 = 11,
    Advanced_SIMD_arch        = 12,
    PCS_config                = 13,
    ABI_PCS_R9_use            = 14,
    ABI_PCS_RW_data           = 15,
    ABI_PCS_RO_data           = 16,
    ABI_PCS_GOT_use           = 17,
    ABI_PCS_wchar_t           = 18,
    ABI_FP_rounding           = 19,
    ABI_FP_denormal           = 20,
    ABI_FP_exceptions         = 21,
    ABI_FP_user_exceptions    = 22,
    ABI_FP_number_model       = 23,
    ABI_align8_needed         = 24,
    ABI_align8_preserved      = 25,
    ABI_enum_size             = 26,
    ABI_HardFP_use            = 27,
    ABI_VFP_args              = 28,
    ABI_WMMX_args             = 29,
    ABI_optimization_goals    = 30,
    ABI_FP_optimization_goals = 31,
    compatibility             = 32,
    CPU_unaligned_access      = 34,
    VFP_HP_extension          = 36,
    ABI_FP_16bit_format       = 38,
    MPextension_use           = 42, // was 70, 2.08 ABI
    DIV_use                   = 44,
    nodefaults                = 64,
    also_compatible_with      = 65,
    T2EE_use                  = 66,
    conformance               = 67,
    Virtualization_use        = 68,
    MPextension_use_old       = 70
  };

  // Magic numbers for .ARM.attributes
  enum AttrMagic {
    Format_Version  = 0x41
  };

  // Legal Values for CPU_arch, (=6), uleb128
  enum CPUArch {
    Pre_v4       = 0,
    v4       = 1,   // e.g. SA110
    v4T      = 2,   // e.g. ARM7TDMI
    v5T      = 3,   // e.g. ARM9TDMI
    v5TE     = 4,   // e.g. ARM946E_S
    v5TEJ    = 5,   // e.g. ARM926EJ_S
    v6       = 6,   // e.g. ARM1136J_S
    v6KZ     = 7,   // e.g. ARM1176JZ_S
    v6T2     = 8,   // e.g. ARM1156T2F_S
    v6K      = 9,   // e.g. ARM1136J_S
    v7       = 10,  // e.g. Cortex A8, Cortex M3
    v6_M     = 11,  // e.g. Cortex M1
    v6S_M    = 12,  // v6_M with the System extensions
    v7E_M    = 13   // v7_M with DSP extensions
  };

}

#endif // __TARGET_ARMBUILDATTRS_H__
