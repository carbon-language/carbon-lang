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
// as defined in ARM ABI addenda document (ABI release 2.07).
//
//===----------------------------------------------------------------------===//

#ifndef __TARGET_ARMBUILDATTRS_H__
#define __TARGET_ARMBUILDATTRS_H__

namespace ARMBuildAttrs {
  enum {
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
    nodefaults                = 64,
    also_compatible_with      = 65,
    T2EE_use                  = 66,
    conformance               = 67,
    Virtualization_use        = 68,
    MPextension_use           = 70
  };
}

#endif // __TARGET_ARMBUILDATTRS_H__
