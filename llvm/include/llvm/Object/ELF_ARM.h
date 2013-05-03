//===-- ELF_ARM.h - ARM ELF ABI ---------------------------------*- C++ -*-===//
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

#ifndef LLVM_OBJECT_ELF_ARM_H
#define LLVM_OBJECT_ELF_ARM_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Endian.h"

namespace llvm {

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
  FP_arch                   = 10,
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
  FP_HP_extension           = 36,
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
  v7E_M    = 13,  // v7_M with DSP extensions
  v8
};

enum CPUArchProfile { // (=7), uleb128
  Not_Applicable = 0, // pre v7, or cross-profile code
  ApplicationProfile = (0x41), // 'A' (e.g. for Cortex A8)
  RealTimeProfile = (0x52), // 'R' (e.g. for Cortex R4)
  MicroControllerProfile = (0x4D), // 'M' (e.g. for Cortex M3)
  SystemProfile = (0x53) // 'S' Application or real-time profile
};

// The following have a lot of common use cases
enum {
  //ARMISAUse (=8), uleb128  and THUMBISAUse (=9), uleb128
  Not_Allowed = 0,
  Allowed = 1
};

enum {
  // FP_arch (=10), uleb128 (formerly Tag_VFP_arch = 10)
  AllowFPv2  = 2, // v2 FP ISA permitted (implies use of the v1 FP ISA)
  AllowFPv3A = 3, // v3 FP ISA permitted (implies use of the v2 FP ISA)
  AllowFPv3B = 4, // v3 FP ISA permitted, but only D0-D15, S0-S31
  AllowFPv4A = 5, // v4 FP ISA permitted (implies use of v3 FP ISA)
  AllowFPv4B = 6, // v4 FP ISA was permitted, but only D0-D15, S0-S31
  AllowV8FP  = 7, // ARMv8-A FP ISA permitted
  AllowV8FPB = 8  // ARMv8-A FP ISA permitted, but only D0-D15, S0-D31
};

enum {
  // Tag_THUMB_ISA_use, (=9), uleb128
  AllowThumb32 = 2 // 32-bit Thumb (implies 16-bit instructions)
};

enum {
  // Tag_WMMX_arch, (=11), uleb128
  AllowWMMXv1 = 1,  // The user permitted this entity to use WMMX v1
  AllowWMMXv2 = 2   // The user permitted this entity to use WMMX v2
};

enum {
  // Tag_ABI_FP_denormal, (=20), uleb128
  MightFlushToZero = 0, // Denormal numbers might be flushed to (+) zero
  IEEE754Denormal = 1 , // Depends on IEEE 754 denormal numbers
  PreserveFPSign = 2    // Sign when flushed-to-zero is preserved
};

enum {
  // Tag_ABI_FP_number_model, (=23), uleb128
  AllowNormal = 1, // Use IEEE 754 format normal numbers only
  AllowRTABI = 2,  // numbers, infinities, and one quiet NaN (see [RTABI])
  AllowIEE754 = 3  // this code to use all the IEEE 754-defined FP encodings
};

enum {
  // Tag_ABI_FP_rounding, (=19), uleb128
  FPRoundingNearest = 0,  // Use the IEEE 754 round to nearest rounding mode
  FPRoundingRuntime = 1   // Choose the IEEE 754 rounding mode at run time
};

enum {
  // Tag_DIV_use, (=44), uleb128
  AllowDIVThumb = 0,  // Allow SDIV, UDIV on Thumb ISA, e.g. Cortex R4 or M3
  NotAllowedDIV = 1,  // Disallow SDIV and UDIV
  AllowDIVv7a   = 2   // Allow SDIV, UDIV on v7-a with integer div extension
};

enum {
  // Tag_Virtualization_use, (=42), uleb128
  TrustZone = 1,  // Use of the TrustZone extension was permitted
  VirtExts  = 2,  // Use of virtualization extensions (HVC, ERET) permitted
  TrustZoneVirtExts = 3  // TrustZone and virtualization extensions permitted
};

enum {
  // Tag_PCS_config, (=13), uleb128
  PCS_none = 0,  // No standard configuration used, or no information recorded
  PCS_bare = 1,        // Bare platform configuration
  PCS_linux = 2,       // Linux application configuration
  PCS_linux_dso = 3,   // Linux DSO configuration
  PCS_palm_2004 = 4,   // Palm OS 2004 configuration
  PCS_palm_future = 5, // Reserved to future Palm OS configuration
  PCS_symbian_2004 = 6,   // Symbian OS 2004 configuration
  PCS_symbian_future = 7  // Reserved to future Symbian OS configuration
};

enum {
  // Tag_ABI_PCS_R9_use, (=14), uleb128
  PCS_R9_normal = 0, // R9 used as V6 (just another callee-saved register,
                     // implied by omitting the tag)
  PCS_R9_SB = 1,     // R9 used as SB, a global Static Base register
  PCS_R9_TLS = 2,    // R9 used as a Thread Local Storage (TLS) pointer
  PCS_R9_none = 3    // R9 not used at all by code associated with
                     // the attributed entity.
};

enum {
  // Tag_ABI_PCS_RW_data, (=15), uleb128
  PCS_RW_data_abs = 0, // RW static data permitted to be addressed absolutely
  PCS_RW_data_pcrel = 1, // RW static data was only permitted to be
                         // addressed PC-relative.
  PCS_RW_data_sbrel = 2, // RW static data was only permitted to be addressed
                         // SB-relative.
  PCS_RW_data_none = 3   // No permission to use RW static data
};

enum {
  // Tag_ABI_PCS_RO_data, (=16), uleb128
  PCS_RO_data_abs = 0,  // RO static data permitted to be addressed absolutely
  PCS_RO_data_pcrel = 1, // RO static data was only permitted to be
                         // addressed PC-relative.
  PCS_RO_data_none = 2   // No permission to use RO static data
};

enum {
  // Tag_ABI_PCS_GOT_use, (=17), uleb128
  PCS_GOT_none = 0,   // No permission to import static data
  PCS_GOT_direct = 1, // Permission to address imported data directly
  PCS_GOT_indirect = 2  // The user permitted this entity to address imported
                        // data indirectly (e.g. via a GOT)
};

enum {
  // Tag_ABI_PCS_wchar_t, (=18), uleb128
  PCS_wchar_t_disallowed = 0, // The user prohibited the use of wchar_t
  PCS_wchar_t_2 = 2,          // The user intended the size of wchar_t to be 2
  PCS_wchar_t_4 = 4           // The user intended the size of wchar_t to be 4
};

enum {
  // Tag_ABI_enum_size, (=26), uleb128
  PCS_enum_size = 1,  // Enum values occupy the smallest container big enough
                      // to hold all their values.
  PCS_enum_size_32 = 2, // The user intended Enum containers to be 32-bit
  PCS_enum_size_abi32 = 3  // The user intended that every enumeration visible
    // across an ABI-complying interface contains a value needing 32 bits to
    // encode it; other enums can be containerized.
};

enum {
  // Tag_ABI_align_needed, (=24), uleb128
  PCS_Align_needed_disallowed = 0, //  The user did not permit code to depend
    // the alignment of 8-byte data or data with extended (>8-byte) alignment.
  PCS_Align_needed_8 = 1, // Code was permitted to depend on the 8-byte
                          // alignment of 8-byte data items.
  PCS_Align_needed_4 = 2, // Code was permitted to depend on the 4-byte
                          // alignment of 8-byte data items.
  PCS_Align_needed_res = 3  // Reserved
  // OR: n (in 4..12) Code was permitted to depend on the 8-byte alignment of
  // 8-byte data items and the alignment of data items having up to 2^n byte
  // extended alignment.
};

enum {
  // Tag_ABI_align_preserved, (=25), uleb128
  PCS_Align_preserved_none = 0, // The user did not require code to preserve
                                // 8-byte alignment of 8-byte data objects.
  PCS_Align_preserved_8 = 1, // Code was required to preserve 8-byte alignment
                             // of 8-byte data objects.
  PCS_Align_preserved_8sp = 2, // Code was required to preserve 8-byte
    // alignment of 8-byte data objects and to ensure (SP MOD 8) = 0 at all
    // instruction boundaries (not just at function calls).
  PCS_Align_preserved_res = 3  // Reserved
  // OR: n (in 4..12) Code was required to preserve the alignments of case 2
  // and the alignment of data items having up to 2^n byte extended alignment
};

enum {
  // Tag_ABI_HardFP_use, (=27), uleb128
  HardFPImplied = 0,   // FP use should be implied by Tag_FP_arch
  HardFP_SP_VFP = 1,   // Use only SP FP instructions
  HardFP_Reserved = 2,      // Reserved
  HardFP_ImpliedDup = 3  // Deprecated duplicate of the default HardFPImplied
};

enum {
  // Tag_ABI_VFP_args, (=28), uleb128
  VFPArgs_base = 0,  // FP parameter/result passing using AAPCS, base variant
  VFPArgs_VFP = 1,   // FP parameter/result passing using AAPCS, VFP variant
  VFPArgs_toolchain = 2, // FP parameter/result passing to conform to tool
    // chain-specific conventions
  VFPArgs_baseVFP = 3 // FP parameter/result passing using both base and VFP
    //variants. Did not permit non-variadic functions to pass FP params/results.
};

/// Contains build ARM aeabi attribute values.
class ARMGenericBuildAttrInfo {
public:
  ARMGenericBuildAttrInfo()
    : Valid(false) {}

  CPUArch Tag_CPU_arch;
  CPUArchProfile Tag_CPU_arch_profile;
  std::string Tag_CPU_raw_name;
  std::string Tag_CPU_name;
  unsigned Tag_ARM_ISA_use;
  unsigned Tag_THUMB_ISA_use;
  unsigned Tag_FP_arch;
  unsigned Tag_WMMX_arch;
  unsigned Tag_Advanced_SIMD_arch;
  unsigned Tag_FP_HP_extension;
  unsigned Tag_CPU_unaligned_access;
  unsigned Tag_MPextension_use;
  unsigned Tag_DIV_use;
  unsigned Tag_T2EE_use;
  unsigned Tag_Virtualization_use;
  unsigned Tag_ABI_optimization_goals;
  unsigned Tag_ABI_FP_optimization_goals;

  //PCS/ABI attributes
  unsigned Tag_PCS_config;
  unsigned Tag_ABI_PCS_R9_use;
  unsigned Tag_ABI_PCS_RW_data;
  unsigned Tag_ABI_PCS_RO_data;
  unsigned Tag_ABI_PCS_GOT_use;
  unsigned Tag_ABI_PCS_wchar_t;
  unsigned Tag_ABI_enum_size;
  unsigned Tag_ABI_align8_needed;
  unsigned Tag_ABI_align8_preserved;

  //FP
  unsigned Tag_ABI_FP_rounding;
  unsigned Tag_ABI_FP_denormal;
  unsigned Tag_ABI_FP_number_model;
  unsigned Tag_ABI_FP_exceptions;
  unsigned Tag_ABI_FP_user_exceptions;
  unsigned Tag_ABI_HardFP_use;
  unsigned Tag_ABI_VFP_args;

private:
  bool Valid;

public:
  /// Indicates whether this instance contains valid or default values.
  bool isValid() { return Valid; }
  void setValid(bool v) { Valid = v; }
};

}
} // llvm

#endif // LLVM_OBJECT_ELF_ARM_H
