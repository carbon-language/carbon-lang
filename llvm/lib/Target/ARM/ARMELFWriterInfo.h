//===-- ARMELFWriterInfo.h - ELF Writer Info for ARM ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements ELF writer information for the ARM backend.
//
//===----------------------------------------------------------------------===//

#ifndef ARM_ELF_WRITER_INFO_H
#define ARM_ELF_WRITER_INFO_H

#include "llvm/Target/TargetELFWriterInfo.h"

namespace llvm {

  class ARMELFWriterInfo : public TargetELFWriterInfo {

    // ELF Relocation types for ARM
    // Meets 2.08 ABI Specs. Most of these are not used by the assembler.
    // They are here for completeness's sake
    
    enum ARMRelocationType {
      R_ARM_NONE                  = 0x00,
      R_ARM_PC24                  = 0x01,
      R_ARM_ABS32                 = 0x02,
      R_ARM_REL32                 = 0x03,
      R_ARM_LDR_PC_G0             = 0x04,
      R_ARM_ABS16                 = 0x05,
      R_ARM_ABS12                 = 0x06,
      R_ARM_THM_ABS5              = 0x07,
      R_ARM_ABS8                  = 0x08,
      R_ARM_SBREL32               = 0x09,
      R_ARM_THM_CALL              = 0x0a,
      R_ARM_THM_PC8               = 0x0b,
      R_ARM_BREL_ADJ              = 0x0c,
      R_ARM_TLS_DESC              = 0x0d,
      R_ARM_THM_SWI8              = 0x0e,
      R_ARM_XPC25                 = 0x0f,
      R_ARM_THM_XPC22             = 0x10,
      R_ARM_TLS_DTPMOD32          = 0x11,
      R_ARM_TLS_DTPOFF32          = 0x12,
      R_ARM_TLS_TPOFF32           = 0x13,
      R_ARM_COPY                  = 0x14,
      R_ARM_GLOB_DAT              = 0x15,
      R_ARM_JUMP_SLOT             = 0x16,
      R_ARM_RELATIVE              = 0x17,
      R_ARM_GOTOFF32              = 0x18,
      R_ARM_BASE_PREL             = 0x19,
      R_ARM_GOT_BREL              = 0x1a,
      R_ARM_PLT32                 = 0x1b,
      R_ARM_CALL                  = 0x1c,
      R_ARM_JUMP24                = 0x1d,
      R_ARM_THM_JUMP24            = 0x1e,
      R_ARM_BASE_ABS              = 0x1f,
      R_ARM_ALU_PCREL_7_0         = 0x20,
      R_ARM_ALU_PCREL_15_8        = 0x21,
      R_ARM_ALU_PCREL_23_15       = 0x22,
      R_ARM_LDR_SBREL_11_0_NC     = 0x23,
      R_ARM_ALU_SBREL_19_12_NC    = 0x24,
      R_ARM_ALU_SBREL_27_20_CK    = 0x25,
      R_ARM_TARGET1               = 0x26,
      R_ARM_SBREL31               = 0x27,
      R_ARM_V4BX                  = 0x28,
      R_ARM_TARGET2               = 0x29,
      R_ARM_PREL31                = 0x2a,
      R_ARM_MOVW_ABS_NC           = 0x2b,
      R_ARM_MOVT_ABS              = 0x2c,
      R_ARM_MOVW_PREL_NC          = 0x2d,
      R_ARM_MOVT_PREL             = 0x2e,
      R_ARM_THM_MOVW_ABS_NC       = 0x2f,
      R_ARM_THM_MOVT_ABS          = 0x30,
      R_ARM_THM_MOVW_PREL_NC      = 0x31,
      R_ARM_THM_MOVT_PREL         = 0x32,
      R_ARM_THM_JUMP19            = 0x33,
      R_ARM_THM_JUMP6             = 0x34,
      R_ARM_THM_ALU_PREL_11_0     = 0x35,
      R_ARM_THM_PC12              = 0x36,
      R_ARM_ABS32_NOI             = 0x37,
      R_ARM_REL32_NOI             = 0x38,
      R_ARM_ALU_PC_G0_NC          = 0x39,
      R_ARM_ALU_PC_G0             = 0x3a,
      R_ARM_ALU_PC_G1_NC          = 0x3b,
      R_ARM_ALU_PC_G1             = 0x3c,
      R_ARM_ALU_PC_G2             = 0x3d,
      R_ARM_LDR_PC_G1             = 0x3e,
      R_ARM_LDR_PC_G2             = 0x3f,
      R_ARM_LDRS_PC_G0            = 0x40,
      R_ARM_LDRS_PC_G1            = 0x41,
      R_ARM_LDRS_PC_G2            = 0x42,
      R_ARM_LDC_PC_G0             = 0x43,
      R_ARM_LDC_PC_G1             = 0x44,
      R_ARM_LDC_PC_G2             = 0x45,
      R_ARM_ALU_SB_G0_NC          = 0x46,
      R_ARM_ALU_SB_G0             = 0x47,
      R_ARM_ALU_SB_G1_NC          = 0x48,
      R_ARM_ALU_SB_G1             = 0x49,
      R_ARM_ALU_SB_G2             = 0x4a,
      R_ARM_LDR_SB_G0             = 0x4b,
      R_ARM_LDR_SB_G1             = 0x4c,
      R_ARM_LDR_SB_G2             = 0x4d,
      R_ARM_LDRS_SB_G0            = 0x4e,
      R_ARM_LDRS_SB_G1            = 0x4f,
      R_ARM_LDRS_SB_G2            = 0x50,
      R_ARM_LDC_SB_G0             = 0x51,
      R_ARM_LDC_SB_G1             = 0x52,
      R_ARM_LDC_SB_G2             = 0x53,
      R_ARM_MOVW_BREL_NC          = 0x54,
      R_ARM_MOVT_BREL             = 0x55,
      R_ARM_MOVW_BREL             = 0x56,
      R_ARM_THM_MOVW_BREL_NC      = 0x57,
      R_ARM_THM_MOVT_BREL         = 0x58,
      R_ARM_THM_MOVW_BREL         = 0x59,
      R_ARM_TLS_GOTDESC           = 0x5a,
      R_ARM_TLS_CALL              = 0x5b,
      R_ARM_TLS_DESCSEQ           = 0x5c,
      R_ARM_THM_TLS_CALL          = 0x5d,
      R_ARM_PLT32_ABS             = 0x5e,
      R_ARM_GOT_ABS               = 0x5f,
      R_ARM_GOT_PREL              = 0x60,
      R_ARM_GOT_BREL12            = 0x61,
      R_ARM_GOTOFF12              = 0x62,
      R_ARM_GOTRELAX              = 0x63,
      R_ARM_GNU_VTENTRY           = 0x64,
      R_ARM_GNU_VTINHERIT         = 0x65,
      R_ARM_THM_JUMP11            = 0x66,
      R_ARM_THM_JUMP8             = 0x67,
      R_ARM_TLS_GD32              = 0x68,
      R_ARM_TLS_LDM32             = 0x69,
      R_ARM_TLS_LDO32             = 0x6a,
      R_ARM_TLS_IE32              = 0x6b,
      R_ARM_TLS_LE32              = 0x6c,
      R_ARM_TLS_LDO12             = 0x6d,
      R_ARM_TLS_LE12              = 0x6e,
      R_ARM_TLS_IE12GP            = 0x6f,
      R_ARM_PRIVATE_0             = 0x70,
      R_ARM_PRIVATE_1             = 0x71,
      R_ARM_PRIVATE_2             = 0x72,
      R_ARM_PRIVATE_3             = 0x73,
      R_ARM_PRIVATE_4             = 0x74,
      R_ARM_PRIVATE_5             = 0x75,
      R_ARM_PRIVATE_6             = 0x76,
      R_ARM_PRIVATE_7             = 0x77,
      R_ARM_PRIVATE_8             = 0x78,
      R_ARM_PRIVATE_9             = 0x79,
      R_ARM_PRIVATE_10            = 0x7a,
      R_ARM_PRIVATE_11            = 0x7b,
      R_ARM_PRIVATE_12            = 0x7c,
      R_ARM_PRIVATE_13            = 0x7d,
      R_ARM_PRIVATE_14            = 0x7e,
      R_ARM_PRIVATE_15            = 0x7f,
      R_ARM_ME_TOO                = 0x80,
      R_ARM_THM_TLS_DESCSEQ16     = 0x81,
      R_ARM_THM_TLS_DESCSEQ32     = 0x82
    };

  public:
    ARMELFWriterInfo(TargetMachine &TM);
    virtual ~ARMELFWriterInfo();

    /// getRelocationType - Returns the target specific ELF Relocation type.
    /// 'MachineRelTy' contains the object code independent relocation type
    virtual unsigned getRelocationType(unsigned MachineRelTy) const;

    /// hasRelocationAddend - True if the target uses an addend in the
    /// ELF relocation entry.
    virtual bool hasRelocationAddend() const { return false; }

    /// getDefaultAddendForRelTy - Gets the default addend value for a
    /// relocation entry based on the target ELF relocation type.
    virtual long int getDefaultAddendForRelTy(unsigned RelTy,
                                              long int Modifier = 0) const;

    /// getRelTySize - Returns the size of relocatable field in bits
    virtual unsigned getRelocationTySize(unsigned RelTy) const;

    /// isPCRelativeRel - True if the relocation type is pc relative
    virtual bool isPCRelativeRel(unsigned RelTy) const;

    /// getJumpTableRelocationTy - Returns the machine relocation type used
    /// to reference a jumptable.
    virtual unsigned getAbsoluteLabelMachineRelTy() const;

    /// computeRelocation - Some relocatable fields could be relocated
    /// directly, avoiding the relocation symbol emission, compute the
    /// final relocation value for this symbol.
    virtual long int computeRelocation(unsigned SymOffset, unsigned RelOffset,
                                       unsigned RelTy) const;
  };

} // end llvm namespace

#endif // ARM_ELF_WRITER_INFO_H
