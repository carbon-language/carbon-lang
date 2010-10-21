//===-- MBlazeELFWriterInfo.h - ELF Writer Info for MBlaze ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements ELF writer information for the MBlaze backend.
//
//===----------------------------------------------------------------------===//

#ifndef MBLAZE_ELF_WRITER_INFO_H
#define MBLAZE_ELF_WRITER_INFO_H

#include "llvm/Target/TargetELFWriterInfo.h"

namespace llvm {

  class MBlazeELFWriterInfo : public TargetELFWriterInfo {

    // ELF Relocation types for MBlaze
    enum MBlazeRelocationType {
      R_MICROBLAZE_NONE = 0,
      R_MICROBLAZE_32 = 1,
      R_MICROBLAZE_32_PCREL = 2,
      R_MICROBLAZE_64_PCREL = 3,
      R_MICROBLAZE_32_PCREL_LO = 4,
      R_MICROBLAZE_64 = 5,
      R_MICROBLAZE_32_LO = 6,
      R_MICROBLAZE_SRO32 = 7,
      R_MICROBLAZE_SRW32 = 8,
      R_MICROBLAZE_64_NONE = 9, 
      R_MICROBLAZE_32_SYM_OP_SYM = 10,
      R_MICROBLAZE_GNU_VTINHERIT = 11,
      R_MICROBLAZE_GNU_VTENTRY = 12,
      R_MICROBLAZE_GOTPC_64 = 13,
      R_MICROBLAZE_GOT_64 = 14,
      R_MICROBLAZE_PLT_64 = 15,
      R_MICROBLAZE_REL = 16,
      R_MICROBLAZE_JUMP_SLOT = 17,
      R_MICROBLAZE_GLOB_DAT = 18,
      R_MICROBLAZE_GOTOFF_64 = 19,
      R_MICROBLAZE_GOTOFF_32 = 20,
      R_MICROBLAZE_COPY = 21
    };

  public:
    MBlazeELFWriterInfo(TargetMachine &TM);
    virtual ~MBlazeELFWriterInfo();

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

#endif // MBLAZE_ELF_WRITER_INFO_H
