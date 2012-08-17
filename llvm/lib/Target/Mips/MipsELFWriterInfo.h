//===-- MipsELFWriterInfo.h - ELF Writer Info for Mips ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements ELF writer information for the Mips backend.
//
//===----------------------------------------------------------------------===//

#ifndef MIPS_ELF_WRITER_INFO_H
#define MIPS_ELF_WRITER_INFO_H

#include "llvm/Target/TargetELFWriterInfo.h"

namespace llvm {

  class MipsELFWriterInfo : public TargetELFWriterInfo {

  public:
    MipsELFWriterInfo(bool is64Bit_, bool isLittleEndian_);
    virtual ~MipsELFWriterInfo();

    /// getRelocationType - Returns the target specific ELF Relocation type.
    /// 'MachineRelTy' contains the object code independent relocation type
    virtual unsigned getRelocationType(unsigned MachineRelTy) const;

    /// hasRelocationAddend - True if the target uses an addend in the
    /// ELF relocation entry.
    virtual bool hasRelocationAddend() const { return is64Bit ? true : false; }

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

#endif // MIPS_ELF_WRITER_INFO_H
