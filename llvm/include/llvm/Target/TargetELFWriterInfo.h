//===-- llvm/Target/TargetELFWriterInfo.h - ELF Writer Info -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the TargetELFWriterInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETELFWRITERINFO_H
#define LLVM_TARGET_TARGETELFWRITERINFO_H

namespace llvm {

  //===--------------------------------------------------------------------===//
  //                          TargetELFWriterInfo
  //===--------------------------------------------------------------------===//

  class TargetELFWriterInfo {
  protected:
    // EMachine - This field is the target specific value to emit as the
    // e_machine member of the ELF header.
    unsigned short EMachine;
    bool is64Bit, isLittleEndian;
  public:

    // Machine architectures
    enum MachineType {
      EM_NONE = 0,     // No machine
      EM_M32 = 1,      // AT&T WE 32100
      EM_SPARC = 2,    // SPARC
      EM_386 = 3,      // Intel 386
      EM_68K = 4,      // Motorola 68000
      EM_88K = 5,      // Motorola 88000
      EM_486 = 6,      // Intel 486 (deprecated)
      EM_860 = 7,      // Intel 80860
      EM_MIPS = 8,     // MIPS R3000
      EM_PPC = 20,     // PowerPC
      EM_ARM = 40,     // ARM
      EM_ALPHA = 41,   // DEC Alpha
      EM_SPARCV9 = 43, // SPARC V9
      EM_X86_64 = 62,  // AMD64
      EM_HEXAGON = 164 // Qualcomm Hexagon
    };

    // ELF File classes
    enum {
      ELFCLASS32 = 1, // 32-bit object file
      ELFCLASS64 = 2  // 64-bit object file
    };

    // ELF Endianess
    enum {
      ELFDATA2LSB = 1, // Little-endian object file
      ELFDATA2MSB = 2  // Big-endian object file
    };

    explicit TargetELFWriterInfo(bool is64Bit_, bool isLittleEndian_);
    virtual ~TargetELFWriterInfo();

    unsigned short getEMachine() const { return EMachine; }
    unsigned getEFlags() const { return 0; }
    unsigned getEIClass() const { return is64Bit ? ELFCLASS64 : ELFCLASS32; }
    unsigned getEIData() const {
      return isLittleEndian ? ELFDATA2LSB : ELFDATA2MSB;
    }

    /// ELF Header and ELF Section Header Info
    unsigned getHdrSize() const { return is64Bit ? 64 : 52; }
    unsigned getSHdrSize() const { return is64Bit ? 64 : 40; }

    /// Symbol Table Info
    unsigned getSymTabEntrySize() const { return is64Bit ? 24 : 16; }

    /// getPrefELFAlignment - Returns the preferred alignment for ELF. This
    /// is used to align some sections.
    unsigned getPrefELFAlignment() const { return is64Bit ? 8 : 4; }

    /// getRelocationEntrySize - Entry size used in the relocation section
    unsigned getRelocationEntrySize() const {
      return is64Bit ? (hasRelocationAddend() ? 24 : 16)
                     : (hasRelocationAddend() ? 12 : 8);
    }

    /// getRelocationType - Returns the target specific ELF Relocation type.
    /// 'MachineRelTy' contains the object code independent relocation type
    virtual unsigned getRelocationType(unsigned MachineRelTy) const = 0;

    /// hasRelocationAddend - True if the target uses an addend in the
    /// ELF relocation entry.
    virtual bool hasRelocationAddend() const = 0;

    /// getDefaultAddendForRelTy - Gets the default addend value for a
    /// relocation entry based on the target ELF relocation type.
    virtual long int getDefaultAddendForRelTy(unsigned RelTy,
                                              long int Modifier = 0) const = 0;

    /// getRelTySize - Returns the size of relocatable field in bits
    virtual unsigned getRelocationTySize(unsigned RelTy) const = 0;

    /// isPCRelativeRel - True if the relocation type is pc relative
    virtual bool isPCRelativeRel(unsigned RelTy) const = 0;

    /// getJumpTableRelocationTy - Returns the machine relocation type used
    /// to reference a jumptable.
    virtual unsigned getAbsoluteLabelMachineRelTy() const = 0;

    /// computeRelocation - Some relocatable fields could be relocated
    /// directly, avoiding the relocation symbol emission, compute the
    /// final relocation value for this symbol.
    virtual long int computeRelocation(unsigned SymOffset, unsigned RelOffset,
                                       unsigned RelTy) const = 0;
  };

} // end llvm namespace

#endif // LLVM_TARGET_TARGETELFWRITERINFO_H
