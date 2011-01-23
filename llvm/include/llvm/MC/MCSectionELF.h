//===- MCSectionELF.h - ELF Machine Code Sections ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the MCSectionELF class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCSECTIONELF_H
#define LLVM_MC_MCSECTIONELF_H

#include "llvm/MC/MCSection.h"

namespace llvm {

class MCSymbol;

/// MCSectionELF - This represents a section on linux, lots of unix variants
/// and some bare metal systems.
class MCSectionELF : public MCSection {
  /// SectionName - This is the name of the section.  The referenced memory is
  /// owned by TargetLoweringObjectFileELF's ELFUniqueMap.
  StringRef SectionName;

  /// Type - This is the sh_type field of a section, drawn from the enums below.
  unsigned Type;

  /// Flags - This is the sh_flags field of a section, drawn from the enums.
  /// below.
  unsigned Flags;

  /// EntrySize - The size of each entry in this section. This size only
  /// makes sense for sections that contain fixed-sized entries. If a
  /// section does not contain fixed-sized entries 'EntrySize' will be 0.
  unsigned EntrySize;

  const MCSymbol *Group;

private:
  friend class MCContext;
  MCSectionELF(StringRef Section, unsigned type, unsigned flags,
               SectionKind K, unsigned entrySize, const MCSymbol *group)
    : MCSection(SV_ELF, K), SectionName(Section), Type(type), Flags(flags),
      EntrySize(entrySize), Group(group) {}
  ~MCSectionELF();
public:

  /// ShouldOmitSectionDirective - Decides whether a '.section' directive
  /// should be printed before the section name
  bool ShouldOmitSectionDirective(StringRef Name, const MCAsmInfo &MAI) const;

  /// HasCommonSymbols - True if this section holds common symbols, this is
  /// indicated on the ELF object file by a symbol with SHN_COMMON section
  /// header index.
  bool HasCommonSymbols() const;

  /// Valid section flags.
  enum {
    // The section contains data that should be writable.
    SHF_WRITE            = 0x1U,

    // The section occupies memory during execution.
    SHF_ALLOC            = 0x2U,

    // The section contains executable machine instructions.
    SHF_EXECINSTR        = 0x4U,

    // The data in the section may be merged to eliminate duplication.
    SHF_MERGE            = 0x10U,

    // Elements in the section consist of null-terminated character strings.
    SHF_STRINGS          = 0x20U,

    // A field in this section holds a section header table index.
    SHF_INFO_LINK        = 0x40U,

    // Adds special ordering requirements for link editors.
    SHF_LINK_ORDER       = 0x80U,

    // This section requires special OS-specific processing to avoid incorrect
    // behavior.
    SHF_OS_NONCONFORMING = 0x100U,

    // This section is a member of a section group.
    SHF_GROUP            = 0x200U,

    // This section holds Thread-Local Storage.
    SHF_TLS              = 0x400U,


    // Start of target-specific flags.

    /// XCORE_SHF_CP_SECTION - All sections with the "c" flag are grouped
    /// together by the linker to form the constant pool and the cp register is
    /// set to the start of the constant pool by the boot code.
    XCORE_SHF_CP_SECTION = 0x800U,

    /// XCORE_SHF_DP_SECTION - All sections with the "d" flag are grouped
    /// together by the linker to form the data section and the dp register is
    /// set to the start of the section by the boot code.
    XCORE_SHF_DP_SECTION = 0x1000U
  };

  StringRef getSectionName() const { return SectionName; }
  unsigned getType() const { return Type; }
  unsigned getFlags() const { return Flags; }
  unsigned getEntrySize() const { return EntrySize; }
  const MCSymbol *getGroup() const { return Group; }

  void PrintSwitchToSection(const MCAsmInfo &MAI,
                            raw_ostream &OS) const;
  virtual bool UseCodeAlign() const;
  virtual bool isVirtualSection() const;

  /// isBaseAddressKnownZero - We know that non-allocatable sections (like
  /// debug info) have a base of zero.
  virtual bool isBaseAddressKnownZero() const {
    return (getFlags() & SHF_ALLOC) == 0;
  }

  static bool classof(const MCSection *S) {
    return S->getVariant() == SV_ELF;
  }
  static bool classof(const MCSectionELF *) { return true; }

  // Return the entry size for sections with fixed-width data.
  static unsigned DetermineEntrySize(SectionKind Kind);

};

} // end namespace llvm

#endif
