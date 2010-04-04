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

  /// IsExplicit - Indicates that this section comes from globals with an
  /// explicit section specified.
  bool IsExplicit;
  
protected:
  MCSectionELF(StringRef Section, unsigned type, unsigned flags,
               SectionKind K, bool isExplicit)
    : MCSection(K), SectionName(Section), Type(type), Flags(flags), 
      IsExplicit(isExplicit) {}
public:
  
  static MCSectionELF *Create(StringRef Section, unsigned Type, 
                              unsigned Flags, SectionKind K, bool isExplicit,
                              MCContext &Ctx);

  /// ShouldOmitSectionDirective - Decides whether a '.section' directive
  /// should be printed before the section name
  bool ShouldOmitSectionDirective(StringRef Name, const MCAsmInfo &MAI) const;

  /// ShouldPrintSectionType - Only prints the section type if supported
  bool ShouldPrintSectionType(unsigned Ty) const;

  /// HasCommonSymbols - True if this section holds common symbols, this is
  /// indicated on the ELF object file by a symbol with SHN_COMMON section 
  /// header index.
  bool HasCommonSymbols() const;
  
  /// These are the section type and flags fields.  An ELF section can have
  /// only one Type, but can have more than one of the flags specified.
  ///
  /// Valid section types.
  enum {
    // This value marks the section header as inactive.
    SHT_NULL             = 0x00U,

    // Holds information defined by the program, with custom format and meaning.
    SHT_PROGBITS         = 0x01U,

    // This section holds a symbol table.
    SHT_SYMTAB           = 0x02U,

    // The section holds a string table.
    SHT_STRTAB           = 0x03U,

    // The section holds relocation entries with explicit addends.
    SHT_RELA             = 0x04U,

    // The section holds a symbol hash table.
    SHT_HASH             = 0x05U,
    
    // Information for dynamic linking.
    SHT_DYNAMIC          = 0x06U,

    // The section holds information that marks the file in some way.
    SHT_NOTE             = 0x07U,

    // A section of this type occupies no space in the file.
    SHT_NOBITS           = 0x08U,

    // The section holds relocation entries without explicit addends.
    SHT_REL              = 0x09U,

    // This section type is reserved but has unspecified semantics. 
    SHT_SHLIB            = 0x0AU,

    // This section holds a symbol table.
    SHT_DYNSYM           = 0x0BU,

    // This section contains an array of pointers to initialization functions.
    SHT_INIT_ARRAY       = 0x0EU,

    // This section contains an array of pointers to termination functions.
    SHT_FINI_ARRAY       = 0x0FU,

    // This section contains an array of pointers to functions that are invoked
    // before all other initialization functions.
    SHT_PREINIT_ARRAY    = 0x10U,

    // A section group is a set of sections that are related and that must be
    // treated specially by the linker.
    SHT_GROUP            = 0x11U,

    // This section is associated with a section of type SHT_SYMTAB, when the
    // referenced symbol table contain the escape value SHN_XINDEX
    SHT_SYMTAB_SHNDX     = 0x12U,

    LAST_KNOWN_SECTION_TYPE = SHT_SYMTAB_SHNDX
  }; 

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
    
    /// FIRST_TARGET_DEP_FLAG - This is the first flag that subclasses are
    /// allowed to specify.
    FIRST_TARGET_DEP_FLAG = 0x800U,

    /// TARGET_INDEP_SHF - This is the bitmask for all the target independent
    /// section flags.  Targets can define their own target flags above these.
    /// If they do that, they should implement their own MCSectionELF subclasses
    /// and implement the virtual method hooks below to handle printing needs.
    TARGET_INDEP_SHF     = FIRST_TARGET_DEP_FLAG-1U
  };

  StringRef getSectionName() const { return SectionName; }
  unsigned getType() const { return Type; }
  unsigned getFlags() const { return Flags; }
  
  virtual void PrintSwitchToSection(const MCAsmInfo &MAI,
                                    raw_ostream &OS) const;
  
  /// isBaseAddressKnownZero - We know that non-allocatable sections (like
  /// debug info) have a base of zero.
  virtual bool isBaseAddressKnownZero() const {
    return (getFlags() & SHF_ALLOC) == 0;
  }
  
  /// PrintTargetSpecificSectionFlags - Targets that define their own
  /// MCSectionELF subclasses with target specific section flags should
  /// implement this method if they end up adding letters to the attributes
  /// list.
  virtual void PrintTargetSpecificSectionFlags(const MCAsmInfo &MAI,
                                               raw_ostream &OS) const {
  }
                                               
  
};

} // end namespace llvm

#endif
