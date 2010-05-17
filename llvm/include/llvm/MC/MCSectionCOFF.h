//===- MCSectionCOFF.h - COFF Machine Code Sections -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the MCSectionCOFF class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCSECTIONCOFF_H
#define LLVM_MC_MCSECTIONCOFF_H

#include "llvm/MC/MCSection.h"

namespace llvm {
  
/// MCSectionCOFF - This represents a section on Windows
  class MCSectionCOFF : public MCSection {
    // The memory for this string is stored in the same MCContext as *this.
    StringRef SectionName;
    
    /// Characteristics - This is the Characteristics field of a section,
    //  drawn from the enums below.
    unsigned Characteristics;

    /// Selection - This is the Selection field for the section symbol, if
    /// it is a COMDAT section (Characteristics & IMAGE_SCN_LNK_COMDAT) != 0
    int Selection;

  private:
    friend class MCContext;
    MCSectionCOFF(StringRef Section, unsigned Characteristics,
                  int Selection, SectionKind K)
      : MCSection(SV_COFF, K), SectionName(Section),
        Characteristics(Characteristics), Selection (Selection) {
      assert ((Characteristics & 0x00F00000) == 0 &&
        "alignment must not be set upon section creation");
    }
    ~MCSectionCOFF();

  public:
    /// ShouldOmitSectionDirective - Decides whether a '.section' directive
    /// should be printed before the section name
    bool ShouldOmitSectionDirective(StringRef Name, const MCAsmInfo &MAI) const;

    //FIXME: all COFF enumerations/flags should be standardized into one place...
    // Target/X86COFF.h doesn't seem right as COFF can be used for other targets,
    // MC/WinCOFF.h maybe right as it isn't target or entity specific, and it is
    //   pretty low on the dependancy graph (is there any need to support non
    //   windows COFF?)
    // here is good for section stuff, but others should go elsewhere

    /// Valid section flags.
    enum {
      IMAGE_SCN_TYPE_NO_PAD                     = 0x00000008,
      IMAGE_SCN_CNT_CODE                        = 0x00000020,
      IMAGE_SCN_CNT_INITIALIZED_DATA            = 0x00000040,
      IMAGE_SCN_CNT_UNINITIALIZED_DATA          = 0x00000080,
      IMAGE_SCN_LNK_OTHER                       = 0x00000100,
      IMAGE_SCN_LNK_INFO                        = 0x00000200,
      IMAGE_SCN_LNK_REMOVE                      = 0x00000800,
      IMAGE_SCN_LNK_COMDAT                      = 0x00001000,
      IMAGE_SCN_MEM_FARDATA                     = 0x00008000,
      IMAGE_SCN_MEM_PURGEABLE                   = 0x00020000,
      IMAGE_SCN_MEM_16BIT                       = 0x00020000,
      IMAGE_SCN_MEM_LOCKED                      = 0x00040000,
      IMAGE_SCN_MEM_PRELOAD                     = 0x00080000,
      /* these are handled elsewhere
      IMAGE_SCN_ALIGN_1BYTES                    = 0x00100000,
      IMAGE_SCN_ALIGN_2BYTES                    = 0x00200000,
      IMAGE_SCN_ALIGN_4BYTES                    = 0x00300000,
      IMAGE_SCN_ALIGN_8BYTES                    = 0x00400000,
      IMAGE_SCN_ALIGN_16BYTES                   = 0x00500000,
      IMAGE_SCN_ALIGN_32BYTES                   = 0x00600000,
      IMAGE_SCN_ALIGN_64BYTES                   = 0x00700000,
      */
      IMAGE_SCN_LNK_NRELOC_OVFL                 = 0x01000000,
      IMAGE_SCN_MEM_DISCARDABLE                 = 0x02000000,
      IMAGE_SCN_MEM_NOT_CACHED                  = 0x04000000,
      IMAGE_SCN_MEM_NOT_PAGED                   = 0x08000000,
      IMAGE_SCN_MEM_SHARED                      = 0x10000000,
      IMAGE_SCN_MEM_EXECUTE                     = 0x20000000,
      IMAGE_SCN_MEM_READ                        = 0x40000000,
      IMAGE_SCN_MEM_WRITE                       = 0x80000000
    };

    enum {
      IMAGE_COMDAT_SELECT_NODUPLICATES = 1,
      IMAGE_COMDAT_SELECT_ANY,
      IMAGE_COMDAT_SELECT_SAME_SIZE,
      IMAGE_COMDAT_SELECT_EXACT_MATCH,
      IMAGE_COMDAT_SELECT_ASSOCIATIVE,
      IMAGE_COMDAT_SELECT_LARGEST
    };

    StringRef getSectionName() const { return SectionName; }
    unsigned getCharacteristics() const { return Characteristics; }
    int getSelection () const { return Selection; }
    
    virtual void PrintSwitchToSection(const MCAsmInfo &MAI,
                                      raw_ostream &OS) const;

    static bool classof(const MCSection *S) {
      return S->getVariant() == SV_COFF;
    }
    static bool classof(const MCSectionCOFF *) { return true; }
  };

} // end namespace llvm

#endif
