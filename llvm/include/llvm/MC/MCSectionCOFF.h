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
    
    /// Flags - This is the Characteristics field of a section, drawn
    /// from the enums below.
    unsigned Flags;

  private:
    friend class MCContext;
    MCSectionCOFF(StringRef Section, unsigned flags, SectionKind K)
      : MCSection(K), SectionName(Section), Flags(flags) {
    }
    ~MCSectionCOFF();

  public:
    /// ShouldOmitSectionDirective - Decides whether a '.section' directive
    /// should be printed before the section name
    bool ShouldOmitSectionDirective(StringRef Name, const MCAsmInfo &MAI) const;

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
      IMAGE_SCN_ALIGN_1BYTES                    = 0x00100000,
      IMAGE_SCN_ALIGN_2BYTES                    = 0x00200000,
      IMAGE_SCN_ALIGN_4BYTES                    = 0x00300000,
      IMAGE_SCN_ALIGN_8BYTES                    = 0x00400000,
      IMAGE_SCN_ALIGN_16BYTES                   = 0x00500000,
      IMAGE_SCN_ALIGN_32BYTES                   = 0x00600000,
      IMAGE_SCN_ALIGN_64BYTES                   = 0x00700000,
      IMAGE_SCN_LNK_NRELOC_OVFL                 = 0x01000000,
      IMAGE_SCN_MEM_DISCARDABLE                 = 0x02000000,
      IMAGE_SCN_MEM_NOT_CACHED                  = 0x04000000,
      IMAGE_SCN_MEM_NOT_PAGED                   = 0x08000000,
      IMAGE_SCN_MEM_SHARED                      = 0x10000000,
      IMAGE_SCN_MEM_EXECUTE                     = 0x20000000,
      IMAGE_SCN_MEM_READ                        = 0x40000000,
      IMAGE_SCN_MEM_WRITE                       = 0x80000000
    };

    StringRef getSectionName() const { return SectionName; }
    unsigned getFlags() const { return Flags; }
    
    virtual void PrintSwitchToSection(const MCAsmInfo &MAI,
                                      raw_ostream &OS) const;
  };

} // end namespace llvm

#endif
