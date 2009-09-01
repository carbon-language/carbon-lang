//===- MCSectionPIC16.h - PIC16-specific section representation -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the MCSectionPIC16 class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_PIC16SECTION_H
#define LLVM_PIC16SECTION_H

#include "llvm/MC/MCSection.h"

namespace llvm {

  /// MCSectionPIC16 - Represents a physical section in PIC16 COFF.
  /// Contains data objects.
  ///
  class MCSectionPIC16 : public MCSection {
    /// Name of the section to uniquely identify it.
    std::string Name;

    /// User can specify an address at which a section should be placed. 
    /// Negative value here means user hasn't specified any. 
    int Address; 

    /// Overlay information - Sections with same color can be overlaid on
    /// one another.
    int Color; 

    /// Conatined data objects.
    std::vector<const GlobalVariable *>Items;

    /// Total size of all data objects contained here.
    unsigned Size;
    
    MCSectionPIC16(const StringRef &name, SectionKind K, int addr, int color)
      : MCSection(K), Name(name), Address(addr), Color(color) {
    }
    
  public:
    /// Return the name of the section.
    const std::string &getName() const { return Name; }

    /// Return the Address of the section.
    int getAddress() const { return Address; }

    /// Return the Color of the section.
    int getColor() const { return Color; }

    /// PIC16 Terminology for section kinds is as below.
    /// UDATA - BSS
    /// IDATA - initialized data (equiv to Metadata) 
    /// ROMDATA - ReadOnly.
    /// UDATA_OVR - Sections that can be overlaid. Section of such type is
    ///             used to contain function autos an frame. We can think of
    ///             it as equiv to llvm ThreadBSS)
    /// So, let's have some convenience functions to Map PIC16 Section types 
    /// to SectionKind just for the sake of better readability.
    static SectionKind UDATA_Kind() { return SectionKind::getBSS(); } 
    static SectionKind IDATA_Kind() { return SectionKind::getMetadata(); }
    static SectionKind ROMDATA_Kind() { return SectionKind::getReadOnly(); }
    static SectionKind UDATA_OVR_Kind() { return SectionKind::getThreadBSS(); }

    // If we could just do getKind() == UDATA_Kind() ?
    bool isUDATA_Kind() { return getKind().isBSS(); }
    bool isIDATA_Kind() { return getKind().isMetadata(); }
    bool isROMDATA_Kind() { return getKind().isMetadata(); }
    bool isUDATA_OVR_Kind() { return getKind().isThreadBSS(); }

    /// This would be the only way to create a section. 
    static MCSectionPIC16 *Create(const StringRef &Name, SectionKind K, 
                                  int Address, int Color, MCContext &Ctx);
    
    /// Override this as PIC16 has its own way of printing switching
    /// to a section.
    virtual void PrintSwitchToSection(const MCAsmInfo &MAI,
                                      raw_ostream &OS) const;
  };

} // end namespace llvm

#endif
