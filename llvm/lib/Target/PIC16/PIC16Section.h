//===- PIC16Section.h - PIC16-specific section representation ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the MCSection class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_PIC16SECTION_H
#define LLVM_PIC16SECTION_H

#include "llvm/MC/MCSection.h"

namespace llvm {

  class MCSectionPIC16 : public MCSection {
    std::string Name;
    
    MCSectionPIC16(const StringRef &name, SectionKind K)
      : MCSection(K), Name(name) {
    }
    
  public:
    
    const std::string &getName() const { return Name; }
    
    static MCSectionPIC16 *Create(const StringRef &Name, 
                                  SectionKind K, MCContext &Ctx);
    
    virtual void PrintSwitchToSection(const TargetAsmInfo &TAI,
                                      raw_ostream &OS) const;
  };

} // end namespace llvm

#endif
