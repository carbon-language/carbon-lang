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
#include "llvm/MC/MCContext.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

  class MCSectionPIC16 : public MCSection {
    MCSectionPIC16(const StringRef &Name, bool IsDirective, SectionKind K,
                 MCContext &Ctx) : MCSection(Name, IsDirective, K, Ctx) {}
  public:
    
    static MCSectionPIC16 *Create(const StringRef &Name, bool IsDirective, 
                                  SectionKind K, MCContext &Ctx) {
      return new (Ctx) MCSectionPIC16(Name, IsDirective, K, Ctx);
    }
    
    
    virtual void PrintSwitchToSection(const TargetAsmInfo &TAI,
                                      raw_ostream &OS) const {
      OS << getName() << '\n';
    }

  };

} // end namespace llvm

#endif
