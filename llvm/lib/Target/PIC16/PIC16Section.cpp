//===-- PIC16Section.cpp - PIC16 Section ----------- --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "PIC16.h"
#include "PIC16ABINames.h"
#include "PIC16Section.h"
#include "llvm/MC/MCContext.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;


// This is the only way to create a PIC16Section. Sections created here
// do not need to be explicitly deleted as they are managed by auto_ptrs.
PIC16Section *PIC16Section::Create(const StringRef &Name,
                                   PIC16SectionType Ty,
                                   const std::string &Address, 
                                   int Color, MCContext &Ctx) {

  /// Determine the internal SectionKind info.
  /// Users of PIC16Section class should not need to know the internal
  /// SectionKind. They should work only with PIC16SectionType.
  ///
  /// PIC16 Terminology for section kinds is as below.
  /// UDATA - BSS
  /// IDATA - initialized data (equiv to Metadata) 
  /// ROMDATA - ReadOnly.
  /// UDATA_OVR - Sections that can be overlaid. Section of such type is
  ///             used to contain function autos an frame. We can think of
  ///             it as equiv to llvm ThreadBSS)
  /// UDATA_SHR - Shared RAM. Memory area that is mapped to all banks.

  SectionKind K;
  switch (Ty) {
    default: llvm_unreachable ("can not create unknown section type");
    case UDATA_OVR: {
      K = SectionKind::getThreadBSS();
      break;
    }
    case UDATA_SHR:
    case UDATA: {
      K = SectionKind::getBSS();
      break;
    }
    case ROMDATA:
    case IDATA: {
      K = SectionKind::getMetadata();
      break;
    }
    case CODE: {
      K = SectionKind::getText();
      break;
    }
      
  }

  // Create the Section.
  PIC16Section *S = new (Ctx) PIC16Section(Name, K, Address, Color);
  S->T = Ty;
  return S;
}

// A generic way to print all types of sections.
void PIC16Section::PrintSwitchToSection(const MCAsmInfo &MAI,
                                          raw_ostream &OS) const {
  // Print name.
  OS << getName() << '\t';

  // Print type.
  switch (getType()) {
  default : llvm_unreachable ("unknown section type"); 
  case UDATA: OS << "UDATA"; break;
  case IDATA: OS << "IDATA"; break;
  case ROMDATA: OS << "ROMDATA"; break;
  case UDATA_SHR: OS << "UDATA_SHR"; break;
  case UDATA_OVR: OS << "UDATA_OVR"; break;
  case CODE: OS << "CODE"; break;
  }

  OS << '\t';

  // Print Address.
  OS << Address;

  OS << '\n';
}
