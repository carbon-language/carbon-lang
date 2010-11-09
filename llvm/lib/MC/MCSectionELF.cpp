//===- lib/MC/MCSectionELF.cpp - ELF Code Section Representation ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

MCSectionELF::~MCSectionELF() {} // anchor.

// ShouldOmitSectionDirective - Decides whether a '.section' directive
// should be printed before the section name
bool MCSectionELF::ShouldOmitSectionDirective(StringRef Name,
                                              const MCAsmInfo &MAI) const {
  
  // FIXME: Does .section .bss/.data/.text work everywhere??
  if (Name == ".text" || Name == ".data" ||
      (Name == ".bss" && !MAI.usesELFSectionDirectiveForBSS()))
    return true;

  return false;
}

void MCSectionELF::PrintSwitchToSection(const MCAsmInfo &MAI,
                                        raw_ostream &OS) const {
   
  if (ShouldOmitSectionDirective(SectionName, MAI)) {
    OS << '\t' << getSectionName() << '\n';
    return;
  }

  OS << "\t.section\t" << getSectionName();
  
  // Handle the weird solaris syntax if desired.
  if (MAI.usesSunStyleELFSectionSwitchSyntax() && 
      !(Flags & MCSectionELF::SHF_MERGE)) {
    if (Flags & MCSectionELF::SHF_ALLOC)
      OS << ",#alloc";
    if (Flags & MCSectionELF::SHF_EXECINSTR)
      OS << ",#execinstr";
    if (Flags & MCSectionELF::SHF_WRITE)
      OS << ",#write";
    if (Flags & MCSectionELF::SHF_TLS)
      OS << ",#tls";
    OS << '\n';
    return;
  }
  
  OS << ",\"";
  if (Flags & MCSectionELF::SHF_ALLOC)
    OS << 'a';
  if (Flags & MCSectionELF::SHF_EXECINSTR)
    OS << 'x';
  if (Flags & MCSectionELF::SHF_WRITE)
    OS << 'w';
  if (Flags & MCSectionELF::SHF_MERGE)
    OS << 'M';
  if (Flags & MCSectionELF::SHF_STRINGS)
    OS << 'S';
  if (Flags & MCSectionELF::SHF_TLS)
    OS << 'T';
  
  // If there are target-specific flags, print them.
  if (Flags & MCSectionELF::XCORE_SHF_CP_SECTION)
    OS << 'c';
  if (Flags & MCSectionELF::XCORE_SHF_DP_SECTION)
    OS << 'd';
  
  OS << '"';

  OS << ',';

  // If comment string is '@', e.g. as on ARM - use '%' instead
  if (MAI.getCommentString()[0] == '@')
    OS << '%';
  else
    OS << '@';

  if (Type == MCSectionELF::SHT_INIT_ARRAY)
    OS << "init_array";
  else if (Type == MCSectionELF::SHT_FINI_ARRAY)
    OS << "fini_array";
  else if (Type == MCSectionELF::SHT_PREINIT_ARRAY)
    OS << "preinit_array";
  else if (Type == MCSectionELF::SHT_NOBITS)
    OS << "nobits";
  else if (Type == MCSectionELF::SHT_PROGBITS)
    OS << "progbits";

  if (EntrySize) {
    assert(Flags & MCSectionELF::SHF_MERGE);
    OS << "," << EntrySize;
  }

  OS << '\n';
}

bool MCSectionELF::UseCodeAlign() const {
  return getFlags() & MCSectionELF::SHF_EXECINSTR;
}

// HasCommonSymbols - True if this section holds common symbols, this is
// indicated on the ELF object file by a symbol with SHN_COMMON section 
// header index.
bool MCSectionELF::HasCommonSymbols() const {
  
  if (StringRef(SectionName).startswith(".gnu.linkonce."))
    return true;

  return false;
}

unsigned MCSectionELF::DetermineEntrySize(SectionKind Kind) {
  if (Kind.isMergeable1ByteCString()) return 1;
  if (Kind.isMergeable2ByteCString()) return 2;
  if (Kind.isMergeable4ByteCString()) return 4;
  if (Kind.isMergeableConst4())       return 4;
  if (Kind.isMergeableConst8())       return 8;
  if (Kind.isMergeableConst16())      return 16;
  return 0;
}
