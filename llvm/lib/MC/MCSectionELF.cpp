//===- lib/MC/MCSectionELF.cpp - ELF Code Section Representation ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCContext.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/MC/MCAsmInfo.h"

using namespace llvm;

MCSectionELF *MCSectionELF::
Create(StringRef Section, unsigned Type, unsigned Flags,
       SectionKind K, bool isExplicit, MCContext &Ctx) {
  return new (Ctx) MCSectionELF(Section, Type, Flags, K, isExplicit);
}

// ShouldOmitSectionDirective - Decides whether a '.section' directive
// should be printed before the section name
bool MCSectionELF::ShouldOmitSectionDirective(const char *Name,
                                        const MCAsmInfo &MAI) const {
  
  // FIXME: Does .section .bss/.data/.text work everywhere??
  if (strcmp(Name, ".text") == 0 ||
      strcmp(Name, ".data") == 0 ||
      (strcmp(Name, ".bss") == 0 &&
       !MAI.usesELFSectionDirectiveForBSS())) 
    return true;

  return false;
}

// ShouldPrintSectionType - Only prints the section type if supported
bool MCSectionELF::ShouldPrintSectionType(unsigned Ty) const {
  
  if (IsExplicit && !(Ty == SHT_NOBITS || Ty == SHT_PROGBITS))
    return false;

  return true;
}

void MCSectionELF::PrintSwitchToSection(const MCAsmInfo &MAI,
                                        raw_ostream &OS) const {
   
  if (ShouldOmitSectionDirective(SectionName.c_str(), MAI)) {
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
  } else {
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
    if (Flags & ~MCSectionELF::TARGET_INDEP_SHF)
      PrintTargetSpecificSectionFlags(MAI, OS);
    
    OS << '"';

    if (ShouldPrintSectionType(Type)) {
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
    
      if (getKind().isMergeable1ByteCString()) {
        OS << ",1";
      } else if (getKind().isMergeable2ByteCString()) {
        OS << ",2";
      } else if (getKind().isMergeable4ByteCString() || 
                 getKind().isMergeableConst4()) {
        OS << ",4";
      } else if (getKind().isMergeableConst8()) {
        OS << ",8";
      } else if (getKind().isMergeableConst16()) {
        OS << ",16";
      }
    }
  }
  
  OS << '\n';
}

// HasCommonSymbols - True if this section holds common symbols, this is
// indicated on the ELF object file by a symbol with SHN_COMMON section 
// header index.
bool MCSectionELF::HasCommonSymbols() const {
  
  if (strncmp(SectionName.c_str(), ".gnu.linkonce.", 14) == 0)
    return true;

  return false;
}


