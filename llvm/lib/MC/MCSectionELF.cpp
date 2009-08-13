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
#include "llvm/Target/TargetAsmInfo.h"

using namespace llvm;

MCSectionELF *MCSectionELF::
Create(const StringRef &Section, unsigned Type, unsigned Flags,
       SectionKind K, bool hasCrazyBSS, bool isExplicit, MCContext &Ctx) {
  return new 
    (Ctx) MCSectionELF(Section, Type, Flags, K, hasCrazyBSS, isExplicit);
}

// ShouldOmitSectionDirective - Decides whether a '.section' directive
// should be printed before the section name
bool MCSectionELF::ShouldOmitSectionDirective(const char *Name) const {
  
  // PPC/Linux doesn't support the .bss directive, it needs .section .bss.
  // FIXME: Does .section .bss/.data/.text work everywhere??
  if ((!HasCrazyBSS && strncmp(Name, ".bss", 4) == 0) || 
      strncmp(Name, ".text", 5) == 0 || 
      strncmp(Name, ".data", 5) == 0)
    return true;

  return false;
}

// ShouldPrintSectionType - Only prints the section type if supported
bool MCSectionELF::ShouldPrintSectionType(unsigned Ty) const {
  
  if (IsExplicit && !(Ty == SHT_NOBITS || Ty == SHT_PROGBITS))
    return false;

  return true;
}

void MCSectionELF::PrintSwitchToSection(const TargetAsmInfo &TAI,
                                        raw_ostream &OS) const {
  
  if (ShouldOmitSectionDirective(SectionName.c_str())) {
    OS << '\t' << getSectionName() << '\n';
    return;
  }

  OS << "\t.section\t" << getSectionName();
  
  // Handle the weird solaris syntax if desired.
  if (TAI.usesSunStyleELFSectionSwitchSyntax() && 
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
   
    OS << '"';

    if (ShouldPrintSectionType(Type)) {
      OS << ',';
   
      // If comment string is '@', e.g. as on ARM - use '%' instead
      if (TAI.getCommentString()[0] == '@')
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

// IsCommon - True if this section contains only common symbols
bool MCSectionELF::IsCommon() const {
  
  if (strncmp(SectionName.c_str(), ".gnu.linkonce.", 14) == 0)
    return true;

  return false;
}


