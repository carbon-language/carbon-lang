//===- lib/MC/MCSection.cpp - Machine Code Section Representation ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCContext.h"
#include "llvm/Target/TargetAsmInfo.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
// MCSection
//===----------------------------------------------------------------------===//

MCSection::~MCSection() {
}

MCSection::MCSection(const StringRef &N, bool isDirective, SectionKind K, 
                     MCContext &Ctx)
  : Name(N), IsDirective(isDirective), Kind(K) {
  MCSection *&Entry = Ctx.Sections[Name];
  assert(Entry == 0 && "Multiple sections with the same name created");
  Entry = this;
}

//===----------------------------------------------------------------------===//
// MCSectionELF
//===----------------------------------------------------------------------===//

MCSectionELF *MCSectionELF::
Create(const StringRef &Name, bool IsDirective, SectionKind K, MCContext &Ctx) {
  return new (Ctx) MCSectionELF(Name, IsDirective, K, Ctx);
}

void MCSectionELF::PrintSwitchToSection(const TargetAsmInfo &TAI,
                                        raw_ostream &OS) const {
  if (isDirective()) {
    OS << getName() << '\n';
    return;
  }

  OS << "\t.section\t" << getName();
  
  // Handle the weird solaris syntax if desired.
  if (TAI.usesSunStyleELFSectionSwitchSyntax() &&
      !getKind().isMergeableConst() && !getKind().isMergeableCString()) {
    if (!getKind().isMetadata())
      OS << ",#alloc";
    if (getKind().isText())
      OS << ",#execinstr";
    if (getKind().isWriteable())
      OS << ",#write";
    if (getKind().isThreadLocal())
      OS << ",#tls";
  } else {
    OS << ",\"";
  
    if (!getKind().isMetadata())
      OS << 'a';
    if (getKind().isText())
      OS << 'x';
    if (getKind().isWriteable())
      OS << 'w';
    if (getKind().isMergeable1ByteCString() ||
        getKind().isMergeable2ByteCString() ||
        getKind().isMergeable4ByteCString() ||
        getKind().isMergeableConst4() ||
        getKind().isMergeableConst8() ||
        getKind().isMergeableConst16())
      OS << 'M';
    if (getKind().isMergeable1ByteCString() ||
        getKind().isMergeable2ByteCString() ||
        getKind().isMergeable4ByteCString())
      OS << 'S';
    if (getKind().isThreadLocal())
      OS << 'T';
    
    OS << "\",";
    
    // If comment string is '@', e.g. as on ARM - use '%' instead
    if (TAI.getCommentString()[0] == '@')
      OS << '%';
    else
      OS << '@';
    
    if (getKind().isBSS() || getKind().isThreadBSS())
      OS << "nobits";
    else
      OS << "progbits";
    
    if (getKind().isMergeable1ByteCString()) {
      OS << ",1";
    } else if (getKind().isMergeable2ByteCString()) {
      OS << ",2";
    } else if (getKind().isMergeable4ByteCString()) {
      OS << ",4";
    } else if (getKind().isMergeableConst4()) {
      OS << ",4";
    } else if (getKind().isMergeableConst8()) {
      OS << ",8";
    } else if (getKind().isMergeableConst16()) {
      OS << ",16";
    }
  }
}

//===----------------------------------------------------------------------===//
// MCSectionMachO
//===----------------------------------------------------------------------===//

MCSectionMachO *MCSectionMachO::
Create(const StringRef &Name, bool IsDirective, SectionKind K, MCContext &Ctx) {
  return new (Ctx) MCSectionMachO(Name, IsDirective, K, Ctx);
}

void MCSectionMachO::PrintSwitchToSection(const TargetAsmInfo &TAI,
                                          raw_ostream &OS) const {
  if (!isDirective())
    OS << "\t.section\t" << getName() << '\n';
  else
    OS << getName() << '\n';
}

//===----------------------------------------------------------------------===//
// MCSectionCOFF
//===----------------------------------------------------------------------===//

MCSectionCOFF *MCSectionCOFF::
Create(const StringRef &Name, bool IsDirective, SectionKind K, MCContext &Ctx) {
  return new (Ctx) MCSectionCOFF(Name, IsDirective, K, Ctx);
}

void MCSectionCOFF::PrintSwitchToSection(const TargetAsmInfo &TAI,
                                         raw_ostream &OS) const {
  
  if (isDirective()) {
    OS << getName() << '\n';
    return;
  }
  OS << "\t.section\t" << getName() << ",\"";
  if (getKind().isText())
    OS << 'x';
  if (getKind().isWriteable())
    OS << 'w';
  OS << "\"\n";
}
