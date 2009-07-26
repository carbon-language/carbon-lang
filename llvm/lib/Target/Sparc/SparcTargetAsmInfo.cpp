//===-- SparcTargetAsmInfo.cpp - Sparc asm properties -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the SparcTargetAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "SparcTargetAsmInfo.h"
#include "llvm/ADT/SmallVector.h"
using namespace llvm;

SparcELFTargetAsmInfo::SparcELFTargetAsmInfo(const TargetMachine &TM)
  : ELFTargetAsmInfo(TM) {
  Data16bitsDirective = "\t.half\t";
  Data32bitsDirective = "\t.word\t";
  Data64bitsDirective = 0;  // .xword is only supported by V9.
  ZeroDirective = "\t.skip\t";
  CommentString = "!";
  ConstantPoolSection = "\t.section \".rodata\",#alloc\n";
  COMMDirectiveTakesAlignment = true;
  CStringSection=".rodata.str";

  // Sparc normally uses named section for BSS.
  BSSSection_ = getNamedSection("\t.bss",
                                SectionFlags::Writable | SectionFlags::BSS);
}


void SparcELFTargetAsmInfo::getSectionFlags(unsigned Flags,
                                            SmallVectorImpl<char> &Str) const {
  if (Flags & SectionFlags::Mergeable)
    return ELFTargetAsmInfo::getSectionFlags(Flags, Str);

  // FIXME: Inefficient.
  std::string Res;
  if (!(Flags & SectionFlags::Debug))
    Res += ",#alloc";
  if (Flags & SectionFlags::Code)
    Res += ",#execinstr";
  if (Flags & SectionFlags::Writable)
    Res += ",#write";
  if (Flags & SectionFlags::TLS)
    Res += ",#tls";

  Str.append(Res.begin(), Res.end());
}
