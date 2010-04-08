//===-- SystemZMCAsmInfo.cpp - SystemZ asm properties ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the SystemZMCAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "SystemZMCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSectionELF.h"
using namespace llvm;

SystemZMCAsmInfo::SystemZMCAsmInfo(const Target &T, const StringRef &TT) {
  PrivateGlobalPrefix = ".L";
  WeakRefDirective = "\t.weak\t";
  PCSymbol = ".";
}

const MCSection *SystemZMCAsmInfo::
getNonexecutableStackSection(MCContext &Ctx) const{
  return Ctx.getELFSection(".note.GNU-stack", MCSectionELF::SHT_PROGBITS,
                           0, SectionKind::getMetadata(), false);
}
