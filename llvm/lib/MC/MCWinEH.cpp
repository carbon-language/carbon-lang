//===- lib/MC/MCWinEH.cpp - Windows EH implementation ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCSectionCOFF.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCWinEH.h"
#include "llvm/Support/COFF.h"

namespace llvm {
namespace WinEH {
const MCSection *UnwindEmitter::GetPDataSection(StringRef Suffix,
                                                MCContext &Context) {
  if (Suffix.empty())
    return Context.getObjectFileInfo()->getPDataSection();
  return Context.getCOFFSection((".pdata" + Suffix).str(),
                                COFF::IMAGE_SCN_CNT_INITIALIZED_DATA |
                                COFF::IMAGE_SCN_MEM_READ,
                                SectionKind::getDataRel());
}

const MCSection *UnwindEmitter::GetXDataSection(StringRef Suffix,
                                                MCContext &Context) {
  if (Suffix.empty())
    return Context.getObjectFileInfo()->getXDataSection();
  return Context.getCOFFSection((".xdata" + Suffix).str(),
                                COFF::IMAGE_SCN_CNT_INITIALIZED_DATA |
                                COFF::IMAGE_SCN_MEM_READ,
                                SectionKind::getDataRel());
}

StringRef UnwindEmitter::GetSectionSuffix(const MCSymbol *Function) {
  if (!Function || !Function->isInSection())
    return "";

  const MCSection *FunctionSection = &Function->getSection();
  if (const auto Section = dyn_cast<MCSectionCOFF>(FunctionSection)) {
    StringRef Name = Section->getSectionName();
    size_t Dollar = Name.find('$');
    size_t Dot = Name.find('.', 1);

    if (Dollar == StringRef::npos && Dot == StringRef::npos)
      return "";
    if (Dot == StringRef::npos)
      return Name.substr(Dollar);
    if (Dollar == StringRef::npos || Dot < Dollar)
      return Name.substr(Dot);

    return Name.substr(Dollar);
  }

  return "";
}
}
}

