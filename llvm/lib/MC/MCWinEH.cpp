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
static StringRef getSectionSuffix(const MCSymbol *Function) {
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

static const MCSection *getUnwindInfoSection(
    StringRef SecName, const MCSectionCOFF *UnwindSec, const MCSymbol *Function,
    MCContext &Context) {
  // If Function is in a COMDAT, get or create an unwind info section in that
  // COMDAT group.
  if (Function && Function->isInSection()) {
    const MCSectionCOFF *FunctionSection =
        cast<MCSectionCOFF>(&Function->getSection());
    if (FunctionSection->getCharacteristics() & COFF::IMAGE_SCN_LNK_COMDAT) {
      return Context.getAssociativeCOFFSection(
          UnwindSec, FunctionSection->getCOMDATSymbol());
    }
  }

  // If Function is in a section other than .text, create a new .pdata section.
  // Otherwise use the plain .pdata section.
  StringRef Suffix = getSectionSuffix(Function);
  if (Suffix.empty())
    return UnwindSec;
  return Context.getCOFFSection((SecName + Suffix).str(),
                                COFF::IMAGE_SCN_CNT_INITIALIZED_DATA |
                                COFF::IMAGE_SCN_MEM_READ,
                                SectionKind::getDataRel());
}

const MCSection *UnwindEmitter::getPDataSection(const MCSymbol *Function,
                                                MCContext &Context) {
  const MCSectionCOFF *PData =
      cast<MCSectionCOFF>(Context.getObjectFileInfo()->getPDataSection());
  return getUnwindInfoSection(".pdata", PData, Function, Context);
}

const MCSection *UnwindEmitter::getXDataSection(const MCSymbol *Function,
                                                MCContext &Context) {
  const MCSectionCOFF *XData =
      cast<MCSectionCOFF>(Context.getObjectFileInfo()->getXDataSection());
  return getUnwindInfoSection(".xdata", XData, Function, Context);
}

}
}

