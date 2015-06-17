//===- lib/ReaderWriter/ELF/X86_64/X86_64SectionChunks.cpp --------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "X86_64SectionChunks.h"
#include "TargetLayout.h"

namespace lld {
namespace elf {

X86_64GOTSection::X86_64GOTSection(const ELFLinkingContext &ctx)
  : AtomSection<ELF64LE>(ctx, ".got", DefinedAtom::typeGOT, DefinedAtom::permRW_,
    TargetLayout<ELF64LE>::ORDER_GOT) {
  this->_alignment = 8;
}

const AtomLayout *X86_64GOTSection::appendAtom(const Atom *atom) {
  const DefinedAtom *da = dyn_cast<DefinedAtom>(atom);

  for (const auto &r : *da) {
    if (r->kindNamespace() != Reference::KindNamespace::ELF)
      continue;
    assert(r->kindArch() == Reference::KindArch::x86_64);
    if (r->kindValue() == R_X86_64_TPOFF64)
      _tlsMap[r->target()] = _tlsMap.size();
  }

  return AtomSection<ELF64LE>::appendAtom(atom);
}

} // elf
} // lld
