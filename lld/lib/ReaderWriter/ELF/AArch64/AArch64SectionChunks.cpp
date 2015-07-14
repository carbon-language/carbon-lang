//===- lib/ReaderWriter/ELF/AArch64/AArch64SectionChunks.cpp --------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "AArch64SectionChunks.h"
#include "TargetLayout.h"

namespace lld {
namespace elf {

AArch64GOTSection::AArch64GOTSection(const ELFLinkingContext &ctx,
  StringRef name, int32_t order)
  : AtomSection<ELF64LE>(ctx, name, DefinedAtom::typeGOT, DefinedAtom::permRW_,
    order) {
  _alignment = 8;
}

const AtomLayout *AArch64GOTSection::appendAtom(const Atom *atom) {
  const DefinedAtom *da = dyn_cast<DefinedAtom>(atom);

  for (const auto &r : *da) {
    if (r->kindNamespace() != Reference::KindNamespace::ELF)
      continue;
    assert(r->kindArch() == Reference::KindArch::AArch64);
    if ((r->kindValue() == R_AARCH64_TLS_TPREL64) ||
        (r->kindValue() == R_AARCH64_TLSDESC))
      _tlsMap[r->target()] = _tlsMap.size();
  }

  return AtomSection<ELF64LE>::appendAtom(atom);
}

} // elf
} // lld
