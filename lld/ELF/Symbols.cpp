//===- Symbols.cpp --------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Symbols.h"
#include "Chunks.h"

using namespace llvm::object;

using namespace lld;
using namespace lld::elf2;

template <class ELFT>
DefinedRegular<ELFT>::DefinedRegular(StringRef Name)
    : Defined(DefinedRegularKind), Name(Name) {}

// Returns 1, 0 or -1 if this symbol should take precedence
// over the Other, tie or lose, respectively.
template <class ELFT> int DefinedRegular<ELFT>::compare(SymbolBody *Other) {
  if (Other->kind() < kind())
    return -Other->compare(this);
  auto *R = dyn_cast<DefinedRegular>(Other);
  if (!R)
    return 1;

  return 0;
}

int Defined::compare(SymbolBody *Other) {
  if (Other->kind() < kind())
    return -Other->compare(this);
  if (isa<Defined>(Other))
    return 0;
  return 1;
}

int Undefined::compare(SymbolBody *Other) {
  if (Other->kind() < kind())
    return -Other->compare(this);
  return 1;
}

template <class ELFT> StringRef DefinedRegular<ELFT>::getName() { return Name; }

namespace lld {
namespace elf2 {
template class DefinedRegular<llvm::object::ELF32LE>;
template class DefinedRegular<llvm::object::ELF32BE>;
template class DefinedRegular<llvm::object::ELF64LE>;
template class DefinedRegular<llvm::object::ELF64BE>;
}
}
