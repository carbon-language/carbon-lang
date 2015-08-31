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
#include "Error.h"
#include "InputFiles.h"

using namespace llvm::object;

using namespace lld;
using namespace lld::elf2;

// Returns 1, 0 or -1 if this symbol should take precedence
// over the Other, tie or lose, respectively.
template <class ELFT> int SymbolBody::compare(SymbolBody *Other) {
  std::pair<bool, bool> L(isDefined(), !isWeak());
  std::pair<bool, bool> R(Other->isDefined(), !Other->isWeak());

  // Normalize
  if (L > R)
    return -Other->compare<ELFT>(this);

  if (L != R)
    return -1;

  if (L.first && L.second) {
    if (isCommon()) {
      if (Other->isCommon()) {
        // FIXME: We also need to remember the alignment restriction.
        if (cast<DefinedCommon<ELFT>>(this)->Sym.st_size >=
            cast<DefinedCommon<ELFT>>(Other)->Sym.st_size)
          return 1;
        return -1;
      }
      return -1;
    }
    if (Other->isCommon())
      return 1;
    return 0;
  }
  return 1;
}

template int SymbolBody::compare<ELF32LE>(SymbolBody *Other);
template int SymbolBody::compare<ELF32BE>(SymbolBody *Other);
template int SymbolBody::compare<ELF64LE>(SymbolBody *Other);
template int SymbolBody::compare<ELF64BE>(SymbolBody *Other);
