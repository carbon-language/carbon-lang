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
int SymbolBody::compare(SymbolBody *Other) {
  std::pair<bool, bool> L(isDefined(), isWeak());
  std::pair<bool, bool> R(Other->isDefined(), Other->isWeak());

  // Normalize
  if (L > R)
    return -Other->compare(this);

  if (L != R)
    return -1;

  if (L.first && !L.second)
    return 0;
  return 1;
}
