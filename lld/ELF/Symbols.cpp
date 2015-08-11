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

DefinedRegular::DefinedRegular(StringRef Name)
    : Defined(DefinedRegularKind, Name) {}

DefinedWeak::DefinedWeak(StringRef Name) : Defined(DefinedWeakKind, Name) {}

// Returns 1, 0 or -1 if this symbol should take precedence
// over the Other, tie or lose, respectively.
int SymbolBody::compare(SymbolBody *Other) {
  Kind LK = kind();
  Kind RK = Other->kind();

  // Normalize so that the smaller kind is on the left.
  if (LK > RK)
    return -Other->compare(this);

  // First handle comparisons between two different kinds.
  if (LK != RK)
    return 1;

  // Now handle the case where the kinds are the same.
  switch (LK) {
  case DefinedRegularKind:
    return 0;
  case DefinedWeakKind:
  case UndefinedKind:
  case UndefinedWeakKind:
    return 1;
  }
  llvm_unreachable("unknown symbol kind");
}
