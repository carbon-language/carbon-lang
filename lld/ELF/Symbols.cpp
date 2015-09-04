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

#include "llvm/ADT/STLExtras.h"

using namespace llvm;
using namespace llvm::object;
using namespace llvm::ELF;

using namespace lld;
using namespace lld::elf2;

static uint8_t getMinVisibility(uint8_t VA, uint8_t VB) {
  if (VA == STV_DEFAULT)
    return VB;
  if (VB == STV_DEFAULT)
    return VA;
  return std::min(VA, VB);
}

// Returns 1, 0 or -1 if this symbol should take precedence
// over the Other, tie or lose, respectively.
template <class ELFT> int SymbolBody::compare(SymbolBody *Other) {
  assert(!isLazy() && !Other->isLazy());
  std::pair<bool, bool> L(isDefined(), !isWeak());
  std::pair<bool, bool> R(Other->isDefined(), !Other->isWeak());

  // Normalize
  if (L > R)
    return -Other->compare<ELFT>(this);

  uint8_t LV = getMostConstrainingVisibility();
  uint8_t RV = Other->getMostConstrainingVisibility();
  MostConstrainingVisibility = getMinVisibility(LV, RV);
  Other->MostConstrainingVisibility = MostConstrainingVisibility;

  if (L != R)
    return -1;

  if (L.first && L.second) {
    if (isCommon()) {
      if (Other->isCommon()) {
        auto *ThisC = cast<DefinedCommon<ELFT>>(this);
        auto *OtherC = cast<DefinedCommon<ELFT>>(Other);
        typename DefinedCommon<ELFT>::uintX_t MaxAlign =
            std::max(ThisC->MaxAlignment, OtherC->MaxAlignment);
        if (ThisC->Sym.st_size >= OtherC->Sym.st_size) {
          ThisC->MaxAlignment = MaxAlign;
          return 1;
        }
        OtherC->MaxAlignment = MaxAlign;
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

std::unique_ptr<InputFile> Lazy::getMember() {
  MemoryBufferRef MBRef = File->getMember(&Sym);

  // getMember returns an empty buffer if the member was already
  // read from the library.
  if (MBRef.getBuffer().empty())
    return std::unique_ptr<InputFile>(nullptr);

  return createELFFile<ObjectFile>(MBRef);
}

template int SymbolBody::compare<ELF32LE>(SymbolBody *Other);
template int SymbolBody::compare<ELF32BE>(SymbolBody *Other);
template int SymbolBody::compare<ELF64LE>(SymbolBody *Other);
template int SymbolBody::compare<ELF64BE>(SymbolBody *Other);
