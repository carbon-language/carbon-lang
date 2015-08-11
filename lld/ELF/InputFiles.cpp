//===- InputFiles.cpp -----------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "InputFiles.h"
#include "Chunks.h"
#include "Error.h"
#include "Symbols.h"
#include "llvm/ADT/STLExtras.h"

using namespace llvm::ELF;

using namespace lld;
using namespace lld::elf2;

template <class ELFT>
bool ObjectFile<ELFT>::isCompatibleWith(const ObjectFileBase &Other) const {
  if (kind() != Other.kind())
    return false;
  return getObj()->getHeader()->e_machine ==
         cast<ObjectFile<ELFT>>(Other).getObj()->getHeader()->e_machine;
}

template <class ELFT> void elf2::ObjectFile<ELFT>::parse() {
  // Parse a memory buffer as a ELF file.
  std::error_code EC;
  ELFObj = llvm::make_unique<ELFFile<ELFT>>(MB.getBuffer(), EC);
  error(EC);

  // Read section and symbol tables.
  initializeChunks();
  initializeSymbols();
}

template <class ELFT> void elf2::ObjectFile<ELFT>::initializeChunks() {
  uint64_t Size = ELFObj->getNumSections();
  Chunks.reserve(Size);
  for (const Elf_Shdr &Sec : ELFObj->sections()) {
    if (Sec.sh_type == SHT_SYMTAB) {
      Symtab = &Sec;
      continue;
    }
    if (Sec.sh_flags & SHF_ALLOC) {
      auto *C = new (Alloc) SectionChunk<ELFT>(this->getObj(), &Sec);
      Chunks.push_back(C);
    }
  }
}

template <class ELFT> void elf2::ObjectFile<ELFT>::initializeSymbols() {
  ErrorOr<StringRef> StringTableOrErr =
      ELFObj->getStringTableForSymtab(*Symtab);
  error(StringTableOrErr.getError());
  StringRef StringTable = *StringTableOrErr;

  Elf_Sym_Range Syms = ELFObj->symbols(Symtab);
  auto NumSymbols = std::distance(Syms.begin(), Syms.end());
  uint32_t FirstNonLocal = Symtab->sh_info;
  if (FirstNonLocal > NumSymbols)
    error("Invalid sh_info in symbol table");
  Syms = llvm::make_range(Syms.begin() + FirstNonLocal, Syms.end());
  SymbolBodies.reserve(NumSymbols);
  for (const Elf_Sym &Sym : Syms)
    SymbolBodies.push_back(createSymbolBody(StringTable, &Sym));
}

template <class ELFT>
SymbolBody *elf2::ObjectFile<ELFT>::createSymbolBody(StringRef StringTable,
                                                     const Elf_Sym *Sym) {
  ErrorOr<StringRef> NameOrErr = Sym->getName(StringTable);
  error(NameOrErr.getError());
  StringRef Name = *NameOrErr;
  switch (Sym->getBinding()) {
  default:
    error("unexpected binding");
  case STB_GLOBAL:
    if (Sym->isUndefined())
      return new (Alloc) Undefined(Name);
    return new (Alloc) DefinedRegular<ELFT>(Name);
  case STB_WEAK:
    // FIXME: add support for weak undefined
    return new (Alloc) DefinedWeak<ELFT>(Name);
  }
}

namespace lld {
namespace elf2 {
template class elf2::ObjectFile<llvm::object::ELF32LE>;
template class elf2::ObjectFile<llvm::object::ELF32BE>;
template class elf2::ObjectFile<llvm::object::ELF64LE>;
template class elf2::ObjectFile<llvm::object::ELF64BE>;
}
}
