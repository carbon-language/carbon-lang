//===- Chunks.cpp ---------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Chunks.h"
#include "Error.h"

using namespace llvm;
using namespace llvm::ELF;

using namespace lld;
using namespace lld::elf2;

template <class ELFT>
SectionChunk<ELFT>::SectionChunk(object::ELFFile<ELFT> *Obj,
                                 const Elf_Shdr *Header)
    : Obj(Obj), Header(Header) {
  Align = Header->sh_addralign;
}

template <class ELFT> void SectionChunk<ELFT>::writeTo(uint8_t *Buf) {
  if (Header->sh_type == SHT_NOBITS)
    return;
  // Copy section contents from source object file to output file.
  ArrayRef<uint8_t> Data = *Obj->getSectionContents(Header);
  memcpy(Buf + OutputSectionOff, Data.data(), Data.size());

  // FIXME: Relocations
}

template <class ELFT> StringRef SectionChunk<ELFT>::getSectionName() const {
  ErrorOr<StringRef> Name = Obj->getSectionName(Header);
  error(Name);
  return *Name;
}

namespace lld {
namespace elf2 {
template class SectionChunk<object::ELF32LE>;
template class SectionChunk<object::ELF32BE>;
template class SectionChunk<object::ELF64LE>;
template class SectionChunk<object::ELF64BE>;
}
}
