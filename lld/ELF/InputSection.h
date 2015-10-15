//===- InputSection.h -------------------------------------------*- C++ -*-===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_ELF_INPUT_SECTION_H
#define LLD_ELF_INPUT_SECTION_H

#include "lld/Core/LLVM.h"
#include "llvm/Object/ELF.h"

namespace lld {
namespace elf2 {

template <class ELFT> class ObjectFile;
template <class ELFT> class OutputSection;
template <class ELFT> class OutputSectionBase;

// This corresponds to a section of an input file.
template <class ELFT> class InputSection {
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Shdr Elf_Shdr;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Rela Elf_Rela;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Rel Elf_Rel;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Sym Elf_Sym;
  typedef typename llvm::object::ELFFile<ELFT>::uintX_t uintX_t;

public:
  InputSection(ObjectFile<ELFT> *F, const Elf_Shdr *Header);

  // Returns the size of this section (even if this is a common or BSS.)
  size_t getSize() const { return Header->sh_size; }

  // Write this section to a mmap'ed file, assuming Buf is pointing to
  // beginning of the output section.
  void writeTo(uint8_t *Buf);

  StringRef getSectionName() const;
  const Elf_Shdr *getSectionHdr() const { return Header; }
  ObjectFile<ELFT> *getFile() const { return File; }

  // The writer sets and uses the addresses.
  uintX_t getAlign() {
    // The ELF spec states that a value of 0 means the section has no alignment
    // constraits.
    return std::max<uintX_t>(Header->sh_addralign, 1);
  }

  // Relocation sections that refer to this one.
  SmallVector<const Elf_Shdr *, 1> RelocSections;

  // The offset from beginning of the output sections this section was assigned
  // to. The writer sets a value.
  uint64_t OutSecOff = 0;
  OutputSectionBase<ELFT> *OutSec = nullptr;

  static InputSection<ELFT> Discarded;

private:
  static void relocateOne(uint8_t *Buf, uint8_t *BufEnd, const Elf_Rel &Rel,
                          uint32_t Type, uintX_t BaseAddr, uintX_t SymVA);

  static void relocateOne(uint8_t *Buf, uint8_t *BufEnd, const Elf_Rela &Rel,
                          uint32_t Type, uintX_t BaseAddr, uintX_t SymVA);

  template <bool isRela>
  void relocate(uint8_t *Buf, uint8_t *BufEnd,
                llvm::iterator_range<
                    const llvm::object::Elf_Rel_Impl<ELFT, isRela> *> Rels,
                const ObjectFile<ELFT> &File, uintX_t BaseAddr);

  // The file this section is from.
  ObjectFile<ELFT> *File;

  const Elf_Shdr *Header;
};

template <class ELFT>
InputSection<ELFT> InputSection<ELFT>::Discarded(nullptr, nullptr);

} // namespace elf2
} // namespace lld

#endif
