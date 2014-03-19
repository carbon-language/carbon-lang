//===- lib/ReaderWriter/ELF/MipsELFFile.h ---------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef LLD_READER_WRITER_ELF_MIPS_MIPS_ELF_FILE_H
#define LLD_READER_WRITER_ELF_MIPS_MIPS_ELF_FILE_H

#include "ELFReader.h"
#include "MipsLinkingContext.h"

namespace lld {
namespace elf {

template <class ELFT> class MipsELFFile : public ELFFile<ELFT> {
public:
  MipsELFFile(StringRef name, bool atomizeStrings)
      : ELFFile<ELFT>(name, atomizeStrings) {}

  MipsELFFile(std::unique_ptr<MemoryBuffer> mb, bool atomizeStrings,
              error_code &ec)
      : ELFFile<ELFT>(std::move(mb), atomizeStrings, ec) {}

  static ErrorOr<std::unique_ptr<MipsELFFile>>
  create(std::unique_ptr<MemoryBuffer> mb, bool atomizeStrings) {
    error_code ec;
    std::unique_ptr<MipsELFFile<ELFT>> file(
        new MipsELFFile<ELFT>(mb->getBufferIdentifier(), atomizeStrings));

    file->_objFile.reset(new llvm::object::ELFFile<ELFT>(mb.release(), ec));

    if (ec)
      return ec;

    // Read input sections from the input file that need to be converted to
    // atoms
    if ((ec = file->createAtomizableSections()))
      return ec;

    // For mergeable strings, we would need to split the section into various
    // atoms
    if ((ec = file->createMergeableAtoms()))
      return ec;

    // Create the necessary symbols that are part of the section that we
    // created in createAtomizableSections function
    if ((ec = file->createSymbolsFromAtomizableSections()))
      return ec;

    // Create the appropriate atoms from the file
    if ((ec = file->createAtoms()))
      return ec;

    return std::move(file);
  }

private:
  typedef llvm::object::Elf_Sym_Impl<ELFT> Elf_Sym;
  typedef llvm::object::Elf_Rel_Impl<ELFT, false> Elf_Rel;

  ELFReference<ELFT> *
  createRelocationReference(const Elf_Sym &symbol, const Elf_Rel &ri,
                            ArrayRef<uint8_t> content) override {
    bool isMips64EL = this->_objFile->isMips64EL();
    auto *ref = new (this->_readerStorage)
        ELFReference<ELFT>(&ri, ri.r_offset - symbol.st_value, this->kindArch(),
                           ri.getType(isMips64EL), ri.getSymbol(isMips64EL));
    const uint8_t *ap = content.data() + ri.r_offset - symbol.st_value;
    switch (ri.getType(isMips64EL)) {
    case R_MIPS_32:
      ref->setAddend(*(int32_t *)ap);
      break;
    case R_MIPS_26:
      ref->setAddend(*(int32_t *)ap & 0x3ffffff);
      break;
    case R_MIPS_HI16:
    case R_MIPS_LO16:
    case R_MIPS_GOT16:
      ref->setAddend(*(int16_t *)ap);
      break;
    }
    return ref;
  }
};

} // elf
} // lld

#endif
