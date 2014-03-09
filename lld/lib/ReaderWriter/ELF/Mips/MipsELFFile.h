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
};

} // elf
} // lld

#endif
