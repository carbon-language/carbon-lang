//===- lib/ReaderWriter/ELF/X86_64/X86_64ELFFile.h ------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_X86_64_ELF_FILE_H
#define LLD_READER_WRITER_ELF_X86_64_ELF_FILE_H

#include "ELFReader.h"

namespace lld {
namespace elf {

class X86_64LinkingContext;

template <class ELFT> class X86_64ELFFile : public ELFFile<ELFT> {
public:
  X86_64ELFFile(StringRef name, bool atomizeStrings)
      : ELFFile<ELFT>(name, atomizeStrings) {}

  X86_64ELFFile(std::unique_ptr<MemoryBuffer> mb, bool atomizeStrings,
                std::error_code &ec)
      : ELFFile<ELFT>(std::move(mb), atomizeStrings, ec) {}

  static ErrorOr<std::unique_ptr<X86_64ELFFile>>
  create(std::unique_ptr<MemoryBuffer> mb, bool atomizeStrings) {
    std::error_code ec;
    std::unique_ptr<X86_64ELFFile<ELFT>> file(
        new X86_64ELFFile<ELFT>(mb->getBufferIdentifier(), atomizeStrings));

    file->_objFile.reset(
        new llvm::object::ELFFile<ELFT>(mb.release()->getBuffer(), ec));

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

template <class ELFT> class X86_64DynamicFile : public DynamicFile<ELFT> {
public:
  X86_64DynamicFile(const X86_64LinkingContext &context, StringRef name)
      : DynamicFile<ELFT>(context, name) {}
};

} // elf
} // lld

#endif // LLD_READER_WRITER_ELF_X86_64_ELF_FILE_H
