//===- lib/ReaderWriter/ELF/Writer.h --------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_WRITER_H
#define LLD_READER_WRITER_ELF_WRITER_H

#include "lld/Core/File.h"
#include "lld/Core/InputFiles.h"
#include "lld/ReaderWriter/Writer.h"

namespace lld {
namespace elf {
/// \brief The Writer class is a base class for the linker to write
///        various kinds of ELF files.
class ELFWriter : public Writer {
public:
  ELFWriter() { }

public:
  /// \brief builds the chunks that needs to be written to the output
  ///        ELF file
  virtual void buildChunks(const File &file) = 0;

  /// \brief Writes the chunks into the output file specified by path
  virtual error_code writeFile(const File &file, StringRef path) = 0;

  /// \brief Get the virtual address of \p atom after layout.
  virtual uint64_t addressOfAtom(const Atom *atom) = 0;
};
} // end namespace elf
} // end namespace lld

#endif
