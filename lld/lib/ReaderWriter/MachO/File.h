//===- lib/ReaderWriter/MachO/File.h --------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_MACHO_FILE_H
#define LLD_READER_WRITER_MACHO_FILE_H

#include "Atoms.h"

#include "lld/ReaderWriter/Simple.h"

namespace lld {
namespace mach_o {

class MachOFile : public SimpleFile {
public:
  MachOFile(StringRef path) : SimpleFile(path) {}

  void addDefinedAtom(StringRef name, ArrayRef<uint8_t> content) {
    MachODefinedAtom *atom =
        new (_allocator) MachODefinedAtom(*this, name, content);
    addAtom(*atom);
  }

private:
  llvm::BumpPtrAllocator _allocator;
};

} // end namespace mach_o
} // end namespace lld

#endif
