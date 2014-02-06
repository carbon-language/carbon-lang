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

  void addDefinedAtom(StringRef name, ArrayRef<uint8_t> content,
                      Atom::Scope scope, bool copyRefs) {
    if (copyRefs) {
      // Make a copy of the atom's name and content that is owned by this file.
      name = name.copy(_allocator);
      content = content.copy(_allocator);
    }
    MachODefinedAtom *atom =
        new (_allocator) MachODefinedAtom(*this, name, content, scope);
    addAtom(*atom);
  }

  void addUndefinedAtom(StringRef name, bool copyRefs) {
    if (copyRefs) {
      // Make a copy of the atom's name that is owned by this file.
      name = name.copy(_allocator);
    }
    SimpleUndefinedAtom *atom =
        new (_allocator) SimpleUndefinedAtom(*this, name);
    addAtom(*atom);
  }

private:
  llvm::BumpPtrAllocator _allocator;
};

} // end namespace mach_o
} // end namespace lld

#endif
