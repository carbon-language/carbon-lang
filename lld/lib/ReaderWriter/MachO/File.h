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
      char *s = _allocator.Allocate<char>(name.size());
      memcpy(s, name.data(), name.size());
      name = StringRef(s, name.size());
      uint8_t *bytes = _allocator.Allocate<uint8_t>(content.size());
      memcpy(bytes, content.data(), content.size());
      content = llvm::makeArrayRef(bytes, content.size());
    }
    MachODefinedAtom *atom =
        new (_allocator) MachODefinedAtom(*this, name, content, scope);
    addAtom(*atom);
  }

  void addUndefinedAtom(StringRef name, bool copyRefs) {
    if (copyRefs) {
      // Make a copy of the atom's name and content that is owned by this file.
      char *s = _allocator.Allocate<char>(name.size());
      memcpy(s, name.data(), name.size());
      name = StringRef(s, name.size());
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
