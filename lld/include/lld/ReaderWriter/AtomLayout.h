//===- include/lld/ReaderWriter/AtomLayout.h ------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_LAYOUT_H
#define LLD_READER_WRITER_LAYOUT_H

#include <cstdint>

namespace lld {
class Atom;

/// AtomLayouts are used by a writer to manage physical positions of atoms.
/// AtomLayout has two positions; one is file offset, and the other is the
/// address when loaded into memory.
///
/// Construction of AtomLayouts is usually a multi-pass process. When an atom
/// is appended to a section, we don't know the starting address of the
/// section. Thus, we have no choice but to store the offset from the
/// beginning of the section as AtomLayout values. After all sections starting
/// address are fixed, AtomLayout is revisited to get the offsets updated by
/// adding the starting addresses of the section.
struct AtomLayout {
  AtomLayout(const Atom *a, uint64_t fileOff, uint64_t virAddr)
      : _atom(a), _fileOffset(fileOff), _virtualAddr(virAddr) {}

  AtomLayout() : _atom(nullptr), _fileOffset(0), _virtualAddr(0) {}

  const Atom *_atom;
  uint64_t _fileOffset;
  uint64_t _virtualAddr;
};

}

#endif
