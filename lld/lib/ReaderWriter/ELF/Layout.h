//===- lib/ReaderWriter/ELF/Layout.h --------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_LAYOUT_H
#define LLD_READER_WRITER_ELF_LAYOUT_H

#include "lld/Core/DefinedAtom.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Object/ELF.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ELF.h"

namespace lld {
namespace elf {
/// \brief The Layout is an abstract class for managing the final layout for
///        the kind of binaries(Shared Libraries / Relocatables / Executables 0
///        Each architecture (Hexagon, PowerPC, MIPS) would have a concrete
///        subclass derived from Layout for generating each binary thats
//         needed by the lld linker
class Layout {
public:
  typedef uint32_t SectionOrder;
  typedef uint32_t SegmentType;
  typedef uint32_t Flags;

public:
  /// Return the order the section would appear in the output file
  virtual SectionOrder getSectionOrder
                        (const StringRef name,
                         int32_t contentType,
                         int32_t contentPerm) = 0;
  /// append the Atom to the layout and create appropriate sections
  virtual error_code addAtom(const Atom *atom) = 0;
  /// find the Atom Address in the current layout
  virtual bool findAtomAddrByName(const StringRef name, 
                                  uint64_t &addr) = 0;
  /// associates a section to a segment
  virtual void assignSectionsToSegments() = 0;
  /// associates a virtual address to the segment, section, and the atom
  virtual void assignVirtualAddress() = 0;
  /// associates a file offset to the segment, section and the atom
  virtual void assignFileOffsets() = 0;

public:
  Layout() {}

  virtual ~Layout() { }
};

struct AtomLayout {
  AtomLayout(const Atom *a, uint64_t fileOff, uint64_t virAddr)
    : _atom(a), _fileOffset(fileOff), _virtualAddr(virAddr) {}

  AtomLayout()
    : _atom(nullptr), _fileOffset(0), _virtualAddr(0) {}

  const Atom *_atom;
  uint64_t _fileOffset;
  uint64_t _virtualAddr;
};
} // end namespace elf
} // end namespace lld

#endif
