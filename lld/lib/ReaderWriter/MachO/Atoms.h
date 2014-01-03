//===- lib/ReaderWriter/MachO/Atoms.h -------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_MACHO_ATOMS_H
#define LLD_READER_WRITER_MACHO_ATOMS_H

#include "lld/ReaderWriter/Simple.h"

namespace lld {
namespace mach_o {
class MachODefinedAtom : public SimpleDefinedAtom {
public:
  // FIXME: This constructor should also take the ContentType.
  MachODefinedAtom(const File &f, const StringRef name,
                   const ArrayRef<uint8_t> content)
      : SimpleDefinedAtom(f), _name(name), _content(content) {}

  virtual uint64_t size() const { return rawContent().size(); }

  virtual ContentType contentType() const { return DefinedAtom::typeCode; }

  virtual StringRef name() const { return _name; }

  virtual Scope scope() const { return scopeGlobal; }

  virtual ArrayRef<uint8_t> rawContent() const { return _content; }

private:
  const StringRef _name;
  const ArrayRef<uint8_t> _content;
};
} // mach_o
} // lld

#endif
