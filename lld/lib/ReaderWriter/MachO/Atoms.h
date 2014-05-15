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
                   const ArrayRef<uint8_t> content, Scope scope)
      : SimpleDefinedAtom(f), _name(name), _content(content), _scope(scope) {}

  uint64_t size() const override { return rawContent().size(); }

  ContentType contentType() const override { return DefinedAtom::typeCode; }

  StringRef name() const override { return _name; }

  Scope scope() const override { return _scope; }

  ArrayRef<uint8_t> rawContent() const override { return _content; }

private:
  const StringRef _name;
  const ArrayRef<uint8_t> _content;
  const Scope _scope;
};


class MachOTentativeDefAtom : public SimpleDefinedAtom {
public:
  MachOTentativeDefAtom(const File &f, const StringRef name, Scope scope,
                        uint64_t size, DefinedAtom::Alignment align)
      : SimpleDefinedAtom(f), _name(name), _scope(scope), _size(size),
        _align(align) {}

  uint64_t size() const override { return _size; }

  Merge merge() const override { return DefinedAtom::mergeAsTentative; }

  ContentType contentType() const override { return DefinedAtom::typeZeroFill; }

  Alignment alignment() const override { return _align; }

  StringRef name() const override { return _name; }

  Scope scope() const override { return _scope; }

  ArrayRef<uint8_t> rawContent() const override { return ArrayRef<uint8_t>(); }

private:
  const StringRef _name;
  const Scope _scope;
  const uint64_t _size;
  const DefinedAtom::Alignment _align;
};


} // mach_o
} // lld

#endif
