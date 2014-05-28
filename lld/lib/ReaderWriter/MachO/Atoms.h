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
  MachODefinedAtom(const File &f, const StringRef name, Scope scope,
                   ContentType type, Merge merge,
                   const ArrayRef<uint8_t> content)
      : SimpleDefinedAtom(f), _name(name), _content(content),
        _contentType(type), _scope(scope), _merge(merge) {}

  // Constructor for zero-fill content
  MachODefinedAtom(const File &f, const StringRef name, Scope scope,
                   uint64_t size)
      : SimpleDefinedAtom(f), _name(name),
        _content(ArrayRef<uint8_t>(nullptr, size)),
        _contentType(DefinedAtom::typeZeroFill),
        _scope(scope), _merge(mergeNo) {}

  uint64_t size() const override { return _content.size(); }

  ContentType contentType() const override { return _contentType; }

  StringRef name() const override { return _name; }

  Scope scope() const override { return _scope; }

  Merge merge() const override { return _merge; }

  DeadStripKind deadStrip() const override {
    if (_contentType == DefinedAtom::typeInitializerPtr)
      return deadStripNever;
    if (_contentType == DefinedAtom::typeTerminatorPtr)
      return deadStripNever;
    return deadStripNormal;
  }

  ArrayRef<uint8_t> rawContent() const override {
    // Zerofill atoms have a content pointer which is null.
    assert(_content.data() != nullptr);
    return _content;
  }

private:
  const StringRef _name;
  const ArrayRef<uint8_t> _content;
  const ContentType _contentType;
  const Scope _scope;
  const Merge _merge;
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
