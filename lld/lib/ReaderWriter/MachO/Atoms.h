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

#include "lld/Core/Simple.h"

namespace lld {
namespace mach_o {
class MachODefinedAtom : public SimpleDefinedAtom {
public:
  MachODefinedAtom(const File &f, const StringRef name, Scope scope,
                   ContentType type, Merge merge, bool thumb,
                   const ArrayRef<uint8_t> content)
      : SimpleDefinedAtom(f), _name(name), _content(content),
        _contentType(type), _scope(scope), _merge(merge), _thumb(thumb) {}

  // Constructor for zero-fill content
  MachODefinedAtom(const File &f, const StringRef name, Scope scope,
                   uint64_t size)
      : SimpleDefinedAtom(f), _name(name),
        _content(ArrayRef<uint8_t>(nullptr, size)),
        _contentType(DefinedAtom::typeZeroFill),
        _scope(scope), _merge(mergeNo), _thumb(false) {}

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
    // Note: Zerofill atoms have a content pointer which is null.
    return _content;
  }

  bool isThumb() const { return _thumb; }

  void addReference(uint32_t offsetInAtom, uint16_t relocType, 
               const Atom *target, Reference::Addend addend, 
               Reference::KindArch arch = Reference::KindArch::x86_64,
               Reference::KindNamespace ns = Reference::KindNamespace::mach_o) {
    SimpleDefinedAtom::addReference(ns, arch, relocType, offsetInAtom, target, addend);
  }
  
private:
  const StringRef _name;
  const ArrayRef<uint8_t> _content;
  const ContentType _contentType;
  const Scope _scope;
  const Merge _merge;
  const bool _thumb;
};

class MachODefinedCustomSectionAtom : public MachODefinedAtom {
public:
  MachODefinedCustomSectionAtom(const File &f, const StringRef name, 
                                Scope scope, ContentType type, Merge merge,
                                bool thumb, const ArrayRef<uint8_t> content,
                                StringRef sectionName)
      : MachODefinedAtom(f, name, scope, type, merge, thumb, content), 
        _sectionName(sectionName) {}

  SectionChoice sectionChoice() const override {
    return DefinedAtom::sectionCustomRequired;
  }
  
  StringRef customSectionName() const override {
    return _sectionName;
  }
private:  
  StringRef _sectionName;
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

class MachOSharedLibraryAtom : public SharedLibraryAtom {
public:
  MachOSharedLibraryAtom(const File &file, StringRef name,
                         StringRef dylibInstallName, bool weakDef)
      : SharedLibraryAtom(), _file(file), _name(name),
        _dylibInstallName(dylibInstallName) {}
  virtual ~MachOSharedLibraryAtom() {}

  virtual StringRef loadName() const override {
    return _dylibInstallName;
  }

  virtual bool canBeNullAtRuntime() const override {
    // FIXME: this may actually be changeable. For now, all symbols are strongly
    // defined though.
    return false;
  }

  virtual const File& file() const override {
    return _file;
  }

  virtual StringRef name() const override {
    return _name;
  }

  virtual Type type() const override {
    // Unused in MachO (I think).
    return Type::Unknown;
  }

  virtual uint64_t size() const override {
    // Unused in MachO (I think)
    return 0;
  }

private:
  const File &_file;
  StringRef _name;
  StringRef _dylibInstallName;
};


} // mach_o
} // lld

#endif
