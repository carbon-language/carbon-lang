//===- lib/ReaderWriter/PECOFF/Atoms.h ------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_PE_COFF_ATOMS_H
#define LLD_READER_WRITER_PE_COFF_ATOMS_H

#include "lld/Core/File.h"
#include "lld/ReaderWriter/Simple.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Object/COFF.h"

#include <vector>

namespace lld {
namespace pecoff {
class COFFDefinedAtom;

/// A COFFReference represents relocation information for an atom. For
/// example, if atom X has a reference to atom Y with offsetInAtom=8, that
/// means that the address starting at 8th byte of the content of atom X needs
/// to be fixed up so that the address points to atom Y's address.
class COFFReference final : public Reference {
public:
  COFFReference(const Atom *target, uint32_t offsetInAtom, uint16_t relocType,
                Reference::KindNamespace ns = Reference::KindNamespace::COFF,
                Reference::KindArch arch = Reference::KindArch::x86)
      : Reference(ns, arch, relocType), _target(target),
        _offsetInAtom(offsetInAtom) {}

  const Atom *target() const override { return _target; }
  void setTarget(const Atom *newAtom) override { _target = newAtom; }

  // Addend is a value to be added to the relocation target. For example, if
  // target=AtomX and addend=4, the relocation address will become the address
  // of AtomX + 4. COFF does not support that sort of relocation, thus addend
  // is always zero.
  Addend addend() const override { return 0; }
  void setAddend(Addend) override {}
  uint64_t offsetInAtom() const override { return _offsetInAtom; }

private:
  const Atom *_target;
  uint32_t _offsetInAtom;
};

class COFFAbsoluteAtom : public AbsoluteAtom {
public:
  COFFAbsoluteAtom(const File &f, StringRef name, Scope scope, uint64_t value)
      : _owningFile(f), _name(name), _scope(scope), _value(value) {}

  const File &file() const override { return _owningFile; }
  Scope scope() const override { return _scope; }
  StringRef name() const override { return _name; }
  uint64_t value() const override { return _value; }

private:
  const File &_owningFile;
  StringRef _name;
  Scope _scope;
  uint64_t _value;
};

class COFFUndefinedAtom : public UndefinedAtom {
public:
  COFFUndefinedAtom(const File &file, StringRef name,
                    const UndefinedAtom *fallback = nullptr)
      : _owningFile(file), _name(name), _fallback(fallback) {}

  const File &file() const override { return _owningFile; }
  StringRef name() const override { return _name; }
  CanBeNull canBeNull() const override { return CanBeNull::canBeNullNever; }
  const UndefinedAtom *fallback() const override { return _fallback; }

private:
  const File &_owningFile;
  StringRef _name;
  const UndefinedAtom *_fallback;
};

/// The base class of all COFF defined atoms. A derived class of
/// COFFBaseDefinedAtom may represent atoms read from a file or atoms created
/// by the linker. An example of the latter case is the jump table for symbols
/// in a DLL.
class COFFBaseDefinedAtom : public DefinedAtom {
public:
  enum class Kind {
    File,
    Internal
  };

  const File &file() const override { return _file; }
  StringRef name() const override { return _name; }
  Interposable interposable() const override { return interposeNo; }
  Merge merge() const override { return mergeNo; }
  Alignment alignment() const override { return Alignment(0); }
  SectionChoice sectionChoice() const = 0;
  StringRef customSectionName() const override { return ""; }
  SectionPosition sectionPosition() const override {
    return sectionPositionAny;
  }
  DeadStripKind deadStrip() const override { return deadStripNormal; }
  bool isAlias() const override { return false; }

  Kind getKind() const { return _kind; }

  void addReference(std::unique_ptr<COFFReference> reference) {
    _references.push_back(std::move(reference));
  }

  reference_iterator begin() const override {
    return reference_iterator(*this, reinterpret_cast<const void *>(0));
  }

  reference_iterator end() const override {
    return reference_iterator(
        *this, reinterpret_cast<const void *>(_references.size()));
  }

protected:
  COFFBaseDefinedAtom(const File &file, StringRef name, Kind kind)
      : _file(file), _name(name), _kind(kind) {}

private:
  const Reference *derefIterator(const void *iter) const override {
    size_t index = reinterpret_cast<size_t>(iter);
    return _references[index].get();
  }

  void incrementIterator(const void *&iter) const override {
    size_t index = reinterpret_cast<size_t>(iter);
    iter = reinterpret_cast<const void *>(index + 1);
  }

  const File &_file;
  StringRef _name;
  Kind _kind;
  std::vector<std::unique_ptr<COFFReference> > _references;
};

/// This is the root class of the atom read from a file. This class have two
/// subclasses; one for the regular atom and another for the BSS atom.
class COFFDefinedFileAtom : public COFFBaseDefinedAtom {
public:
  COFFDefinedFileAtom(const File &file, StringRef name, StringRef sectionName,
                      Scope scope, ContentType contentType,
                      ContentPermissions perms, uint64_t ordinal)
      : COFFBaseDefinedAtom(file, name, Kind::File), _sectionName(sectionName),
        _scope(scope), _contentType(contentType), _permissions(perms),
        _ordinal(ordinal), _alignment(0) {}

  static bool classof(const COFFBaseDefinedAtom *atom) {
    return atom->getKind() == Kind::File;
  }

  void setAlignment(Alignment val) { _alignment = val; }

  SectionChoice sectionChoice() const override { return sectionCustomRequired; }
  StringRef customSectionName() const override { return _sectionName; }
  Scope scope() const override { return _scope; }
  ContentType contentType() const override { return _contentType; }
  ContentPermissions permissions() const override { return _permissions; }
  uint64_t ordinal() const override { return _ordinal; }
  Alignment alignment() const override { return _alignment; }

private:
  StringRef _sectionName;
  Scope _scope;
  ContentType _contentType;
  ContentPermissions _permissions;
  uint64_t _ordinal;
  Alignment _alignment;
  std::vector<std::unique_ptr<COFFReference> > _references;
};

// A COFFDefinedAtom represents an atom read from a file and has contents.
class COFFDefinedAtom : public COFFDefinedFileAtom {
public:
  COFFDefinedAtom(const File &file, StringRef name, StringRef sectionName,
                  Scope scope, ContentType type, bool isComdat,
                  ContentPermissions perms, Merge merge, ArrayRef<uint8_t> data,
                  uint64_t ordinal)
      : COFFDefinedFileAtom(file, name, sectionName, scope, type, perms,
                            ordinal),
        _isComdat(isComdat), _merge(merge), _dataref(data) {}

  Merge merge() const override { return _merge; }
  uint64_t size() const override { return _dataref.size(); }
  ArrayRef<uint8_t> rawContent() const override { return _dataref; }

  DeadStripKind deadStrip() const override {
    // Only COMDAT symbols would be dead-stripped.
    return _isComdat ? deadStripNormal : deadStripNever;
  }

private:
  bool _isComdat;
  Merge _merge;
  ArrayRef<uint8_t> _dataref;
};

// A COFFDefinedAtom represents an atom for BSS section.
class COFFBSSAtom : public COFFDefinedFileAtom {
public:
  COFFBSSAtom(const File &file, StringRef name, Scope scope,
              ContentPermissions perms, Merge merge, uint32_t size,
              uint64_t ordinal)
      : COFFDefinedFileAtom(file, name, ".bss", scope, typeZeroFill, perms,
                            ordinal),
        _merge(merge), _size(size) {}

  Merge merge() const override { return _merge; }
  uint64_t size() const override { return _size; }
  ArrayRef<uint8_t> rawContent() const override { return _contents; }

private:
  Merge _merge;
  uint32_t _size;
  std::vector<uint8_t> _contents;
};

/// A COFFLinkerInternalAtom represents a defined atom created by the linker,
/// not read from file.
class COFFLinkerInternalAtom : public COFFBaseDefinedAtom {
public:
  SectionChoice sectionChoice() const override { return sectionBasedOnContent; }
  uint64_t ordinal() const override { return _ordinal; }
  Scope scope() const override { return scopeGlobal; }
  Alignment alignment() const override { return Alignment(0); }
  uint64_t size() const override { return _data.size(); }
  ArrayRef<uint8_t> rawContent() const override { return _data; }

protected:
  COFFLinkerInternalAtom(const File &file, uint64_t ordinal,
                         std::vector<uint8_t> data, StringRef symbolName = "")
      : COFFBaseDefinedAtom(file, symbolName, Kind::Internal),
        _ordinal(ordinal), _data(std::move(data)) {}

private:
  uint64_t _ordinal;
  std::vector<uint8_t> _data;
};

class COFFStringAtom : public COFFLinkerInternalAtom {
public:
  COFFStringAtom(const File &file, uint64_t ordinal, StringRef sectionName,
                 StringRef contents)
      : COFFLinkerInternalAtom(file, ordinal, stringRefToVector(contents)),
        _sectionName(sectionName) {}

  SectionChoice sectionChoice() const override { return sectionCustomRequired; }
  StringRef customSectionName() const override { return _sectionName; }
  ContentType contentType() const override { return typeData; }
  ContentPermissions permissions() const override { return permR__; }

private:
  StringRef _sectionName;

  std::vector<uint8_t> stringRefToVector(StringRef name) const {
    std::vector<uint8_t> ret(name.size() + 1);
    memcpy(&ret[0], name.data(), name.size());
    ret[name.size()] = 0;
    return ret;
  }
};

// A COFFSharedLibraryAtom represents a symbol for data in an import library.  A
// reference to a COFFSharedLibraryAtom will be transformed to a real reference
// to an import address table entry in Idata pass.
class COFFSharedLibraryAtom : public SharedLibraryAtom {
public:
  COFFSharedLibraryAtom(const File &file, uint16_t hint, StringRef symbolName,
                        StringRef importName, StringRef dllName)
      : _file(file), _hint(hint), _mangledName(addImpPrefix(symbolName)),
        _importName(importName), _dllName(dllName), _importTableEntry(nullptr) {
  }

  const File &file() const override { return _file; }
  uint16_t hint() const { return _hint; }

  /// Returns the symbol name to be used by the core linker.
  StringRef name() const override { return _mangledName; }

  /// Returns the symbol name to be used in the import description table in the
  /// COFF header.
  virtual StringRef importName() const { return _importName; }

  StringRef loadName() const override { return _dllName; }
  bool canBeNullAtRuntime() const override { return false; }
  Type type() const override { return Type::Unknown; }
  uint64_t size() const override { return 0; }

  void setImportTableEntry(const DefinedAtom *atom) {
    _importTableEntry = atom;
  }

  const DefinedAtom *getImportTableEntry() const { return _importTableEntry; }

private:
  /// Mangle the symbol name by adding "__imp_" prefix. See the file comment of
  /// ReaderImportHeader.cpp for details about the prefix.
  std::string addImpPrefix(StringRef symbolName) {
    std::string ret("__imp_");
    ret.append(symbolName);
    return ret;
  }

  const File &_file;
  uint16_t _hint;
  std::string _mangledName;
  std::string _importName;
  StringRef _dllName;
  const DefinedAtom *_importTableEntry;
};

// An instance of this class represents "input file" for atoms created in a
// pass. Atoms need to be associated to an input file even if it's not read from
// a file, so we use this class for that.
class VirtualFile : public SimpleFile {
public:
  VirtualFile(const LinkingContext &ctx)
      : SimpleFile("<virtual-file>"), _nextOrdinal(0) {
    setOrdinal(ctx.getNextOrdinalAndIncrement());
  }

  uint64_t getNextOrdinal() { return _nextOrdinal++; }

private:
  uint64_t _nextOrdinal;
};

//===----------------------------------------------------------------------===//
//
// Utility functions to handle layout edges.
//
//===----------------------------------------------------------------------===//

template <typename T, typename U>
void addLayoutEdge(T *a, U *b, uint32_t which) {
  auto ref = new COFFReference(nullptr, 0, which, Reference::KindNamespace::all,
                               Reference::KindArch::all);
  ref->setTarget(b);
  a->addReference(std::unique_ptr<COFFReference>(ref));
}

template <typename T, typename U> void connectWithLayoutEdge(T *a, U *b) {
  addLayoutEdge(a, b, lld::Reference::kindLayoutAfter);
  addLayoutEdge(b, a, lld::Reference::kindLayoutBefore);
}

/// Connect atoms with layout-{before,after} edges. It usually serves two
/// purposes.
///
///   - To prevent atoms from being GC'ed (aka dead-stripped) if there is a
///     reference to one of the atoms. In that case we want to emit all the
///     atoms appeared in the same section, because the referenced "live" atom
///     may reference other atoms in the same section. If we don't add layout
///     edges between atoms, unreferenced atoms in the same section would be
///     GC'ed.
///   - To preserve the order of atmos. We want to emit the atoms in the
///     same order as they appeared in the input object file.
template <typename T> void connectAtomsWithLayoutEdge(std::vector<T *> &atoms) {
  if (atoms.size() < 2)
    return;
  for (auto it = atoms.begin(), e = atoms.end(); it + 1 != e; ++it)
    connectWithLayoutEdge(*it, *(it + 1));
}

} // namespace pecoff
} // namespace lld

#endif
