//===- lld/Core/Simple.h - Simple implementations of Atom and File --------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Provide simple implementations for Atoms and File.
///
//===----------------------------------------------------------------------===//

#ifndef LLD_CORE_SIMPLE_H
#define LLD_CORE_SIMPLE_H

#include "lld/Core/DefinedAtom.h"
#include "lld/Core/File.h"
#include "lld/Core/Reference.h"
#include "lld/Core/UndefinedAtom.h"

namespace lld {

class SimpleFile : public MutableFile {
public:
  SimpleFile(StringRef path) : MutableFile(path) {}

  void addAtom(const Atom &atom) override {
    if (auto *defAtom = dyn_cast<DefinedAtom>(&atom)) {
      _definedAtoms._atoms.push_back(defAtom);
    } else if (auto *undefAtom = dyn_cast<UndefinedAtom>(&atom)) {
      _undefinedAtoms._atoms.push_back(undefAtom);
    } else if (auto *shlibAtom = dyn_cast<SharedLibraryAtom>(&atom)) {
      _sharedLibraryAtoms._atoms.push_back(shlibAtom);
    } else if (auto *absAtom = dyn_cast<AbsoluteAtom>(&atom)) {
      _absoluteAtoms._atoms.push_back(absAtom);
    } else {
      llvm_unreachable("atom has unknown definition kind");
    }
  }

  const atom_collection<DefinedAtom> &defined() const override {
    return _definedAtoms;
  }

  const atom_collection<UndefinedAtom> &undefined() const override {
    return _undefinedAtoms;
  }

  const atom_collection<SharedLibraryAtom> &sharedLibrary() const override {
    return _sharedLibraryAtoms;
  }

  const atom_collection<AbsoluteAtom> &absolute() const override {
    return _absoluteAtoms;
  }

  DefinedAtomRange definedAtoms() override {
    return make_range(_definedAtoms._atoms);
  }

protected:
  atom_collection_vector<DefinedAtom>        _definedAtoms;
  atom_collection_vector<UndefinedAtom>      _undefinedAtoms;
  atom_collection_vector<SharedLibraryAtom>  _sharedLibraryAtoms;
  atom_collection_vector<AbsoluteAtom>       _absoluteAtoms;
};

class SimpleFileWrapper : public SimpleFile {
public:
  SimpleFileWrapper(const LinkingContext &context, const File &file)
      : SimpleFile(file.path()) {
    for (auto definedAtom : file.defined())
      _definedAtoms._atoms.push_back(std::move(definedAtom));
    for (auto undefAtom : file.undefined())
      _undefinedAtoms._atoms.push_back(std::move(undefAtom));
    for (auto shlibAtom : file.sharedLibrary())
      _sharedLibraryAtoms._atoms.push_back(std::move(shlibAtom));
    for (auto absAtom : file.absolute())
      _absoluteAtoms._atoms.push_back(std::move(absAtom));
  }
};

class SimpleReference : public Reference {
public:
  SimpleReference(Reference::KindNamespace ns, Reference::KindArch arch,
                  Reference::KindValue value, uint64_t off, const Atom *t,
                  Reference::Addend a)
      : Reference(ns, arch, value), _target(t), _offsetInAtom(off), _addend(a) {
  }

  uint64_t offsetInAtom() const override { return _offsetInAtom; }

  const Atom *target() const override {
    assert(_target);
    return _target;
  }

  Addend addend() const override { return _addend; }
  void setAddend(Addend a) override { _addend = a; }
  void setTarget(const Atom *newAtom) override { _target = newAtom; }

private:
  const Atom *_target;
  uint64_t _offsetInAtom;
  Addend _addend;
};

class SimpleDefinedAtom : public DefinedAtom {
public:
  explicit SimpleDefinedAtom(const File &f) : _file(f) {
    static uint32_t lastOrdinal = 0;
    _ordinal = lastOrdinal++;
  }

  const File &file() const override { return _file; }

  StringRef name() const override { return StringRef(); }

  uint64_t ordinal() const override { return _ordinal; }

  Scope scope() const override { return DefinedAtom::scopeLinkageUnit; }

  Interposable interposable() const override { return DefinedAtom::interposeNo; }

  Merge merge() const override { return DefinedAtom::mergeNo; }

  Alignment alignment() const override { return Alignment(0, 0); }

  SectionChoice sectionChoice() const override {
    return DefinedAtom::sectionBasedOnContent;
  }

  SectionPosition sectionPosition() const override {
    return DefinedAtom::sectionPositionAny;
  }

  StringRef customSectionName() const override { return StringRef(); }
  DeadStripKind deadStrip() const override {
    return DefinedAtom::deadStripNormal;
  }

  DefinedAtom::reference_iterator begin() const override {
    uintptr_t index = 0;
    const void *it = reinterpret_cast<const void *>(index);
    return reference_iterator(*this, it);
  }

  DefinedAtom::reference_iterator end() const override {
    uintptr_t index = _references.size();
    const void *it = reinterpret_cast<const void *>(index);
    return reference_iterator(*this, it);
  }

  const Reference *derefIterator(const void *it) const override {
    uintptr_t index = reinterpret_cast<uintptr_t>(it);
    assert(index < _references.size());
    return &_references[index];
  }

  void incrementIterator(const void *&it) const override {
    uintptr_t index = reinterpret_cast<uintptr_t>(it);
    ++index;
    it = reinterpret_cast<const void *>(index);
  }

  void addReference(Reference::KindNamespace ns, Reference::KindArch arch,
                    Reference::KindValue kindValue, uint64_t off,
                    const Atom *target, Reference::Addend a) {
    assert(target && "trying to create reference to nothing");
    _references.push_back(SimpleReference(ns, arch, kindValue, off, target, a));
  }

  /// Sort references in a canonical order (by offset, then by kind).
  void sortReferences() const {
    std::sort(_references.begin(), _references.end(),
        [] (const SimpleReference &lhs, const SimpleReference &rhs) -> bool {
          uint64_t lhsOffset = lhs.offsetInAtom();
          uint64_t rhsOffset = rhs.offsetInAtom();
          if (rhsOffset != lhsOffset)
            return (lhsOffset < rhsOffset);
          if (rhs.kindNamespace() != lhs.kindNamespace())
            return (lhs.kindNamespace() < rhs.kindNamespace());
          if (rhs.kindArch() != lhs.kindArch())
            return (lhs.kindArch() < rhs.kindArch());
          return (lhs.kindValue() < rhs.kindValue());
        });
  }
  void setOrdinal(uint64_t ord) { _ordinal = ord; }

private:
  const File                   &_file;
  uint64_t                      _ordinal;
  mutable std::vector<SimpleReference>  _references;
};

class SimpleUndefinedAtom : public UndefinedAtom {
public:
  SimpleUndefinedAtom(const File &f, StringRef name) : _file(f), _name(name) {
    assert(!name.empty() && "UndefinedAtoms must have a name");
  }

  /// file - returns the File that produced/owns this Atom
  const File &file() const override { return _file; }

  /// name - The name of the atom. For a function atom, it is the (mangled)
  /// name of the function.
  StringRef name() const override { return _name; }

  CanBeNull canBeNull() const override { return UndefinedAtom::canBeNullNever; }

private:
  const File &_file;
  StringRef _name;
};

} // end namespace lld

#endif
