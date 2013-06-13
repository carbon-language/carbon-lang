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
  SimpleFile(const TargetInfo &ti, StringRef path) : MutableFile(ti, path) {
    static uint32_t lastOrdinal = 0;
    _ordinal = lastOrdinal++;
  }

  virtual void addAtom(const Atom &atom) {
    if (const DefinedAtom *defAtom = dyn_cast<DefinedAtom>(&atom)) {
      _definedAtoms._atoms.push_back(defAtom);
    } else if (
        const UndefinedAtom *undefAtom = dyn_cast<UndefinedAtom>(&atom)) {
      _undefinedAtoms._atoms.push_back(undefAtom);
    } else if (
        const SharedLibraryAtom *slAtom = dyn_cast<SharedLibraryAtom>(&atom)) {
      _sharedLibraryAtoms._atoms.push_back(slAtom);
    } else if (const AbsoluteAtom *abAtom = dyn_cast<AbsoluteAtom>(&atom)) {
      _absoluteAtoms._atoms.push_back(abAtom);
    } else {
      llvm_unreachable("atom has unknown definition kind");
    }
  }

  virtual const atom_collection<DefinedAtom> &defined() const {
    return _definedAtoms;
  }

  virtual const atom_collection<UndefinedAtom> &undefined() const {
    return _undefinedAtoms;
  }

  virtual const atom_collection<SharedLibraryAtom> &sharedLibrary() const {
    return _sharedLibraryAtoms;
  }

  virtual const atom_collection<AbsoluteAtom> &absolute() const {
    return _absoluteAtoms;
  }

  virtual DefinedAtomRange definedAtoms() {
    return make_range(_definedAtoms._atoms);
  }

private:
  atom_collection_vector<DefinedAtom> _definedAtoms;
  atom_collection_vector<UndefinedAtom> _undefinedAtoms;
  atom_collection_vector<SharedLibraryAtom> _sharedLibraryAtoms;
  atom_collection_vector<AbsoluteAtom> _absoluteAtoms;
};

class SimpleReference : public Reference {
public:
  SimpleReference(Reference::Kind k, uint64_t off, const Atom *t,
                  Reference::Addend a)
      : _target(t), _offsetInAtom(off), _addend(a) {
    _kind = k;
  }

  virtual uint64_t offsetInAtom() const { return _offsetInAtom; }

  virtual const Atom *target() const { return _target; }

  virtual Addend addend() const { return _addend; }

  virtual void setAddend(Addend a) { _addend = a; }

  virtual void setTarget(const Atom *newAtom) { _target = newAtom; }
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

  virtual const File &file() const { return _file; }

  virtual StringRef name() const { return StringRef(); }

  virtual uint64_t ordinal() const { return _ordinal; }

  virtual Scope scope() const { return DefinedAtom::scopeLinkageUnit; }

  virtual Interposable interposable() const { return DefinedAtom::interposeNo; }

  virtual Merge merge() const { return DefinedAtom::mergeNo; }

  virtual Alignment alignment() const { return Alignment(0, 0); }

  virtual SectionChoice sectionChoice() const {
    return DefinedAtom::sectionBasedOnContent;
  }

  virtual SectionPosition sectionPosition() const {
    return DefinedAtom::sectionPositionAny;
  }

  virtual StringRef customSectionName() const { return StringRef(); }
  virtual DeadStripKind deadStrip() const {
    return DefinedAtom::deadStripNormal;
  }

  virtual bool isAlias() const { return false; }

  virtual DefinedAtom::reference_iterator begin() const {
    uintptr_t index = 0;
    const void *it = reinterpret_cast<const void *>(index);
    return reference_iterator(*this, it);
  }

  virtual DefinedAtom::reference_iterator end() const {
    uintptr_t index = _references.size();
    const void *it = reinterpret_cast<const void *>(index);
    return reference_iterator(*this, it);
  }

  virtual const Reference *derefIterator(const void *it) const {
    uintptr_t index = reinterpret_cast<uintptr_t>(it);
    assert(index < _references.size());
    return &_references[index];
  }

  virtual void incrementIterator(const void *&it) const {
    uintptr_t index = reinterpret_cast<uintptr_t>(it);
    ++index;
    it = reinterpret_cast<const void *>(index);
  }

  void addReference(Reference::Kind kind, uint64_t offset, const Atom *target,
                    Reference::Addend addend) {
    _references.push_back(SimpleReference(kind, offset, target, addend));
  }

  void setOrdinal(uint64_t ord) { _ordinal = ord; }

private:
  const File &_file;
  uint64_t _ordinal;
  std::vector<SimpleReference> _references;
};

class SimpleUndefinedAtom : public UndefinedAtom {
public:
  SimpleUndefinedAtom(const File &f, StringRef name) 
    : _file(f)
    , _name(name) {
    assert(!name.empty() && "UndefinedAtoms must have a name");
  }

  /// file - returns the File that produced/owns this Atom
  virtual const class File &file() const { return _file; }

  /// name - The name of the atom. For a function atom, it is the (mangled)
  /// name of the function.
  virtual StringRef name() const { return _name; }

  virtual CanBeNull canBeNull() const { return UndefinedAtom::canBeNullNever; }

private:
  const File &_file;
  StringRef _name;
};
} // end namespace lld

#endif
