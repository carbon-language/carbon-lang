//===- Core/File.h - A Container of Atoms ---------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_CORE_FILE_H
#define LLD_CORE_FILE_H

#include "lld/Core/AbsoluteAtom.h"
#include "lld/Core/DefinedAtom.h"
#include "lld/Core/range.h"
#include "lld/Core/SharedLibraryAtom.h"
#include "lld/Core/LinkingContext.h"
#include "lld/Core/UndefinedAtom.h"

#include "llvm/Support/ErrorHandling.h"

#include <vector>

namespace lld {

class LinkingContext;

/// Every Atom is owned by some File. A common scenario is for a single
/// object file (.o) to be parsed by some reader and produce a single
/// File object that represents the content of that object file.
///
/// To iterate through the Atoms in a File there are four methods that
/// return collections.  For instance to iterate through all the DefinedAtoms
/// in a File object use:
///      for (const DefinedAtoms *atom : file->defined()) {
///      }
///
/// The Atom objects in a File are owned by the File object.  The Atom objects
/// are destroyed when the File object is destroyed.
class File {
public:
  virtual ~File() {}

  /// \brief Kinds of files that are supported.
  enum Kind {
    kindObject,        ///< object file (.o)
    kindSharedLibrary, ///< shared library (.so)
    kindArchiveLibrary ///< archive (.a)
  };

  /// \brief Returns file kind.  Need for dyn_cast<> on File objects.
  Kind kind() const {
    return _kind;
  }

  /// \brief For error messages and debugging, this returns the path to the file
  /// which was used to create this object (e.g. "/tmp/foo.o").
  StringRef path() const  {
    return _path;
  }

  /// Returns the command line order of the file.
  uint64_t ordinal() const {
    assert(_ordinal != UINT64_MAX);
    return _ordinal;
  }

  /// Returns true/false depending on whether an ordinal has been set.
  bool hasOrdinal() const { return (_ordinal != UINT64_MAX); }

  /// Sets the command line order of the file.
  void setOrdinal(uint64_t ordinal) const { _ordinal = ordinal; }

  template <typename T> class atom_iterator; // forward reference

  /// \brief For use interating over DefinedAtoms in this File.
  typedef atom_iterator<DefinedAtom>  defined_iterator;

  /// \brief For use interating over UndefinedAtoms in this File.
  typedef atom_iterator<UndefinedAtom> undefined_iterator;

  /// \brief For use interating over SharedLibraryAtoms in this File.
  typedef atom_iterator<SharedLibraryAtom> shared_library_iterator;

  /// \brief For use interating over AbsoluteAtoms in this File.
  typedef atom_iterator<AbsoluteAtom> absolute_iterator;

  /// \brief Different object file readers may instantiate and manage atoms with
  /// different data structures.  This class is a collection abstraction.
  /// Each concrete File instance must implement these atom_collection
  /// methods to enable clients to interate the File's atoms.
  template <typename T>
  class atom_collection {
  public:
    virtual ~atom_collection() { }
    virtual atom_iterator<T> begin() const = 0;
    virtual atom_iterator<T> end() const = 0;
    virtual const T *deref(const void *it) const = 0;
    virtual void next(const void *&it) const = 0;
    virtual uint64_t size() const = 0;
    bool empty() const { return size() == 0; }
  };

  /// \brief The class is the iterator type used to iterate through a File's
  /// Atoms. This iterator delegates the work to the associated atom_collection
  /// object. There are four kinds of Atoms, so this iterator is templated on
  /// the four base Atom kinds.
  template <typename T>
  class atom_iterator {
  public:
    atom_iterator(const atom_collection<T> &c, const void *it)
              : _collection(c), _it(it) { }

    const T *operator*() const {
      return _collection.deref(_it);
    }
    const T *operator->() const {

      return _collection.deref(_it);
    }

    bool operator!=(const atom_iterator<T> &other) const {
      return (this->_it != other._it);
    }

    atom_iterator<T> &operator++() {
      _collection.next(_it);
      return *this;
    }
  private:
    const atom_collection<T> &_collection;
    const void               *_it;
  };


  /// \brief Must be implemented to return the atom_collection object for
  /// all DefinedAtoms in this File.
  virtual const atom_collection<DefinedAtom> &defined() const = 0;

  /// \brief Must be implemented to return the atom_collection object for
  /// all UndefinedAtomw in this File.
  virtual const atom_collection<UndefinedAtom> &undefined() const = 0;

  /// \brief Must be implemented to return the atom_collection object for
  /// all SharedLibraryAtoms in this File.
  virtual const atom_collection<SharedLibraryAtom> &sharedLibrary() const = 0;

  /// \brief Must be implemented to return the atom_collection object for
  /// all AbsoluteAtoms in this File.
  virtual const atom_collection<AbsoluteAtom> &absolute() const = 0;

protected:
  /// \brief only subclasses of File can be instantiated
  File(StringRef p, Kind kind) : _path(p), _kind(kind), _ordinal(UINT64_MAX) {}

  /// \brief This is a convenience class for File subclasses which manage their
  /// atoms as a simple std::vector<>.
  template <typename T>
  class atom_collection_vector : public atom_collection<T> {
  public:
    atom_iterator<T> begin() const override {
      auto *it = _atoms.empty() ? nullptr
                                : reinterpret_cast<const void *>(_atoms.data());
      return atom_iterator<T>(*this, it);
    }

    atom_iterator<T> end() const override {
      auto *it = _atoms.empty() ? nullptr : reinterpret_cast<const void *>(
                                                _atoms.data() + _atoms.size());
      return atom_iterator<T>(*this, it);
    }

    const T *deref(const void *it) const override {
      return *reinterpret_cast<const T *const *>(it);
    }

    void next(const void *&it) const override {
      const T *const *p = reinterpret_cast<const T *const *>(it);
      ++p;
      it = reinterpret_cast<const void*>(p);
    }

    uint64_t size() const override { return _atoms.size(); }

    std::vector<const T *> _atoms;
  };

  /// \brief This is a convenience class for File subclasses which need to
  /// return an empty collection.
  template <typename T>
  class atom_collection_empty : public atom_collection<T> {
  public:
    atom_iterator<T> begin() const override {
      return atom_iterator<T>(*this, nullptr);
    }
    atom_iterator<T> end() const override {
      return atom_iterator<T>(*this, nullptr);
    }
    const T *deref(const void *it) const override {
      llvm_unreachable("empty collection should never be accessed");
    }
    void next(const void *&it) const override {}
    uint64_t size() const override { return 0; }
  };

  static atom_collection_empty<DefinedAtom>       _noDefinedAtoms;
  static atom_collection_empty<UndefinedAtom>     _noUndefinedAtoms;
  static atom_collection_empty<SharedLibraryAtom> _noSharedLibraryAtoms;
  static atom_collection_empty<AbsoluteAtom>      _noAbsoluteAtoms;

private:
  StringRef _path;
  Kind              _kind;
  mutable uint64_t  _ordinal;
};

/// \brief A mutable File.
class MutableFile : public File {
public:
  /// \brief Add an atom to the file. Invalidates iterators for all returned
  /// containters.
  virtual void addAtom(const Atom&) = 0;

  typedef range<std::vector<const DefinedAtom *>::iterator> DefinedAtomRange;
  virtual DefinedAtomRange definedAtoms() = 0;

protected:
  /// \brief only subclasses of MutableFile can be instantiated
  MutableFile(StringRef p) : File(p, kindObject) {}
};
} // end namespace lld

#endif
