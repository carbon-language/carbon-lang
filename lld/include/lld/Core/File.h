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
#include "lld/Core/TargetInfo.h"
#include "lld/Core/UndefinedAtom.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"

#include <vector>

namespace lld {
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
  virtual ~File();

  /// \brief Kinds of files that are supported.
  enum Kind {
    kindObject,            ///< object file (.o)
    kindSharedLibrary,     ///< shared library (.so)
    kindArchiveLibrary,    ///< archive (.a)
    kindLinkerScript,      ///< linker script
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

  /// \brief Returns the path of the source file used to create the object
  /// file which this (File) object represents.  This information is usually
  /// parsed out of the DWARF debug information. If the source file cannot
  /// be ascertained, this method returns the empty string.
  virtual StringRef translationUnitSource() const;

  /// Returns the command line order of the file.
  uint64_t ordinal() const {
    assert(_ordinal != UINT64_MAX);
    return _ordinal;
  }

  /// Sets the command line order of the file.  The parameter must
  /// also be incremented to the next available ordinal number.
  virtual void setOrdinalAndIncrement(uint64_t &ordinal) const {
    _ordinal = ordinal;
    ++ordinal;
  }

public:
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

  virtual const TargetInfo &getTargetInfo() const = 0;

protected:
  /// \brief only subclasses of File can be instantiated
  File(StringRef p, Kind kind) : _path(p), _kind(kind), _ordinal(UINT64_MAX) {}

  /// \brief This is a convenience class for File subclasses which manage their
  /// atoms as a simple std::vector<>.
  template <typename T>
  class atom_collection_vector : public atom_collection<T> {
  public:
    virtual atom_iterator<T> begin() const {
      return atom_iterator<T>(*this,
          _atoms.empty() ? 0 : reinterpret_cast<const void *>(_atoms.data()));
    }

    virtual atom_iterator<T> end() const{
      return atom_iterator<T>(*this, _atoms.empty() ? 0 :
          reinterpret_cast<const void *>(_atoms.data() + _atoms.size()));
    }

    virtual const T *deref(const void *it) const {
      return *reinterpret_cast<const T* const*>(it);
    }

    virtual void next(const void *&it) const {
      const T *const *p = reinterpret_cast<const T *const*>(it);
      ++p;
      it = reinterpret_cast<const void*>(p);
    }

    std::vector<const T *> _atoms;
  };

  /// \brief This is a convenience class for File subclasses which need to
  /// return an empty collection.
  template <typename T>
  class atom_collection_empty : public atom_collection<T> {
  public:
    virtual atom_iterator<T> begin() const {
      return atom_iterator<T>(*this, nullptr);
    }
    virtual atom_iterator<T> end() const{
      return atom_iterator<T>(*this, nullptr);
    }
    virtual const T *deref(const void *it) const {
      llvm_unreachable("empty collection should never be accessed");
    }
    virtual void next(const void *&it) const {
    }
    virtual void push_back(const T *element) {
      llvm_unreachable("empty collection should never be grown");
    }
  };

  static atom_collection_empty<DefinedAtom>       _noDefinedAtoms;
  static atom_collection_empty<UndefinedAtom>     _noUndefinedAtoms;
  static atom_collection_empty<SharedLibraryAtom> _noSharedLibaryAtoms;
  static atom_collection_empty<AbsoluteAtom>      _noAbsoluteAtoms;

  StringRef         _path;
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

  virtual const TargetInfo &getTargetInfo() const { return _targetInfo; }

protected:
  /// \brief only subclasses of MutableFile can be instantiated
  MutableFile(const TargetInfo &ti, StringRef p)
      : File(p, kindObject), _targetInfo(ti) {}

private:
  const TargetInfo &_targetInfo;
};
} // end namespace lld

#endif
