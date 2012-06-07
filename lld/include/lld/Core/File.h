//===- Core/File.h - A Container of Atoms ---------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_CORE_FILE_H_
#define LLD_CORE_FILE_H_

#include "lld/Core/AbsoluteAtom.h"
#include "lld/Core/DefinedAtom.h"
#include "lld/Core/SharedLibraryAtom.h"
#include "lld/Core/UndefinedAtom.h"

#include "llvm/ADT/StringRef.h"

#include <vector>

namespace lld {


///
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
///
class File {
public:
  virtual ~File();

  /// Kinds of files that are supported.
  enum Kind {
    kindObject,            ///< object file (.o)
    kindSharedLibrary,     ///< shared library (.so)
    kindArchiveLibrary,    ///< archive (.a)
  };

  /// Returns file kind.  Need for dyn_cast<> on File objects.
  virtual Kind kind() const {
    return kindObject;
  }

  /// For error messages and debugging, this returns the path to the file
  /// which was used to create this object (e.g. "/tmp/foo.o").
  StringRef path() const  {
    return _path;
  }

  /// Returns the path of the source file used to create the object
  /// file which this (File) object represents.  This information is usually 
  /// parsed out of the DWARF debug information. If the source file cannot 
  /// be ascertained, this method returns the empty string.
  virtual StringRef translationUnitSource() const;


  static inline bool classof(const File *) { 
    return true; 
  }

protected:
  template <typename T> class atom_iterator; // forward reference
public:
  
  /// For use interating over DefinedAtoms in this File.
  typedef atom_iterator<DefinedAtom>  defined_iterator;

  /// For use interating over UndefinedAtoms in this File.
  typedef atom_iterator<UndefinedAtom> undefined_iterator;

  /// For use interating over SharedLibraryAtoms in this File.
  typedef atom_iterator<SharedLibraryAtom> shared_library_iterator;

  /// For use interating over AbsoluteAtoms in this File.
  typedef atom_iterator<AbsoluteAtom> absolute_iterator;



  /// Note: this method is not const.  All File objects instantiated by reading
  /// an object file from disk are "const File*" objects and cannot be 
  /// modified.  This method can only be used with temporay File objects
  /// such as is seen by each Pass object when it runs.
  /// This method is *not* safe to call while iterating through this File's 
  /// Atoms.  A Pass should queue up any Atoms it want to add and then 
  /// call this method when no longer iterating over the File's Atoms.
  virtual void addAtom(const Atom&) = 0;



protected:
  /// only subclasses of File can be instantiated 
  File(StringRef p) : _path(p) {}


  /// Different object file readers may instantiate and manage atoms with
  /// different data structures.  This class is a collection abstraction.
  /// Each concrete File instance must implement these atom_collection
  /// methods to enable clients to interate the File's atoms.
  template <typename T>
  class atom_collection {
  public:
    virtual ~atom_collection() { }
    virtual atom_iterator<T> begin() const = 0;
    virtual atom_iterator<T> end() const = 0;
    virtual const T* deref(const void* it) const = 0;
    virtual void next(const void*& it) const = 0;
  };


  /// The class is the iterator type used to iterate through a File's Atoms.
  /// This iterator delegates the work to the associated atom_collection object.
  /// There are four kinds of Atoms, so this iterator is templated on
  /// the four base Atom kinds.
  template <typename T>
  class atom_iterator {
  public:
    atom_iterator(const atom_collection<T>& c, const void* it) 
              : _collection(c), _it(it) { }

    const T* operator*() const {
      return _collection.deref(_it);
    }
    
    const T* operator->() const {
      return _collection.deref(_it);
    }

    bool operator!=(const atom_iterator<T>& other) const {
      return (this->_it != other._it);
    }

    atom_iterator<T>& operator++() {
      _collection.next(_it);
      return *this;
    }
  private:
    const atom_collection<T>&   _collection;
    const void*                 _it;
  };
  
public:
  /// Must be implemented to return the atom_collection object for 
  /// all DefinedAtoms in this File.
  virtual const atom_collection<DefinedAtom>& defined() const = 0;

  /// Must be implemented to return the atom_collection object for 
  /// all UndefinedAtomw in this File.
  virtual const atom_collection<UndefinedAtom>& undefined() const = 0;

  /// Must be implemented to return the atom_collection object for 
  /// all SharedLibraryAtoms in this File.
  virtual const atom_collection<SharedLibraryAtom>& sharedLibrary() const = 0;

  /// Must be implemented to return the atom_collection object for 
  /// all AbsoluteAtoms in this File.
  virtual const atom_collection<AbsoluteAtom>& absolute() const = 0;

protected:
  /// This is a convenience class for File subclasses which manage their
  /// atoms as a simple std::vector<>.  
  template <typename T>
  class atom_collection_vector : public atom_collection<T> {
  public:
    virtual atom_iterator<T> begin() const { 
      return atom_iterator<T>(*this, reinterpret_cast<const void*>
                                                              (_atoms.data()));
    }
    virtual atom_iterator<T> end() const{ 
      return atom_iterator<T>(*this, reinterpret_cast<const void*>
                                              (_atoms.data() + _atoms.size()));
    }
    virtual const T* deref(const void* it) const {
      return *reinterpret_cast<const T* const*>(it);
    }
    virtual void next(const void*& it) const {
      const T * const * p = reinterpret_cast<const T * const *>(it);
      ++p;
      it = reinterpret_cast<const void*>(p);
    }
    std::vector<const T*>   _atoms;
  };
  
  StringRef _path;
};

} // namespace lld

#endif // LLD_CORE_FILE_H_
