//===- Core/SharedLibraryAtom.h - A Shared Library Atom -------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_CORE_SHARED_LIBRARY_ATOM_H_
#define LLD_CORE_SHARED_LIBRARY_ATOM_H_

#include "lld/Core/Atom.h"

namespace llvm {
  class StringRef;
}

namespace lld {

/// A SharedLibraryAtom has no content.
/// It exists to represent a symbol which will be bound at runtime.
class SharedLibraryAtom : public Atom {
public:
  virtual Definition definition() const {
    return Atom::definitionSharedLibrary;
  }

  /// like dynamic_cast, if atom is definitionSharedLibrary
  /// returns atom cast to SharedLibraryAtom*, else returns NULL
  virtual const SharedLibraryAtom* sharedLibraryAtom() const { 
    return this;
  }

  /// Returns shared library name used to load it at runtime.
  /// On linux that is the DT_NEEDED name.
  /// On Darwin it is the LC_DYLIB_LOAD dylib name.
  virtual llvm::StringRef loadName() const = 0;

  /// Returns if shared library symbol can be missing at runtime and if
  /// so the loader should silently resolve address of symbol to be NULL.
  virtual bool canBeNullAtRuntime() const = 0;
  
protected:
           SharedLibraryAtom() {}
  virtual ~SharedLibraryAtom() {}
};

} // namespace lld

#endif // LLD_CORE_SHARED_LIBRARY_ATOM_H_
