//===- Core/SharedLibraryFile.h - Models shared libraries as Atoms --------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_CORE_SHARED_LIBRARY_FILE_H
#define LLD_CORE_SHARED_LIBRARY_FILE_H

#include "lld/Core/File.h"

namespace lld {

///
/// The SharedLibraryFile subclass of File is used to represent dynamic
/// shared libraries being linked against.
///
class SharedLibraryFile : public File {
public:
  virtual ~SharedLibraryFile() {}

  static inline bool classof(const File *f) {
    return f->kind() == kindSharedLibrary;
  }

  /// Check if the shared library exports a symbol with the specified name.
  /// If so, return a SharedLibraryAtom which represents that exported
  /// symbol.  Otherwise return nullptr.
  virtual const SharedLibraryAtom *exports(StringRef name,
                                           bool dataSymbolOnly) const = 0;

  // Returns DSO name. It's the soname (ELF), the install name (MachO) or
  // the import name (Windows).
  virtual StringRef getDSOName() const = 0;

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

protected:
  /// only subclasses of SharedLibraryFile can be instantiated
  explicit SharedLibraryFile(StringRef path) : File(path, kindSharedLibrary) {}

  atom_collection_vector<DefinedAtom>        _definedAtoms;
  atom_collection_vector<UndefinedAtom>      _undefinedAtoms;
  atom_collection_vector<SharedLibraryAtom>  _sharedLibraryAtoms;
  atom_collection_vector<AbsoluteAtom>       _absoluteAtoms;
};

} // namespace lld

#endif // LLD_CORE_SHARED_LIBRARY_FILE_H
