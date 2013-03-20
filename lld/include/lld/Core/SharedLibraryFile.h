//===- Core/SharedLibraryFile.h - Models shared libraries as Atoms --------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_CORE_SHARED_LIBRARY_FILE_H_
#define LLD_CORE_SHARED_LIBRARY_FILE_H_

#include "lld/Core/File.h"
#include "lld/Core/SharedLibraryAtom.h"

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
protected:
  /// only subclasses of SharedLibraryFile can be instantiated
  SharedLibraryFile(StringRef path) : File(path, kindSharedLibrary) {}
};

} // namespace lld

#endif // LLD_CORE_SHARED_LIBRARY_FILE_H_
