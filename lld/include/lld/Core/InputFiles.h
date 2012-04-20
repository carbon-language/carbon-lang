//===- Core/InputFiles.h - The set of Input Files to the Linker -----------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_CORE_INPUT_FILES_H_
#define LLD_CORE_INPUT_FILES_H_

#include "lld/Core/File.h"

#include <vector>

namespace lld {

/// This InputFiles class manages access to all input files to the linker.
///
/// The forEachInitialAtom() method iterates object files to add at
/// the start of the link.
///
/// The searchLibraries() method is used to lazily search libraries.
class InputFiles {
public:
  class Handler {
  public:
    virtual ~Handler() {}
    virtual void doFile(const class File &) = 0;
    virtual void doDefinedAtom(const class DefinedAtom &) = 0;
    virtual void doUndefinedAtom(const class UndefinedAtom &) = 0;
    virtual void doSharedLibraryAtom(const class SharedLibraryAtom &) = 0;
    virtual void doAbsoluteAtom(const class AbsoluteAtom &) = 0;
  };

  InputFiles();
  virtual ~InputFiles();

  /// Used by platforms to insert platform specific files.
  virtual void prependFile(const File&);
  
  /// Used by platforms to insert platform specific files.
  virtual void appendFile(const File&);
 

  /// @brief iterates all atoms in initial files
  virtual void forEachInitialAtom(Handler &) const;

  /// @brief searches libraries for name
  virtual bool searchLibraries(  StringRef name
                               , bool searchSharedLibs
                               , bool searchArchives
                               , bool dataSymbolOnly
                               , Handler &) const;

protected:
  void handleFile(const File *file, InputFiles::Handler &handler) const;
  
  std::vector<const File*>        _files;
};

} // namespace lld

#endif // LLD_CORE_INPUT_FILES_H_
