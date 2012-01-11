//===- Core/File.h - A Contaier of Atoms ----------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_CORE_FILE_H_
#define LLD_CORE_FILE_H_

#include "llvm/ADT/StringRef.h"

#include "lld/Core/DefinedAtom.h"
#include "lld/Core/UndefinedAtom.h"

namespace lld {

class File {
public:
  File(llvm::StringRef p) : _path(p) {}
  ~File();

  class AtomHandler {
  public:
    virtual ~AtomHandler() {}
    virtual void doDefinedAtom(const class DefinedAtom &) = 0;
    virtual void doUndefinedAtom(const class UndefinedAtom &) = 0;
    virtual void doFile(const class File &) = 0;
  };

  llvm::StringRef path() const  {
    return _path;
  }

  virtual bool forEachAtom(AtomHandler &) const = 0;
  virtual bool justInTimeforEachAtom( llvm::StringRef name
                                    , AtomHandler &) const = 0;

  virtual bool translationUnitSource(llvm::StringRef &path) const;


private:
  llvm::StringRef _path;
};

} // namespace lld

#endif // LLD_CORE_FILE_H_
