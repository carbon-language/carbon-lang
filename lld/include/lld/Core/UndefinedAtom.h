//===- Core/UndefinedAtom.h - An Undefined Atom ---------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_CORE_UNDEFINED_ATOM_H_
#define LLD_CORE_UNDEFINED_ATOM_H_

#include "lld/Core/Atom.h"

#include "llvm/ADT/StringRef.h"

namespace lld {

/// An UndefinedAtom has no content.
/// It exists as a place holder for a future atom.
class UndefinedAtom : public Atom {
public:
  UndefinedAtom(llvm::StringRef nm, bool weakImport, const File& f)
          : _name(nm), _file(f), _weakImport(weakImport) {}

  virtual const File& file() const {
    return _file;
  }

  virtual llvm::StringRef name() const {
    return _name;
  }

  virtual Definition definition() const {
    return Atom::definitionUndefined;
  }

  virtual bool weakImport() const {
    return _weakImport;
  }

protected:
  virtual ~UndefinedAtom() {}

  llvm::StringRef _name;
  const File&     _file;
  bool            _weakImport;
};

} // namespace lld

#endif // LLD_CORE_UNDEFINED_ATOM_H_
