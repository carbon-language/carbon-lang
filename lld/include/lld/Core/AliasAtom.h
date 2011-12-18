//===- Core/AliasAtom.h - Alias to another Atom ---------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_CORE_ALIAS_ATOM_H_
#define LLD_CORE_ALIAS_ATOM_H_

#include "lld/Core/Atom.h"

#include "llvm/ADT/StringRef.h"

namespace lld {

class AliasAtom : public Atom {
public:
  AliasAtom(llvm::StringRef nm, const Atom &target,  Atom::Scope scope)
    : Atom( target.definition()
          , Atom::combineNever
          , scope
          , target.contentType()
          , target.sectionChoice()
          , target.userVisibleName()
          , target.deadStrip()
          , target.isThumb()
          , true
          , target.alignment()
          )
    , _name(nm)
    , _aliasOf(target) {}

  // overrides of Atom
  virtual const File *file() const {
    return _aliasOf.file();
  }

  virtual bool translationUnitSource(llvm::StringRef &path) const {
    return _aliasOf.translationUnitSource(path);
  }

  virtual llvm::StringRef name() const {
    return _name;
  }

private:
  const llvm::StringRef _name;
  const Atom &_aliasOf;
};

} // namespace lld

#endif // LLD_CORE_ALIAS_ATOM_H_
