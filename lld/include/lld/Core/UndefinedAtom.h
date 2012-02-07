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
  virtual Definition definition() const {
    return Atom::definitionUndefined;
  }

  /// like dynamic_cast, if atom is definitionUndefined
  /// returns atom cast to UndefinedAtom*, else returns NULL
  virtual const UndefinedAtom* undefinedAtom() const { 
    return this;
  }

  /// returns if undefined symbol can be missing at runtime
  virtual bool weakImport() const = 0;
  
protected:
           UndefinedAtom() {}
  virtual ~UndefinedAtom() {}
};

} // namespace lld

#endif // LLD_CORE_UNDEFINED_ATOM_H_
