//===- Core/AbsoluteAtom.h - An absolute Atom -----------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_CORE_ABSOLUTE_ATOM_H_
#define LLD_CORE_ABSOLUTE_ATOM_H_

#include "lld/Core/Atom.h"

namespace lld {

/// An AbsoluteAtom has no content.
/// It exists to represent content at fixed addresses in memory.
class AbsoluteAtom : public Atom {
public:
  virtual Definition definition() const {
    return Atom::definitionAbsolute;
  }

  /// like dynamic_cast, if atom is definitionAbsolute
  /// returns atom cast to AbsoluteAtom*, else returns nullptr
  virtual const AbsoluteAtom* absoluteAtom() const { 
    return this;
  }
  
  virtual uint64_t value() const = 0;
  
  
protected:
           AbsoluteAtom() {}
  virtual ~AbsoluteAtom() {}
};

} // namespace lld

#endif // LLD_CORE_ABSOLUTE_ATOM_H_
