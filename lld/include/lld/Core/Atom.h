//===- Core/Atom.h - A node in linking graph ------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_CORE_ATOM_H_
#define LLD_CORE_ATOM_H_

#include <assert.h>

namespace llvm {
  class StringRef;
}

namespace lld {

class File;
class DefinedAtom;

///
/// The linker has a Graph Theory model of linking. An object file is seen
/// as a set of Atoms with References to other Atoms.  Each Atom is a node
/// and each Reference is an edge. An Atom can be a DefinedAtom which has
/// content or a UndefinedAtom which is a placeholder and represents an 
/// undefined symbol (extern declaration).
///
class Atom {
public:
  /// Whether this atom is defined or a proxy for an undefined symbol
  enum Definition {
    definitionRegular,      ///< Normal C/C++ function or global variable.
    definitionAbsolute,     ///< Asm-only (foo = 10). Not tied to any content.
    definitionUndefined,    ///< Only in .o files to model reference to undef.
    definitionSharedLibrary ///< Only in shared libraries to model export.
  };

  /// file - returns the File that produced/owns this Atom
  virtual const class File& file() const = 0;

  /// name - The name of the atom. For a function atom, it is the (mangled)
  /// name of the function. 
  virtual llvm::StringRef name() const = 0;
  
  /// definition - Whether this atom is a definition or represents an undefined
  /// symbol.
  virtual Definition definition() const = 0;
  
  /// definedAtom - like dynamic_cast, if atom is definitionRegular
  /// returns atom cast to DefinedAtom*, else returns NULL;
  virtual const DefinedAtom* definedAtom() const { return NULL; }
  
protected:
  /// Atom is an abstract base class.  Only subclasses can access constructor.
  Atom() {}
  
  /// The memory for Atom objects is always managed by the owning File
  /// object.  Therefore, no one but the owning File object should call
  /// delete on an Atom.  In fact, some File objects may bulk allocate
  /// an array of Atoms, so they cannot be individually deleted by anyone.
  virtual ~Atom() {}
};

} // namespace lld

#endif // LLD_CORE_ATOM_H_
