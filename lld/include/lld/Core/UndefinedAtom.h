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

  /// Whether this undefined symbol needs to be resolved,
  /// or whether it can just evaluate to NULL.
  /// This concept is often called "weak", but that term
  /// is overloaded to mean other things too.
  enum CanBeNull {
    /// Normal symbols must be resolved at build time
    canBeNullNever,
    
    /// This symbol can be missing at runtime and will evalute to NULL.
    /// That is, the static linker still must find a definition (usually
    /// is some shared library), but at runtime, the dynamic loader
    /// will allow the symbol to be missing and resolved to NULL.
    ///
    /// On Darwin this is generated using a function prototype with
    /// __attribute__((weak_import)).  
    /// On linux this is generated using a function prototype with
    ///  __attribute__((weak)).
    canBeNullAtRuntime,
    
    
    /// This symbol can be missing at build time.
    /// That is, the static linker will not error if a definition for
    /// this symbol is not found at build time. Instead, the linker 
    /// will build an executable that lets the dynamic loader find the
    /// symbol at runtime.  
    /// This feature is not supported on Darwin.
    /// On linux this is generated using a function prototype with
    ///  __attribute__((weak)).
    canBeNullAtBuildtime
  };
  
  virtual CanBeNull canBeNull() const = 0;
  
   
protected:
           UndefinedAtom() {}
  virtual ~UndefinedAtom() {}
};

} // namespace lld

#endif // LLD_CORE_UNDEFINED_ATOM_H_
