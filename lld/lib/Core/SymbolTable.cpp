//===- Core/SymbolTable.cpp - Main Symbol Table ---------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/Core/SymbolTable.h"
#include "lld/Core/Atom.h"
#include "lld/Core/File.h"
#include "lld/Core/InputFiles.h"
#include "lld/Core/Resolver.h"
#include "lld/Core/UndefinedAtom.h"
#include "lld/Platform/Platform.h"

#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <cassert>
#include <stdlib.h>
#include <vector>

namespace lld {

SymbolTable::SymbolTable(Platform& plat)
  : _platform(plat) {
}

void SymbolTable::add(const Atom &atom) {
  assert(atom.scope() != Atom::scopeTranslationUnit);
  if ( !atom.internalName() ) {
    this->addByName(atom);
  }
  else if ( atom.mergeDuplicates() ) {
    // TO DO: support constants merging
  }
}

enum NameCollisionResolution {
  NCR_First,
  NCR_Second,
  NCR_Weak,
  NCR_Larger,
  NCR_Dup,
  NCR_Error
};

static NameCollisionResolution cases[6][6] = {
  //regular     weak         tentative   absolute    undef      sharedLib
  {
    // first is regular
    NCR_Dup,    NCR_First,   NCR_First,  NCR_Error,  NCR_First, NCR_First
  },
  {
    // first is weak
    NCR_Second, NCR_Weak,   NCR_Larger, NCR_Error,  NCR_First, NCR_First
  },
  {
    // first is tentative
    NCR_Second, NCR_Second, NCR_Larger, NCR_Error,  NCR_First, NCR_First
  },
  {
    // first is absolute
    NCR_Error,  NCR_Error,  NCR_Error,  NCR_Error,  NCR_First, NCR_First
  },
  {
    // first is undef
    NCR_Second, NCR_Second, NCR_Second, NCR_Second, NCR_First, NCR_Second
  },
  {
    // first is sharedLib
    NCR_Second, NCR_Second, NCR_Second, NCR_Second, NCR_First, NCR_First
  }
};

static NameCollisionResolution collide(Atom::Definition first,
                                       Atom::Definition second) {
  return cases[first][second];
}

void SymbolTable::addByName(const Atom & newAtom) {
  llvm::StringRef name = newAtom.name();
  const Atom *existing = this->findByName(name);
  if (existing == NULL) {
    // name is not in symbol table yet, add it associate with this atom
    _nameTable[name] = &newAtom;
  } else {
    // name is already in symbol table and associated with another atom
    switch (collide(existing->definition(), newAtom.definition())) {
    case NCR_First:
      // using first, just add new to _replacedAtoms
      _replacedAtoms[&newAtom] = existing;
      break;
    case NCR_Second:
      // using second, update tables
      _nameTable[name] = &newAtom;
      _replacedAtoms[existing] = &newAtom;
      break;
    case NCR_Dup:
      if ( existing->mergeDuplicates() && newAtom.mergeDuplicates() ) {
          // using existing atom, add new atom to _replacedAtoms
        _replacedAtoms[&newAtom] = existing;
      }
      else {
        const Atom& use = _platform.handleMultipleDefinitions(*existing, newAtom);
        if ( &use == existing ) {
          // using existing atom, add new atom to _replacedAtoms
          _replacedAtoms[&newAtom] = existing;
        }
        else {
          // using new atom, update tables
          _nameTable[name] = &newAtom;
          _replacedAtoms[existing] = &newAtom;
        }
      }
      break;
    default:
      llvm::report_fatal_error("SymbolTable::addByName(): unhandled switch clause");
    }
  }
}

const Atom *SymbolTable::findByName(llvm::StringRef sym) {
  NameToAtom::iterator pos = _nameTable.find(sym);
  if (pos == _nameTable.end())
    return NULL;
  return pos->second;
}

bool SymbolTable::isDefined(llvm::StringRef sym) {
  const Atom *atom = this->findByName(sym);
  if (atom == NULL)
    return false;
  if (atom->definition() == Atom::definitionUndefined)
    return false;
  return true;
}

const Atom *SymbolTable::replacement(const Atom *atom) {
  AtomToAtom::iterator pos = _replacedAtoms.find(atom);
  if (pos == _replacedAtoms.end())
    return atom;
  // might be chain, recurse to end
  return this->replacement(pos->second);
}

unsigned int SymbolTable::size() {
  return _nameTable.size();
}

void SymbolTable::undefines(std::vector<const Atom *> &undefs) {
  for (NameToAtom::iterator it = _nameTable.begin(),
       end = _nameTable.end(); it != end; ++it) {
    const Atom *atom = it->second;
    assert(atom != NULL);
    if (atom->definition() == Atom::definitionUndefined)
      undefs.push_back(atom);
  }
}

} // namespace lld
