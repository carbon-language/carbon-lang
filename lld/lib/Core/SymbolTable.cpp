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
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/ArrayRef.h"

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
    this->addByContent(atom);
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
    // Name is not in symbol table yet, add it associate with this atom.
    _nameTable[name] = &newAtom;
  } 
  else {
    // Name is already in symbol table and associated with another atom.
    bool useNew = true;
    switch (collide(existing->definition(), newAtom.definition())) {
    case NCR_First:
      useNew = false;
      break;
    case NCR_Second:
      useNew = true;
      break;
    case NCR_Dup:
      if ( existing->mergeDuplicates() && newAtom.mergeDuplicates() ) {
        // Both mergeable.  Use auto-hide bit as tie breaker
        if ( existing->autoHide() != newAtom.autoHide() ) {
          // They have different autoHide values, keep non-autohide one
          useNew = existing->autoHide();
        }
        else {
          // They have same autoHide, so just keep using existing
          useNew = false;
        }
      }
      else {
        const Atom& use = _platform.handleMultipleDefinitions(*existing, newAtom);
        useNew = ( &use != existing ); 
      }
      break;
    default:
      llvm::report_fatal_error("SymbolTable::addByName(): unhandled switch clause");
    }
    if ( useNew ) {
      // Update name table to use new atom.
      _nameTable[name] = &newAtom;
      // Add existing atom to replacement table.
      _replacedAtoms[existing] = &newAtom;
    }
    else {
      // New atom is not being used.  Add it to replacement table.
      _replacedAtoms[&newAtom] = existing;
    }
  }
}


unsigned SymbolTable::MyMappingInfo::getHashValue(const Atom * const atom) {
  unsigned hash = atom->size();
  if ( atom->contentType() != Atom::typeZeroFill ) {
    llvm::ArrayRef<uint8_t> content = atom->rawContent();
    for (unsigned int i=0; i < content.size(); ++i) {
      hash = hash * 33 + content[i];
    }
  }
  hash &= 0x00FFFFFF;
  hash |= ((unsigned)atom->contentType()) << 24;
  //fprintf(stderr, "atom=%p, hash=0x%08X\n", atom, hash);
  return hash;
}


bool SymbolTable::MyMappingInfo::isEqual(const Atom * const l, 
                                         const Atom * const r) {
  if ( l == r )
    return true;
  if ( l == getEmptyKey() )
    return false;
  if ( r == getEmptyKey() )
    return false;
  if ( l == getTombstoneKey() )
    return false;
  if ( r == getTombstoneKey() )
    return false;
    
  if ( l->contentType() != r->contentType() )
    return false;
  if ( l->size() != r->size() )
    return false;
  llvm::ArrayRef<uint8_t> lc = l->rawContent();
  llvm::ArrayRef<uint8_t> rc = r->rawContent();
  return lc.equals(rc);
}


void SymbolTable::addByContent(const Atom & newAtom) {
  AtomContentSet::iterator pos = _contentTable.find(&newAtom);
  if ( pos == _contentTable.end() ) {
    _contentTable.insert(&newAtom);
    return;
  }
  const Atom* existing = *pos;
    // New atom is not being used.  Add it to replacement table.
    _replacedAtoms[&newAtom] = existing;
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
