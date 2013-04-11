//===- Core/Resolver.cpp - Resolves Atom References -----------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/Core/Atom.h"
#include "lld/Core/File.h"
#include "lld/Core/InputFiles.h"
#include "lld/Core/LLVM.h"
#include "lld/Core/Resolver.h"
#include "lld/Core/SymbolTable.h"
#include "lld/Core/TargetInfo.h"
#include "lld/Core/UndefinedAtom.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <vector>

namespace lld {

/// This is used as a filter function to std::remove_if to dead strip atoms.
class NotLive {
public:
       NotLive(const llvm::DenseSet<const Atom*>& la) : _liveAtoms(la) { }

  bool operator()(const Atom *atom) const {
    // don't remove if live
    if ( _liveAtoms.count(atom) )
      return false;
   // don't remove if marked never-dead-strip
    if (const DefinedAtom* defAtom = dyn_cast<DefinedAtom>(atom)) {
      if ( defAtom->deadStrip() == DefinedAtom::deadStripNever )
        return false;
    }
    // do remove this atom
    return true;
  }

private:
  const llvm::DenseSet<const Atom*> _liveAtoms;
};


/// This is used as a filter function to std::remove_if to coalesced atoms.
class AtomCoalescedAway {
public:
  AtomCoalescedAway(SymbolTable &sym) : _symbolTable(sym) {}

  bool operator()(const Atom *atom) const {
    const Atom *rep = _symbolTable.replacement(atom);
    return rep != atom;
  }

private:
  SymbolTable &_symbolTable;
};


// add all atoms from all initial .o files
void Resolver::buildInitialAtomList() {
 DEBUG_WITH_TYPE("resolver", llvm::dbgs() << "Resolver initial atom list:\n");

  // each input files contributes initial atoms
  _atoms.reserve(1024);
  _inputFiles.forEachInitialAtom(*this);

  _completedInitialObjectFiles = true;
}


// called before the first atom in any file is added with doAtom()
void Resolver::doFile(const File &file) {
}


void Resolver::doUndefinedAtom(const class UndefinedAtom& atom) {
  DEBUG_WITH_TYPE("resolver", llvm::dbgs()
                    << "       UndefinedAtom: "
                    << llvm::format("0x%09lX", &atom)
                    << ", name="
                    << atom.name()
                    << "\n");

 // add to list of known atoms
  _atoms.push_back(&atom);

  // tell symbol table
  _symbolTable.add(atom);
}


// called on each atom when a file is added
void Resolver::doDefinedAtom(const DefinedAtom &atom) {
  DEBUG_WITH_TYPE("resolver", llvm::dbgs()
                    << "         DefinedAtom: "
                    << llvm::format("0x%09lX", &atom)
                    << ", file=#"
                    << atom.file().ordinal()
                    << ", atom=#"
                    << atom.ordinal()
                    << ", name="
                    << atom.name()
                    << "\n");

  // Verify on zero-size atoms are pinned to start or end of section.
  switch ( atom.sectionPosition() ) {
  case DefinedAtom::sectionPositionStart:
  case DefinedAtom::sectionPositionEnd:
    assert(atom.size() == 0);
    break;
  case DefinedAtom::sectionPositionEarly:
  case DefinedAtom::sectionPositionAny:
    break;
  }

  // add to list of known atoms
  _atoms.push_back(&atom);

  // tell symbol table
  _symbolTable.add(atom);

  if (_targetInfo.deadStrip()) {
    // add to set of dead-strip-roots, all symbols that
    // the compiler marks as don't strip
    if (atom.deadStrip() == DefinedAtom::deadStripNever)
      _deadStripRoots.insert(&atom);
  }
}

void Resolver::doSharedLibraryAtom(const SharedLibraryAtom& atom) {
   DEBUG_WITH_TYPE("resolver", llvm::dbgs()
                    << "   SharedLibraryAtom: "
                    << llvm::format("0x%09lX", &atom)
                    << ", name="
                    << atom.name()
                    << "\n");

  // add to list of known atoms
  _atoms.push_back(&atom);

  // tell symbol table
  _symbolTable.add(atom);
}

void Resolver::doAbsoluteAtom(const AbsoluteAtom& atom) {
   DEBUG_WITH_TYPE("resolver", llvm::dbgs()
                    << "       AbsoluteAtom: "
                    << llvm::format("0x%09lX", &atom)
                    << ", name="
                     << atom.name()
                    << "\n");

  // add to list of known atoms
  _atoms.push_back(&atom);

  // tell symbol table
  if (atom.scope() != Atom::scopeTranslationUnit) {
    _symbolTable.add(atom);
  }
}



// utility to add a vector of atoms
void Resolver::addAtoms(const std::vector<const DefinedAtom*>& newAtoms) {
  for (std::vector<const DefinedAtom *>::const_iterator it = newAtoms.begin();
       it != newAtoms.end(); ++it) {
    this->doDefinedAtom(**it);
  }
}

// ask symbol table if any definitionUndefined atoms still exist
// if so, keep searching libraries until no more atoms being added
void Resolver::resolveUndefines() {
  const bool searchArchives =
      _targetInfo.searchArchivesToOverrideTentativeDefinitions();
  const bool searchSharedLibs =
      _targetInfo.searchSharedLibrariesToOverrideTentativeDefinitions();

  // keep looping until no more undefines were added in last loop
  unsigned int undefineGenCount = 0xFFFFFFFF;
  while (undefineGenCount != _symbolTable.size()) {
    undefineGenCount = _symbolTable.size();
    std::vector<const UndefinedAtom *> undefines;
    _symbolTable.undefines(undefines);
    for ( const Atom *undefAtom : undefines ) {
      StringRef undefName = undefAtom->name();
      // load for previous undefine may also have loaded this undefine
      if (!_symbolTable.isDefined(undefName)) {
        _inputFiles.searchLibraries(undefName, true, true, false, *this);
      }
    }
    // search libraries for overrides of common symbols
    if (searchArchives || searchSharedLibs) {
      std::vector<StringRef> tentDefNames;
      _symbolTable.tentativeDefinitions(tentDefNames);
      for ( StringRef tentDefName : tentDefNames ) {
        // Load for previous tentative may also have loaded
        // something that overrode this tentative, so always check.
        const Atom *curAtom = _symbolTable.findByName(tentDefName);
        assert(curAtom != nullptr);
        if (const DefinedAtom* curDefAtom = dyn_cast<DefinedAtom>(curAtom)) {
          if (curDefAtom->merge() == DefinedAtom::mergeAsTentative ) {
            // Still tentative definition, so look for override.
            _inputFiles.searchLibraries(tentDefName, searchSharedLibs,
                                        searchArchives, true, *this);
          }
        }
      }
    }
  }
}


// switch all references to undefined or coalesced away atoms
// to the new defined atom
void Resolver::updateReferences() {
  for(const Atom *atom : _atoms) {
    if (const DefinedAtom* defAtom = dyn_cast<DefinedAtom>(atom)) {
      for (const Reference *ref : *defAtom) {
        const Atom* newTarget = _symbolTable.replacement(ref->target());
        (const_cast<Reference*>(ref))->setTarget(newTarget);
      }
    }
  }
}


// for dead code stripping, recursively mark atoms "live"
void Resolver::markLive(const Atom &atom) {
  // if already marked live, then done (stop recursion)
  if ( _liveAtoms.count(&atom) )
    return;

  // mark this atom is live
  _liveAtoms.insert(&atom);

  // mark all atoms it references as live
  if ( const DefinedAtom* defAtom = dyn_cast<DefinedAtom>(&atom)) {
    for (const Reference *ref : *defAtom) {
      const Atom *target = ref->target();
      if ( target != nullptr )
        this->markLive(*target);
    }
  }
}


// remove all atoms not actually used
void Resolver::deadStripOptimize() {
  // only do this optimization with -dead_strip
  if (!_targetInfo.deadStrip())
    return;

  // clear liveness on all atoms
  _liveAtoms.clear();

  // By default, shared libraries are built with all globals as dead strip roots
  if (_targetInfo.globalsAreDeadStripRoots()) {
    for ( const Atom *atom : _atoms ) {
      const DefinedAtom *defAtom = dyn_cast<DefinedAtom>(atom);
      if (defAtom == nullptr)
        continue;
      if ( defAtom->scope() == DefinedAtom::scopeGlobal )
        _deadStripRoots.insert(defAtom);
    }
  }

  // Or, use list of names that are dead stip roots.
  for (const StringRef &name : _targetInfo.deadStripRoots()) {
    const Atom *symAtom = _symbolTable.findByName(name);
    assert(symAtom->definition() != Atom::definitionUndefined);
    _deadStripRoots.insert(symAtom);
  }

  // mark all roots as live, and recursively all atoms they reference
  for ( const Atom *dsrAtom : _deadStripRoots) {
    this->markLive(*dsrAtom);
  }

  // now remove all non-live atoms from _atoms
  _atoms.erase(std::remove_if(_atoms.begin(), _atoms.end(),
                              NotLive(_liveAtoms)), _atoms.end());
}


// error out if some undefines remain
bool Resolver::checkUndefines(bool final) {
  // when using LTO, undefines are checked after bitcode is optimized
  if (_haveLLVMObjs && !final)
    return false;

  // build vector of remaining undefined symbols
  std::vector<const UndefinedAtom *> undefinedAtoms;
  _symbolTable.undefines(undefinedAtoms);
  if (_targetInfo.deadStrip()) {
    // When dead code stripping, we don't care if dead atoms are undefined.
    undefinedAtoms.erase(std::remove_if(
                           undefinedAtoms.begin(), undefinedAtoms.end(),
                           NotLive(_liveAtoms)), undefinedAtoms.end());
  }

  // error message about missing symbols
  if (!undefinedAtoms.empty()) {
    // FIXME: need diagnostics interface for writing error messages
    bool foundUndefines = false;
    for (const UndefinedAtom *undefAtom : undefinedAtoms) {
      const File &f = undefAtom->file();
      bool isAtomUndefined = false;
      if (isa<SharedLibraryFile>(f)) {
        if (!_targetInfo.allowShlibUndefines()) {
          foundUndefines = true;
          isAtomUndefined = true;
        }
      } else if (undefAtom->canBeNull() == UndefinedAtom::canBeNullNever) {
        foundUndefines = true;
        isAtomUndefined = true;
      }
      if (isAtomUndefined && _targetInfo.printRemainingUndefines()) {
        llvm::errs() << "Undefined Symbol: " << undefAtom->file().path()
                     << " : " << undefAtom->name() << "\n";
      }
    }
    if (foundUndefines) {
      if (_targetInfo.printRemainingUndefines())
        llvm::errs() << "symbol(s) not found\n";
      return true;
    }
  }
  return false;
}


// remove from _atoms all coaleseced away atoms
void Resolver::removeCoalescedAwayAtoms() {
  _atoms.erase(std::remove_if(_atoms.begin(), _atoms.end(),
                              AtomCoalescedAway(_symbolTable)), _atoms.end());
}

// check for interactions between symbols defined in this linkage unit
// and same symbol name in linked dynamic shared libraries
void Resolver::checkDylibSymbolCollisions() {
  for ( const Atom *atom : _atoms ) {
    const DefinedAtom* defAtom = dyn_cast<DefinedAtom>(atom);
    if (defAtom == nullptr)
      continue;
    if ( defAtom->merge() != DefinedAtom::mergeAsTentative )
      continue;
    assert(defAtom->scope() != DefinedAtom::scopeTranslationUnit);
    // See if any shared library also has symbol which
    // collides with the tentative definition.
    // SymbolTable will warn if needed.
    _inputFiles.searchLibraries(defAtom->name(), true, false, false, *this);
  }
}


void Resolver::linkTimeOptimize() {
  // FIX ME
}

bool Resolver::resolve() {
  this->buildInitialAtomList();
  this->resolveUndefines();
  this->updateReferences();
  this->deadStripOptimize();
  if (this->checkUndefines(false)) {
    if (!_targetInfo.allowRemainingUndefines())
      return true;
  }
  this->removeCoalescedAwayAtoms();
  this->checkDylibSymbolCollisions();
  this->linkTimeOptimize();
  this->_result.addAtoms(_atoms);
  return false;
}

void Resolver::MergedFile::addAtom(const Atom& atom) {
  if (const DefinedAtom* defAtom = dyn_cast<DefinedAtom>(&atom)) {
    _definedAtoms._atoms.push_back(defAtom);
  } else if (const UndefinedAtom* undefAtom = dyn_cast<UndefinedAtom>(&atom)) {
    _undefinedAtoms._atoms.push_back(undefAtom);
  } else if (const SharedLibraryAtom* slAtom =
               dyn_cast<SharedLibraryAtom>(&atom)) {
    _sharedLibraryAtoms._atoms.push_back(slAtom);
  } else if (const AbsoluteAtom* abAtom = dyn_cast<AbsoluteAtom>(&atom)) {
    _absoluteAtoms._atoms.push_back(abAtom);
  } else {
    llvm_unreachable("atom has unknown definition kind");
  }
}


MutableFile::DefinedAtomRange Resolver::MergedFile::definedAtoms() {
  return range<std::vector<const DefinedAtom*>::iterator>(
                    _definedAtoms._atoms.begin(), _definedAtoms._atoms.end());
}



void Resolver::MergedFile::addAtoms(std::vector<const Atom*>& all) {
  DEBUG_WITH_TYPE("resolver", llvm::dbgs() << "Resolver final atom list:\n");
  for ( const Atom *atom : all ) {
    DEBUG_WITH_TYPE("resolver", llvm::dbgs()
                    << llvm::format("    0x%09lX", atom)
                    << ", name="
                    << atom->name()
                    << "\n");
    this->addAtom(*atom);
  }
}


} // namespace lld
