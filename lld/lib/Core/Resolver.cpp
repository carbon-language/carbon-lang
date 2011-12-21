//===- Core/Resolver.cpp - Resolves Atom References -----------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/Core/Resolver.h"
#include "lld/Core/Atom.h"
#include "lld/Core/File.h"
#include "lld/Core/InputFiles.h"
#include "lld/Core/SymbolTable.h"
#include "lld/Core/UndefinedAtom.h"
#include "lld/Platform/Platform.h"

#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <vector>

namespace lld {

class NotLive {
public:
  bool operator()(const Atom *atom) const {
    return !(atom->live() || !atom->deadStrip());
  }
};

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

void Resolver::initializeState() {
  _platform.initialize();
}

// add initial undefines from -u option
void Resolver::addInitialUndefines() {

}

// add all atoms from all initial .o files
void Resolver::buildInitialAtomList() {
  // each input files contributes initial atoms
  _atoms.reserve(1024);
  _inputFiles.forEachInitialAtom(*this);

  _completedInitialObjectFiles = true;
}


// called before the first atom in any file is added with doAtom()
void Resolver::doFile(const File &file) {
  // notify platform
  _platform.fileAdded(file);
}

// called on each atom when a file is added
void Resolver::doAtom(const Atom &atom) {
  // notify platform
  _platform.atomAdded(atom);

  // add to list of known atoms
  _atoms.push_back(&atom);

  // adjust scope (e.g. force some globals to be hidden)
  _platform.adjustScope(atom);

  // non-static atoms need extra handling
  if (atom.scope() != Atom::scopeTranslationUnit) {
    // tell symbol table about non-static atoms
    _symbolTable.add(atom);

    // platform can add aliases for any symbol
    std::vector<const Atom *> aliases;
    if (_platform.getAliasAtoms(atom, aliases))
      this->addAtoms(aliases);
  }

  if (_platform.deadCodeStripping()) {
    // add to set of dead-strip-roots, all symbols that
    // the compiler marks as don't strip
    if (!atom.deadStrip())
      _deadStripRoots.insert(&atom);

    // add to set of dead-strip-roots, all symbols that
    // the platform decided must remain
    if (_platform.isDeadStripRoot(atom))
      _deadStripRoots.insert(&atom);
  }
}

// utility to add a vector of atoms
void Resolver::addAtoms(const std::vector<const Atom *> &newAtoms) {
  for (std::vector<const Atom *>::const_iterator it = newAtoms.begin();
       it != newAtoms.end(); ++it) {
    this->doAtom(**it);
  }
}

// ask symbol table if any definitionUndefined atoms still exist
// if so, keep searching libraries until no more atoms being added
void Resolver::resolveUndefines() {
  const bool searchArchives =
    _platform.searchArchivesToOverrideTentativeDefinitions();
  const bool searchDylibs =
    _platform.searchSharedLibrariesToOverrideTentativeDefinitions();

  // keep looping until no more undefines were added in last loop
  unsigned int undefineGenCount = 0xFFFFFFFF;
  while (undefineGenCount != _symbolTable.size()) {
    undefineGenCount = _symbolTable.size();
    std::vector<const Atom *> undefines;
    _symbolTable.undefines(undefines);
    for (std::vector<const Atom *>::iterator it = undefines.begin();
         it != undefines.end(); ++it) {
      llvm::StringRef undefName = (*it)->name();
      // load for previous undefine may also have loaded this undefine
      if (!_symbolTable.isDefined(undefName)) {
        _inputFiles.searchLibraries(undefName, true, true, false, *this);

        // give platform a chance to instantiate platform
        // specific atoms (e.g. section boundary)
        if (!_symbolTable.isDefined(undefName)) {
          std::vector<const Atom *> platAtoms;
          if (_platform.getPlatformAtoms(undefName, platAtoms))
            this->addAtoms(platAtoms);
        }
      }
    }
    // search libraries for overrides of common symbols
    if (searchArchives || searchDylibs) {
      std::vector<const Atom *> tents;
      for (std::vector<const Atom *>::iterator ait = _atoms.begin();
           ait != _atoms.end(); ++ait) {
        const Atom *atom = *ait;
        if (atom->definition() == Atom::definitionTentative)
          tents.push_back(atom);
      }
      for (std::vector<const Atom *>::iterator dit = tents.begin();
           dit != tents.end(); ++dit) {
        // load for previous tentative may also have loaded
        // this tentative, so check again
        llvm::StringRef tentName = (*dit)->name();
        const Atom *curAtom = _symbolTable.findByName(tentName);
        assert(curAtom != NULL);
        if (curAtom->definition() == Atom::definitionTentative) {
          _inputFiles.searchLibraries(tentName, searchDylibs, true, true,
                                      *this);
        }
      }
    }
  }
}

// switch all references to undefined or coalesced away atoms
// to the new defined atom
void Resolver::updateReferences() {
  for (std::vector<const Atom *>::iterator it = _atoms.begin();
       it != _atoms.end(); ++it) {
    const Atom *atom = *it;
    for (Reference::iterator rit = atom->referencesBegin(),
         end = atom->referencesEnd(); rit != end; ++rit) {
      rit->target = _symbolTable.replacement(rit->target);
    }
  }
}

// for dead code stripping, recursively mark atom "live"
void Resolver::markLive(const Atom &atom, WhyLiveBackChain *previous) {
  // if -why_live cares about this symbol, then dump chain
  if ((previous->referer != NULL) && _platform.printWhyLive(atom.name())) {
    llvm::errs() << atom.name() << " from " << atom.file().path() << "\n";
    int depth = 1;
    for (WhyLiveBackChain *p = previous; p != NULL;
         p = p->previous, ++depth) {
      for (int i = depth; i > 0; --i)
        llvm::errs() << "  ";
      llvm::errs() << p->referer->name() << " from "
                   << p->referer->file().path() << "\n";
    }
  }

  // if already marked live, then done (stop recursion)
  if (atom.live())
    return;

  // mark this atom is live
  const_cast<Atom *>(&atom)->setLive(true);

  // mark all atoms it references as live
  WhyLiveBackChain thisChain;
  thisChain.previous = previous;
  thisChain.referer = &atom;
  for (Reference::iterator rit = atom.referencesBegin(),
       end = atom.referencesEnd(); rit != end; ++rit) {
    this->markLive(*(rit->target), &thisChain);
  }
}

// remove all atoms not actually used
void Resolver::deadStripOptimize() {
  // only do this optimization with -dead_strip
  if (!_platform.deadCodeStripping())
    return;

  // clear liveness on all atoms
  for (std::vector<const Atom *>::iterator it = _atoms.begin();
       it != _atoms.end(); ++it) {
    const Atom *atom = *it;
    const_cast<Atom *>(atom)->setLive(0);
  }

  // add entry point (main) to live roots
  const Atom *entry = this->entryPoint();
  if (entry != NULL)
    _deadStripRoots.insert(entry);

  // add -exported_symbols_list, -init, and -u entries to live roots
  for (Platform::UndefinesIterator uit = _platform.initialUndefinesBegin();
       uit != _platform.initialUndefinesEnd(); ++uit) {
    llvm::StringRef sym = *uit;
    const Atom *symAtom = _symbolTable.findByName(sym);
    assert(symAtom->definition() != Atom::definitionUndefined);
    _deadStripRoots.insert(symAtom);
  }

  // add platform specific helper atoms
  std::vector<const Atom *> platRootAtoms;
  if (_platform.getImplicitDeadStripRoots(platRootAtoms))
    this->addAtoms(platRootAtoms);

  // mark all roots as live, and recursively all atoms they reference
  for (std::set<const Atom *>::iterator it = _deadStripRoots.begin();
       it != _deadStripRoots.end(); ++it) {
    WhyLiveBackChain rootChain;
    rootChain.previous = NULL;
    rootChain.referer = *it;
    this->markLive(**it, &rootChain);
  }

  // now remove all non-live atoms from _atoms
  _atoms.erase(std::remove_if(_atoms.begin(), _atoms.end(),
                              NotLive()), _atoms.end());
}

// error out if some undefines remain
void Resolver::checkUndefines(bool final) {
  // when using LTO, undefines are checked after bitcode is optimized
  if (_haveLLVMObjs && !final)
    return;

  // build vector of remaining undefined symbols
  std::vector<const Atom *> undefinedAtoms;
  _symbolTable.undefines(undefinedAtoms);
  if (_platform.deadCodeStripping()) {
    // when dead code stripping we don't care if dead atoms are undefined
    undefinedAtoms.erase(std::remove_if(
                           undefinedAtoms.begin(), undefinedAtoms.end(),
                           NotLive()), undefinedAtoms.end());
  }

  // let platform make error message about missing symbols
  if (undefinedAtoms.size() != 0)
    _platform.errorWithUndefines(undefinedAtoms, _atoms);
}

// remove from _atoms all coaleseced away atoms
void Resolver::removeCoalescedAwayAtoms() {
  _atoms.erase(std::remove_if(_atoms.begin(), _atoms.end(),
                              AtomCoalescedAway(_symbolTable)), _atoms.end());
}

// check for interactions between symbols defined in this linkage unit
// and same symbol name in linked dynamic shared libraries
void Resolver::checkDylibSymbolCollisions() {
  for (std::vector<const Atom *>::const_iterator it = _atoms.begin();
       it != _atoms.end(); ++it) {
    const Atom *atom = *it;
    if (atom->scope() == Atom::scopeGlobal) {
      if (atom->definition() == Atom::definitionTentative) {
        // See if any shared library also has symbol which
        // collides with the tentative definition.
        // SymbolTable will warn if needed.
        _inputFiles.searchLibraries(atom->name(), true, false, false, *this);
      }
    }
  }
}

// get "main" atom for linkage unit
const Atom *Resolver::entryPoint() {
  llvm::StringRef symbolName = _platform.entryPointName();
  if (symbolName != NULL)
    return _symbolTable.findByName(symbolName);

  return NULL;
}

// give platform a chance to tweak the set of atoms
void Resolver::tweakAtoms() {
  _platform.postResolveTweaks(_atoms);
}

void Resolver::linkTimeOptimize() {
  // FIX ME
}

std::vector<const Atom *> &Resolver::resolve() {
  this->initializeState();
  this->addInitialUndefines();
  this->buildInitialAtomList();
  this->resolveUndefines();
  this->updateReferences();
  this->deadStripOptimize();
  this->checkUndefines(false);
  this->removeCoalescedAwayAtoms();
  this->checkDylibSymbolCollisions();
  this->linkTimeOptimize();
  this->tweakAtoms();
  return _atoms;
}

} // namespace lld
