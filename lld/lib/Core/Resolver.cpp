//===- Core/Resolver.cpp - Resolves Atom References -----------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/Core/Atom.h"
#include "lld/Core/ArchiveLibraryFile.h"
#include "lld/Core/File.h"
#include "lld/Core/SharedLibraryFile.h"
#include "lld/Core/Instrumentation.h"
#include "lld/Core/LLVM.h"
#include "lld/Core/Resolver.h"
#include "lld/Core/SymbolTable.h"
#include "lld/Core/LinkingContext.h"
#include "lld/Core/UndefinedAtom.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <vector>

namespace lld {

void Resolver::handleFile(const File &file) {
  bool undefAdded = false;
  for (const DefinedAtom *atom : file.defined())
    doDefinedAtom(*atom);
  for (const UndefinedAtom *atom : file.undefined())
    if (doUndefinedAtom(*atom))
      undefAdded = true;
  for (const SharedLibraryAtom *atom : file.sharedLibrary())
    doSharedLibraryAtom(*atom);
  for (const AbsoluteAtom *atom : file.absolute())
    doAbsoluteAtom(*atom);

  // Notify the input file manager of the fact that we have made some progress
  // on linking using the current input file. It may want to know the fact for
  // --start-group/--end-group.
  if (undefAdded) {
    _context.getInputGraph().notifyProgress();
  }
}

void Resolver::forEachUndefines(bool searchForOverrides,
                                UndefCallback callback) {
  // Handle normal archives
  unsigned undefineGenCount = 0;
  do {
    undefineGenCount = _symbolTable.size();
    for (const UndefinedAtom *undefAtom : _symbolTable.undefines()) {
      StringRef undefName = undefAtom->name();
      // load for previous undefine may also have loaded this undefine
      if (!_symbolTable.isDefined(undefName))
        callback(undefName, false);
    }

    // search libraries for overrides of common symbols
    if (searchForOverrides) {
      for (StringRef tentDefName : _symbolTable.tentativeDefinitions()) {
        // Load for previous tentative may also have loaded
        // something that overrode this tentative, so always check.
        const Atom *curAtom = _symbolTable.findByName(tentDefName);
        assert(curAtom != nullptr);
        if (const DefinedAtom *curDefAtom = dyn_cast<DefinedAtom>(curAtom)) {
          if (curDefAtom->merge() == DefinedAtom::mergeAsTentative)
            callback(tentDefName, true);
        }
      }
    }
  } while (undefineGenCount != _symbolTable.size());
}

void Resolver::handleArchiveFile(const File &file) {
  const ArchiveLibraryFile *archiveFile = cast<ArchiveLibraryFile>(&file);
  bool searchForOverrides =
      _context.searchArchivesToOverrideTentativeDefinitions();
  forEachUndefines(searchForOverrides,
                   [&](StringRef undefName, bool dataSymbolOnly) {
    if (const File *member = archiveFile->find(undefName, dataSymbolOnly)) {
      member->setOrdinal(_context.getNextOrdinalAndIncrement());
      handleFile(*member);
    }
  });
}

void Resolver::handleSharedLibrary(const File &file) {
  // Add all the atoms from the shared library
  const SharedLibraryFile *sharedLibrary = cast<SharedLibraryFile>(&file);
  handleFile(*sharedLibrary);
  bool searchForOverrides =
      _context.searchSharedLibrariesToOverrideTentativeDefinitions();
  forEachUndefines(searchForOverrides,
                   [&](StringRef undefName, bool dataSymbolOnly) {
    if (const SharedLibraryAtom *atom =
            sharedLibrary->exports(undefName, dataSymbolOnly))
      doSharedLibraryAtom(*atom);
  });
}

bool Resolver::doUndefinedAtom(const UndefinedAtom &atom) {
  DEBUG_WITH_TYPE("resolver", llvm::dbgs()
                    << "       UndefinedAtom: "
                    << llvm::format("0x%09lX", &atom)
                    << ", name=" << atom.name() << "\n");

  // add to list of known atoms
  _atoms.push_back(&atom);

  // tell symbol table
  bool newUndefAdded = _symbolTable.add(atom);

  // If the undefined symbol has an alternative name, try to resolve the
  // symbol with the name to give it a second chance. This feature is used
  // for COFF "weak external" symbol.
  if (!_symbolTable.isDefined(atom.name())) {
    if (const UndefinedAtom *fallbackAtom = atom.fallback()) {
      doUndefinedAtom(*fallbackAtom);
      _symbolTable.addReplacement(&atom, fallbackAtom);
    }
  }
  return newUndefAdded;
}

/// \brief Add the section group and the group-child reference members.
void Resolver::maybeAddSectionGroupOrGnuLinkOnce(const DefinedAtom &atom) {
  // First time adding a group?
  bool isFirstTime = _symbolTable.addGroup(atom);

  if (!isFirstTime) {
    // If duplicate symbols are allowed, select the first group.
    if (_context.getAllowDuplicates())
      return;
    auto *prevGroup = dyn_cast<DefinedAtom>(_symbolTable.findGroup(atom.name()));
    assert(prevGroup &&
           "Internal Error: The group atom could only be a defined atom");
    // The atoms should be of the same content type, reject invalid group
    // resolution behaviors.
    if (atom.contentType() == prevGroup->contentType())
      return;
    llvm::errs() << "SymbolTable: error while merging " << atom.name()
                 << "\n";
    llvm::report_fatal_error("duplicate symbol error");
    return;
  }

  for (const Reference *r : atom) {
    if (r->kindNamespace() == lld::Reference::KindNamespace::all &&
        r->kindValue() == lld::Reference::kindGroupChild) {
      const DefinedAtom *target = dyn_cast<DefinedAtom>(r->target());
      assert(target && "Internal Error: kindGroupChild references need to "
                       "be associated with Defined Atoms only");
      _atoms.push_back(target);
      _symbolTable.add(*target);
    }
  }
}

// Called on each atom when a file is added. Returns true if a given
// atom is added to the symbol table.
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
  if (atom.sectionPosition() == DefinedAtom::sectionPositionStart ||
      atom.sectionPosition() == DefinedAtom::sectionPositionEnd) {
    assert(atom.size() == 0);
  }

  // add to list of known atoms
  _atoms.push_back(&atom);

  if (atom.isGroupParent()) {
    maybeAddSectionGroupOrGnuLinkOnce(atom);
  } else {
    _symbolTable.add(atom);
  }

  // An atom that should never be dead-stripped is a dead-strip root.
  if (_context.deadStrip() && atom.deadStrip() == DefinedAtom::deadStripNever) {
    _deadStripRoots.insert(&atom);
  }
}

void Resolver::doSharedLibraryAtom(const SharedLibraryAtom &atom) {
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

void Resolver::doAbsoluteAtom(const AbsoluteAtom &atom) {
  DEBUG_WITH_TYPE("resolver", llvm::dbgs()
                    << "       AbsoluteAtom: "
                    << llvm::format("0x%09lX", &atom)
                    << ", name="
                    << atom.name()
                    << "\n");

  // add to list of known atoms
  _atoms.push_back(&atom);

  // tell symbol table
  if (atom.scope() != Atom::scopeTranslationUnit)
    _symbolTable.add(atom);
}

// utility to add a vector of atoms
void Resolver::addAtoms(const std::vector<const DefinedAtom *> &newAtoms) {
  for (const DefinedAtom *newAtom : newAtoms)
    doDefinedAtom(*newAtom);
}

// Keep adding atoms until _context.getNextFile() returns an error. This
// function is where undefined atoms are resolved.
bool Resolver::resolveUndefines() {
  ScopedTask task(getDefaultDomain(), "resolveUndefines");

  for (;;) {
    ErrorOr<File &> file = _context.getInputGraph().getNextFile();
    std::error_code ec = file.getError();
    if (ec == InputGraphError::no_more_files)
      return true;
    if (!file) {
      llvm::errs() << "Error occurred in getNextFile: " << ec.message() << "\n";
      return false;
    }

    switch (file->kind()) {
    case File::kindObject:
      assert(!file->hasOrdinal());
      file->setOrdinal(_context.getNextOrdinalAndIncrement());
      handleFile(*file);
      break;
    case File::kindArchiveLibrary:
      if (!file->hasOrdinal())
        file->setOrdinal(_context.getNextOrdinalAndIncrement());
      handleArchiveFile(*file);
      break;
    case File::kindSharedLibrary:
      if (!file->hasOrdinal())
        file->setOrdinal(_context.getNextOrdinalAndIncrement());
      handleSharedLibrary(*file);
      break;
    }
  }
}

// switch all references to undefined or coalesced away atoms
// to the new defined atom
void Resolver::updateReferences() {
  ScopedTask task(getDefaultDomain(), "updateReferences");
  for (const Atom *atom : _atoms) {
    if (const DefinedAtom *defAtom = dyn_cast<DefinedAtom>(atom)) {
      for (const Reference *ref : *defAtom) {
        // A reference of type kindAssociate should't be updated.
        // Instead, an atom having such reference will be removed
        // if the target atom is coalesced away, so that they will
        // go away as a group.
        if (ref->kindNamespace() == lld::Reference::KindNamespace::all &&
            ref->kindValue() == lld::Reference::kindAssociate) {
          if (_symbolTable.isCoalescedAway(atom))
            _deadAtoms.insert(ref->target());
          continue;
        }
        const Atom *newTarget = _symbolTable.replacement(ref->target());
        const_cast<Reference *>(ref)->setTarget(newTarget);
      }
    }
  }
}

// For dead code stripping, recursively mark atoms "live"
void Resolver::markLive(const Atom *atom) {
  // Mark the atom is live. If it's already marked live, then stop recursion.
  auto exists = _liveAtoms.insert(atom);
  if (!exists.second)
    return;

  // Mark all atoms it references as live
  if (const DefinedAtom *defAtom = dyn_cast<DefinedAtom>(atom)) {
    for (const Reference *ref : *defAtom)
      markLive(ref->target());
    for (const Atom *target : _reverseRef.lookup(defAtom))
      markLive(target);
  }
}

static bool isBackref(const Reference *ref) {
  if (ref->kindNamespace() != lld::Reference::KindNamespace::all)
    return false;
  return (ref->kindValue() == lld::Reference::kindLayoutBefore ||
          ref->kindValue() == lld::Reference::kindGroupChild);
}

// remove all atoms not actually used
void Resolver::deadStripOptimize() {
  ScopedTask task(getDefaultDomain(), "deadStripOptimize");
  // only do this optimization with -dead_strip
  if (!_context.deadStrip())
    return;

  // Some type of references prevent referring atoms to be dead-striped.
  // Make a reverse map of such references before traversing the graph.
  for (const Atom *atom : _atoms)
    if (const DefinedAtom *defAtom = dyn_cast<DefinedAtom>(atom))
      for (const Reference *ref : *defAtom)
        if (isBackref(ref))
          _reverseRef[ref->target()].insert(atom);

  // By default, shared libraries are built with all globals as dead strip roots
  if (_context.globalsAreDeadStripRoots())
    for (const Atom *atom : _atoms)
      if (const DefinedAtom *defAtom = dyn_cast<DefinedAtom>(atom))
        if (defAtom->scope() == DefinedAtom::scopeGlobal)
          _deadStripRoots.insert(defAtom);

  // Or, use list of names that are dead strip roots.
  for (const StringRef &name : _context.deadStripRoots()) {
    const Atom *symAtom = _symbolTable.findByName(name);
    assert(symAtom);
    _deadStripRoots.insert(symAtom);
  }

  // mark all roots as live, and recursively all atoms they reference
  for (const Atom *dsrAtom : _deadStripRoots)
    markLive(dsrAtom);

  // now remove all non-live atoms from _atoms
  _atoms.erase(std::remove_if(_atoms.begin(), _atoms.end(), [&](const Atom *a) {
                 return _liveAtoms.count(a) == 0;
               }),
               _atoms.end());
}

// error out if some undefines remain
bool Resolver::checkUndefines() {
  // build vector of remaining undefined symbols
  std::vector<const UndefinedAtom *> undefinedAtoms = _symbolTable.undefines();
  if (_context.deadStrip()) {
    // When dead code stripping, we don't care if dead atoms are undefined.
    undefinedAtoms.erase(
        std::remove_if(undefinedAtoms.begin(), undefinedAtoms.end(),
                       [&](const Atom *a) { return _liveAtoms.count(a) == 0; }),
        undefinedAtoms.end());
  }

  // error message about missing symbols
  if (!undefinedAtoms.empty()) {
    // FIXME: need diagnostics interface for writing error messages
    bool foundUndefines = false;
    for (const UndefinedAtom *undefAtom : undefinedAtoms) {
      const File &f = undefAtom->file();

      // Skip over a weak symbol.
      if (undefAtom->canBeNull() != UndefinedAtom::canBeNullNever)
        continue;

      // If this is a library and undefined symbols are allowed on the
      // target platform, skip over it.
      if (isa<SharedLibraryFile>(f) && _context.allowShlibUndefines())
        continue;

      // If the undefine is coalesced away, skip over it.
      if (_symbolTable.isCoalescedAway(undefAtom))
        continue;

      // Seems like this symbol is undefined. Warn that.
      foundUndefines = true;
      if (_context.printRemainingUndefines()) {
        llvm::errs() << "Undefined symbol: " << undefAtom->file().path()
                     << ": " << undefAtom->name() << "\n";
      }
    }
    if (foundUndefines) {
      if (_context.printRemainingUndefines())
        llvm::errs() << "symbol(s) not found\n";
      return true;
    }
  }
  return false;
}

// remove from _atoms all coaleseced away atoms
void Resolver::removeCoalescedAwayAtoms() {
  ScopedTask task(getDefaultDomain(), "removeCoalescedAwayAtoms");
  _atoms.erase(std::remove_if(_atoms.begin(), _atoms.end(), [&](const Atom *a) {
                 return _symbolTable.isCoalescedAway(a) || _deadAtoms.count(a);
               }),
               _atoms.end());
}

bool Resolver::resolve() {
  if (!resolveUndefines())
    return false;
  updateReferences();
  deadStripOptimize();
  if (checkUndefines())
    if (!_context.allowRemainingUndefines())
      return false;
  removeCoalescedAwayAtoms();
  _result->addAtoms(_atoms);
  return true;
}

void Resolver::MergedFile::addAtom(const Atom &atom) {
  if (auto *def = dyn_cast<DefinedAtom>(&atom)) {
    _definedAtoms._atoms.push_back(def);
  } else if (auto *undef = dyn_cast<UndefinedAtom>(&atom)) {
    _undefinedAtoms._atoms.push_back(undef);
  } else if (auto *shared = dyn_cast<SharedLibraryAtom>(&atom)) {
    _sharedLibraryAtoms._atoms.push_back(shared);
  } else if (auto *abs = dyn_cast<AbsoluteAtom>(&atom)) {
    _absoluteAtoms._atoms.push_back(abs);
  } else {
    llvm_unreachable("atom has unknown definition kind");
  }
}

MutableFile::DefinedAtomRange Resolver::MergedFile::definedAtoms() {
  return range<std::vector<const DefinedAtom *>::iterator>(
      _definedAtoms._atoms.begin(), _definedAtoms._atoms.end());
}

void Resolver::MergedFile::addAtoms(std::vector<const Atom *> &all) {
  ScopedTask task(getDefaultDomain(), "addAtoms");
  DEBUG_WITH_TYPE("resolver", llvm::dbgs() << "Resolver final atom list:\n");
  for (const Atom *atom : all) {
    DEBUG_WITH_TYPE("resolver", llvm::dbgs()
                    << llvm::format("    0x%09lX", atom)
                    << ", name="
                    << atom->name()
                    << "\n");
    addAtom(*atom);
  }
}

} // namespace lld
