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

namespace {

/// This is used as a filter function to std::remove_if to dead strip atoms.
class NotLive {
public:
  explicit NotLive(const llvm::DenseSet<const Atom *> &la) : _liveAtoms(la) {}

  bool operator()(const Atom *atom) const {
    // don't remove if live
    if (_liveAtoms.count(atom))
      return false;
    // don't remove if marked never-dead-strip
    if (const DefinedAtom *defAtom = dyn_cast<DefinedAtom>(atom))
      if (defAtom->deadStrip() == DefinedAtom::deadStripNever)
        return false;
    // do remove this atom
    return true;
  }

private:
  const llvm::DenseSet<const Atom *> &_liveAtoms;
};

/// This is used as a filter function to std::remove_if to coalesced atoms.
class AtomCoalescedAway {
public:
  explicit AtomCoalescedAway(SymbolTable &sym) : _symbolTable(sym) {}

  bool operator()(const Atom *atom) const {
    const Atom *rep = _symbolTable.replacement(atom);
    return rep != atom;
  }

private:
  SymbolTable &_symbolTable;
};

} // namespace

// called before the first atom in any file is added with doAtom()
void Resolver::doFile(const File &file) {}

void Resolver::handleFile(const File &file) {
  uint32_t resolverState = Resolver::StateNoChange;
  const SharedLibraryFile *sharedLibraryFile =
      llvm::dyn_cast<SharedLibraryFile>(&file);

  doFile(file);
  for (const DefinedAtom *atom : file.defined()) {
    doDefinedAtom(*atom);
    resolverState |= StateNewDefinedAtoms;
  }
  if (!sharedLibraryFile ||
      _context.addUndefinedAtomsFromSharedLibrary(sharedLibraryFile)) {
    for (const UndefinedAtom *undefAtom : file.undefined()) {
      doUndefinedAtom(*undefAtom);
      resolverState |= StateNewUndefinedAtoms;
      // If the undefined symbol has an alternative name, try to resolve the
      // symbol with the name to give it a second chance. This feature is used
      // for COFF "weak external" symbol.
      if (!_symbolTable.isDefined(undefAtom->name())) {
        if (const UndefinedAtom *fallbackAtom = undefAtom->fallback()) {
          doUndefinedAtom(*fallbackAtom);
          _symbolTable.addReplacement(undefAtom, fallbackAtom);
        }
      }
    }
  }
  for (const SharedLibraryAtom *shlibAtom : file.sharedLibrary()) {
    doSharedLibraryAtom(*shlibAtom);
    resolverState |= StateNewSharedLibraryAtoms;
  }
  for (const AbsoluteAtom *absAtom : file.absolute()) {
    doAbsoluteAtom(*absAtom);
    resolverState |= StateNewAbsoluteAtoms;
  }
  _context.setResolverState(resolverState);
}

void Resolver::forEachUndefines(UndefCallback callback,
                                bool searchForOverrides) {
  // Handle normal archives
  int64_t undefineGenCount = 0;
  do {
    undefineGenCount = _symbolTable.size();
    std::vector<const UndefinedAtom *> undefines;
    _symbolTable.undefines(undefines);
    for (const UndefinedAtom *undefAtom : undefines) {
      StringRef undefName = undefAtom->name();
      // load for previous undefine may also have loaded this undefine
      if (!_symbolTable.isDefined(undefName))
        callback(undefName, false);
    }
    // search libraries for overrides of common symbols
    if (searchForOverrides) {
      std::vector<StringRef> tentDefNames;
      _symbolTable.tentativeDefinitions(tentDefNames);
      for (StringRef tentDefName : tentDefNames) {
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
  const ArchiveLibraryFile *archiveFile = dyn_cast<ArchiveLibraryFile>(&file);
  auto callback = [&](StringRef undefName, bool dataSymbolOnly) {
    if (const File *member = archiveFile->find(undefName, dataSymbolOnly)) {
      member->setOrdinal(_context.getNextOrdinalAndIncrement());
      handleFile(*member);
    }
  };
  bool searchForOverrides =
      _context.searchArchivesToOverrideTentativeDefinitions();
  forEachUndefines(callback, searchForOverrides);
}

void Resolver::handleSharedLibrary(const File &file) {
  // Add all the atoms from the shared library
  const SharedLibraryFile *sharedLibrary = dyn_cast<SharedLibraryFile>(&file);
  handleFile(*sharedLibrary);

  auto callback = [&](StringRef undefName, bool dataSymbolOnly) {
    if (const SharedLibraryAtom *shAtom =
            sharedLibrary->exports(undefName, dataSymbolOnly))
      doSharedLibraryAtom(*shAtom);
  };
  bool searchForOverrides =
      _context.searchSharedLibrariesToOverrideTentativeDefinitions();
  forEachUndefines(callback, searchForOverrides);
}

void Resolver::doUndefinedAtom(const UndefinedAtom &atom) {
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
  switch (atom.sectionPosition()) {
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

  if (_context.deadStrip()) {
    // add to set of dead-strip-roots, all symbols that
    // the compiler marks as don't strip
    if (atom.deadStrip() == DefinedAtom::deadStripNever)
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
    this->doDefinedAtom(*newAtom);
}

// Keep adding atoms until _context.nextFile() returns an error. This function
// is where undefined atoms are resolved.
bool Resolver::resolveUndefines() {
  ScopedTask task(getDefaultDomain(), "resolveUndefines");

  for (;;) {
    ErrorOr<File &> file = _context.nextFile();
    _context.setResolverState(Resolver::StateNoChange);
    if (error_code(file) == InputGraphError::no_more_files)
      return true;
    if (!file) {
      llvm::errs() << "Error occurred in nextFile: "
                   << error_code(file).message() << "\n";
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
        const Atom *newTarget = _symbolTable.replacement(ref->target());
        const_cast<Reference *>(ref)->setTarget(newTarget);
      }
    }
  }
}

// For dead code stripping, recursively mark atoms "live"
void Resolver::markLive(const Atom &atom) {
  // Mark the atom is live. If it's already marked live, then stop recursion.
  auto exists = _liveAtoms.insert(&atom);
  if (!exists.second)
    return;

  // Mark all atoms it references as live
  if (const DefinedAtom *defAtom = dyn_cast<DefinedAtom>(&atom))
    for (const Reference *ref : *defAtom)
      if (const Atom *target = ref->target())
        this->markLive(*target);
}

// remove all atoms not actually used
void Resolver::deadStripOptimize() {
  ScopedTask task(getDefaultDomain(), "deadStripOptimize");
  // only do this optimization with -dead_strip
  if (!_context.deadStrip())
    return;
  assert(_liveAtoms.empty());

  // By default, shared libraries are built with all globals as dead strip roots
  if (_context.globalsAreDeadStripRoots()) {
    for (const Atom *atom : _atoms) {
      const DefinedAtom *defAtom = dyn_cast<DefinedAtom>(atom);
      if (defAtom == nullptr)
        continue;
      if (defAtom->scope() == DefinedAtom::scopeGlobal)
        _deadStripRoots.insert(defAtom);
    }
  }

  // Or, use list of names that are dead strip roots.
  for (const StringRef &name : _context.deadStripRoots()) {
    const Atom *symAtom = _symbolTable.findByName(name);
    assert(symAtom);
    _deadStripRoots.insert(symAtom);
  }

  // mark all roots as live, and recursively all atoms they reference
  for (const Atom *dsrAtom : _deadStripRoots)
    this->markLive(*dsrAtom);

  // now remove all non-live atoms from _atoms
  _atoms.erase(
      std::remove_if(_atoms.begin(), _atoms.end(), NotLive(_liveAtoms)),
      _atoms.end());
}

// error out if some undefines remain
bool Resolver::checkUndefines(bool final) {
  // when using LTO, undefines are checked after bitcode is optimized
  if (_haveLLVMObjs && !final)
    return false;

  // build vector of remaining undefined symbols
  std::vector<const UndefinedAtom *> undefinedAtoms;
  _symbolTable.undefines(undefinedAtoms);
  if (_context.deadStrip()) {
    // When dead code stripping, we don't care if dead atoms are undefined.
    undefinedAtoms.erase(std::remove_if(undefinedAtoms.begin(),
                                        undefinedAtoms.end(),
                                        NotLive(_liveAtoms)),
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
      if (_symbolTable.replacement(undefAtom) != undefAtom)
        continue;

      // Seems like this symbol is undefined. Warn that.
      foundUndefines = true;
      if (_context.printRemainingUndefines()) {
        llvm::errs() << "Undefined Symbol: " << undefAtom->file().path()
                     << " : " << undefAtom->name() << "\n";
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
  _atoms.erase(std::remove_if(_atoms.begin(), _atoms.end(),
                              AtomCoalescedAway(_symbolTable)),
               _atoms.end());
}

void Resolver::linkTimeOptimize() {
  // FIX ME
}

bool Resolver::resolve() {
  if (!this->resolveUndefines())
    return false;
  this->updateReferences();
  this->deadStripOptimize();
  if (this->checkUndefines(false))
    if (!_context.allowRemainingUndefines())
      return false;
  this->removeCoalescedAwayAtoms();
  this->linkTimeOptimize();
  this->_result->addAtoms(_atoms);
  return true;
}

void Resolver::MergedFile::addAtom(const Atom &atom) {
  if (const DefinedAtom *defAtom = dyn_cast<DefinedAtom>(&atom)) {
    _definedAtoms._atoms.push_back(defAtom);
  } else if (const UndefinedAtom *undefAtom = dyn_cast<UndefinedAtom>(&atom)) {
    _undefinedAtoms._atoms.push_back(undefAtom);
  } else if (const SharedLibraryAtom *slAtom =
                 dyn_cast<SharedLibraryAtom>(&atom)) {
    _sharedLibraryAtoms._atoms.push_back(slAtom);
  } else if (const AbsoluteAtom *abAtom = dyn_cast<AbsoluteAtom>(&atom)) {
    _absoluteAtoms._atoms.push_back(abAtom);
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
    this->addAtom(*atom);
  }
}

} // namespace lld
