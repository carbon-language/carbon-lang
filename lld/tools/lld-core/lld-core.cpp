//===- tools/lld/lld-core.cpp - Linker Core Test Driver -----------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/Core/InputFiles.h"
#include "lld/Core/Atom.h"
#include "lld/Core/DefinedAtom.h"
#include "lld/Core/UndefinedAtom.h"
#include "lld/Core/Resolver.h"
#include "lld/Core/YamlReader.h"
#include "lld/Core/YamlWriter.h"
#include "lld/Platform/Platform.h"

#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/system_error.h"
#include "llvm/Support/ErrorHandling.h"

#include <vector>

using namespace lld;

static void error(llvm::Twine message) {
  llvm::errs() << "lld-core: " << message << ".\n";
}

static bool error(llvm::error_code ec) {
  if (ec) {
    error(ec.message());
    return true;
  }
  return false;
}

namespace {
class LdCore : public InputFiles, public Platform {
public:
  LdCore(std::vector<File *> &f) : _files(f) { }

  // InputFiles interface
  virtual void forEachInitialAtom(File::AtomHandler &) const;
  virtual bool searchLibraries(llvm::StringRef name, bool searchDylibs,
                               bool searchArchives, bool dataSymbolOnly,
                               File::AtomHandler &) const;

  virtual void initialize() { }

  // tell platform object another file has been added
  virtual void fileAdded(const File &file) { }

  // tell platform object another atom has been added
  virtual void atomAdded(const Atom &file) { }

  // give platform a chance to change each atom's scope
  virtual void adjustScope(const DefinedAtom &atom) { }

  // if specified atom needs alternate names, return AliasAtom(s)
  virtual bool getAliasAtoms(const Atom &atom,
                             std::vector<const DefinedAtom *>&) {
    return false;
  }

  // give platform a chance to resolve platform-specific undefs
  virtual bool getPlatformAtoms(llvm::StringRef undefined,
                                std::vector<const DefinedAtom *>&) {
    return false;
  }

  // resolver should remove unreferenced atoms
  virtual bool deadCodeStripping() {
    return false;
  }

  // atom must be kept so should be root of dead-strip graph
  virtual bool isDeadStripRoot(const Atom &atom) {
    return false;
  }

  // if target must have some atoms, denote here
  virtual bool getImplicitDeadStripRoots(std::vector<const DefinedAtom *>&) {
    return false;
  }

  // return entry point for output file (e.g. "main") or NULL
  virtual llvm::StringRef entryPointName() {
    return NULL;
  }

  // for iterating must-be-defined symbols ("main" or -u command line option)
  typedef llvm::StringRef const *UndefinesIterator;
  virtual UndefinesIterator initialUndefinesBegin() const {
    return NULL;
  }
  virtual UndefinesIterator initialUndefinesEnd() const {
    return NULL;
  }

  // if platform wants resolvers to search libraries for overrides
  virtual bool searchArchivesToOverrideTentativeDefinitions() {
    return false;
  }

  virtual bool searchSharedLibrariesToOverrideTentativeDefinitions() {
    return false;
  }

  // if platform allows symbol to remain undefined (e.g. -r)
  virtual bool allowUndefinedSymbol(llvm::StringRef name) {
    return true;
  }

  // for debugging dead code stripping, -why_live
  virtual bool printWhyLive(llvm::StringRef name) {
    return false;
  }

  virtual const Atom& handleMultipleDefinitions(const Atom& def1, 
                                                const Atom& def2) {
    llvm::report_fatal_error("symbol '" 
                            + llvm::Twine(def1.name()) 
                            + "' multiply defined");
  }

  // print out undefined symbol error messages in platform specific way
  virtual void errorWithUndefines(const std::vector<const Atom *> &undefs,
                                  const std::vector<const Atom *> &all) {}

  // last chance for platform to tweak atoms
  virtual void postResolveTweaks(std::vector<const Atom *> &all) {}

private:
  std::vector<File *> &_files;
};
}

void LdCore::forEachInitialAtom(File::AtomHandler &handler) const {
  for (std::vector<File *>::iterator it = _files.begin();
       it != _files.end(); ++it) {
    const File *file = *it;
    handler.doFile(*file);
    file->forEachAtom(handler);
  }
}

bool LdCore::searchLibraries(llvm::StringRef name, bool searchDylibs,
                             bool searchArchives, bool dataSymbolOnly,
                             File::AtomHandler &) const {
  return false;
}

namespace {
class MergedFile : public File {
public:
  MergedFile(std::vector<const Atom *> &a)
    : File("path"), _atoms(a) { }

  virtual bool forEachAtom(File::AtomHandler &handler) const {
    handler.doFile(*this);
    for (std::vector<const Atom *>::iterator it = _atoms.begin();
         it != _atoms.end(); ++it) {
      const Atom* atom = *it;
      switch ( atom->definition() ) {
        case Atom::definitionRegular:
          handler.doDefinedAtom(*(DefinedAtom*)atom);
          break;
        case Atom::definitionUndefined:
          handler.doUndefinedAtom(*(UndefinedAtom*)atom);
          break;
        default:
          // TO DO
          break;
      }
    }
    return true;
  }

  virtual bool justInTimeforEachAtom(llvm::StringRef name,
                                     File::AtomHandler &) const {
    return false;
  }

private:
  std::vector<const Atom *> &_atoms;
};
}

int main(int argc, const char *argv[]) {
  // read input YAML doc into object file(s)
  std::vector<File *> files;
  if (error(yaml::parseObjectTextFileOrSTDIN(llvm::StringRef(argv[1]), files)))
    return 1;

  // merge all atom graphs
  LdCore core(files);
  Resolver resolver(core, core);
  std::vector<const Atom *> &mergedAtoms = resolver.resolve();
  MergedFile outFile(mergedAtoms);

  // write new atom graph out as YAML doc
  std::string errorInfo;
  llvm::raw_fd_ostream out("-", errorInfo);
  yaml::writeObjectText(outFile, out);
  return 0;
}
