//===- Core/Resolver.h - Resolves Atom References -------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_CORE_RESOLVER_H_
#define LLD_CORE_RESOLVER_H_

#include "lld/Core/File.h"
#include "lld/Core/SharedLibraryFile.h"
#include "lld/Core/SymbolTable.h"

#include "llvm/ADT/DenseSet.h"

#include <set>
#include <vector>

namespace lld {

class Atom;
class LinkingContext;

/// \brief The Resolver is responsible for merging all input object files
/// and producing a merged graph.
class Resolver {
public:
  enum ResolverState {
    StateNoChange = 0,              // The default resolver state
    StateNewDefinedAtoms = 1,       // New defined atoms were added
    StateNewUndefinedAtoms = 2,     // New undefined atoms were added
    StateNewSharedLibraryAtoms = 4, // New shared library atoms were added
    StateNewAbsoluteAtoms = 8       // New absolute atoms were added
  };

  Resolver(const LinkingContext &context)
      : _context(context), _symbolTable(context), _result(context),
        _haveLLVMObjs(false), _addToFinalSection(false) {}

  virtual ~Resolver() {}

  // InputFiles::Handler methods
  virtual void doDefinedAtom(const DefinedAtom&);
  virtual void doUndefinedAtom(const UndefinedAtom&);
  virtual void doSharedLibraryAtom(const SharedLibraryAtom &);
  virtual void doAbsoluteAtom(const AbsoluteAtom &);
  virtual void doFile(const File&);

  // Handle files, this adds atoms from the current file thats
  // being processed by the resolver
  virtual void handleFile(const File &);

  // Handle an archive library file.
  virtual void handleArchiveFile(const File &);

  // Handle a shared library file.
  virtual void handleSharedLibrary(const File &);

  /// @brief do work of merging and resolving and return list
  bool resolve();

  MutableFile& resultFile() {
    return _result;
  }

private:

  /// \brief The main function that iterates over the files to resolve
  void resolveUndefines();
  void updateReferences();
  void deadStripOptimize();
  bool checkUndefines(bool final);
  void removeCoalescedAwayAtoms();
  void checkDylibSymbolCollisions();
  void linkTimeOptimize();
  void tweakAtoms();

  void markLive(const Atom &atom);
  void addAtoms(const std::vector<const DefinedAtom *>&);

  class MergedFile : public MutableFile {
  public:
    MergedFile(const LinkingContext &context)
        : MutableFile(context, "<linker-internal>") {}

    virtual const atom_collection<DefinedAtom> &defined() const {
      return _definedAtoms;
    }
    virtual const atom_collection<UndefinedAtom>& undefined() const {
      return _undefinedAtoms;
    }
    virtual const atom_collection<SharedLibraryAtom>& sharedLibrary() const {
      return _sharedLibraryAtoms;
    }
    virtual const atom_collection<AbsoluteAtom>& absolute() const {
      return _absoluteAtoms;
    }

    void addAtoms(std::vector<const Atom*>& atoms);

    virtual void addAtom(const Atom& atom);
    virtual DefinedAtomRange definedAtoms();

  private:
    atom_collection_vector<DefinedAtom>         _definedAtoms;
    atom_collection_vector<UndefinedAtom>       _undefinedAtoms;
    atom_collection_vector<SharedLibraryAtom>   _sharedLibraryAtoms;
    atom_collection_vector<AbsoluteAtom>        _absoluteAtoms;
  };

  const LinkingContext &_context;
  SymbolTable _symbolTable;
  std::vector<const Atom *>     _atoms;
  std::set<const Atom *>        _deadStripRoots;
  std::vector<const Atom *>     _atomsWithUnresolvedReferences;
  llvm::DenseSet<const Atom *>  _liveAtoms;
  MergedFile                    _result;
  bool                          _haveLLVMObjs;
  bool _addToFinalSection;
};

} // namespace lld

#endif // LLD_CORE_RESOLVER_H_
