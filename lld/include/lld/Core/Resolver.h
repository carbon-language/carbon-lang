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
#include "lld/Core/InputFiles.h"
#include "lld/Core/SharedLibraryFile.h"
#include "lld/Core/SymbolTable.h"

#include "llvm/ADT/DenseSet.h"

#include <set>
#include <vector>

namespace lld {

class Atom;
class TargetInfo;

/// \brief The Resolver is responsible for merging all input object files
/// and producing a merged graph.
class Resolver : public InputFiles::Handler {
public:
  Resolver(const TargetInfo &ti, const InputFiles &inputs)
      : _targetInfo(ti), _inputFiles(inputs), _symbolTable(ti), _result(ti),
        _haveLLVMObjs(false), _addToFinalSection(false),
        _completedInitialObjectFiles(false) {
  }

  // InputFiles::Handler methods
  virtual void doDefinedAtom(const DefinedAtom&);
  virtual void doUndefinedAtom(const UndefinedAtom&);
  virtual void doSharedLibraryAtom(const SharedLibraryAtom &);
  virtual void doAbsoluteAtom(const AbsoluteAtom &);
  virtual void doFile(const File&);

  /// @brief do work of merging and resolving and return list
  bool resolve();

  MutableFile& resultFile() {
    return _result;
  }

private:

  void buildInitialAtomList();
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
    MergedFile(const TargetInfo &ti) : MutableFile(ti, "<linker-internal>") {}

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
    friend Resolver;
    atom_collection_vector<DefinedAtom>         _definedAtoms;
    atom_collection_vector<UndefinedAtom>       _undefinedAtoms;
    atom_collection_vector<SharedLibraryAtom>   _sharedLibraryAtoms;
    atom_collection_vector<AbsoluteAtom>        _absoluteAtoms;
  };


  const TargetInfo             &_targetInfo;
  const InputFiles             &_inputFiles;
  SymbolTable                   _symbolTable;
  std::vector<const Atom *>     _atoms;
  std::set<const Atom *>        _deadStripRoots;
  std::vector<const Atom *>     _atomsWithUnresolvedReferences;
  llvm::DenseSet<const Atom *>  _liveAtoms;
  MergedFile                    _result;
  bool                          _haveLLVMObjs;
  bool                          _addToFinalSection;
  bool                          _completedInitialObjectFiles;
};

} // namespace lld

#endif // LLD_CORE_RESOLVER_H_
