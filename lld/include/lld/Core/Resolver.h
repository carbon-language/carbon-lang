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
#include "lld/Core/SymbolTable.h"

#include "llvm/ADT/DenseSet.h"

#include <set>
#include <vector>

namespace lld {

class Atom;
class InputFiles;
class SymbolTable;

/// 
/// The ResolverOptions class encapsulates options needed during core linking.
/// To use, create a subclass whose constructor sets up the ivars.
///
class ResolverOptions {
public:
  ResolverOptions()
    : _deadCodeStrip(false)
    , _globalsAreDeadStripRoots(false)
    , _searchArchivesToOverrideTentativeDefinitions(false)
    , _searchSharedLibrariesToOverrideTentativeDefinitions(false)
    , _warnSharedLibrariesOverridesTentativeDefinitions(false)
    , _undefinesAreErrors(false)
    , _warnIfCoalesableAtomsHaveDifferentCanBeNull(false)
    , _warnIfCoalesableAtomsHaveDifferentLoadName(false) {
  }
  
  /// Whether the resolver should removed unreferenced atoms.
  bool deadCodeStripping() const {
    return _deadCodeStrip;
  }
  
  /// If dead stripping, whether all global symbols are kept. 
  bool allGlobalsAreDeadStripRoots() const {
    return _globalsAreDeadStripRoots;
  }
  
  /// If dead stripping, names of atoms that must be kept. 
  const std::vector<StringRef>& deadStripRootNames() const {
    return _deadStripRootNames;
  }
  
  /// Whether resolver should look in archives for a definition to 
  /// replace a tentative defintion.
  bool searchArchivesToOverrideTentativeDefinitions() const {
    return _searchArchivesToOverrideTentativeDefinitions;
  }
  
  /// Whether resolver should look in shared libraries for a definition to 
  /// replace a tentative defintion.
  bool searchSharedLibrariesToOverrideTentativeDefinitions() const {
    return _searchSharedLibrariesToOverrideTentativeDefinitions;
  }
  
  /// Whether resolver should look warn if shared library definition replaced
  /// a tentative defintion.
  bool warnSharedLibrariesOverridesTentativeDefinitions() const {
    return _warnSharedLibrariesOverridesTentativeDefinitions;
  }

  /// Whether resolver should error if there are any UndefinedAtoms 
  /// left when resolving is done.
  bool undefinesAreErrors() const {
    return _undefinesAreErrors;
  }
  
  /// Whether resolver should warn if it discovers two UndefinedAtoms 
  /// or two SharedLibraryAtoms with the same name, but different 
  /// canBeNull attributes.
  bool warnIfCoalesableAtomsHaveDifferentCanBeNull() const {
    return _warnIfCoalesableAtomsHaveDifferentCanBeNull;
  }
 
  /// Whether resolver should warn if it discovers two SharedLibraryAtoms
  /// with the same name, but different loadNames.
   bool warnIfCoalesableAtomsHaveDifferentLoadName() const {
    return _warnIfCoalesableAtomsHaveDifferentLoadName;
  }
 
protected:
  bool  _deadCodeStrip;
  bool  _globalsAreDeadStripRoots;
  bool  _searchArchivesToOverrideTentativeDefinitions;
  bool  _searchSharedLibrariesToOverrideTentativeDefinitions;
  bool  _warnSharedLibrariesOverridesTentativeDefinitions;
  bool  _undefinesAreErrors;
  bool  _warnIfCoalesableAtomsHaveDifferentCanBeNull;
  bool  _warnIfCoalesableAtomsHaveDifferentLoadName;
  std::vector<StringRef> _deadStripRootNames;
};



///
/// The Resolver is responsible for merging all input object files
/// and producing a merged graph.
///
/// All variations in resolving are controlled by the 
/// ResolverOptions object specified.
///
class Resolver : public InputFiles::Handler {
public:
  Resolver(ResolverOptions &opts, const InputFiles &inputs)
    : _options(opts)
    , _inputFiles(inputs)
    , _symbolTable(opts)
    , _haveLLVMObjs(false)
    , _addToFinalSection(false)
    , _completedInitialObjectFiles(false) {}

  // InputFiles::Handler methods
  virtual void doDefinedAtom(const class DefinedAtom&);
  virtual void doUndefinedAtom(const class UndefinedAtom&);
  virtual void doSharedLibraryAtom(const class SharedLibraryAtom &);
  virtual void doAbsoluteAtom(const class AbsoluteAtom &);
  virtual void doFile(const File&);

  /// @brief do work of merging and resolving and return list
  void resolve();

  MutableFile& resultFile() {
    return _result;
  }

private:

  void buildInitialAtomList();
  void resolveUndefines();
  void updateReferences();
  void deadStripOptimize();
  void checkUndefines(bool final);
  void removeCoalescedAwayAtoms();
  void checkDylibSymbolCollisions();
  void linkTimeOptimize();
  void tweakAtoms();

  void markLive(const Atom &atom);
  void addAtoms(const std::vector<const DefinedAtom *>&);


  class MergedFile : public MutableFile {
  public:
    MergedFile() : MutableFile("<linker-internal>") { }

  virtual const atom_collection<DefinedAtom>& defined() const {
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

  private:
    friend class Resolver;
    atom_collection_vector<DefinedAtom>         _definedAtoms;
    atom_collection_vector<UndefinedAtom>       _undefinedAtoms;
    atom_collection_vector<SharedLibraryAtom>   _sharedLibraryAtoms;
    atom_collection_vector<AbsoluteAtom>        _absoluteAtoms;
  };


  ResolverOptions              &_options;
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
