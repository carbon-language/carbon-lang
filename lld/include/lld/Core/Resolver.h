//===- Core/Resolver.h - Resolves Atom References -------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_CORE_RESOLVER_H
#define LLD_CORE_RESOLVER_H

#include "lld/Core/ArchiveLibraryFile.h"
#include "lld/Core/File.h"
#include "lld/Core/SharedLibraryFile.h"
#include "lld/Core/Simple.h"
#include "lld/Core/SymbolTable.h"
#include "llvm/ADT/DenseMap.h"
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
  Resolver(LinkingContext &context)
      : _context(context), _symbolTable(context), _result(new MergedFile()),
        _fileIndex(0) {}

  // InputFiles::Handler methods
  void doDefinedAtom(const DefinedAtom&);
  bool doUndefinedAtom(const UndefinedAtom &);
  void doSharedLibraryAtom(const SharedLibraryAtom &);
  void doAbsoluteAtom(const AbsoluteAtom &);

  // Handle files, this adds atoms from the current file thats
  // being processed by the resolver
  bool handleFile(const File &);

  // Handle an archive library file.
  bool handleArchiveFile(const File &);

  // Handle a shared library file.
  void handleSharedLibrary(const File &);

  /// @brief do work of merging and resolving and return list
  bool resolve();

  std::unique_ptr<MutableFile> resultFile() { return std::move(_result); }

private:
  typedef std::function<void(StringRef, bool)> UndefCallback;

  bool undefinesAdded(int begin, int end);
  File *getFile(int &index, int &groupLevel);

  /// \brief Add section group/.gnu.linkonce if it does not exist previously.
  void maybeAddSectionGroupOrGnuLinkOnce(const DefinedAtom &atom);

  /// \brief The main function that iterates over the files to resolve
  void makePreloadArchiveMap();
  bool resolveUndefines();
  void updateReferences();
  void deadStripOptimize();
  bool checkUndefines();
  void removeCoalescedAwayAtoms();
  void checkDylibSymbolCollisions();
  void forEachUndefines(bool searchForOverrides, UndefCallback callback);

  void markLive(const Atom *atom);
  void addAtoms(const std::vector<const DefinedAtom *>&);
  void maybePreloadArchiveMember(StringRef sym);

  class MergedFile : public SimpleFile {
  public:
    MergedFile() : SimpleFile("<linker-internal>") {}
    void addAtoms(std::vector<const Atom*>& atoms);
  };

  LinkingContext &_context;
  SymbolTable _symbolTable;
  std::vector<const Atom *>     _atoms;
  std::set<const Atom *>        _deadStripRoots;
  llvm::DenseSet<const Atom *>  _liveAtoms;
  llvm::DenseSet<const Atom *>  _deadAtoms;
  std::unique_ptr<MergedFile>   _result;
  llvm::DenseMap<const Atom *, llvm::DenseSet<const Atom *>> _reverseRef;

  // --start-group and --end-group
  std::vector<File *> _files;
  std::map<File *, bool> _newUndefinesAdded;
  size_t _fileIndex;

  // Preloading
  std::map<StringRef, ArchiveLibraryFile *> _archiveMap;
};

} // namespace lld

#endif // LLD_CORE_RESOLVER_H
