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
#include "lld/Core/SymbolTable.h"

#include <vector>
#include <set>

namespace lld {

class Atom;
class InputFiles;
class Platform;
class SymbolTable;

/// The Resolver is responsible for merging all input object files
/// and producing a merged graph.
///
/// All platform specific resolving is done by delegating to the
/// Platform object specified.
class Resolver : public File::AtomHandler {
public:
  Resolver(Platform &plat, const InputFiles &inputs)
    : _platform(plat)
    , _inputFiles(inputs)
    , _haveLLVMObjs(false)
    , _addToFinalSection(false)
    , _completedInitialObjectFiles(false) {}

  // AtomHandler methods
  virtual void doAtom(const Atom &);
  virtual void doFile(const File &);

  /// @brief do work of merging and resolving and return list
  std::vector<const Atom *> &resolve();

private:
  struct WhyLiveBackChain {
    WhyLiveBackChain *previous;
    const Atom *referer;
  };

  void initializeState();
  void addInitialUndefines();
  void buildInitialAtomList();
  void resolveUndefines();
  void updateReferences();
  void deadStripOptimize();
  void checkUndefines(bool final);
  void removeCoalescedAwayAtoms();
  void checkDylibSymbolCollisions();
  void linkTimeOptimize();
  void tweakAtoms();

  const Atom *entryPoint();
  void markLive(const Atom &atom, WhyLiveBackChain *previous);
  void addAtoms(const std::vector<const Atom *>&);

  Platform &_platform;
  const InputFiles &_inputFiles;
  SymbolTable _symbolTable;
  std::vector<const Atom *> _atoms;
  std::set<const Atom *> _deadStripRoots;
  std::vector<const Atom *> _atomsWithUnresolvedReferences;
  bool _haveLLVMObjs;
  bool _addToFinalSection;
  bool _completedInitialObjectFiles;
};

} // namespace lld

#endif // LLD_CORE_RESOLVER_H_
