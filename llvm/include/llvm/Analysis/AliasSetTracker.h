//===- llvm/Analysis/AliasSetTracker.h - Build Alias Sets -------*- C++ -*-===//
//
// This file defines two classes: AliasSetTracker and AliasSet.  These interface
// are used to classify a collection of pointer references into a maximal number
// of disjoint sets.  Each AliasSet object constructed by the AliasSetTracker
// object refers to memory disjoint from the other sets.
// 
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_ALIASSETTRACKER_H
#define LLVM_ANALYSIS_ALIASSETTRACKER_H

#include <vector>
class AliasAnalysis;
class LoadInst;
class StoreInst;
class CallInst;
class InvokeInst;
class Value;
class AliasSetTracker;

class AliasSet {
  friend class AliasSetTracker;
  std::vector<LoadInst*> Loads;
  std::vector<StoreInst*> Stores;
  std::vector<CallInst*> Calls;
  std::vector<InvokeInst*> Invokes;
public:
  /// AccessType - Keep track of whether this alias set merely refers to the
  /// locations of memory, whether it modifies the memory, or whether it does
  /// both.  The lattice goes from "None" (alias set not present) to either Refs
  /// or Mods, then to ModRef as neccesary.
  ///
  enum AccessType {
    Refs, Mods, ModRef
  };

  /// AliasType - Keep track the relationships between the pointers in the set.
  /// Lattice goes from MustAlias to MayAlias.
  ///
  enum AliasType {
    MustAlias, MayAlias
  };
private:
  enum AccessType AccessTy;
  enum AliasType  AliasTy;
public:
  /// Accessors...
  enum AccessType getAccessType() const { return AccessTy; }
  enum AliasType  getAliasType()  const { return AliasTy; }

  // TODO: in the future, add a fixed size (4? 2?) cache of pointers that we
  // know are in the alias set, to cut down time answering "pointeraliasesset"
  // queries.

  /// pointerAliasesSet - Return true if the specified pointer "may" (or must)
  /// alias one of the members in the set.
  ///
  bool pointerAliasesSet(const Value *Ptr, AliasAnalysis &AA) const;

  /// mergeSetIn - Merge the specified alias set into this alias set...
  ///
  void mergeSetIn(const AliasSet &AS);

  const std::vector<LoadInst*>   &getLoads()   const { return Loads; }
  const std::vector<StoreInst*>  &getStores()  const { return Stores; }
  const std::vector<CallInst*>   &getCalls()   const { return Calls; }
  const std::vector<InvokeInst*> &getInvokes() const { return Invokes; }

private:
  AliasSet() : AliasTy(MustAlias) {} // Can only be created by AliasSetTracker
  void updateAccessType();
  Value *getSomePointer() const;
};


class AliasSetTracker {
  AliasAnalysis &AA;
  std::vector<AliasSet> AliasSets;
public:
  /// AliasSetTracker ctor - Create an empty collection of AliasSets, and use
  /// the specified alias analysis object to disambiguate load and store
  /// addresses.
  AliasSetTracker(AliasAnalysis &aa) : AA(aa) {}


  /// add methods - These methods are used to add different types of
  /// instructions to the alias sets.  Adding a new instruction can result in
  /// one of three actions happening:
  ///
  ///   1. If the instruction doesn't alias any other sets, create a new set.
  ///   2. If the instruction aliases exactly one set, add it to the set
  ///   3. If the instruction aliases multiple sets, merge the sets, and add
  ///      the instruction to the result.
  ///
  void add(LoadInst *LI);
  void add(StoreInst *SI);
  void add(CallInst *CI);
  void add(InvokeInst *II);

  /// getAliasSets - Return the alias sets that are active.
  const std::vector<AliasSet> &getAliasSets() const { return AliasSets; }

private:
  AliasSet *findAliasSetForPointer(const Value *Ptr);
  void mergeAllSets();
};

#endif
