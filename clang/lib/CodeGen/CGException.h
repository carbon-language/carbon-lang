//===-- CGException.h - Classes for exceptions IR generation ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// These classes support the generation of LLVM IR for exceptions in
// C++ and Objective C.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CODEGEN_CGEXCEPTION_H
#define CLANG_CODEGEN_CGEXCEPTION_H

/// EHScopeStack is defined in CodeGenFunction.h, but its
/// implementation is in this file and in CGException.cpp.
#include "CodeGenFunction.h"

namespace llvm {
  class Value;
  class BasicBlock;
}

namespace clang {
namespace CodeGen {

/// The exceptions personality for a function.  When 
class EHPersonality {
  const char *PersonalityFn;

  // If this is non-null, this personality requires a non-standard
  // function for rethrowing an exception after a catchall cleanup.
  // This function must have prototype void(void*).
  const char *CatchallRethrowFn;

  EHPersonality(const char *PersonalityFn,
                const char *CatchallRethrowFn = 0)
    : PersonalityFn(PersonalityFn),
      CatchallRethrowFn(CatchallRethrowFn) {}

public:
  static const EHPersonality &get(const LangOptions &Lang);
  static const EHPersonality GNU_C;
  static const EHPersonality GNU_ObjC;
  static const EHPersonality NeXT_ObjC;
  static const EHPersonality GNU_CPlusPlus;
  static const EHPersonality GNU_CPlusPlus_SJLJ;

  const char *getPersonalityFnName() const { return PersonalityFn; }
  const char *getCatchallRethrowFnName() const { return CatchallRethrowFn; }
};

/// A protected scope for zero-cost EH handling.
class EHScope {
  llvm::BasicBlock *CachedLandingPad;

  unsigned K : 2;

protected:
  enum { BitsRemaining = 30 };

public:
  enum Kind { Cleanup, Catch, Terminate, Filter };

  EHScope(Kind K) : CachedLandingPad(0), K(K) {}

  Kind getKind() const { return static_cast<Kind>(K); }

  llvm::BasicBlock *getCachedLandingPad() const {
    return CachedLandingPad;
  }

  void setCachedLandingPad(llvm::BasicBlock *Block) {
    CachedLandingPad = Block;
  }
};

/// A scope which attempts to handle some, possibly all, types of
/// exceptions.
///
/// Objective C @finally blocks are represented using a cleanup scope
/// after the catch scope.
class EHCatchScope : public EHScope {
  unsigned NumHandlers : BitsRemaining;

  // In effect, we have a flexible array member
  //   Handler Handlers[0];
  // But that's only standard in C99, not C++, so we have to do
  // annoying pointer arithmetic instead.

public:
  struct Handler {
    /// A type info value, or null (C++ null, not an LLVM null pointer)
    /// for a catch-all.
    llvm::Value *Type;

    /// The catch handler for this type.
    llvm::BasicBlock *Block;

    /// The unwind destination index for this handler.
    unsigned Index;
  };

private:
  friend class EHScopeStack;

  Handler *getHandlers() {
    return reinterpret_cast<Handler*>(this+1);
  }

  const Handler *getHandlers() const {
    return reinterpret_cast<const Handler*>(this+1);
  }

public:
  static size_t getSizeForNumHandlers(unsigned N) {
    return sizeof(EHCatchScope) + N * sizeof(Handler);
  }

  EHCatchScope(unsigned NumHandlers)
    : EHScope(Catch), NumHandlers(NumHandlers) {
  }

  unsigned getNumHandlers() const {
    return NumHandlers;
  }

  void setCatchAllHandler(unsigned I, llvm::BasicBlock *Block) {
    setHandler(I, /*catchall*/ 0, Block);
  }

  void setHandler(unsigned I, llvm::Value *Type, llvm::BasicBlock *Block) {
    assert(I < getNumHandlers());
    getHandlers()[I].Type = Type;
    getHandlers()[I].Block = Block;
  }

  const Handler &getHandler(unsigned I) const {
    assert(I < getNumHandlers());
    return getHandlers()[I];
  }

  typedef const Handler *iterator;
  iterator begin() const { return getHandlers(); }
  iterator end() const { return getHandlers() + getNumHandlers(); }

  static bool classof(const EHScope *Scope) {
    return Scope->getKind() == Catch;
  }
};

/// A cleanup scope which generates the cleanup blocks lazily.
class EHCleanupScope : public EHScope {
  /// Whether this cleanup needs to be run along normal edges.
  bool IsNormalCleanup : 1;

  /// Whether this cleanup needs to be run along exception edges.
  bool IsEHCleanup : 1;

  /// Whether this cleanup was activated before all normal uses.
  bool ActivatedBeforeNormalUse : 1;

  /// Whether this cleanup was activated before all EH uses.
  bool ActivatedBeforeEHUse : 1;

  /// The amount of extra storage needed by the Cleanup.
  /// Always a multiple of the scope-stack alignment.
  unsigned CleanupSize : 12;

  /// The number of fixups required by enclosing scopes (not including
  /// this one).  If this is the top cleanup scope, all the fixups
  /// from this index onwards belong to this scope.
  unsigned FixupDepth : BitsRemaining - 16;

  /// The nearest normal cleanup scope enclosing this one.
  EHScopeStack::stable_iterator EnclosingNormal;

  /// The nearest EH cleanup scope enclosing this one.
  EHScopeStack::stable_iterator EnclosingEH;

  /// The dual entry/exit block along the normal edge.  This is lazily
  /// created if needed before the cleanup is popped.
  llvm::BasicBlock *NormalBlock;

  /// The dual entry/exit block along the EH edge.  This is lazily
  /// created if needed before the cleanup is popped.
  llvm::BasicBlock *EHBlock;

  /// An optional i1 variable indicating whether this cleanup has been
  /// activated yet.  This has one of three states:
  ///   - it is null if the cleanup is inactive
  ///   - it is activeSentinel() if the cleanup is active and was not
  ///     required before activation
  ///   - it points to a valid variable
  llvm::AllocaInst *ActiveVar;

  /// Extra information required for cleanups that have resolved
  /// branches through them.  This has to be allocated on the side
  /// because everything on the cleanup stack has be trivially
  /// movable.
  struct ExtInfo {
    /// The destinations of normal branch-afters and branch-throughs.
    llvm::SmallPtrSet<llvm::BasicBlock*, 4> Branches;

    /// Normal branch-afters.
    llvm::SmallVector<std::pair<llvm::BasicBlock*,llvm::ConstantInt*>, 4>
      BranchAfters;

    /// The destinations of EH branch-afters and branch-throughs.
    /// TODO: optimize for the extremely common case of a single
    /// branch-through.
    llvm::SmallPtrSet<llvm::BasicBlock*, 4> EHBranches;

    /// EH branch-afters.
    llvm::SmallVector<std::pair<llvm::BasicBlock*,llvm::ConstantInt*>, 4>
    EHBranchAfters;
  };
  mutable struct ExtInfo *ExtInfo;

  struct ExtInfo &getExtInfo() {
    if (!ExtInfo) ExtInfo = new struct ExtInfo();
    return *ExtInfo;
  }

  const struct ExtInfo &getExtInfo() const {
    if (!ExtInfo) ExtInfo = new struct ExtInfo();
    return *ExtInfo;
  }

public:
  /// Gets the size required for a lazy cleanup scope with the given
  /// cleanup-data requirements.
  static size_t getSizeForCleanupSize(size_t Size) {
    return sizeof(EHCleanupScope) + Size;
  }

  size_t getAllocatedSize() const {
    return sizeof(EHCleanupScope) + CleanupSize;
  }

  EHCleanupScope(bool IsNormal, bool IsEH, bool IsActive,
                 unsigned CleanupSize, unsigned FixupDepth,
                 EHScopeStack::stable_iterator EnclosingNormal,
                 EHScopeStack::stable_iterator EnclosingEH)
    : EHScope(EHScope::Cleanup),
      IsNormalCleanup(IsNormal), IsEHCleanup(IsEH),
      ActivatedBeforeNormalUse(IsActive),
      ActivatedBeforeEHUse(IsActive),
      CleanupSize(CleanupSize), FixupDepth(FixupDepth),
      EnclosingNormal(EnclosingNormal), EnclosingEH(EnclosingEH),
      NormalBlock(0), EHBlock(0),
      ActiveVar(IsActive ? activeSentinel() : 0),
      ExtInfo(0)
  {
    assert(this->CleanupSize == CleanupSize && "cleanup size overflow");
  }

  ~EHCleanupScope() {
    delete ExtInfo;
  }

  bool isNormalCleanup() const { return IsNormalCleanup; }
  llvm::BasicBlock *getNormalBlock() const { return NormalBlock; }
  void setNormalBlock(llvm::BasicBlock *BB) { NormalBlock = BB; }

  bool isEHCleanup() const { return IsEHCleanup; }
  llvm::BasicBlock *getEHBlock() const { return EHBlock; }
  void setEHBlock(llvm::BasicBlock *BB) { EHBlock = BB; }

  static llvm::AllocaInst *activeSentinel() {
    return reinterpret_cast<llvm::AllocaInst*>(1);
  }

  bool isActive() const { return ActiveVar != 0; }
  llvm::AllocaInst *getActiveVar() const { return ActiveVar; }
  void setActiveVar(llvm::AllocaInst *Var) { ActiveVar = Var; }

  bool wasActivatedBeforeNormalUse() const { return ActivatedBeforeNormalUse; }
  void setActivatedBeforeNormalUse(bool B) { ActivatedBeforeNormalUse = B; }

  bool wasActivatedBeforeEHUse() const { return ActivatedBeforeEHUse; }
  void setActivatedBeforeEHUse(bool B) { ActivatedBeforeEHUse = B; }

  unsigned getFixupDepth() const { return FixupDepth; }
  EHScopeStack::stable_iterator getEnclosingNormalCleanup() const {
    return EnclosingNormal;
  }
  EHScopeStack::stable_iterator getEnclosingEHCleanup() const {
    return EnclosingEH;
  }

  size_t getCleanupSize() const { return CleanupSize; }
  void *getCleanupBuffer() { return this + 1; }

  EHScopeStack::Cleanup *getCleanup() {
    return reinterpret_cast<EHScopeStack::Cleanup*>(getCleanupBuffer());
  }

  /// True if this cleanup scope has any branch-afters or branch-throughs.
  bool hasBranches() const { return ExtInfo && !ExtInfo->Branches.empty(); }

  /// Add a branch-after to this cleanup scope.  A branch-after is a
  /// branch from a point protected by this (normal) cleanup to a
  /// point in the normal cleanup scope immediately containing it.
  /// For example,
  ///   for (;;) { A a; break; }
  /// contains a branch-after.
  ///
  /// Branch-afters each have their own destination out of the
  /// cleanup, guaranteed distinct from anything else threaded through
  /// it.  Therefore branch-afters usually force a switch after the
  /// cleanup.
  void addBranchAfter(llvm::ConstantInt *Index,
                      llvm::BasicBlock *Block) {
    struct ExtInfo &ExtInfo = getExtInfo();
    if (ExtInfo.Branches.insert(Block))
      ExtInfo.BranchAfters.push_back(std::make_pair(Block, Index));
  }

  /// Return the number of unique branch-afters on this scope.
  unsigned getNumBranchAfters() const {
    return ExtInfo ? ExtInfo->BranchAfters.size() : 0;
  }

  llvm::BasicBlock *getBranchAfterBlock(unsigned I) const {
    assert(I < getNumBranchAfters());
    return ExtInfo->BranchAfters[I].first;
  }

  llvm::ConstantInt *getBranchAfterIndex(unsigned I) const {
    assert(I < getNumBranchAfters());
    return ExtInfo->BranchAfters[I].second;
  }

  /// Add a branch-through to this cleanup scope.  A branch-through is
  /// a branch from a scope protected by this (normal) cleanup to an
  /// enclosing scope other than the immediately-enclosing normal
  /// cleanup scope.
  ///
  /// In the following example, the branch through B's scope is a
  /// branch-through, while the branch through A's scope is a
  /// branch-after:
  ///   for (;;) { A a; B b; break; }
  ///
  /// All branch-throughs have a common destination out of the
  /// cleanup, one possibly shared with the fall-through.  Therefore
  /// branch-throughs usually don't force a switch after the cleanup.
  ///
  /// \return true if the branch-through was new to this scope
  bool addBranchThrough(llvm::BasicBlock *Block) {
    return getExtInfo().Branches.insert(Block);
  }

  /// Determines if this cleanup scope has any branch throughs.
  bool hasBranchThroughs() const {
    if (!ExtInfo) return false;
    return (ExtInfo->BranchAfters.size() != ExtInfo->Branches.size());
  }

  // Same stuff, only for EH branches instead of normal branches.
  // It's quite possible that we could find a better representation
  // for this.

  bool hasEHBranches() const { return ExtInfo && !ExtInfo->EHBranches.empty(); }
  void addEHBranchAfter(llvm::ConstantInt *Index,
                        llvm::BasicBlock *Block) {
    struct ExtInfo &ExtInfo = getExtInfo();
    if (ExtInfo.EHBranches.insert(Block))
      ExtInfo.EHBranchAfters.push_back(std::make_pair(Block, Index));
  }

  unsigned getNumEHBranchAfters() const {
    return ExtInfo ? ExtInfo->EHBranchAfters.size() : 0;
  }

  llvm::BasicBlock *getEHBranchAfterBlock(unsigned I) const {
    assert(I < getNumEHBranchAfters());
    return ExtInfo->EHBranchAfters[I].first;
  }

  llvm::ConstantInt *getEHBranchAfterIndex(unsigned I) const {
    assert(I < getNumEHBranchAfters());
    return ExtInfo->EHBranchAfters[I].second;
  }

  bool addEHBranchThrough(llvm::BasicBlock *Block) {
    return getExtInfo().EHBranches.insert(Block);
  }

  bool hasEHBranchThroughs() const {
    if (!ExtInfo) return false;
    return (ExtInfo->EHBranchAfters.size() != ExtInfo->EHBranches.size());
  }

  static bool classof(const EHScope *Scope) {
    return (Scope->getKind() == Cleanup);
  }
};

/// An exceptions scope which filters exceptions thrown through it.
/// Only exceptions matching the filter types will be permitted to be
/// thrown.
///
/// This is used to implement C++ exception specifications.
class EHFilterScope : public EHScope {
  unsigned NumFilters : BitsRemaining;

  // Essentially ends in a flexible array member:
  // llvm::Value *FilterTypes[0];

  llvm::Value **getFilters() {
    return reinterpret_cast<llvm::Value**>(this+1);
  }

  llvm::Value * const *getFilters() const {
    return reinterpret_cast<llvm::Value* const *>(this+1);
  }

public:
  EHFilterScope(unsigned NumFilters) :
    EHScope(Filter), NumFilters(NumFilters) {}

  static size_t getSizeForNumFilters(unsigned NumFilters) {
    return sizeof(EHFilterScope) + NumFilters * sizeof(llvm::Value*);
  }

  unsigned getNumFilters() const { return NumFilters; }

  void setFilter(unsigned I, llvm::Value *FilterValue) {
    assert(I < getNumFilters());
    getFilters()[I] = FilterValue;
  }

  llvm::Value *getFilter(unsigned I) const {
    assert(I < getNumFilters());
    return getFilters()[I];
  }

  static bool classof(const EHScope *Scope) {
    return Scope->getKind() == Filter;
  }
};

/// An exceptions scope which calls std::terminate if any exception
/// reaches it.
class EHTerminateScope : public EHScope {
  unsigned DestIndex : BitsRemaining;
public:
  EHTerminateScope(unsigned Index) : EHScope(Terminate), DestIndex(Index) {}
  static size_t getSize() { return sizeof(EHTerminateScope); }

  unsigned getDestIndex() const { return DestIndex; }

  static bool classof(const EHScope *Scope) {
    return Scope->getKind() == Terminate;
  }
};

/// A non-stable pointer into the scope stack.
class EHScopeStack::iterator {
  char *Ptr;

  friend class EHScopeStack;
  explicit iterator(char *Ptr) : Ptr(Ptr) {}

public:
  iterator() : Ptr(0) {}

  EHScope *get() const { 
    return reinterpret_cast<EHScope*>(Ptr);
  }

  EHScope *operator->() const { return get(); }
  EHScope &operator*() const { return *get(); }

  iterator &operator++() {
    switch (get()->getKind()) {
    case EHScope::Catch:
      Ptr += EHCatchScope::getSizeForNumHandlers(
          static_cast<const EHCatchScope*>(get())->getNumHandlers());
      break;

    case EHScope::Filter:
      Ptr += EHFilterScope::getSizeForNumFilters(
          static_cast<const EHFilterScope*>(get())->getNumFilters());
      break;

    case EHScope::Cleanup:
      Ptr += static_cast<const EHCleanupScope*>(get())
        ->getAllocatedSize();
      break;

    case EHScope::Terminate:
      Ptr += EHTerminateScope::getSize();
      break;
    }

    return *this;
  }

  iterator next() {
    iterator copy = *this;
    ++copy;
    return copy;
  }

  iterator operator++(int) {
    iterator copy = *this;
    operator++();
    return copy;
  }

  bool encloses(iterator other) const { return Ptr >= other.Ptr; }
  bool strictlyEncloses(iterator other) const { return Ptr > other.Ptr; }

  bool operator==(iterator other) const { return Ptr == other.Ptr; }
  bool operator!=(iterator other) const { return Ptr != other.Ptr; }
};

inline EHScopeStack::iterator EHScopeStack::begin() const {
  return iterator(StartOfData);
}

inline EHScopeStack::iterator EHScopeStack::end() const {
  return iterator(EndOfBuffer);
}

inline void EHScopeStack::popCatch() {
  assert(!empty() && "popping exception stack when not empty");

  assert(isa<EHCatchScope>(*begin()));
  StartOfData += EHCatchScope::getSizeForNumHandlers(
                          cast<EHCatchScope>(*begin()).getNumHandlers());

  if (empty()) NextEHDestIndex = FirstEHDestIndex;

  assert(CatchDepth > 0 && "mismatched catch/terminate push/pop");
  CatchDepth--;
}

inline void EHScopeStack::popTerminate() {
  assert(!empty() && "popping exception stack when not empty");

  assert(isa<EHTerminateScope>(*begin()));
  StartOfData += EHTerminateScope::getSize();

  if (empty()) NextEHDestIndex = FirstEHDestIndex;

  assert(CatchDepth > 0 && "mismatched catch/terminate push/pop");
  CatchDepth--;
}

inline EHScopeStack::iterator EHScopeStack::find(stable_iterator sp) const {
  assert(sp.isValid() && "finding invalid savepoint");
  assert(sp.Size <= stable_begin().Size && "finding savepoint after pop");
  return iterator(EndOfBuffer - sp.Size);
}

inline EHScopeStack::stable_iterator
EHScopeStack::stabilize(iterator ir) const {
  assert(StartOfData <= ir.Ptr && ir.Ptr <= EndOfBuffer);
  return stable_iterator(EndOfBuffer - ir.Ptr);
}

inline EHScopeStack::stable_iterator
EHScopeStack::getInnermostActiveNormalCleanup() const {
  for (EHScopeStack::stable_iterator
         I = getInnermostNormalCleanup(), E = stable_end(); I != E; ) {
    EHCleanupScope &S = cast<EHCleanupScope>(*find(I));
    if (S.isActive()) return I;
    I = S.getEnclosingNormalCleanup();
  }
  return stable_end();
}

inline EHScopeStack::stable_iterator
EHScopeStack::getInnermostActiveEHCleanup() const {
  for (EHScopeStack::stable_iterator
         I = getInnermostEHCleanup(), E = stable_end(); I != E; ) {
    EHCleanupScope &S = cast<EHCleanupScope>(*find(I));
    if (S.isActive()) return I;
    I = S.getEnclosingEHCleanup();
  }
  return stable_end();
}

}
}

#endif
