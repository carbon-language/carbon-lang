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

/// A protected scope for zero-cost EH handling.
class EHScope {
  llvm::BasicBlock *CachedLandingPad;

  unsigned K : 3;

protected:
  enum { BitsRemaining = 29 };

public:
  enum Kind { Cleanup, LazyCleanup, Catch, Terminate, Filter };

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

    static Handler make(llvm::Value *Type, llvm::BasicBlock *Block) {
      Handler Temp;
      Temp.Type = Type;
      Temp.Block = Block;
      return Temp;
    }
  };

private:
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
    getHandlers()[I] = Handler::make(Type, Block);
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
class EHLazyCleanupScope : public EHScope {
  /// Whether this cleanup needs to be run along normal edges.
  bool IsNormalCleanup : 1;

  /// Whether this cleanup needs to be run along exception edges.
  bool IsEHCleanup : 1;

  /// The amount of extra storage needed by the LazyCleanup.
  /// Always a multiple of the scope-stack alignment.
  unsigned CleanupSize : 12;

  /// The number of fixups required by enclosing scopes (not including
  /// this one).  If this is the top cleanup scope, all the fixups
  /// from this index onwards belong to this scope.
  unsigned FixupDepth : BitsRemaining - 14;

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

public:
  /// Gets the size required for a lazy cleanup scope with the given
  /// cleanup-data requirements.
  static size_t getSizeForCleanupSize(size_t Size) {
    return sizeof(EHLazyCleanupScope) + Size;
  }

  size_t getAllocatedSize() const {
    return sizeof(EHLazyCleanupScope) + CleanupSize;
  }

  EHLazyCleanupScope(bool IsNormal, bool IsEH, unsigned CleanupSize,
                     unsigned FixupDepth,
                     EHScopeStack::stable_iterator EnclosingNormal,
                     EHScopeStack::stable_iterator EnclosingEH)
    : EHScope(EHScope::LazyCleanup),
      IsNormalCleanup(IsNormal), IsEHCleanup(IsEH),
      CleanupSize(CleanupSize), FixupDepth(FixupDepth),
      EnclosingNormal(EnclosingNormal), EnclosingEH(EnclosingEH),
      NormalBlock(0), EHBlock(0)
  {}

  bool isNormalCleanup() const { return IsNormalCleanup; }
  llvm::BasicBlock *getNormalBlock() const { return NormalBlock; }
  void setNormalBlock(llvm::BasicBlock *BB) { NormalBlock = BB; }

  bool isEHCleanup() const { return IsEHCleanup; }
  llvm::BasicBlock *getEHBlock() const { return EHBlock; }
  void setEHBlock(llvm::BasicBlock *BB) { EHBlock = BB; }

  unsigned getFixupDepth() const { return FixupDepth; }
  EHScopeStack::stable_iterator getEnclosingNormalCleanup() const {
    return EnclosingNormal;
  }
  EHScopeStack::stable_iterator getEnclosingEHCleanup() const {
    return EnclosingEH;
  }

  size_t getCleanupSize() const { return CleanupSize; }
  void *getCleanupBuffer() { return this + 1; }

  EHScopeStack::LazyCleanup *getCleanup() {
    return reinterpret_cast<EHScopeStack::LazyCleanup*>(getCleanupBuffer());
  }

  static bool classof(const EHScope *Scope) {
    return (Scope->getKind() == LazyCleanup);
  }
};

/// A scope which needs to execute some code if we try to unwind ---
/// either normally, via the EH mechanism, or both --- through it.
class EHCleanupScope : public EHScope {
  /// The number of fixups required by enclosing scopes (not including
  /// this one).  If this is the top cleanup scope, all the fixups
  /// from this index onwards belong to this scope.
  unsigned FixupDepth : BitsRemaining;

  /// The nearest normal cleanup scope enclosing this one.
  EHScopeStack::stable_iterator EnclosingNormal;

  /// The nearest EH cleanup scope enclosing this one.
  EHScopeStack::stable_iterator EnclosingEH;

  llvm::BasicBlock *NormalEntry;
  llvm::BasicBlock *NormalExit;
  llvm::BasicBlock *EHEntry;
  llvm::BasicBlock *EHExit;

public:
  static size_t getSize() { return sizeof(EHCleanupScope); }

  EHCleanupScope(unsigned FixupDepth,
                 EHScopeStack::stable_iterator EnclosingNormal,
                 EHScopeStack::stable_iterator EnclosingEH,
                 llvm::BasicBlock *NormalEntry, llvm::BasicBlock *NormalExit,
                 llvm::BasicBlock *EHEntry, llvm::BasicBlock *EHExit)
    : EHScope(Cleanup), FixupDepth(FixupDepth),
      EnclosingNormal(EnclosingNormal), EnclosingEH(EnclosingEH),
      NormalEntry(NormalEntry), NormalExit(NormalExit),
      EHEntry(EHEntry), EHExit(EHExit) {
    assert((NormalEntry != 0) == (NormalExit != 0));
    assert((EHEntry != 0) == (EHExit != 0));
  }

  bool isNormalCleanup() const { return NormalEntry != 0; }
  bool isEHCleanup() const { return EHEntry != 0; }

  llvm::BasicBlock *getNormalEntry() const { return NormalEntry; }
  llvm::BasicBlock *getNormalExit() const { return NormalExit; }
  llvm::BasicBlock *getEHEntry() const { return EHEntry; }
  llvm::BasicBlock *getEHExit() const { return EHExit; }
  unsigned getFixupDepth() const { return FixupDepth; }
  EHScopeStack::stable_iterator getEnclosingNormalCleanup() const {
    return EnclosingNormal;
  }
  EHScopeStack::stable_iterator getEnclosingEHCleanup() const {
    return EnclosingEH;
  }

  static bool classof(const EHScope *Scope) {
    return Scope->getKind() == Cleanup;
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
public:
  EHTerminateScope() : EHScope(Terminate) {}
  static size_t getSize() { return sizeof(EHTerminateScope); }

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

    case EHScope::LazyCleanup:
      Ptr += static_cast<const EHLazyCleanupScope*>(get())
        ->getAllocatedSize();
      break;

    case EHScope::Cleanup:
      Ptr += EHCleanupScope::getSize();
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

  assert(CatchDepth > 0 && "mismatched catch/terminate push/pop");
  CatchDepth--;
}

inline void EHScopeStack::popTerminate() {
  assert(!empty() && "popping exception stack when not empty");

  assert(isa<EHTerminateScope>(*begin()));
  StartOfData += EHTerminateScope::getSize();

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

}
}

#endif
