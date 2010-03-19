//===-- UnresolvedSet.h - Unresolved sets of declarations  ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the UnresolvedSet class, which is used to store
//  collections of declarations in the AST.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_UNRESOLVEDSET_H
#define LLVM_CLANG_AST_UNRESOLVEDSET_H

#include <iterator>
#include "llvm/ADT/SmallVector.h"
#include "clang/Basic/Specifiers.h"

namespace clang {

class NamedDecl;

/// A POD class for pairing a NamedDecl* with an access specifier.
/// Can be put into unions.
class DeclAccessPair {
  NamedDecl *Ptr; // we'd use llvm::PointerUnion, but it isn't trivial

  enum { Mask = 0x3 };

public:
  static DeclAccessPair make(NamedDecl *D, AccessSpecifier AS) {
    DeclAccessPair p;
    p.set(D, AS);
    return p;
  }

  NamedDecl *getDecl() const {
    return (NamedDecl*) (~Mask & (uintptr_t) Ptr);
  }
  AccessSpecifier getAccess() const {
    return AccessSpecifier(Mask & (uintptr_t) Ptr);
  }

  void setDecl(NamedDecl *D) {
    set(D, getAccess());
  }
  void setAccess(AccessSpecifier AS) {
    set(getDecl(), AS);
  }
  void set(NamedDecl *D, AccessSpecifier AS) {
    Ptr = reinterpret_cast<NamedDecl*>(uintptr_t(AS) |
                                       reinterpret_cast<uintptr_t>(D));
  }

  operator NamedDecl*() const { return getDecl(); }
  NamedDecl *operator->() const { return getDecl(); }
};
}

// Take a moment to tell SmallVector that this is POD.
namespace llvm {
template<typename> struct isPodLike;
template<> struct isPodLike<clang::DeclAccessPair> {
   static const bool value = true;
};
}

namespace clang {

/// The iterator over UnresolvedSets.  Serves as both the const and
/// non-const iterator.
class UnresolvedSetIterator {
private:
  typedef llvm::SmallVectorImpl<DeclAccessPair> DeclsTy;
  typedef DeclsTy::iterator IteratorTy;

  IteratorTy ir;

  friend class UnresolvedSetImpl;
  explicit UnresolvedSetIterator(DeclsTy::iterator ir) : ir(ir) {}
  explicit UnresolvedSetIterator(DeclsTy::const_iterator ir) :
    ir(const_cast<DeclsTy::iterator>(ir)) {}
public:
  UnresolvedSetIterator() {}

  typedef std::iterator_traits<IteratorTy>::difference_type difference_type;
  typedef NamedDecl *value_type;
  typedef NamedDecl **pointer;
  typedef NamedDecl *reference;
  typedef std::iterator_traits<IteratorTy>::iterator_category iterator_category;

  NamedDecl *getDecl() const { return ir->getDecl(); }
  AccessSpecifier getAccess() const { return ir->getAccess(); }
  DeclAccessPair getPair() const { return *ir; }

  NamedDecl *operator*() const { return getDecl(); }
  
  UnresolvedSetIterator &operator++() { ++ir; return *this; }
  UnresolvedSetIterator operator++(int) { return UnresolvedSetIterator(ir++); }
  UnresolvedSetIterator &operator--() { --ir; return *this; }
  UnresolvedSetIterator operator--(int) { return UnresolvedSetIterator(ir--); }

  UnresolvedSetIterator &operator+=(difference_type d) {
    ir += d; return *this;
  }
  UnresolvedSetIterator operator+(difference_type d) const {
    return UnresolvedSetIterator(ir + d);
  }
  UnresolvedSetIterator &operator-=(difference_type d) {
    ir -= d; return *this;
  }
  UnresolvedSetIterator operator-(difference_type d) const {
    return UnresolvedSetIterator(ir - d);
  }
  value_type operator[](difference_type d) const { return *(*this + d); }

  difference_type operator-(const UnresolvedSetIterator &o) const {
    return ir - o.ir;
  }

  bool operator==(const UnresolvedSetIterator &o) const { return ir == o.ir; }
  bool operator!=(const UnresolvedSetIterator &o) const { return ir != o.ir; }
  bool operator<(const UnresolvedSetIterator &o) const { return ir < o.ir; }
  bool operator<=(const UnresolvedSetIterator &o) const { return ir <= o.ir; }
  bool operator>=(const UnresolvedSetIterator &o) const { return ir >= o.ir; }
  bool operator>(const UnresolvedSetIterator &o) const { return ir > o.ir; }
};

/// UnresolvedSet - A set of unresolved declarations.  This is needed
/// in a lot of places, but isn't really worth breaking into its own
/// header right now.
class UnresolvedSetImpl {
  typedef UnresolvedSetIterator::DeclsTy DeclsTy;

  // Don't allow direct construction, and only permit subclassing by
  // UnresolvedSet.
private:
  template <unsigned N> friend class UnresolvedSet;
  UnresolvedSetImpl() {}
  UnresolvedSetImpl(const UnresolvedSetImpl &) {}

public:
  // We don't currently support assignment through this iterator, so we might
  // as well use the same implementation twice.
  typedef UnresolvedSetIterator iterator;
  typedef UnresolvedSetIterator const_iterator;

  iterator begin() { return iterator(decls().begin()); }
  iterator end() { return iterator(decls().end()); }

  const_iterator begin() const { return const_iterator(decls().begin()); }
  const_iterator end() const { return const_iterator(decls().end()); }

  void addDecl(NamedDecl *D) {
    addDecl(D, AS_none);
  }

  void addDecl(NamedDecl *D, AccessSpecifier AS) {
    decls().push_back(DeclAccessPair::make(D, AS));
  }

  /// Replaces the given declaration with the new one, once.
  ///
  /// \return true if the set changed
  bool replace(const NamedDecl* Old, NamedDecl *New) {
    for (DeclsTy::iterator I = decls().begin(), E = decls().end(); I != E; ++I)
      if (I->getDecl() == Old)
        return (I->setDecl(New), true);
    return false;
  }

  /// Replaces the declaration at the given iterator with the new one,
  /// preserving the original access bits.
  void replace(iterator I, NamedDecl *New) {
    I.ir->setDecl(New);
  }

  void replace(iterator I, NamedDecl *New, AccessSpecifier AS) {
    I.ir->set(New, AS);
  }

  void erase(unsigned I) {
    decls()[I] = decls().back();
    decls().pop_back();
  }

  void erase(iterator I) {
    *I.ir = decls().back();
    decls().pop_back();
  }

  void setAccess(iterator I, AccessSpecifier AS) {
    I.ir->setAccess(AS);
  }

  void clear() { decls().clear(); }
  void set_size(unsigned N) { decls().set_size(N); }

  bool empty() const { return decls().empty(); }
  unsigned size() const { return decls().size(); }

  void append(iterator I, iterator E) {
    decls().append(I.ir, E.ir);
  }

  DeclAccessPair &operator[](unsigned I) { return decls()[I]; }
  const DeclAccessPair &operator[](unsigned I) const { return decls()[I]; }

private:
  // These work because the only permitted subclass is UnresolvedSetImpl

  DeclsTy &decls() {
    return *reinterpret_cast<DeclsTy*>(this);
  }
  const DeclsTy &decls() const {
    return *reinterpret_cast<const DeclsTy*>(this);
  }
};

/// A set of unresolved declarations 
template <unsigned InlineCapacity> class UnresolvedSet :
    public UnresolvedSetImpl {
  llvm::SmallVector<DeclAccessPair, InlineCapacity> Decls;
};

  
} // namespace clang

#endif
