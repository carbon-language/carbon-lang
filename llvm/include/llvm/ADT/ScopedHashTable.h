//===- ScopedHashTable.h - A simple scoped hash table ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements an efficient scoped hash table, which is useful for
// things like dominator-based optimizations.  This allows clients to do things
// like this:
//
//  ScopedHashTable<int, int> HT;
//  {
//    ScopedHashTableScope<int, int> Scope1(HT);
//    HT.insert(0, 0);
//    HT.insert(1, 1);
//    {
//      ScopedHashTableScope<int, int> Scope2(HT);
//      HT.insert(0, 42);
//    }
//  }
//
// Looking up the value for "0" in the Scope2 block will return 42.  Looking
// up the value for 0 before 42 is inserted or after Scope2 is popped will
// return 0.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_SCOPEDHASHTABLE_H
#define LLVM_ADT_SCOPEDHASHTABLE_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Allocator.h"

namespace llvm {

template <typename K, typename V, typename KInfo = DenseMapInfo<K>,
          typename AllocatorTy = MallocAllocator>
class ScopedHashTable;

template <typename K, typename V, typename KInfo = DenseMapInfo<K> >
class ScopedHashTableVal {
  ScopedHashTableVal *NextInScope;
  ScopedHashTableVal *NextForKey;
  K Key;
  V Val;
  ScopedHashTableVal(const K &key, const V &val) : Key(key), Val(val) {}
public:

  const K &getKey() const { return Key; }
  const V &getValue() const { return Val; }
  V &getValue() { return Val; }

  ScopedHashTableVal *getNextForKey() { return NextForKey; }
  const ScopedHashTableVal *getNextForKey() const { return NextForKey; }
  ScopedHashTableVal *getNextInScope() { return NextInScope; }
  
  template <typename AllocatorTy>
  static ScopedHashTableVal *Create(ScopedHashTableVal *nextInScope,
                                    ScopedHashTableVal *nextForKey,
                                    const K &key, const V &val,
                                    AllocatorTy &Allocator) {
    ScopedHashTableVal *New = Allocator.template Allocate<ScopedHashTableVal>();
    // Set up the value.
    new (New) ScopedHashTableVal(key, val);
    New->NextInScope = nextInScope;
    New->NextForKey = nextForKey; 
    return New;
  }
  
  template <typename AllocatorTy>
  void Destroy(AllocatorTy &Allocator) {
    // Free memory referenced by the item.
    this->~ScopedHashTableVal();
    Allocator.Deallocate(this);
  }
};

template <typename K, typename V, typename KInfo = DenseMapInfo<K> >
class ScopedHashTableScope {
  /// HT - The hashtable that we are active for.
  ScopedHashTable<K, V, KInfo> &HT;

  /// PrevScope - This is the scope that we are shadowing in HT.
  ScopedHashTableScope *PrevScope;

  /// LastValInScope - This is the last value that was inserted for this scope
  /// or null if none have been inserted yet.
  ScopedHashTableVal<K, V, KInfo> *LastValInScope;
  void operator=(ScopedHashTableScope&);       // DO NOT IMPLEMENT
  ScopedHashTableScope(ScopedHashTableScope&); // DO NOT IMPLEMENT
public:
  ScopedHashTableScope(ScopedHashTable<K, V, KInfo> &HT);
  ~ScopedHashTableScope();

private:
  friend class ScopedHashTable<K, V, KInfo>;
  ScopedHashTableVal<K, V, KInfo> *getLastValInScope() {
    return LastValInScope;
  }
  void setLastValInScope(ScopedHashTableVal<K, V, KInfo> *Val) {
    LastValInScope = Val;
  }
};


template <typename K, typename V, typename KInfo = DenseMapInfo<K> >
class ScopedHashTableIterator {
  ScopedHashTableVal<K, V, KInfo> *Node;
public:
  ScopedHashTableIterator(ScopedHashTableVal<K, V, KInfo> *node) : Node(node) {}

  V &operator*() const {
    assert(Node && "Dereference end()");
    return Node->getValue();
  }
  V *operator->() const {
    return &Node->getValue();
  }

  bool operator==(const ScopedHashTableIterator &RHS) const {
    return Node == RHS.Node;
  }
  bool operator!=(const ScopedHashTableIterator &RHS) const {
    return Node != RHS.Node;
  }

  inline ScopedHashTableIterator& operator++() {          // Preincrement
    assert(Node && "incrementing past end()");
    Node = Node->getNextForKey();
    return *this;
  }
  ScopedHashTableIterator operator++(int) {        // Postincrement
    ScopedHashTableIterator tmp = *this; ++*this; return tmp;
  }
};


template <typename K, typename V, typename KInfo, typename AllocatorTy>
class ScopedHashTable {
  typedef ScopedHashTableVal<K, V, KInfo> ValTy;
  DenseMap<K, ValTy*, KInfo> TopLevelMap;
  ScopedHashTableScope<K, V, KInfo> *CurScope;
  
  AllocatorTy Allocator;
  
  ScopedHashTable(const ScopedHashTable&); // NOT YET IMPLEMENTED
  void operator=(const ScopedHashTable&);  // NOT YET IMPLEMENTED
  friend class ScopedHashTableScope<K, V, KInfo>;
public:
  ScopedHashTable() : CurScope(0) {}
  ScopedHashTable(AllocatorTy A) : CurScope(0), Allocator(A) {}
  ~ScopedHashTable() {
    assert(CurScope == 0 && TopLevelMap.empty() && "Scope imbalance!");
  }
  
  typedef typename ReferenceAdder<AllocatorTy>::result AllocatorRefTy;
  typedef typename ReferenceAdder<const AllocatorTy>::result AllocatorCRefTy;
  AllocatorRefTy getAllocator() { return Allocator; }
  AllocatorCRefTy getAllocator() const { return Allocator; }

  bool count(const K &Key) const {
    return TopLevelMap.count(Key);
  }

  V lookup(const K &Key) {
    typename DenseMap<K, ValTy*, KInfo>::iterator I = TopLevelMap.find(Key);
    if (I != TopLevelMap.end())
      return I->second->getValue();
      
    return V();
  }

  void insert(const K &Key, const V &Val) {
    assert(CurScope && "No scope active!");

    ScopedHashTableVal<K, V, KInfo> *&KeyEntry = TopLevelMap[Key];

    KeyEntry = ValTy::Create(CurScope->getLastValInScope(), KeyEntry, Key, Val,
                             Allocator);
    CurScope->setLastValInScope(KeyEntry);
  }

  typedef ScopedHashTableIterator<K, V, KInfo> iterator;

  iterator end() { return iterator(0); }

  iterator begin(const K &Key) {
    typename DenseMap<K, ValTy*, KInfo>::iterator I =
      TopLevelMap.find(Key);
    if (I == TopLevelMap.end()) return end();
    return iterator(I->second);
  }
};

/// ScopedHashTableScope ctor - Install this as the current scope for the hash
/// table.
template <typename K, typename V, typename KInfo>
ScopedHashTableScope<K, V, KInfo>::
  ScopedHashTableScope(ScopedHashTable<K, V, KInfo> &ht) : HT(ht) {
  PrevScope = HT.CurScope;
  HT.CurScope = this;
  LastValInScope = 0;
}

template <typename K, typename V, typename KInfo>
ScopedHashTableScope<K, V, KInfo>::~ScopedHashTableScope() {
  assert(HT.CurScope == this && "Scope imbalance!");
  HT.CurScope = PrevScope;

  // Pop and delete all values corresponding to this scope.
  while (ScopedHashTableVal<K, V, KInfo> *ThisEntry = LastValInScope) {
    // Pop this value out of the TopLevelMap.
    if (ThisEntry->getNextForKey() == 0) {
      assert(HT.TopLevelMap[ThisEntry->getKey()] == ThisEntry &&
             "Scope imbalance!");
      HT.TopLevelMap.erase(ThisEntry->getKey());
    } else {
      ScopedHashTableVal<K, V, KInfo> *&KeyEntry =
        HT.TopLevelMap[ThisEntry->getKey()];
      assert(KeyEntry == ThisEntry && "Scope imbalance!");
      KeyEntry = ThisEntry->getNextForKey();
    }

    // Pop this value out of the scope.
    LastValInScope = ThisEntry->getNextInScope();

    // Delete this entry.
    ThisEntry->Destroy(HT.getAllocator());
  }
}

} // end namespace llvm

#endif
