//===- llvm/ADT/DenseMap.h - Dense probed hash table ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the DenseMap class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_DENSEMAP_H
#define LLVM_ADT_DENSEMAP_H

#include "llvm/Support/DataTypes.h"
#include <cassert>
#include <utility>

namespace llvm {
  
template<typename T>
struct DenseMapKeyInfo {
  //static inline T getEmptyKey();
  //static inline T getTombstoneKey();
  //static unsigned getHashValue(const T &Val);
  //static bool isPod()
};

template<typename T>
struct DenseMapKeyInfo<T*> {
  static inline T* getEmptyKey() { return (T*)-1; }
  static inline T* getTombstoneKey() { return (T*)-2; }
  static unsigned getHashValue(const T *PtrVal) {
    return (unsigned)((uintptr_t)PtrVal >> 4) ^
           (unsigned)((uintptr_t)PtrVal >> 9);
  }
  static bool isPod() { return true; }
};

template<typename KeyT, typename ValueT>
class DenseMapIterator;

template<typename KeyT, typename ValueT>
class DenseMap {
  typedef std::pair<KeyT, ValueT> BucketT;
  unsigned NumBuckets;
  BucketT *Buckets;
  
  unsigned NumEntries;
  DenseMap(const DenseMap &); // not implemented.
public:
  explicit DenseMap(unsigned NumInitBuckets = 8) {
    init(NumInitBuckets);
  }
  ~DenseMap() {
    const KeyT EmptyKey = getEmptyKey(), TombstoneKey = getTombstoneKey();
    for (BucketT *P = Buckets, *E = Buckets+NumBuckets; P != E; ++P) {
      if (P->first != EmptyKey && P->first != TombstoneKey)
        P->second.~ValueT();
      P->first.~KeyT();
    }
    delete[] (char*)Buckets;
  }
  
  typedef DenseMapIterator<KeyT, ValueT> iterator;
  typedef DenseMapIterator<KeyT, ValueT> const_iterator;
  inline iterator begin() const;
  inline iterator end() const;
  
  unsigned size() const { return NumEntries; }
  
  void clear() {
    const KeyT EmptyKey = getEmptyKey(), TombstoneKey = getTombstoneKey();
    for (BucketT *P = Buckets, *E = Buckets+NumBuckets; P != E; ++P) {
      if (P->first != EmptyKey && P->first != TombstoneKey) {
        P->first = EmptyKey;
        P->second.~ValueT();
        --NumEntries;
      }
    }
    assert(NumEntries == 0 && "Node count imbalance!");
  }
  
  /// count - Return true if the specified key is in the map.
  bool count(const KeyT &Val) const {
    BucketT *TheBucket;
    return LookupBucketFor(Val, TheBucket);
  }
  
  ValueT &operator[](const KeyT &Val) {
    BucketT *TheBucket;
    if (LookupBucketFor(Val, TheBucket))
      return TheBucket->second;

    // If the load of the hash table is more than 3/4, grow it.
    if (NumEntries*4 >= NumBuckets*3) {
      this->grow();
      LookupBucketFor(Val, TheBucket);
    }
    ++NumEntries;
    TheBucket->first = Val;
    new (&TheBucket->second) ValueT();
    return TheBucket->second;
  }
  
private:
  unsigned getHashValue(const KeyT &Val) const {
    return DenseMapKeyInfo<KeyT>::getHashValue(Val);
  }
  const KeyT getEmptyKey() const { return DenseMapKeyInfo<KeyT>::getEmptyKey();}
  const KeyT getTombstoneKey() const {
    return DenseMapKeyInfo<KeyT>::getTombstoneKey();
  }
  
  /// LookupBucketFor - Lookup the appropriate bucket for Val, returning it in
  /// FoundBucket.  If the bucket contains the key and a value, this returns
  /// true, otherwise it returns a bucket with an empty marker or tombstone and
  /// returns false.
  bool LookupBucketFor(const KeyT &Val, BucketT *&FoundBucket) const {
    unsigned BucketNo = getHashValue(Val);
    unsigned ProbeAmt = 1;
    BucketT *BucketsPtr = Buckets;
    
    // FoundTombstone - Keep track of whether we find a tombstone while probing.
    BucketT *FoundTombstone = 0;
    const KeyT EmptyKey = getEmptyKey();
    const KeyT TombstoneKey = getTombstoneKey();
    assert(Val != EmptyKey && Val != TombstoneKey &&
           "Empty/Tombstone value shouldn't be inserted into map!");
      
    while (1) {
      BucketT *ThisBucket = BucketsPtr + (BucketNo & (NumBuckets-1));
      // Found Val's bucket?  If so, return it.
      if (ThisBucket->first == Val) {
        FoundBucket = ThisBucket;
        return true;
      }
      
      // If we found an empty bucket, the key doesn't exist in the set.
      // Insert it and return the default value.
      if (ThisBucket->first == EmptyKey) {
        // If we've already seen a tombstone while probing, fill it in instead
        // of the empty bucket we eventually probed to.
        if (FoundTombstone) ThisBucket = FoundTombstone;
        FoundBucket = FoundTombstone ? FoundTombstone : ThisBucket;
        return false;
      }
      
      // If this is a tombstone, remember it.  If Val ends up not in the map, we
      // prefer to return it than something that would require more probing.
      if (ThisBucket->first == TombstoneKey && !FoundTombstone)
        FoundTombstone = ThisBucket;  // Remember the first tombstone found.
      
      // Otherwise, it's a hash collision or a tombstone, continue quadratic
      // probing.
      BucketNo += ProbeAmt++;
    }
  }

  void init(unsigned InitBuckets) {
    NumEntries = 0;
    NumBuckets = InitBuckets;
    assert(InitBuckets && (InitBuckets & InitBuckets-1) == 0 &&
           "# initial buckets must be a power of two!");
    Buckets = (BucketT*)new char[sizeof(BucketT)*InitBuckets];
    // Initialize all the keys to EmptyKey.
    const KeyT EmptyKey = getEmptyKey();
    for (unsigned i = 0; i != InitBuckets; ++i)
      new (&Buckets[i].first) KeyT(EmptyKey);
  }
  
  void grow() {
    unsigned OldNumBuckets = NumBuckets;
    BucketT *OldBuckets = Buckets;
    
    // Double the number of buckets.
    NumBuckets <<= 1;
    Buckets = (BucketT*)new char[sizeof(BucketT)*NumBuckets];

    // Initialize all the keys to EmptyKey.
    const KeyT EmptyKey = getEmptyKey();
    for (unsigned i = 0, e = NumBuckets; i != e; ++i)
      new (&Buckets[i].first) KeyT(EmptyKey);

    // Insert all the old elements.
    const KeyT TombstoneKey = getTombstoneKey();
    for (BucketT *B = OldBuckets, *E = OldBuckets+OldNumBuckets; B != E; ++B) {
      if (B->first != EmptyKey && B->first != TombstoneKey) {
        // Insert the key/value into the new table.
        BucketT *DestBucket;
        bool FoundVal = LookupBucketFor(B->first, DestBucket);
        assert(!FoundVal && "Key already in new map?");
        DestBucket->first = B->first;
        new (&DestBucket->second) ValueT(B->second);
        
        // Free the value.
        B->second.~ValueT();
      }
      B->first.~KeyT();
    }
    
    // Free the old table.
    delete[] (char*)OldBuckets;
  }
};

template<typename KeyT, typename ValueT>
class DenseMapIterator {
  typedef std::pair<KeyT, ValueT> BucketT;
  const BucketT *Ptr, *End;
public:
  DenseMapIterator(const BucketT *Pos, const BucketT *E) : Ptr(Pos), End(E) {
    AdvancePastEmptyBuckets();
  }
  
  const std::pair<KeyT, ValueT> &operator*() const {
    return *Ptr;
  }
  const std::pair<KeyT, ValueT> *operator->() const {
    return Ptr;
  }
  
  bool operator==(const DenseMapIterator &RHS) const {
    return Ptr == RHS.Ptr;
  }
  bool operator!=(const DenseMapIterator &RHS) const {
    return Ptr != RHS.Ptr;
  }
  
  inline DenseMapIterator& operator++() {          // Preincrement
    ++Ptr;
    AdvancePastEmptyBuckets();
    return *this;
  }
  DenseMapIterator operator++(int) {        // Postincrement
    DenseMapIterator tmp = *this; ++*this; return tmp;
  }
  
private:
  void AdvancePastEmptyBuckets() {
    const KeyT Empty = DenseMapKeyInfo<KeyT>::getEmptyKey();
    const KeyT Tombstone = DenseMapKeyInfo<KeyT>::getTombstoneKey();

    while (Ptr != End && Ptr->first != Empty && Ptr->first != Tombstone)
      ++Ptr;
  }
};


template<typename KeyT, typename ValueT>
inline DenseMapIterator<KeyT, ValueT> DenseMap<KeyT, ValueT>::begin() const {
  return DenseMapIterator<KeyT, ValueT>(Buckets, Buckets+NumBuckets);
}
template<typename KeyT, typename ValueT>
inline DenseMapIterator<KeyT, ValueT> DenseMap<KeyT, ValueT>::end() const {
  return DenseMapIterator<KeyT, ValueT>(Buckets+NumBuckets, Buckets+NumBuckets);
}

} // end namespace llvm

#endif
