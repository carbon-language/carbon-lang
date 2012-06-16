//===- llvm/ADT/DenseMap.h - Dense probed hash table ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the DenseMap class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_DENSEMAP_H
#define LLVM_ADT_DENSEMAP_H

#include "llvm/Support/Compiler.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/PointerLikeTypeTraits.h"
#include "llvm/Support/type_traits.h"
#include "llvm/ADT/DenseMapInfo.h"
#include <algorithm>
#include <iterator>
#include <new>
#include <utility>
#include <cassert>
#include <cstddef>
#include <cstring>

namespace llvm {

template<typename KeyT, typename ValueT,
         typename KeyInfoT = DenseMapInfo<KeyT>,
         bool IsConst = false>
class DenseMapIterator;

template<typename DerivedT,
         typename KeyT, typename ValueT, typename KeyInfoT>
class DenseMapBase {
protected:
  typedef std::pair<KeyT, ValueT> BucketT;

public:
  typedef KeyT key_type;
  typedef ValueT mapped_type;
  typedef BucketT value_type;

  typedef DenseMapIterator<KeyT, ValueT, KeyInfoT> iterator;
  typedef DenseMapIterator<KeyT, ValueT,
                           KeyInfoT, true> const_iterator;
  inline iterator begin() {
    // When the map is empty, avoid the overhead of AdvancePastEmptyBuckets().
    return empty() ? end() : iterator(getBuckets(), getBucketsEnd());
  }
  inline iterator end() {
    return iterator(getBucketsEnd(), getBucketsEnd(), true);
  }
  inline const_iterator begin() const {
    return empty() ? end() : const_iterator(getBuckets(), getBucketsEnd());
  }
  inline const_iterator end() const {
    return const_iterator(getBucketsEnd(), getBucketsEnd(), true);
  }

  bool empty() const { return getNumEntries() == 0; }
  unsigned size() const { return getNumEntries(); }

  /// Grow the densemap so that it has at least Size buckets. Does not shrink
  void resize(size_t Size) {
    if (Size > getNumBuckets())
      grow(Size);
  }

  void clear() {
    if (getNumEntries() == 0 && getNumTombstones() == 0) return;
    
    // If the capacity of the array is huge, and the # elements used is small,
    // shrink the array.
    if (getNumEntries() * 4 < getNumBuckets() && getNumBuckets() > 64) {
      shrink_and_clear();
      return;
    }

    const KeyT EmptyKey = getEmptyKey(), TombstoneKey = getTombstoneKey();
    for (BucketT *P = getBuckets(), *E = getBucketsEnd(); P != E; ++P) {
      if (!KeyInfoT::isEqual(P->first, EmptyKey)) {
        if (!KeyInfoT::isEqual(P->first, TombstoneKey)) {
          P->second.~ValueT();
          decrementNumEntries();
        }
        P->first = EmptyKey;
      }
    }
    assert(getNumEntries() == 0 && "Node count imbalance!");
    setNumTombstones(0);
  }

  /// count - Return true if the specified key is in the map.
  bool count(const KeyT &Val) const {
    BucketT *TheBucket;
    return LookupBucketFor(Val, TheBucket);
  }

  iterator find(const KeyT &Val) {
    BucketT *TheBucket;
    if (LookupBucketFor(Val, TheBucket))
      return iterator(TheBucket, getBucketsEnd(), true);
    return end();
  }
  const_iterator find(const KeyT &Val) const {
    BucketT *TheBucket;
    if (LookupBucketFor(Val, TheBucket))
      return const_iterator(TheBucket, getBucketsEnd(), true);
    return end();
  }

  /// Alternate version of find() which allows a different, and possibly
  /// less expensive, key type.
  /// The DenseMapInfo is responsible for supplying methods
  /// getHashValue(LookupKeyT) and isEqual(LookupKeyT, KeyT) for each key
  /// type used.
  template<class LookupKeyT>
  iterator find_as(const LookupKeyT &Val) {
    BucketT *TheBucket;
    if (LookupBucketFor(Val, TheBucket))
      return iterator(TheBucket, getBucketsEnd(), true);
    return end();
  }
  template<class LookupKeyT>
  const_iterator find_as(const LookupKeyT &Val) const {
    BucketT *TheBucket;
    if (LookupBucketFor(Val, TheBucket))
      return const_iterator(TheBucket, getBucketsEnd(), true);
    return end();
  }

  /// lookup - Return the entry for the specified key, or a default
  /// constructed value if no such entry exists.
  ValueT lookup(const KeyT &Val) const {
    BucketT *TheBucket;
    if (LookupBucketFor(Val, TheBucket))
      return TheBucket->second;
    return ValueT();
  }

  // Inserts key,value pair into the map if the key isn't already in the map.
  // If the key is already in the map, it returns false and doesn't update the
  // value.
  std::pair<iterator, bool> insert(const std::pair<KeyT, ValueT> &KV) {
    BucketT *TheBucket;
    if (LookupBucketFor(KV.first, TheBucket))
      return std::make_pair(iterator(TheBucket, getBucketsEnd(), true),
                            false); // Already in map.

    // Otherwise, insert the new element.
    TheBucket = InsertIntoBucket(KV.first, KV.second, TheBucket);
    return std::make_pair(iterator(TheBucket, getBucketsEnd(), true), true);
  }

  /// insert - Range insertion of pairs.
  template<typename InputIt>
  void insert(InputIt I, InputIt E) {
    for (; I != E; ++I)
      insert(*I);
  }


  bool erase(const KeyT &Val) {
    BucketT *TheBucket;
    if (!LookupBucketFor(Val, TheBucket))
      return false; // not in map.

    TheBucket->second.~ValueT();
    TheBucket->first = getTombstoneKey();
    decrementNumEntries();
    incrementNumTombstones();
    return true;
  }
  void erase(iterator I) {
    BucketT *TheBucket = &*I;
    TheBucket->second.~ValueT();
    TheBucket->first = getTombstoneKey();
    decrementNumEntries();
    incrementNumTombstones();
  }

  value_type& FindAndConstruct(const KeyT &Key) {
    BucketT *TheBucket;
    if (LookupBucketFor(Key, TheBucket))
      return *TheBucket;

    return *InsertIntoBucket(Key, ValueT(), TheBucket);
  }

  ValueT &operator[](const KeyT &Key) {
    return FindAndConstruct(Key).second;
  }

#if LLVM_USE_RVALUE_REFERENCES
  value_type& FindAndConstruct(KeyT &&Key) {
    BucketT *TheBucket;
    if (LookupBucketFor(Key, TheBucket))
      return *TheBucket;

    return *InsertIntoBucket(Key, ValueT(), TheBucket);
  }

  ValueT &operator[](KeyT &&Key) {
    return FindAndConstruct(Key).second;
  }
#endif

  /// isPointerIntoBucketsArray - Return true if the specified pointer points
  /// somewhere into the DenseMap's array of buckets (i.e. either to a key or
  /// value in the DenseMap).
  bool isPointerIntoBucketsArray(const void *Ptr) const {
    return Ptr >= getBuckets() && Ptr < getBucketsEnd();
  }

  /// getPointerIntoBucketsArray() - Return an opaque pointer into the buckets
  /// array.  In conjunction with the previous method, this can be used to
  /// determine whether an insertion caused the DenseMap to reallocate.
  const void *getPointerIntoBucketsArray() const { return getBuckets(); }

protected:
  DenseMapBase() {}

  void destroyAll() {
    if (getNumBuckets() == 0) // Nothing to do.
      return;

    const KeyT EmptyKey = getEmptyKey(), TombstoneKey = getTombstoneKey();
    for (BucketT *P = getBuckets(), *E = getBucketsEnd(); P != E; ++P) {
      if (!KeyInfoT::isEqual(P->first, EmptyKey) &&
          !KeyInfoT::isEqual(P->first, TombstoneKey))
        P->second.~ValueT();
      P->first.~KeyT();
    }

#ifndef NDEBUG
    memset((void*)getBuckets(), 0x5a, sizeof(BucketT)*getNumBuckets());
#endif
  }

  void initEmpty() {
    setNumEntries(0);
    setNumTombstones(0);

    assert((getNumBuckets() & (getNumBuckets()-1)) == 0 &&
           "# initial buckets must be a power of two!");
    const KeyT EmptyKey = getEmptyKey();
    for (BucketT *B = getBuckets(), *E = getBucketsEnd(); B != E; ++B)
      new (&B->first) KeyT(EmptyKey);
  }

  void moveFromOldBuckets(BucketT *OldBucketsBegin, BucketT *OldBucketsEnd) {
    initEmpty();

    // Insert all the old elements.
    const KeyT EmptyKey = getEmptyKey();
    const KeyT TombstoneKey = getTombstoneKey();
    for (BucketT *B = OldBucketsBegin, *E = OldBucketsEnd; B != E; ++B) {
      if (!KeyInfoT::isEqual(B->first, EmptyKey) &&
          !KeyInfoT::isEqual(B->first, TombstoneKey)) {
        // Insert the key/value into the new table.
        BucketT *DestBucket;
        bool FoundVal = LookupBucketFor(B->first, DestBucket);
        (void)FoundVal; // silence warning.
        assert(!FoundVal && "Key already in new map?");
        DestBucket->first = llvm_move(B->first);
        new (&DestBucket->second) ValueT(llvm_move(B->second));
        incrementNumEntries();

        // Free the value.
        B->second.~ValueT();
      }
      B->first.~KeyT();
    }

#ifndef NDEBUG
    if (OldBucketsBegin != OldBucketsEnd)
      memset((void*)OldBucketsBegin, 0x5a,
             sizeof(BucketT) * (OldBucketsEnd - OldBucketsBegin));
#endif
  }

  template <typename OtherBaseT>
  void copyFrom(const DenseMapBase<OtherBaseT, KeyT, ValueT, KeyInfoT>& other) {
    assert(getNumBuckets() == other.getNumBuckets());

    setNumEntries(other.getNumEntries());
    setNumTombstones(other.getNumTombstones());

    if (isPodLike<KeyT>::value && isPodLike<ValueT>::value)
      memcpy(getBuckets(), other.getBuckets(),
             getNumBuckets() * sizeof(BucketT));
    else
      for (size_t i = 0; i < getNumBuckets(); ++i) {
        new (&getBuckets()[i].first) KeyT(other.getBuckets()[i].first);
        if (!KeyInfoT::isEqual(getBuckets()[i].first, getEmptyKey()) &&
            !KeyInfoT::isEqual(getBuckets()[i].first, getTombstoneKey()))
          new (&getBuckets()[i].second) ValueT(other.getBuckets()[i].second);
      }
  }

  void swap(DenseMapBase& RHS) {
    std::swap(getNumEntries(), RHS.getNumEntries());
    std::swap(getNumTombstones(), RHS.getNumTombstones());
  }

private:
  static unsigned getHashValue(const KeyT &Val) {
    return KeyInfoT::getHashValue(Val);
  }
  template<typename LookupKeyT>
  static unsigned getHashValue(const LookupKeyT &Val) {
    return KeyInfoT::getHashValue(Val);
  }
  static const KeyT getEmptyKey() {
    return KeyInfoT::getEmptyKey();
  }
  static const KeyT getTombstoneKey() {
    return KeyInfoT::getTombstoneKey();
  }

  unsigned getNumEntries() const {
    return static_cast<const DerivedT *>(this)->getNumEntries();
  }
  void setNumEntries(unsigned Num) {
    static_cast<DerivedT *>(this)->setNumEntries(Num);
  }
  void incrementNumEntries() {
    setNumEntries(getNumEntries() + 1);
  }
  void decrementNumEntries() {
    setNumEntries(getNumEntries() - 1);
  }
  unsigned getNumTombstones() const {
    return static_cast<const DerivedT *>(this)->getNumTombstones();
  }
  void setNumTombstones(unsigned Num) {
    static_cast<DerivedT *>(this)->setNumTombstones(Num);
  }
  void incrementNumTombstones() {
    setNumTombstones(getNumTombstones() + 1);
  }
  void decrementNumTombstones() {
    setNumTombstones(getNumTombstones() - 1);
  }
  BucketT *getBuckets() const {
    return static_cast<const DerivedT *>(this)->getBuckets();
  }
  unsigned getNumBuckets() const {
    return static_cast<const DerivedT *>(this)->getNumBuckets();
  }
  BucketT *getBucketsEnd() const {
    return getBuckets() + getNumBuckets();
  }

  void grow(unsigned AtLeast) {
    static_cast<DerivedT *>(this)->grow(AtLeast);
  }

  void shrink_and_clear() {
    static_cast<DerivedT *>(this)->shrink_and_clear();
  }


  BucketT *InsertIntoBucket(const KeyT &Key, const ValueT &Value,
                            BucketT *TheBucket) {
    TheBucket = InsertIntoBucketImpl(Key, TheBucket);

    TheBucket->first = Key;
    new (&TheBucket->second) ValueT(Value);
    return TheBucket;
  }

#if LLVM_USE_RVALUE_REFERENCES
  BucketT *InsertIntoBucket(const KeyT &Key, ValueT &&Value,
                            BucketT *TheBucket) {
    TheBucket = InsertIntoBucketImpl(Key, TheBucket);

    TheBucket->first = Key;
    new (&TheBucket->second) ValueT(std::move(Value));
    return TheBucket;
  }

  BucketT *InsertIntoBucket(KeyT &&Key, ValueT &&Value, BucketT *TheBucket) {
    TheBucket = InsertIntoBucketImpl(Key, TheBucket);

    TheBucket->first = std::move(Key);
    new (&TheBucket->second) ValueT(std::move(Value));
    return TheBucket;
  }
#endif

  BucketT *InsertIntoBucketImpl(const KeyT &Key, BucketT *TheBucket) {
    // If the load of the hash table is more than 3/4, or if fewer than 1/8 of
    // the buckets are empty (meaning that many are filled with tombstones),
    // grow the table.
    //
    // The later case is tricky.  For example, if we had one empty bucket with
    // tons of tombstones, failing lookups (e.g. for insertion) would have to
    // probe almost the entire table until it found the empty bucket.  If the
    // table completely filled with tombstones, no lookup would ever succeed,
    // causing infinite loops in lookup.
    unsigned NewNumEntries = getNumEntries() + 1;
    if (NewNumEntries*4 >= getNumBuckets()*3) {
      this->grow(getNumBuckets() * 2);
      LookupBucketFor(Key, TheBucket);
    }
    if (getNumBuckets()-(NewNumEntries+getNumTombstones()) < getNumBuckets()/8) {
      this->grow(getNumBuckets());
      LookupBucketFor(Key, TheBucket);
    }

    // Only update the state after we've grown our bucket space appropriately
    // so that when growing buckets we have self-consistent entry count.
    incrementNumEntries();

    // If we are writing over a tombstone, remember this.
    if (!KeyInfoT::isEqual(TheBucket->first, getEmptyKey()))
      decrementNumTombstones();

    return TheBucket;
  }

  /// LookupBucketFor - Lookup the appropriate bucket for Val, returning it in
  /// FoundBucket.  If the bucket contains the key and a value, this returns
  /// true, otherwise it returns a bucket with an empty marker or tombstone and
  /// returns false.
  template<typename LookupKeyT>
  bool LookupBucketFor(const LookupKeyT &Val, BucketT *&FoundBucket) const {
    unsigned BucketNo = getHashValue(Val);
    unsigned ProbeAmt = 1;
    BucketT *BucketsPtr = getBuckets();

    if (getNumBuckets() == 0) {
      FoundBucket = 0;
      return false;
    }

    // FoundTombstone - Keep track of whether we find a tombstone while probing.
    BucketT *FoundTombstone = 0;
    const KeyT EmptyKey = getEmptyKey();
    const KeyT TombstoneKey = getTombstoneKey();
    assert(!KeyInfoT::isEqual(Val, EmptyKey) &&
           !KeyInfoT::isEqual(Val, TombstoneKey) &&
           "Empty/Tombstone value shouldn't be inserted into map!");

    while (1) {
      BucketT *ThisBucket = BucketsPtr + (BucketNo & (getNumBuckets()-1));
      // Found Val's bucket?  If so, return it.
      if (KeyInfoT::isEqual(Val, ThisBucket->first)) {
        FoundBucket = ThisBucket;
        return true;
      }

      // If we found an empty bucket, the key doesn't exist in the set.
      // Insert it and return the default value.
      if (KeyInfoT::isEqual(ThisBucket->first, EmptyKey)) {
        // If we've already seen a tombstone while probing, fill it in instead
        // of the empty bucket we eventually probed to.
        if (FoundTombstone) ThisBucket = FoundTombstone;
        FoundBucket = FoundTombstone ? FoundTombstone : ThisBucket;
        return false;
      }

      // If this is a tombstone, remember it.  If Val ends up not in the map, we
      // prefer to return it than something that would require more probing.
      if (KeyInfoT::isEqual(ThisBucket->first, TombstoneKey) && !FoundTombstone)
        FoundTombstone = ThisBucket;  // Remember the first tombstone found.

      // Otherwise, it's a hash collision or a tombstone, continue quadratic
      // probing.
      BucketNo += ProbeAmt++;
    }
  }

public:
  /// Return the approximate size (in bytes) of the actual map.
  /// This is just the raw memory used by DenseMap.
  /// If entries are pointers to objects, the size of the referenced objects
  /// are not included.
  size_t getMemorySize() const {
    return getNumBuckets() * sizeof(BucketT);
  }
};

template<typename KeyT, typename ValueT,
         typename KeyInfoT = DenseMapInfo<KeyT> >
class DenseMap
    : public DenseMapBase<DenseMap<KeyT, ValueT, KeyInfoT>,
                          KeyT, ValueT, KeyInfoT> {
  // Lift some types from the dependent base class into this class for
  // simplicity of referring to them.
  typedef DenseMapBase<DenseMap, KeyT, ValueT, KeyInfoT> BaseT;
  typedef typename BaseT::BucketT BucketT;
  friend class DenseMapBase<DenseMap, KeyT, ValueT, KeyInfoT>;

  BucketT *Buckets;
  unsigned NumEntries;
  unsigned NumTombstones;
  unsigned NumBuckets;

public:
  explicit DenseMap(unsigned NumInitBuckets = 0) {
    init(NumInitBuckets);
  }

  DenseMap(const DenseMap &other) {
    init(0);
    copyFrom(other);
  }

#if LLVM_USE_RVALUE_REFERENCES
  DenseMap(DenseMap &&other) {
    init(0);
    swap(other);
  }
#endif

  template<typename InputIt>
  DenseMap(const InputIt &I, const InputIt &E) {
    init(NextPowerOf2(std::distance(I, E)));
    this->insert(I, E);
  }

  ~DenseMap() {
    this->destroyAll();
    operator delete(Buckets);
  }

  void swap(DenseMap& RHS) {
    std::swap(Buckets, RHS.Buckets);
    std::swap(NumEntries, RHS.NumEntries);
    std::swap(NumTombstones, RHS.NumTombstones);
    std::swap(NumBuckets, RHS.NumBuckets);
  }

  DenseMap& operator=(const DenseMap& other) {
    copyFrom(other);
    return *this;
  }

#if LLVM_USE_RVALUE_REFERENCES
  DenseMap& operator=(DenseMap &&other) {
    this->destroyAll();
    operator delete(Buckets);
    init(0);
    swap(other);
    return *this;
  }
#endif

  void copyFrom(const DenseMap& other) {
    this->destroyAll();
    operator delete(Buckets);
    if (allocateBuckets(other.NumBuckets)) {
      this->BaseT::copyFrom(other);
    } else {
      NumEntries = 0;
      NumTombstones = 0;
    }
  }

  void init(unsigned InitBuckets) {
    if (allocateBuckets(InitBuckets)) {
      this->BaseT::initEmpty();
    } else {
      NumEntries = 0;
      NumTombstones = 0;
    }
  }

  void grow(unsigned AtLeast) {
    unsigned OldNumBuckets = NumBuckets;
    BucketT *OldBuckets = Buckets;

    allocateBuckets(std::max<unsigned>(64, NextPowerOf2(AtLeast)));
    assert(Buckets);
    if (!OldBuckets) {
      this->BaseT::initEmpty();
      return;
    }

    this->moveFromOldBuckets(OldBuckets, OldBuckets+OldNumBuckets);

    // Free the old table.
    operator delete(OldBuckets);
  }

  void shrink_and_clear() {
    unsigned OldNumEntries = NumEntries;
    this->destroyAll();

    // Reduce the number of buckets.
    unsigned NewNumBuckets
      = std::max(64, 1 << (Log2_32_Ceil(OldNumEntries) + 1));
    if (NewNumBuckets == NumBuckets) {
      this->BaseT::initEmpty();
      return;
    }

    operator delete(Buckets);
    init(NewNumBuckets);
  }

private:
  unsigned getNumEntries() const {
    return NumEntries;
  }
  void setNumEntries(unsigned Num) {
    NumEntries = Num;
  }

  unsigned getNumTombstones() const {
    return NumTombstones;
  }
  void setNumTombstones(unsigned Num) {
    NumTombstones = Num;
  }

  BucketT *getBuckets() const {
    return Buckets;
  }

  unsigned getNumBuckets() const {
    return NumBuckets;
  }

  bool allocateBuckets(unsigned Num) {
    NumBuckets = Num;
    if (NumBuckets == 0) {
      Buckets = 0;
      return false;
    }

    Buckets = static_cast<BucketT*>(operator new(sizeof(BucketT) * NumBuckets));
    return true;
  }
};

template<typename KeyT, typename ValueT,
         typename KeyInfoT, bool IsConst>
class DenseMapIterator {
  typedef std::pair<KeyT, ValueT> Bucket;
  typedef DenseMapIterator<KeyT, ValueT,
                           KeyInfoT, true> ConstIterator;
  friend class DenseMapIterator<KeyT, ValueT, KeyInfoT, true>;
public:
  typedef ptrdiff_t difference_type;
  typedef typename conditional<IsConst, const Bucket, Bucket>::type value_type;
  typedef value_type *pointer;
  typedef value_type &reference;
  typedef std::forward_iterator_tag iterator_category;
private:
  pointer Ptr, End;
public:
  DenseMapIterator() : Ptr(0), End(0) {}

  DenseMapIterator(pointer Pos, pointer E, bool NoAdvance = false)
    : Ptr(Pos), End(E) {
    if (!NoAdvance) AdvancePastEmptyBuckets();
  }

  // If IsConst is true this is a converting constructor from iterator to
  // const_iterator and the default copy constructor is used.
  // Otherwise this is a copy constructor for iterator.
  DenseMapIterator(const DenseMapIterator<KeyT, ValueT,
                                          KeyInfoT, false>& I)
    : Ptr(I.Ptr), End(I.End) {}

  reference operator*() const {
    return *Ptr;
  }
  pointer operator->() const {
    return Ptr;
  }

  bool operator==(const ConstIterator &RHS) const {
    return Ptr == RHS.operator->();
  }
  bool operator!=(const ConstIterator &RHS) const {
    return Ptr != RHS.operator->();
  }

  inline DenseMapIterator& operator++() {  // Preincrement
    ++Ptr;
    AdvancePastEmptyBuckets();
    return *this;
  }
  DenseMapIterator operator++(int) {  // Postincrement
    DenseMapIterator tmp = *this; ++*this; return tmp;
  }

private:
  void AdvancePastEmptyBuckets() {
    const KeyT Empty = KeyInfoT::getEmptyKey();
    const KeyT Tombstone = KeyInfoT::getTombstoneKey();

    while (Ptr != End &&
           (KeyInfoT::isEqual(Ptr->first, Empty) ||
            KeyInfoT::isEqual(Ptr->first, Tombstone)))
      ++Ptr;
  }
};
  
template<typename KeyT, typename ValueT, typename KeyInfoT>
static inline size_t
capacity_in_bytes(const DenseMap<KeyT, ValueT, KeyInfoT> &X) {
  return X.getMemorySize();
}

} // end namespace llvm

#endif
