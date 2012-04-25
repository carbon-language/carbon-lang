//===- llvm/ADT/FlatArrayMap.h - 'Normally small' pointer set ----*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the FlatArrayMap class.
// See FlatArrayMap doxygen comments for more details.
//
//===----------------------------------------------------------------------===//

#ifndef FLATARRAYMAP_H_
#define FLATARRAYMAP_H_

#include <algorithm>
#include <utility>
#include "llvm/Support/type_traits.h"

namespace llvm {
  
  template <typename KeyTy, typename MappedTy> 
  struct FlatArrayMapTypes {
    typedef KeyTy key_type;
    typedef MappedTy mapped_type;
    typedef typename std::pair<key_type, mapped_type> value_type;
  };
  
  template<typename KeyTy, typename MappedTy, bool IsConst = false>
  class FlatArrayMapIterator;  
  
  //===--------------------------------------------------------------------===//
  /// FlatArrayMap presents map container interface.
  /// It uses flat array implementation inside:
  /// [ <key0, value0>, <key1, value1>, ... <keyN, valueN> ]
  /// It works fast for small amount of elements.
  /// User should pass key type, mapped type (type of value), and maximum
  /// number of elements.
  /// After maximum number of elements is reached, map declines any farther
  /// attempts to insert new elements ("insert" method returns <end(),false>).
  ///
  template <typename KeyTy, typename MappedTy, unsigned MaxArraySize>
  class FlatArrayMap {
  public:
    typedef FlatArrayMapTypes<KeyTy, MappedTy> Types;
    
    typedef typename Types::key_type key_type;
    typedef typename Types::mapped_type mapped_type;
    typedef typename Types::value_type value_type;
    
    typedef FlatArrayMapIterator<KeyTy, MappedTy> iterator;
    typedef FlatArrayMapIterator<KeyTy, MappedTy, true> const_iterator; 
    
    typedef FlatArrayMap<KeyTy, MappedTy, MaxArraySize> self;
    
  private:
    
    enum { BadIndex = ~0UL };
    
    key_type EmptyKey;
    mapped_type EmptyValue;
    
    value_type Array[MaxArraySize + 1];
    unsigned NumElements;
   
  unsigned findFor(const KeyTy Ptr) const {
    // Linear search for the item.
    for (const value_type *APtr = Array, *E = Array + NumElements;
               APtr != E; ++APtr) {
      if (APtr->first == Ptr) {
        return APtr - Array;
      }
    }
    return BadIndex;
  } 
  
  bool lookupFor(const KeyTy &Ptr, const value_type*& Found) const {
    unsigned FoundIdx = findFor(Ptr);
    if (FoundIdx != BadIndex) {
      Found = Array + FoundIdx;
      return true;
    }
    return false;
  }
  
  bool lookupFor(const KeyTy &Ptr, value_type*& Found) {
    unsigned FoundIdx = findFor(Ptr);
    if (FoundIdx != BadIndex) {
      Found = Array + FoundIdx;
      return true;
    }
    return false;
  }  
  
  
  void copyFrom(const self &RHS) {
    memcpy(Array, RHS.Array, sizeof(value_type) * (MaxArraySize + 1));
    NumElements = RHS.NumElements;
  }   
   
  void init () {
    memset(Array + MaxArraySize, 0, sizeof(value_type));
    NumElements = 0;     
  }  
  
  bool insertInternal(KeyTy Ptr, MappedTy Val, value_type*& Item) {
    // Check to see if it is already in the set.
    value_type *Found;
    if (lookupFor(Ptr, Found)) {
      Item = Found;
      return false;
    }
    if (NumElements < MaxArraySize) {
      unsigned Idx = NumElements++;
      Array[Idx] = std::make_pair(Ptr, Val);
      Item = Array + Idx;
      return true;
    }
    Item = Array + MaxArraySize; // return end()
    return false;
  }
  
  public:
  
    // Constructors
  
    FlatArrayMap() : EmptyKey(), EmptyValue() {
      init();
    }
    
    FlatArrayMap(const self &that) :
      EmptyKey(), EmptyValue() {
      copyFrom(that);
    }
  
    template<typename It>
    FlatArrayMap(It I, It E) :
      EmptyKey(), EmptyValue() {
      init();
      insert(I, E);
    }
    
    // Size
    
    unsigned size() const {
      return NumElements;
    }
    
    bool empty() const {
      return !NumElements;
    }
    
    // Iterators
    
    iterator begin() {
      return iterator(Array);
    }
    const_iterator begin() const {
      return const_iterator(Array);
    }

    iterator end() {
      return iterator(Array + MaxArraySize);
    }
    const_iterator end() const {
      return const_iterator(Array + MaxArraySize);
    }
    
    // Modifiers
    
    void clear() {
      for (unsigned i = 0; i < NumElements; ++i) {
        Array[i].first = EmptyKey;
        Array[i].second = EmptyValue;
      }
      NumElements = 0;
    }
    
    // The map container is extended by inserting a single new element.
    // The behavior is the same as the std::map::insert, except the
    // case when maximum number of elements is reached;
    // in this case map declines any farther attempts
    // to insert new elements ("insert" method returns <end(),false>).    
    std::pair<iterator, bool> insert(const value_type& KV) {
      value_type* Item;
      bool Res = insertInternal(KV.first, KV.second, Item);
      return std::make_pair(iterator(Item), Res);
    }
    
    template <typename IterT>
    void insert(IterT I, IterT E) {
      for (; I != E; ++I)
        insert(*I);
    } 
    
    void erase(key_type K) {
      unsigned Found = findFor(K);
      if (Found != BadIndex) {
        value_type *APtr = Array + Found;
        value_type *E = Array + NumElements;
        *APtr = E[-1];
        E[-1].first.~key_type();
        E[-1].second.~mapped_type();
        --NumElements;
      }
    }
        
    void erase(iterator i) {
      erase(i->first);
    }
    
    void swap(self& RHS) {
      std::swap_ranges(Array, Array+MaxArraySize,  RHS.Array);
      std::swap(this->NumElements, RHS.NumElements);
    }
    
    // Search operations
    
    iterator find(const key_type& K) {
      value_type *Found;
      if (lookupFor(K, Found))
        return iterator(Found);
      return end();
    }
    
    const_iterator find(const key_type& K) const {
      const value_type *Found;
      if (lookupFor(K, Found))
        return const_iterator(Found);
      return end();
    }
    
    bool count(const key_type& K) const {
      return find(K) != end();
    }
    
    mapped_type &operator[](const key_type &Key) {
      std::pair<iterator, bool> res = insert(Key, mapped_type());
      return res.first->second;
    }

    // Other operations
    
    self& operator=(const self& other) {
      clear();
      copyFrom(other);
      return *this;
    }       
    
    /// isPointerIntoBucketsArray - Return true if the specified pointer points
    /// somewhere into the map's array of buckets (i.e. either to a key or
    /// value).
    bool isPointerIntoBucketsArray(const void *Ptr) const {
      return Ptr >= Array && Ptr < Array + NumElements;
    }

    /// getPointerIntoBucketsArray() - Return an opaque pointer into the buckets
    /// array.
    const void *getPointerIntoBucketsArray() const { return Array; }    
  };
  
  template<typename KeyTy, typename MappedTy, bool IsConst>
  class FlatArrayMapIterator {
    
    typedef FlatArrayMapTypes<KeyTy, MappedTy> Types;
    
    typedef typename conditional<IsConst,
                                 const typename Types::value_type,
                                 typename Types::value_type>::type value_type;
    typedef value_type *pointer;
    typedef value_type &reference;
    
    typedef FlatArrayMapIterator<KeyTy, MappedTy, IsConst> self;
    typedef FlatArrayMapIterator<KeyTy, MappedTy, false> non_const_self;
    typedef FlatArrayMapIterator<KeyTy, MappedTy, true> const_self;    

    friend class FlatArrayMapIterator<KeyTy, MappedTy, false>;
    friend class FlatArrayMapIterator<KeyTy, MappedTy, true>;    
  
    pointer TheBucket;
    
  public:
    
    FlatArrayMapIterator() : TheBucket(0) {}
    
    explicit FlatArrayMapIterator(pointer BP) :
        TheBucket(BP) {}
    
    // If IsConst is true this is a converting constructor from iterator to
    // const_iterator and the default copy constructor is used.
    // Otherwise this is a copy constructor for iterator.
    FlatArrayMapIterator(const non_const_self& I)
      : TheBucket(I.TheBucket) {}  
    
    bool operator==(const const_self &RHS) const {
      return TheBucket->first == RHS.TheBucket->first;
    }
    bool operator!=(const const_self &RHS) const {
      return TheBucket->first != RHS.TheBucket->first;
    }
  
    reference operator*() const {
      return *TheBucket;
    }  
  
    pointer operator->() const {
      return TheBucket;
    }  
    
    inline self& operator++() {   // Preincrement
      ++TheBucket;
      return *this;
    }
  
    self operator++(int) {        // Postincrement
      FlatArrayMapIterator tmp = *this; ++*this; return tmp;
    }
  };
}

#endif /* FLATARRAYMAP_H_ */
