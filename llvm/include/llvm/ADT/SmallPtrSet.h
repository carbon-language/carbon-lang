//===- llvm/ADT/SmallPtrSet.h - 'Normally small' pointer set ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the SmallPtrSet class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_SMALLPTRSET_H
#define LLVM_ADT_SMALLPTRSET_H

#include <cassert>
#include <cstring>

namespace llvm {

class SmallPtrSetImpl {
protected:
  /// CurArray - This is the current set of buckets.  If it points to
  /// SmallArray, then the set is in 'small mode'.
  void **CurArray;
  /// CurArraySize - The allocated size of CurArray, always a power of two.
  /// Note that CurArray points to an array that has CurArraySize+1 elements in
  /// it, so that the end iterator actually points to valid memory.
  unsigned CurArraySize;
  
  // If small, this is # elts allocated consequtively
  unsigned NumElements;
  void *SmallArray[1];  // Must be last ivar.
public:
  SmallPtrSetImpl(unsigned SmallSize) {
    assert(SmallSize && (SmallSize & (SmallSize-1)) == 0 &&
           "Initial size must be a power of two!");
    CurArray = &SmallArray[0];
    CurArraySize = SmallSize;
    // The end pointer, always valid, is set to a valid element to help the
    // iterator.
    CurArray[SmallSize] = 0;
    clear();
  }
  ~SmallPtrSetImpl() {
    if (!isSmall())
      delete[] CurArray;
  }
  
  bool isSmall() const { return CurArray == &SmallArray[0]; }

  static void *getTombstoneMarker() { return reinterpret_cast<void*>(-2); }
  static void *getEmptyMarker() {
    // Note that -1 is chosen to make clear() efficiently implementable with
    // memset and because it's not a valid pointer value.
    return reinterpret_cast<void*>(-1);
  }
  
  void clear() {
    // Fill the array with empty markers.
    memset(CurArray, -1, CurArraySize*sizeof(void*));
    NumElements = 0;
  }
  
  /// insert - This returns true if the pointer was new to the set, false if it
  /// was already in the set.
  bool insert(void *Ptr);
  
  bool count(void *Ptr) const {
    if (isSmall()) {
      // Linear search for the item.
      for (void *const *APtr = SmallArray, *const *E = SmallArray+NumElements;
           APtr != E; ++APtr)
        if (*APtr == Ptr)
          return true;
      return false;
    }
    
    // Big set case.
    return *FindBucketFor(Ptr) == Ptr;
  }
  
private:
  unsigned Hash(void *Ptr) const {
    return ((uintptr_t)Ptr >> 4) & (CurArraySize-1);
  }
  void * const *FindBucketFor(void *Ptr) const;
  
  /// Grow - Allocate a larger backing store for the buckets and move it over.
  void Grow();
};

/// SmallPtrSetIteratorImpl - This is the common base class shared between all
/// instances of SmallPtrSetIterator.
class SmallPtrSetIteratorImpl {
protected:
  void *const *Bucket;
public:
  SmallPtrSetIteratorImpl(void *const *BP) : Bucket(BP) {
    AdvanceIfNotValid();
  }
  
  bool operator==(const SmallPtrSetIteratorImpl &RHS) const {
    return Bucket == RHS.Bucket;
  }
  bool operator!=(const SmallPtrSetIteratorImpl &RHS) const {
    return Bucket != RHS.Bucket;
  }
  
protected:
  /// AdvanceIfNotValid - If the current bucket isn't valid, advance to a bucket
  /// that is.   This is guaranteed to stop because the end() bucket is marked
  /// valid.
  void AdvanceIfNotValid() {
    while (*Bucket == SmallPtrSetImpl::getEmptyMarker() ||
           *Bucket == SmallPtrSetImpl::getTombstoneMarker())
      ++Bucket;
  }
};

/// SmallPtrSetIterator - This implements a const_iterator for SmallPtrSet.
template<typename PtrTy>
class SmallPtrSetIterator : public SmallPtrSetIteratorImpl {
public:
  SmallPtrSetIterator(void *const *BP) : SmallPtrSetIteratorImpl(BP) {}

  // Most methods provided by baseclass.
  
  PtrTy operator*() const {
    return static_cast<PtrTy>(*Bucket);
  }
  
  inline SmallPtrSetIterator& operator++() {          // Preincrement
    ++Bucket;
    AdvanceIfNotValid();
    return *this;
  }
  
  SmallPtrSetIterator operator++(int) {        // Postincrement
    SmallPtrSetIterator tmp = *this; ++*this; return tmp;
  }
};


/// SmallPtrSet - This class implements 
template<class PtrType, unsigned SmallSize>
class SmallPtrSet : public SmallPtrSetImpl {
  void *SmallArray[SmallSize];
public:
  SmallPtrSet() : SmallPtrSetImpl(SmallSize) {}
  
  typedef SmallPtrSetIterator<PtrType> iterator;
  typedef SmallPtrSetIterator<PtrType> const_iterator;
  inline iterator begin() const {
    return iterator(CurArray);
  }
  inline iterator end() const {
    return iterator(CurArray+CurArraySize);
  }
};

}

#endif
