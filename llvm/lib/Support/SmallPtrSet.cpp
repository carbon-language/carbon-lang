//===- llvm/ADT/SmallPtrSet.cpp - 'Normally small' pointer set ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the SmallPtrSet class.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallPtrSet.h"
using namespace llvm;

bool SmallPtrSetImpl::insert(void *Ptr) {
  if (isSmall()) {
    // Check to see if it is already in the set.
    for (void **APtr = SmallArray, **E = SmallArray+NumElements;
         APtr != E; ++APtr)
      if (*APtr == Ptr)
        return false;
    
    // Nope, there isn't.  If we stay small, just 'pushback' now.
    if (NumElements < CurArraySize-1) {
      SmallArray[NumElements++] = Ptr;
      return true;
    }
    // Otherwise, hit the big set case, which will call grow.
  }
  
  // If more than 3/4 of the array is full, grow.
  if (NumElements*4 >= CurArraySize*3)
    Grow();
  
  // Okay, we know we have space.  Find a hash bucket.
  void **Bucket = const_cast<void**>(FindBucketFor(Ptr));
  if (*Bucket == Ptr) return false; // Already inserted, good.
  
  // Otherwise, insert it!
  *Bucket = Ptr;
  ++NumElements;  // Track density.
  return true;
}

void * const *SmallPtrSetImpl::FindBucketFor(void *Ptr) const {
  unsigned Bucket = Hash(Ptr);
  unsigned ArraySize = CurArraySize;
  unsigned ProbeAmt = 1;
  void *const *Array = CurArray;
  void *const *Tombstone = 0;
  while (1) {
    // Found Ptr's bucket?
    if (Array[Bucket] == Ptr)
      return Array+Bucket;
    
    // If we found an empty bucket, the pointer doesn't exist in the set.
    // Return a tombstone if we've seen one so far, or the empty bucket if
    // not.
    if (Array[Bucket] == getEmptyMarker())
      return Tombstone ? Tombstone : Array+Bucket;
    
    // If this is a tombstone, remember it.  If Ptr ends up not in the set, we
    // prefer to return it than something that would require more probing.
    if (Array[Bucket] == getTombstoneMarker() && !Tombstone)
      Tombstone = Array+Bucket;  // Remember the first tombstone found.
    
    // It's a hash collision or a tombstone. Reprobe.
    Bucket = (Bucket + ProbeAmt++) & (ArraySize-1);
  }
}

/// Grow - Allocate a larger backing store for the buckets and move it over.
///
void SmallPtrSetImpl::Grow() {
  // Allocate at twice as many buckets, but at least 128.
  unsigned OldSize = CurArraySize;
  unsigned NewSize = OldSize < 64 ? 128 : OldSize*2;
  
  void **OldBuckets = CurArray;
  bool WasSmall = isSmall();
  
  // Install the new array.  Clear all the buckets to empty.
  CurArray = new void*[NewSize+1];
  CurArraySize = NewSize;
  memset(CurArray, -1, NewSize*sizeof(void*));
  
  // The end pointer, always valid, is set to a valid element to help the
  // iterator.
  CurArray[NewSize] = 0;
  
  // Copy over all the elements.
  if (WasSmall) {
    // Small sets store their elements in order.
    for (void **BucketPtr = OldBuckets, **E = OldBuckets+NumElements;
         BucketPtr != E; ++BucketPtr) {
      void *Elt = *BucketPtr;
      *const_cast<void**>(FindBucketFor(Elt)) = Elt;
    }
  } else {
    // Copy over all valid entries.
    for (void **BucketPtr = OldBuckets, **E = OldBuckets+OldSize;
         BucketPtr != E; ++BucketPtr) {
      // Copy over the element if it is valid.
      void *Elt = *BucketPtr;
      if (Elt != getTombstoneMarker() && Elt != getEmptyMarker())
        *const_cast<void**>(FindBucketFor(Elt)) = Elt;
    }
    
    delete [] OldBuckets;
  }
}
