//===--- StringMap.cpp - String Hash table map implementation -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the StringMap class.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringMap.h"
#include <cassert>
using namespace llvm;

StringMapVisitor::~StringMapVisitor() {
}

StringMapImpl::StringMapImpl(unsigned InitSize, unsigned itemSize) {
  assert((InitSize & (InitSize-1)) == 0 &&
         "Init Size must be a power of 2 or zero!");
  NumBuckets = InitSize ? InitSize : 512;
  ItemSize = itemSize;
  NumItems = 0;
  
  TheTable = new ItemBucket[NumBuckets+1]();
  memset(TheTable, 0, NumBuckets*sizeof(ItemBucket));
  
  // Allocate one extra bucket, set it to look filled so the iterators stop at
  // end.
  TheTable[NumBuckets].Item = (StringMapEntryBase*)2;
}


/// HashString - Compute a hash code for the specified string.
///
static unsigned HashString(const char *Start, const char *End) {
  // Bernstein hash function.
  unsigned int Result = 0;
  // TODO: investigate whether a modified bernstein hash function performs
  // better: http://eternallyconfuzzled.com/tuts/algorithms/jsw_tut_hashing.aspx
  //   X*33+c -> X*33^c
  while (Start != End)
    Result = Result * 33 + *Start++;
  Result = Result + (Result >> 5);
  return Result;
}

/// LookupBucketFor - Look up the bucket that the specified string should end
/// up in.  If it already exists as a key in the map, the Item pointer for the
/// specified bucket will be non-null.  Otherwise, it will be null.  In either
/// case, the FullHashValue field of the bucket will be set to the hash value
/// of the string.
unsigned StringMapImpl::LookupBucketFor(const char *NameStart,
                                         const char *NameEnd) {
  unsigned HTSize = NumBuckets;
  unsigned FullHashValue = HashString(NameStart, NameEnd);
  unsigned BucketNo = FullHashValue & (HTSize-1);
  
  unsigned ProbeAmt = 1;
  while (1) {
    ItemBucket &Bucket = TheTable[BucketNo];
    StringMapEntryBase *BucketItem = Bucket.Item;
    // If we found an empty bucket, this key isn't in the table yet, return it.
    if (BucketItem == 0) {
      Bucket.FullHashValue = FullHashValue;
      return BucketNo;
    }
    
    // If the full hash value matches, check deeply for a match.  The common
    // case here is that we are only looking at the buckets (for item info
    // being non-null and for the full hash value) not at the items.  This
    // is important for cache locality.
    if (Bucket.FullHashValue == FullHashValue) {
      // Do the comparison like this because NameStart isn't necessarily
      // null-terminated!
      char *ItemStr = (char*)BucketItem+ItemSize;
      unsigned ItemStrLen = BucketItem->getKeyLength();
      if (unsigned(NameEnd-NameStart) == ItemStrLen &&
          memcmp(ItemStr, NameStart, ItemStrLen) == 0) {
        // We found a match!
        return BucketNo;
      }
    }
    
    // Okay, we didn't find the item.  Probe to the next bucket.
    BucketNo = (BucketNo+ProbeAmt) & (HTSize-1);
    
    // Use quadratic probing, it has fewer clumping artifacts than linear
    // probing and has good cache behavior in the common case.
    ++ProbeAmt;
  }
}

/// RehashTable - Grow the table, redistributing values into the buckets with
/// the appropriate mod-of-hashtable-size.
void StringMapImpl::RehashTable() {
  unsigned NewSize = NumBuckets*2;
  // Allocate one extra bucket which will always be non-empty.  This allows the
  // iterators to stop at end.
  ItemBucket *NewTableArray = new ItemBucket[NewSize+1]();
  memset(NewTableArray, 0, NewSize*sizeof(ItemBucket));
  NewTableArray[NewSize].Item = (StringMapEntryBase*)2;
  
  // Rehash all the items into their new buckets.  Luckily :) we already have
  // the hash values available, so we don't have to rehash any strings.
  for (ItemBucket *IB = TheTable, *E = TheTable+NumBuckets; IB != E; ++IB) {
    if (IB->Item) {
      // Fast case, bucket available.
      unsigned FullHash = IB->FullHashValue;
      unsigned NewBucket = FullHash & (NewSize-1);
      if (NewTableArray[NewBucket].Item == 0) {
        NewTableArray[FullHash & (NewSize-1)].Item = IB->Item;
        NewTableArray[FullHash & (NewSize-1)].FullHashValue = FullHash;
        continue;
      }
      
      unsigned ProbeSize = 1;
      do {
        NewBucket = (NewBucket + ProbeSize++) & (NewSize-1);
      } while (NewTableArray[NewBucket].Item);
      
      // Finally found a slot.  Fill it in.
      NewTableArray[NewBucket].Item = IB->Item;
      NewTableArray[NewBucket].FullHashValue = FullHash;
    }
  }
  
  delete[] TheTable;
  
  TheTable = NewTableArray;
  NumBuckets = NewSize;
}


/// VisitEntries - This method walks through all of the items,
/// invoking Visitor.Visit for each of them.
void StringMapImpl::VisitEntries(const StringMapVisitor &Visitor) const {
  for (ItemBucket *IB = TheTable, *E = TheTable+NumBuckets; IB != E; ++IB) {
    if (StringMapEntryBase *Id = IB->Item)
      Visitor.Visit((char*)Id + ItemSize, Id);
  }
}
