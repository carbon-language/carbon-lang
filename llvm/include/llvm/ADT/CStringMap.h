//===--- CStringMap.h - CString Hash table map interface --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the CStringMap class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_CSTRINGMAP_H
#define LLVM_ADT_CSTRINGMAP_H

#include "llvm/Support/Allocator.h"
#include <cstring>

namespace llvm {
  
/// CStringMapVisitor - Subclasses of this class may be implemented to walk all
/// of the items in a CStringMap.
class CStringMapVisitor {
public:
  virtual ~CStringMapVisitor();
  virtual void Visit(const char *Key, void *Value) const = 0;
};
  
/// CStringMapImpl - This is the base class of CStringMap that is shared among
/// all of its instantiations.
class CStringMapImpl {
protected:
  /// ItemBucket - The hash table consists of an array of these.  If Item is
  /// non-null, this is an extant entry, otherwise, it is a hole.
  struct ItemBucket {
    /// FullHashValue - This remembers the full hash value of the key for
    /// easy scanning.
    unsigned FullHashValue;
    
    /// Item - This is a pointer to the actual item object.
    void *Item;
  };
  
  ItemBucket *TheTable;
  unsigned NumBuckets;
  unsigned NumItems;
  unsigned ItemSize;
protected:
  CStringMapImpl(unsigned InitSize, unsigned ItemSize);
  void RehashTable();
  
  /// LookupBucketFor - Look up the bucket that the specified string should end
  /// up in.  If it already exists as a key in the map, the Item pointer for the
  /// specified bucket will be non-null.  Otherwise, it will be null.  In either
  /// case, the FullHashValue field of the bucket will be set to the hash value
  /// of the string.
  unsigned LookupBucketFor(const char *KeyStart, const char *KeyEnd);
  
public:
  unsigned getNumBuckets() const { return NumBuckets; }
  unsigned getNumItems() const { return NumItems; }

  void VisitEntries(const CStringMapVisitor &Visitor) const;
};


/// CStringMap - This is an unconventional map that is specialized for handling
/// keys that are "C strings", that is, null-terminated strings.  This does some
/// funky memory allocation and hashing things to make it extremely efficient,
/// storing the string data *after* the value in the map.
template<typename ValueTy, typename AllocatorTy = MallocAllocator>
class CStringMap : public CStringMapImpl {
  AllocatorTy Allocator;
public:
  CStringMap(unsigned InitialSize = 0)
    : CStringMapImpl(InitialSize, sizeof(ValueTy)) {}
  
  AllocatorTy &getAllocator() { return Allocator; }
  const AllocatorTy &getAllocator() const { return Allocator; }

  /// FindValue - Look up the specified key in the map.  If it exists, return a
  /// pointer to the element, otherwise return null.
  ValueTy *FindValue(const char *KeyStart, const char *KeyEnd) {
    unsigned BucketNo = LookupBucketFor(KeyStart, KeyEnd);
    return static_cast<ValueTy*>(TheTable[BucketNo].Item);
  }
  
  /// GetKeyForValueInMap - Given a value that is inserted into this map, return
  /// the string that corresponds to it.  This is an efficient operation that
  /// is provided by CStringMap.  The string is live as long as the value is in
  /// the map.
  static const char *GetKeyForValueInMap(const ValueTy &Val) {
    return reinterpret_cast<const char*>(&Val+1);
  }
  
  /// GetOrCreateValue - Look up the specified key in the table.  If a value
  /// exists, return it.  Otherwise, default construct a value, insert it, and
  /// return.
  ValueTy &GetOrCreateValue(const char *KeyStart, const char *KeyEnd) {
    unsigned BucketNo = LookupBucketFor(KeyStart, KeyEnd);
    ItemBucket &Bucket = TheTable[BucketNo];
    if (Bucket.Item)
      return *static_cast<ValueTy*>(Bucket.Item);
    
    unsigned KeyLength = KeyEnd-KeyStart;
    
    // Okay, the item doesn't already exist, and Bucket is the bucket to fill
    // in.  Allocate a new item with space for the null-terminated string at the
    // end.
    unsigned AllocSize = sizeof(ValueTy)+KeyLength+1;
    
#ifdef __GNUC__
    unsigned Alignment = __alignof__(ValueTy);
#else
    // FIXME: ugly.
    unsigned Alignment = 8;
#endif
    ValueTy *NewItem = (ValueTy*)Allocator.Allocate(AllocSize, Alignment);
    new (NewItem) ValueTy();
    ++NumItems;
    
    // Copy the string information.
    char *StrBuffer = (char*)(NewItem+1);
    memcpy(StrBuffer, KeyStart, KeyLength);
    StrBuffer[KeyLength] = 0;  // Null terminate string.
    
    // Fill in the bucket for the hash table.  The FullHashValue was already
    // filled in by LookupBucketFor.
    Bucket.Item = NewItem;
    
    // If the hash table is now more than 3/4 full, rehash into a larger table.
    if (NumItems > NumBuckets*3/4)
      RehashTable();
    return *NewItem;
  }
  
  ~CStringMap() {
    for (ItemBucket *I = TheTable, *E = TheTable+NumBuckets; I != E; ++I) {
      if (ValueTy *Id = static_cast<ValueTy*>(I->Item)) {
        // Free memory referenced by the item.
        Id->~ValueTy();
        Allocator.Deallocate(Id);
      }
    }
    delete [] TheTable;
  }
};
  
}

#endif

