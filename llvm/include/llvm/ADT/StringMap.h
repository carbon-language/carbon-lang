//===--- StringMap.h - String Hash table map interface ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the StringMap class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_STRINGMAP_H
#define LLVM_ADT_STRINGMAP_H

#include "llvm/Support/Allocator.h"
#include <cstring>

namespace llvm {
  template<typename ValueT>
  class StringMapConstIterator;
  template<typename ValueT>
  class StringMapIterator;

  
/// StringMapEntryBase - Shared base class of StringMapEntry instances.
class StringMapEntryBase {
  unsigned StrLen;
public:
  StringMapEntryBase(unsigned Len) : StrLen(Len) {}
  
  unsigned getKeyLength() const { return StrLen; }
};
  
/// StringMapImpl - This is the base class of StringMap that is shared among
/// all of its instantiations.
class StringMapImpl {
public:
  /// ItemBucket - The hash table consists of an array of these.  If Item is
  /// non-null, this is an extant entry, otherwise, it is a hole.
  struct ItemBucket {
    /// FullHashValue - This remembers the full hash value of the key for
    /// easy scanning.
    unsigned FullHashValue;
    
    /// Item - This is a pointer to the actual item object.
    StringMapEntryBase *Item;
  };
  
protected:
  ItemBucket *TheTable;
  unsigned NumBuckets;
  unsigned NumItems;
  unsigned ItemSize;
protected:
  StringMapImpl(unsigned InitSize, unsigned ItemSize);
  void RehashTable();
  
  /// LookupBucketFor - Look up the bucket that the specified string should end
  /// up in.  If it already exists as a key in the map, the Item pointer for the
  /// specified bucket will be non-null.  Otherwise, it will be null.  In either
  /// case, the FullHashValue field of the bucket will be set to the hash value
  /// of the string.
  unsigned LookupBucketFor(const char *KeyStart, const char *KeyEnd);
  
  /// FindKey - Look up the bucket that contains the specified key. If it exists
  /// in the map, return the bucket number of the key.  Otherwise return -1.
  /// This does not modify the map.
  int FindKey(const char *KeyStart, const char *KeyEnd) const;
  
public:
  static StringMapEntryBase *getTombstoneVal() {
    return (StringMapEntryBase*)-1;
  }
  
  unsigned getNumBuckets() const { return NumBuckets; }
  unsigned getNumItems() const { return NumItems; }

  bool empty() const { return NumItems == 0; }
  unsigned size() const { return NumItems; }
};

/// StringMapEntry - This is used to represent one value that is inserted into
/// a StringMap.  It contains the Value itself and the key: the string length
/// and data.
template<typename ValueTy>
class StringMapEntry : public StringMapEntryBase {
  ValueTy Val;
public:
  StringMapEntry(unsigned StrLen)
    : StringMapEntryBase(StrLen), Val() {}
  StringMapEntry(unsigned StrLen, const ValueTy &V)
    : StringMapEntryBase(StrLen), Val(V) {}

  const ValueTy &getValue() const { return Val; }
  ValueTy &getValue() { return Val; }
  
  void setValue(const ValueTy &V) { Val = V; }
  
  /// getKeyData - Return the start of the string data that is the key for this
  /// value.  The string data is always stored immediately after the
  /// StringMapEntry object.
  const char *getKeyData() const {return reinterpret_cast<const char*>(this+1);}
  
  /// Create - Create a StringMapEntry for the specified key and default
  /// construct the value.
  template<typename AllocatorTy>
  static StringMapEntry *Create(const char *KeyStart, const char *KeyEnd,
                                AllocatorTy &Allocator) {
    unsigned KeyLength = KeyEnd-KeyStart;
    
    // Okay, the item doesn't already exist, and 'Bucket' is the bucket to fill
    // in.  Allocate a new item with space for the string at the end and a null
    // terminator.
    unsigned AllocSize = sizeof(StringMapEntry)+KeyLength+1;
    
#ifdef __GNUC__
    unsigned Alignment = __alignof__(StringMapEntry);
#else
    // FIXME: ugly.
    unsigned Alignment = 8;
#endif
    StringMapEntry *NewItem = 
      static_cast<StringMapEntry*>(Allocator.Allocate(AllocSize, Alignment));
    
    // Default construct the value.
    new (NewItem) StringMapEntry(KeyLength);
    
    // Copy the string information.
    char *StrBuffer = const_cast<char*>(NewItem->getKeyData());
    memcpy(StrBuffer, KeyStart, KeyLength);
    StrBuffer[KeyLength] = 0;  // Null terminate for convenience of clients.
    return NewItem;
  }
  
  /// Create - Create a StringMapEntry with normal malloc/free.
  static StringMapEntry *Create(const char *KeyStart, const char *KeyEnd) {
    MallocAllocator A;
    return Create(KeyStart, KeyEnd, A);
  }

  /// Destroy - Destroy this StringMapEntry, releasing memory back to the
  /// specified allocator.
  template<typename AllocatorTy>
  void Destroy(AllocatorTy &Allocator) {
    // Free memory referenced by the item.
    this->~StringMapEntry();
    Allocator.Deallocate(this);
  }
  
  /// Destroy this object, releasing memory back to the malloc allocator.
  void Destroy() {
    MallocAllocator A;
    Destroy(A);
  }
};


/// StringMap - This is an unconventional map that is specialized for handling
/// keys that are "strings", which are basically ranges of bytes. This does some
/// funky memory allocation and hashing things to make it extremely efficient,
/// storing the string data *after* the value in the map.
template<typename ValueTy, typename AllocatorTy = MallocAllocator>
class StringMap : public StringMapImpl {
  AllocatorTy Allocator;
  typedef StringMapEntry<ValueTy> MapEntryTy;
public:
  StringMap(unsigned InitialSize = 0)
    : StringMapImpl(InitialSize, sizeof(MapEntryTy)) {}
  
  AllocatorTy &getAllocator() { return Allocator; }
  const AllocatorTy &getAllocator() const { return Allocator; }

  typedef StringMapConstIterator<ValueTy> const_iterator;
  typedef StringMapIterator<ValueTy> iterator;
  
  iterator begin() { return iterator(TheTable); }
  iterator end() { return iterator(TheTable+NumBuckets); }
  const_iterator begin() const { return const_iterator(TheTable); }
  const_iterator end() const { return const_iterator(TheTable+NumBuckets); }
  
  
  iterator find(const char *KeyStart, const char *KeyEnd) {
    int Bucket = FindKey(KeyStart, KeyEnd);
    if (Bucket == -1) return end();
    return iterator(TheTable+Bucket);
  }

  const_iterator find(const char *KeyStart, const char *KeyEnd) const {
    int Bucket = FindKey(KeyStart, KeyEnd);
    if (Bucket == -1) return end();
    return const_iterator(TheTable+Bucket);
  }
  
  /// GetOrCreateValue - Look up the specified key in the table.  If a value
  /// exists, return it.  Otherwise, default construct a value, insert it, and
  /// return.
  StringMapEntry<ValueTy> &GetOrCreateValue(const char *KeyStart, 
                                            const char *KeyEnd) {
    unsigned BucketNo = LookupBucketFor(KeyStart, KeyEnd);
    ItemBucket &Bucket = TheTable[BucketNo];
    if (Bucket.Item)
      return *static_cast<MapEntryTy*>(Bucket.Item);
    
    MapEntryTy *NewItem = MapEntryTy::Create(KeyStart, KeyEnd, Allocator);
    ++NumItems;
    
    // Fill in the bucket for the hash table.  The FullHashValue was already
    // filled in by LookupBucketFor.
    Bucket.Item = NewItem;
    
    // If the hash table is now more than 3/4 full, rehash into a larger table.
    if (NumItems > NumBuckets*3/4)
      RehashTable();
    return *NewItem;
  }
  
  ~StringMap() {
    for (ItemBucket *I = TheTable, *E = TheTable+NumBuckets; I != E; ++I) {
      if (MapEntryTy *Id = static_cast<MapEntryTy*>(I->Item))
        Id->Destroy(Allocator);
    }
    delete [] TheTable;
  }
};
  

template<typename ValueTy>
class StringMapConstIterator {
  StringMapImpl::ItemBucket *Ptr;
public:
  StringMapConstIterator(StringMapImpl::ItemBucket *Bucket) : Ptr(Bucket) {
    AdvancePastEmptyBuckets();
  }
  
  const StringMapEntry<ValueTy> &operator*() const {
    return *static_cast<StringMapEntry<ValueTy>*>(Ptr->Item);
  }
  const StringMapEntry<ValueTy> *operator->() const {
    return static_cast<StringMapEntry<ValueTy>*>(Ptr->Item);
  }
  
  bool operator==(const StringMapConstIterator &RHS) const {
    return Ptr == RHS.Ptr;
  }
  bool operator!=(const StringMapConstIterator &RHS) const {
    return Ptr != RHS.Ptr;
  }
  
  inline StringMapConstIterator& operator++() {          // Preincrement
    ++Ptr;
    AdvancePastEmptyBuckets();
    return *this;
  }
  StringMapConstIterator operator++(int) {        // Postincrement
    StringMapConstIterator tmp = *this; ++*this; return tmp;
  }
  
private:
  void AdvancePastEmptyBuckets() {
    while (Ptr->Item == 0 || Ptr->Item == StringMapImpl::getTombstoneVal())
      ++Ptr;
  }
};

template<typename ValueTy>
class StringMapIterator : public StringMapConstIterator<ValueTy> {
public:  
  StringMapIterator(StringMapImpl::ItemBucket *Bucket)
    : StringMapConstIterator<ValueTy>(Bucket) {
  }
  StringMapEntry<ValueTy> &operator*() const {
    return *static_cast<StringMapEntry<ValueTy>*>(this->Ptr->Item);
  }
  StringMapEntry<ValueTy> *operator->() const {
    return static_cast<StringMapEntry<ValueTy>*>(this->Ptr->Item);
  }
};

}

#endif

