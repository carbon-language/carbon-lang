//===--- Preprocess.cpp - C Language Family Preprocessor Implementation ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the IdentifierInfo, IdentifierVisitor, and
// IdentifierTable interfaces.
//
//===----------------------------------------------------------------------===//

#include "clang/Lex/IdentifierTable.h"
#include "clang/Lex/MacroInfo.h"
#include <iostream>
using namespace llvm;
using namespace clang;

//===----------------------------------------------------------------------===//
// IdentifierInfo Implementation
//===----------------------------------------------------------------------===//

void IdentifierInfo::Destroy() {
  delete Macro;
}

//===----------------------------------------------------------------------===//
// IdentifierVisitor Implementation
//===----------------------------------------------------------------------===//

IdentifierVisitor::~IdentifierVisitor() {
}

//===----------------------------------------------------------------------===//
// Memory Allocation Support
//===----------------------------------------------------------------------===//

/// The identifier table has a very simple memory allocation pattern: it just
/// keeps allocating identifiers, then never frees them unless it frees them
/// all.  As such, we use a simple bump-pointer memory allocator to make
/// allocation speedy.  Shark showed that malloc was 27% of the time spent in
/// IdentifierTable::getIdentifier with malloc, and takes a 4.3% time with this.
#define USE_ALLOCATOR 1
#if USE_ALLOCATOR

namespace {
class MemRegion {
  unsigned RegionSize;
  MemRegion *Next;
  char *NextPtr;
public:
  void Init(unsigned size, MemRegion *next) {
    RegionSize = size;
    Next = next;
    NextPtr = (char*)(this+1);
    
    // FIXME: uses GCC extension.
    unsigned Alignment = __alignof__(IdentifierInfo);
    NextPtr = (char*)((intptr_t)(NextPtr+Alignment-1) &
                      ~(intptr_t)(Alignment-1));
  }
  
  const MemRegion *getNext() const { return Next; }
  unsigned getNumBytesAllocated() const {
    return NextPtr-(const char*)this;
  }
  
  /// Allocate - Allocate and return at least the specified number of bytes.
  ///
  void *Allocate(unsigned AllocSize, MemRegion **RegPtr) {
    // FIXME: uses GCC extension.
    unsigned Alignment = __alignof__(IdentifierInfo);
    // Round size up to an even multiple of the alignment.
    AllocSize = (AllocSize+Alignment-1) & ~(Alignment-1);
    
    // If there is space in this region for the identifier, return it.
    if (unsigned(NextPtr+AllocSize-(char*)this) <= RegionSize) {
      void *Result = NextPtr;
      NextPtr += AllocSize;
      return Result;
    }
    
    // Otherwise, we have to allocate a new chunk.  Create one twice as big as
    // this one.
    MemRegion *NewRegion = (MemRegion *)malloc(RegionSize*2);
    NewRegion->Init(RegionSize*2, this);

    // Update the current "first region" pointer  to point to the new region.
    *RegPtr = NewRegion;
    
    // Try allocating from it now.
    return NewRegion->Allocate(AllocSize, RegPtr);
  }
  
  /// Deallocate - Release all memory for this region to the system.
  ///
  void Deallocate() {
    MemRegion *next = Next;
    free(this);
    if (next)
      next->Deallocate();
  }
};
}

#endif

//===----------------------------------------------------------------------===//
// IdentifierTable Implementation
//===----------------------------------------------------------------------===//


/// IdentifierLink - There is one of these allocated by IdentifierInfo.
/// These form the linked list of buckets for the hash table.
struct IdentifierBucket {
  /// Next - This is the next bucket in the linked list.
  IdentifierBucket *Next;
  
  IdentifierInfo TokInfo;
  // NOTE: TokInfo must be the last element in this structure, as the string
  // information for the identifier is allocated right after it.
};

// FIXME: start hashtablesize off at 8K entries, GROW when density gets to 3.
static unsigned HASH_TABLE_SIZE = 8096;

IdentifierTable::IdentifierTable() {
  IdentifierBucket **TableArray = new IdentifierBucket*[HASH_TABLE_SIZE]();
  TheTable = TableArray;
  NumIdentifiers = 0;
#if USE_ALLOCATOR
  TheMemory = malloc(8*4096);
  ((MemRegion*)TheMemory)->Init(8*4096, 0);
#endif
  
  memset(TheTable, 0, HASH_TABLE_SIZE*sizeof(IdentifierBucket*));
}

IdentifierTable::~IdentifierTable() {
  IdentifierBucket **TableArray = (IdentifierBucket**)TheTable;
  for (unsigned i = 0, e = HASH_TABLE_SIZE; i != e; ++i) {
    IdentifierBucket *Id = TableArray[i]; 
    while (Id) {
      // Free memory referenced by the identifier (e.g. macro info).
      Id->TokInfo.Destroy();
      
      IdentifierBucket *Next = Id->Next;
#if !USE_ALLOCATOR
      free(Id);
#endif
      Id = Next;
    }
  }
#if USE_ALLOCATOR
  ((MemRegion*)TheMemory)->Deallocate();
#endif
  delete [] TableArray;
}

/// HashString - Compute a hash code for the specified string.
///
static unsigned HashString(const char *Start, const char *End) {
  unsigned int Result = 0;
  // Perl hash function.
  while (Start != End)
    Result = Result * 33 + *Start++;
  Result = Result + (Result >> 5);
  return Result;
}

IdentifierInfo &IdentifierTable::get(const char *NameStart,
                                     const char *NameEnd) {
  IdentifierBucket **TableArray = (IdentifierBucket**)TheTable;

  unsigned Hash = HashString(NameStart, NameEnd) % HASH_TABLE_SIZE;
  unsigned Length = NameEnd-NameStart;
  
  IdentifierBucket *IdentHead = TableArray[Hash];
  for (IdentifierBucket *Identifier = IdentHead, *LastID = 0; Identifier; 
       LastID = Identifier, Identifier = Identifier->Next) {
    if (Identifier->TokInfo.getNameLength() == Length &&
        memcmp(Identifier->TokInfo.getName(), NameStart, Length) == 0) {
      // If found identifier wasn't at start of bucket, move it there so
      // that frequently searched for identifiers are found earlier, even if
      // they first occur late in the source file.
      if (LastID) {
        LastID->Next = Identifier->Next;
        Identifier->Next = IdentHead;
        TableArray[Hash] = Identifier;
      }
      
      return Identifier->TokInfo;
    }
  }

  // Allocate a new identifier, with space for the null-terminated string at the
  // end.
  unsigned AllocSize = sizeof(IdentifierBucket)+Length+1;
#if USE_ALLOCATOR
  IdentifierBucket *Identifier = (IdentifierBucket*)
    ((MemRegion*)TheMemory)->Allocate(AllocSize, (MemRegion**)&TheMemory);
#else
  IdentifierBucket *Identifier = (IdentifierBucket*)malloc(AllocSize);
#endif
  Identifier->TokInfo.NameLen = Length;
  Identifier->TokInfo.Macro = 0;
  Identifier->TokInfo.TokenID = tok::identifier;
  Identifier->TokInfo.IsExtension = false;
  Identifier->TokInfo.IsPoisoned = false;
  Identifier->TokInfo.IsMacroArg = false;
  Identifier->TokInfo.FETokenInfo = 0;

  // Copy the string information.
  char *StrBuffer = (char*)(Identifier+1);
  memcpy(StrBuffer, NameStart, Length);
  StrBuffer[Length] = 0;  // Null terminate string.
  
  // Link it into the hash table.  Adding it to the start of the hash table is
  // useful for buckets with lots of entries.  This means that more recently
  // referenced identifiers will be near the head of the bucket.
  Identifier->Next = IdentHead;
  TableArray[Hash] = Identifier;
  return Identifier->TokInfo;
}

IdentifierInfo &IdentifierTable::get(const std::string &Name) {
  // Don't use c_str() here: no need to be null terminated.
  const char *NameBytes = &Name[0];
  unsigned Size = Name.size();
  return get(NameBytes, NameBytes+Size);
}

/// VisitIdentifiers - This method walks through all of the identifiers,
/// invoking IV->VisitIdentifier for each of them.
void IdentifierTable::VisitIdentifiers(const IdentifierVisitor &IV) {
  IdentifierBucket **TableArray = (IdentifierBucket**)TheTable;
  for (unsigned i = 0, e = HASH_TABLE_SIZE; i != e; ++i) {
    for (IdentifierBucket *Id = TableArray[i]; Id; Id = Id->Next)
      IV.VisitIdentifier(Id->TokInfo);
  }
}

/// PrintStats - Print statistics about how well the identifier table is doing
/// at hashing identifiers.
void IdentifierTable::PrintStats() const {
  unsigned NumIdentifiers = 0;
  unsigned NumEmptyBuckets = 0;
  unsigned MaxBucketLength = 0;
  unsigned AverageIdentifierSize = 0;
  unsigned MaxIdentifierLength = 0;
  
  IdentifierBucket **TableArray = (IdentifierBucket**)TheTable;
  for (unsigned i = 0, e = HASH_TABLE_SIZE; i != e; ++i) {
    
    unsigned NumIdentifiersInBucket = 0;
    for (IdentifierBucket *Id = TableArray[i]; Id; Id = Id->Next) {
      AverageIdentifierSize += Id->TokInfo.getNameLength();
      if (MaxIdentifierLength < Id->TokInfo.getNameLength())
        MaxIdentifierLength = Id->TokInfo.getNameLength();
      ++NumIdentifiersInBucket;
    }
    if (NumIdentifiersInBucket > MaxBucketLength) 
      MaxBucketLength = NumIdentifiersInBucket;
    if (NumIdentifiersInBucket == 0)
      ++NumEmptyBuckets;

    NumIdentifiers += NumIdentifiersInBucket;
  }
  
  std::cerr << "\n*** Identifier Table Stats:\n";
  std::cerr << "# Identifiers:   " << NumIdentifiers << "\n";
  std::cerr << "# Empty Buckets: " << NumEmptyBuckets << "\n";
  std::cerr << "Max identifiers in one bucket: " << MaxBucketLength << "\n";
  std::cerr << "Hash density (#identifiers per bucket): "
            << NumIdentifiers/(double)HASH_TABLE_SIZE << "\n";
  std::cerr << "Nonempty hash density (average chain length): "
            << NumIdentifiers/(double)(HASH_TABLE_SIZE-NumEmptyBuckets) << "\n";
  std::cerr << "Ave identifier length: "
            << (AverageIdentifierSize/(double)NumIdentifiers) << "\n";
  std::cerr << "Max identifier length: " << MaxIdentifierLength << "\n";
  
  // Compute statistics about the memory allocated for identifiers.
#if USE_ALLOCATOR
  unsigned BytesUsed = 0;
  unsigned NumRegions = 0;
  const MemRegion *R = (MemRegion*)TheMemory;
  for (; R; R = R->getNext(), ++NumRegions) {
    BytesUsed += R->getNumBytesAllocated();
  }
  std::cerr << "\nNumber of memory regions: " << NumRegions << "\n";
  std::cerr << "Bytes allocated for identifiers: " << BytesUsed << "\n";
#endif
}


