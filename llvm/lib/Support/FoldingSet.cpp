//===-- Support/FoldingSet.cpp - Uniquing Hash Set --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a hash set that can be used to remove duplication of
// nodes in a graph.  This code was originally created by Chris Lattner for use
// with SelectionDAGCSEMap, but was isolated to provide use across the llvm code
// set. 
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/FoldingSet.h"
#include "llvm/Support/MathExtras.h"
#include <cassert>
#include <cstring>
using namespace llvm;

//===----------------------------------------------------------------------===//
// FoldingSetNodeID Implementation

/// Add* - Add various data types to Bit data.
///
void FoldingSetNodeID::AddPointer(const void *Ptr) {
  // Note: this adds pointers to the hash using sizes and endianness that
  // depend on the host.  It doesn't matter however, because hashing on
  // pointer values in inherently unstable.  Nothing  should depend on the 
  // ordering of nodes in the folding set.
  intptr_t PtrI = (intptr_t)Ptr;
  Bits.push_back(unsigned(PtrI));
  if (sizeof(intptr_t) > sizeof(unsigned))
    Bits.push_back(unsigned(uint64_t(PtrI) >> 32));
}
void FoldingSetNodeID::AddInteger(signed I) {
  Bits.push_back(I);
}
void FoldingSetNodeID::AddInteger(unsigned I) {
  Bits.push_back(I);
}
void FoldingSetNodeID::AddInteger(int64_t I) {
  AddInteger((uint64_t)I);
}
void FoldingSetNodeID::AddInteger(uint64_t I) {
  Bits.push_back(unsigned(I));
  
  // If the integer is small, encode it just as 32-bits.
  if ((uint64_t)(int)I != I)
    Bits.push_back(unsigned(I >> 32));
}
void FoldingSetNodeID::AddFloat(float F) {
  Bits.push_back(FloatToBits(F));
}
void FoldingSetNodeID::AddDouble(double D) {
 AddInteger(DoubleToBits(D));
}

void FoldingSetNodeID::AddString(const char *String) {
  unsigned Size = static_cast<unsigned>(strlen(String));
  Bits.push_back(Size);
  if (!Size) return;

  unsigned Units = Size / 4;
  unsigned Pos = 0;
  const unsigned *Base = (const unsigned *)String;
  
  // If the string is aligned do a bulk transfer.
  if (!((intptr_t)Base & 3)) {
    Bits.append(Base, Base + Units);
    Pos = (Units + 1) * 4;
  } else {
    // Otherwise do it the hard way.
    for ( Pos += 4; Pos <= Size; Pos += 4) {
      unsigned V = ((unsigned char)String[Pos - 4] << 24) |
                   ((unsigned char)String[Pos - 3] << 16) |
                   ((unsigned char)String[Pos - 2] << 8) |
                    (unsigned char)String[Pos - 1];
      Bits.push_back(V);
    }
  }
  
  // With the leftover bits.
  unsigned V = 0;
  // Pos will have overshot size by 4 - #bytes left over. 
  switch (Pos - Size) {
  case 1: V = (V << 8) | (unsigned char)String[Size - 3]; // Fall thru.
  case 2: V = (V << 8) | (unsigned char)String[Size - 2]; // Fall thru.
  case 3: V = (V << 8) | (unsigned char)String[Size - 1]; break;
  default: return; // Nothing left.
  }

  Bits.push_back(V);
}

void FoldingSetNodeID::AddString(const std::string &String) {
  unsigned Size = static_cast<unsigned>(String.size());
  Bits.push_back(Size);
  if (!Size) return;

  unsigned Units = Size / 4;
  unsigned Pos = 0;
  const unsigned *Base = (const unsigned *)String.data();
  
  // If the string is aligned do a bulk transfer.
  if (!((intptr_t)Base & 3)) {
    Bits.append(Base, Base + Units);
    Pos = (Units + 1) * 4;
  } else {
    // Otherwise do it the hard way.
    for ( Pos += 4; Pos <= Size; Pos += 4) {
      unsigned V = ((unsigned char)String[Pos - 4] << 24) |
                   ((unsigned char)String[Pos - 3] << 16) |
                   ((unsigned char)String[Pos - 2] << 8) |
                    (unsigned char)String[Pos - 1];
      Bits.push_back(V);
    }
  }
  
  // With the leftover bits.
  unsigned V = 0;
  // Pos will have overshot size by 4 - #bytes left over. 
  switch (Pos - Size) {
  case 1: V = (V << 8) | (unsigned char)String[Size - 3]; // Fall thru.
  case 2: V = (V << 8) | (unsigned char)String[Size - 2]; // Fall thru.
  case 3: V = (V << 8) | (unsigned char)String[Size - 1]; break;
  default: return; // Nothing left.
  }

  Bits.push_back(V);
}

/// ComputeHash - Compute a strong hash value for this FoldingSetNodeID, used to 
/// lookup the node in the FoldingSetImpl.
unsigned FoldingSetNodeID::ComputeHash() const {
  // This is adapted from SuperFastHash by Paul Hsieh.
  unsigned Hash = static_cast<unsigned>(Bits.size());
  for (const unsigned *BP = &Bits[0], *E = BP+Bits.size(); BP != E; ++BP) {
    unsigned Data = *BP;
    Hash         += Data & 0xFFFF;
    unsigned Tmp  = ((Data >> 16) << 11) ^ Hash;
    Hash          = (Hash << 16) ^ Tmp;
    Hash         += Hash >> 11;
  }
  
  // Force "avalanching" of final 127 bits.
  Hash ^= Hash << 3;
  Hash += Hash >> 5;
  Hash ^= Hash << 4;
  Hash += Hash >> 17;
  Hash ^= Hash << 25;
  Hash += Hash >> 6;
  return Hash;
}

/// operator== - Used to compare two nodes to each other.
///
bool FoldingSetNodeID::operator==(const FoldingSetNodeID &RHS)const{
  if (Bits.size() != RHS.Bits.size()) return false;
  return memcmp(&Bits[0], &RHS.Bits[0], Bits.size()*sizeof(Bits[0])) == 0;
}


//===----------------------------------------------------------------------===//
/// Helper functions for FoldingSetImpl.

/// GetNextPtr - In order to save space, each bucket is a
/// singly-linked-list. In order to make deletion more efficient, we make
/// the list circular, so we can delete a node without computing its hash.
/// The problem with this is that the start of the hash buckets are not
/// Nodes.  If NextInBucketPtr is a bucket pointer, this method returns null:
/// use GetBucketPtr when this happens.
static FoldingSetImpl::Node *GetNextPtr(void *NextInBucketPtr) {
  // The low bit is set if this is the pointer back to the bucket.
  if (reinterpret_cast<intptr_t>(NextInBucketPtr) & 1)
    return 0;
  
  return static_cast<FoldingSetImpl::Node*>(NextInBucketPtr);
}


/// testing.
static void **GetBucketPtr(void *NextInBucketPtr) {
  intptr_t Ptr = reinterpret_cast<intptr_t>(NextInBucketPtr);
  assert((Ptr & 1) && "Not a bucket pointer");
  return reinterpret_cast<void**>(Ptr & ~intptr_t(1));
}

/// GetBucketFor - Hash the specified node ID and return the hash bucket for
/// the specified ID.
static void **GetBucketFor(const FoldingSetNodeID &ID,
                           void **Buckets, unsigned NumBuckets) {
  // NumBuckets is always a power of 2.
  unsigned BucketNum = ID.ComputeHash() & (NumBuckets-1);
  return Buckets + BucketNum;
}

//===----------------------------------------------------------------------===//
// FoldingSetImpl Implementation

FoldingSetImpl::FoldingSetImpl(unsigned Log2InitSize) : NumNodes(0) {
  assert(5 < Log2InitSize && Log2InitSize < 32 &&
         "Initial hash table size out of range");
  NumBuckets = 1 << Log2InitSize;
  Buckets = new void*[NumBuckets+1];
  memset(Buckets, 0, NumBuckets*sizeof(void*));
  
  // Set the very last bucket to be a non-null "pointer".
  Buckets[NumBuckets] = reinterpret_cast<void*>(-1);
}
FoldingSetImpl::~FoldingSetImpl() {
  delete [] Buckets;
}

/// GrowHashTable - Double the size of the hash table and rehash everything.
///
void FoldingSetImpl::GrowHashTable() {
  void **OldBuckets = Buckets;
  unsigned OldNumBuckets = NumBuckets;
  NumBuckets <<= 1;
  
  // Reset the node count to zero: we're going to reinsert everything.
  NumNodes = 0;
  
  // Clear out new buckets.
  Buckets = new void*[NumBuckets+1];
  memset(Buckets, 0, NumBuckets*sizeof(void*));

  // Set the very last bucket to be a non-null "pointer".
  Buckets[NumBuckets] = reinterpret_cast<void*>(-1);

  // Walk the old buckets, rehashing nodes into their new place.
  for (unsigned i = 0; i != OldNumBuckets; ++i) {
    void *Probe = OldBuckets[i];
    if (!Probe) continue;
    while (Node *NodeInBucket = GetNextPtr(Probe)) {
      // Figure out the next link, remove NodeInBucket from the old link.
      Probe = NodeInBucket->getNextInBucket();
      NodeInBucket->SetNextInBucket(0);

      // Insert the node into the new bucket, after recomputing the hash.
      FoldingSetNodeID ID;
      GetNodeProfile(ID, NodeInBucket);
      InsertNode(NodeInBucket, GetBucketFor(ID, Buckets, NumBuckets));
    }
  }
  
  delete[] OldBuckets;
}

/// FindNodeOrInsertPos - Look up the node specified by ID.  If it exists,
/// return it.  If not, return the insertion token that will make insertion
/// faster.
FoldingSetImpl::Node
*FoldingSetImpl::FindNodeOrInsertPos(const FoldingSetNodeID &ID,
                                     void *&InsertPos) {
  
  void **Bucket = GetBucketFor(ID, Buckets, NumBuckets);
  void *Probe = *Bucket;
  
  InsertPos = 0;
  
  while (Node *NodeInBucket = GetNextPtr(Probe)) {
    FoldingSetNodeID OtherID;
    GetNodeProfile(OtherID, NodeInBucket);
    if (OtherID == ID)
      return NodeInBucket;

    Probe = NodeInBucket->getNextInBucket();
  }
  
  // Didn't find the node, return null with the bucket as the InsertPos.
  InsertPos = Bucket;
  return 0;
}

/// InsertNode - Insert the specified node into the folding set, knowing that it
/// is not already in the map.  InsertPos must be obtained from 
/// FindNodeOrInsertPos.
void FoldingSetImpl::InsertNode(Node *N, void *InsertPos) {
  assert(N->getNextInBucket() == 0);
  // Do we need to grow the hashtable?
  if (NumNodes+1 > NumBuckets*2) {
    GrowHashTable();
    FoldingSetNodeID ID;
    GetNodeProfile(ID, N);
    InsertPos = GetBucketFor(ID, Buckets, NumBuckets);
  }

  ++NumNodes;
  
  /// The insert position is actually a bucket pointer.
  void **Bucket = static_cast<void**>(InsertPos);
  
  void *Next = *Bucket;
  
  // If this is the first insertion into this bucket, its next pointer will be
  // null.  Pretend as if it pointed to itself, setting the low bit to indicate
  // that it is a pointer to the bucket.
  if (Next == 0)
    Next = reinterpret_cast<void*>(reinterpret_cast<intptr_t>(Bucket)|1);

  // Set the node's next pointer, and make the bucket point to the node.
  N->SetNextInBucket(Next);
  *Bucket = N;
}

/// RemoveNode - Remove a node from the folding set, returning true if one was
/// removed or false if the node was not in the folding set.
bool FoldingSetImpl::RemoveNode(Node *N) {
  // Because each bucket is a circular list, we don't need to compute N's hash
  // to remove it.
  void *Ptr = N->getNextInBucket();
  if (Ptr == 0) return false;  // Not in folding set.

  --NumNodes;
  N->SetNextInBucket(0);

  // Remember what N originally pointed to, either a bucket or another node.
  void *NodeNextPtr = Ptr;
  
  // Chase around the list until we find the node (or bucket) which points to N.
  while (true) {
    if (Node *NodeInBucket = GetNextPtr(Ptr)) {
      // Advance pointer.
      Ptr = NodeInBucket->getNextInBucket();
      
      // We found a node that points to N, change it to point to N's next node,
      // removing N from the list.
      if (Ptr == N) {
        NodeInBucket->SetNextInBucket(NodeNextPtr);
        return true;
      }
    } else {
      void **Bucket = GetBucketPtr(Ptr);
      Ptr = *Bucket;
      
      // If we found that the bucket points to N, update the bucket to point to
      // whatever is next.
      if (Ptr == N) {
        *Bucket = NodeNextPtr;
        return true;
      }
    }
  }
}

/// GetOrInsertNode - If there is an existing simple Node exactly
/// equal to the specified node, return it.  Otherwise, insert 'N' and it
/// instead.
FoldingSetImpl::Node *FoldingSetImpl::GetOrInsertNode(FoldingSetImpl::Node *N) {
  FoldingSetNodeID ID;
  GetNodeProfile(ID, N);
  void *IP;
  if (Node *E = FindNodeOrInsertPos(ID, IP))
    return E;
  InsertNode(N, IP);
  return N;
}

//===----------------------------------------------------------------------===//
// FoldingSetIteratorImpl Implementation

FoldingSetIteratorImpl::FoldingSetIteratorImpl(void **Bucket) {
  // Skip to the first non-null non-self-cycle bucket.
  while (*Bucket != reinterpret_cast<void*>(-1) &&
         (*Bucket == 0 || GetNextPtr(*Bucket) == 0))
    ++Bucket;
  
  NodePtr = static_cast<FoldingSetNode*>(*Bucket);
}

void FoldingSetIteratorImpl::advance() {
  // If there is another link within this bucket, go to it.
  void *Probe = NodePtr->getNextInBucket();

  if (FoldingSetNode *NextNodeInBucket = GetNextPtr(Probe))
    NodePtr = NextNodeInBucket;
  else {
    // Otherwise, this is the last link in this bucket.  
    void **Bucket = GetBucketPtr(Probe);

    // Skip to the next non-null non-self-cycle bucket.
    do {
      ++Bucket;
    } while (*Bucket != reinterpret_cast<void*>(-1) &&
             (*Bucket == 0 || GetNextPtr(*Bucket) == 0));
    
    NodePtr = static_cast<FoldingSetNode*>(*Bucket);
  }
}

//===----------------------------------------------------------------------===//
// FoldingSetBucketIteratorImpl Implementation

FoldingSetBucketIteratorImpl::FoldingSetBucketIteratorImpl(void **Bucket) {
  Ptr = (*Bucket == 0 || GetNextPtr(*Bucket) == 0) ? (void*) Bucket : *Bucket;
}
