//===-- Support/FoldingSet.cpp - Uniquing Hash Set --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by James M. Laskey and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
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

#include "llvm/ADT/MathExtras.h"

using namespace llvm;

//===----------------------------------------------------------------------===//
// FoldingSetImpl::NodeID Implementation

/// Add* - Add various data types to Bit data.
///
void FoldingSetImpl::NodeID::AddPointer(const void *Ptr) {
  // Note: this adds pointers to the hash using sizes and endianness that
  // depend on the host.  It doesn't matter however, because hashing on
  // pointer values in inherently unstable.  Nothing  should depend on the 
  // ordering of nodes in the folding set.
  intptr_t PtrI = (intptr_t)Ptr;
  Bits.push_back(unsigned(PtrI));
  if (sizeof(intptr_t) > sizeof(unsigned))
    Bits.push_back(unsigned(uint64_t(PtrI) >> 32));
}
void FoldingSetImpl::NodeID::AddInteger(signed I) {
  Bits.push_back(I);
}
void FoldingSetImpl::NodeID::AddInteger(unsigned I) {
  Bits.push_back(I);
}
void FoldingSetImpl::NodeID::AddInteger(uint64_t I) {
  Bits.push_back(unsigned(I));
  Bits.push_back(unsigned(I >> 32));
}
void FoldingSetImpl::NodeID::AddFloat(float F) {
  Bits.push_back(FloatToBits(F));
}
void FoldingSetImpl::NodeID::AddDouble(double D) {
  Bits.push_back(DoubleToBits(D));
}
void FoldingSetImpl::NodeID::AddString(const std::string &String) {
  // Note: An assumption is made here that strings are composed of one byte
  // chars.
  unsigned Size = String.size();
  unsigned Units = Size / sizeof(unsigned);
  const unsigned *Base = (const unsigned *)String.data();
  Bits.insert(Bits.end(), Base, Base + Units);
  if (Size & 3) {
    unsigned V = 0;
    for (unsigned i = Units * sizeof(unsigned); i < Size; ++i)
      V = (V << 8) | String[i];
    Bits.push_back(V);
  }
}

/// ComputeHash - Compute a strong hash value for this NodeID, used to 
/// lookup the node in the FoldingSetImpl.
unsigned FoldingSetImpl::NodeID::ComputeHash() const {
  // This is adapted from SuperFastHash by Paul Hsieh.
  unsigned Hash = Bits.size();
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
bool FoldingSetImpl::NodeID::operator==(const FoldingSetImpl::NodeID &RHS)const{
  if (Bits.size() != RHS.Bits.size()) return false;
  return memcmp(&Bits[0], &RHS.Bits[0], Bits.size()*sizeof(Bits[0])) == 0;
}


//===----------------------------------------------------------------------===//
// FoldingSetImpl Implementation

FoldingSetImpl::FoldingSetImpl() : NumNodes(0) {
  NumBuckets = 64;
  Buckets = new void*[NumBuckets];
  memset(Buckets, 0, NumBuckets*sizeof(void*));
}
FoldingSetImpl::~FoldingSetImpl() {
  delete [] Buckets;
}

/// GetNextPtr - In order to save space, each bucket is a
/// singly-linked-list. In order to make deletion more efficient, we make
/// the list circular, so we can delete a node without computing its hash.
/// The problem with this is that the start of the hash buckets are not
/// Nodes.  If NextInBucketPtr is a bucket pointer, this method returns null
/// : use GetBucketPtr when this happens.
FoldingSetImpl::Node *FoldingSetImpl::GetNextPtr(void *NextInBucketPtr) {
  if (NextInBucketPtr >= Buckets && NextInBucketPtr < Buckets+NumBuckets)
    return 0;
  return static_cast<Node*>(NextInBucketPtr);
}

/// GetNextPtr - This is just like the previous GetNextPtr implementation,
/// but allows a bucket array to be specified.
FoldingSetImpl::Node *FoldingSetImpl::GetNextPtr(void *NextInBucketPtr,
                                                 void **Bucks,
                                                 unsigned NumBuck) {
  if (NextInBucketPtr >= Bucks && NextInBucketPtr < Bucks+NumBuck)
    return 0;
  return static_cast<Node*>(NextInBucketPtr);
}

/// GetBucketPtr - Provides a casting of a bucket pointer for isNode
/// testing.
void **FoldingSetImpl::GetBucketPtr(void *NextInBucketPtr) {
  return static_cast<void**>(NextInBucketPtr);
}

/// GetBucketFor - Hash the specified node ID and return the hash bucket for
/// the specified ID.
void **FoldingSetImpl::GetBucketFor(const NodeID &ID) const {
  // NumBuckets is always a power of 2.
  unsigned BucketNum = ID.ComputeHash() & (NumBuckets-1);
  return Buckets+BucketNum;
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
  Buckets = new void*[NumBuckets];
  memset(Buckets, 0, NumBuckets*sizeof(void*));

  // Walk the old buckets, rehashing nodes into their new place.
  for (unsigned i = 0; i != OldNumBuckets; ++i) {
    void *Probe = OldBuckets[i];
    if (!Probe) continue;
    while (Node *NodeInBucket = GetNextPtr(Probe, OldBuckets, OldNumBuckets)){
      // Figure out the next link, remove NodeInBucket from the old link.
      Probe = NodeInBucket->getNextInBucket();
      NodeInBucket->SetNextInBucket(0);

      // Insert the node into the new bucket, after recomputing the hash.
      NodeID ID;
      GetNodeProfile(ID, NodeInBucket);
      InsertNode(NodeInBucket, GetBucketFor(ID));
    }
  }
  
  delete[] OldBuckets;
}

/// FindNodeOrInsertPos - Look up the node specified by ID.  If it exists,
/// return it.  If not, return the insertion token that will make insertion
/// faster.
FoldingSetImpl::Node *FoldingSetImpl::FindNodeOrInsertPos(const NodeID &ID,
                                                          void *&InsertPos) {
  void **Bucket = GetBucketFor(ID);
  void *Probe = *Bucket;
  
  InsertPos = 0;
  
  while (Node *NodeInBucket = GetNextPtr(Probe)) {
    NodeID OtherID;
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
  ++NumNodes;
  // Do we need to grow the hashtable?
  if (NumNodes > NumBuckets*2) {
    GrowHashTable();
    NodeID ID;
    GetNodeProfile(ID, N);
    InsertPos = GetBucketFor(ID);
  }
  
  /// The insert position is actually a bucket pointer.
  void **Bucket = static_cast<void**>(InsertPos);
  
  void *Next = *Bucket;
  
  // If this is the first insertion into this bucket, its next pointer will be
  // null.  Pretend as if it pointed to itself.
  if (Next == 0)
    Next = Bucket;

  // Set the nodes next pointer, and make the bucket point to the node.
  N->SetNextInBucket(Next);
  *Bucket = N;
}

/// RemoveNode - Remove a node from the folding set, returning true if one was
/// removed or false if the node was not in the folding set.
bool FoldingSetImpl::RemoveNode(Node *N) {
  // Because each bucket is a circular list, we don't need to compute N's hash
  // to remove it.  Chase around the list until we find the node (or bucket)
  // which points to N.
  void *Ptr = N->getNextInBucket();
  if (Ptr == 0) return false;  // Not in folding set.

  --NumNodes;

  void *NodeNextPtr = Ptr;
  N->SetNextInBucket(0);
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
  NodeID ID;
  GetNodeProfile(ID, N);
  void *IP;
  if (Node *E = FindNodeOrInsertPos(ID, IP))
    return E;
  InsertNode(N, IP);
  return N;
}
