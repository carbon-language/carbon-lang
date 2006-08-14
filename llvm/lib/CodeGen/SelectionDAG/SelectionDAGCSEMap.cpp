//===-- SelectionDAGCSEMap.cpp - Implement the SelectionDAG CSE Map -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements the SelectionDAGCSEMap class.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/Support/MathExtras.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
// SelectionDAGCSEMap::NodeID Implementation

SelectionDAGCSEMap::NodeID::NodeID(SDNode *N) {
  SetOpcode(N->getOpcode());
  // Add the return value info.
  SetValueTypes(N->value_begin());
  // Add the operand info.
  SetOperands(N->op_begin(), N->getNumOperands());

  // Handle SDNode leafs with special info.
  if (N->getNumOperands() == 0) {
    switch (N->getOpcode()) {
    default: break;  // Normal nodes don't need extra info.
    case ISD::TargetConstant:
    case ISD::Constant:
      AddInteger(cast<ConstantSDNode>(N)->getValue());
      break;
    case ISD::TargetConstantFP:
    case ISD::ConstantFP:
      AddInteger(DoubleToBits(cast<ConstantFPSDNode>(N)->getValue()));
      break;
    case ISD::TargetGlobalAddress:
    case ISD::GlobalAddress:
      AddPointer(cast<GlobalAddressSDNode>(N)->getGlobal());
      AddInteger(cast<GlobalAddressSDNode>(N)->getOffset());
      break;
    case ISD::BasicBlock:
      AddPointer(cast<BasicBlockSDNode>(N)->getBasicBlock());
      break;
    case ISD::Register:
      AddInteger(cast<RegisterSDNode>(N)->getReg());
      break;
    case ISD::SRCVALUE:
      AddPointer(cast<SrcValueSDNode>(N)->getValue());
      AddInteger(cast<SrcValueSDNode>(N)->getOffset());
      break;
    case ISD::FrameIndex:
    case ISD::TargetFrameIndex:
      AddInteger(cast<FrameIndexSDNode>(N)->getIndex());
      break;
    case ISD::JumpTable:
    case ISD::TargetJumpTable:
      AddInteger(cast<JumpTableSDNode>(N)->getIndex());
      break;
    case ISD::ConstantPool:
    case ISD::TargetConstantPool:
      AddInteger(cast<ConstantPoolSDNode>(N)->getAlignment());
      AddInteger(cast<ConstantPoolSDNode>(N)->getOffset());
      AddPointer(cast<ConstantPoolSDNode>(N)->get());
      break;
    }
  }
}

SelectionDAGCSEMap::NodeID::NodeID(unsigned short ID, const void *VTList) {
  SetOpcode(ID);
  SetValueTypes(VTList);
  SetOperands();
}
SelectionDAGCSEMap::NodeID::NodeID(unsigned short ID, const void *VTList,
                                   SDOperand Op) {
  SetOpcode(ID);
  SetValueTypes(VTList);
  SetOperands(Op);
}
SelectionDAGCSEMap::NodeID::NodeID(unsigned short ID, const void *VTList, 
                                   SDOperand Op1, SDOperand Op2) {
  SetOpcode(ID);
  SetValueTypes(VTList);
  SetOperands(Op1, Op2);
}
SelectionDAGCSEMap::NodeID::NodeID(unsigned short ID, const void *VTList, 
                                   SDOperand Op1, SDOperand Op2,
                                   SDOperand Op3) {
  SetOpcode(ID);
  SetValueTypes(VTList);
  SetOperands(Op1, Op2, Op3);
}
SelectionDAGCSEMap::NodeID::NodeID(unsigned short ID, const void *VTList, 
                                   const SDOperand *OpList, unsigned N) {
  SetOpcode(ID);
  SetValueTypes(VTList);
  SetOperands(OpList, N);
}

void SelectionDAGCSEMap::NodeID::AddPointer(const void *Ptr) {
  // Note: this adds pointers to the hash using sizes and endianness that depend
  // on the host.  It doesn't matter however, because hashing on pointer values
  // in inherently unstable.  Nothing in the SelectionDAG should depend on the
  // ordering of nodes in the CSEMap.
  intptr_t PtrI = (intptr_t)Ptr;
  Bits.push_back(unsigned(PtrI));
  if (sizeof(intptr_t) > sizeof(unsigned))
    Bits.push_back(unsigned(uint64_t(PtrI) >> 32));
}

void SelectionDAGCSEMap::NodeID::AddOperand(SDOperand Op) {
  AddPointer(Op.Val);
  Bits.push_back(Op.ResNo);
}

void SelectionDAGCSEMap::NodeID::SetOperands(const SDOperand *Ops, 
                                             unsigned NumOps) {
  for (; NumOps; --NumOps, ++Ops)
    AddOperand(*Ops);
}


/// ComputeHash - Compute a strong hash value for this NodeID, for lookup in
/// the SelectionDAGCSEMap.
unsigned SelectionDAGCSEMap::NodeID::ComputeHash() const {
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

bool SelectionDAGCSEMap::NodeID::operator==(const NodeID &RHS) const {
  if (Bits.size() != RHS.Bits.size()) return false;
  return memcmp(&Bits[0], &RHS.Bits[0], Bits.size()*sizeof(Bits[0])) == 0;
}


//===----------------------------------------------------------------------===//
// SelectionDAGCSEMap Implementation

SelectionDAGCSEMap::SelectionDAGCSEMap() : NumNodes(0) {
  NumBuckets = 64;
  Buckets = new void*[NumBuckets];
  memset(Buckets, 0, NumBuckets*sizeof(void*));
}
SelectionDAGCSEMap::~SelectionDAGCSEMap() {
  delete [] Buckets;
}

/// GetNextPtr - In order to save space, each bucket is a singly-linked-list. In
/// order to make deletion more efficient, we make the list circular, so we can
/// delete a node without computing its hash.  The problem with this is that the
/// start of the hash buckets are not SDNodes.  If NextInBucketPtr is a bucket
/// pointer, this method returns null: use GetBucketPtr when this happens.
SDNode *SelectionDAGCSEMap::GetNextPtr(void *NextInBucketPtr) {
  if (NextInBucketPtr >= Buckets && NextInBucketPtr < Buckets+NumBuckets)
    return 0;
  return static_cast<SDNode*>(NextInBucketPtr);
}

/// GetNextPtr - This is just like the previous GetNextPtr implementation, but
/// allows a bucket array to be specified.
SDNode *SelectionDAGCSEMap::GetNextPtr(void *NextInBucketPtr, void **Bucks, 
                                       unsigned NumBuck) {
  if (NextInBucketPtr >= Bucks && NextInBucketPtr < Bucks+NumBuck)
    return 0;
  return static_cast<SDNode*>(NextInBucketPtr);
}

void **SelectionDAGCSEMap::GetBucketPtr(void *NextInBucketPtr) {
  //assert(NextInBucketPtr >= Buckets && NextInBucketPtr < Buckets+NumBuckets &&
  //       "NextInBucketPtr is not a bucket ptr");
  return static_cast<void**>(NextInBucketPtr);
}

/// GetBucketFor - Hash the specified node ID and return the hash bucket for the
/// specified ID.
void **SelectionDAGCSEMap::GetBucketFor(const NodeID &ID) const {
  // NumBuckets is always a power of 2.
  unsigned BucketNum = ID.ComputeHash() & (NumBuckets-1);
  return Buckets+BucketNum;
}

/// GrowHashTable - Double the size of the hash table and rehash everything.
///
void SelectionDAGCSEMap::GrowHashTable() {
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
    while (SDNode *NodeInBucket = GetNextPtr(Probe, OldBuckets, OldNumBuckets)){
      // Figure out the next link, remove NodeInBucket from the old link.
      Probe = NodeInBucket->getNextInBucket();
      NodeInBucket->SetNextInBucket(0);

      // Insert the node into the new bucket, after recomputing the hash.
      InsertNode(NodeInBucket, GetBucketFor(NodeID(NodeInBucket)));
    }
  }
  
  delete[] OldBuckets;
}

/// FindNodeOrInsertPos - Look up the node specified by ID.  If it exists,
/// return it.  If not, return the insertion token that will make insertion
/// faster.
SDNode *SelectionDAGCSEMap::FindNodeOrInsertPos(const NodeID &ID,
                                                void *&InsertPos) {
  void **Bucket = GetBucketFor(ID);
  void *Probe = *Bucket;
  
  InsertPos = 0;
  
  unsigned Opc = ID.getOpcode();
  while (SDNode *NodeInBucket = GetNextPtr(Probe)) {
    // If we found a node with the same opcode, it might be a matching node.
    // Because it is in the same bucket as this one, we know the hash values
    // match.  Compute the NodeID for the possible match and do a final compare.
    if (NodeInBucket->getOpcode() == Opc) {
      NodeID OtherID(NodeInBucket);
      if (OtherID == ID)
        return NodeInBucket;
    }

    Probe = NodeInBucket->getNextInBucket();
  }
  
  // Didn't find the node, return null with the bucket as the InsertPos.
  InsertPos = Bucket;
  return 0;
}

/// InsertNode - Insert the specified node into the CSE Map, knowing that it
/// is not already in the map.  InsertPos must be obtained from 
/// FindNodeOrInsertPos.
void SelectionDAGCSEMap::InsertNode(SDNode *N, void *InsertPos) {
  ++NumNodes;
  // Do we need to grow the hashtable?
  if (NumNodes > NumBuckets*2) {
    GrowHashTable();
    InsertPos = GetBucketFor(NodeID(N));
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


/// RemoveNode - Remove a node from the CSE map, returning true if one was
/// removed or false if the node was not in the CSE map.
bool SelectionDAGCSEMap::RemoveNode(SDNode *N) {

  // Because each bucket is a circular list, we don't need to compute N's hash
  // to remove it.  Chase around the list until we find the node (or bucket)
  // which points to N.
  void *Ptr = N->getNextInBucket();
  if (Ptr == 0) return false;  // Not in CSEMap.

  --NumNodes;

  void *NodeNextPtr = Ptr;
  N->SetNextInBucket(0);
  while (1) {
    if (SDNode *NodeInBucket = GetNextPtr(Ptr)) {
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

/// GetOrInsertSimpleNode - If there is an existing simple SDNode exactly
/// equal to the specified node, return it.  Otherwise, insert 'N' and it
/// instead.  This only works on *simple* SDNodes, not ConstantSDNode or any
/// other classes derived from SDNode.
SDNode *SelectionDAGCSEMap::GetOrInsertNode(SDNode *N) {
  SelectionDAGCSEMap::NodeID ID(N);
  void *IP;
  if (SDNode *E = FindNodeOrInsertPos(ID, IP))
    return E;
  InsertNode(N, IP);
  return N;
}
