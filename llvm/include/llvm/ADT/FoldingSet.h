//===-- llvm/ADT/FoldingSet.h - Uniquing Hash Set ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a hash set that can be used to remove duplication of nodes
// in a graph.  This code was originally created by Chris Lattner for use with
// SelectionDAGCSEMap, but was isolated to provide use across the llvm code set. 
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_FOLDINGSET_H
#define LLVM_ADT_FOLDINGSET_H

#include "llvm/Support/DataTypes.h"
#include "llvm/ADT/SmallVector.h"
#include <string>

namespace llvm {
  class APFloat;

/// This folding set used for two purposes:
///   1. Given information about a node we want to create, look up the unique
///      instance of the node in the set.  If the node already exists, return
///      it, otherwise return the bucket it should be inserted into.
///   2. Given a node that has already been created, remove it from the set.
/// 
/// This class is implemented as a single-link chained hash table, where the
/// "buckets" are actually the nodes themselves (the next pointer is in the
/// node).  The last node points back to the bucket to simplified node removal.
///
/// Any node that is to be included in the folding set must be a subclass of
/// FoldingSetNode.  The node class must also define a Profile method used to
/// establish the unique bits of data for the node.  The Profile method is
/// passed a FoldingSetNodeID object which is used to gather the bits.  Just 
/// call one of the Add* functions defined in the FoldingSetImpl::NodeID class.
/// NOTE: That the folding set does not own the nodes and it is the
/// responsibility of the user to dispose of the nodes.
///
/// Eg.
///    class MyNode : public FoldingSetNode {
///    private:
///      std::string Name;
///      unsigned Value;
///    public:
///      MyNode(const char *N, unsigned V) : Name(N), Value(V) {}
///       ...
///      void Profile(FoldingSetNodeID &ID) {
///        ID.AddString(Name);
///        ID.AddInteger(Value);
///       }
///       ...
///     };
///
/// To define the folding set itself use the FoldingSet template;
///
/// Eg.
///    FoldingSet<MyNode> MyFoldingSet;
///
/// Four public methods are available to manipulate the folding set; 
///
/// 1) If you have an existing node that you want add to the set but unsure
/// that the node might already exist then call;
///
///    MyNode *M = MyFoldingSet.GetOrInsertNode(N);
///
/// If The result is equal to the input then the node has been inserted.
/// Otherwise, the result is the node existing in the folding set, and the
/// input can be discarded (use the result instead.)
///
/// 2) If you are ready to construct a node but want to check if it already
/// exists, then call FindNodeOrInsertPos with a FoldingSetNodeID of the bits to
/// check;
///
///   FoldingSetNodeID ID;
///   ID.AddString(Name);
///   ID.AddInteger(Value);
///   void *InsertPoint;
///
///    MyNode *M = MyFoldingSet.FindNodeOrInsertPos(ID, InsertPoint);
///
/// If found then M with be non-NULL, else InsertPoint will point to where it
/// should be inserted using InsertNode.
///
/// 3) If you get a NULL result from FindNodeOrInsertPos then you can as a new
/// node with FindNodeOrInsertPos;
///
///    InsertNode(N, InsertPoint);
///
/// 4) Finally, if you want to remove a node from the folding set call;
///
///    bool WasRemoved = RemoveNode(N);
///
/// The result indicates whether the node existed in the folding set.


//===----------------------------------------------------------------------===//
/// FoldingSetImpl - Implements the folding set functionality.  The main
/// structure is an array of buckets.  Each bucket is indexed by the hash of
/// the nodes it contains.  The bucket itself points to the nodes contained
/// in the bucket via a singly linked list.  The last node in the list points
/// back to the bucket to facilitate node removal.
/// 
class FoldingSetImpl {
protected:
  /// Buckets - Array of bucket chains.
  ///
  void **Buckets;
  
  /// NumBuckets - Length of the Buckets array.  Always a power of 2.
  ///
  unsigned NumBuckets;
  
  /// NumNodes - Number of nodes in the folding set. Growth occurs when NumNodes
  /// is greater than twice the number of buckets.
  unsigned NumNodes;
  
public:
  explicit FoldingSetImpl(unsigned Log2InitSize = 6);
  virtual ~FoldingSetImpl();
  
  // Forward declaration.
  class Node;

  //===--------------------------------------------------------------------===//
  /// NodeID - This class is used to gather all the unique data bits of a
  /// node.  When all the bits are gathered this class is used to produce a
  /// hash value for the node.  
  ///
  class NodeID {
    /// Bits - Vector of all the data bits that make the node unique.
    /// Use a SmallVector to avoid a heap allocation in the common case.
    SmallVector<unsigned, 32> Bits;
    
  public:
    NodeID() {}
    
    /// getRawData - Return the ith entry in the Bits data.
    ///
    unsigned getRawData(unsigned i) const {
      return Bits[i];
    }
    
    /// Add* - Add various data types to Bit data.
    ///
    void AddPointer(const void *Ptr);
    void AddInteger(signed I);
    void AddInteger(unsigned I);
    void AddInteger(int64_t I);
    void AddInteger(uint64_t I);
    void AddFloat(float F);
    void AddDouble(double D);
    void AddAPFloat(const APFloat& apf);
    void AddString(const std::string &String);
    
    /// ComputeHash - Compute a strong hash value for this NodeID, used to 
    /// lookup the node in the FoldingSetImpl.
    unsigned ComputeHash() const;
    
    /// operator== - Used to compare two nodes to each other.
    ///
    bool operator==(const NodeID &RHS) const;
  };

  //===--------------------------------------------------------------------===//
  /// Node - This class is used to maintain the singly linked bucket list in
  /// a folding set.
  ///
  class Node {
  private:
    // NextInFoldingSetBucket - next link in the bucket list.
    void *NextInFoldingSetBucket;
    
  public:

    Node() : NextInFoldingSetBucket(0) {}
    
    // Accessors
    void *getNextInBucket() const { return NextInFoldingSetBucket; }
    void SetNextInBucket(void *N) { NextInFoldingSetBucket = N; }
  };

  /// RemoveNode - Remove a node from the folding set, returning true if one
  /// was removed or false if the node was not in the folding set.
  bool RemoveNode(Node *N);
  
  /// GetOrInsertNode - If there is an existing simple Node exactly
  /// equal to the specified node, return it.  Otherwise, insert 'N' and return
  /// it instead.
  Node *GetOrInsertNode(Node *N);
  
  /// FindNodeOrInsertPos - Look up the node specified by ID.  If it exists,
  /// return it.  If not, return the insertion token that will make insertion
  /// faster.
  Node *FindNodeOrInsertPos(const NodeID &ID, void *&InsertPos);
  
  /// InsertNode - Insert the specified node into the folding set, knowing that
  /// it is not already in the folding set.  InsertPos must be obtained from 
  /// FindNodeOrInsertPos.
  void InsertNode(Node *N, void *InsertPos);
  
  /// size - Returns the number of nodes in the folding set.
  unsigned size() const { return NumNodes; }
  
private:

  /// GrowHashTable - Double the size of the hash table and rehash everything.
  ///
  void GrowHashTable();
  
protected:

  /// GetNodeProfile - Instantiations of the FoldingSet template implement
  /// this function to gather data bits for the given node.
  virtual void GetNodeProfile(NodeID &ID, Node *N) const = 0;
};

// Convenience types to hide the implementation of the folding set.
typedef FoldingSetImpl::Node FoldingSetNode;
typedef FoldingSetImpl::NodeID FoldingSetNodeID;

template<class T> class FoldingSetIterator;

//===----------------------------------------------------------------------===//
/// FoldingSet - This template class is used to instantiate a specialized
/// implementation of the folding set to the node class T.  T must be a 
/// subclass of FoldingSetNode and implement a Profile function.
///
template<class T> class FoldingSet : public FoldingSetImpl {
private:
  /// GetNodeProfile - Each instantiatation of the FoldingSet needs to provide a
  /// way to convert nodes into a unique specifier.
  virtual void GetNodeProfile(NodeID &ID, Node *N) const {
    T *TN = static_cast<T *>(N);
    TN->Profile(ID);
  }
  
public:
  explicit FoldingSet(unsigned Log2InitSize = 6)
  : FoldingSetImpl(Log2InitSize)
  {}
  
  typedef FoldingSetIterator<T> iterator;
  iterator begin() { return iterator(Buckets); }
  iterator end() { return iterator(Buckets+NumBuckets); }

  typedef FoldingSetIterator<const T> const_iterator;
  const_iterator begin() const { return const_iterator(Buckets); }
  const_iterator end() const { return const_iterator(Buckets+NumBuckets); }

  /// GetOrInsertNode - If there is an existing simple Node exactly
  /// equal to the specified node, return it.  Otherwise, insert 'N' and
  /// return it instead.
  T *GetOrInsertNode(Node *N) {
    return static_cast<T *>(FoldingSetImpl::GetOrInsertNode(N));
  }
  
  /// FindNodeOrInsertPos - Look up the node specified by ID.  If it exists,
  /// return it.  If not, return the insertion token that will make insertion
  /// faster.
  T *FindNodeOrInsertPos(const FoldingSetNodeID &ID, void *&InsertPos) {
    return static_cast<T *>(FoldingSetImpl::FindNodeOrInsertPos(ID, InsertPos));
  }
};

//===----------------------------------------------------------------------===//
/// FoldingSetIteratorImpl - This is the common iterator support shared by all
/// folding sets, which knows how to walk the folding set hash table.
class FoldingSetIteratorImpl {
protected:
  FoldingSetNode *NodePtr;
  FoldingSetIteratorImpl(void **Bucket);
  void advance();
  
public:
  bool operator==(const FoldingSetIteratorImpl &RHS) const {
    return NodePtr == RHS.NodePtr;
  }
  bool operator!=(const FoldingSetIteratorImpl &RHS) const {
    return NodePtr != RHS.NodePtr;
  }
};


template<class T>
class FoldingSetIterator : public FoldingSetIteratorImpl {
public:
  FoldingSetIterator(void **Bucket) : FoldingSetIteratorImpl(Bucket) {}
  
  T &operator*() const {
    return *static_cast<T*>(NodePtr);
  }
  
  T *operator->() const {
    return static_cast<T*>(NodePtr);
  }
  
  inline FoldingSetIterator& operator++() {          // Preincrement
    advance();
    return *this;
  }
  FoldingSetIterator operator++(int) {        // Postincrement
    FoldingSetIterator tmp = *this; ++*this; return tmp;
  }
};

} // End of namespace llvm.


#endif

