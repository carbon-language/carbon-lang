//===--- ImmutableSet.h - Immutable (functional) set interface --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the ImutAVLTree and ImmutableSet classes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_IMSET_H
#define LLVM_ADT_IMSET_H

#include "llvm/Support/Allocator.h"
#include "llvm/ADT/FoldingSet.h"
#include <cassert>

namespace llvm {
  
//===----------------------------------------------------------------------===//    
// Immutable AVL-Tree Definition.
//===----------------------------------------------------------------------===//

template <typename ImutInfo> class ImutAVLFactory;

template <typename ImutInfo> class ImutAVLTreeInOrderIterator;
  
template <typename ImutInfo >
class ImutAVLTree : public FoldingSetNode {
public:
  typedef typename ImutInfo::key_type_ref   key_type_ref;
  typedef typename ImutInfo::value_type     value_type;
  typedef typename ImutInfo::value_type_ref value_type_ref;

  typedef ImutAVLFactory<ImutInfo>          Factory;
  friend class ImutAVLFactory<ImutInfo>;
  
  typedef ImutAVLTreeInOrderIterator<ImutInfo>  iterator;
  
  //===----------------------------------------------------===//  
  // Public Interface.
  //===----------------------------------------------------===//  
  
  /// getLeft - Returns a pointer to the left subtree.  This value
  ///  is NULL if there is no left subtree.
  ImutAVLTree* getLeft() const { 
    assert (!isMutable() && "Node is incorrectly marked mutable.");
    
    return reinterpret_cast<ImutAVLTree*>(Left);
  }
  
  /// getRight - Returns a pointer to the right subtree.  This value is
  ///  NULL if there is no right subtree.
  ImutAVLTree* getRight() const { return Right; }  
  
  
  /// getHeight - Returns the height of the tree.  A tree with no subtrees
  ///  has a height of 1.
  unsigned getHeight() const { return Height; }  
  
  /// getValue - Returns the data value associated with the tree node.
  const value_type& getValue() const { return Value; }
  
  /// find - Finds the subtree associated with the specified key value.
  ///  This method returns NULL if no matching subtree is found.
  ImutAVLTree* find(key_type_ref K) {
    ImutAVLTree *T = this;
    
    while (T) {
      key_type_ref CurrentKey = ImutInfo::KeyOfValue(T->getValue());
      
      if (ImutInfo::isEqual(K,CurrentKey))
        return T;
      else if (ImutInfo::isLess(K,CurrentKey))
        T = T->getLeft();
      else
        T = T->getRight();
    }
    
    return NULL;
  }
  
  /// size - Returns the number of nodes in the tree, which includes
  ///  both leaves and non-leaf nodes.
  unsigned size() const {
    unsigned n = 1;
    
    if (const ImutAVLTree* L = getLeft())  n += L->size();
    if (const ImutAVLTree* R = getRight()) n += R->size();
    
    return n;
  }
  
  /// begin - Returns an iterator that iterates over the nodes of the tree
  ///  in an inorder traversal.  The returned iterator thus refers to the
  ///  the tree node with the minimum data element.
  iterator begin() const { return iterator(this); }
  
  /// end - Returns an iterator for the tree that denotes the end of an
  ///  inorder traversal.
  iterator end() const { return iterator(); }
  
  /// isEqual - Compares two trees for structural equality and returns true
  ///   if they are equal.  This worst case performance of this operation is
  //    linear in the sizes of the trees.
  bool isEqual(const ImutAVLTree& RHS) const {
    if (&RHS == this)
      return true;
    
    iterator LItr = begin(), LEnd = end();
    iterator RItr = RHS.begin(), REnd = RHS.end();
    
    while (LItr != LEnd && RItr != REnd) {
      if (*LItr == *RItr) {
        LItr.SkipSubTree();
        RItr.SkipSubTree();
        continue;
      }
      
      // FIXME: need to compare data values, not key values, but our
      // traits don't support this yet.
      if (!ImutInfo::isEqual(ImutInfo::KeyOfValue(LItr->getValue()),
                             ImutInfo::KeyOfValue(RItr->getValue())))
        return false;
      
      ++LItr;
      ++RItr;
    }
    
    return LItr == LEnd && RItr == REnd;
  }

  /// isNotEqual - Compares two trees for structural inequality.  Performance
  ///  is the same is isEqual.
  bool isNotEqual(const ImutAVLTree& RHS) const { return !isEqual(RHS); }
  
  /// contains - Returns true if this tree contains a subtree (node) that
  ///  has an data element that matches the specified key.  Complexity
  ///  is logarithmic in the size of the tree.
  bool contains(const key_type_ref K) { return (bool) find(K); }
  
  /// foreach - A member template the accepts invokes operator() on a functor
  ///  object (specifed by Callback) for every node/subtree in the tree.
  ///  Nodes are visited using an inorder traversal.
  template <typename Callback>
  void foreach(Callback& C) {
    if (ImutAVLTree* L = getLeft()) L->foreach(C);
    
    C(Value);    
    
    if (ImutAVLTree* R = getRight()) R->foreach(C);
  }
  
  /// verify - A utility method that checks that the balancing and
  ///  ordering invariants of the tree are satisifed.  It is a recursive
  ///  method that returns the height of the tree, which is then consumed
  ///  by the enclosing verify call.  External callers should ignore the
  ///  return value.  An invalid tree will cause an assertion to fire in
  ///  a debug build.
  unsigned verify() const {
    unsigned HL = getLeft() ? getLeft()->verify() : 0;
    unsigned HR = getRight() ? getRight()->verify() : 0;
    
    assert (getHeight() == ( HL > HR ? HL : HR ) + 1 
            && "Height calculation wrong.");
    
    assert ((HL > HR ? HL-HR : HR-HL) <= 2
            && "Balancing invariant violated.");
    
    
    assert (!getLeft()
            || ImutInfo::isLess(ImutInfo::KeyOfValue(getLeft()->getValue()),
                                ImutInfo::KeyOfValue(getValue()))
            && "Value in left child is not less that current value.");
    
    
    assert (!getRight()
            || ImutInfo::isLess(ImutInfo::KeyOfValue(getValue()),
                                ImutInfo::KeyOfValue(getRight()->getValue()))
            && "Current value is not less that value of right child.");
    
    return getHeight();
  }  
  
  //===----------------------------------------------------===//  
  // Internal Values.
  //===----------------------------------------------------===//
  
private:
  uintptr_t        Left;
  ImutAVLTree*     Right;
  unsigned         Height;
  value_type       Value;
  
  //===----------------------------------------------------===//  
  // Profiling or FoldingSet.
  //===----------------------------------------------------===//

private:

  /// Profile - Generates a FoldingSet profile for a tree node before it is
  ///   created.  This is used by the ImutAVLFactory when creating
  ///   trees.
  static inline
  void Profile(FoldingSetNodeID& ID, ImutAVLTree* L, ImutAVLTree* R,
               value_type_ref V) {    
    ID.AddPointer(L);
    ID.AddPointer(R);
    ImutInfo::Profile(ID,V);
  }
  
public:

  /// Profile - Generates a FoldingSet profile for an existing tree node.
  void Profile(FoldingSetNodeID& ID) {
    Profile(ID,getSafeLeft(),getRight(),getValue());    
  }
  
  //===----------------------------------------------------===//    
  // Internal methods (node manipulation; used by Factory).
  //===----------------------------------------------------===//
  
private:
  
  enum { Mutable = 0x1 };
  
  /// ImutAVLTree - Internal constructor that is only called by
  ///   ImutAVLFactory.
  ImutAVLTree(ImutAVLTree* l, ImutAVLTree* r, value_type_ref v, unsigned height)
  : Left(reinterpret_cast<uintptr_t>(l) | Mutable),
  Right(r), Height(height), Value(v) {}
  
  
  /// isMutable - Returns true if the left and right subtree references
  ///  (as well as height) can be changed.  If this method returns false,
  ///  the tree is truly immutable.  Trees returned from an ImutAVLFactory
  ///  object should always have this method return true.  Further, if this
  ///  method returns false for an instance of ImutAVLTree, all subtrees
  ///  will also have this method return false.  The converse is not true.
  bool isMutable() const { return Left & Mutable; }
  
  /// getSafeLeft - Returns the pointer to the left tree by always masking
  ///  out the mutable bit.  This is used internally by ImutAVLFactory,
  ///  as no trees returned to the client should have the mutable flag set.
  ImutAVLTree* getSafeLeft() const { 
    return reinterpret_cast<ImutAVLTree*>(Left & ~Mutable);
  }
  
  //===----------------------------------------------------===//    
  // Mutating operations.  A tree root can be manipulated as
  // long as its reference has not "escaped" from internal 
  // methods of a factory object (see below).  When a tree
  // pointer is externally viewable by client code, the 
  // internal "mutable bit" is cleared to mark the tree 
  // immutable.  Note that a tree that still has its mutable
  // bit set may have children (subtrees) that are themselves
  // immutable.
  //===----------------------------------------------------===//
  
  
  /// MarkImmutable - Clears the mutable flag for a tree.  After this happens,
  ///   it is an error to call setLeft(), setRight(), and setHeight().  It
  ///   is also then safe to call getLeft() instead of getSafeLeft().  
  void MarkImmutable() {
    assert (isMutable() && "Mutable flag already removed.");
    Left &= ~Mutable;
  }
  
  /// setLeft - Changes the reference of the left subtree.  Used internally
  ///   by ImutAVLFactory.
  void setLeft(ImutAVLTree* NewLeft) {
    assert (isMutable() && 
            "Only a mutable tree can have its left subtree changed.");
    
    Left = reinterpret_cast<uintptr_t>(NewLeft) | Mutable;
  }
  
  /// setRight - Changes the reference of the right subtree.  Used internally
  ///  by ImutAVLFactory.
  void setRight(ImutAVLTree* NewRight) {
    assert (isMutable() &&
            "Only a mutable tree can have its right subtree changed.");
    
    Right = NewRight;
  }
  
  /// setHeight - Changes the height of the tree.  Used internally by
  ///  ImutAVLFactory.
  void setHeight(unsigned h) {
    assert (isMutable() && "Only a mutable tree can have its height changed.");
    Height = h;
  }
};

//===----------------------------------------------------------------------===//    
// Immutable AVL-Tree Factory class.
//===----------------------------------------------------------------------===//

template <typename ImutInfo >  
class ImutAVLFactory {
  typedef ImutAVLTree<ImutInfo> TreeTy;
  typedef typename TreeTy::value_type_ref value_type_ref;
  typedef typename TreeTy::key_type_ref   key_type_ref;
  
  typedef FoldingSet<TreeTy> CacheTy;
  
  CacheTy Cache;  
  BumpPtrAllocator Allocator;    
  
  //===--------------------------------------------------===//    
  // Public interface.
  //===--------------------------------------------------===//
  
public:
  ImutAVLFactory() {}
  
  TreeTy* Add(TreeTy* T, value_type_ref V) {
    T = Add_internal(V,T);
    MarkImmutable(T);
    return T;
  }
  
  TreeTy* Remove(TreeTy* T, key_type_ref V) {
    T = Remove_internal(V,T);
    MarkImmutable(T);
    return T;
  }
  
  TreeTy* GetEmptyTree() const { return NULL; }
  
  BumpPtrAllocator& getAllocator() { return Allocator; }
  
  //===--------------------------------------------------===//    
  // A bunch of quick helper functions used for reasoning
  // about the properties of trees and their children.
  // These have succinct names so that the balancing code
  // is as terse (and readable) as possible.
  //===--------------------------------------------------===//
private:
  
  bool           isEmpty(TreeTy* T) const { return !T; }
  unsigned        Height(TreeTy* T) const { return T ? T->getHeight() : 0; }  
  TreeTy*           Left(TreeTy* T) const { return T->getSafeLeft(); }
  TreeTy*          Right(TreeTy* T) const { return T->getRight(); }  
  value_type_ref   Value(TreeTy* T) const { return T->Value; }
  
  unsigned IncrementHeight(TreeTy* L, TreeTy* R) const {
    unsigned hl = Height(L);
    unsigned hr = Height(R);
    return ( hl > hr ? hl : hr ) + 1;
  }
  
  //===--------------------------------------------------===//    
  // "CreateNode" is used to generate new tree roots that link
  // to other trees.  The functon may also simply move links
  // in an existing root if that root is still marked mutable.
  // This is necessary because otherwise our balancing code
  // would leak memory as it would create nodes that are
  // then discarded later before the finished tree is
  // returned to the caller.
  //===--------------------------------------------------===//
  
  TreeTy* CreateNode(TreeTy* L, value_type_ref V, TreeTy* R) {
    FoldingSetNodeID ID;      
    TreeTy::Profile(ID,L,R,V);      
    void* InsertPos;
    
    if (TreeTy* T = Cache.FindNodeOrInsertPos(ID,InsertPos))
      return T;
    
    assert (InsertPos != NULL);
    
    // Allocate the new tree node and insert it into the cache.
    TreeTy* T = (TreeTy*) Allocator.Allocate<TreeTy>();    
    new (T) TreeTy(L,R,V,IncrementHeight(L,R));
    Cache.InsertNode(T,InsertPos);

    return T;      
  }
  
  TreeTy* CreateNode(TreeTy* L, TreeTy* OldTree, TreeTy* R) {      
    assert (!isEmpty(OldTree));
    
    if (OldTree->isMutable()) {
      OldTree->setLeft(L);
      OldTree->setRight(R);
      OldTree->setHeight(IncrementHeight(L,R));
      return OldTree;
    }
    else return CreateNode(L, Value(OldTree), R);
  }
  
  /// Balance - Used by Add_internal and Remove_internal to
  ///  balance a newly created tree.
  TreeTy* Balance(TreeTy* L, value_type_ref V, TreeTy* R) {
    
    unsigned hl = Height(L);
    unsigned hr = Height(R);
    
    if (hl > hr + 2) {
      assert (!isEmpty(L) &&
              "Left tree cannot be empty to have a height >= 2.");
      
      TreeTy* LL = Left(L);
      TreeTy* LR = Right(L);
      
      if (Height(LL) >= Height(LR))
        return CreateNode(LL, L, CreateNode(LR,V,R));
      
      assert (!isEmpty(LR) &&
              "LR cannot be empty because it has a height >= 1.");
      
      TreeTy* LRL = Left(LR);
      TreeTy* LRR = Right(LR);
      
      return CreateNode(CreateNode(LL,L,LRL), LR, CreateNode(LRR,V,R));                              
    }
    else if (hr > hl + 2) {
      assert (!isEmpty(R) &&
              "Right tree cannot be empty to have a height >= 2.");
      
      TreeTy* RL = Left(R);
      TreeTy* RR = Right(R);
      
      if (Height(RR) >= Height(RL))
        return CreateNode(CreateNode(L,V,RL), R, RR);
      
      assert (!isEmpty(RL) &&
              "RL cannot be empty because it has a height >= 1.");
      
      TreeTy* RLL = Left(RL);
      TreeTy* RLR = Right(RL);
      
      return CreateNode(CreateNode(L,V,RLL), RL, CreateNode(RLR,R,RR));
    }
    else
      return CreateNode(L,V,R);
  }
  
  /// Add_internal - Creates a new tree that includes the specified
  ///  data and the data from the original tree.  If the original tree
  ///  already contained the data item, the original tree is returned.
  TreeTy* Add_internal(value_type_ref V, TreeTy* T) {
    if (isEmpty(T))
      return CreateNode(T, V, T);
    
    assert (!T->isMutable());
    
    key_type_ref K = ImutInfo::KeyOfValue(V);
    key_type_ref KCurrent = ImutInfo::KeyOfValue(Value(T));
    
    if (ImutInfo::isEqual(K,KCurrent))
      return CreateNode(Left(T), V, Right(T));
    else if (ImutInfo::isLess(K,KCurrent))
      return Balance(Add_internal(V,Left(T)), Value(T), Right(T));
    else
      return Balance(Left(T), Value(T), Add_internal(V,Right(T)));
  }
  
  /// Remove_interal - Creates a new tree that includes all the data
  ///  from the original tree except the specified data.  If the
  ///  specified data did not exist in the original tree, the original
  ///  tree is returned.
  TreeTy* Remove_internal(key_type_ref K, TreeTy* T) {
    if (isEmpty(T))
      return T;
    
    assert (!T->isMutable());
    
    key_type_ref KCurrent = ImutInfo::KeyOfValue(Value(T));
    
    if (ImutInfo::isEqual(K,KCurrent))
      return CombineLeftRightTrees(Left(T),Right(T));
    else if (ImutInfo::isLess(K,KCurrent))
      return Balance(Remove_internal(K,Left(T)), Value(T), Right(T));
    else
      return Balance(Left(T), Value(T), Remove_internal(K,Right(T)));
  }
  
  TreeTy* CombineLeftRightTrees(TreeTy* L, TreeTy* R) {
    if (isEmpty(L)) return R;      
    if (isEmpty(R)) return L;
    
    TreeTy* OldNode;          
    TreeTy* NewRight = RemoveMinBinding(R,OldNode);
    return Balance(L,Value(OldNode),NewRight);
  }
  
  TreeTy* RemoveMinBinding(TreeTy* T, TreeTy*& NodeRemoved) {
    assert (!isEmpty(T));
    
    if (isEmpty(Left(T))) {
      NodeRemoved = T;
      return Right(T);
    }
    
    return Balance(RemoveMinBinding(Left(T),NodeRemoved),Value(T),Right(T));
  }    
  
  /// MarkImmutable - Clears the mutable bits of a root and all of its
  ///  descendants.
  void MarkImmutable(TreeTy* T) {
    if (!T || !T->isMutable())
      return;
    
    T->MarkImmutable();
    MarkImmutable(Left(T));
    MarkImmutable(Right(T));
  }
};
  
  
//===----------------------------------------------------------------------===//    
// Immutable AVL-Tree Iterators.
//===----------------------------------------------------------------------===//  

template <typename ImutInfo>
class ImutAVLTreeGenericIterator {
  SmallVector<uintptr_t,20> stack;
public:
  enum VisitFlag { VisitedNone=0x0, VisitedLeft=0x1, VisitedRight=0x3, 
                   Flags=0x3 };
  
  typedef ImutAVLTree<ImutInfo> TreeTy;      
  typedef ImutAVLTreeGenericIterator<ImutInfo> _Self;

  inline ImutAVLTreeGenericIterator() {}
  inline ImutAVLTreeGenericIterator(const TreeTy* Root) {
    if (Root) stack.push_back(reinterpret_cast<uintptr_t>(Root));
  }  
  
  TreeTy* operator*() const {
    assert (!stack.empty());    
    return reinterpret_cast<TreeTy*>(stack.back() & ~Flags);
  }
  
  uintptr_t getVisitState() {
    assert (!stack.empty());
    return stack.back() & Flags;
  }
  
  
  bool AtEnd() const { return stack.empty(); }

  bool AtBeginning() const { 
    return stack.size() == 1 && getVisitState() == VisitedNone;
  }
  
  void SkipToParent() {
    assert (!stack.empty());
    stack.pop_back();
    
    if (stack.empty())
      return;
    
    switch (getVisitState()) {
      case VisitedNone:
        stack.back() |= VisitedLeft;
        break;
      case VisitedLeft:
        stack.back() |= VisitedRight;
        break;
      default:
        assert (false && "Unreachable.");            
    }
  }
  
  inline bool operator==(const _Self& x) const {
    if (stack.size() != x.stack.size())
      return false;
    
    for (unsigned i = 0 ; i < stack.size(); i++)
      if (stack[i] != x.stack[i])
        return false;
    
    return true;
  }
  
  inline bool operator!=(const _Self& x) const { return !operator==(x); }  
  
  _Self& operator++() {
    assert (!stack.empty());
    
    TreeTy* Current = reinterpret_cast<TreeTy*>(stack.back() & ~Flags);
    assert (Current);
    
    switch (getVisitState()) {
      case VisitedNone:
        if (TreeTy* L = Current->getLeft())
          stack.push_back(reinterpret_cast<uintptr_t>(L));
        else
          stack.back() |= VisitedLeft;
        
        break;
        
      case VisitedLeft:
        if (TreeTy* R = Current->getRight())
          stack.push_back(reinterpret_cast<uintptr_t>(R));
        else
          stack.back() |= VisitedRight;
        
        break;
        
      case VisitedRight:
        SkipToParent();        
        break;
        
      default:
        assert (false && "Unreachable.");
    }
    
    return *this;
  }
  
  _Self& operator--() {
    assert (!stack.empty());
    
    TreeTy* Current = reinterpret_cast<TreeTy*>(stack.back() & ~Flags);
    assert (Current);
    
    switch (getVisitState()) {
      case VisitedNone:
        stack.pop_back();
        break;
        
      case VisitedLeft:                
        stack.back() &= ~Flags; // Set state to "VisitedNone."
        
        if (TreeTy* L = Current->getLeft())
          stack.push_back(reinterpret_cast<uintptr_t>(L) | VisitedRight);
          
        break;
        
      case VisitedRight:        
        stack.back() &= ~Flags;
        stack.back() |= VisitedLeft;
        
        if (TreeTy* R = Current->getRight())
          stack.push_back(reinterpret_cast<uintptr_t>(R) | VisitedRight);
          
        break;
        
      default:
        assert (false && "Unreachable.");
    }
    
    return *this;
  }
};
  
template <typename ImutInfo>
class ImutAVLTreeInOrderIterator {
  typedef ImutAVLTreeGenericIterator<ImutInfo> InternalIteratorTy;
  InternalIteratorTy InternalItr;

public:
  typedef ImutAVLTree<ImutInfo> TreeTy;
  typedef ImutAVLTreeInOrderIterator<ImutInfo> _Self;

  ImutAVLTreeInOrderIterator(const TreeTy* Root) : InternalItr(Root) { 
    if (Root) operator++(); // Advance to first element.
  }
  
  ImutAVLTreeInOrderIterator() : InternalItr() {}

  inline bool operator==(const _Self& x) const {
    return InternalItr == x.InternalItr;
  }
  
  inline bool operator!=(const _Self& x) const { return !operator==(x); }  
  
  inline TreeTy* operator*() const { return *InternalItr; }
  inline TreeTy* operator->() const { return *InternalItr; }
  
  inline _Self& operator++() { 
    do ++InternalItr;
    while (!InternalItr.AtEnd() && 
           InternalItr.getVisitState() != InternalIteratorTy::VisitedLeft);

    return *this;
  }
  
  inline _Self& operator--() { 
    do --InternalItr;
    while (!InternalItr.AtBeginning() && 
           InternalItr.getVisitState() != InternalIteratorTy::VisitedLeft);
    
    return *this;
  }
  
  inline void SkipSubTree() {
    InternalItr.SkipToParent();
    
    while (!InternalItr.AtEnd() &&
           InternalItr.getVisitState() != InternalIteratorTy::VisitedLeft)
      ++InternalItr;        
  }
};
    
//===----------------------------------------------------------------------===//    
// Trait classes for Profile information.
//===----------------------------------------------------------------------===//

/// Generic profile template.  The default behavior is to invoke the
/// profile method of an object.  Specializations for primitive integers
/// and generic handling of pointers is done below.
template <typename T>
struct ImutProfileInfo {
  typedef const T  value_type;
  typedef const T& value_type_ref;
  
  static inline void Profile(FoldingSetNodeID& ID, value_type_ref X) {
    X.Profile(ID);
  }  
};

/// Profile traits for integers.
template <typename T>
struct ImutProfileInteger {    
  typedef const T  value_type;
  typedef const T& value_type_ref;
  
  static inline void Profile(FoldingSetNodeID& ID, value_type_ref X) {
    ID.AddInteger(X);
  }  
};

#define PROFILE_INTEGER_INFO(X)\
template<> struct ImutProfileInfo<X> : ImutProfileInteger<X> {};

PROFILE_INTEGER_INFO(char)
PROFILE_INTEGER_INFO(unsigned char)
PROFILE_INTEGER_INFO(short)
PROFILE_INTEGER_INFO(unsigned short)
PROFILE_INTEGER_INFO(unsigned)
PROFILE_INTEGER_INFO(signed)
PROFILE_INTEGER_INFO(long)
PROFILE_INTEGER_INFO(unsigned long)
PROFILE_INTEGER_INFO(long long)
PROFILE_INTEGER_INFO(unsigned long long)

#undef PROFILE_INTEGER_INFO

/// Generic profile trait for pointer types.  We treat pointers as
/// references to unique objects.
template <typename T>
struct ImutProfileInfo<T*> {
  typedef const T*   value_type;
  typedef value_type value_type_ref;
  
  static inline void Profile(FoldingSetNodeID &ID, value_type_ref X) {
    ID.AddPointer(X);
  }
};

//===----------------------------------------------------------------------===//    
// Trait classes that contain element comparison operators and type
//  definitions used by ImutAVLTree, ImmutableSet, and ImmutableMap.  These
//  inherit from the profile traits (ImutProfileInfo) to include operations
//  for element profiling.
//===----------------------------------------------------------------------===//


/// ImutContainerInfo - Generic definition of comparison operations for
///   elements of immutable containers that defaults to using
///   std::equal_to<> and std::less<> to perform comparison of elements.
template <typename T>
struct ImutContainerInfo : public ImutProfileInfo<T> {
  typedef typename ImutProfileInfo<T>::value_type      value_type;
  typedef typename ImutProfileInfo<T>::value_type_ref  value_type_ref;
  typedef value_type      key_type;
  typedef value_type_ref  key_type_ref;
  
  static inline key_type_ref KeyOfValue(value_type_ref D) { return D; }
  
  static inline bool isEqual(key_type_ref LHS, key_type_ref RHS) { 
    return std::equal_to<key_type>()(LHS,RHS);
  }
  
  static inline bool isLess(key_type_ref LHS, key_type_ref RHS) {
    return std::less<key_type>()(LHS,RHS);
  }
};

/// ImutContainerInfo - Specialization for pointer values to treat pointers
///  as references to unique objects.  Pointers are thus compared by
///  their addresses.
template <typename T>
struct ImutContainerInfo<T*> : public ImutProfileInfo<T*> {
  typedef typename ImutProfileInfo<T*>::value_type      value_type;
  typedef typename ImutProfileInfo<T*>::value_type_ref  value_type_ref;
  typedef value_type      key_type;
  typedef value_type_ref  key_type_ref;
  
  static inline key_type_ref KeyOfValue(value_type_ref D) { return D; }
  
  static inline bool isEqual(key_type_ref LHS, key_type_ref RHS) {
    return LHS == RHS;
  }
  
  static inline bool isLess(key_type_ref LHS, key_type_ref RHS) {
    return LHS < RHS;
  }
};

//===----------------------------------------------------------------------===//    
// Immutable Set
//===----------------------------------------------------------------------===//

template <typename ValT, typename ValInfo = ImutContainerInfo<ValT> >
class ImmutableSet {
public:
  typedef typename ValInfo::value_type      value_type;
  typedef typename ValInfo::value_type_ref  value_type_ref;
  
private:  
  typedef ImutAVLTree<ValInfo> TreeTy;
  TreeTy* Root;
  
  ImmutableSet(TreeTy* R) : Root(R) {}
  
public:
  
  class Factory {
    typename TreeTy::Factory F;
    
  public:
    Factory() {}
    
    /// GetEmptySet - Returns an immutable set that contains no elements.
    ImmutableSet GetEmptySet() { return ImmutableSet(F.GetEmptyTree()); }
    
    /// Add - Creates a new immutable set that contains all of the values
    ///  of the original set with the addition of the specified value.  If
    ///  the original set already included the value, then the original set is
    ///  returned and no memory is allocated.  The time and space complexity
    ///  of this operation is logarithmic in the size of the original set.
    ///  The memory allocated to represent the set is released when the
    ///  factory object that created the set is destroyed.
    ImmutableSet Add(ImmutableSet Old, value_type_ref V) {
      return ImmutableSet(F.Add(Old.Root,V));
    }
    
    /// Remove - Creates a new immutable set that contains all of the values
    ///  of the original set with the exception of the specified value.  If
    ///  the original set did not contain the value, the original set is
    ///  returned and no memory is allocated.  The time and space complexity
    ///  of this operation is logarithmic in the size of the original set.
    ///  The memory allocated to represent the set is released when the
    ///  factory object that created the set is destroyed.
    ImmutableSet Remove(ImmutableSet Old, value_type_ref V) {
      return ImmutableSet(F.Remove(Old.Root,V));
    }
    
    BumpPtrAllocator& getAllocator() { return F.getAllocator(); }

  private:
    Factory(const Factory& RHS) {};
    void operator=(const Factory& RHS) {};    
  };
  
  friend class Factory;  

  /// contains - Returns true if the set contains the specified value.
  bool contains(const value_type_ref V) const {
    return Root ? Root->contains(V) : false;
  }
  
  bool operator==(ImmutableSet RHS) const {
    return Root && RHS.Root ? Root->isEqual(*RHS.Root) : Root == RHS.Root;
  }
  
  bool operator!=(ImmutableSet RHS) const {
    return Root && RHS.Root ? Root->isNotEqual(*RHS.Root) : Root != RHS.Root;
  }
  
  /// isEmpty - Return true if the set contains no elements.
  bool isEmpty() const { return !Root; }
  
  template <typename Callback>
  void foreach(Callback& C) { if (Root) Root->foreach(C); }
  
  template <typename Callback>
  void foreach() { if (Root) { Callback C; Root->foreach(C); } }
    
  //===--------------------------------------------------===//    
  // Iterators.
  //===--------------------------------------------------===//  

  class iterator {
    typename TreeTy::iterator itr;
    
    iterator() {}
    iterator(TreeTy* t) : itr(t) {}
    friend class ImmutableSet<ValT,ValInfo>;
  public:
    inline value_type_ref operator*() const { return itr->getValue(); }
    inline iterator& operator++() { ++itr; return *this; }
    inline iterator  operator++(int) { iterator tmp(*this); ++itr; return tmp; }
    inline iterator& operator--() { --itr; return *this; }
    inline iterator  operator--(int) { iterator tmp(*this); --itr; return tmp; }
    inline bool operator==(const iterator& RHS) const { return RHS.itr == itr; }
    inline bool operator!=(const iterator& RHS) const { return RHS.itr != itr; }        
  };
  
  iterator begin() const { return iterator(Root); }
  iterator end() const { return iterator(); }  
  
  //===--------------------------------------------------===//    
  // For testing.
  //===--------------------------------------------------===//  
  
  void verify() const { if (Root) Root->verify(); }
  unsigned getHeight() const { return Root ? Root->getHeight() : 0; }
};

} // end namespace llvm

#endif
