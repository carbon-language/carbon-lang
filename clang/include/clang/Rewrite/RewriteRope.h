//===--- RewriteRope.h - Rope specialized for rewriter ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the RewriteRope class, which is a powerful string class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_REWRITEROPE_H
#define LLVM_CLANG_REWRITEROPE_H

#include "llvm/ADT/iterator"
#include <list>
#include <cstring>

#include "llvm/Support/Casting.h"

//#define USE_ROPE_VECTOR

namespace clang {

struct RopeRefCountString {
  unsigned RefCount;
  char Data[1];  //  Variable sized.
  
  void addRef() {
    if (this) ++RefCount;
  }
  
  void dropRef() {
    if (this && --RefCount == 0)
      delete [] (char*)this;
  }
};
  
struct RopePiece {
  RopeRefCountString *StrData;
  unsigned StartOffs;
  unsigned EndOffs;
  
  RopePiece() : StrData(0), StartOffs(0), EndOffs(0) {}
  
  RopePiece(RopeRefCountString *Str, unsigned Start, unsigned End)
    : StrData(Str), StartOffs(Start), EndOffs(End) {
    StrData->addRef();
  }
  RopePiece(const RopePiece &RP)
    : StrData(RP.StrData), StartOffs(RP.StartOffs), EndOffs(RP.EndOffs) {
    StrData->addRef();
  }

  ~RopePiece() {
    StrData->dropRef();
  }
  
  void operator=(const RopePiece &RHS) {
    if (StrData != RHS.StrData) {
      StrData->dropRef();
      StrData = RHS.StrData;
      StrData->addRef();
    }
    StartOffs = RHS.StartOffs;
    EndOffs = RHS.EndOffs;
  }
  
  const char &operator[](unsigned Offset) const {
    return StrData->Data[Offset+StartOffs];
  }
  char &operator[](unsigned Offset) {
    return StrData->Data[Offset+StartOffs];
  }
  
  unsigned size() const { return EndOffs-StartOffs; }
};
  
  
  
#ifndef USE_ROPE_VECTOR
  using llvm::dyn_cast;
  using llvm::cast;

/// This is an adapted B+ Tree, ... erases don't keep the tree balanced.
  
class RopePieceBTreeNode;
struct InsertResult {
  RopePieceBTreeNode *LHS, *RHS;
};
  
class RopePieceBTreeNode {
protected:
  /// WidthFactor - This controls the number of K/V slots held in the BTree:
  /// how wide it is.  Each level of the BTree is guaranteed to have at least
  /// 'WidthFactor' elements in it (either ropepieces or children), (except the
  /// root, which may have less) and may have at most 2*WidthFactor elements.
  enum { WidthFactor = 8 };

  /// Size - This is the number of bytes of file this node (including any
  /// potential children) covers.
  unsigned Size;
  
  /// IsLeaf - True if this is an instance of RopePieceBTreeLeaf, false if it is
  /// an instance of RopePieceBTreeInterior.
  bool IsLeaf;
  
  RopePieceBTreeNode(bool isLeaf) : IsLeaf(isLeaf) {}
  ~RopePieceBTreeNode() {}
public:
  
  bool isLeaf() const { return IsLeaf; }
  unsigned size() const { return Size; }

  void Destroy();

  /// split - Split the range containing the specified offset so that we are
  /// guaranteed that there is a place to do an insertion at the specified
  /// offset.  The offset is relative, so "0" is the start of the node.  This
  /// returns true if the insertion could not be done in place, and returns
  /// information in 'Res' about the piece that is percolated up.
  bool split(unsigned Offset, InsertResult *Res);
  
  /// insert - Insert the specified ropepiece into this tree node at the
  /// specified offset.  The offset is relative, so "0" is the start of the
  /// node.  This returns true if the insertion could not be done in place, and
  /// returns information in 'Res' about the piece that is percolated up.
  bool insert(unsigned Offset, const RopePiece &R, InsertResult *Res);

  /// erase - Remove NumBytes from this node at the specified offset.  We are
  /// guaranteed that there is a split at Offset.
  void erase(unsigned Offset, unsigned NumBytes);
  
  static inline bool classof(const RopePieceBTreeNode *) { return true; }

};
  
  

  
class RopePieceBTreeLeaf : public RopePieceBTreeNode {
  /// NumPieces - This holds the number of rope pieces currently active in the
  /// Pieces array.
  unsigned char NumPieces;
  
  /// Pieces - This tracks the file chunks currently in this leaf.
  ///
  RopePiece Pieces[2*WidthFactor];
  
  /// NextLeaf - This is a pointer to the next leaf in the tree, allowing
  /// efficient in-order forward iteration of the tree without traversal.
  const RopePieceBTreeLeaf *NextLeaf;
public:
  RopePieceBTreeLeaf() : RopePieceBTreeNode(true), NextLeaf(0) {}
 
  bool isFull() const { return NumPieces == 2*WidthFactor; }

  /// clear - Remove all rope pieces from this leaf.
  void clear() {
    while (NumPieces)
      Pieces[--NumPieces] = RopePiece();
    Size = 0;
  }
  
  unsigned getNumPieces() const { return NumPieces; }
  
  const RopePiece &getPiece(unsigned i) const {
    assert(i < getNumPieces() && "Invalid piece ID");
    return Pieces[i];
  }
  
  const RopePieceBTreeLeaf *getNextLeafInOrder() const { return NextLeaf; }
  void setNextLeafInOrder(const RopePieceBTreeLeaf *NL) { NextLeaf = NL; }
  
  void FullRecomputeSizeLocally() {
    Size = 0;
    for (unsigned i = 0, e = getNumPieces(); i != e; ++i)
      Size += getPiece(i).size();
  }
  
  /// split - Split the range containing the specified offset so that we are
  /// guaranteed that there is a place to do an insertion at the specified
  /// offset.  The offset is relative, so "0" is the start of the node.  This
  /// returns true if the insertion could not be done in place, and returns
  /// information in 'Res' about the piece that is percolated up.
  bool split(unsigned Offset, InsertResult *Res);
  
  /// insert - Insert the specified ropepiece into this tree node at the
  /// specified offset.  The offset is relative, so "0" is the start of the
  /// node.  This returns true if the insertion could not be done in place, and
  /// returns information in 'Res' about the piece that is percolated up.
  bool insert(unsigned Offset, const RopePiece &R, InsertResult *Res);
  
  
  /// erase - Remove NumBytes from this node at the specified offset.  We are
  /// guaranteed that there is a split at Offset.
  void erase(unsigned Offset, unsigned NumBytes);
  
  static inline bool classof(const RopePieceBTreeLeaf *) { return true; }
  static inline bool classof(const RopePieceBTreeNode *N) {
    return N->isLeaf();
  }
};

/// split - Split the range containing the specified offset so that we are
/// guaranteed that there is a place to do an insertion at the specified
/// offset.  The offset is relative, so "0" is the start of the node.  This
/// returns true if the insertion could not be done in place, and returns
/// information in 'Res' about the piece that is percolated up.
inline bool RopePieceBTreeLeaf::split(unsigned Offset, InsertResult *Res) {
  // Find the insertion point.  We are guaranteed that there is a split at the
  // specified offset so find it.
  if (Offset == 0 || Offset == size()) {
    // Fastpath for a common case.  There is already a splitpoint at the end.
    return false;
  }
  
  // Find the piece that this offset lands in.
  unsigned PieceOffs = 0;
  unsigned i = 0;
  while (Offset >= PieceOffs+Pieces[i].size()) {
    PieceOffs += Pieces[i].size();
    ++i;
  }
  
  // If there is already a split point at the specified offset, just return
  // success.
  if (PieceOffs == Offset)
    return false;
  
  // Otherwise, we need to split piece 'i' at Offset-PieceOffs.  Convert Offset
  // to being Piece relative.
  unsigned IntraPieceOffset = Offset-PieceOffs;
  
  // We do this by shrinking the RopePiece and then doing an insert of the tail.
  RopePiece Tail(Pieces[i].StrData, Pieces[i].StartOffs+IntraPieceOffset,
                 Pieces[i].EndOffs);
  Size -= Pieces[i].size();
  Pieces[i].EndOffs = Pieces[i].StartOffs+IntraPieceOffset;
  Size += Pieces[i].size();

  return insert(Offset, Tail, Res);
}
  
  
/// insert - Insert the specified RopePiece into this tree node at the
/// specified offset.  The offset is relative, so "0" is the start of the
/// node.  This returns true if the insertion could not be done in place, and
/// returns information in 'Res' about the piece that is percolated up.
inline bool RopePieceBTreeLeaf::insert(unsigned Offset, const RopePiece &R,
                                       InsertResult *Res) {
  // If this node is not full, insert the piece.
  if (!isFull()) {
    // Find the insertion point.  We are guaranteed that there is a split at the
    // specified offset so find it.
    unsigned i = 0, e = getNumPieces();
    if (Offset == size()) {
      // Fastpath for a common case.
      i = e;
    } else {
      unsigned SlotOffs = 0;
      for (; Offset > SlotOffs; ++i)
        SlotOffs += getPiece(i).size();
      assert(SlotOffs == Offset && "Split didn't occur before insertion!");
    }
    
    // For an insertion into a non-full leaf node, just insert the value in
    // its sorted position.  This requires moving later values over.
    for (; i != e; --e)
      Pieces[e] = Pieces[e-1];
    Pieces[i] = R;
    ++NumPieces;
    Size += R.size();
    return false;
  }
  
  // Otherwise, if this is leaf is full, split it in two halves.  Since this
  // node is full, it contains 2*WidthFactor values.  We move the first
  // 'WidthFactor' values to the LHS child (which we leave in this node) and
  // move the last 'WidthFactor' values into the RHS child.
  
  // Create the new node.
  RopePieceBTreeLeaf *NewNode = new RopePieceBTreeLeaf();

  // Move over the last 'WidthFactor' values from here to NewNode.
  std::copy(&Pieces[WidthFactor], &Pieces[2*WidthFactor],
            &NewNode->Pieces[0]);
  // Replace old pieces with null RopePieces to drop refcounts.
  std::fill(&Pieces[WidthFactor], &Pieces[2*WidthFactor], RopePiece());
  
  // Decrease the number of values in the two nodes.
  NewNode->NumPieces = NumPieces = WidthFactor;
  
  // Recompute the two nodes' size.
  NewNode->FullRecomputeSizeLocally();
  FullRecomputeSizeLocally();
  
  // Update the list of leaves.
  NewNode->setNextLeafInOrder(this->getNextLeafInOrder());
  this->setNextLeafInOrder(NewNode);
  
  assert(Res && "No result location specified");
  Res->LHS = this;
  Res->RHS = NewNode;
  
  if (this->size() >= Offset)
    this->insert(Offset, R, 0 /*can't fail*/);
  else
    NewNode->insert(Offset - this->size(), R, 0 /*can't fail*/);
  return true;
}
  
/// erase - Remove NumBytes from this node at the specified offset.  We are
/// guaranteed that there is a split at Offset.
inline void RopePieceBTreeLeaf::erase(unsigned Offset, unsigned NumBytes) {
  // Since we are guaranteed that there is a split at Offset, we start by
  // finding the Piece that starts there.
  unsigned PieceOffs = 0;
  unsigned i = 0;
  for (; Offset > PieceOffs; ++i)
    PieceOffs += getPiece(i).size();
  assert(PieceOffs == Offset && "Split didn't occur before erase!");
  
  unsigned StartPiece = i;
  
  // Figure out how many pieces completely cover 'NumBytes'.  We want to remove
  // all of them.
  for (; Offset+NumBytes > PieceOffs+getPiece(i).size(); ++i)
    PieceOffs += getPiece(i).size();

  // If we exactly include the last one, include it in the region to delete.
  if (Offset+NumBytes == PieceOffs+getPiece(i).size())
    PieceOffs += getPiece(i).size(), ++i;
  
  // If we completely cover some RopePieces, erase them now.
  if (i != StartPiece) {
    unsigned NumDeleted = i-StartPiece;
    for (; i != getNumPieces(); ++i)
      Pieces[i-NumDeleted] = Pieces[i];
    
    // Drop references to dead rope pieces.
    std::fill(&Pieces[getNumPieces()-NumDeleted], &Pieces[getNumPieces()],
              RopePiece());
    NumPieces -= NumDeleted;
    
    unsigned CoverBytes = PieceOffs-Offset;
    NumBytes -= CoverBytes;
    Size -= CoverBytes;
  }

  // If we completely removed some stuff, we could be done.
  if (NumBytes == 0) return;
  
  // Okay, now might be erasing part of some Piece.  If this is the case, then
  // move the start point of the piece.
  assert(getPiece(StartPiece).size() > NumBytes);
  Pieces[StartPiece].StartOffs += NumBytes;
  
  // The size of this node just shrunk by NumBytes.
  Size -= NumBytes;
}
  
// Holds up to 2*WidthFactor children.
class RopePieceBTreeInterior : public RopePieceBTreeNode {
  /// NumChildren - This holds the number of children currently active in the
  /// Children array.
  unsigned char NumChildren;
  RopePieceBTreeNode *Children[2*WidthFactor];
public:
  RopePieceBTreeInterior() : RopePieceBTreeNode(false) {}

  RopePieceBTreeInterior(RopePieceBTreeNode *LHS, RopePieceBTreeNode *RHS)
    : RopePieceBTreeNode(false) {
    Children[0] = LHS;
    Children[1] = RHS;
    NumChildren = 2;
    Size = LHS->size() + RHS->size();
  }

  bool isFull() const { return NumChildren == 2*WidthFactor; }

  unsigned getNumChildren() const { return NumChildren; }
  const RopePieceBTreeNode *getChild(unsigned i) const {
    assert(i < NumChildren && "invalid child #");
    return Children[i];
  }
  RopePieceBTreeNode *getChild(unsigned i) {
    assert(i < NumChildren && "invalid child #");
    return Children[i];
  }
  
  void FullRecomputeSizeLocally() {
    Size = 0;
    for (unsigned i = 0, e = getNumChildren(); i != e; ++i)
      Size += getChild(i)->size();
  }
  
  
  /// split - Split the range containing the specified offset so that we are
  /// guaranteed that there is a place to do an insertion at the specified
  /// offset.  The offset is relative, so "0" is the start of the node.  This
  /// returns true if the insertion could not be done in place, and returns
  /// information in 'Res' about the piece that is percolated up.
  bool split(unsigned Offset, InsertResult *Res);
  
  
  /// insert - Insert the specified ropepiece into this tree node at the
  /// specified offset.  The offset is relative, so "0" is the start of the
  /// node.  This returns true if the insertion could not be done in place, and
  /// returns information in 'Res' about the piece that is percolated up.
  bool insert(unsigned Offset, const RopePiece &R, InsertResult *Res);
  
  /// HandleChildPiece - A child propagated an insertion result up to us.
  /// Insert the new child, and/or propagate the result further up the tree.
  bool HandleChildPiece(unsigned i, InsertResult &Res);
  
  /// erase - Remove NumBytes from this node at the specified offset.  We are
  /// guaranteed that there is a split at Offset.
  void erase(unsigned Offset, unsigned NumBytes);
  
  static inline bool classof(const RopePieceBTreeInterior *) { return true; }
  static inline bool classof(const RopePieceBTreeNode *N) {
    return !N->isLeaf(); 
  }
};
  
/// split - Split the range containing the specified offset so that we are
/// guaranteed that there is a place to do an insertion at the specified
/// offset.  The offset is relative, so "0" is the start of the node.  This
/// returns true if the insertion could not be done in place, and returns
/// information in 'Res' about the piece that is percolated up.
inline bool RopePieceBTreeInterior::split(unsigned Offset, InsertResult *Res) {
  // Figure out which child to split.
  if (Offset == 0 || Offset == size())
    return false;  // If we have an exact offset, we're already split.

  unsigned ChildOffset = 0;
  unsigned i = 0;
  for (; Offset >= ChildOffset+getChild(i)->size(); ++i)
    ChildOffset += getChild(i)->size();
  
  // If already split there, we're done.
  if (ChildOffset == Offset)
    return false;
  
  // Otherwise, recursively split the child.
  if (getChild(i)->split(Offset-ChildOffset, Res)) 
    return HandleChildPiece(i, *Res);
  return false;  // Done!
}
  
/// insert - Insert the specified ropepiece into this tree node at the
/// specified offset.  The offset is relative, so "0" is the start of the
/// node.  This returns true if the insertion could not be done in place, and
/// returns information in 'Res' about the piece that is percolated up.
inline bool RopePieceBTreeInterior::insert(unsigned Offset, const RopePiece &R,
                                           InsertResult *Res) {
  // Find the insertion point.  We are guaranteed that there is a split at the
  // specified offset so find it.
  unsigned i = 0, e = getNumChildren();
  
  unsigned ChildOffs = 0;
  if (Offset == size()) {
    // Fastpath for a common case.  Insert at end of last child.
    i = e-1;
    ChildOffs = size()-getChild(i)->size();
  } else {
    for (; Offset > ChildOffs+getChild(i)->size(); ++i)
      ChildOffs += getChild(i)->size();
  }
  
  Size += R.size();

  // Insert at the end of this child.
  if (getChild(i)->insert(Offset-ChildOffs, R, Res))
    return HandleChildPiece(i, *Res);
  
  return false;
}
  
/// HandleChildPiece - A child propagated an insertion result up to us.
/// Insert the new child, and/or propagate the result further up the tree.
inline bool RopePieceBTreeInterior::HandleChildPiece(unsigned i, 
                                                     InsertResult &Res) {
  // Otherwise the child propagated a subtree up to us as a new child.  See if
  // we have space for it here.
  if (!isFull()) {
    // Replace child 'i' with the two children specified in Res.
    if (i + 1 != getNumChildren())
      memmove(&Children[i+2], &Children[i+1],
              (getNumChildren()-i-1)*sizeof(Children[0]));
    Children[i] = Res.LHS;
    Children[i+1] = Res.RHS;
    ++NumChildren;
    return false;
  }
  
  // Okay, this node is full.  Split it in half, moving WidthFactor children to
  // a newly allocated interior node.
  
  // Create the new node.
  RopePieceBTreeInterior *NewNode = new RopePieceBTreeInterior();
  
  // Move over the last 'WidthFactor' values from here to NewNode.
  memcpy(&NewNode->Children[0], &Children[WidthFactor],
         WidthFactor*sizeof(Children[0]));
  
  // Decrease the number of values in the two nodes.
  NewNode->NumChildren = NumChildren = WidthFactor;
  
  // Finally, insert the two new children in the side the can (now) hold them.
  if (i < WidthFactor)
    this->HandleChildPiece(i, Res);
  else
    NewNode->HandleChildPiece(i-WidthFactor, Res);
  
  // Recompute the two nodes' size.
  NewNode->FullRecomputeSizeLocally();
  FullRecomputeSizeLocally();
  
  Res.LHS = this;
  Res.RHS = NewNode;
  return true;
}

/// erase - Remove NumBytes from this node at the specified offset.  We are
/// guaranteed that there is a split at Offset.
inline void RopePieceBTreeInterior::erase(unsigned Offset, unsigned NumBytes) {
  // This will shrink this node by NumBytes.
  Size -= NumBytes;
  
  // Find the first child that overlaps with Offset.
  unsigned i = 0;
  for (; Offset >= getChild(i)->size(); ++i)
    Offset -= getChild(i)->size();
  
  // Propagate the delete request into overlapping children, or completely
  // delete the children as appropriate.
  while (NumBytes) {
    RopePieceBTreeNode *CurChild = getChild(i);

    // If we are deleting something contained entirely in the child, pass on the
    // request.
    if (Offset+NumBytes < CurChild->size()) {
      CurChild->erase(Offset, NumBytes);
      return;
    }
    
    // If this deletion request starts somewhere in the middle of the child, it
    // must be deleting to the end of the child.
    if (Offset) {
      unsigned BytesFromChild = CurChild->size()-Offset;
      CurChild->erase(Offset, BytesFromChild);
      NumBytes -= BytesFromChild;
      ++i;
      continue;
    }
    
    // If the deletion request completely covers the child, delete it and move
    // the rest down.
    NumBytes -= CurChild->size();
    CurChild->Destroy();
    --NumChildren;
    if (i+1 != getNumChildren())
      memmove(&Children[i], &Children[i+1],
              (getNumChildren()-i)*sizeof(Children[0]));
  }
}
  
inline void RopePieceBTreeNode::Destroy() {
  if (RopePieceBTreeLeaf *Leaf = dyn_cast<RopePieceBTreeLeaf>(this))
    delete Leaf;
  else
    delete cast<RopePieceBTreeInterior>(this);
}

/// split - Split the range containing the specified offset so that we are
/// guaranteed that there is a place to do an insertion at the specified
/// offset.  The offset is relative, so "0" is the start of the node.  This
/// returns true if the insertion could not be done in place, and returns
/// information in 'Res' about the piece that is percolated up.
inline bool RopePieceBTreeNode::split(unsigned Offset, InsertResult *Res) {
  assert(Offset <= size() && "Invalid offset to split!");
  if (RopePieceBTreeLeaf *Leaf = dyn_cast<RopePieceBTreeLeaf>(this))
    return Leaf->split(Offset, Res);
  return cast<RopePieceBTreeInterior>(this)->split(Offset, Res);
}
  
/// insert - Insert the specified ropepiece into this tree node at the
/// specified offset.  The offset is relative, so "0" is the start of the
/// node.
inline bool RopePieceBTreeNode::insert(unsigned Offset, const RopePiece &R,
                                       InsertResult *Res) {
  assert(Offset <= size() && "Invalid offset to insert!");
  if (RopePieceBTreeLeaf *Leaf = dyn_cast<RopePieceBTreeLeaf>(this))
    return Leaf->insert(Offset, R, Res);
  return cast<RopePieceBTreeInterior>(this)->insert(Offset, R, Res);
}
  
/// erase - Remove NumBytes from this node at the specified offset.  We are
/// guaranteed that there is a split at Offset.
inline void RopePieceBTreeNode::erase(unsigned Offset, unsigned NumBytes) {
  assert(Offset+NumBytes <= size() && "Invalid offset to erase!");
  if (RopePieceBTreeLeaf *Leaf = dyn_cast<RopePieceBTreeLeaf>(this))
    return Leaf->erase(Offset, NumBytes);
  return cast<RopePieceBTreeInterior>(this)->erase(Offset, NumBytes);
}
  
  
  
/// RewritePieceBTreeIterator - Provide read-only forward iteration.
class RewritePieceBTreeIterator :
    public forward_iterator<const char, ptrdiff_t> {
  /// CurNode - The current B+Tree node that we are inspecting.
  const RopePieceBTreeLeaf *CurNode;
  /// CurPiece - The current RopePiece in the B+Tree node that we're inspecting.
  const RopePiece *CurPiece;
  /// CurChar - The current byte in the RopePiece we are pointing to.
  unsigned CurChar;
  friend class RewriteRope;
public:
  RewritePieceBTreeIterator(const RopePieceBTreeNode *N) {  // begin iterator.
    // Walk down the left side of the tree until we get to a leaf.
    while (const RopePieceBTreeInterior *IN = 
               dyn_cast<RopePieceBTreeInterior>(N))
      N = IN->getChild(0);
    
    // We must have at least one leaf.
    CurNode = cast<RopePieceBTreeLeaf>(N);
    
    // If we found a leaf that happens to be empty, skip over it until we get to
    // something full.
    while (CurNode && CurNode->getNumPieces() == 0)
      CurNode = CurNode->getNextLeafInOrder();
      
    if (CurNode != 0)
      CurPiece = &CurNode->getPiece(0);
    else  // Empty tree, this is an end() iterator.
      CurPiece = 0;
    CurChar = 0;
  }
  // end iterator
  RewritePieceBTreeIterator() : CurNode(0), CurPiece(0), CurChar(0) {}
  
  const char operator*() const {
    return (*CurPiece)[CurChar];
  }
  
  bool operator==(const RewritePieceBTreeIterator &RHS) const {
    return CurPiece == RHS.CurPiece && CurChar == RHS.CurChar;
  }
  bool operator!=(const RewritePieceBTreeIterator &RHS) const {
    return !operator==(RHS);
  }
  
  inline RewritePieceBTreeIterator& operator++() {   // Preincrement
    if (CurChar+1 < CurPiece->size())
      ++CurChar;
    else if (CurPiece != &CurNode->getPiece(CurNode->getNumPieces()-1)) {
      CurChar = 0;
      ++CurPiece;
    } else {
      // Find the next non-empty leaf node.
      do
        CurNode = CurNode->getNextLeafInOrder();
      while (CurNode && CurNode->getNumPieces() == 0);

      if (CurNode != 0)
        CurPiece = &CurNode->getPiece(0);
      else // Hit end().
        CurPiece = 0;
      CurChar = 0;
    }
    return *this;
  }
  
  inline RewritePieceBTreeIterator operator++(int) { // Postincrement
    RewritePieceBTreeIterator tmp = *this; ++*this; return tmp;
  }
};
  
  
class RopePieceBTree {
  RopePieceBTreeNode *Root;
  void operator=(const RopePieceBTree &); // DO NOT IMPLEMENT
public:
  RopePieceBTree() {
    Root = new RopePieceBTreeLeaf();
  }
  RopePieceBTree(const RopePieceBTree &RHS) {
    assert(RHS.empty() && "Can't copy non-empty tree yet");
    Root = new RopePieceBTreeLeaf();
  }
  ~RopePieceBTree() {
    Root->Destroy();
  }
  
  typedef RewritePieceBTreeIterator iterator;
  iterator begin() const { return iterator(Root); }
  iterator end() const { return iterator(); }
  unsigned size() const { return Root->size(); }
  unsigned empty() const { return size() == 0; }
  
  void clear() {
    if (RopePieceBTreeLeaf *Leaf = dyn_cast<RopePieceBTreeLeaf>(Root))
      Leaf->clear();
    else {
      Root->Destroy();
      Root = new RopePieceBTreeLeaf();
    }
  }
  
  void insert(unsigned Offset, const RopePiece &R) {
    InsertResult Result;
    // #1. Split at Offset.
    if (Root->split(Offset, &Result))
      Root = new RopePieceBTreeInterior(Result.LHS, Result.RHS);
    
    // #2. Do the insertion.
    if (Root->insert(Offset, R, &Result))
      Root = new RopePieceBTreeInterior(Result.LHS, Result.RHS);
  }

  void erase(unsigned Offset, unsigned NumBytes) {
    InsertResult Result;
    // #1. Split at Offset.
    if (Root->split(Offset, &Result))
      Root = new RopePieceBTreeInterior(Result.LHS, Result.RHS);
    
    // #2. Do the erasing.
    Root->erase(Offset, NumBytes);
  }
};

  
#endif // ifndef USE_ROPE_VECTOR
  
#ifdef USE_ROPE_VECTOR
  class RewriteRope;
  
template <typename CharType, typename PieceIterType>
class RewriteRopeIterator : 
             public bidirectional_iterator<CharType, ptrdiff_t> {
  PieceIterType CurPiece;
  unsigned CurChar;
  friend class RewriteRope;
public:
  RewriteRopeIterator(const PieceIterType &curPiece, unsigned curChar)
    : CurPiece(curPiece), CurChar(curChar) {}
  
  CharType &operator*() const {
    return (*CurPiece)[CurChar];
  }
      
  bool operator==(const RewriteRopeIterator &RHS) const {
    return CurPiece == RHS.CurPiece && CurChar == RHS.CurChar;
  }
  bool operator!=(const RewriteRopeIterator &RHS) const {
    return !operator==(RHS);
  }

  inline RewriteRopeIterator& operator++() {   // Preincrement
    if (CurChar+1 < CurPiece->size())
      ++CurChar;
    else {
      CurChar = 0;
      ++CurPiece;
    }
    return *this;
  }
 
  RewriteRopeIterator operator+(int Offset) const {
    assert(Offset >= 0 && "FIXME: Only handle forward case so far!");
    
    PieceIterType Piece = CurPiece;
    unsigned Char = CurChar;
    while (Char+Offset >= Piece->size()) {
      Offset -= Piece->size()-Char;
      ++Piece;
      Char = 0;
    }
    Char += Offset;
    return RewriteRopeIterator(Piece, Char);
  }
  
  inline RewriteRopeIterator operator++(int) { // Postincrement
    RewriteRopeIterator tmp = *this; ++*this; return tmp;
  }
};
#endif
  
  
/// RewriteRope - A powerful string class, todo generalize this.
class RewriteRope {
#ifdef USE_ROPE_VECTOR
  // FIXME: This could be significantly faster by using a balanced binary tree
  // instead of a list.
  std::list<RopePiece> Chunks;
  unsigned CurSize;
#else
  RopePieceBTree Chunks;
#endif
  
  /// We allocate space for string data out of a buffer of size AllocChunkSize.
  /// This keeps track of how much space is left.
  RopeRefCountString *AllocBuffer;
  unsigned AllocOffs;
  enum { AllocChunkSize = 4080 };
  
public:
  RewriteRope() : 
#ifdef USE_ROPE_VECTOR
  CurSize(0), 
#endif
                   AllocBuffer(0), AllocOffs(AllocChunkSize) {}
  RewriteRope(const RewriteRope &RHS) : Chunks(RHS.Chunks),
#ifdef USE_ROPE_VECTOR
  CurSize(RHS.CurSize), 
#endif
    AllocBuffer(0), AllocOffs(AllocChunkSize) {
  }

  ~RewriteRope() {
    // If we had an allocation buffer, drop our reference to it.
    AllocBuffer->dropRef();
  }
  
#ifdef USE_ROPE_VECTOR
  typedef RewriteRopeIterator<char, std::list<RopePiece>::iterator> iterator;
  typedef RewriteRopeIterator<const char, 
                           std::list<RopePiece>::const_iterator> const_iterator;

  iterator begin() { return iterator(Chunks.begin(), 0); }
  iterator end() { return iterator(Chunks.end(), 0); }
  const_iterator begin() const { return const_iterator(Chunks.begin(), 0); }
  const_iterator end() const { return const_iterator(Chunks.end(), 0); }
  
  unsigned size() const { return CurSize; }
#else
  typedef RopePieceBTree::iterator iterator;
  typedef RopePieceBTree::iterator const_iterator;
  iterator begin() const { return Chunks.begin(); }
  iterator end() const  { return Chunks.end(); }
  unsigned size() const { return Chunks.size(); }
#endif
  
  void clear() {
    Chunks.clear();
#ifdef USE_ROPE_VECTOR
   CurSize = 0;
#endif
  }
  
#ifndef USE_ROPE_VECTOR
  void assign(const char *Start, const char *End) {
    clear();
    Chunks.insert(0, MakeRopeString(Start, End));
  }
  
  void insert(unsigned Offset, const char *Start, const char *End) {
    if (Start == End) return;
    Chunks.insert(Offset, MakeRopeString(Start, End));
  }

  void erase(unsigned Offset, unsigned NumBytes) {
    if (NumBytes == 0) return;
    Chunks.erase(Offset, NumBytes);
  }
#endif

#ifdef USE_ROPE_VECTOR
  void assign(const char *Start, const char *End) {
    clear();
    Chunks.push_back(MakeRopeString(Start, End));
    CurSize = End-Start;
  }

  iterator getAtOffset(unsigned Offset) {
    assert(Offset <= CurSize && "Offset out of range!");
    if (Offset == CurSize) return iterator(Chunks.end(), 0);
    std::list<RopePiece>::iterator Piece = Chunks.begin();
    while (Offset >= Piece->size()) {
      Offset -= Piece->size();
      ++Piece;
    }
    return iterator(Piece, Offset);
  }

  const_iterator getAtOffset(unsigned Offset) const {
    assert(Offset <= CurSize && "Offset out of range!");
    if (Offset == CurSize) return const_iterator(Chunks.end(), 0);
    std::list<RopePiece>::const_iterator Piece = Chunks.begin();
    while (Offset >= Piece->size()) {
      Offset -= Piece->size();
      ++Piece;
    }
    return const_iterator(Piece, Offset);
  }
  
  
  void insert(iterator Loc, const char *Start, const char *End) {
    if (Start == End) return;
    Chunks.insert(SplitAt(Loc), MakeRopeString(Start, End));
    CurSize += End-Start;
  }

  void erase(iterator Start, iterator End) {
    if (Start == End) return;
    
    // If erase is localized within the same chunk, this is a degenerate case.
    if (Start.CurPiece == End.CurPiece) {
      RopePiece &Chunk = *Start.CurPiece;
      unsigned NumDel = End.CurChar-Start.CurChar;
      CurSize -= NumDel;

      // If deleting from start of chunk, just adjust range.
      if (Start.CurChar == 0) {
        if (Chunk.EndOffs != End.CurChar)
          Chunk.StartOffs += NumDel;
        else // Deleting entire chunk.
          Chunks.erase(End.CurPiece);
        return;
      }

      // If deleting to the end of chunk, just adjust range.
      if (End.CurChar == Chunk.size()) {
        Chunk.EndOffs -= NumDel;
        return;
      }
      
      // If deleting the middle of a chunk, split this chunk and adjust the end
      // piece.
      SplitAt(Start)->StartOffs += NumDel;
      return;
    }
    
    // Otherwise, the start chunk and the end chunk are different.
    std::list<RopePiece>::iterator CurPiece = Start.CurPiece;
      
    // Delete the end of the start chunk.  If it is the whole thing, remove it.
    {
      RopePiece &StartChunk = *CurPiece;
      unsigned NumDel = StartChunk.size()-Start.CurChar;
      CurSize -= NumDel;
      if (Start.CurChar == 0) {
        // Delete the whole chunk.
        Chunks.erase(CurPiece++);
      } else {
        // Otherwise, just move the end of chunk marker up.
        StartChunk.EndOffs -= NumDel;
        ++CurPiece;
      }
    }
    
    // If deleting a span of chunks, nuke them all now.
    while (CurPiece != End.CurPiece) {
      CurSize -= CurPiece->size();
      Chunks.erase(CurPiece++);
    }
    
    // Finally, erase the start of the end chunk if appropriate.
    if (End.CurChar != 0) {
      End.CurPiece->StartOffs += End.CurChar;
      CurSize -= End.CurChar;
    }
  }
#endif

private:
  RopePiece MakeRopeString(const char *Start, const char *End) {
    unsigned Len = End-Start;
    
    // If we have space for this string in the current alloc buffer, use it.
    if (AllocOffs+Len <= AllocChunkSize) {
      memcpy(AllocBuffer->Data+AllocOffs, Start, Len);
      AllocOffs += Len;
      return RopePiece(AllocBuffer, AllocOffs-Len, AllocOffs);
    }

    // If we don't have enough room because this specific allocation is huge,
    // just allocate a new rope piece for it alone.
    if (Len > AllocChunkSize) {
      unsigned Size = End-Start+sizeof(RopeRefCountString)-1;
      RopeRefCountString *Res = 
        reinterpret_cast<RopeRefCountString *>(new char[Size]);
      Res->RefCount = 0;
      memcpy(Res->Data, Start, End-Start);
      return RopePiece(Res, 0, End-Start);
    }
    
    // Otherwise, this was a small request but we just don't have space for it
    // Make a new chunk and share it with later allocations.
    
    // If we had an old allocation, drop our reference to it.
    if (AllocBuffer && --AllocBuffer->RefCount == 0)
      delete [] (char*)AllocBuffer;
    
    unsigned AllocSize = sizeof(RopeRefCountString)-1+AllocChunkSize;
    AllocBuffer = reinterpret_cast<RopeRefCountString *>(new char[AllocSize]);
    AllocBuffer->RefCount = 0;
    memcpy(AllocBuffer->Data, Start, Len);
    AllocOffs = Len;
    
    // Start out the new allocation with a refcount of 1, since we have an
    // internal reference to it.
    AllocBuffer->addRef();
    return RopePiece(AllocBuffer, 0, Len);
  }
  
#ifdef USE_ROPE_VECTOR

  /// SplitAt - If the specified iterator position has a non-zero character
  /// number, split the specified buffer up.  This guarantees that the specified
  /// iterator is at the start of a chunk.  Return the chunk it is at the start
  /// of.
  std::list<RopePiece>::iterator SplitAt(iterator Loc) {
    std::list<RopePiece>::iterator Chunk = Loc.CurPiece;
    
    // If the specified position is at the start of a piece, return it.
    if (Loc.CurChar == 0)
      return Chunk;
    
    // Otherwise, we have to split the specified piece in half, inserting the 
    // new piece into the list of pieces.
    
    // Make a new piece for the prefix part.
    Chunks.insert(Chunk, RopePiece(Chunk->StrData, Chunk->StartOffs,
                                   Chunk->StartOffs+Loc.CurChar));
    
    // Make the current piece refer the suffix part.
    Chunk->StartOffs += Loc.CurChar;
    
    // Return the old chunk, which is the suffix.
    return Chunk;
  }
#endif
};
  
} // end namespace clang

#endif
