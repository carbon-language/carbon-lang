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
#include <cstring>

namespace clang {
  //===--------------------------------------------------------------------===//
  // RopeRefCountString Class
  //===--------------------------------------------------------------------===//
  
  /// RopeRefCountString
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
  
  //===--------------------------------------------------------------------===//
  // RopePiece Class
  //===--------------------------------------------------------------------===//
  
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
  
  //===--------------------------------------------------------------------===//
  // RopePieceBTreeIterator Class
  //===--------------------------------------------------------------------===//
  
  /// RopePieceBTreeIterator - Provide read-only forward iteration.
  class RopePieceBTreeIterator :
      public forward_iterator<const char, ptrdiff_t> {
    /// CurNode - The current B+Tree node that we are inspecting.
    const void /*RopePieceBTreeLeaf*/ *CurNode;
    /// CurPiece - The current RopePiece in the B+Tree node that we're
    /// inspecting.
    const RopePiece *CurPiece;
    /// CurChar - The current byte in the RopePiece we are pointing to.
    unsigned CurChar;
    friend class RewriteRope;
  public:
    // begin iterator.
    RopePieceBTreeIterator(const void /*RopePieceBTreeNode*/ *N);
    // end iterator
    RopePieceBTreeIterator() : CurNode(0), CurPiece(0), CurChar(0) {}
    
    const char operator*() const {
      return (*CurPiece)[CurChar];
    }
    
    bool operator==(const RopePieceBTreeIterator &RHS) const {
      return CurPiece == RHS.CurPiece && CurChar == RHS.CurChar;
    }
    bool operator!=(const RopePieceBTreeIterator &RHS) const {
      return !operator==(RHS);
    }
    
    RopePieceBTreeIterator& operator++() {   // Preincrement
      if (CurChar+1 < CurPiece->size())
        ++CurChar;
      else
        MoveToNextPiece();
      return *this;
    }
        
    inline RopePieceBTreeIterator operator++(int) { // Postincrement
      RopePieceBTreeIterator tmp = *this; ++*this; return tmp;
    }
        
  private:
    void MoveToNextPiece();
  };
  
  //===--------------------------------------------------------------------===//
  // RopePieceBTree Class
  //===--------------------------------------------------------------------===//
  
  class RopePieceBTree {
    void /*RopePieceBTreeNode*/ *Root;
    void operator=(const RopePieceBTree &); // DO NOT IMPLEMENT
  public:
    RopePieceBTree();
    RopePieceBTree(const RopePieceBTree &RHS);
    ~RopePieceBTree();
    
    typedef RopePieceBTreeIterator iterator;
    iterator begin() const { return iterator(Root); }
    iterator end() const { return iterator(); }
    unsigned size() const;
    unsigned empty() const { return size() == 0; }
    
    void clear();
    
    void insert(unsigned Offset, const RopePiece &R);

    void erase(unsigned Offset, unsigned NumBytes);
  };

  //===--------------------------------------------------------------------===//
  // RewriteRope Class
  //===--------------------------------------------------------------------===//
  
/// RewriteRope - A powerful string class, todo generalize this.
class RewriteRope {
  RopePieceBTree Chunks;
  
  /// We allocate space for string data out of a buffer of size AllocChunkSize.
  /// This keeps track of how much space is left.
  RopeRefCountString *AllocBuffer;
  unsigned AllocOffs;
  enum { AllocChunkSize = 4080 };
  
public:
  RewriteRope() :  AllocBuffer(0), AllocOffs(AllocChunkSize) {}
  RewriteRope(const RewriteRope &RHS) 
    : Chunks(RHS.Chunks), AllocBuffer(0), AllocOffs(AllocChunkSize) {
  }

  ~RewriteRope() {
    // If we had an allocation buffer, drop our reference to it.
    AllocBuffer->dropRef();
  }
  
  typedef RopePieceBTree::iterator iterator;
  typedef RopePieceBTree::iterator const_iterator;
  iterator begin() const { return Chunks.begin(); }
  iterator end() const  { return Chunks.end(); }
  unsigned size() const { return Chunks.size(); }
  
  void clear() {
    Chunks.clear();
  }
  
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
};
  
} // end namespace clang

#endif
