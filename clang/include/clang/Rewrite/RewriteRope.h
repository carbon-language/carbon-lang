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

namespace clang {

struct RopeRefCountString {
  unsigned RefCount;
  char Data[1];  //  Variable sized.
};
  
struct RopePiece {
  RopeRefCountString *StrData;
  unsigned StartOffs;
  unsigned EndOffs;
  
  RopePiece(RopeRefCountString *Str, unsigned Start, unsigned End)
    : StrData(Str), StartOffs(Start), EndOffs(End) {
    ++StrData->RefCount;
  }
  RopePiece(const RopePiece &RP)
    : StrData(RP.StrData), StartOffs(RP.StartOffs), EndOffs(RP.EndOffs) {
      ++StrData->RefCount;
  }

  ~RopePiece() {
    if (--StrData->RefCount == 0)
      delete [] (char*)StrData;
  }
  
  const char &operator[](unsigned Offset) const {
    return StrData->Data[Offset+StartOffs];
  }
  char &operator[](unsigned Offset) {
    return StrData->Data[Offset+StartOffs];
  }
  
  unsigned size() const { return EndOffs-StartOffs; }
};

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

  
  
/// RewriteRope - A powerful string class, todo generalize this.
class RewriteRope {
  // FIXME: This could be significantly faster by using a balanced binary tree
  // instead of a list.
  std::list<RopePiece> Chunks;
  unsigned CurSize;
  
  /// We allocate space for string data out of a buffer of size AllocChunkSize.
  /// This keeps track of how much space is left.
  RopeRefCountString *AllocBuffer;
  unsigned AllocOffs;
  enum { AllocChunkSize = 4080 };
  
public:
  RewriteRope() : CurSize(0), AllocBuffer(0), AllocOffs(AllocChunkSize) {}
  ~RewriteRope() { clear(); }
  
  typedef RewriteRopeIterator<char, std::list<RopePiece>::iterator> iterator;
  typedef RewriteRopeIterator<const char, 
                           std::list<RopePiece>::const_iterator> const_iterator;

  iterator begin() { return iterator(Chunks.begin(), 0); }
  iterator end() { return iterator(Chunks.end(), 0); }
  const_iterator begin() const { return const_iterator(Chunks.begin(), 0); }
  const_iterator end() const { return const_iterator(Chunks.end(), 0); }
  
  unsigned size() const { return CurSize; }
  
  void clear() {
    Chunks.clear();
    CurSize = 0;
  }
  
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
    unsigned AllocSize = sizeof(RopeRefCountString)-1+AllocChunkSize;
    AllocBuffer = reinterpret_cast<RopeRefCountString *>(new char[AllocSize]);
    AllocBuffer->RefCount = 0;
    memcpy(AllocBuffer->Data, Start, Len);
    AllocOffs = Len;
    return RopePiece(AllocBuffer, 0, Len);
  }
  
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
};
  
} // end namespace clang

#endif
