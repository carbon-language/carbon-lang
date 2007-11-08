//===--- RewriteRope.h - Rope specialized for rewriter ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the RewriteRope class, which is a powerful string class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_REWRITEROPE_H
#define LLVM_CLANG_REWRITEROPE_H

#include "llvm/ADT/iterator"
#include <vector>


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
    StrData->RefCount++;
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
  
template <typename CharType, typename PieceType>
class RewriteRopeIterator : 
    public std::iterator<std::random_access_iterator_tag, CharType, ptrdiff_t> {
  PieceType *CurPiece;
  unsigned CurChar;
  friend class RewriteRope;
public:
  RewriteRopeIterator(PieceType *curPiece, unsigned curChar)
    : CurPiece(curPiece), CurChar(curChar) {}
  
  CharType &operator*() const {
    return (**CurPiece)[CurChar];
  }
      
  bool operator==(const RewriteRopeIterator &RHS) const {
    return CurPiece == RHS.CurPiece && CurChar == RHS.CurChar;
  }
  bool operator!=(const RewriteRopeIterator &RHS) const {
    return !operator==(RHS);
  }

  inline RewriteRopeIterator& operator++() {   // Preincrement
    if (CurChar+1 < (*CurPiece)->size())
      ++CurChar;
    else {
      CurChar = 0;
      ++CurPiece;
    }
    return *this;
  }
 
  inline RewriteRopeIterator operator++(int) { // Postincrement
    RewriteRopeIterator tmp = *this; ++*this; return tmp;
  }
         
  RewriteRopeIterator operator+(int Offset) const {
    assert(Offset >= 0 && "FIXME: Only handle forward case so far!");
    
    PieceType *Piece = CurPiece;
    unsigned Char = CurChar;
    while (Char+Offset >= (*Piece)->size()) {
      Offset -= (*Piece)->size()-Char;
      ++Piece;
      Char = 0;
    }
    Char += Offset;
    return RewriteRopeIterator(Piece, Char);
  }
      
  ptrdiff_t operator-(const RewriteRopeIterator &RHS) const {
    if (CurPiece < RHS.CurPiece ||
        (CurPiece == RHS.CurPiece && CurChar < RHS.CurChar))
      return -RHS.operator-(*this);

    PieceType *Piece = RHS.CurPiece;
    unsigned Char = RHS.CurChar;

    unsigned Offset = 0;
    while (Piece != CurPiece) {
      Offset += (*Piece)->size()-Char;
      Char = 0;
      ++Piece;
    }
    
    return Offset + CurChar-Char;
  }
};

  
  
/// RewriteRope - A powerful string class, todo generalize this.
class RewriteRope {
  std::vector<RopePiece*> Chunks;
  unsigned CurSize;
public:
  RewriteRope() : CurSize(0) {}
  ~RewriteRope() { clear(); }
  
  typedef RewriteRopeIterator<char, RopePiece*> iterator;
  typedef RewriteRopeIterator<const char, RopePiece* const> const_iterator;
  iterator begin() { 
    if (Chunks.empty()) return iterator(0,0);
    return iterator(&Chunks[0], 0);
  }
  iterator end() {
    if (Chunks.empty()) return iterator(0,0);
    return iterator(&Chunks[0]+Chunks.size(), 0);
  }
  
  const_iterator begin() const { 
    if (Chunks.empty()) return const_iterator(0,0);
    return const_iterator(&Chunks[0], 0);
  }
  const_iterator end() const {
    if (Chunks.empty()) return const_iterator(0,0);
    return const_iterator(&Chunks[0]+Chunks.size(), 0);
  }
  
  
  unsigned size() const { return CurSize; }
  
  void clear() {
    for (unsigned i = 0, e = Chunks.size(); i != e; ++i)
      delete Chunks[i];
    Chunks.clear();
    CurSize = 0;
  }
  
  void assign(const char *Start, const char *End) {
    clear();
    Chunks.push_back(new RopePiece(MakeRopeString(Start, End), 0,
                                   End-Start));
    CurSize = End-Start;
  }
  
  void insert(iterator Loc, const char *Start, const char *End) {
    if (Start == End) return;
    
    unsigned ChunkNo = SplitAt(Loc);
    
    RopeRefCountString *Str = MakeRopeString(Start, End);
    Chunks.insert(Chunks.begin()+ChunkNo, new RopePiece(Str, 0, End-Start));
    CurSize += End-Start;
  }

  void erase(iterator Start, iterator End) {
    if (Start == End) return;
    
    unsigned StartChunkIdx = getChunkIdx(Start);
    unsigned EndChunkIdx   = getChunkIdx(End);
    
    // If erase is localized within the same chunk, this is a degenerate case.
    if (StartChunkIdx == EndChunkIdx) {
      RopePiece *Chunk = Chunks[StartChunkIdx];
      unsigned NumDel = End.CurChar-Start.CurChar;
      CurSize -= NumDel;

      // If deleting from start of chunk, just adjust range.
      if (Start.CurChar == 0) {
        if (Chunk->EndOffs != End.CurChar) {
          Chunk->StartOffs += NumDel;
        } else {
          // Deleting entire chunk, remove it.
          delete Chunk;
          Chunks.erase(Chunks.begin()+StartChunkIdx);
        }
        return;
      }

      // If deleting to the end of chunk, just adjust range.
      if (End.CurChar == Chunk->size()) {
        Chunk->EndOffs -= NumDel;
        return;
      }
      
      // If deleting the middle of a chunk, split this chunk and adjust the end
      // piece.
      unsigned NewIdx = SplitAt(Start);
      Chunk = Chunks[NewIdx];
      Chunk->StartOffs += End.CurChar-Start.CurChar;

      return;
    }

    
    // Otherwise, the start chunk and the end chunk are different.
    
    // Delete the end of the start chunk.  If it is the whole thing, remove it.
    {
      RopePiece *StartChunk = Chunks[StartChunkIdx];
      unsigned NumDel = StartChunk->size()-Start.CurChar;
      CurSize -= NumDel;
      if (Start.CurChar == 0) {
        // Delete the whole chunk.
        delete StartChunk;
        Chunks.erase(Chunks.begin()+StartChunkIdx);
        --EndChunkIdx;
      } else {
        // Otherwise, just move the end of chunk marker up.
        StartChunk->EndOffs -= NumDel;
        ++StartChunkIdx;
      }
    }
    
    // If deleting a span of chunks, nuke them all now.
    while (StartChunkIdx != EndChunkIdx) {
      CurSize -= Chunks[StartChunkIdx]->size();
      delete Chunks[StartChunkIdx];
      Chunks.erase(Chunks.begin()+StartChunkIdx);
      --EndChunkIdx;
    }
    
    // Finally, erase the start of the end chunk if appropriate.
    if (End.CurChar != 0) {
      RopePiece *EndChunk = Chunks[EndChunkIdx];
      EndChunk->StartOffs += End.CurChar;
      CurSize -= End.CurChar;
    }
  }

private:
  RopeRefCountString *MakeRopeString(const char *Start, const char *End) {
    unsigned Size = End-Start+sizeof(RopeRefCountString)-1;
    RopeRefCountString *Res = 
      reinterpret_cast<RopeRefCountString *>(new char[Size]);
    Res->RefCount = 0;
    memcpy(Res->Data, Start, End-Start);
    return Res;
  }
  
  unsigned getChunkIdx(iterator Loc) const {
    // Return the loc idx of the specified chunk, handling empty ropes.
    return Loc.CurPiece == 0 ? 0 : Loc.CurPiece - &Chunks[0];
  }
  
  /// SplitAt - If the specified iterator position has a non-zero character
  /// number, split the specified buffer up.  This guarantees that the specified
  /// iterator is at the start of a chunk.  Return the chunk it is at the start
  /// of.
  unsigned SplitAt(iterator Loc) {
    unsigned ChunkIdx = getChunkIdx(Loc);
    
    // If the specified position is at the start of a piece, return it.
    if (Loc.CurChar == 0)
      return ChunkIdx;
    
    // Otherwise, we have to split the specified piece in half, inserting the 
    // new piece into the vector of pieces.
    RopePiece *CurPiece = *Loc.CurPiece;
    
    // Make a new piece for the prefix part.
    RopePiece *NewPiece = new RopePiece(CurPiece->StrData, CurPiece->StartOffs,
                                        CurPiece->StartOffs+Loc.CurChar);
    
    // Make the current piece refer the suffix part.
    CurPiece->StartOffs += Loc.CurChar;
    
    // Insert the new piece.
    Chunks.insert(Chunks.begin()+ChunkIdx, NewPiece);
    
    // Return the old chunk, which is the suffix.
    return ChunkIdx+1;
  }
};
  
} // end namespace clang

#endif
