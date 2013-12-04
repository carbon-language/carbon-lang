//===-- llvm/Support/GCOV.h - LLVM coverage tool ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This header provides the interface to read and write coverage files that 
// use 'gcov' format.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_GCOV_H
#define LLVM_SUPPORT_GCOV_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

class GCOVFunction;
class GCOVBlock;
class FileInfo;

namespace GCOV {
  enum GCOVVersion {
    V402,
    V404
  };
} // end GCOV namespace

/// GCOVBuffer - A wrapper around MemoryBuffer to provide GCOV specific
/// read operations.
class GCOVBuffer {
public:
  GCOVBuffer(MemoryBuffer *B) : Buffer(B), Cursor(0) {}
  
  /// readGCNOFormat - Check GCNO signature is valid at the beginning of buffer.
  bool readGCNOFormat() {
    StringRef File = Buffer->getBuffer().slice(0, 4);
    if (File != "oncg") {
      errs() << "Unexpected file type: " << File << ".\n";
      return false;
    }
    Cursor = 4;
    return true;
  }

  /// readGCDAFormat - Check GCDA signature is valid at the beginning of buffer.
  bool readGCDAFormat() {
    StringRef File = Buffer->getBuffer().slice(0, 4);
    if (File != "adcg") {
      errs() << "Unexpected file type: " << File << ".\n";
      return false;
    }
    Cursor = 4;
    return true;
  }

  /// readGCOVVersion - Read GCOV version.
  bool readGCOVVersion(GCOV::GCOVVersion &Version) {
    StringRef VersionStr = Buffer->getBuffer().slice(Cursor, Cursor+4);
    if (VersionStr == "*204") {
      Cursor += 4;
      Version = GCOV::V402;
      return true;
    }
    if (VersionStr == "*404") {
      Cursor += 4;
      Version = GCOV::V404;
      return true;
    }
    errs() << "Unexpected version: " << VersionStr << ".\n";
    return false;
  }

  /// readFunctionTag - If cursor points to a function tag then increment the
  /// cursor and return true otherwise return false.
  bool readFunctionTag() {
    StringRef Tag = Buffer->getBuffer().slice(Cursor, Cursor+4);
    if (Tag.empty() || 
        Tag[0] != '\0' || Tag[1] != '\0' ||
        Tag[2] != '\0' || Tag[3] != '\1') {
      return false;
    }
    Cursor += 4;
    return true;
  }

  /// readBlockTag - If cursor points to a block tag then increment the
  /// cursor and return true otherwise return false.
  bool readBlockTag() {
    StringRef Tag = Buffer->getBuffer().slice(Cursor, Cursor+4);
    if (Tag.empty() || 
        Tag[0] != '\0' || Tag[1] != '\0' ||
        Tag[2] != '\x41' || Tag[3] != '\x01') {
      return false;
    }
    Cursor += 4;
    return true;
  }

  /// readEdgeTag - If cursor points to an edge tag then increment the
  /// cursor and return true otherwise return false.
  bool readEdgeTag() {
    StringRef Tag = Buffer->getBuffer().slice(Cursor, Cursor+4);
    if (Tag.empty() || 
        Tag[0] != '\0' || Tag[1] != '\0' ||
        Tag[2] != '\x43' || Tag[3] != '\x01') {
      return false;
    }
    Cursor += 4;
    return true;
  }

  /// readLineTag - If cursor points to a line tag then increment the
  /// cursor and return true otherwise return false.
  bool readLineTag() {
    StringRef Tag = Buffer->getBuffer().slice(Cursor, Cursor+4);
    if (Tag.empty() || 
        Tag[0] != '\0' || Tag[1] != '\0' ||
        Tag[2] != '\x45' || Tag[3] != '\x01') {
      return false;
    }
    Cursor += 4;
    return true;
  }

  /// readArcTag - If cursor points to an gcda arc tag then increment the
  /// cursor and return true otherwise return false.
  bool readArcTag() {
    StringRef Tag = Buffer->getBuffer().slice(Cursor, Cursor+4);
    if (Tag.empty() || 
        Tag[0] != '\0' || Tag[1] != '\0' ||
        Tag[2] != '\xa1' || Tag[3] != '\1') {
      return false;
    }
    Cursor += 4;
    return true;
  }

  /// readObjectTag - If cursor points to an object summary tag then increment
  /// the cursor and return true otherwise return false.
  bool readObjectTag() {
    StringRef Tag = Buffer->getBuffer().slice(Cursor, Cursor+4);
    if (Tag.empty() ||
        Tag[0] != '\0' || Tag[1] != '\0' ||
        Tag[2] != '\0' || Tag[3] != '\xa1') {
      return false;
    }
    Cursor += 4;
    return true;
  }

  /// readProgramTag - If cursor points to a program summary tag then increment
  /// the cursor and return true otherwise return false.
  bool readProgramTag() {
    StringRef Tag = Buffer->getBuffer().slice(Cursor, Cursor+4);
    if (Tag.empty() ||
        Tag[0] != '\0' || Tag[1] != '\0' ||
        Tag[2] != '\0' || Tag[3] != '\xa3') {
      return false;
    }
    Cursor += 4;
    return true;
  }

  bool readInt(uint32_t &Val) {
    if (Buffer->getBuffer().size() < Cursor+4) {
      errs() << "Unexpected end of memory buffer: " << Cursor+4 << ".\n";
      return false;
    }
    StringRef Str = Buffer->getBuffer().slice(Cursor, Cursor+4);
    Cursor += 4;
    Val = *(const uint32_t *)(Str.data());
    return true;
  }

  bool readInt64(uint64_t &Val) {
    uint32_t Lo, Hi;
    if (!readInt(Lo) || !readInt(Hi)) return false;
    Val = ((uint64_t)Hi << 32) | Lo;
    return true;
  }

  bool readString(StringRef &Str) {
    uint32_t Len;
    if (!readInt(Len)) return false;
    Len *= 4;
    if (Buffer->getBuffer().size() < Cursor+Len) {
      errs() << "Unexpected end of memory buffer: " << Cursor+Len << ".\n";
      return false;
    }
    Str = Buffer->getBuffer().slice(Cursor, Cursor+Len).split('\0').first;
    Cursor += Len;
    return true;
  }

  uint64_t getCursor() const { return Cursor; }
  void advanceCursor(uint32_t n) { Cursor += n*4; }
private:
  MemoryBuffer *Buffer;
  uint64_t Cursor;
};

/// GCOVFile - Collects coverage information for one pair of coverage file
/// (.gcno and .gcda).
class GCOVFile {
public:
  GCOVFile() : GCNOInitialized(false), Checksum(0), Functions(), RunCount(0),
               ProgramCount(0) {}
  ~GCOVFile();
  bool readGCNO(GCOVBuffer &Buffer);
  bool readGCDA(GCOVBuffer &Buffer);
  uint32_t getChecksum() const { return Checksum; }
  void dump() const;
  void collectLineCounts(FileInfo &FI);
private:
  bool GCNOInitialized;
  GCOV::GCOVVersion Version;
  uint32_t Checksum;
  SmallVector<GCOVFunction *, 16> Functions;
  uint32_t RunCount;
  uint32_t ProgramCount;
};

struct GCOVEdge {
  GCOVEdge(GCOVBlock *S, GCOVBlock *D): Src(S), Dst(D), Count(0) {}

  GCOVBlock *Src;
  GCOVBlock *Dst;
  uint64_t Count;
};

/// GCOVFunction - Collects function information.
class GCOVFunction {
public:
  GCOVFunction(GCOVFile &P) : Parent(P), Ident(0), LineNumber(0) {}
  ~GCOVFunction();
  bool readGCNO(GCOVBuffer &Buffer, GCOV::GCOVVersion Version);
  bool readGCDA(GCOVBuffer &Buffer, GCOV::GCOVVersion Version);
  StringRef getFilename() const { return Filename; }
  void dump() const;
  void collectLineCounts(FileInfo &FI);
private:
  GCOVFile &Parent;
  uint32_t Ident;
  uint32_t LineNumber;
  StringRef Name;
  StringRef Filename;
  SmallVector<GCOVBlock *, 16> Blocks;
  SmallVector<GCOVEdge *, 16> Edges;
};

/// GCOVBlock - Collects block information.
class GCOVBlock {
public:
  typedef SmallVectorImpl<GCOVEdge *>::const_iterator EdgeIterator;

  GCOVBlock(GCOVFunction &P, uint32_t N) :
    Parent(P), Number(N), Counter(0), SrcEdges(), DstEdges(), Lines() {}
  ~GCOVBlock();
  void addSrcEdge(GCOVEdge *Edge) {
    assert(Edge->Dst == this); // up to caller to ensure edge is valid
    SrcEdges.push_back(Edge);
  }
  void addDstEdge(GCOVEdge *Edge) {
    assert(Edge->Src == this); // up to caller to ensure edge is valid
    DstEdges.push_back(Edge);
  }
  void addLine(uint32_t N) { Lines.push_back(N); }
  void addCount(size_t DstEdgeNo, uint64_t N);
  uint64_t getCount() const { return Counter; }
  size_t getNumSrcEdges() const { return SrcEdges.size(); }
  size_t getNumDstEdges() const { return DstEdges.size(); }

  EdgeIterator src_begin() const { return SrcEdges.begin(); }
  EdgeIterator src_end() const { return SrcEdges.end(); }
  EdgeIterator dst_begin() const { return DstEdges.begin(); }
  EdgeIterator dst_end() const { return DstEdges.end(); }

  void dump() const;
  void collectLineCounts(FileInfo &FI);
private:
  GCOVFunction &Parent;
  uint32_t Number;
  uint64_t Counter;
  SmallVector<GCOVEdge *, 16> SrcEdges;
  SmallVector<GCOVEdge *, 16> DstEdges;
  SmallVector<uint32_t, 16> Lines;
};

typedef SmallVector<const GCOVBlock *, 4> BlockVector;
typedef DenseMap<uint32_t, BlockVector> LineData;
class FileInfo {
public:
  void addBlockLine(StringRef Filename, uint32_t Line, const GCOVBlock *Block) {
    LineInfo[Filename][Line-1].push_back(Block);
  }
  void setRunCount(uint32_t Runs) { RunCount = Runs; }
  void setProgramCount(uint32_t Programs) { ProgramCount = Programs; }
  void print(StringRef GCNOFile, StringRef GCDAFile) const;
private:
  StringMap<LineData> LineInfo;
  uint32_t RunCount;
  uint32_t ProgramCount;
};

}

#endif
