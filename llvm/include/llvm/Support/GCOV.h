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
  enum GCOVFormat {
    InvalidGCOV,
    GCNO_402,
    GCNO_404,
    GCDA_402,
    GCDA_404
  };
} // end GCOV namespace

/// GCOVBuffer - A wrapper around MemoryBuffer to provide GCOV specific
/// read operations.
class GCOVBuffer {
public:
  GCOVBuffer(MemoryBuffer *B) : Buffer(B), Cursor(0) {}
  
  /// readGCOVFormat - Read GCOV signature at the beginning of buffer.
  GCOV::GCOVFormat readGCOVFormat() {
    StringRef Magic = Buffer->getBuffer().slice(0, 12);
    Cursor = 12;
    if (Magic == "oncg*404MVLL")
      return GCOV::GCNO_404;
    else if (Magic == "oncg*204MVLL")
      return GCOV::GCNO_402;
    else if (Magic == "adcg*404MVLL")
      return GCOV::GCDA_404;
    else if (Magic == "adcg*204MVLL")
      return GCOV::GCDA_402;
    
    Cursor = 0;
    return GCOV::InvalidGCOV;
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
  GCOVFile() : Functions(), RunCount(0), ProgramCount(0) {}
  ~GCOVFile();
  bool read(GCOVBuffer &Buffer);
  void dump();
  void collectLineCounts(FileInfo &FI);
private:
  SmallVector<GCOVFunction *, 16> Functions;
  uint32_t RunCount;
  uint32_t ProgramCount;
};

/// GCOVFunction - Collects function information.
class GCOVFunction {
public:
  GCOVFunction() : Ident(0), LineNumber(0) {}
  ~GCOVFunction();
  bool read(GCOVBuffer &Buffer, GCOV::GCOVFormat Format);
  StringRef getFilename() const { return Filename; }
  void dump();
  void collectLineCounts(FileInfo &FI);
private:
  uint32_t Ident;
  uint32_t LineNumber;
  StringRef Name;
  StringRef Filename;
  SmallVector<GCOVBlock *, 16> Blocks;
};

/// GCOVBlock - Collects block information.
class GCOVBlock {
public:
  GCOVBlock(GCOVFunction &P, uint32_t N) :
    Parent(P), Number(N), Counter(0), Edges(), Lines() {}
  ~GCOVBlock();
  void addEdge(uint32_t N) { Edges.push_back(N); }
  void addLine(uint32_t N) { Lines.push_back(N); }
  void addCount(uint64_t N) { Counter += N; }
  size_t getNumEdges() { return Edges.size(); }
  void dump();
  void collectLineCounts(FileInfo &FI);
private:
  GCOVFunction &Parent;
  uint32_t Number;
  uint64_t Counter;
  SmallVector<uint32_t, 16> Edges;
  SmallVector<uint32_t, 16> Lines;
};

typedef DenseMap<uint32_t, uint64_t> LineCounts;
class FileInfo {
public:
  void addLineCount(StringRef Filename, uint32_t Line, uint64_t Count) {
    LineInfo[Filename][Line-1] += Count;
  }
  void setRunCount(uint32_t Runs) { RunCount = Runs; }
  void setProgramCount(uint32_t Programs) { ProgramCount = Programs; }
  void print(raw_fd_ostream &OS, StringRef gcnoFile, StringRef gcdaFile);
private:
  StringMap<LineCounts> LineInfo;
  uint32_t RunCount;
  uint32_t ProgramCount;
};

}

#endif
