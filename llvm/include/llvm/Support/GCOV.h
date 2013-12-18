//===- GCOV.h - LLVM coverage tool ----------------------------------------===//
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

/// GCOVOptions - A struct for passing gcov options between functions.
struct GCOVOptions {
  GCOVOptions(bool A, bool B, bool C, bool U) :
    AllBlocks(A), BranchInfo(B), BranchCount(C), UncondBranch(U) {}

  bool AllBlocks;
  bool BranchInfo;
  bool BranchCount;
  bool UncondBranch;
};

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

/// GCOVEdge - Collects edge information.
struct GCOVEdge {
  GCOVEdge(GCOVBlock *S, GCOVBlock *D): Src(S), Dst(D), Count(0) {}

  GCOVBlock *Src;
  GCOVBlock *Dst;
  uint64_t Count;
};

/// GCOVFunction - Collects function information.
class GCOVFunction {
public:
  typedef SmallVectorImpl<GCOVBlock *>::const_iterator BlockIterator;

  GCOVFunction(GCOVFile &P) : Parent(P), Ident(0), LineNumber(0) {}
  ~GCOVFunction();
  bool readGCNO(GCOVBuffer &Buffer, GCOV::GCOVVersion Version);
  bool readGCDA(GCOVBuffer &Buffer, GCOV::GCOVVersion Version);
  StringRef getName() const { return Name; }
  StringRef getFilename() const { return Filename; }
  size_t getNumBlocks() const { return Blocks.size(); }
  uint64_t getEntryCount() const;
  uint64_t getExitCount() const;

  BlockIterator block_begin() const { return Blocks.begin(); }
  BlockIterator block_end() const { return Blocks.end(); }

  void dump() const;
  void collectLineCounts(FileInfo &FI);
private:
  GCOVFile &Parent;
  uint32_t Ident;
  uint32_t Checksum;
  uint32_t LineNumber;
  StringRef Name;
  StringRef Filename;
  SmallVector<GCOVBlock *, 16> Blocks;
  SmallVector<GCOVEdge *, 16> Edges;
};

/// GCOVBlock - Collects block information.
class GCOVBlock {
  struct EdgeWeight {
    EdgeWeight(GCOVBlock *D): Dst(D), Count(0) {}

    GCOVBlock *Dst;
    uint64_t Count;
  };

  struct SortDstEdgesFunctor {
    bool operator()(const GCOVEdge *E1, const GCOVEdge *E2) {
      return E1->Dst->Number < E2->Dst->Number;
    }
  };
public:
  typedef SmallVectorImpl<GCOVEdge *>::const_iterator EdgeIterator;

  GCOVBlock(GCOVFunction &P, uint32_t N) : Parent(P), Number(N), Counter(0),
    DstEdgesAreSorted(true), SrcEdges(), DstEdges(), Lines() {}
  ~GCOVBlock();
  void addLine(uint32_t N) { Lines.push_back(N); }
  uint32_t getLastLine() const { return Lines.back(); }
  void addCount(size_t DstEdgeNo, uint64_t N);
  uint64_t getCount() const { return Counter; }

  void addSrcEdge(GCOVEdge *Edge) {
    assert(Edge->Dst == this); // up to caller to ensure edge is valid
    SrcEdges.push_back(Edge);
  }
  void addDstEdge(GCOVEdge *Edge) {
    assert(Edge->Src == this); // up to caller to ensure edge is valid
    // Check if adding this edge causes list to become unsorted.
    if (DstEdges.size() && DstEdges.back()->Dst->Number > Edge->Dst->Number)
      DstEdgesAreSorted = false;
    DstEdges.push_back(Edge);
  }
  size_t getNumSrcEdges() const { return SrcEdges.size(); }
  size_t getNumDstEdges() const { return DstEdges.size(); }
  void sortDstEdges();

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
  bool DstEdgesAreSorted;
  SmallVector<GCOVEdge *, 16> SrcEdges;
  SmallVector<GCOVEdge *, 16> DstEdges;
  SmallVector<uint32_t, 16> Lines;
};

class FileInfo {
  // It is unlikely--but possible--for multiple functions to be on the same line.
  // Therefore this typedef allows LineData.Functions to store multiple functions
  // per instance. This is rare, however, so optimize for the common case.
  typedef SmallVector<const GCOVFunction *, 1> FunctionVector;
  typedef DenseMap<uint32_t, FunctionVector> FunctionLines;
  typedef SmallVector<const GCOVBlock *, 4> BlockVector;
  typedef DenseMap<uint32_t, BlockVector> BlockLines;

  struct LineData {
    BlockLines Blocks;
    FunctionLines Functions;
  };

  struct GCOVCoverage {
    GCOVCoverage() :
      LogicalLines(0), LinesExec(0), Branches(0), BranchesExec(0),
      BranchesTaken(0) {}

    uint32_t LogicalLines;
    uint32_t LinesExec;

    uint32_t Branches;
    uint32_t BranchesExec;
    uint32_t BranchesTaken;
  };
public:
  FileInfo(const GCOVOptions &Options) :
    Options(Options), LineInfo(), RunCount(0), ProgramCount(0) {}

  void addBlockLine(StringRef Filename, uint32_t Line, const GCOVBlock *Block) {
    LineInfo[Filename].Blocks[Line-1].push_back(Block);
  }
  void addFunctionLine(StringRef Filename, uint32_t Line,
                       const GCOVFunction *Function) {
    LineInfo[Filename].Functions[Line-1].push_back(Function);
  }
  void setRunCount(uint32_t Runs) { RunCount = Runs; }
  void setProgramCount(uint32_t Programs) { ProgramCount = Programs; }
  void print(StringRef GCNOFile, StringRef GCDAFile) const;
private:
  void printFunctionSummary(raw_fd_ostream &OS,
                            const FunctionVector &Funcs) const;
  void printBlockInfo(raw_fd_ostream &OS, const GCOVBlock &Block,
                      uint32_t LineIndex, uint32_t &BlockNo) const;
  void printBranchInfo(raw_fd_ostream &OS, const GCOVBlock &Block,
                       GCOVCoverage &Coverage, uint32_t &EdgeNo) const;
  void printUncondBranchInfo(raw_fd_ostream &OS, uint32_t &EdgeNo,
                             uint64_t Count) const;
  void printFileCoverage(StringRef Filename, GCOVCoverage &Coverage) const;

  const GCOVOptions &Options;
  StringMap<LineData> LineInfo;
  uint32_t RunCount;
  uint32_t ProgramCount;
};

}

#endif
