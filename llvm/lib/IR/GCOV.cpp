//===- GCOVr.cpp - LLVM coverage tool -------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// GCOV implements the interface to read and write coverage files that use
// 'gcov' format.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Debug.h"
#include "llvm/Support/GCOV.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MemoryObject.h"
#include "llvm/Support/system_error.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
// GCOVFile implementation.

/// ~GCOVFile - Delete GCOVFile and its content.
GCOVFile::~GCOVFile() {
  DeleteContainerPointers(Functions);
}

/// isGCDAFile - Return true if Format identifies a .gcda file.
static bool isGCDAFile(GCOV::GCOVFormat Format) {
  return Format == GCOV::GCDA_402 || Format == GCOV::GCDA_404;
}

/// isGCNOFile - Return true if Format identifies a .gcno file.
static bool isGCNOFile(GCOV::GCOVFormat Format) {
  return Format == GCOV::GCNO_402 || Format == GCOV::GCNO_404;
}

/// read - Read GCOV buffer.
bool GCOVFile::read(GCOVBuffer &Buffer) {
  GCOV::GCOVFormat Format = Buffer.readGCOVFormat();
  if (Format == GCOV::InvalidGCOV)
    return false;

  if (isGCNOFile(Format)) {
    while (true) {
      if (!Buffer.readFunctionTag()) break;
      GCOVFunction *GFun = new GCOVFunction();
      if (!GFun->read(Buffer, Format))
        return false;
      Functions.push_back(GFun);
    }
  }
  else if (isGCDAFile(Format)) {
    for (size_t i = 0, e = Functions.size(); i < e; ++i) {
      if (!Buffer.readFunctionTag()) {
        errs() << "Unexpected number of functions.\n";
        return false;
      }
      if (!Functions[i]->read(Buffer, Format))
        return false;
    }
    if (Buffer.readObjectTag()) {
      uint32_t Length;
      uint32_t Dummy;
      if (!Buffer.readInt(Length)) return false;
      if (!Buffer.readInt(Dummy)) return false; // checksum
      if (!Buffer.readInt(Dummy)) return false; // num
      if (!Buffer.readInt(RunCount)) return false;;
      Buffer.advanceCursor(Length-3);
    }
    while (Buffer.readProgramTag()) {
      uint32_t Length;
      if (!Buffer.readInt(Length)) return false;
      Buffer.advanceCursor(Length);
      ++ProgramCount;
    }
  }

  return true;
}

/// dump - Dump GCOVFile content to dbgs() for debugging purposes.
void GCOVFile::dump() {
  for (SmallVectorImpl<GCOVFunction *>::iterator I = Functions.begin(),
         E = Functions.end(); I != E; ++I)
    (*I)->dump();
}

/// collectLineCounts - Collect line counts. This must be used after
/// reading .gcno and .gcda files.
void GCOVFile::collectLineCounts(FileInfo &FI) {
  for (SmallVectorImpl<GCOVFunction *>::iterator I = Functions.begin(),
         E = Functions.end(); I != E; ++I)
    (*I)->collectLineCounts(FI);
  FI.setRunCount(RunCount);
  FI.setProgramCount(ProgramCount);
}

//===----------------------------------------------------------------------===//
// GCOVFunction implementation.

/// ~GCOVFunction - Delete GCOVFunction and its content.
GCOVFunction::~GCOVFunction() {
  DeleteContainerPointers(Blocks);
}

/// read - Read a function from the buffer. Return false if buffer cursor
/// does not point to a function tag.
bool GCOVFunction::read(GCOVBuffer &Buff, GCOV::GCOVFormat Format) {
  uint32_t Dummy;
  if (!Buff.readInt(Dummy)) return false; // Function header length
  if (!Buff.readInt(Ident)) return false;
  if (!Buff.readInt(Dummy)) return false; // Checksum #1
  if (Format != GCOV::GCNO_402 && Format != GCOV::GCDA_402)
    if (!Buff.readInt(Dummy)) return false; // Checksum #2

  if (!Buff.readString(Name)) return false;

  if (Format == GCOV::GCNO_402 || Format == GCOV::GCNO_404)
    if (!Buff.readString(Filename)) return false;

  if (Format == GCOV::GCDA_402 || Format == GCOV::GCDA_404) {
    if (!Buff.readArcTag()) {
      errs() << "Arc tag not found.\n";
      return false;
    }
    uint32_t Count;
    if (!Buff.readInt(Count)) return false;
    Count /= 2;

    // This for loop adds the counts for each block. A second nested loop is
    // required to combine the edge counts that are contained in the GCDA file.
    for (uint32_t Line = 0; Count > 0; ++Line) {
      if (Line >= Blocks.size()) {
        errs() << "Unexpected number of edges.\n";
        return false;
      }
      GCOVBlock &Block = *Blocks[Line];
      for (size_t Edge = 0, End = Block.getNumEdges(); Edge < End; ++Edge) {
        if (Count == 0) {
          errs() << "Unexpected number of edges.\n";
          return false;
        }
        uint64_t ArcCount;
        if (!Buff.readInt64(ArcCount)) return false;
        Block.addCount(ArcCount);
        --Count;
      }
    }
    return true;
  }

  if (!Buff.readInt(LineNumber)) return false;

  // read blocks.
  if (!Buff.readBlockTag()) {
    errs() << "Block tag not found.\n";
    return false;
  }
  uint32_t BlockCount;
  if (!Buff.readInt(BlockCount)) return false;
  for (uint32_t i = 0, e = BlockCount; i != e; ++i) {
    if (!Buff.readInt(Dummy)) return false; // Block flags;
    Blocks.push_back(new GCOVBlock(*this, i));
  }

  // read edges.
  while (Buff.readEdgeTag()) {
    uint32_t EdgeCount;
    if (!Buff.readInt(EdgeCount)) return false;
    EdgeCount = (EdgeCount - 1) / 2;
    uint32_t BlockNo;
    if (!Buff.readInt(BlockNo)) return false;
    if (BlockNo >= BlockCount) {
      errs() << "Unexpected block number.\n";
      return false;
    }
    for (uint32_t i = 0, e = EdgeCount; i != e; ++i) {
      uint32_t Dst;
      if (!Buff.readInt(Dst)) return false;
      Blocks[BlockNo]->addEdge(Dst);
      if (!Buff.readInt(Dummy)) return false; // Edge flag
    }
  }

  // read line table.
  while (Buff.readLineTag()) {
    uint32_t LineTableLength;
    if (!Buff.readInt(LineTableLength)) return false;
    uint32_t EndPos = Buff.getCursor() + LineTableLength*4;
    uint32_t BlockNo;
    if (!Buff.readInt(BlockNo)) return false;
    if (BlockNo >= BlockCount) {
      errs() << "Unexpected block number.\n";
      return false;
    }
    GCOVBlock *Block = Blocks[BlockNo];
    if (!Buff.readInt(Dummy)) return false; // flag
    while (Buff.getCursor() != (EndPos - 4)) {
      StringRef F;
      if (!Buff.readString(F)) return false;
      if (F != Filename) {
        errs() << "Multiple sources for a single basic block.\n";
        return false;
      }
      if (Buff.getCursor() == (EndPos - 4)) break;
      while (true) {
        uint32_t Line;
        if (!Buff.readInt(Line)) return false;
        if (!Line) break;
        Block->addLine(Line);
      }
    }
    if (!Buff.readInt(Dummy)) return false; // flag
  }
  return true;
}

/// dump - Dump GCOVFunction content to dbgs() for debugging purposes.
void GCOVFunction::dump() {
  dbgs() <<  "===== " << Name << " @ " << Filename << ":" << LineNumber << "\n";
  for (SmallVectorImpl<GCOVBlock *>::iterator I = Blocks.begin(),
         E = Blocks.end(); I != E; ++I)
    (*I)->dump();
}

/// collectLineCounts - Collect line counts. This must be used after
/// reading .gcno and .gcda files.
void GCOVFunction::collectLineCounts(FileInfo &FI) {
  for (SmallVectorImpl<GCOVBlock *>::iterator I = Blocks.begin(),
         E = Blocks.end(); I != E; ++I)
    (*I)->collectLineCounts(FI);
}

//===----------------------------------------------------------------------===//
// GCOVBlock implementation.

/// ~GCOVBlock - Delete GCOVBlock and its content.
GCOVBlock::~GCOVBlock() {
  Edges.clear();
  Lines.clear();
}

/// collectLineCounts - Collect line counts. This must be used after
/// reading .gcno and .gcda files.
void GCOVBlock::collectLineCounts(FileInfo &FI) {
  for (SmallVectorImpl<uint32_t>::iterator I = Lines.begin(),
         E = Lines.end(); I != E; ++I)
    FI.addLineCount(Parent.getFilename(), *I, Counter);
}

/// dump - Dump GCOVBlock content to dbgs() for debugging purposes.
void GCOVBlock::dump() {
  dbgs() << "Block : " << Number << " Counter : " << Counter << "\n";
  if (!Edges.empty()) {
    dbgs() << "\tEdges : ";
    for (SmallVectorImpl<uint32_t>::iterator I = Edges.begin(), E = Edges.end();
         I != E; ++I)
      dbgs() << (*I) << ",";
    dbgs() << "\n";
  }
  if (!Lines.empty()) {
    dbgs() << "\tLines : ";
    for (SmallVectorImpl<uint32_t>::iterator I = Lines.begin(),
           E = Lines.end(); I != E; ++I)
      dbgs() << (*I) << ",";
    dbgs() << "\n";
  }
}

//===----------------------------------------------------------------------===//
// FileInfo implementation.

/// print -  Print source files with collected line count information.
void FileInfo::print(raw_fd_ostream &OS, StringRef gcnoFile,
                     StringRef gcdaFile) {
  for (StringMap<LineCounts>::iterator I = LineInfo.begin(), E = LineInfo.end();
       I != E; ++I) {
    StringRef Filename = I->first();
    OS << "        -:    0:Source:" << Filename << "\n";
    OS << "        -:    0:Graph:" << gcnoFile << "\n";
    OS << "        -:    0:Data:" << gcdaFile << "\n";
    OS << "        -:    0:Runs:" << RunCount << "\n";
    OS << "        -:    0:Programs:" << ProgramCount << "\n";
    LineCounts &L = LineInfo[Filename];
    OwningPtr<MemoryBuffer> Buff;
    if (error_code ec = MemoryBuffer::getFileOrSTDIN(Filename, Buff)) {
      errs() << Filename << ": " << ec.message() << "\n";
      return;
    }
    StringRef AllLines = Buff->getBuffer();
    uint32_t i = 0;
    while (!AllLines.empty()) {
      if (L.find(i) != L.end()) {
        if (L[i] == 0)
          OS << "    #####:";
        else
          OS << format("%9" PRIu64 ":", L[i]);
      } else {
        OS << "        -:";
      }
      std::pair<StringRef, StringRef> P = AllLines.split('\n');
      if (AllLines != P.first)
        OS << format("%5u:", i+1) << P.first;
      OS << "\n";
      AllLines = P.second;
      ++i;
    }
  }
}
