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
      GCOVFunction *GFun = new GCOVFunction();
      if (!GFun->read(Buffer, Format))
        break;
      Functions.push_back(GFun);
    }
  }
  else if (isGCDAFile(Format)) {
    for (size_t i = 0, e = Functions.size(); i < e; ++i) {
      bool ReadGCDA = Functions[i]->read(Buffer, Format);
      (void)ReadGCDA;
      assert(ReadGCDA && ".gcda data does not match .gcno data");
    }
    while (Buffer.readProgramTag())
      ++ProgramCount;
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
  if (!Buff.readFunctionTag())
    return false;

  Buff.readInt(); // Function header length
  Ident = Buff.readInt(); 
  Buff.readInt(); // Checksum #1
  if (Format != GCOV::GCNO_402 && Format != GCOV::GCDA_402)
    Buff.readInt(); // Checksum #2

  Name = Buff.readString();
  if (Format == GCOV::GCNO_402 || Format == GCOV::GCNO_404)
    Filename = Buff.readString();

  if (Format == GCOV::GCDA_402 || Format == GCOV::GCDA_404) {
    Buff.readArcTag();
    uint32_t i = 0;
    uint32_t Count = Buff.readInt() / 2;

    // This for loop adds the counts for each block. A second nested loop is
    // required to combine the edge counts that are contained in the GCDA file.
    for (uint32_t Line = 0; i < Count; ++Line) {
      GCOVBlock &Block = *Blocks[Line];
      for (size_t Edge = 0, End = Block.getNumEdges(); Edge < End; ++Edge) {
        assert(i < Count && "Unexpected number of Edges!");
        Block.addCount(Buff.readInt64());
        ++i;
      }
    }
    return true;
  }

  LineNumber = Buff.readInt();

  // read blocks.
  bool BlockTagFound = Buff.readBlockTag();
  (void)BlockTagFound;
  assert(BlockTagFound && "Block Tag not found!");
  uint32_t BlockCount = Buff.readInt();
  for (uint32_t i = 0, e = BlockCount; i != e; ++i) {
    Buff.readInt(); // Block flags;
    Blocks.push_back(new GCOVBlock(i));
  }

  // read edges.
  while (Buff.readEdgeTag()) {
    uint32_t EdgeCount = (Buff.readInt() - 1) / 2;
    uint32_t BlockNo = Buff.readInt();
    assert(BlockNo < BlockCount && "Unexpected Block number!");
    for (uint32_t i = 0, e = EdgeCount; i != e; ++i) {
      Blocks[BlockNo]->addEdge(Buff.readInt());
      Buff.readInt(); // Edge flag
    }
  }

  // read line table.
  while (Buff.readLineTag()) {
    uint32_t LineTableLength = Buff.readInt();
    uint32_t EndPos = Buff.getCursor() + LineTableLength*4;
    uint32_t BlockNo = Buff.readInt();
    assert(BlockNo < BlockCount && "Unexpected Block number!");
    GCOVBlock *Block = Blocks[BlockNo];
    Buff.readInt(); // flag
    while (Buff.getCursor() != (EndPos - 4)) {
      StringRef Filename = Buff.readString();
      if (Buff.getCursor() == (EndPos - 4)) break;
      while (uint32_t L = Buff.readInt())
        Block->addLine(Filename, L);
    }
    Buff.readInt(); // flag
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
  DeleteContainerSeconds(Lines);
}

void GCOVBlock::addLine(StringRef Filename, uint32_t LineNo) {
  GCOVLines *&LinesForFile = Lines[Filename];
  if (!LinesForFile)
    LinesForFile = new GCOVLines();
  LinesForFile->add(LineNo);
}

/// collectLineCounts - Collect line counts. This must be used after
/// reading .gcno and .gcda files.
void GCOVBlock::collectLineCounts(FileInfo &FI) {
  for (StringMap<GCOVLines *>::iterator I = Lines.begin(),
         E = Lines.end(); I != E; ++I)
    I->second->collectLineCounts(FI, I->first(), Counter);
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
    for (StringMap<GCOVLines *>::iterator LI = Lines.begin(),
           LE = Lines.end(); LI != LE; ++LI) {
      dbgs() << LI->first() << " -> ";
      LI->second->dump();
      dbgs() << "\n";
    }
  }
}

//===----------------------------------------------------------------------===//
// GCOVLines implementation.

/// collectLineCounts - Collect line counts. This must be used after
/// reading .gcno and .gcda files.
void GCOVLines::collectLineCounts(FileInfo &FI, StringRef Filename, 
                                  uint64_t Count) {
  for (SmallVectorImpl<uint32_t>::iterator I = Lines.begin(),
         E = Lines.end(); I != E; ++I)
    FI.addLineCount(Filename, *I, Count);
}

/// dump - Dump GCOVLines content to dbgs() for debugging purposes.
void GCOVLines::dump() {
  for (SmallVectorImpl<uint32_t>::iterator I = Lines.begin(),
         E = Lines.end(); I != E; ++I)
    dbgs() << (*I) << ",";
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
    OS << "        -:    0:Programs:" << ProgramCount << "\n";
    LineCounts &L = LineInfo[Filename];
    OwningPtr<MemoryBuffer> Buff;
    if (error_code ec = MemoryBuffer::getFileOrSTDIN(Filename, Buff)) {
      errs() << Filename << ": " << ec.message() << "\n";
      return;
    }
    StringRef AllLines = Buff.take()->getBuffer();
    uint32_t i = 0;
    while (!AllLines.empty()) {
      if (L.find(i) != L.end()) {
        if (L[i] == 0)
          OS << "    #####:";
        else
          OS << format("%9lu:", L[i]);
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

