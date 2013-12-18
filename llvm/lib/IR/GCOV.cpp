//===- GCOV.cpp - LLVM coverage tool --------------------------------------===//
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
#include <algorithm>
using namespace llvm;

//===----------------------------------------------------------------------===//
// GCOVFile implementation.

/// ~GCOVFile - Delete GCOVFile and its content.
GCOVFile::~GCOVFile() {
  DeleteContainerPointers(Functions);
}

/// readGCNO - Read GCNO buffer.
bool GCOVFile::readGCNO(GCOVBuffer &Buffer) {
  if (!Buffer.readGCNOFormat()) return false;
  if (!Buffer.readGCOVVersion(Version)) return false;

  if (!Buffer.readInt(Checksum)) return false;
  while (true) {
    if (!Buffer.readFunctionTag()) break;
    GCOVFunction *GFun = new GCOVFunction(*this);
    if (!GFun->readGCNO(Buffer, Version))
      return false;
    Functions.push_back(GFun);
  }

  GCNOInitialized = true;
  return true;
}

/// readGCDA - Read GCDA buffer. It is required that readGCDA() can only be
/// called after readGCNO().
bool GCOVFile::readGCDA(GCOVBuffer &Buffer) {
  assert(GCNOInitialized && "readGCDA() can only be called after readGCNO()");
  if (!Buffer.readGCDAFormat()) return false;
  GCOV::GCOVVersion GCDAVersion;
  if (!Buffer.readGCOVVersion(GCDAVersion)) return false;
  if (Version != GCDAVersion) {
    errs() << "GCOV versions do not match.\n";
    return false;
  }

  uint32_t GCDAChecksum;
  if (!Buffer.readInt(GCDAChecksum)) return false;
  if (Checksum != GCDAChecksum) {
    errs() << "File checksums do not match: " << Checksum << " != "
           << GCDAChecksum << ".\n";
    return false;
  }
  for (size_t i = 0, e = Functions.size(); i < e; ++i) {
    if (!Buffer.readFunctionTag()) {
      errs() << "Unexpected number of functions.\n";
      return false;
    }
    if (!Functions[i]->readGCDA(Buffer, Version))
      return false;
  }
  if (Buffer.readObjectTag()) {
    uint32_t Length;
    uint32_t Dummy;
    if (!Buffer.readInt(Length)) return false;
    if (!Buffer.readInt(Dummy)) return false; // checksum
    if (!Buffer.readInt(Dummy)) return false; // num
    if (!Buffer.readInt(RunCount)) return false;
    Buffer.advanceCursor(Length-3);
  }
  while (Buffer.readProgramTag()) {
    uint32_t Length;
    if (!Buffer.readInt(Length)) return false;
    Buffer.advanceCursor(Length);
    ++ProgramCount;
  }

  return true;
}

/// dump - Dump GCOVFile content to dbgs() for debugging purposes.
void GCOVFile::dump() const {
  for (SmallVectorImpl<GCOVFunction *>::const_iterator I = Functions.begin(),
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
  DeleteContainerPointers(Edges);
}

/// readGCNO - Read a function from the GCNO buffer. Return false if an error
/// occurs.
bool GCOVFunction::readGCNO(GCOVBuffer &Buff, GCOV::GCOVVersion Version) {
  uint32_t Dummy;
  if (!Buff.readInt(Dummy)) return false; // Function header length
  if (!Buff.readInt(Ident)) return false;
  if (!Buff.readInt(Checksum)) return false;
  if (Version != GCOV::V402) {
    uint32_t CfgChecksum;
    if (!Buff.readInt(CfgChecksum)) return false;
    if (Parent.getChecksum() != CfgChecksum) {
      errs() << "File checksums do not match: " << Parent.getChecksum()
             << " != " << CfgChecksum << " in (" << Name << ").\n";
      return false;
    }
  }
  if (!Buff.readString(Name)) return false;
  if (!Buff.readString(Filename)) return false;
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
      errs() << "Unexpected block number: " << BlockNo << " (in " << Name
             << ").\n";
      return false;
    }
    for (uint32_t i = 0, e = EdgeCount; i != e; ++i) {
      uint32_t Dst;
      if (!Buff.readInt(Dst)) return false;
      GCOVEdge *Edge = new GCOVEdge(Blocks[BlockNo], Blocks[Dst]);
      Edges.push_back(Edge);
      Blocks[BlockNo]->addDstEdge(Edge);
      Blocks[Dst]->addSrcEdge(Edge);
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
      errs() << "Unexpected block number: " << BlockNo << " (in " << Name
             << ").\n";
      return false;
    }
    GCOVBlock *Block = Blocks[BlockNo];
    if (!Buff.readInt(Dummy)) return false; // flag
    while (Buff.getCursor() != (EndPos - 4)) {
      StringRef F;
      if (!Buff.readString(F)) return false;
      if (Filename != F) {
        errs() << "Multiple sources for a single basic block: " << Filename
               << " != " << F << " (in " << Name << ").\n";
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

/// readGCDA - Read a function from the GCDA buffer. Return false if an error
/// occurs.
bool GCOVFunction::readGCDA(GCOVBuffer &Buff, GCOV::GCOVVersion Version) {
  uint32_t Dummy;
  if (!Buff.readInt(Dummy)) return false; // Function header length

  uint32_t GCDAIdent;
  if (!Buff.readInt(GCDAIdent)) return false;
  if (Ident != GCDAIdent) {
    errs() << "Function identifiers do not match: " << Ident << " != "
           << GCDAIdent << " (in " << Name << ").\n";
    return false;
  }

  uint32_t GCDAChecksum;
  if (!Buff.readInt(GCDAChecksum)) return false;
  if (Checksum != GCDAChecksum) {
    errs() << "Function checksums do not match: " << Checksum << " != "
           << GCDAChecksum << " (in " << Name << ").\n";
    return false;
  }

  uint32_t CfgChecksum;
  if (Version != GCOV::V402) {
    if (!Buff.readInt(CfgChecksum)) return false;
    if (Parent.getChecksum() != CfgChecksum) {
      errs() << "File checksums do not match: " << Parent.getChecksum()
             << " != " << CfgChecksum << " (in " << Name << ").\n";
      return false;
    }
  }

  StringRef GCDAName;
  if (!Buff.readString(GCDAName)) return false;
  if (Name != GCDAName) {
    errs() << "Function names do not match: " << Name << " != " << GCDAName
           << ".\n";
    return false;
  }

  if (!Buff.readArcTag()) {
    errs() << "Arc tag not found (in " << Name << ").\n";
    return false;
  }

  uint32_t Count;
  if (!Buff.readInt(Count)) return false;
  Count /= 2;

  // This for loop adds the counts for each block. A second nested loop is
  // required to combine the edge counts that are contained in the GCDA file.
  for (uint32_t BlockNo = 0; Count > 0; ++BlockNo) {
    // The last block is always reserved for exit block
    if (BlockNo >= Blocks.size()-1) {
      errs() << "Unexpected number of edges (in " << Name << ").\n";
      return false;
    }
    GCOVBlock &Block = *Blocks[BlockNo];
    for (size_t EdgeNo = 0, End = Block.getNumDstEdges(); EdgeNo < End;
           ++EdgeNo) {
      if (Count == 0) {
        errs() << "Unexpected number of edges (in " << Name << ").\n";
        return false;
      }
      uint64_t ArcCount;
      if (!Buff.readInt64(ArcCount)) return false;
      Block.addCount(EdgeNo, ArcCount);
      --Count;
    }
    Block.sortDstEdges();
  }
  return true;
}

/// getEntryCount - Get the number of times the function was called by
/// retrieving the entry block's count.
uint64_t GCOVFunction::getEntryCount() const {
  return Blocks.front()->getCount();
}

/// getExitCount - Get the number of times the function returned by retrieving
/// the exit block's count.
uint64_t GCOVFunction::getExitCount() const {
  return Blocks.back()->getCount();
}

/// dump - Dump GCOVFunction content to dbgs() for debugging purposes.
void GCOVFunction::dump() const {
  dbgs() <<  "===== " << Name << " @ " << Filename << ":" << LineNumber << "\n";
  for (SmallVectorImpl<GCOVBlock *>::const_iterator I = Blocks.begin(),
         E = Blocks.end(); I != E; ++I)
    (*I)->dump();
}

/// collectLineCounts - Collect line counts. This must be used after
/// reading .gcno and .gcda files.
void GCOVFunction::collectLineCounts(FileInfo &FI) {
  for (SmallVectorImpl<GCOVBlock *>::iterator I = Blocks.begin(),
         E = Blocks.end(); I != E; ++I)
    (*I)->collectLineCounts(FI);
  FI.addFunctionLine(Filename, LineNumber, this);
}

//===----------------------------------------------------------------------===//
// GCOVBlock implementation.

/// ~GCOVBlock - Delete GCOVBlock and its content.
GCOVBlock::~GCOVBlock() {
  SrcEdges.clear();
  DstEdges.clear();
  Lines.clear();
}

/// addCount - Add to block counter while storing the edge count. If the
/// destination has no outgoing edges, also update that block's count too.
void GCOVBlock::addCount(size_t DstEdgeNo, uint64_t N) {
  assert(DstEdgeNo < DstEdges.size()); // up to caller to ensure EdgeNo is valid
  DstEdges[DstEdgeNo]->Count = N;
  Counter += N;
  if (!DstEdges[DstEdgeNo]->Dst->getNumDstEdges())
    DstEdges[DstEdgeNo]->Dst->Counter += N;
}

/// sortDstEdges - Sort destination edges by block number, nop if already
/// sorted. This is required for printing branch info in the correct order.
void GCOVBlock::sortDstEdges() {
  if (!DstEdgesAreSorted) {
    SortDstEdgesFunctor SortEdges;
    std::stable_sort(DstEdges.begin(), DstEdges.end(), SortEdges);
  }
}

/// collectLineCounts - Collect line counts. This must be used after
/// reading .gcno and .gcda files.
void GCOVBlock::collectLineCounts(FileInfo &FI) {
  for (SmallVectorImpl<uint32_t>::iterator I = Lines.begin(),
         E = Lines.end(); I != E; ++I)
    FI.addBlockLine(Parent.getFilename(), *I, this);
}

/// dump - Dump GCOVBlock content to dbgs() for debugging purposes.
void GCOVBlock::dump() const {
  dbgs() << "Block : " << Number << " Counter : " << Counter << "\n";
  if (!SrcEdges.empty()) {
    dbgs() << "\tSource Edges : ";
    for (EdgeIterator I = SrcEdges.begin(), E = SrcEdges.end(); I != E; ++I) {
      const GCOVEdge *Edge = *I;
      dbgs() << Edge->Src->Number << " (" << Edge->Count << "), ";
    }
    dbgs() << "\n";
  }
  if (!DstEdges.empty()) {
    dbgs() << "\tDestination Edges : ";
    for (EdgeIterator I = DstEdges.begin(), E = DstEdges.end(); I != E; ++I) {
      const GCOVEdge *Edge = *I;
      dbgs() << Edge->Dst->Number << " (" << Edge->Count << "), ";
    }
    dbgs() << "\n";
  }
  if (!Lines.empty()) {
    dbgs() << "\tLines : ";
    for (SmallVectorImpl<uint32_t>::const_iterator I = Lines.begin(),
           E = Lines.end(); I != E; ++I)
      dbgs() << (*I) << ",";
    dbgs() << "\n";
  }
}

//===----------------------------------------------------------------------===//
// FileInfo implementation.

// Safe integer division, returns 0 if numerator is 0.
static uint32_t safeDiv(uint64_t Numerator, uint64_t Divisor) {
  if (!Numerator)
    return 0;
  return Numerator/Divisor;
}

// This custom division function mimics gcov's branch ouputs:
//   - Round to closest whole number
//   - Only output 0% or 100% if it's exactly that value
static uint32_t branchDiv(uint64_t Numerator, uint64_t Divisor) {
  if (!Numerator)
    return 0;
  if (Numerator == Divisor)
    return 100;

  uint8_t Res = (Numerator*100+Divisor/2) / Divisor;
  if (Res == 0)
    return 1;
  if (Res == 100)
    return 99;
  return Res;
}

struct formatBranchInfo {
  formatBranchInfo(const GCOVOptions &Options, uint64_t Count,
                   uint64_t Total) :
    Options(Options), Count(Count), Total(Total) {}

  void print(raw_ostream &OS) const {
    if (!Total)
      OS << "never executed";
    else if (Options.BranchCount)
      OS << "taken " << Count;
    else
      OS << "taken " << branchDiv(Count, Total) << "%";
  }

  const GCOVOptions &Options;
  uint64_t Count;
  uint64_t Total;
};

static raw_ostream &operator<<(raw_ostream &OS, const formatBranchInfo &FBI) {
  FBI.print(OS);
  return OS;
}

/// print -  Print source files with collected line count information.
void FileInfo::print(StringRef GCNOFile, StringRef GCDAFile) const {
  for (StringMap<LineData>::const_iterator I = LineInfo.begin(),
         E = LineInfo.end(); I != E; ++I) {
    StringRef Filename = I->first();
    OwningPtr<MemoryBuffer> Buff;
    if (error_code ec = MemoryBuffer::getFileOrSTDIN(Filename, Buff)) {
      errs() << Filename << ": " << ec.message() << "\n";
      return;
    }
    StringRef AllLines = Buff->getBuffer();

    std::string CovFilename = Filename.str() + ".gcov";
    std::string ErrorInfo;
    raw_fd_ostream OS(CovFilename.c_str(), ErrorInfo);
    if (!ErrorInfo.empty())
      errs() << ErrorInfo << "\n";

    OS << "        -:    0:Source:" << Filename << "\n";
    OS << "        -:    0:Graph:" << GCNOFile << "\n";
    OS << "        -:    0:Data:" << GCDAFile << "\n";
    OS << "        -:    0:Runs:" << RunCount << "\n";
    OS << "        -:    0:Programs:" << ProgramCount << "\n";

    const LineData &Line = I->second;
    for (uint32_t LineIndex = 0; !AllLines.empty(); ++LineIndex) {
      if (Options.BranchInfo) {
        FunctionLines::const_iterator FuncsIt = Line.Functions.find(LineIndex);
        if (FuncsIt != Line.Functions.end())
          printFunctionSummary(OS, FuncsIt->second);
      }

      BlockLines::const_iterator BlocksIt = Line.Blocks.find(LineIndex);
      if (BlocksIt == Line.Blocks.end()) {
        // No basic blocks are on this line. Not an executable line of code.
        OS << "        -:";
        std::pair<StringRef, StringRef> P = AllLines.split('\n');
        OS << format("%5u:", LineIndex+1) << P.first << "\n";
        AllLines = P.second;
      } else {
        const BlockVector &Blocks = BlocksIt->second;

        // Add up the block counts to form line counts.
        uint64_t LineCount = 0;
        for (BlockVector::const_iterator I = Blocks.begin(), E = Blocks.end();
               I != E; ++I) {
          const GCOVBlock *Block = *I;
          if (Options.AllBlocks) {
            // Only take the highest block count for that line.
            uint64_t BlockCount = Block->getCount();
            LineCount = LineCount > BlockCount ? LineCount : BlockCount;
          } else {
            // Sum up all of the block counts.
            LineCount += Block->getCount();
          }
        }
        if (LineCount == 0)
          OS << "    #####:";
        else
          OS << format("%9" PRIu64 ":", LineCount);

        std::pair<StringRef, StringRef> P = AllLines.split('\n');
        OS << format("%5u:", LineIndex+1) << P.first << "\n";
        AllLines = P.second;

        uint32_t BlockNo = 0;
        uint32_t EdgeNo = 0;
        for (BlockVector::const_iterator I = Blocks.begin(), E = Blocks.end();
               I != E; ++I) {
          const GCOVBlock *Block = *I;

          // Only print block and branch information at the end of the block.
          if (Block->getLastLine() != LineIndex+1)
            continue;
          if (Options.AllBlocks)
            printBlockInfo(OS, *Block, LineIndex, BlockNo);
          if (Options.BranchInfo) {
            size_t NumEdges = Block->getNumDstEdges();
            if (NumEdges > 1)
              printBranchInfo(OS, *Block, EdgeNo);
            else if (Options.UncondBranch && NumEdges == 1)
              printUncondBranchInfo(OS, EdgeNo, (*Block->dst_begin())->Count);
          }
        }
      }
    }
  }
}

/// printFunctionSummary - Print function and block summary.
void FileInfo::printFunctionSummary(raw_fd_ostream &OS,
                                    const FunctionVector &Funcs) const {
  for (FunctionVector::const_iterator I = Funcs.begin(), E = Funcs.end();
         I != E; ++I) {
    const GCOVFunction *Func = *I;
    uint64_t EntryCount = Func->getEntryCount();
    uint32_t BlocksExec = 0;
    for (GCOVFunction::BlockIterator I = Func->block_begin(),
           E = Func->block_end(); I != E; ++I) {
      const GCOVBlock *Block = *I;
      if (Block->getNumDstEdges() && Block->getCount())
          ++BlocksExec;
    }

    OS << "function " << Func->getName() << " called " << EntryCount
       << " returned " << safeDiv(Func->getExitCount()*100, EntryCount)
       << "% blocks executed "
       << safeDiv(BlocksExec*100, Func->getNumBlocks()-1) << "%\n";
  }
}

/// printBlockInfo - Output counts for each block.
void FileInfo::printBlockInfo(raw_fd_ostream &OS, const GCOVBlock &Block,
                              uint32_t LineIndex, uint32_t &BlockNo) const {
  if (Block.getCount() == 0)
    OS << "    $$$$$:";
  else
    OS << format("%9" PRIu64 ":", Block.getCount());
  OS << format("%5u-block %2u\n", LineIndex+1, BlockNo++);
}

/// printBranchInfo - Print conditional branch probabilities.
void FileInfo::printBranchInfo(raw_fd_ostream &OS, const GCOVBlock &Block,
                               uint32_t &EdgeNo) const {
  SmallVector<uint64_t, 16> BranchCounts;
  uint64_t TotalCounts = 0;
  for (GCOVBlock::EdgeIterator I = Block.dst_begin(), E = Block.dst_end();
         I != E; ++I) {
    const GCOVEdge *Edge = *I;
    BranchCounts.push_back(Edge->Count);
    TotalCounts += Edge->Count;
  }

  for (SmallVectorImpl<uint64_t>::const_iterator I = BranchCounts.begin(),
         E = BranchCounts.end(); I != E; ++I) {
    OS << format("branch %2u ", EdgeNo++)
       << formatBranchInfo(Options, *I, TotalCounts) << "\n";
  }
}

/// printUncondBranchInfo - Print unconditional branch probabilities.
void FileInfo::printUncondBranchInfo(raw_fd_ostream &OS, uint32_t &EdgeNo,
                                     uint64_t Count) const {
  OS << format("unconditional %2u ", EdgeNo++)
     << formatBranchInfo(Options, Count, Count) << "\n";
}
