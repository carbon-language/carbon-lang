//===- ProfileInfo.cpp - Represents profile information -------------------===//
// 
//                      The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// The ProfileInfo class is used to represent profiling information read in from
// the dump file.
//
//===----------------------------------------------------------------------===//

#include "ProfileInfo.h"
#include "llvm/Module.h"
#include <iostream>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>

enum ProfilingType {
  ArgumentInfo = 1,   // The command line argument block
  FunctionInfo = 2,   // Function profiling information
  BlockInfo    = 3,   // Block profiling information
};

// ByteSwap - Byteswap 'Var' if 'Really' is true.
//
static inline unsigned ByteSwap(unsigned Var, bool Really) {
  if (!Really) return Var;
  return ((Var & (255<< 0)) << 24) | 
         ((Var & (255<< 8)) <<  8) | 
         ((Var & (255<<16)) >>  8) | 
         ((Var & (255<<24)) >> 24);
}

static void ReadProfilingBlock(const char *ToolName, FILE *F,
                               bool ShouldByteSwap,
                               std::vector<unsigned> &Data) {
  // Read the number of entries...
  unsigned NumEntries;
  if (fread(&NumEntries, sizeof(unsigned), 1, F) != 1) {
    std::cerr << ToolName << ": data packet truncated!\n";
    perror(0);
    exit(1);
  }
  NumEntries = ByteSwap(NumEntries, ShouldByteSwap);

  // Read the counts...
  std::vector<unsigned> TempSpace(NumEntries);

  // Read in the block of data...
  if (fread(&TempSpace[0], sizeof(unsigned)*NumEntries, 1, F) != 1) {
    std::cerr << ToolName << ": data packet truncated!\n";
    perror(0);
    exit(1);
  }

  // Make sure we have enough space...
  if (Data.size() < NumEntries)
    Data.resize(NumEntries);
  
  // Accumulate the data we just read into the data.
  if (!ShouldByteSwap) {
    for (unsigned i = 0; i != NumEntries; ++i)
      Data[i] += TempSpace[i];
  } else {
    for (unsigned i = 0; i != NumEntries; ++i)
      Data[i] += ByteSwap(TempSpace[i], true);
  }
}

// ProfileInfo ctor - Read the specified profiling data file, exiting the
// program if the file is invalid or broken.
//
ProfileInfo::ProfileInfo(const char *ToolName, const std::string &Filename,
                         Module &TheModule) : M(TheModule) {
  FILE *F = fopen(Filename.c_str(), "r");
  if (F == 0) {
    std::cerr << ToolName << ": Error opening '" << Filename << ": ";
    perror(0);
    exit(1);
  }

  // Keep reading packets until we run out of them.
  unsigned PacketType;
  while (fread(&PacketType, sizeof(unsigned), 1, F) == 1) {
    // If the low eight bits of the packet are zero, we must be dealing with an
    // endianness mismatch.  Byteswap all words read from the profiling
    // information.
    bool ShouldByteSwap = (char)PacketType == 0;
    PacketType = ByteSwap(PacketType, ShouldByteSwap);

    switch (PacketType) {
    case ArgumentInfo: {
      unsigned ArgLength;
      if (fread(&ArgLength, sizeof(unsigned), 1, F) != 1) {
        std::cerr << ToolName << ": arguments packet truncated!\n";
        perror(0);
        exit(1);
      }
      ArgLength = ByteSwap(ArgLength, ShouldByteSwap);

      // Read in the arguments...
      std::vector<char> Chars(ArgLength+4);

      if (ArgLength)
        if (fread(&Chars[0], (ArgLength+3) & ~3, 1, F) != 1) {
          std::cerr << ToolName << ": arguments packet truncated!\n";
          perror(0);
          exit(1);
        }
      CommandLines.push_back(std::string(&Chars[0], &Chars[ArgLength]));
      break;
    }
      
    case FunctionInfo:
      ReadProfilingBlock(ToolName, F, ShouldByteSwap, FunctionCounts);
      break;
      
    case BlockInfo:
      ReadProfilingBlock(ToolName, F, ShouldByteSwap, BlockCounts);
      break;

    default:
      std::cerr << ToolName << ": Unknown packet type #" << PacketType << "!\n";
      exit(1);
    }
  }
  
  fclose(F);
}


// getFunctionCounts - This method is used by consumers of function counting
// information.  If we do not directly have function count information, we
// compute it from other, more refined, types of profile information.
//
void ProfileInfo::getFunctionCounts(std::vector<std::pair<Function*,
                                                         unsigned> > &Counts) {
  if (FunctionCounts.empty()) {
    // Synthesize function frequency information from the number of times their
    // entry blocks were executed.
    std::vector<std::pair<BasicBlock*, unsigned> > BlockCounts;
    getBlockCounts(BlockCounts);

    for (unsigned i = 0, e = BlockCounts.size(); i != e; ++i)
      if (&BlockCounts[i].first->getParent()->front() == BlockCounts[i].first)
        Counts.push_back(std::make_pair(BlockCounts[i].first->getParent(),
                                        BlockCounts[i].second));
    return;
  }
  
  unsigned Counter = 0;
  for (Module::iterator I = M.begin(), E = M.end();
       I != E && Counter != FunctionCounts.size(); ++I)
    if (!I->isExternal())
      Counts.push_back(std::make_pair(I, FunctionCounts[Counter++]));
}

// getBlockCounts - This method is used by consumers of block counting
// information.  If we do not directly have block count information, we
// compute it from other, more refined, types of profile information.
//
void ProfileInfo::getBlockCounts(std::vector<std::pair<BasicBlock*,
                                                       unsigned> > &Counts) {
  if (BlockCounts.empty()) {
    std::cerr << "Block counts not available, and no synthesis "
              << "is implemented yet!\n";
    return;
  }

  unsigned Counter = 0;
  for (Module::iterator F = M.begin(), E = M.end(); F != E; ++F)
    for (Function::iterator BB = F->begin(), E = F->end(); BB != E; ++BB) {
      Counts.push_back(std::make_pair(BB, BlockCounts[Counter++]));
      if (Counter == BlockCounts.size())
        return;
    }
}
