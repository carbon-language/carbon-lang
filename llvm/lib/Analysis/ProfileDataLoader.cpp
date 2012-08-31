//===- ProfileDataLoader.cpp - Load profile information from disk ---------===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// The ProfileDataLoader class is used to load raw profiling data from the dump
// file.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/Module.h"
#include "llvm/InstrTypes.h"
#include "llvm/Analysis/ProfileDataLoader.h"
#include "llvm/Analysis/ProfileDataTypes.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/system_error.h"
#include <cstdio>
#include <cstdlib>
using namespace llvm;

namespace llvm {

template<>
char ProfileDataT<Function,BasicBlock>::ID = 0;

raw_ostream& operator<<(raw_ostream &O, const Function *F) {
  return O << F->getName();
}

raw_ostream& operator<<(raw_ostream &O, const BasicBlock *BB) {
  return O << BB->getName();
}

raw_ostream& operator<<(raw_ostream &O, std::pair<const BasicBlock *,
                        const BasicBlock *> E) {
  O << "(";

  if (E.first)
    O << E.first;
  else
    O << "0";

  O << ",";

  if (E.second)
    O << E.second;
  else
    O << "0";

  return O << ")";
}

} // namespace llvm

/// ByteSwap - Byteswap 'Var'.  Required when the compiler host and target have
/// different endianness.
static inline unsigned ByteSwap(unsigned Var) {
  return ((Var & (255U<< 0U)) << 24U) |
         ((Var & (255U<< 8U)) <<  8U) |
         ((Var & (255U<<16U)) >>  8U) |
         ((Var & (255U<<24U)) >> 24U);
}

/// AddCounts - Add 'A' and 'B', accounting for the fact that the value of one
/// (or both) may not be defined.
static unsigned AddCounts(unsigned A, unsigned B) {
  // If either value is undefined, use the other.
  // Undefined + undefined = undefined.
  if (A == ProfileDataLoader::Uncounted) return B;
  if (B == ProfileDataLoader::Uncounted) return A;

  // Saturate to the maximum storable value.  This could change taken/nottaken
  // ratios, but is presumably better than wrapping and thus potentially
  // inverting ratios.
  uint64_t tmp = (uint64_t)A + (uint64_t)B;
  if (tmp > (uint64_t)ProfileDataLoader::MaxCount)
    tmp = ProfileDataLoader::MaxCount;
  return (unsigned)tmp;
}

/// ReadProfilingData - Load 'NumEntries' items of type 'T' from file 'F'
template <typename T, unsigned N>
static void ReadProfilingData(const char *ToolName, FILE *F,
                              SmallVector<T, N> &Data, size_t NumEntries) {
  // Read in the block of data...
  if (fread(&Data[0], sizeof(T), NumEntries, F) != NumEntries) {
    report_fatal_error(std::string(ToolName) + ": Profiling data truncated");
  }
}

/// ReadProfilingNumEntries - Read how many entries are in this profiling data
/// packet.
static unsigned ReadProfilingNumEntries(const char *ToolName, FILE *F,
                                        bool ShouldByteSwap) {
  SmallVector<unsigned, 1> NumEntries(1);
  ReadProfilingData<unsigned, 1>(ToolName, F, NumEntries, 1);
  return ShouldByteSwap ? ByteSwap(NumEntries[0]) : NumEntries[0];
}

/// ReadProfilingBlock - Read the number of entries in the next profiling data
/// packet and then accumulate the entries into 'Data'.
static void ReadProfilingBlock(const char *ToolName, FILE *F,
                               bool ShouldByteSwap,
                               SmallVector<unsigned, 32> &Data) {
  // Read the number of entries...
  unsigned NumEntries = ReadProfilingNumEntries(ToolName, F, ShouldByteSwap);

  // Read in the data.
  SmallVector<unsigned, 8> TempSpace(NumEntries);
  ReadProfilingData<unsigned, 8>(ToolName, F, TempSpace, (size_t)NumEntries);

  // Make sure we have enough space ...
  if (Data.size() < NumEntries)
    Data.resize(NumEntries, ProfileDataLoader::Uncounted);

  // Accumulate the data we just read into the existing data.
  for (unsigned i = 0; i < NumEntries; ++i)
    Data[i] = AddCounts(ShouldByteSwap ? ByteSwap(TempSpace[i]) : TempSpace[i],
                        Data[i]);
}

/// ReadProfilingArgBlock - Read the command line arguments that the progam was
/// run with when the current profiling data packet(s) were generated.
static void ReadProfilingArgBlock(const char *ToolName, FILE *F,
                                  bool ShouldByteSwap,
                                  SmallVector<std::string, 1> &CommandLines) {
  // Read the number of bytes ...
  unsigned ArgLength = ReadProfilingNumEntries(ToolName, F, ShouldByteSwap);

  // Read in the arguments (if there are any to read).  Round up the length to
  // the nearest 4-byte multiple.
  SmallVector<char, 8> Args(ArgLength+4);
  if (ArgLength)
    ReadProfilingData<char, 8>(ToolName, F, Args, (ArgLength+3) & ~3);

  // Store the arguments.
  CommandLines.push_back(std::string(&Args[0], &Args[ArgLength]));
}

const unsigned ProfileDataLoader::Uncounted = ~0U;
const unsigned ProfileDataLoader::MaxCount = ~0U - 1U;

/// ProfileDataLoader ctor - Read the specified profiling data file, reporting
/// a fatal error if the file is invalid or broken.
ProfileDataLoader::ProfileDataLoader(const char *ToolName,
                                     const std::string &Filename)
  : Filename(Filename) {
  FILE *F = fopen(Filename.c_str(), "rb");
  if (F == 0)
    report_fatal_error(std::string(ToolName) + ": Error opening '" +
                       Filename + "': ");

  // Keep reading packets until we run out of them.
  unsigned PacketType;
  while (fread(&PacketType, sizeof(unsigned), 1, F) == 1) {
    // If the low eight bits of the packet are zero, we must be dealing with an
    // endianness mismatch.  Byteswap all words read from the profiling
    // information.  This can happen when the compiler host and target have
    // different endianness.
    bool ShouldByteSwap = (char)PacketType == 0;
    PacketType = ShouldByteSwap ? ByteSwap(PacketType) : PacketType;

    switch (PacketType) {
      case ArgumentInfo:
        ReadProfilingArgBlock(ToolName, F, ShouldByteSwap, CommandLines);
        break;

      case EdgeInfo:
        ReadProfilingBlock(ToolName, F, ShouldByteSwap, EdgeCounts);
        break;

      default:
        report_fatal_error(std::string(ToolName)
                           + ": Unknown profiling packet type");
        break;
    }
  }

  fclose(F);
}
