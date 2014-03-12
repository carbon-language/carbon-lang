//=-- ProfileDataReader.cpp - Instrumented profiling reader -----------------=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for reading profiling data for clang's
// instrumentation based PGO and coverage.
//
//===----------------------------------------------------------------------===//

#include "llvm/Profile/ProfileDataReader.h"
#include "llvm/Profile/ProfileData.h"
#include "llvm/Support/Endian.h"

#include <cassert>

using namespace llvm;

error_code ProfileDataReader::create(
    std::string Path, std::unique_ptr<ProfileDataReader> &Result) {
  std::unique_ptr<MemoryBuffer> Buffer;
  if (error_code EC = MemoryBuffer::getFileOrSTDIN(Path, Buffer))
    return EC;

  if (Buffer->getBufferSize() > std::numeric_limits<unsigned>::max())
    return profiledata_error::too_large;

  Result.reset(new ProfileDataReader(Buffer));
  if (error_code EC = Result->readIndex())
    return EC;
  return profiledata_error::success;
}

class llvm::ProfileDataCursor {
  const char *Start;
  const char *Next;
  const char *End;

  error_code skip(unsigned bytes) {
    if (Next + bytes > End)
      return profiledata_error::malformed;
    Next += bytes;
    return profiledata_error::success;
  }

  template <typename T>
  error_code read(T &Result) {
    typedef support::detail::packed_endian_specific_integral
        <T, support::little, support::unaligned> Endian_t;
    const char *Prev = Next;
    if (error_code EC = skip(sizeof(T)))
      return EC;
    Result = *reinterpret_cast<const Endian_t*>(Prev);
    return profiledata_error::success;
  }
public:
  ProfileDataCursor(const MemoryBuffer *Buf)
      : Start(Buf->getBufferStart()), Next(Start), End(Buf->getBufferEnd()) {}
  bool offsetReached(size_t Offset) { return Start + Offset <= Next; }
  bool offsetInBounds(size_t Offset) { return Start + Offset < End; }

  error_code skipToOffset(size_t Offset) {
    if (!offsetInBounds(Offset))
      return profiledata_error::malformed;
    Next = Start + Offset;
    return profiledata_error::success;
  }

  error_code skip32() { return skip(4); }
  error_code skip64() { return skip(8); }
  error_code read32(uint32_t &Result) { return read<uint32_t>(Result); }
  error_code read64(uint64_t &Result) { return read<uint64_t>(Result); }

  error_code readChars(StringRef &Result, uint32_t Len) {
    error_code EC;
    const char *Prev = Next;
    if (error_code EC = skip(Len))
      return EC;
    Result = StringRef(Prev, Len);
    return profiledata_error::success;
  }
  error_code readString(StringRef &Result) {
    uint32_t Len;
    if (error_code EC = read32(Len))
      return EC;
    return readChars(Result, Len);
  }
};

error_code ProfileDataReader::readIndex() {
  ProfileDataCursor Cursor(DataBuffer.get());
  error_code EC;
  StringRef Magic;
  uint32_t Version, IndexEnd, DataStart;

  if ((EC = Cursor.readChars(Magic, 4)))
    return EC;
  if (StringRef(PROFILEDATA_MAGIC, 4) != Magic)
    return profiledata_error::bad_magic;
  if ((EC = Cursor.read32(Version)))
    return EC;
  if (Version != PROFILEDATA_VERSION)
    return profiledata_error::unsupported_version;
  if ((EC = Cursor.read32(IndexEnd)))
    return EC;
  if ((EC = Cursor.skip32()))
    return EC;
  if ((EC = Cursor.read64(MaxFunctionCount)))
    return EC;

  DataStart = IndexEnd + (sizeof(uint64_t) - IndexEnd % sizeof(uint64_t));
  while (!Cursor.offsetReached(IndexEnd)) {
    StringRef FuncName;
    uint32_t Offset, TotalOffset;
    if ((EC = Cursor.readString(FuncName)))
      return EC;
    if ((EC = Cursor.read32(Offset)))
      return EC;
    TotalOffset = DataStart + Offset;
    if (!Cursor.offsetInBounds(TotalOffset))
      return profiledata_error::truncated;
    DataOffsets[FuncName] = TotalOffset;
  }

  return profiledata_error::success;
}

error_code ProfileDataReader::findFunctionCounts(StringRef FuncName,
                                                 uint64_t &FunctionHash,
                                                 ProfileDataCursor &Cursor) {
  error_code EC;
  // Find the relevant section of the pgo-data file.
  const auto &OffsetIter = DataOffsets.find(FuncName);
  if (OffsetIter == DataOffsets.end())
    return profiledata_error::unknown_function;
  // Go there and read the function data
  if ((EC = Cursor.skipToOffset(OffsetIter->getValue())))
    return EC;
  if ((EC = Cursor.read64(FunctionHash)))
    return EC;
  return profiledata_error::success;
}

error_code ProfileDataReader::getFunctionCounts(StringRef FuncName,
                                                uint64_t &FunctionHash,
                                                std::vector<uint64_t> &Counts) {
  ProfileDataCursor Cursor(DataBuffer.get());
  error_code EC;
  if ((EC = findFunctionCounts(FuncName, FunctionHash, Cursor)))
    return EC;

  uint64_t NumCounters;
  if ((EC = Cursor.read64(NumCounters)))
    return EC;
  for (uint64_t I = 0; I < NumCounters; ++I) {
    uint64_t Count;
    if ((EC = Cursor.read64(Count)))
      return EC;
    Counts.push_back(Count);
  }

  return profiledata_error::success;
}

error_code ProfileDataReader::getCallFrequency(StringRef FuncName,
                                               uint64_t &FunctionHash,
                                               double &Frequency) {
  ProfileDataCursor Cursor(DataBuffer.get());
  error_code EC;
  if ((EC = findFunctionCounts(FuncName, FunctionHash, Cursor)))
    return EC;
  if ((EC = Cursor.skip64()))
    return EC;
  uint64_t CallCount;
  if ((EC = Cursor.read64(CallCount)))
    return EC;
  Frequency = CallCount / (double)MaxFunctionCount;
  return profiledata_error::success;
}
