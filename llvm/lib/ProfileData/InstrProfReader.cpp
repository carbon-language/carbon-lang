//=-- InstrProfReader.cpp - Instrumented profiling reader -------------------=//
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

#include "llvm/ProfileData/InstrProfReader.h"
#include "llvm/ProfileData/InstrProf.h"

#include <cassert>

using namespace llvm;

error_code InstrProfReader::create(std::string Path,
                                   std::unique_ptr<InstrProfReader> &Result) {
  std::unique_ptr<MemoryBuffer> Buffer;
  if (error_code EC = MemoryBuffer::getFileOrSTDIN(Path, Buffer))
    return EC;

  // Sanity check the file.
  if (Buffer->getBufferSize() > std::numeric_limits<unsigned>::max())
    return instrprof_error::too_large;

  // Create the reader.
  if (RawInstrProfReader64::hasFormat(*Buffer))
    Result.reset(new RawInstrProfReader64(std::move(Buffer)));
  else if (RawInstrProfReader32::hasFormat(*Buffer))
    Result.reset(new RawInstrProfReader32(std::move(Buffer)));
  else
    Result.reset(new TextInstrProfReader(std::move(Buffer)));

  // Read the header and return the result.
  return Result->readHeader();
}

void InstrProfIterator::Increment() {
  if (Reader->readNextRecord(Record))
    *this = InstrProfIterator();
}

error_code TextInstrProfReader::readNextRecord(InstrProfRecord &Record) {
  // Skip empty lines.
  while (!Line.is_at_end() && Line->empty())
    ++Line;
  // If we hit EOF while looking for a name, we're done.
  if (Line.is_at_end())
    return error(instrprof_error::eof);

  // Read the function name.
  Record.Name = *Line++;

  // Read the function hash.
  if (Line.is_at_end())
    return error(instrprof_error::truncated);
  if ((Line++)->getAsInteger(10, Record.Hash))
    return error(instrprof_error::malformed);

  // Read the number of counters.
  uint64_t NumCounters;
  if (Line.is_at_end())
    return error(instrprof_error::truncated);
  if ((Line++)->getAsInteger(10, NumCounters))
    return error(instrprof_error::malformed);

  // Read each counter and fill our internal storage with the values.
  Counts.clear();
  Counts.reserve(NumCounters);
  for (uint64_t I = 0; I < NumCounters; ++I) {
    if (Line.is_at_end())
      return error(instrprof_error::truncated);
    uint64_t Count;
    if ((Line++)->getAsInteger(10, Count))
      return error(instrprof_error::malformed);
    Counts.push_back(Count);
  }
  // Give the record a reference to our internal counter storage.
  Record.Counts = Counts;

  return success();
}

template <class IntPtrT>
static uint64_t getRawMagic();

template <>
uint64_t getRawMagic<uint64_t>() {
  return
    uint64_t(255) << 56 |
    uint64_t('l') << 48 |
    uint64_t('p') << 40 |
    uint64_t('r') << 32 |
    uint64_t('o') << 24 |
    uint64_t('f') << 16 |
    uint64_t('r') <<  8 |
    uint64_t(129);
}

template <>
uint64_t getRawMagic<uint32_t>() {
  return
    uint64_t(255) << 56 |
    uint64_t('l') << 48 |
    uint64_t('p') << 40 |
    uint64_t('r') << 32 |
    uint64_t('o') << 24 |
    uint64_t('f') << 16 |
    uint64_t('R') <<  8 |
    uint64_t(129);
}

template <class IntPtrT>
bool RawInstrProfReader<IntPtrT>::hasFormat(const MemoryBuffer &DataBuffer) {
  if (DataBuffer.getBufferSize() < sizeof(uint64_t))
    return false;
  uint64_t Magic =
    *reinterpret_cast<const uint64_t *>(DataBuffer.getBufferStart());
  return getRawMagic<IntPtrT>() == Magic ||
    sys::SwapByteOrder(getRawMagic<IntPtrT>()) == Magic;
}

template <class IntPtrT>
error_code RawInstrProfReader<IntPtrT>::readHeader() {
  if (!hasFormat(*DataBuffer))
    return error(instrprof_error::bad_magic);
  if (DataBuffer->getBufferSize() < sizeof(RawHeader))
    return error(instrprof_error::bad_header);
  auto *Header =
    reinterpret_cast<const RawHeader *>(DataBuffer->getBufferStart());
  ShouldSwapBytes = Header->Magic != getRawMagic<IntPtrT>();
  return readHeader(*Header);
}

static uint64_t getRawVersion() {
  return 1;
}

template <class IntPtrT>
error_code RawInstrProfReader<IntPtrT>::readHeader(const RawHeader &Header) {
  if (swap(Header.Version) != getRawVersion())
    return error(instrprof_error::unsupported_version);

  CountersDelta = swap(Header.CountersDelta);
  NamesDelta = swap(Header.NamesDelta);
  auto DataSize = swap(Header.DataSize);
  auto CountersSize = swap(Header.CountersSize);
  auto NamesSize = swap(Header.NamesSize);

  ptrdiff_t DataOffset = sizeof(RawHeader);
  ptrdiff_t CountersOffset = DataOffset + sizeof(ProfileData) * DataSize;
  ptrdiff_t NamesOffset = CountersOffset + sizeof(uint64_t) * CountersSize;
  size_t FileSize = NamesOffset + sizeof(char) * NamesSize;

  if (FileSize != DataBuffer->getBufferSize())
    return error(instrprof_error::bad_header);

  const char *Start = DataBuffer->getBufferStart();
  Data = reinterpret_cast<const ProfileData *>(Start + DataOffset);
  DataEnd = Data + DataSize;
  CountersStart = reinterpret_cast<const uint64_t *>(Start + CountersOffset);
  NamesStart = Start + NamesOffset;

  return success();
}

template <class IntPtrT>
error_code
RawInstrProfReader<IntPtrT>::readNextRecord(InstrProfRecord &Record) {
  if (Data == DataEnd)
    return error(instrprof_error::eof);

  // Get the raw data.
  StringRef RawName(getName(Data->NamePtr), swap(Data->NameSize));
  auto RawCounts = makeArrayRef(getCounter(Data->CounterPtr),
                                swap(Data->NumCounters));

  // Check bounds.
  auto *NamesStartAsCounter = reinterpret_cast<const uint64_t *>(NamesStart);
  if (RawName.data() < NamesStart ||
      RawName.data() + RawName.size() > DataBuffer->getBufferEnd() ||
      RawCounts.data() < CountersStart ||
      RawCounts.data() + RawCounts.size() > NamesStartAsCounter)
    return error(instrprof_error::malformed);

  // Store the data in Record, byte-swapping as necessary.
  Record.Hash = swap(Data->FuncHash);
  Record.Name = RawName;
  if (ShouldSwapBytes) {
    Counts.clear();
    Counts.reserve(RawCounts.size());
    for (uint64_t Count : RawCounts)
      Counts.push_back(swap(Count));
    Record.Counts = Counts;
  } else
    Record.Counts = RawCounts;

  // Iterate.
  ++Data;
  return success();
}

namespace llvm {
template class RawInstrProfReader<uint32_t>;
template class RawInstrProfReader<uint64_t>;
}
