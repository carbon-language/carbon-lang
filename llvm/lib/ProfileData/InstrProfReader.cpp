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

#include "InstrProfIndexed.h"

#include <cassert>

using namespace llvm;

static error_code setupMemoryBuffer(std::string Path,
                                    std::unique_ptr<MemoryBuffer> &Buffer) {
  if (error_code EC = MemoryBuffer::getFileOrSTDIN(Path, Buffer))
    return EC;

  // Sanity check the file.
  if (Buffer->getBufferSize() > std::numeric_limits<unsigned>::max())
    return instrprof_error::too_large;
  return instrprof_error::success;
}

static error_code initializeReader(InstrProfReader &Reader) {
  return Reader.readHeader();
}

error_code InstrProfReader::create(std::string Path,
                                   std::unique_ptr<InstrProfReader> &Result) {
  // Set up the buffer to read.
  std::unique_ptr<MemoryBuffer> Buffer;
  if (error_code EC = setupMemoryBuffer(Path, Buffer))
    return EC;

  // Create the reader.
  if (IndexedInstrProfReader::hasFormat(*Buffer))
    Result.reset(new IndexedInstrProfReader(std::move(Buffer)));
  else if (RawInstrProfReader64::hasFormat(*Buffer))
    Result.reset(new RawInstrProfReader64(std::move(Buffer)));
  else if (RawInstrProfReader32::hasFormat(*Buffer))
    Result.reset(new RawInstrProfReader32(std::move(Buffer)));
  else
    Result.reset(new TextInstrProfReader(std::move(Buffer)));

  // Initialize the reader and return the result.
  return initializeReader(*Result);
}

error_code IndexedInstrProfReader::create(
    std::string Path, std::unique_ptr<IndexedInstrProfReader> &Result) {
  // Set up the buffer to read.
  std::unique_ptr<MemoryBuffer> Buffer;
  if (error_code EC = setupMemoryBuffer(Path, Buffer))
    return EC;

  // Create the reader.
  if (!IndexedInstrProfReader::hasFormat(*Buffer))
    return instrprof_error::bad_magic;
  Result.reset(new IndexedInstrProfReader(std::move(Buffer)));

  // Initialize the reader and return the result.
  return initializeReader(*Result);
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
  if (NumCounters == 0)
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

template <class IntPtrT>
error_code RawInstrProfReader<IntPtrT>::readNextHeader(const char *CurrentPos) {
  const char *End = DataBuffer->getBufferEnd();
  // Skip zero padding between profiles.
  while (CurrentPos != End && *CurrentPos == 0)
    ++CurrentPos;
  // If there's nothing left, we're done.
  if (CurrentPos == End)
    return instrprof_error::eof;
  // If there isn't enough space for another header, this is probably just
  // garbage at the end of the file.
  if (CurrentPos + sizeof(RawHeader) > End)
    return instrprof_error::malformed;
  // The magic should have the same byte order as in the previous header.
  uint64_t Magic = *reinterpret_cast<const uint64_t *>(CurrentPos);
  if (Magic != swap(getRawMagic<IntPtrT>()))
    return instrprof_error::bad_magic;

  // There's another profile to read, so we need to process the header.
  auto *Header = reinterpret_cast<const RawHeader *>(CurrentPos);
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
  size_t ProfileSize = NamesOffset + sizeof(char) * NamesSize;

  auto *Start = reinterpret_cast<const char *>(&Header);
  if (Start + ProfileSize > DataBuffer->getBufferEnd())
    return error(instrprof_error::bad_header);

  Data = reinterpret_cast<const ProfileData *>(Start + DataOffset);
  DataEnd = Data + DataSize;
  CountersStart = reinterpret_cast<const uint64_t *>(Start + CountersOffset);
  NamesStart = Start + NamesOffset;
  ProfileEnd = Start + ProfileSize;

  return success();
}

template <class IntPtrT>
error_code
RawInstrProfReader<IntPtrT>::readNextRecord(InstrProfRecord &Record) {
  if (Data == DataEnd)
    if (error_code EC = readNextHeader(ProfileEnd))
      return EC;

  // Get the raw data.
  StringRef RawName(getName(Data->NamePtr), swap(Data->NameSize));
  uint32_t NumCounters = swap(Data->NumCounters);
  if (NumCounters == 0)
    return error(instrprof_error::malformed);
  auto RawCounts = makeArrayRef(getCounter(Data->CounterPtr), NumCounters);

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

InstrProfLookupTrait::hash_value_type
InstrProfLookupTrait::ComputeHash(StringRef K) {
  return IndexedInstrProf::ComputeHash(HashType, K);
}

bool IndexedInstrProfReader::hasFormat(const MemoryBuffer &DataBuffer) {
  if (DataBuffer.getBufferSize() < 8)
    return false;
  using namespace support;
  uint64_t Magic =
      endian::read<uint64_t, little, aligned>(DataBuffer.getBufferStart());
  return Magic == IndexedInstrProf::Magic;
}

error_code IndexedInstrProfReader::readHeader() {
  const unsigned char *Start =
      (const unsigned char *)DataBuffer->getBufferStart();
  const unsigned char *Cur = Start;
  if ((const unsigned char *)DataBuffer->getBufferEnd() - Cur < 24)
    return error(instrprof_error::truncated);

  using namespace support;

  // Check the magic number.
  uint64_t Magic = endian::readNext<uint64_t, little, unaligned>(Cur);
  if (Magic != IndexedInstrProf::Magic)
    return error(instrprof_error::bad_magic);

  // Read the version.
  uint64_t Version = endian::readNext<uint64_t, little, unaligned>(Cur);
  if (Version != IndexedInstrProf::Version)
    return error(instrprof_error::unsupported_version);

  // Read the maximal function count.
  MaxFunctionCount = endian::readNext<uint64_t, little, unaligned>(Cur);

  // Read the hash type and start offset.
  IndexedInstrProf::HashT HashType = static_cast<IndexedInstrProf::HashT>(
      endian::readNext<uint64_t, little, unaligned>(Cur));
  if (HashType > IndexedInstrProf::HashT::Last)
    return error(instrprof_error::unsupported_hash_type);
  uint64_t HashOffset = endian::readNext<uint64_t, little, unaligned>(Cur);

  // The rest of the file is an on disk hash table.
  Index.reset(InstrProfReaderIndex::Create(Start + HashOffset, Cur, Start,
                                           InstrProfLookupTrait(HashType)));
  // Set up our iterator for readNextRecord.
  RecordIterator = Index->data_begin();

  return success();
}

error_code IndexedInstrProfReader::getFunctionCounts(
    StringRef FuncName, uint64_t &FuncHash, std::vector<uint64_t> &Counts) {
  const auto &Iter = Index->find(FuncName);
  if (Iter == Index->end())
    return error(instrprof_error::unknown_function);

  // Found it. Make sure it's valid before giving back a result.
  const InstrProfRecord &Record = *Iter;
  if (Record.Name.empty())
    return error(instrprof_error::malformed);
  FuncHash = Record.Hash;
  Counts = Record.Counts;
  return success();
}

error_code IndexedInstrProfReader::readNextRecord(InstrProfRecord &Record) {
  // Are we out of records?
  if (RecordIterator == Index->data_end())
    return error(instrprof_error::eof);

  // Read the next one.
  Record = *RecordIterator;
  ++RecordIterator;
  if (Record.Name.empty())
    return error(instrprof_error::malformed);
  return success();
}
