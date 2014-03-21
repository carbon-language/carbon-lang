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
#include "llvm/Support/Endian.h"

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

  // FIXME: This needs to determine which format the file is and construct the
  // correct subclass.
  Result.reset(new TextInstrProfReader(Buffer));

  return instrprof_error::success;
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
