//=-- InstrProfReader.h - Instrumented profiling readers ----------*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for reading profiling data for instrumentation
// based PGO and coverage.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_PROFILEDATA_INSTRPROF_READER_H__
#define LLVM_PROFILEDATA_INSTRPROF_READER_H__

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"

#include <iterator>

namespace llvm {

class InstrProfReader;

/// Profiling information for a single function.
struct InstrProfRecord {
  StringRef Name;
  uint64_t Hash;
  ArrayRef<uint64_t> Counts;
};

/// A file format agnostic iterator over profiling data.
class InstrProfIterator : public std::iterator<std::input_iterator_tag,
                                               InstrProfRecord> {
  InstrProfReader *Reader;
  InstrProfRecord Record;

  void Increment();
public:
  InstrProfIterator() : Reader(nullptr) {}
  InstrProfIterator(InstrProfReader *Reader) : Reader(Reader) { Increment(); }

  InstrProfIterator &operator++() { Increment(); return *this; }
  bool operator==(const InstrProfIterator &RHS) { return Reader == RHS.Reader; }
  bool operator!=(const InstrProfIterator &RHS) { return Reader != RHS.Reader; }
  InstrProfRecord &operator*() { return Record; }
  InstrProfRecord *operator->() { return &Record; }
};

/// Base class and interface for reading profiling data of any known instrprof
/// format. Provides an iterator over InstrProfRecords.
class InstrProfReader {
  error_code LastError;
public:
  InstrProfReader() : LastError(instrprof_error::success) {}
  virtual ~InstrProfReader() {}

  /// Read a single record.
  virtual error_code readNextRecord(InstrProfRecord &Record) = 0;
  /// Iterator over profile data.
  InstrProfIterator begin() { return InstrProfIterator(this); }
  InstrProfIterator end() { return InstrProfIterator(); }

  /// Set the current error_code and return same.
  error_code error(error_code EC) {
    LastError = EC;
    return EC;
  }
  /// Clear the current error code and return a successful one.
  error_code success() { return error(instrprof_error::success); }

  /// Return true if the reader has finished reading the profile data.
  bool isEOF() { return LastError == instrprof_error::eof; }
  /// Return true if the reader encountered an error reading profiling data.
  bool hasError() { return LastError && !isEOF(); }
  /// Get the current error code.
  error_code getError() { return LastError; }

  /// Factory method to create an appropriately typed reader for the given
  /// instrprof file.
  static error_code create(std::string Path,
                           std::unique_ptr<InstrProfReader> &Result);
};

/// Reader for the simple text based instrprof format.
///
/// This format is a simple text format that's suitable for test data. Records
/// are separated by one or more blank lines, and record fields are separated by
/// new lines.
///
/// Each record consists of a function name, a function hash, a number of
/// counters, and then each counter value, in that order.
class TextInstrProfReader : public InstrProfReader {
private:
  /// The profile data file contents.
  std::unique_ptr<MemoryBuffer> DataBuffer;
  /// Iterator over the profile data.
  line_iterator Line;
  /// The current set of counter values.
  std::vector<uint64_t> Counts;

  TextInstrProfReader(const TextInstrProfReader &) LLVM_DELETED_FUNCTION;
  TextInstrProfReader &operator=(const TextInstrProfReader &)
    LLVM_DELETED_FUNCTION;
public:
  TextInstrProfReader(std::unique_ptr<MemoryBuffer> &DataBuffer_)
      : DataBuffer(DataBuffer_.release()), Line(*DataBuffer, '#') {}

  /// Read a single record.
  error_code readNextRecord(InstrProfRecord &Record) override;
};

} // end namespace llvm

#endif // LLVM_PROFILEDATA_INSTRPROF_READER_H__
