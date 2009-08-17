//===-- llvm/Support/FormattedStream.cpp - Formatted streams ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of formatted_raw_ostream.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/FormattedStream.h"

using namespace llvm;

/// CountColumns - Examine the given char sequence and figure out which
/// column we end up in after output.
///
static unsigned CountColumns(unsigned Column, const char *Ptr, size_t Size) {
  // Keep track of the current column by scanning the string for
  // special characters

  for (const char *End = Ptr + Size; Ptr != End; ++Ptr) {
    ++Column;
    if (*Ptr == '\n' || *Ptr == '\r')
      Column = 0;
    else if (*Ptr == '\t')
      // Assumes tab stop = 8 characters.
      Column += (8 - (Column & 0x7)) & 0x7;
  }

  return Column;
}

/// ComputeColumn - Examine the current output and figure out which
/// column we end up in after output.
void formatted_raw_ostream::ComputeColumn() {
  // The buffer may have been allocated underneath us.
  if (Scanned == 0) Scanned = begin();
  // Scan all characters added since our last scan to determine the new column.
  ColumnScanned = CountColumns(ColumnScanned, Scanned, end() - Scanned);
  // We're now current with everything in the buffer.
  Scanned = end();
}

/// PadToColumn - Align the output to some column number.
///
/// \param NewCol - The column to move to.
/// \param MinPad - The minimum space to give after the most recent
/// I/O, even if the current column + minpad > newcol.
///
void formatted_raw_ostream::PadToColumn(unsigned NewCol) { 
  // Figure out what's in the buffer and add it to the column count.
  ComputeColumn();

  // Output spaces until we reach the desired column.
  unsigned num = NewCol - ColumnScanned;
  if (NewCol < ColumnScanned || num < 1)
    num = 1;

  // Keep a buffer of spaces handy to speed up processing.
  const char *Spaces = "                                                      "
    "                                                                         ";

  assert(num < MAX_COLUMN_PAD && "Unexpectedly large column padding");
  write(Spaces, num);
}

void formatted_raw_ostream::write_impl(const char *Ptr, size_t Size) {
  // Figure out what's in the buffer and add it to the column count.
  ComputeColumn();

  // Write the data to the underlying stream (which is unbuffered, so
  // the data will be immediately written out).
  TheStream->write(Ptr, Size);

  // If this FormattedStream is unbuffered, scan the string that
  // was just written to determine the new column.
  if (Ptr == begin()) {
    // Buffered mode. The buffer is being flushed; reset the scanning
    // position to the beginning of the buffer.
    assert(Ptr + Size == end() && "Buffer is not being fully flushed!");
    Scanned = begin();
  } else {
    // Unbuffered mode. Immediately scan the string that was just
    // written to determine the new column.
    ColumnScanned = CountColumns(ColumnScanned, Ptr, Size);
  }
}

/// fouts() - This returns a reference to a formatted_raw_ostream for
/// standard output.  Use it like: fouts() << "foo" << "bar";
formatted_raw_ostream &llvm::fouts() {
  static formatted_raw_ostream S(outs());
  return S;
}

/// ferrs() - This returns a reference to a formatted_raw_ostream for
/// standard error.  Use it like: ferrs() << "foo" << "bar";
formatted_raw_ostream &llvm::ferrs() {
  static formatted_raw_ostream S(errs());
  return S;
}
