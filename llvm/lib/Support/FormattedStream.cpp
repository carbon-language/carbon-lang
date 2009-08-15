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

/// ComputeColumn - Examine the current output and figure out which
/// column we end up in after output.
///
void formatted_raw_ostream::ComputeColumn() {
  // Keep track of the current column by scanning the string for
  // special characters

  // The buffer may have been allocated underneath us.
  if (Scanned == 0 && GetNumBytesInBuffer() != 0) {
    Scanned = begin();
  }

  while (Scanned != end()) {
    ++ColumnScanned;
    if (*Scanned == '\n' || *Scanned == '\r')
      ColumnScanned = 0;
    else if (*Scanned == '\t')
      // Assumes tab stop = 8 characters.
      ColumnScanned += (8 - (ColumnScanned & 0x7)) & 0x7;
    ++Scanned;
  }
}

/// PadToColumn - Align the output to some column number.
///
/// \param NewCol - The column to move to.
/// \param MinPad - The minimum space to give after the most recent
/// I/O, even if the current column + minpad > newcol.
///
void formatted_raw_ostream::PadToColumn(unsigned NewCol, unsigned MinPad) { 
  // Figure out what's in the buffer and add it to the column count.
  ComputeColumn();

  // Output spaces until we reach the desired column.
  unsigned num = NewCol - ColumnScanned;
  if (NewCol < ColumnScanned || num < MinPad)
    num = MinPad;

  // Keep a buffer of spaces handy to speed up processing.
  const char *Spaces = "                                                      "
    "                                                                         ";

  assert(num < MAX_COLUMN_PAD && "Unexpectedly large column padding");

  write(Spaces, num);
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
