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
#include <algorithm>

using namespace llvm;

/// ComputeColumn - Examine the current output and figure out which
/// column we end up in after output.
///
void formatted_raw_ostream::ComputeColumn(const char *Ptr, size_t Size) {
  // Keep track of the current column by scanning the string for
  // special characters

  for (const char *epos = Ptr + Size; Ptr != epos; ++Ptr) {
    ++Column;
    if (*Ptr == '\n' || *Ptr == '\r')
      Column = 0;
    else if (*Ptr == '\t')
      Column += (8 - (Column & 0x7)) & 0x7;
  }
}

/// PadToColumn - Align the output to some column number.
///
/// \param NewCol - The column to move to.
/// \param MinPad - The minimum space to give after the most recent
/// I/O, even if the current column + minpad > newcol.
///
void formatted_raw_ostream::PadToColumn(unsigned NewCol, unsigned MinPad) {
  flush();

  // Output spaces until we reach the desired column.
  unsigned num = NewCol - Column;
  if (NewCol < Column || num < MinPad)
    num = MinPad;

  // Keep a buffer of spaces handy to speed up processing.
  static char Spaces[MAX_COLUMN_PAD];
  static bool Initialized = false;
  if (!Initialized) {
    std::fill_n(Spaces, MAX_COLUMN_PAD, ' '),
    Initialized = true;
  }

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
