//===-- llvm/CodeGen/AsmStream.cpp - AsmStream Framework --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains instantiations of "standard" AsmOStreams.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/FormattedStream.h"

namespace llvm {
  /// ComputeColumn - Examine the current output and figure out which
  /// column we end up in after output.
  ///
  void formatted_raw_ostream::ComputeColumn(const char *Ptr, unsigned Size)
  {
    // Keep track of the current column by scanning the string for
    // special characters

    // Find the last newline.  This is our column start.  If there
    // is no newline, start with the current column.
    const char *nlpos = NULL;
    for (const char *pos = Ptr + Size, *epos = Ptr; pos > epos; --pos) {
      if (*(pos-1) == '\n') {
        nlpos = pos-1;
        // The newline will be counted, setting this to zero.  We
        // need to do it this way in case nlpos is Ptr.
        Column = -1;
        break;
      }
    }

    if (nlpos == NULL) {
      nlpos = Ptr;
    }

    // Walk through looking for tabs and advance column as appropriate
    for (const char *pos = nlpos, *epos = Ptr + Size; pos != epos; ++pos) {
      ++Column;
      if (*pos == '\t') {
        // Advance to next tab stop (every eight characters)
        Column += ((8 - (Column & 0x7)) & 0x7);
        assert(!(Column & 0x3) && "Column out of alignment");
      }
    }
  }

  /// PadToColumn - Align the output to some column number
  ///
  /// \param NewCol - The column to move to
  /// \param MinPad - The minimum space to give after the most recent
  /// I/O, even if the current column + minpad > newcol
  ///
  void formatted_raw_ostream::PadToColumn(unsigned NewCol, unsigned MinPad) 
  {
    flush();

    // Output spaces until we reach the desired column
    unsigned num = NewCol - Column;
    if (NewCol < Column || num < MinPad) {
      num = MinPad;
    }

    // TODO: Write a whole string at a time
    while (num-- > 0) {
      write(' ');
    }
  }
}
