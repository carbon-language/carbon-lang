//===- LineIterator.cpp - Implementation of line iteration ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace llvm;

line_iterator::line_iterator(const MemoryBuffer &Buffer, char CommentMarker)
    : Buffer(Buffer.getBufferSize() ? &Buffer : nullptr),
      CommentMarker(CommentMarker), LineNumber(1),
      CurrentLine(Buffer.getBufferSize() ? Buffer.getBufferStart() : nullptr,
                  0) {
  // Ensure that if we are constructed on a non-empty memory buffer that it is
  // a null terminated buffer.
  if (Buffer.getBufferSize()) {
    assert(Buffer.getBufferEnd()[0] == '\0');
    advance();
  }
}

void line_iterator::advance() {
  assert(Buffer && "Cannot advance past the end!");

  const char *Pos = CurrentLine.end();
  assert(Pos == Buffer->getBufferStart() || *Pos == '\n' || *Pos == '\0');

  if (CommentMarker == '\0') {
    // If we're not stripping comments, this is simpler.
    size_t Blanks = 0;
    while (Pos[Blanks] == '\n')
      ++Blanks;
    Pos += Blanks;
    LineNumber += Blanks;
  } else {
    // Skip comments and count line numbers, which is a bit more complex.
    for (;;) {
      if (*Pos == CommentMarker)
        do {
          ++Pos;
        } while (*Pos != '\0' && *Pos != '\n');
      if (*Pos != '\n')
        break;
      ++Pos;
      ++LineNumber;
    }
  }

  if (*Pos == '\0') {
    // We've hit the end of the buffer, reset ourselves to the end state.
    Buffer = nullptr;
    CurrentLine = StringRef();
    return;
  }

  // Measure the line.
  size_t Length = 0;
  do {
    ++Length;
  } while (Pos[Length] != '\0' && Pos[Length] != '\n');

  CurrentLine = StringRef(Pos, Length);
}
