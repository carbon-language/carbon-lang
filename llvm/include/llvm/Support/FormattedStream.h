//===-- llvm/CodeGen/FormattedStream.h - Formatted streams ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains raw_ostream implementations for streams to do
// things like pretty-print comments.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_ASMSTREAM_H
#define LLVM_CODEGEN_ASMSTREAM_H

#include "llvm/Support/raw_ostream.h"

namespace llvm 
{
  /// raw_asm_fd_ostream - Formatted raw_fd_ostream to handle
  /// asm-specific constructs
  ///
  class formatted_raw_ostream : public raw_ostream {
  private:
    /// TheStream - The real stream we output to
    ///
    raw_ostream &TheStream;

    /// Column - The current output column of the stream
    ///
    unsigned Column;

    virtual void write_impl(const char *Ptr, unsigned Size) {
      ComputeColumn(Ptr, Size);
      TheStream.write(Ptr, Size);
    }

    /// current_pos - Return the current position within the stream,
    /// not counting the bytes currently in the buffer.
    virtual uint64_t current_pos() { 
      // This has the same effect as calling TheStream.current_pos(),
      // but that interface is private.
      return TheStream.tell() - TheStream.GetNumBytesInBuffer();
    }

    /// ComputeColumn - Examine the current output and figure out
    /// which column we end up in after output.
    ///
    void ComputeColumn(const char *Ptr, unsigned Size);

  public:
    /// formatted_raw_ostream - Open the specified file for
    /// writing. If an error occurs, information about the error is
    /// put into ErrorInfo, and the stream should be immediately
    /// destroyed; the string will be empty if no error occurred.
    ///
    /// \param Filename - The file to open. If this is "-" then the
    /// stream will use stdout instead.
    /// \param Binary - The file should be opened in binary mode on
    /// platforms that support this distinction.
    formatted_raw_ostream(raw_ostream &Stream) 
        : raw_ostream(), TheStream(Stream), Column(0) {}

    /// PadToColumn - Align the output to some column number
    ///
    /// \param NewCol - The column to move to
    /// \param MinPad - The minimum space to give after the most
    /// recent I/O, even if the current column + minpad > newcol
    ///
    void PadToColumn(unsigned NewCol, unsigned MinPad = 0);
  };

  /// Column - An I/O manipulator to advance the output to a certain column
  ///
  class Column {
  private:
    /// Col - The column to move to
    ///
    unsigned int Col;

  public:
    explicit Column(unsigned int c) 
        : Col(c) {}

    /// operator() - Make Column a functor invokable by a stream
    /// output operator.
    ///
    formatted_raw_ostream &operator()(formatted_raw_ostream &Out) const {
      // Make at least one space before the next output
      Out.PadToColumn(Col, 1);
      return(Out);
    }
  };

  /// operator<< - Support coulmn-setting in formatted streams.
  ///
  inline formatted_raw_ostream &operator<<(formatted_raw_ostream &Out,
                                           const Column &Func)
  {
    return(Func(Out));
  }
}

#endif
