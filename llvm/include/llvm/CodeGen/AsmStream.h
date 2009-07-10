//===-- llvm/CodeGen/AsmStream.h - AsmStream Framework --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains raw_ostream implementations for ASM printers to
// do things like pretty-print comments.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_ASMSTREAM_H
#define LLVM_CODEGEN_ASMSTREAM_H

#include "llvm/Target/TargetAsmInfo.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm 
{
  /// raw_asm_fd_ostream - Formatted raw_fd_ostream to handle
  /// asm-specific constructs
  ///
  class raw_asm_fd_ostream : public raw_fd_ostream {
  private:
    bool formatted;
    int column;

  protected:
    void ComputeColumn(void) {
      if (formatted) {
        // Keep track of the current column by scanning the string for
        // special characters

        // Find the last newline.  This is our column start.  If there
        // is no newline, start with the current column.
        char *nlpos = NULL;        
        for (char *pos = CurBufPtr(), *epos = StartBufPtr(); pos > epos; --pos) {
          if (*(pos-1) == '\n') {
            nlpos = pos-1;
            // The newline will be counted, setting this to zero.  We
            // need to do it this way in case nlpos is CurBufPtr().
            column = -1;
            break;
          }
        }

        if (nlpos == NULL) {
          nlpos = StartBufPtr();
        }

        // Walk through looking for tabs and advance column as appropriate
        for (char *pos = nlpos, *epos = CurBufPtr(); pos != epos; ++pos) {
          ++column;
          if (*pos == '\t') {
            // Advance to next tab stop (every eight characters)
            column += ((8 - (column & 0x7)) & 0x7);
            assert(!(column & 0x3) && "Column out of alignment");
          }
        }
      }
    }

    virtual void AboutToFlush(void) {
      ComputeColumn();
    }

  public:
    /// raw_asm_fd_ostream - Open the specified file for writing. If
    /// an error occurs, information about the error is put into
    /// ErrorInfo, and the stream should be immediately destroyed; the
    /// string will be empty if no error occurred.
    ///
    /// \param Filename - The file to open. If this is "-" then the
    /// stream will use stdout instead.
    /// \param Binary - The file should be opened in binary mode on
    /// platforms that support this distinction.
    raw_asm_fd_ostream(const char *Filename, bool Binary, std::string &ErrorInfo) 
        : raw_fd_ostream(Filename, Binary, ErrorInfo),
            formatted(!Binary), column(0) {}

    /// raw_asm_fd_ostream ctor - FD is the file descriptor that this
    /// writes to.  If ShouldClose is true, this closes the file when
    /// the stream is destroyed.
    raw_asm_fd_ostream(int fd, bool shouldClose, 
                       bool unbuffered=false)
        : raw_fd_ostream(fd, shouldClose, unbuffered),
            formatted(true), column(0) {
      if (unbuffered) {
        assert(0 && "asm stream must be buffered");
        // Force buffering
        SetBufferSize();
      }
    }


    /// SetColumn - Align the output to some column number
    ///
    void setColumn(int newcol, int minpad = 0) {
      if (formatted) {
        flush();

        // Output spaces until we reach the desired column
        int num = newcol - column;
        if (num <= 0) {
          num = minpad;
        }

        // TODO: Write a whole string at a time
        while (num-- > 0) {
          write(' ');
        }
      }
    };
  };

  /// Column - An I/O manipulator to advance the output to a certain column
  ///
  class Column {
  private:
    int column;

  public:
    explicit Column(int c) 
        : column(c) {}

    raw_asm_fd_ostream &operator()(raw_asm_fd_ostream &out) const {
      // Make at least one space before the comment
      out.setColumn(column, 1);
      return(out);
    }
  };

  inline raw_asm_fd_ostream &operator<<(raw_asm_fd_ostream &out, const Column &column)
  {
    return(column(out));
  }

  /// Comment - An I/O manipulator to output an end-of-line comment
  ///
  class Comment 
      : public Column {
  private:
    static const int CommentColumn = 60;
    std::string text;
    const TargetAsmInfo &TAI;
    
  public:
    Comment(const std::string &comment,
            const TargetAsmInfo &tai) 
        : Column(CommentColumn), text(comment), TAI(tai) {}

    raw_asm_fd_ostream &operator()(raw_asm_fd_ostream &out) const {
      Column::operator()(out);
      out << TAI.getCommentString() << " " << text;
      return(out);
    }
  };

  inline raw_asm_fd_ostream &operator<<(raw_asm_fd_ostream &out,
                                        const Comment &comment)
  {
    return(comment(out));
  }

  /// Asm stream equivalent for std streams
  ///

  /// WARNING: Do NOT use these streams in the constructors of global
  /// objects.  There is no mechanism to ensure they are initialized
  /// before other global objects.
  ///
  extern raw_asm_fd_ostream asmouts;
  extern raw_asm_fd_ostream asmerrs;
}

#endif
