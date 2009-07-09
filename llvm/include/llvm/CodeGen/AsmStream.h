//===-- llvm/CodeGen/AsmStream.h - AsmStream Framework --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains streambuf and ostream implementations for ASM printers
// to do things like pretty-print comments.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_ASMSTREAM_H
#define LLVM_CODEGEN_ASMSTREAM_H

#include "llvm/Target/TargetAsmInfo.h"

#include <streambuf>
#include <iomanip>
#include <iterator>
#include <llvm/Support/Streams.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace llvm 
{
  /// BasicAsmStreambuf - A stream buffer to count columns and allow
  /// placement of output at specific columns.
  ///
  /// TODO: This only works on POSIX systems for now
  template<typename charT, typename Traits = std::char_traits<charT> >
  class BasicAsmStreambuf
      : public std::basic_streambuf<charT, Traits> {
  private:
    static const int bufferSize = 10;
    charT buffer_array[bufferSize];
    int column;
    int fd;

    bool formatted;

    typedef typename std::basic_streambuf<charT, Traits>::int_type int_type;

  protected:
    void setFD(int f) {
      fd = f;
    }

    int getFD(void) const {
      return(fd);
    }

    bool FDSet(void) const {
      return(getFD() != -1);
    }

  public:
    BasicAsmStreambuf(void) 
        : std::basic_streambuf<charT, Traits>(), column(0), fd(-1),
            formatted(true) {
      // Assure we have room for one character when overflow is called
      setp(buffer_array, buffer_array+(bufferSize-1));
    }

    explicit BasicAsmStreambuf(int f)
        : std::basic_streambuf<charT, Traits>(), column(0), fd(f),
            formatted(true) {
      // Assure we have room for one character when overflow is called
      setp(buffer_array, buffer_array+(bufferSize-1));
    }

    virtual ~BasicAsmStreambuf(void) {
      sync();
    }

    void setColumn(int newcol, int minpad = 0) {
      if (formatted) {
        // Flush the current buffer to get the most recent column
        if (Traits::eq_int_type(flushBuffer(), Traits::eof())) {
          // Error
          assert(0 && "Corrupted output buffer");
        }

        // Output spaces until we reach the desired column
        int num = newcol - column;
        if (num <= 0) {
          num = minpad;
        }

        // TODO: Use sputn
        while (num-- > 0) {
          if (Traits::eq_int_type(this->sputc(' '), Traits::eof())) {
            assert(0 && "Corrupted output buffer");
          }
        }
      }
    };

  protected:
    void setFormatting(bool v) {
      formatted = v;
    }

    int flushBuffer(void) {
      if (FDSet()) {
        if (formatted) {
          // Keep track of the current column by scanning the string for
          // special characters

          // Find the last newline.  This is our column start.  If there
          // is no newline, start with the current column.
          charT *nlpos = NULL;        
          for (charT *pos = this->pptr(), *epos = this->pbase(); pos > epos; --pos) {
            if (Traits::eq(*(pos-1), '\n')) {
              nlpos = pos-1;
              // The newline will be counted, setting this to zero.
              // We need to do it this way in case nlpos is pptr(),
              // which is one after epptr() after overflow is called.
              column = -1;
              break;
            }
          }

          if (nlpos == NULL) {
            nlpos = this->pbase();
          }

          // Walk through looking for tabs and advance column as appropriate
          for (charT *pos = nlpos, *epos = this->pptr(); pos != epos; ++pos) {
            ++column;
            if (Traits::eq(*pos, '\t')) {
              // Advance to next tab stop (every eight characters)
              column += ((8 - (column & 0x7)) & 0x7);
              assert(!(column & 0x3) && "Column out of alignment");
            }
          }
        }

        // Write out the buffer
        int num = this->pptr() - this->pbase();

        if (write(fd, buffer_array, num) != num) {
          return(Traits::eof());
        }
        this->pbump(-num);
        return(num);
      }
      return(0);
    }

    // Buffer full, so write c and all buffered characters
    virtual int_type overflow(int_type c) {
      if (!Traits::eq_int_type(c, Traits::eof())) {
        *this->pptr() = c;
        this->pbump(1);
        //++column;
        if (Traits::eq_int_type(flushBuffer(), Traits::eof())) {
          // Error
          return Traits::eof();
        }
        
        return Traits::not_eof(c);
      }
      return Traits::eof();
    }

    // Write multiple characters (TODO)
    //    virtual std::streamsize xsputn(const char *s,
    //                             std::streamsize num) {
    //}

    // Flush data in the buffer
    virtual int sync(void) {
      if (Traits::eq_int_type(flushBuffer(),Traits::eof())) {
        // Error
        return -1;
      }
      return 0;
    }
  };

  typedef BasicAsmStreambuf<char> AsmStreambuf;
  typedef BasicAsmStreambuf<wchar_t> AsmWStreambuf;

  /// BasicAsmFilebuf - An AsmStreambuf to write to files
  ///
  template<typename charT, typename Traits = std::char_traits<charT> >
  class BasicAsmFilebuf 
      : public BasicAsmStreambuf<charT, Traits> {
  public:
    BasicAsmFilebuf(void)
        : BasicAsmStreambuf<charT, Traits>() {}
    BasicAsmFilebuf(const std::basic_string<charT, Traits> &name,
                    std::ios_base::openmode mode)
        : BasicAsmStreambuf<charT, Traits>() {
      open(name.c_str(), mode);
    }

    ~BasicAsmFilebuf(void) {
      close();
    }

    BasicAsmFilebuf *open(const charT *name, std::ios_base::openmode mode) {
      int flags = 0;
      if (mode & std::ios_base::in) {
        flags |= O_RDONLY;
      }
      if (mode & std::ios_base::out) {
        if (mode & std::ios_base::in) {
          flags |= O_RDWR;
        }
        else {
          flags |= O_WRONLY;
        }
        flags |= O_CREAT;
      }
      if (mode & std::ios_base::trunc) {
        flags |= O_TRUNC;
      }
      if (mode & std::ios_base::app) {
        flags |= O_APPEND;
      }

      if (mode & std::ios_base::binary) {
        this->setFormatting(false);
      }
      else {
        this->setFormatting(true);
      }

      int fd = ::open(name, flags, S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH|S_IWOTH);
      assert(fd != -1 && "Cannot open output file");

      this->setFD(fd);

      return(this);
    }

    BasicAsmFilebuf *close(void) {
      if (is_open()) {
        int result = ::close(this->getFD());
        assert(result != -1 && "Could not close output file");
        this->setFD(-1);
      }

      return(this);
    }

    bool is_open(void) const {
      return(this->getFD() != -1);
    }
  };

  typedef BasicAsmFilebuf<char> AsmFilebuf;
  typedef BasicAsmFilebuf<wchar_t> AsmWFilebuf;

  /// BasicAsmOStream - std::ostream equivalent for BasicAsmStreambuf
  ///
  template<typename charT, typename Traits = std::char_traits<charT> >
  class BasicAsmOStream 
      : public std::basic_ostream<charT, Traits> {
  protected:
    typedef BasicAsmStreambuf<charT, Traits> streambuf;
    streambuf *bufptr;

  public:
    explicit BasicAsmOStream(streambuf *b)
        : std::basic_ostream<charT, Traits>(b), bufptr(b) {
      assert(b && "Invalid streambuf in OStream constructor");
      assert(this->rdbuf()
             && "Invalid basic_ostream buf in OStream constructor");
    }

    streambuf *buffer(void) {
      assert(bufptr && "Invalid streambuf in buffer");
      assert(this->rdbuf()
             && "Invalid basic_ostream buf in rdbuf");
      return(bufptr);
    }
  };

  typedef BasicAsmOStream<char> AsmOStream;
  typedef BasicAsmOStream<wchar_t> AsmWOStream;

  /// BasicAsmOFStream - std::ofstream equivalent for BasicAsmFilebuf
  ///
  template<typename charT, typename Traits = std::char_traits<charT> >
  class BasicAsmOFStream 
      : public BasicAsmOStream<charT, Traits> {
  protected:
    BasicAsmFilebuf<charT, Traits> buf;

  public:
    BasicAsmOFStream(void) : BasicAsmOStream<charT, Traits>(&buf) {}
    explicit BasicAsmOFStream(const std::basic_string<charT, Traits> &name) 
        : BasicAsmOStream<charT, Traits>(&buf),
            buf(name, std::ios_base::out | std::ios_base::trunc) {
      assert(this->buffer() && "Invalid streambuf in OFStream constructor");
      assert(this->rdbuf()
             && "Invalid basic_ostream buf in OFStream constructor");
    }

    explicit BasicAsmOFStream(const std::basic_string<charT, Traits> &name,
                              std::ios_base::openmode mode) 
        : BasicAsmOStream<charT, Traits>(&buf),
            buf(name, mode) {
      assert(this->buffer() && "Invalid streambuf in OFStream constructor");
      assert(this->rdbuf()
             && "Invalid basic_ostream buf in OFStream constructor");
    }
    
    ~BasicAsmOFStream(void) {
      close();
    }

    bool open(const std::basic_string<charT, Traits> &name) {
      assert(this->buffer() && "Invalid streambuf in open1");
      assert(this->rdbuf()
             && "Invalid basic_ostream buf in open1");

      buf.open(name.c_str(), std::ios_base::out | std::ios_base::trunc);

      assert(this->buffer() && "Invalid streambuf in open2");
      assert(this->rdbuf()
             && "Invalid basic_ostream buf in open2");

      assert(this->good() && "Not good");
      assert((bool)(*this) && "Not true");

      return(true);
    }

    bool close(void) {
      buf.close();
      return(true);
    }

    bool is_open(void) const {
      return(buf.is_open());
    }
  };

  typedef BasicAsmOFStream<char> AsmOFStream;
  typedef BasicAsmOFStream<wchar_t> AsmWOFStream;

  /// Column - An I/O manipulator to advance the output to a certain column
  ///
  class Column {
  private:
    int column;

  public:
    explicit Column(int c) 
        : column(c) {}

    template<typename OStream>
    OStream &operator()(OStream &out) const {
      // Make at least one space before the comment
      out.buffer()->setColumn(column, 1);
      return(out);
    }
  };

  template<typename OStream>
  OStream &operator<<(OStream &out, const Column &column)
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

    template<typename OStream>
    OStream &operator()(OStream &out) const {
      Column::operator()(out);
      out << TAI.getCommentString() << " " << text;
      return(out);
    }
  };

  template<typename OStream>
  OStream &operator<<(OStream &out, const Comment &comment)
  {
    return(comment(out));
  }

  /// raw_asm_ostream - A raw_ostream that writes to an AsmOStream
  /// This exposes additional AsmOStream capability.
  class raw_asm_ostream : public raw_ostream {
    AsmOStream &OS;
  public:
    raw_asm_ostream(AsmOStream &O) : OS(O) {}
    ~raw_asm_ostream();

    AsmOStream &getStream(void) {
      flush();
      return OS;
    }

    raw_asm_ostream &operator<<(const Column &column) {
      flush();
      OS << column;
      return *this;
    }

    raw_asm_ostream &operator<<(const Comment &comment) {
      flush();
      OS << comment;
      return *this;
    }

    /// flush_impl - The is the piece of the class that is implemented by
    /// subclasses.  This outputs the currently buffered data and resets the
    /// buffer to empty.
    virtual void flush_impl();
  };

  /// WARNING: Do NOT use these streams in the constructors of global
  /// objects.  There is no mechanism to ensure they are initialized
  /// before other global objects.
  ///
  extern raw_asm_ostream asmout;
  extern raw_asm_ostream asmerr;
}

#endif
