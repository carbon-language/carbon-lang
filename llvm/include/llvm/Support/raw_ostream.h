//===--- raw_ostream.h - Raw output stream --------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the raw_ostream class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_RAW_OSTREAM_H
#define LLVM_SUPPORT_RAW_OSTREAM_H

#include "llvm/ADT/StringExtras.h"
#include <cassert>
#include <cstring>
#include <string>
#include <iosfwd>

namespace llvm {
  class format_object_base;
  
/// raw_ostream - This class implements an extremely fast bulk output stream
/// that can *only* output to a stream.  It does not support seeking, reopening,
/// rewinding, line buffered disciplines etc. It is a simple buffer that outputs
/// a chunk at a time.
class raw_ostream {
protected:
  char *OutBufStart, *OutBufEnd, *OutBufCur;
public:
  raw_ostream() {
    // Start out ready to flush.
    OutBufStart = OutBufEnd = OutBufCur = 0;
  }
  virtual ~raw_ostream() {}
  
  //===--------------------------------------------------------------------===//
  // Configuration Interface
  //===--------------------------------------------------------------------===//
  
  /// SetBufferSize - Set the internal buffer size to the specified amount
  /// instead of the default.
  void SetBufferSize(unsigned Size) {
    assert(Size >= 64 &&
           "Buffer size must be somewhat large for invariants to hold");
    flush();
    
    delete [] OutBufStart;
    OutBufStart = new char[Size];
    OutBufEnd = OutBufStart+Size;
    OutBufCur = OutBufStart;
  }
  
  //===--------------------------------------------------------------------===//
  // Data Output Interface
  //===--------------------------------------------------------------------===//
  
  void flush() {
    if (OutBufCur != OutBufStart)
      flush_impl();
  }
  
  raw_ostream &operator<<(char C) {
    if (OutBufCur >= OutBufEnd)
      flush_impl();
    *OutBufCur++ = C;
    return *this;
  }
  
  raw_ostream &operator<<(unsigned char C) {
    if (OutBufCur >= OutBufEnd)
      flush_impl();
    *OutBufCur++ = C;
    return *this;
  }
  
  raw_ostream &operator<<(signed char C) {
    if (OutBufCur >= OutBufEnd)
      flush_impl();
    *OutBufCur++ = C;
    return *this;
  }
  
  raw_ostream &operator<<(const char *Str) {
    return write(Str, strlen(Str));
  }
  
  raw_ostream &operator<<(const std::string& Str) {
    return write(Str.data(), Str.length());
  }
  
  raw_ostream &operator<<(unsigned long N);
  raw_ostream &operator<<(long N);
  raw_ostream &operator<<(unsigned long long N);
  raw_ostream &operator<<(long long N);
  
  raw_ostream &operator<<(unsigned int N) {
    return this->operator<<(static_cast<unsigned long>(N));
  }
  
  raw_ostream &operator<<(int N) {
    return this->operator<<(static_cast<long>(N));
  }

  raw_ostream &operator<<(double N) {
    return this->operator<<(ftostr(N));
  }
  
  raw_ostream &write(const char *Ptr, unsigned Size);
  
  // Formatted output, see the format() function below.
  raw_ostream &operator<<(const format_object_base &Fmt);
  
  //===--------------------------------------------------------------------===//
  // Subclass Interface
  //===--------------------------------------------------------------------===//

protected:
  
  /// flush_impl - The is the piece of the class that is implemented by
  /// subclasses.  This outputs the currently buffered data and resets the
  /// buffer to empty.
  virtual void flush_impl() = 0;
  
  /// HandleFlush - A stream's implementation of flush should call this after
  /// emitting the bytes to the data sink.
  void HandleFlush() {
    if (OutBufStart == 0)
      SetBufferSize(4096);
    OutBufCur = OutBufStart;
  }
private:
  // An out of line virtual method to provide a home for the class vtable.
  virtual void handle();
};
  
//===----------------------------------------------------------------------===//
// Formatted Output
//===----------------------------------------------------------------------===//

/// format_object_base - This is a helper class used for handling formatted
/// output.  It is the abstract base class of a templated derived class.
class format_object_base {
protected:
  const char *Fmt;
  virtual void home(); // Out of line virtual method.
public:
  format_object_base(const char *fmt) : Fmt(fmt) {}
  virtual ~format_object_base() {}
  
  /// print - Format the object into the specified buffer.  On success, this
  /// returns the length of the formatted string.  If the buffer is too small,
  /// this returns a length to retry with, which will be larger than BufferSize.
  virtual unsigned print(char *Buffer, unsigned BufferSize) const = 0;
};
  
/// format_object - This is a templated helper class used by the format function
/// that captures the object to be formated and the format string.  When
/// actually printed, this synthesizes the string into a temporary buffer
/// provided and returns whether or not it is big enough.
template <typename T>
  class format_object : public format_object_base {
  T Val;
public:
  format_object(const char *fmt, const T &val)
    : format_object_base(fmt), Val(val) {
  }
  
  /// print - Format the object into the specified buffer.  On success, this
  /// returns the length of the formatted string.  If the buffer is too small,
  /// this returns a length to retry with, which will be larger than BufferSize.
  virtual unsigned print(char *Buffer, unsigned BufferSize) const {
    int N = snprintf(Buffer, BufferSize-1, Fmt, Val);
    if (N < 0)             // VC++ and old GlibC return negative on overflow.
      return BufferSize*2;
    if (unsigned(N) >= BufferSize-1)// Other impls yield number of bytes needed.
      return N+1;
    // If N is positive and <= BufferSize-1, then the string fit, yay.
    return N;
  }
};

/// format - This is a helper function that is used to produce formatted output.
/// This is typically used like:  OS << format("%0.4f", myfloat) << '\n';
template <typename T>
inline format_object<T> format(const char *Fmt, const T &Val) {
  return format_object<T>(Fmt, Val);
}
  
//===----------------------------------------------------------------------===//
// File Output Streams
//===----------------------------------------------------------------------===//
  
/// raw_fd_ostream - A raw_ostream that writes to a file descriptor.
///
class raw_fd_ostream : public raw_ostream {
  int FD;
  bool ShouldClose;
public:
  /// raw_fd_ostream - Open the specified file for writing.  If an error occurs,
  /// information about the error is put into ErrorInfo, and the stream should
  /// be immediately destroyed.
  raw_fd_ostream(const char *Filename, std::string &ErrorInfo);
  
  /// raw_fd_ostream ctor - FD is the file descriptor that this writes to.  If
  /// ShouldClose is true, this closes the file when 
  raw_fd_ostream(int fd, bool shouldClose) : FD(fd), ShouldClose(shouldClose) {}
  
  ~raw_fd_ostream();
    
  /// flush_impl - The is the piece of the class that is implemented by
  /// subclasses.  This outputs the currently buffered data and resets the
  /// buffer to empty.
  virtual void flush_impl();
};
  
/// raw_stdout_ostream - This is a stream that always prints to stdout.
///
class raw_stdout_ostream : public raw_fd_ostream {
  // An out of line virtual method to provide a home for the class vtable.
  virtual void handle();
public:
  raw_stdout_ostream();
};

/// raw_stderr_ostream - This is a stream that always prints to stderr.
///
class raw_stderr_ostream : public raw_fd_ostream {
  // An out of line virtual method to provide a home for the class vtable.
  virtual void handle();
public:
  raw_stderr_ostream();
};
  
/// outs() - This returns a reference to a raw_ostream for standard output.
/// Use it like: outs() << "foo" << "bar";
raw_ostream &outs();

/// errs() - This returns a reference to a raw_ostream for standard error.
/// Use it like: errs() << "foo" << "bar";
raw_ostream &errs();
  
  
//===----------------------------------------------------------------------===//
// Bridge Output Streams
//===----------------------------------------------------------------------===//
  
/// raw_os_ostream - A raw_ostream that writes to an std::ostream.  This is a
/// simple adaptor class.
class raw_os_ostream : public raw_ostream {
  std::ostream &OS;
public:
  raw_os_ostream(std::ostream &O) : OS(O) {}
  
  /// flush_impl - The is the piece of the class that is implemented by
  /// subclasses.  This outputs the currently buffered data and resets the
  /// buffer to empty.
  virtual void flush_impl();
};
  
} // end llvm namespace

#endif
