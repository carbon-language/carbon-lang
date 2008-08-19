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

#include <cassert>
#include <cstring>
#include <string>
#include <iosfwd>

namespace llvm {

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
  
  raw_ostream &operator<<(const char *Str) {
    return write(Str, strlen(Str));
  }
  
  raw_ostream &operator<<(unsigned N) {
    // Zero is a special case.
    if (N == 0)
      return *this << '0';
    
    char NumberBuffer[20];
    char *EndPtr = NumberBuffer+sizeof(NumberBuffer);
    char *CurPtr = EndPtr;
    
    while (N) {
      *--CurPtr = '0' + char(N % 10);
      N /= 10;
    }
    return write(CurPtr, EndPtr-CurPtr);
  }
  
  
  raw_ostream &write(const char *Ptr, unsigned Size) {
    if (OutBufCur+Size > OutBufEnd)
      flush_impl();
    
    // Handle short strings specially, memcpy isn't very good at very short
    // strings.
    switch (Size) {
    case 4: OutBufCur[3] = Ptr[3]; // FALL THROUGH
    case 3: OutBufCur[2] = Ptr[2]; // FALL THROUGH
    case 2: OutBufCur[1] = Ptr[1]; // FALL THROUGH
    case 1: OutBufCur[0] = Ptr[0]; // FALL THROUGH
    case 0: break;
    default:
      // Normally the string to emit is shorter than the buffer.
      if (Size <= unsigned(OutBufEnd-OutBufStart)) {
        memcpy(OutBufCur, Ptr, Size);
        break;
      }

      // If emitting a string larger than our buffer, emit in chunks.  In this
      // case we know that we just flushed the buffer.
      while (Size) {
        unsigned NumToEmit = OutBufEnd-OutBufStart;
        if (Size < NumToEmit) NumToEmit = Size;
        assert(OutBufCur == OutBufStart);
        memcpy(OutBufStart, Ptr, NumToEmit);
        Ptr += NumToEmit;
        OutBufCur = OutBufStart + NumToEmit;
        flush_impl();
      }
      break;
    }
    OutBufCur += Size;
    return *this;
  }
  
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
