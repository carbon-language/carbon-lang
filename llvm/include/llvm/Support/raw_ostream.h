//===--- raw_ostream.h - Raw output stream ----------------------*- C++ -*-===//
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
#include "llvm/ADT/StringRef.h"
#include <cassert>
#include <cstring>
#include <string>
#include <iosfwd>

namespace llvm {
  class format_object_base;
  template <typename T>
  class SmallVectorImpl;

/// raw_ostream - This class implements an extremely fast bulk output stream
/// that can *only* output to a stream.  It does not support seeking, reopening,
/// rewinding, line buffered disciplines etc. It is a simple buffer that outputs
/// a chunk at a time.
class raw_ostream {
private:
  // Do not implement. raw_ostream is noncopyable.
  void operator=(const raw_ostream &);
  raw_ostream(const raw_ostream &);

  /// The buffer is handled in such a way that the buffer is
  /// uninitialized, unbuffered, or out of space when OutBufCur >=
  /// OutBufEnd. Thus a single comparison suffices to determine if we
  /// need to take the slow path to write a single character.
  ///
  /// The buffer is in one of three states:
  ///  1. Unbuffered (Unbuffered == true)
  ///  1. Uninitialized (Unbuffered == false && OutBufStart == 0).
  ///  2. Buffered (Unbuffered == false && OutBufStart != 0 &&
  ///               OutBufEnd - OutBufStart >= 64).
  char *OutBufStart, *OutBufEnd, *OutBufCur;
  bool Unbuffered;

  /// Error This flag is true if an error of any kind has been detected.
  ///
  bool Error;

public:
  // color order matches ANSI escape sequence, don't change
  enum Colors {
    BLACK=0,
    RED,
    GREEN,
    YELLOW,
    BLUE,
    MAGENTA,
    CYAN,
    WHITE,
    SAVEDCOLOR
  };

  explicit raw_ostream(bool unbuffered=false)
    : Unbuffered(unbuffered), Error(false) {
    // Start out ready to flush.
    OutBufStart = OutBufEnd = OutBufCur = 0;
  }

  virtual ~raw_ostream();

  /// tell - Return the current offset with the file.
  uint64_t tell() { return current_pos() + GetNumBytesInBuffer(); }

  /// has_error - Return the value of the flag in this raw_ostream indicating
  /// whether an output error has been encountered.
  bool has_error() const {
    return Error;
  }

  /// clear_error - Set the flag read by has_error() to false. If the error
  /// flag is set at the time when this raw_ostream's destructor is called,
  /// llvm_report_error is called to report the error. Use clear_error()
  /// after handling the error to avoid this behavior.
  void clear_error() {
    Error = false;
  }

  //===--------------------------------------------------------------------===//
  // Configuration Interface
  //===--------------------------------------------------------------------===//

  /// SetBuffered - Set the stream to be buffered, with an automatically
  /// determined buffer size.
  void SetBuffered();

  /// SetBufferrSize - Set the stream to be buffered, using the
  /// specified buffer size.
  void SetBufferSize(size_t Size);

  size_t GetBufferSize() {
    // If we're supposed to be buffered but haven't actually gotten around
    // to allocating the buffer yet, return the value that would be used.
    if (!Unbuffered && !OutBufStart)
      return preferred_buffer_size();

    // Otherwise just return the size of the allocated buffer.
    return OutBufEnd - OutBufStart;
  }

  /// SetUnbuffered - Set the stream to be unbuffered. When
  /// unbuffered, the stream will flush after every write. This routine
  /// will also flush the buffer immediately when the stream is being
  /// set to unbuffered.
  void SetUnbuffered();

  size_t GetNumBytesInBuffer() const {
    return OutBufCur - OutBufStart;
  }

  //===--------------------------------------------------------------------===//
  // Data Output Interface
  //===--------------------------------------------------------------------===//

  void flush() {
    if (OutBufCur != OutBufStart)
      flush_nonempty();
  }

  raw_ostream &operator<<(char C) {
    if (OutBufCur >= OutBufEnd)
      return write(C);
    *OutBufCur++ = C;
    return *this;
  }

  raw_ostream &operator<<(unsigned char C) {
    if (OutBufCur >= OutBufEnd)
      return write(C);
    *OutBufCur++ = C;
    return *this;
  }

  raw_ostream &operator<<(signed char C) {
    if (OutBufCur >= OutBufEnd)
      return write(C);
    *OutBufCur++ = C;
    return *this;
  }

  raw_ostream &operator<<(const StringRef &Str) {
    // Inline fast path, particularly for strings with a known length.
    size_t Size = Str.size();

    // Make sure we can use the fast path.
    if (OutBufCur+Size > OutBufEnd)
      return write(Str.data(), Size);

    memcpy(OutBufCur, Str.data(), Size);
    OutBufCur += Size;
    return *this;
  }

  raw_ostream &operator<<(const char *Str) {
    // Inline fast path, particulary for constant strings where a sufficiently
    // smart compiler will simplify strlen.

    this->operator<<(StringRef(Str));
    return *this;
  }

  raw_ostream &operator<<(const std::string& Str) {
    write(Str.data(), Str.length());
    return *this;
  }

  raw_ostream &operator<<(unsigned long N);
  raw_ostream &operator<<(long N);
  raw_ostream &operator<<(unsigned long long N);
  raw_ostream &operator<<(long long N);
  raw_ostream &operator<<(const void *P);
  raw_ostream &operator<<(unsigned int N) {
    this->operator<<(static_cast<unsigned long>(N));
    return *this;
  }

  raw_ostream &operator<<(int N) {
    this->operator<<(static_cast<long>(N));
    return *this;
  }

  raw_ostream &operator<<(double N) {
    this->operator<<(ftostr(N));
    return *this;
  }

  /// write_hex - Output \arg N in hexadecimal, without any prefix or padding.
  raw_ostream &write_hex(unsigned long long N);

  raw_ostream &write(unsigned char C);
  raw_ostream &write(const char *Ptr, size_t Size);

  // Formatted output, see the format() function in Support/Format.h.
  raw_ostream &operator<<(const format_object_base &Fmt);

  /// Changes the foreground color of text that will be output from this point
  /// forward.
  /// @param colors ANSI color to use, the special SAVEDCOLOR can be used to
  /// change only the bold attribute, and keep colors untouched
  /// @param bold bold/brighter text, default false
  /// @param bg if true change the background, default: change foreground
  /// @returns itself so it can be used within << invocations
  virtual raw_ostream &changeColor(enum Colors colors, bool bold=false,
                                   bool  bg=false) { return *this; }

  /// Resets the colors to terminal defaults. Call this when you are done
  /// outputting colored text, or before program exit.
  virtual raw_ostream &resetColor() { return *this; }

  //===--------------------------------------------------------------------===//
  // Subclass Interface
  //===--------------------------------------------------------------------===//

private:
  /// write_impl - The is the piece of the class that is implemented
  /// by subclasses.  This writes the \args Size bytes starting at
  /// \arg Ptr to the underlying stream.
  /// 
  /// \invariant { Size > 0 }
  virtual void write_impl(const char *Ptr, size_t Size) = 0;

  // An out of line virtual method to provide a home for the class vtable.
  virtual void handle();

  /// current_pos - Return the current position within the stream, not
  /// counting the bytes currently in the buffer.
  virtual uint64_t current_pos() = 0;

protected:
  /// preferred_buffer_size - Return an efficient buffer size for the
  /// underlying output mechanism.
  virtual size_t preferred_buffer_size();

  /// error_detected - Set the flag indicating that an output error has
  /// been encountered.
  void error_detected() { Error = true; }

  typedef char * iterator;
  iterator begin() { return OutBufStart; }
  iterator end() { return OutBufCur; }

  //===--------------------------------------------------------------------===//
  // Private Interface
  //===--------------------------------------------------------------------===//
private:
  /// flush_nonempty - Flush the current buffer, which is known to be
  /// non-empty. This outputs the currently buffered data and resets
  /// the buffer to empty.
  void flush_nonempty();

  /// copy_to_buffer - Copy data into the buffer. Size must not be
  /// greater than the number of unused bytes in the buffer.
  void copy_to_buffer(const char *Ptr, size_t Size);
};

//===----------------------------------------------------------------------===//
// File Output Streams
//===----------------------------------------------------------------------===//

/// raw_fd_ostream - A raw_ostream that writes to a file descriptor.
///
class raw_fd_ostream : public raw_ostream {
  int FD;
  bool ShouldClose;
  uint64_t pos;

  /// write_impl - See raw_ostream::write_impl.
  virtual void write_impl(const char *Ptr, size_t Size);

  /// current_pos - Return the current position within the stream, not
  /// counting the bytes currently in the buffer.
  virtual uint64_t current_pos() { return pos; }

  /// preferred_buffer_size - Determine an efficient buffer size.
  virtual size_t preferred_buffer_size();

public:
  /// raw_fd_ostream - Open the specified file for writing. If an
  /// error occurs, information about the error is put into ErrorInfo,
  /// and the stream should be immediately destroyed; the string will
  /// be empty if no error occurred.
  ///
  /// \param Filename - The file to open. If this is "-" then the
  /// stream will use stdout instead.
  /// \param Binary - The file should be opened in binary mode on
  /// platforms that support this distinction.
  /// \param Force - Don't consider the case where the file already
  /// exists to be an error.
  raw_fd_ostream(const char *Filename, bool Binary, bool Force,
                 std::string &ErrorInfo);

  /// raw_fd_ostream ctor - FD is the file descriptor that this writes to.  If
  /// ShouldClose is true, this closes the file when the stream is destroyed.
  raw_fd_ostream(int fd, bool shouldClose, 
                 bool unbuffered=false) : raw_ostream(unbuffered), FD(fd), 
                                          ShouldClose(shouldClose) {}
  
  ~raw_fd_ostream();

  /// close - Manually flush the stream and close the file.
  void close();

  /// tell - Return the current offset with the file.
  uint64_t tell() { return pos + GetNumBytesInBuffer(); }

  /// seek - Flushes the stream and repositions the underlying file descriptor
  ///  positition to the offset specified from the beginning of the file.
  uint64_t seek(uint64_t off);

  virtual raw_ostream &changeColor(enum Colors colors, bool bold=false,
                                   bool bg=false);
  virtual raw_ostream &resetColor();
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

/// nulls() - This returns a reference to a raw_ostream which simply discards
/// output.
raw_ostream &nulls();

//===----------------------------------------------------------------------===//
// Output Stream Adaptors
//===----------------------------------------------------------------------===//

/// raw_os_ostream - A raw_ostream that writes to an std::ostream.  This is a
/// simple adaptor class.  It does not check for output errors; clients should
/// use the underlying stream to detect errors.
class raw_os_ostream : public raw_ostream {
  std::ostream &OS;

  /// write_impl - See raw_ostream::write_impl.
  virtual void write_impl(const char *Ptr, size_t Size);

  /// current_pos - Return the current position within the stream, not
  /// counting the bytes currently in the buffer.
  virtual uint64_t current_pos();

public:
  raw_os_ostream(std::ostream &O) : raw_ostream(true), OS(O) {}

  /// tell - Return the current offset with the stream.
  uint64_t tell();
};

/// raw_string_ostream - A raw_ostream that writes to an std::string.  This is a
/// simple adaptor class. This class does not encounter output errors.
class raw_string_ostream : public raw_ostream {
  std::string &OS;

  /// write_impl - See raw_ostream::write_impl.
  virtual void write_impl(const char *Ptr, size_t Size);

  /// current_pos - Return the current position within the stream, not
  /// counting the bytes currently in the buffer.
  virtual uint64_t current_pos() { return OS.size(); }
public:
  explicit raw_string_ostream(std::string &O) : raw_ostream(true), OS(O) {}

  /// tell - Return the current offset with the stream.
  uint64_t tell() { return OS.size() + GetNumBytesInBuffer(); }

  /// str - Flushes the stream contents to the target string and returns
  ///  the string's reference.
  std::string& str() {
    flush();
    return OS;
  }
};

/// raw_svector_ostream - A raw_ostream that writes to an SmallVector or
/// SmallString.  This is a simple adaptor class. This class does not
/// encounter output errors.
class raw_svector_ostream : public raw_ostream {
  SmallVectorImpl<char> &OS;

  /// write_impl - See raw_ostream::write_impl.
  virtual void write_impl(const char *Ptr, size_t Size);

  /// current_pos - Return the current position within the stream, not
  /// counting the bytes currently in the buffer.
  virtual uint64_t current_pos();
public:
  explicit raw_svector_ostream(SmallVectorImpl<char> &O)
    : raw_ostream(true), OS(O) {}

  /// tell - Return the current offset with the stream.
  uint64_t tell();
};

/// raw_null_ostream - A raw_ostream that discards all output.
class raw_null_ostream : public raw_ostream {
  /// write_impl - See raw_ostream::write_impl.
  virtual void write_impl(const char *Ptr, size_t size);
  
  /// current_pos - Return the current position within the stream, not
  /// counting the bytes currently in the buffer.
  virtual uint64_t current_pos();

public:
  explicit raw_null_ostream() {}
  ~raw_null_ostream();
};

} // end llvm namespace

#endif
