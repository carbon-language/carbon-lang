//===--- raw_ostream.cpp - Implement the raw_ostream classes --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements support for bulk buffered stream output.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Format.h"
#include "llvm/System/Program.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Config/config.h"
#include "llvm/Support/Compiler.h"
#include <ostream>

#if defined(HAVE_UNISTD_H)
# include <unistd.h>
#endif
#if defined(HAVE_FCNTL_H)
# include <fcntl.h>
#endif

#if defined(_MSC_VER)
#include <io.h>
#include <fcntl.h>
#ifndef STDIN_FILENO
# define STDIN_FILENO 0
#endif
#ifndef STDOUT_FILENO
# define STDOUT_FILENO 1
#endif
#ifndef STDERR_FILENO
# define STDERR_FILENO 2
#endif
#endif

using namespace llvm;


// An out of line virtual method to provide a home for the class vtable.
void raw_ostream::handle() {}

raw_ostream &raw_ostream::operator<<(unsigned long N) {
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

raw_ostream &raw_ostream::operator<<(long N) {
  if (N <  0) {
    *this << '-';
    N = -N;
  }
  
  return this->operator<<(static_cast<unsigned long>(N));
}

raw_ostream &raw_ostream::operator<<(unsigned long long N) {
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

raw_ostream &raw_ostream::operator<<(long long N) {
  if (N <  0) {
    *this << '-';
    N = -N;
  }
  
  return this->operator<<(static_cast<unsigned long long>(N));
}

raw_ostream &raw_ostream::operator<<(const void *P) {
  uintptr_t N = (uintptr_t) P;
  *this << '0' << 'x';
  
  // Zero is a special case.
  if (N == 0)
    return *this << '0';

  char NumberBuffer[20];
  char *EndPtr = NumberBuffer+sizeof(NumberBuffer);
  char *CurPtr = EndPtr;

  while (N) {
    unsigned x = N % 16;
    *--CurPtr = (x < 10 ? '0' + x : 'a' + x - 10);
    N /= 16;
  }

  return write(CurPtr, EndPtr-CurPtr);
}

void raw_ostream::flush_nonempty() {
  assert(OutBufCur > OutBufStart && "Invalid call to flush_nonempty.");
  write_impl(OutBufStart, OutBufCur - OutBufStart);
  OutBufCur = OutBufStart;    
}

raw_ostream &raw_ostream::write(unsigned char C) {
  // Group exceptional cases into a single branch.
  if (OutBufCur >= OutBufEnd) {
    if (Unbuffered) {
      write_impl(reinterpret_cast<char*>(&C), 1);
      return *this;
    }
    
    if (!OutBufStart)
      SetBufferSize();
    else
      flush_nonempty();
  }

  *OutBufCur++ = C;
  return *this;
}

raw_ostream &raw_ostream::write(const char *Ptr, unsigned Size) {
  // Group exceptional cases into a single branch.
  if (BUILTIN_EXPECT(OutBufCur+Size > OutBufEnd, false)) {
    if (Unbuffered) {
      write_impl(Ptr, Size);
      return *this;
    }
    
    if (!OutBufStart)
      SetBufferSize();
    else
      flush_nonempty();
  }
  
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

    // Otherwise we are emitting a string larger than our buffer. We
    // know we already flushed, so just write it out directly.
    write_impl(Ptr, Size);
    Size = 0;
    break;
  }
  OutBufCur += Size;

  return *this;
}

// Formatted output.
raw_ostream &raw_ostream::operator<<(const format_object_base &Fmt) {
  // If we have more than a few bytes left in our output buffer, try
  // formatting directly onto its end.
  //
  // FIXME: This test is a bit silly, since if we don't have enough
  // space in the buffer we will have to flush the formatted output
  // anyway. We should just flush upfront in such cases, and use the
  // whole buffer as our scratch pad. Note, however, that this case is
  // also necessary for correctness on unbuffered streams.
  unsigned NextBufferSize = 127;
  if (OutBufEnd-OutBufCur > 3) {
    unsigned BufferBytesLeft = OutBufEnd-OutBufCur;
    unsigned BytesUsed = Fmt.print(OutBufCur, BufferBytesLeft);
    
    // Common case is that we have plenty of space.
    if (BytesUsed < BufferBytesLeft) {
      OutBufCur += BytesUsed;
      return *this;
    }
    
    // Otherwise, we overflowed and the return value tells us the size to try
    // again with.
    NextBufferSize = BytesUsed;
  }
  
  // If we got here, we didn't have enough space in the output buffer for the
  // string.  Try printing into a SmallVector that is resized to have enough
  // space.  Iterate until we win.
  SmallVector<char, 128> V;
  
  while (1) {
    V.resize(NextBufferSize);
    
    // Try formatting into the SmallVector.
    unsigned BytesUsed = Fmt.print(&V[0], NextBufferSize);
    
    // If BytesUsed fit into the vector, we win.
    if (BytesUsed <= NextBufferSize)
      return write(&V[0], BytesUsed);
    
    // Otherwise, try again with a new size.
    assert(BytesUsed > NextBufferSize && "Didn't grow buffer!?");
    NextBufferSize = BytesUsed;
  }
}

//===----------------------------------------------------------------------===//
//  Formatted Output
//===----------------------------------------------------------------------===//

// Out of line virtual method.
void format_object_base::home() {
}

//===----------------------------------------------------------------------===//
//  raw_fd_ostream
//===----------------------------------------------------------------------===//

/// raw_fd_ostream - Open the specified file for writing. If an error
/// occurs, information about the error is put into ErrorInfo, and the
/// stream should be immediately destroyed; the string will be empty
/// if no error occurred.
raw_fd_ostream::raw_fd_ostream(const char *Filename, bool Binary,
                               std::string &ErrorInfo) : pos(0) {
  ErrorInfo.clear();

  // Handle "-" as stdout.
  if (Filename[0] == '-' && Filename[1] == 0) {
    FD = STDOUT_FILENO;
    // If user requested binary then put stdout into binary mode if
    // possible.
    if (Binary)
      sys::Program::ChangeStdoutToBinary();
    ShouldClose = false;
    return;
  }
  
  int Flags = O_WRONLY|O_CREAT|O_TRUNC;
#ifdef O_BINARY
  if (Binary)
    Flags |= O_BINARY;
#endif
  FD = open(Filename, Flags, 0644);
  if (FD < 0) {
    ErrorInfo = "Error opening output file '" + std::string(Filename) + "'";
    ShouldClose = false;
  } else {
    ShouldClose = true;
  }
}

raw_fd_ostream::~raw_fd_ostream() {
  if (FD >= 0) {
    flush();
    if (ShouldClose)
      ::close(FD);
  }
}

void raw_fd_ostream::write_impl(const char *Ptr, unsigned Size) {
  assert (FD >= 0 && "File already closed.");
  pos += Size;
  ::write(FD, Ptr, Size);
}

void raw_fd_ostream::close() {
  assert (ShouldClose);
  ShouldClose = false;
  flush();
  ::close(FD);
  FD = -1;
}

uint64_t raw_fd_ostream::seek(uint64_t off) {
  flush();
  pos = lseek(FD, off, SEEK_SET);
  return pos;  
}

//===----------------------------------------------------------------------===//
//  raw_stdout/err_ostream
//===----------------------------------------------------------------------===//

raw_stdout_ostream::raw_stdout_ostream():raw_fd_ostream(STDOUT_FILENO, false) {}
raw_stderr_ostream::raw_stderr_ostream():raw_fd_ostream(STDERR_FILENO, false, 
                                                        true) {}

// An out of line virtual method to provide a home for the class vtable.
void raw_stdout_ostream::handle() {}
void raw_stderr_ostream::handle() {}

/// outs() - This returns a reference to a raw_ostream for standard output.
/// Use it like: outs() << "foo" << "bar";
raw_ostream &llvm::outs() {
  static raw_stdout_ostream S;
  return S;
}

/// errs() - This returns a reference to a raw_ostream for standard error.
/// Use it like: errs() << "foo" << "bar";
raw_ostream &llvm::errs() {
  static raw_stderr_ostream S;
  return S;
}

//===----------------------------------------------------------------------===//
//  raw_os_ostream
//===----------------------------------------------------------------------===//

raw_os_ostream::~raw_os_ostream() {
  flush();
}

void raw_os_ostream::write_impl(const char *Ptr, unsigned Size) {
  OS.write(Ptr, Size);
}

uint64_t raw_os_ostream::current_pos() { return OS.tellp(); }

uint64_t raw_os_ostream::tell() { 
  return (uint64_t)OS.tellp() + GetNumBytesInBuffer(); 
}

//===----------------------------------------------------------------------===//
//  raw_string_ostream
//===----------------------------------------------------------------------===//

raw_string_ostream::~raw_string_ostream() {
  flush();
}

void raw_string_ostream::write_impl(const char *Ptr, unsigned Size) {
  OS.append(Ptr, Size);
}

//===----------------------------------------------------------------------===//
//  raw_svector_ostream
//===----------------------------------------------------------------------===//

raw_svector_ostream::~raw_svector_ostream() {
  flush();
}

void raw_svector_ostream::write_impl(const char *Ptr, unsigned Size) {
  OS.append(Ptr, Ptr + Size);
}

uint64_t raw_svector_ostream::current_pos() { return OS.size(); }

uint64_t raw_svector_ostream::tell() { 
  return OS.size() + GetNumBytesInBuffer(); 
}
