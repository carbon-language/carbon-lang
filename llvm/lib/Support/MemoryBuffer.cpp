//===--- MemoryBuffer.cpp - Memory Buffer implementation ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the MemoryBuffer interface.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/MemoryBuffer.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Errno.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/system_error.h"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <cerrno>
#include <sys/types.h>
#include <sys/stat.h>
#if !defined(_MSC_VER) && !defined(__MINGW32__)
#include <unistd.h>
#include <sys/uio.h>
#else
#include <io.h>
#endif
#include <fcntl.h>
using namespace llvm;

//===----------------------------------------------------------------------===//
// MemoryBuffer implementation itself.
//===----------------------------------------------------------------------===//

MemoryBuffer::~MemoryBuffer() { }

/// init - Initialize this MemoryBuffer as a reference to externally allocated
/// memory, memory that we know is already null terminated.
void MemoryBuffer::init(const char *BufStart, const char *BufEnd) {
  assert(BufEnd[0] == 0 && "Buffer is not null terminated!");
  BufferStart = BufStart;
  BufferEnd = BufEnd;
}

//===----------------------------------------------------------------------===//
// MemoryBufferMem implementation.
//===----------------------------------------------------------------------===//

/// CopyStringRef - Copies contents of a StringRef into a block of memory and
/// null-terminates it.
static void CopyStringRef(char *Memory, StringRef Data) {
  memcpy(Memory, Data.data(), Data.size());
  Memory[Data.size()] = 0; // Null terminate string.
}

/// GetNamedBuffer - Allocates a new MemoryBuffer with Name copied after it.
template <typename T>
static T* GetNamedBuffer(StringRef Buffer, StringRef Name) {
  char *Mem = static_cast<char*>(operator new(sizeof(T) + Name.size() + 1));
  CopyStringRef(Mem + sizeof(T), Name);
  return new (Mem) T(Buffer);
}

namespace {
/// MemoryBufferMem - Named MemoryBuffer pointing to a block of memory.
class MemoryBufferMem : public MemoryBuffer {
public:
  MemoryBufferMem(StringRef InputData) {
    init(InputData.begin(), InputData.end());
  }

  virtual const char *getBufferIdentifier() const {
     // The name is stored after the class itself.
    return reinterpret_cast<const char*>(this + 1);
  }
};
}

/// getMemBuffer - Open the specified memory range as a MemoryBuffer.  Note
/// that EndPtr[0] must be a null byte and be accessible!
MemoryBuffer *MemoryBuffer::getMemBuffer(StringRef InputData,
                                         StringRef BufferName) {
  return GetNamedBuffer<MemoryBufferMem>(InputData, BufferName);
}

/// getMemBufferCopy - Open the specified memory range as a MemoryBuffer,
/// copying the contents and taking ownership of it.  This has no requirements
/// on EndPtr[0].
MemoryBuffer *MemoryBuffer::getMemBufferCopy(StringRef InputData,
                                             StringRef BufferName) {
  MemoryBuffer *Buf = getNewUninitMemBuffer(InputData.size(), BufferName);
  if (!Buf) return 0;
  memcpy(const_cast<char*>(Buf->getBufferStart()), InputData.data(),
         InputData.size());
  return Buf;
}

/// getNewUninitMemBuffer - Allocate a new MemoryBuffer of the specified size
/// that is not initialized.  Note that the caller should initialize the
/// memory allocated by this method.  The memory is owned by the MemoryBuffer
/// object.
MemoryBuffer *MemoryBuffer::getNewUninitMemBuffer(size_t Size,
                                                  StringRef BufferName) {
  // Allocate space for the MemoryBuffer, the data and the name. It is important
  // that MemoryBuffer and data are aligned so PointerIntPair works with them.
  size_t AlignedStringLen =
    RoundUpToAlignment(sizeof(MemoryBufferMem) + BufferName.size() + 1,
                       sizeof(void*)); // TODO: Is sizeof(void*) enough?
  size_t RealLen = AlignedStringLen + Size + 1;
  char *Mem = static_cast<char*>(operator new(RealLen, std::nothrow));
  if (!Mem) return 0;

  // The name is stored after the class itself.
  CopyStringRef(Mem + sizeof(MemoryBufferMem), BufferName);

  // The buffer begins after the name and must be aligned.
  char *Buf = Mem + AlignedStringLen;
  Buf[Size] = 0; // Null terminate buffer.

  return new (Mem) MemoryBufferMem(StringRef(Buf, Size));
}

/// getNewMemBuffer - Allocate a new MemoryBuffer of the specified size that
/// is completely initialized to zeros.  Note that the caller should
/// initialize the memory allocated by this method.  The memory is owned by
/// the MemoryBuffer object.
MemoryBuffer *MemoryBuffer::getNewMemBuffer(size_t Size, StringRef BufferName) {
  MemoryBuffer *SB = getNewUninitMemBuffer(Size, BufferName);
  if (!SB) return 0;
  memset(const_cast<char*>(SB->getBufferStart()), 0, Size);
  return SB;
}


/// getFileOrSTDIN - Open the specified file as a MemoryBuffer, or open stdin
/// if the Filename is "-".  If an error occurs, this returns null and fills
/// in *ErrStr with a reason.  If stdin is empty, this API (unlike getSTDIN)
/// returns an empty buffer.
MemoryBuffer *MemoryBuffer::getFileOrSTDIN(StringRef Filename,
                                           error_code &ec,
                                           int64_t FileSize) {
  if (Filename == "-")
    return getSTDIN(ec);
  return getFile(Filename, ec, FileSize);
}

MemoryBuffer *MemoryBuffer::getFileOrSTDIN(const char *Filename,
                                           error_code &ec,
                                           int64_t FileSize) {
  if (strcmp(Filename, "-") == 0)
    return getSTDIN(ec);
  return getFile(Filename, ec, FileSize);
}

//===----------------------------------------------------------------------===//
// MemoryBuffer::getFile implementation.
//===----------------------------------------------------------------------===//

namespace {
/// MemoryBufferMMapFile - This represents a file that was mapped in with the
/// sys::Path::MapInFilePages method.  When destroyed, it calls the
/// sys::Path::UnMapFilePages method.
class MemoryBufferMMapFile : public MemoryBufferMem {
public:
  MemoryBufferMMapFile(StringRef Buffer)
    : MemoryBufferMem(Buffer) { }

  ~MemoryBufferMMapFile() {
    sys::Path::UnMapFilePages(getBufferStart(), getBufferSize());
  }
};

/// FileCloser - RAII object to make sure an FD gets closed properly.
class FileCloser {
  int FD;
public:
  explicit FileCloser(int FD) : FD(FD) {}
  ~FileCloser() { ::close(FD); }
};
}

MemoryBuffer *MemoryBuffer::getFile(StringRef Filename, error_code &ec,
                                    int64_t FileSize) {
  // Ensure the path is null terminated.
  SmallString<256> PathBuf(Filename.begin(), Filename.end());
  return MemoryBuffer::getFile(PathBuf.c_str(), ec, FileSize);
}

MemoryBuffer *MemoryBuffer::getFile(const char *Filename, error_code &ec,
                                    int64_t FileSize) {
  int OpenFlags = O_RDONLY;
#ifdef O_BINARY
  OpenFlags |= O_BINARY;  // Open input file in binary mode on win32.
#endif
  int FD = ::open(Filename, OpenFlags);
  if (FD == -1) {
    ec = error_code(errno, posix_category());
    return 0;
  }

  return getOpenFile(FD, Filename, ec, FileSize);
}

MemoryBuffer *MemoryBuffer::getOpenFile(int FD, const char *Filename,
                                        error_code &ec, int64_t FileSize) {
  FileCloser FC(FD); // Close FD on return.

  // If we don't know the file size, use fstat to find out.  fstat on an open
  // file descriptor is cheaper than stat on a random path.
  if (FileSize == -1) {
    struct stat FileInfo;
    // TODO: This should use fstat64 when available.
    if (fstat(FD, &FileInfo) == -1) {
      ec = error_code(errno, posix_category());
      return 0;
    }
    FileSize = FileInfo.st_size;
  }


  // If the file is large, try to use mmap to read it in.  We don't use mmap
  // for small files, because this can severely fragment our address space. Also
  // don't try to map files that are exactly a multiple of the system page size,
  // as the file would not have the required null terminator.
  //
  // FIXME: Can we just mmap an extra page in the latter case?
  if (FileSize >= 4096*4 &&
      (FileSize & (sys::Process::GetPageSize()-1)) != 0) {
    if (const char *Pages = sys::Path::MapInFilePages(FD, FileSize)) {
      return GetNamedBuffer<MemoryBufferMMapFile>(StringRef(Pages, FileSize),
                                                  Filename);
    }
  }

  MemoryBuffer *Buf = MemoryBuffer::getNewUninitMemBuffer(FileSize, Filename);
  if (!Buf) {
    // Failed to create a buffer. The only way it can fail is if
    // new(std::nothrow) returns 0.
    ec = make_error_code(errc::not_enough_memory);
    return 0;
  }

  OwningPtr<MemoryBuffer> SB(Buf);
  char *BufPtr = const_cast<char*>(SB->getBufferStart());

  size_t BytesLeft = FileSize;
  while (BytesLeft) {
    ssize_t NumRead = ::read(FD, BufPtr, BytesLeft);
    if (NumRead == -1) {
      if (errno == EINTR)
        continue;
      // Error while reading.
      ec = error_code(errno, posix_category());
      return 0;
    } else if (NumRead == 0) {
      // We hit EOF early, truncate and terminate buffer.
      Buf->BufferEnd = BufPtr;
      *BufPtr = 0;
      return SB.take();
    }
    BytesLeft -= NumRead;
    BufPtr += NumRead;
  }

  return SB.take();
}

//===----------------------------------------------------------------------===//
// MemoryBuffer::getSTDIN implementation.
//===----------------------------------------------------------------------===//

MemoryBuffer *MemoryBuffer::getSTDIN(error_code &ec) {
  // Read in all of the data from stdin, we cannot mmap stdin.
  //
  // FIXME: That isn't necessarily true, we should try to mmap stdin and
  // fallback if it fails.
  sys::Program::ChangeStdinToBinary();

  const ssize_t ChunkSize = 4096*4;
  SmallString<ChunkSize> Buffer;
  ssize_t ReadBytes;
  // Read into Buffer until we hit EOF.
  do {
    Buffer.reserve(Buffer.size() + ChunkSize);
    ReadBytes = read(0, Buffer.end(), ChunkSize);
    if (ReadBytes == -1) {
      if (errno == EINTR) continue;
      ec = error_code(errno, posix_category());
      return 0;
    }
    Buffer.set_size(Buffer.size() + ReadBytes);
  } while (ReadBytes != 0);

  return getMemBufferCopy(Buffer, "<stdin>");
}
