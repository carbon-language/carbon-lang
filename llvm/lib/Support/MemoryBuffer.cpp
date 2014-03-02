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
#include "llvm/Config/config.h"
#include "llvm/Support/Errno.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/system_error.h"
#include <cassert>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <new>
#include <sys/stat.h>
#include <sys/types.h>
#if !defined(_MSC_VER) && !defined(__MINGW32__)
#include <unistd.h>
#else
#include <io.h>
// Simplistic definitinos of these macros for use in getOpenFile.
#ifndef S_ISREG
#define S_ISREG(x) (1)
#endif
#ifndef S_ISBLK
#define S_ISBLK(x) (0)
#endif
#endif
using namespace llvm;

//===----------------------------------------------------------------------===//
// MemoryBuffer implementation itself.
//===----------------------------------------------------------------------===//

MemoryBuffer::~MemoryBuffer() { }

/// init - Initialize this MemoryBuffer as a reference to externally allocated
/// memory, memory that we know is already null terminated.
void MemoryBuffer::init(const char *BufStart, const char *BufEnd,
                        bool RequiresNullTerminator) {
  assert((!RequiresNullTerminator || BufEnd[0] == 0) &&
         "Buffer is not null terminated!");
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

namespace {
struct NamedBufferAlloc {
  StringRef Name;
  NamedBufferAlloc(StringRef Name) : Name(Name) {}
};
}

void *operator new(size_t N, const NamedBufferAlloc &Alloc) {
  char *Mem = static_cast<char *>(operator new(N + Alloc.Name.size() + 1));
  CopyStringRef(Mem + N, Alloc.Name);
  return Mem;
}

namespace {
/// MemoryBufferMem - Named MemoryBuffer pointing to a block of memory.
class MemoryBufferMem : public MemoryBuffer {
public:
  MemoryBufferMem(StringRef InputData, bool RequiresNullTerminator) {
    init(InputData.begin(), InputData.end(), RequiresNullTerminator);
  }

  virtual const char *getBufferIdentifier() const override {
     // The name is stored after the class itself.
    return reinterpret_cast<const char*>(this + 1);
  }

  virtual BufferKind getBufferKind() const override {
    return MemoryBuffer_Malloc;
  }
};
}

/// getMemBuffer - Open the specified memory range as a MemoryBuffer.  Note
/// that InputData must be a null terminated if RequiresNullTerminator is true!
MemoryBuffer *MemoryBuffer::getMemBuffer(StringRef InputData,
                                         StringRef BufferName,
                                         bool RequiresNullTerminator) {
  return new (NamedBufferAlloc(BufferName))
      MemoryBufferMem(InputData, RequiresNullTerminator);
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
  // TODO: Is 16-byte alignment enough?  We copy small object files with large
  // alignment expectations into this buffer.
  size_t AlignedStringLen =
      RoundUpToAlignment(sizeof(MemoryBufferMem) + BufferName.size() + 1, 16);
  size_t RealLen = AlignedStringLen + Size + 1;
  char *Mem = static_cast<char*>(operator new(RealLen, std::nothrow));
  if (!Mem) return 0;

  // The name is stored after the class itself.
  CopyStringRef(Mem + sizeof(MemoryBufferMem), BufferName);

  // The buffer begins after the name and must be aligned.
  char *Buf = Mem + AlignedStringLen;
  Buf[Size] = 0; // Null terminate buffer.

  return new (Mem) MemoryBufferMem(StringRef(Buf, Size), true);
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
error_code MemoryBuffer::getFileOrSTDIN(StringRef Filename,
                                        OwningPtr<MemoryBuffer> &result,
                                        int64_t FileSize) {
  if (Filename == "-")
    return getSTDIN(result);
  return getFile(Filename, result, FileSize);
}

//===----------------------------------------------------------------------===//
// MemoryBuffer::getFile implementation.
//===----------------------------------------------------------------------===//

namespace {
/// \brief Memory maps a file descriptor using sys::fs::mapped_file_region.
///
/// This handles converting the offset into a legal offset on the platform.
class MemoryBufferMMapFile : public MemoryBuffer {
  sys::fs::mapped_file_region MFR;

  static uint64_t getLegalMapOffset(uint64_t Offset) {
    return Offset & ~(sys::fs::mapped_file_region::alignment() - 1);
  }

  static uint64_t getLegalMapSize(uint64_t Len, uint64_t Offset) {
    return Len + (Offset - getLegalMapOffset(Offset));
  }

  const char *getStart(uint64_t Len, uint64_t Offset) {
    return MFR.const_data() + (Offset - getLegalMapOffset(Offset));
  }

public:
  MemoryBufferMMapFile(bool RequiresNullTerminator, int FD, uint64_t Len,
                       uint64_t Offset, error_code EC)
      : MFR(FD, false, sys::fs::mapped_file_region::readonly,
            getLegalMapSize(Len, Offset), getLegalMapOffset(Offset), EC) {
    if (!EC) {
      const char *Start = getStart(Len, Offset);
      init(Start, Start + Len, RequiresNullTerminator);
    }
  }

  virtual const char *getBufferIdentifier() const override {
    // The name is stored after the class itself.
    return reinterpret_cast<const char *>(this + 1);
  }

  virtual BufferKind getBufferKind() const override {
    return MemoryBuffer_MMap;
  }
};
}

static error_code getMemoryBufferForStream(int FD,
                                           StringRef BufferName,
                                           OwningPtr<MemoryBuffer> &result) {
  const ssize_t ChunkSize = 4096*4;
  SmallString<ChunkSize> Buffer;
  ssize_t ReadBytes;
  // Read into Buffer until we hit EOF.
  do {
    Buffer.reserve(Buffer.size() + ChunkSize);
    ReadBytes = read(FD, Buffer.end(), ChunkSize);
    if (ReadBytes == -1) {
      if (errno == EINTR) continue;
      return error_code(errno, posix_category());
    }
    Buffer.set_size(Buffer.size() + ReadBytes);
  } while (ReadBytes != 0);

  result.reset(MemoryBuffer::getMemBufferCopy(Buffer, BufferName));
  return error_code::success();
}

static error_code getFileAux(const char *Filename,
                             OwningPtr<MemoryBuffer> &result, int64_t FileSize,
                             bool RequiresNullTerminator);

error_code MemoryBuffer::getFile(Twine Filename,
                                 OwningPtr<MemoryBuffer> &result,
                                 int64_t FileSize,
                                 bool RequiresNullTerminator) {
  // Ensure the path is null terminated.
  SmallString<256> PathBuf;
  StringRef NullTerminatedName = Filename.toNullTerminatedStringRef(PathBuf);
  return getFileAux(NullTerminatedName.data(), result, FileSize,
                    RequiresNullTerminator);
}

static error_code getOpenFileImpl(int FD, const char *Filename,
                                  OwningPtr<MemoryBuffer> &Result,
                                  uint64_t FileSize, uint64_t MapSize,
                                  int64_t Offset, bool RequiresNullTerminator);

static error_code getFileAux(const char *Filename,
                             OwningPtr<MemoryBuffer> &result, int64_t FileSize,
                             bool RequiresNullTerminator) {
  int FD;
  error_code EC = sys::fs::openFileForRead(Filename, FD);
  if (EC)
    return EC;

  error_code ret = getOpenFileImpl(FD, Filename, result, FileSize, FileSize, 0,
                                   RequiresNullTerminator);
  close(FD);
  return ret;
}

static bool shouldUseMmap(int FD,
                          size_t FileSize,
                          size_t MapSize,
                          off_t Offset,
                          bool RequiresNullTerminator,
                          int PageSize) {
  // We don't use mmap for small files because this can severely fragment our
  // address space.
  if (MapSize < 4 * 4096 || MapSize < (unsigned)PageSize)
    return false;

  if (!RequiresNullTerminator)
    return true;


  // If we don't know the file size, use fstat to find out.  fstat on an open
  // file descriptor is cheaper than stat on a random path.
  // FIXME: this chunk of code is duplicated, but it avoids a fstat when
  // RequiresNullTerminator = false and MapSize != -1.
  if (FileSize == size_t(-1)) {
    sys::fs::file_status Status;
    error_code EC = sys::fs::status(FD, Status);
    if (EC)
      return EC;
    FileSize = Status.getSize();
  }

  // If we need a null terminator and the end of the map is inside the file,
  // we cannot use mmap.
  size_t End = Offset + MapSize;
  assert(End <= FileSize);
  if (End != FileSize)
    return false;

#if defined(_WIN32) || defined(__CYGWIN__)
  // Don't peek the next page if file is multiple of *physical* pagesize(4k)
  // but is not multiple of AllocationGranularity(64k),
  // when a null terminator is required.
  // FIXME: It's not good to hardcode 4096 here. dwPageSize shows 4096.
  if ((FileSize & (4096 - 1)) == 0)
    return false;
#endif

  // Don't try to map files that are exactly a multiple of the system page size
  // if we need a null terminator.
  if ((FileSize & (PageSize -1)) == 0)
    return false;

  return true;
}

static error_code getOpenFileImpl(int FD, const char *Filename,
                                  OwningPtr<MemoryBuffer> &result,
                                  uint64_t FileSize, uint64_t MapSize,
                                  int64_t Offset, bool RequiresNullTerminator) {
  static int PageSize = sys::process::get_self()->page_size();

  // Default is to map the full file.
  if (MapSize == uint64_t(-1)) {
    // If we don't know the file size, use fstat to find out.  fstat on an open
    // file descriptor is cheaper than stat on a random path.
    if (FileSize == uint64_t(-1)) {
      sys::fs::file_status Status;
      error_code EC = sys::fs::status(FD, Status);
      if (EC)
        return EC;

      // If this not a file or a block device (e.g. it's a named pipe
      // or character device), we can't trust the size. Create the memory
      // buffer by copying off the stream.
      sys::fs::file_type Type = Status.type();
      if (Type != sys::fs::file_type::regular_file &&
          Type != sys::fs::file_type::block_file)
        return getMemoryBufferForStream(FD, Filename, result);

      FileSize = Status.getSize();
    }
    MapSize = FileSize;
  }

  if (shouldUseMmap(FD, FileSize, MapSize, Offset, RequiresNullTerminator,
                    PageSize)) {
    error_code EC;
    result.reset(new (NamedBufferAlloc(Filename)) MemoryBufferMMapFile(
        RequiresNullTerminator, FD, MapSize, Offset, EC));
    if (!EC)
      return error_code::success();
  }

  MemoryBuffer *Buf = MemoryBuffer::getNewUninitMemBuffer(MapSize, Filename);
  if (!Buf) {
    // Failed to create a buffer. The only way it can fail is if
    // new(std::nothrow) returns 0.
    return make_error_code(errc::not_enough_memory);
  }

  OwningPtr<MemoryBuffer> SB(Buf);
  char *BufPtr = const_cast<char*>(SB->getBufferStart());

  size_t BytesLeft = MapSize;
#ifndef HAVE_PREAD
  if (lseek(FD, Offset, SEEK_SET) == -1)
    return error_code(errno, posix_category());
#endif

  while (BytesLeft) {
#ifdef HAVE_PREAD
    ssize_t NumRead = ::pread(FD, BufPtr, BytesLeft, MapSize-BytesLeft+Offset);
#else
    ssize_t NumRead = ::read(FD, BufPtr, BytesLeft);
#endif
    if (NumRead == -1) {
      if (errno == EINTR)
        continue;
      // Error while reading.
      return error_code(errno, posix_category());
    }
    if (NumRead == 0) {
      assert(0 && "We got inaccurate FileSize value or fstat reported an "
                   "invalid file size.");
      *BufPtr = '\0'; // null-terminate at the actual size.
      break;
    }
    BytesLeft -= NumRead;
    BufPtr += NumRead;
  }

  result.swap(SB);
  return error_code::success();
}

error_code MemoryBuffer::getOpenFile(int FD, const char *Filename,
                                     OwningPtr<MemoryBuffer> &Result,
                                     uint64_t FileSize,
                                     bool RequiresNullTerminator) {
  return getOpenFileImpl(FD, Filename, Result, FileSize, FileSize, 0,
                         RequiresNullTerminator);
}

error_code MemoryBuffer::getOpenFileSlice(int FD, const char *Filename,
                                          OwningPtr<MemoryBuffer> &Result,
                                          uint64_t MapSize, int64_t Offset) {
  return getOpenFileImpl(FD, Filename, Result, -1, MapSize, Offset, false);
}

//===----------------------------------------------------------------------===//
// MemoryBuffer::getSTDIN implementation.
//===----------------------------------------------------------------------===//

error_code MemoryBuffer::getSTDIN(OwningPtr<MemoryBuffer> &result) {
  // Read in all of the data from stdin, we cannot mmap stdin.
  //
  // FIXME: That isn't necessarily true, we should try to mmap stdin and
  // fallback if it fails.
  sys::ChangeStdinToBinary();

  return getMemoryBufferForStream(0, "<stdin>", result);
}
