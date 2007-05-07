//===--- MemoryBuffer.cpp - Memory Buffer implementation ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the MemoryBuffer interface.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/MemoryBuffer.h"
#include "llvm/System/MappedFile.h"
#include "llvm/System/Process.h"
#include "llvm/System/Program.h"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <cerrno>
using namespace llvm;

//===----------------------------------------------------------------------===//
// MemoryBuffer implementation itself.
//===----------------------------------------------------------------------===//

MemoryBuffer::~MemoryBuffer() {
  if (MustDeleteBuffer)
    delete [] BufferStart;
}

/// initCopyOf - Initialize this source buffer with a copy of the specified
/// memory range.  We make the copy so that we can null terminate it
/// successfully.
void MemoryBuffer::initCopyOf(const char *BufStart, const char *BufEnd) {
  size_t Size = BufEnd-BufStart;
  BufferStart = new char[Size+1];
  BufferEnd = BufferStart+Size;
  memcpy(const_cast<char*>(BufferStart), BufStart, Size);
  *const_cast<char*>(BufferEnd) = 0;   // Null terminate buffer.
  MustDeleteBuffer = false;
}

/// init - Initialize this MemoryBuffer as a reference to externally allocated
/// memory, memory that we know is already null terminated.
void MemoryBuffer::init(const char *BufStart, const char *BufEnd) {
  assert(BufEnd[0] == 0 && "Buffer is not null terminated!");
  BufferStart = BufStart;
  BufferEnd = BufEnd;
  MustDeleteBuffer = false;
}

//===----------------------------------------------------------------------===//
// MemoryBufferMem implementation.
//===----------------------------------------------------------------------===//

namespace {
class MemoryBufferMem : public MemoryBuffer {
  std::string FileID;
public:
  MemoryBufferMem(const char *Start, const char *End, const char *FID)
  : FileID(FID) {
    init(Start, End);
  }
  
  virtual const char *getBufferIdentifier() const {
    return FileID.c_str();
  }
};
}

/// getMemBuffer - Open the specified memory range as a MemoryBuffer.  Note
/// that EndPtr[0] must be a null byte and be accessible!
MemoryBuffer *MemoryBuffer::getMemBuffer(const char *StartPtr, 
                                         const char *EndPtr,
                                         const char *BufferName) {
  return new MemoryBufferMem(StartPtr, EndPtr, BufferName);
}

/// getNewUninitMemBuffer - Allocate a new MemoryBuffer of the specified size
/// that is completely initialized to zeros.  Note that the caller should
/// initialize the memory allocated by this method.  The memory is owned by
/// the MemoryBuffer object.
MemoryBuffer *MemoryBuffer::getNewUninitMemBuffer(unsigned Size,
                                                  const char *BufferName) {
  char *Buf = new char[Size+1];
  Buf[Size] = 0;
  MemoryBufferMem *SB = new MemoryBufferMem(Buf, Buf+Size, BufferName);
  // The memory for this buffer is owned by the MemoryBuffer.
  SB->MustDeleteBuffer = true;
  return SB;
}

/// getNewMemBuffer - Allocate a new MemoryBuffer of the specified size that
/// is completely initialized to zeros.  Note that the caller should
/// initialize the memory allocated by this method.  The memory is owned by
/// the MemoryBuffer object.
MemoryBuffer *MemoryBuffer::getNewMemBuffer(unsigned Size,
                                            const char *BufferName) {
  MemoryBuffer *SB = getNewUninitMemBuffer(Size, BufferName);
  memset(const_cast<char*>(SB->getBufferStart()), 0, Size+1);
  return SB;
}


//===----------------------------------------------------------------------===//
// MemoryBufferMMapFile implementation.
//===----------------------------------------------------------------------===//

namespace {
class MemoryBufferMMapFile : public MemoryBuffer {
  sys::MappedFile File;
public:
  MemoryBufferMMapFile() {}
  
  bool open(const sys::Path &Filename, std::string *ErrStr);
  
  virtual const char *getBufferIdentifier() const {
    return File.path().c_str();
  }
    
  ~MemoryBufferMMapFile();
};
}

bool MemoryBufferMMapFile::open(const sys::Path &Filename,
                                std::string *ErrStr) {
  // FIXME: This does an extra stat syscall to figure out the size, but we
  // already know the size!
  bool Failure = File.open(Filename, sys::MappedFile::READ_ACCESS, ErrStr);
  if (Failure) return true;
  
  if (!File.map(ErrStr))
    return true;
  
  size_t Size = File.size();
  
  static unsigned PageSize = sys::Process::GetPageSize();
  assert(((PageSize & (PageSize-1)) == 0) && PageSize &&
         "Page size is not a power of 2!");
  
  // If this file is not an exact multiple of the system page size (common
  // case), then the OS has zero terminated the buffer for us.
  if ((Size & (PageSize-1))) {
    init(File.charBase(), File.charBase()+Size);
  } else {
    // Otherwise, we allocate a new memory buffer and copy the data over
    initCopyOf(File.charBase(), File.charBase()+Size);
    
    // No need to keep the file mapped any longer.
    File.unmap();
  }
  return false;
}

MemoryBufferMMapFile::~MemoryBufferMMapFile() {
  if (File.isMapped())
    File.unmap();
}

//===----------------------------------------------------------------------===//
// MemoryBuffer::getFile implementation.
//===----------------------------------------------------------------------===//

MemoryBuffer *MemoryBuffer::getFile(const char *FilenameStart, unsigned FnSize,
                                    std::string *ErrStr, int64_t FileSize){
  // FIXME: it would be nice if PathWithStatus didn't copy the filename into a
  // temporary string. :(
  sys::PathWithStatus P(FilenameStart, FnSize);
#if 1
  MemoryBufferMMapFile *M = new MemoryBufferMMapFile();
  if (!M->open(P, ErrStr))
    return M;
  delete M;
  return 0;
#else
  // FIXME: We need an efficient and portable method to open a file and then use
  // 'read' to copy the bits out.  The unix implementation is below.  This is
  // an important optimization for clients that want to open large numbers of
  // small files (using mmap on everything can easily exhaust address space!).
  
  // If the user didn't specify a filesize, do a stat to find it.
  if (FileSize == -1) {
    const sys::FileStatus *FS = P.getFileStatus();
    if (FS == 0) return 0;  // Error stat'ing file.
   
    FileSize = FS->fileSize;
  }
  
  // If the file is larger than some threshold, use mmap, otherwise use 'read'.
  if (FileSize >= 4096*4) {
    MemoryBufferMMapFile *M = new MemoryBufferMMapFile();
    if (!M->open(P, ErrStr))
      return M;
    delete M;
    return 0;
  }
  
  MemoryBuffer *SB = getNewUninitMemBuffer(FileSize, FilenameStart);
  char *BufPtr = const_cast<char*>(SB->getBufferStart());
  
  int FD = ::open(FilenameStart, O_RDONLY);
  if (FD == -1) {
    delete SB;
    return 0;
  }
  
  unsigned BytesLeft = FileSize;
  while (BytesLeft) {
    ssize_t NumRead = ::read(FD, BufPtr, BytesLeft);
    if (NumRead != -1) {
      BytesLeft -= NumRead;
      BufPtr += NumRead;
    } else if (errno == EINTR) {
      // try again
    } else {
      // error reading.
      close(FD);
      delete SB;
      return 0;
    }
  }
  close(FD);
  
  return SB;
#endif
}


//===----------------------------------------------------------------------===//
// MemoryBuffer::getSTDIN implementation.
//===----------------------------------------------------------------------===//

namespace {
class STDINBufferFile : public MemoryBuffer {
public:
  virtual const char *getBufferIdentifier() const {
    return "<stdin>";
  }
};
}

MemoryBuffer *MemoryBuffer::getSTDIN() {
  char Buffer[4096*4];
  
  std::vector<char> FileData;
  
  // Read in all of the data from stdin, we cannot mmap stdin.
  sys::Program::ChangeStdinToBinary();
  while (size_t ReadBytes = fread(Buffer, 1, 4096*4, stdin))
    FileData.insert(FileData.end(), Buffer, Buffer+ReadBytes);
  
  size_t Size = FileData.size();
  MemoryBuffer *B = new STDINBufferFile();
  B->initCopyOf(&FileData[0], &FileData[Size]);
  return B;
}
