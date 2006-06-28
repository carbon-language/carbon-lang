//===--- SourceBuffer.cpp - C Language Family Source Buffer Impl. ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the SourceBuffer interface.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/SourceBuffer.h"
#include "clang/Basic/FileManager.h"
#include "llvm/System/MappedFile.h"
#include "llvm/System/Process.h"
#include <cstdio>
#include <cstring>
using namespace llvm;
using namespace clang;

//===----------------------------------------------------------------------===//
// SourceBuffer implementation itself.
//===----------------------------------------------------------------------===//

SourceBuffer::~SourceBuffer() {
  if (MustDeleteBuffer)
    delete [] BufferStart;
}

/// initCopyOf - Initialize this source buffer with a copy of the specified
/// memory range.  We make the copy so that we can null terminate it
/// successfully.
void SourceBuffer::initCopyOf(const char *BufStart, const char *BufEnd) {
  size_t Size = BufEnd-BufStart;
  BufferStart = new char[Size+1];
  BufferEnd = BufferStart+Size;
  memcpy(const_cast<char*>(BufferStart), BufStart, Size);
  *const_cast<char*>(BufferEnd) = 0;   // Null terminate buffer.
  MustDeleteBuffer = false;
}

/// init - Initialize this SourceBuffer as a reference to externally allocated
/// memory, memory that we know is already null terminated.
void SourceBuffer::init(const char *BufStart, const char *BufEnd) {
  assert(BufEnd[0] == 0 && "Buffer is not null terminated!");
  BufferStart = BufStart;
  BufferEnd = BufEnd;
  MustDeleteBuffer = false;
}

//===----------------------------------------------------------------------===//
// SourceBufferMem implementation.
//===----------------------------------------------------------------------===//

namespace {
class SourceBufferMem : public SourceBuffer {
  std::string FileID;
public:
  SourceBufferMem(const char *Start, const char *End, const char *FID)
  : FileID(FID) {
    init(Start, End);
  }
  
  virtual const char *getBufferIdentifier() const {
    return FileID.c_str();
  }
};
}

/// getMemBuffer - Open the specified memory range as a SourceBuffer.  Note
/// that EndPtr[0] must be a null byte and be accessible!
SourceBuffer *SourceBuffer::getMemBuffer(const char *StartPtr, 
                                         const char *EndPtr,
                                         const char *BufferName) {
  return new SourceBufferMem(StartPtr, EndPtr, BufferName);
}

/// getNewMemBuffer - Allocate a new SourceBuffer of the specified size that
/// is completely initialized to zeros.  Note that the caller should
/// initialize the memory allocated by this method.  The memory is owned by
/// the SourceBuffer object.
SourceBuffer *SourceBuffer::getNewMemBuffer(unsigned Size,
                                            const char *BufferName) {
  char *Buf = new char[Size+1];
  memset(Buf, 0, Size+1);
  SourceBufferMem *SB = new SourceBufferMem(Buf, Buf+Size, BufferName);
  // The memory for this buffer is owned by the SourceBuffer.
  SB->MustDeleteBuffer = true;
  return SB;
}


//===----------------------------------------------------------------------===//
// SourceBufferFile implementation.
//===----------------------------------------------------------------------===//

namespace {
class SourceBufferFile : public SourceBuffer {
  sys::MappedFile File;
public:
  SourceBufferFile(const sys::Path &Filename);
  
  virtual const char *getBufferIdentifier() const {
    return File.path().c_str();
  }
    
  ~SourceBufferFile();
};
}

SourceBufferFile::SourceBufferFile(const sys::Path &Filename) : File(Filename) {
  // FIXME: This does an extra stat syscall to figure out the size, but we
  // already know the size!
  File.map();
  
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
}

SourceBufferFile::~SourceBufferFile() {
  File.unmap();
}


SourceBuffer *SourceBuffer::getFile(const FileEntry *FileEnt) {
  try {
    return new SourceBufferFile(sys::Path(FileEnt->getName()));
  } catch (...) {
    return 0;
  }
}

//===----------------------------------------------------------------------===//
// SourceBufferSTDIN implementation.
//===----------------------------------------------------------------------===//

namespace {
class STDINBufferFile : public SourceBuffer {
public:
  virtual const char *getBufferIdentifier() const {
    return "<stdin>";
  }
};
}

SourceBuffer *SourceBuffer::getSTDIN() {
  char Buffer[4096*4];
  
  std::vector<char> FileData;
  
  // Read in all of the data from stdin, we cannot mmap stdin.
  while (size_t ReadBytes = fread(Buffer, 1, 4096*4, stdin))
    FileData.insert(FileData.end(), Buffer, Buffer+ReadBytes);
  
  size_t Size = FileData.size();
  SourceBuffer *B = new STDINBufferFile();
  B->initCopyOf(&FileData[0], &FileData[Size]);
  return B;
}
