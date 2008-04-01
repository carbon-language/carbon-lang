//===- Unix/MappedFile.cpp - Unix MappedFile Implementation -----*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file provides the generic Unix implementation of the MappedFile concept.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only generic UNIX code that
//===          is guaranteed to work on *all* UNIX variants.
//===----------------------------------------------------------------------===//

#include "Unix.h"
#include "llvm/System/Process.h"

#ifdef HAVE_FCNTL_H
#include <fcntl.h>
#endif

#ifdef HAVE_SYS_MMAN_H
#include <sys/mman.h>
#endif

#ifdef HAVE_SYS_STAT_H
#include <sys/stat.h>
#endif

using namespace llvm;
using namespace sys;

namespace llvm {
  namespace sys {
    struct MappedFileInfo {
      int   FD;
      off_t Size;
    };
  }
}

bool MappedFile::initialize(std::string* ErrMsg) {
  int mode = 0;
  if (Options & READ_ACCESS) 
    if (Options & WRITE_ACCESS)
      mode = O_RDWR;
    else
      mode = O_RDONLY;
  else if (Options & WRITE_ACCESS)
    mode = O_WRONLY;

  int FD = ::open(Path.c_str(), mode);
  if (FD < 0) {
    MakeErrMsg(ErrMsg, "can't open file '" + Path.toString() + "'");
    return true;
  } 
  const FileStatus *Status = Path.getFileStatus(false, ErrMsg);
  if (!Status) {
    ::close(FD);
    return true;
  }
  MapInfo = new MappedFileInfo();
  MapInfo->FD = FD;
  MapInfo->Size = Status->getSize();
  return false;
}

void MappedFile::terminate() {
  assert(MapInfo && "MappedFile not initialized");
  ::close(MapInfo->FD);
  delete MapInfo;
  MapInfo = 0;
}

void MappedFile::unmap() {
  assert(MapInfo && "MappedFile not initialized");
  if (!isMapped()) return;
  
  if (Options & WRITE_ACCESS)
    ::msync(BasePtr, MapInfo->Size, MS_SYNC);
  ::munmap(BasePtr, MapInfo->Size);
  BasePtr = 0;  // Mark this as non-mapped.
}

void* MappedFile::map(std::string* ErrMsg) {
  assert(MapInfo && "MappedFile not initialized");
  if (isMapped()) return BasePtr;
  
  int prot = PROT_NONE;
  int flags = 0;
#ifdef MAP_FILE
  flags |= MAP_FILE;
#endif
  if (Options == 0) {
    prot = PROT_READ;
    flags = MAP_PRIVATE;
  } else {
    if (Options & READ_ACCESS)
      prot |= PROT_READ;
    if (Options & WRITE_ACCESS)
      prot |= PROT_WRITE;
    if (Options & EXEC_ACCESS)
      prot |= PROT_EXEC;
    if (Options & SHARED_MAPPING)
      flags |= MAP_SHARED;
    else
      flags |= MAP_PRIVATE;
  }
  size_t map_size = ((MapInfo->Size / Process::GetPageSize())+1) *
    Process::GetPageSize();

  BasePtr = ::mmap(0, map_size, prot, flags, MapInfo->FD, 0);
  if (BasePtr == MAP_FAILED) {
    MakeErrMsg(ErrMsg, "Can't map file:" + Path.toString());
    return 0;
  }
  return BasePtr;
}

size_t MappedFile::size() const {
  assert(MapInfo && "MappedFile not initialized");
  return MapInfo->Size;
}

bool MappedFile::resize(size_t new_size, std::string* ErrMsg) {
  assert(MapInfo && "MappedFile not initialized");

  // Take the mapping out of memory
  unmap();

  // Adjust the current size to a page boundary
  size_t cur_size = ((MapInfo->Size / Process::GetPageSize())+1) *
    Process::GetPageSize();

  // Adjust the new_size to a page boundary
  new_size = ((new_size / Process::GetPageSize())+1) *
    Process::GetPageSize();

  // If the file needs to be extended
  if (new_size > cur_size) {
    // Ensure we can allocate at least the idodes necessary to handle the
    // file size requested. 
    if ((off_t)-1 == ::lseek(MapInfo->FD, new_size, SEEK_SET))
      return MakeErrMsg(ErrMsg, "Can't lseek: ");
    if (-1 == ::write(MapInfo->FD, "\0", 1))
      return MakeErrMsg(ErrMsg, "Can't write: ");
  }

  // Put the mapping back into memory.
  return map(ErrMsg);
}
