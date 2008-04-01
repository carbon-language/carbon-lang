//===- Unix/MappedFile.inc - Unix MappedFile Implementation -----*- C++ -*-===//
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
  int FD = ::open(Path.c_str(), O_RDONLY);
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
  unmap();
  assert(MapInfo && "MappedFile not initialized");
  ::close(MapInfo->FD);
  delete MapInfo;
  MapInfo = 0;
}

void MappedFile::unmap() {
  assert(MapInfo && "MappedFile not initialized");
  if (!isMapped()) return;
  
  ::munmap(BasePtr, MapInfo->Size);
  BasePtr = 0;  // Mark this as non-mapped.
}

const void* MappedFile::map(std::string* ErrMsg) {
  assert(MapInfo && "MappedFile not initialized");
  if (isMapped()) return BasePtr;
  
  int flags = MAP_PRIVATE;
#ifdef MAP_FILE
  flags |= MAP_FILE;
#endif
  size_t PageSize = Process::GetPageSize();
  size_t map_size = ((MapInfo->Size / PageSize)+1) * PageSize;

  BasePtr = ::mmap(0, map_size, PROT_READ, flags, MapInfo->FD, 0);
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

