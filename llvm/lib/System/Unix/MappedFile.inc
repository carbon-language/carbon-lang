//===- Unix/MappedFile.cpp - Unix MappedFile Implementation -----*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
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

namespace llvm {
using namespace sys;

struct sys::MappedFileInfo {
  int fd_;
  struct stat sbuf_;
};

void MappedFile::initialize() {
  if (!path_.exists())
    throw std::string("Can't open file: ") + path_.toString();
  
  int mode = 0;
  if (options_&READ_ACCESS) 
    if (options_&WRITE_ACCESS)
      mode = O_RDWR;
    else
      mode = O_RDONLY;
  else if (options_&WRITE_ACCESS)
    mode = O_WRONLY;

  int FD = ::open(path_.c_str(), mode);
  if (FD < 0)
    ThrowErrno(std::string("Can't open file: ") + path_.toString());

  struct stat sbuf;
  if(::fstat(FD, &sbuf) < 0) {
    ::close(FD);
    ThrowErrno(std::string("Can't stat file: ") + path_.toString());
  }
  info_ = new MappedFileInfo;
  info_->fd_ = FD;
  info_->sbuf_ = sbuf;
}

void MappedFile::terminate() {
  assert(info_ && "MappedFile not initialized");
  if (info_->fd_ >= 0)
    ::close(info_->fd_);
  delete info_;
  info_ = 0;
}

void MappedFile::unmap() {
  assert(info_ && "MappedFile not initialized");
  if (isMapped()) {
    if (options_ & WRITE_ACCESS)
      ::msync(base_, info_->sbuf_.st_size, MS_SYNC);
    ::munmap(base_, info_->sbuf_.st_size);
  }
}

void* MappedFile::map() {
  assert(info_ && "MappedFile not initialized");
  if (!isMapped()) {
    int prot = PROT_NONE;
    int flags = 0;
#ifdef MAP_FILE
    flags |= MAP_FILE;
#endif
    if (options_ == 0) {
      prot = PROT_READ;
      flags = MAP_PRIVATE;
    } else {
      if (options_ & READ_ACCESS)
        prot |= PROT_READ;
      if (options_ & WRITE_ACCESS)
        prot |= PROT_WRITE;
      if (options_ & EXEC_ACCESS)
        prot |= PROT_EXEC;
      if (options_ & SHARED_MAPPING)
        flags |= MAP_SHARED;
      else
        flags |= MAP_PRIVATE;
    }
    size_t map_size = ((info_->sbuf_.st_size / Process::GetPageSize())+1) *
      Process::GetPageSize();

    base_ = ::mmap(0, map_size, prot, flags, info_->fd_, 0);
    if (base_ == MAP_FAILED)
      ThrowErrno(std::string("Can't map file:") + path_.toString());
  }
  return base_;
}

size_t MappedFile::size() const {
  assert(info_ && "MappedFile not initialized");
  return info_->sbuf_.st_size;
}

void MappedFile::size(size_t new_size) {
  assert(info_ && "MappedFile not initialized");

  // Take the mapping out of memory
  this->unmap();

  // Adjust the current size to a page boundary
  size_t cur_size = ((info_->sbuf_.st_size / Process::GetPageSize())+1) *
    Process::GetPageSize();

  // Adjust the new_size to a page boundary
  new_size = ((new_size / Process::GetPageSize())+1) *
    Process::GetPageSize();

  // If the file needs to be extended
  if (new_size > cur_size) {
    // Ensure we can allocate at least the idodes necessary to handle the
    // file size requested. 
    ::lseek(info_->fd_, new_size, SEEK_SET);
    ::write(info_->fd_, "\0", 1);
  }

  // Seek to current end of file. 
  this->map();
}

}

