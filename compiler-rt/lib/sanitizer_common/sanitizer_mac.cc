//===-- sanitizer_mac.cc --------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is shared between AddressSanitizer and ThreadSanitizer
// run-time libraries and implements mac-specific functions from
// sanitizer_libc.h.
//===----------------------------------------------------------------------===//

#ifdef __APPLE__
// Use 64-bit inodes in file operations. ASan does not support OS X 10.5, so
// the clients will most certainly use 64-bit ones as well.
#ifndef _DARWIN_USE_64_BIT_INODE
#define _DARWIN_USE_64_BIT_INODE 1
#endif
#include <stdio.h>

#include "sanitizer_common.h"
#include "sanitizer_internal_defs.h"
#include "sanitizer_libc.h"
#include "sanitizer_procmaps.h"

#include <crt_externs.h>  // for _NSGetEnviron
#include <fcntl.h>
#include <mach-o/dyld.h>
#include <mach-o/loader.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <libkern/OSAtomic.h>

namespace __sanitizer {

// ---------------------- sanitizer_libc.h
void *internal_mmap(void *addr, size_t length, int prot, int flags,
                    int fd, u64 offset) {
  return mmap(addr, length, prot, flags, fd, offset);
}

int internal_munmap(void *addr, uptr length) {
  return munmap(addr, length);
}

int internal_close(fd_t fd) {
  return close(fd);
}

fd_t internal_open(const char *filename, int flags) {
  return open(filename, flags);
}

fd_t internal_open(const char *filename, int flags, u32 mode) {
  return open(filename, flags, mode);
}

fd_t OpenFile(const char *filename, bool write) {
  return internal_open(filename,
      write ? O_WRONLY | O_CREAT : O_RDONLY, 0660);
}

uptr internal_read(fd_t fd, void *buf, uptr count) {
  return read(fd, buf, count);
}

uptr internal_write(fd_t fd, const void *buf, uptr count) {
  return write(fd, buf, count);
}

int internal_stat(const char *path, void *buf) {
  return stat(path, (struct stat *)buf);
}

int internal_lstat(const char *path, void *buf) {
  return lstat(path, (struct stat *)buf);
}

int internal_fstat(fd_t fd, void *buf) {
  return fstat(fd, (struct stat *)buf);
}

uptr internal_filesize(fd_t fd) {
  struct stat st;
  if (internal_fstat(fd, &st))
    return -1;
  return (uptr)st.st_size;
}

int internal_dup2(int oldfd, int newfd) {
  return dup2(oldfd, newfd);
}

uptr internal_readlink(const char *path, char *buf, uptr bufsize) {
  return readlink(path, buf, bufsize);
}

int internal_sched_yield() {
  return sched_yield();
}

void internal__exit(int exitcode) {
  _exit(exitcode);
}

// ----------------- sanitizer_common.h
bool FileExists(const char *filename) {
  struct stat st;
  if (stat(filename, &st))
    return false;
  // Sanity check: filename is a regular file.
  return S_ISREG(st.st_mode);
}

uptr GetTid() {
  return reinterpret_cast<uptr>(pthread_self());
}

void GetThreadStackTopAndBottom(bool at_initialization, uptr *stack_top,
                                uptr *stack_bottom) {
  CHECK(stack_top);
  CHECK(stack_bottom);
  uptr stacksize = pthread_get_stacksize_np(pthread_self());
  void *stackaddr = pthread_get_stackaddr_np(pthread_self());
  *stack_top = (uptr)stackaddr;
  *stack_bottom = *stack_top - stacksize;
}

const char *GetEnv(const char *name) {
  char ***env_ptr = _NSGetEnviron();
  CHECK(env_ptr);
  char **environ = *env_ptr;
  CHECK(environ);
  uptr name_len = internal_strlen(name);
  while (*environ != 0) {
    uptr len = internal_strlen(*environ);
    if (len > name_len) {
      const char *p = *environ;
      if (!internal_memcmp(p, name, name_len) &&
          p[name_len] == '=') {  // Match.
        return *environ + name_len + 1;  // String starting after =.
      }
    }
    environ++;
  }
  return 0;
}

void ReExec() {
  UNIMPLEMENTED();
}

void PrepareForSandboxing() {
  // Nothing here for now.
}

// ----------------- sanitizer_procmaps.h

MemoryMappingLayout::MemoryMappingLayout() {
  Reset();
}

MemoryMappingLayout::~MemoryMappingLayout() {
}

// More information about Mach-O headers can be found in mach-o/loader.h
// Each Mach-O image has a header (mach_header or mach_header_64) starting with
// a magic number, and a list of linker load commands directly following the
// header.
// A load command is at least two 32-bit words: the command type and the
// command size in bytes. We're interested only in segment load commands
// (LC_SEGMENT and LC_SEGMENT_64), which tell that a part of the file is mapped
// into the task's address space.
// The |vmaddr|, |vmsize| and |fileoff| fields of segment_command or
// segment_command_64 correspond to the memory address, memory size and the
// file offset of the current memory segment.
// Because these fields are taken from the images as is, one needs to add
// _dyld_get_image_vmaddr_slide() to get the actual addresses at runtime.

void MemoryMappingLayout::Reset() {
  // Count down from the top.
  // TODO(glider): as per man 3 dyld, iterating over the headers with
  // _dyld_image_count is thread-unsafe. We need to register callbacks for
  // adding and removing images which will invalidate the MemoryMappingLayout
  // state.
  current_image_ = _dyld_image_count();
  current_load_cmd_count_ = -1;
  current_load_cmd_addr_ = 0;
  current_magic_ = 0;
  current_filetype_ = 0;
}

// static
void MemoryMappingLayout::CacheMemoryMappings() {
  // No-op on Mac for now.
}

void MemoryMappingLayout::LoadFromCache() {
  // No-op on Mac for now.
}

// Next and NextSegmentLoad were inspired by base/sysinfo.cc in
// Google Perftools, http://code.google.com/p/google-perftools.

// NextSegmentLoad scans the current image for the next segment load command
// and returns the start and end addresses and file offset of the corresponding
// segment.
// Note that the segment addresses are not necessarily sorted.
template<u32 kLCSegment, typename SegmentCommand>
bool MemoryMappingLayout::NextSegmentLoad(
    uptr *start, uptr *end, uptr *offset,
    char filename[], uptr filename_size, uptr *protection) {
  if (protection)
    UNIMPLEMENTED();
  const char* lc = current_load_cmd_addr_;
  current_load_cmd_addr_ += ((const load_command *)lc)->cmdsize;
  if (((const load_command *)lc)->cmd == kLCSegment) {
    const sptr dlloff = _dyld_get_image_vmaddr_slide(current_image_);
    const SegmentCommand* sc = (const SegmentCommand *)lc;
    if (start) *start = sc->vmaddr + dlloff;
    if (end) *end = sc->vmaddr + sc->vmsize + dlloff;
    if (offset) {
      if (current_filetype_ == /*MH_EXECUTE*/ 0x2) {
        *offset = sc->vmaddr;
      } else {
        *offset = sc->fileoff;
      }
    }
    if (filename) {
      internal_strncpy(filename, _dyld_get_image_name(current_image_),
                       filename_size);
    }
    return true;
  }
  return false;
}

bool MemoryMappingLayout::Next(uptr *start, uptr *end, uptr *offset,
                               char filename[], uptr filename_size,
                               uptr *protection) {
  for (; current_image_ >= 0; current_image_--) {
    const mach_header* hdr = _dyld_get_image_header(current_image_);
    if (!hdr) continue;
    if (current_load_cmd_count_ < 0) {
      // Set up for this image;
      current_load_cmd_count_ = hdr->ncmds;
      current_magic_ = hdr->magic;
      current_filetype_ = hdr->filetype;
      switch (current_magic_) {
#ifdef MH_MAGIC_64
        case MH_MAGIC_64: {
          current_load_cmd_addr_ = (char*)hdr + sizeof(mach_header_64);
          break;
        }
#endif
        case MH_MAGIC: {
          current_load_cmd_addr_ = (char*)hdr + sizeof(mach_header);
          break;
        }
        default: {
          continue;
        }
      }
    }

    for (; current_load_cmd_count_ >= 0; current_load_cmd_count_--) {
      switch (current_magic_) {
        // current_magic_ may be only one of MH_MAGIC, MH_MAGIC_64.
#ifdef MH_MAGIC_64
        case MH_MAGIC_64: {
          if (NextSegmentLoad<LC_SEGMENT_64, struct segment_command_64>(
                  start, end, offset, filename, filename_size, protection))
            return true;
          break;
        }
#endif
        case MH_MAGIC: {
          if (NextSegmentLoad<LC_SEGMENT, struct segment_command>(
                  start, end, offset, filename, filename_size, protection))
            return true;
          break;
        }
      }
    }
    // If we get here, no more load_cmd's in this image talk about
    // segments.  Go on to the next image.
  }
  return false;
}

bool MemoryMappingLayout::GetObjectNameAndOffset(uptr addr, uptr *offset,
                                                 char filename[],
                                                 uptr filename_size,
                                                 uptr *protection) {
  return IterateForObjectNameAndOffset(addr, offset, filename, filename_size,
                                       protection);
}

BlockingMutex::BlockingMutex(LinkerInitialized) {
  // We assume that OS_SPINLOCK_INIT is zero
}

void BlockingMutex::Lock() {
  CHECK(sizeof(OSSpinLock) <= sizeof(opaque_storage_));
  CHECK_EQ(OS_SPINLOCK_INIT, 0);
  CHECK_NE(owner_, (uptr)pthread_self());
  OSSpinLockLock((OSSpinLock*)&opaque_storage_);
  CHECK(!owner_);
  owner_ = (uptr)pthread_self();
}

void BlockingMutex::Unlock() {
  CHECK(owner_ == (uptr)pthread_self());
  owner_ = 0;
  OSSpinLockUnlock((OSSpinLock*)&opaque_storage_);
}

void BlockingMutex::CheckLocked() {
  CHECK_EQ((uptr)pthread_self(), owner_);
}

uptr GetTlsSize() {
  return 0;
}

void InitTlsSize() {
}

}  // namespace __sanitizer

#endif  // __APPLE__
