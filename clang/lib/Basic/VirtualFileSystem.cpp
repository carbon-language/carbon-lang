//===- VirtualFileSystem.cpp - Virtual File System Layer --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This file implements the VirtualFileSystem interface.
//===----------------------------------------------------------------------===//

#include "clang/Basic/VirtualFileSystem.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/Path.h"

using namespace clang;
using namespace clang::vfs;
using namespace llvm;
using llvm::sys::fs::file_status;
using llvm::sys::fs::file_type;
using llvm::sys::fs::perms;
using llvm::sys::fs::UniqueID;

Status::Status(const file_status &Status)
    : UID(Status.getUniqueID()), MTime(Status.getLastModificationTime()),
      User(Status.getUser()), Group(Status.getGroup()), Size(Status.getSize()),
      Type(Status.type()), Perms(Status.permissions()) {}

Status::Status(StringRef Name, StringRef ExternalName,
               UniqueID UID, sys::TimeValue MTime,
               uint32_t User, uint32_t Group, uint64_t Size,
               file_type Type, perms Perms)
    : Name(Name), ExternalName(ExternalName), UID(UID), MTime(MTime),
      User(User), Group(Group), Size(Size), Type(Type), Perms(Perms) {}

bool Status::equivalent(const Status &Other) const {
  return getUniqueID() == Other.getUniqueID();
}
bool Status::isDirectory() const {
  return Type == file_type::directory_file;
}
bool Status::isRegularFile() const {
  return Type == file_type::regular_file;
}
bool Status::isOther() const {
  return Type == exists() && !isRegularFile() && !isDirectory() && !isSymlink();
}
bool Status::isSymlink() const {
  return Type == file_type::symlink_file;
}
bool Status::isStatusKnown() const {
  return Type != file_type::status_error;
}
bool Status::exists() const {
  return isStatusKnown() && Type != file_type::file_not_found;
}

File::~File() {}

FileSystem::~FileSystem() {}

error_code FileSystem::getBufferForFile(const llvm::Twine &Name,
                                        OwningPtr<MemoryBuffer> &Result,
                                        int64_t FileSize,
                                        bool RequiresNullTerminator) {
  llvm::OwningPtr<File> F;
  if (error_code EC = openFileForRead(Name, F))
    return EC;

  error_code EC = F->getBuffer(Name, Result, FileSize, RequiresNullTerminator);
  return EC;
}

//===-----------------------------------------------------------------------===/
// RealFileSystem implementation
//===-----------------------------------------------------------------------===/

/// \brief Wrapper around a raw file descriptor.
class RealFile : public File {
  int FD;
  friend class RealFileSystem;
  RealFile(int FD) : FD(FD) {
    assert(FD >= 0 && "Invalid or inactive file descriptor");
  }
public:
  ~RealFile();
  ErrorOr<Status> status() LLVM_OVERRIDE;
  error_code getBuffer(const Twine &Name, OwningPtr<MemoryBuffer> &Result,
                       int64_t FileSize = -1,
                       bool RequiresNullTerminator = true) LLVM_OVERRIDE;
  error_code close() LLVM_OVERRIDE;
};
RealFile::~RealFile() {
  close();
}

ErrorOr<Status> RealFile::status() {
  assert(FD != -1 && "cannot stat closed file");
  file_status RealStatus;
  if (error_code EC = sys::fs::status(FD, RealStatus))
    return EC;
  return Status(RealStatus);
}

error_code RealFile::getBuffer(const Twine &Name,
                               OwningPtr<MemoryBuffer> &Result,
                               int64_t FileSize, bool RequiresNullTerminator) {
  assert(FD != -1 && "cannot get buffer for closed file");
  return MemoryBuffer::getOpenFile(FD, Name.str().c_str(), Result, FileSize,
                                   RequiresNullTerminator);
}

// FIXME: This is terrible, we need this for ::close.
#if !defined(_MSC_VER) && !defined(__MINGW32__)
#include <unistd.h>
#include <sys/uio.h>
#else
#include <io.h>
#ifndef S_ISFIFO
#define S_ISFIFO(x) (0)
#endif
#endif
error_code RealFile::close() {
  if (::close(FD))
    return error_code(errno, system_category());
  FD = -1;
  return error_code::success();
}

/// \brief The file system according to your operating system.
class RealFileSystem : public FileSystem {
public:
  ErrorOr<Status> status(const Twine &Path) LLVM_OVERRIDE;
  error_code openFileForRead(const Twine &Path, OwningPtr<File> &Result)
    LLVM_OVERRIDE;
};

ErrorOr<Status> RealFileSystem::status(const Twine &Path) {
  sys::fs::file_status RealStatus;
  if (error_code EC = sys::fs::status(Path, RealStatus))
    return EC;
  Status Result(RealStatus);
  Result.setName(Path.str());
  Result.setExternalName(Path.str());
  return Result;
}

error_code RealFileSystem::openFileForRead(const Twine &Name,
                                           OwningPtr<File> &Result) {
  int FD;
  if (error_code EC = sys::fs::openFileForRead(Name, FD))
    return EC;
  Result.reset(new RealFile(FD));
  return error_code::success();
}

IntrusiveRefCntPtr<FileSystem> vfs::getRealFileSystem() {
  static IntrusiveRefCntPtr<FileSystem> FS = new RealFileSystem();
  return FS;
}

//===-----------------------------------------------------------------------===/
// OverlayFileSystem implementation
//===-----------------------------------------------------------------------===/
OverlayFileSystem::OverlayFileSystem(
    IntrusiveRefCntPtr<FileSystem> BaseFS) {
  pushOverlay(BaseFS);
}

void OverlayFileSystem::pushOverlay(IntrusiveRefCntPtr<FileSystem> FS) {
  FSList.push_back(FS);
}

ErrorOr<Status> OverlayFileSystem::status(const Twine &Path) {
  // FIXME: handle symlinks that cross file systems
  for (iterator I = overlays_begin(), E = overlays_end(); I != E; ++I) {
    ErrorOr<Status> Status = (*I)->status(Path);
    if (Status || Status.getError() != errc::no_such_file_or_directory)
      return Status;
  }
  return error_code(errc::no_such_file_or_directory, system_category());
}

error_code OverlayFileSystem::openFileForRead(const llvm::Twine &Path,
                                              OwningPtr<File> &Result) {
  // FIXME: handle symlinks that cross file systems
  for (iterator I = overlays_begin(), E = overlays_end(); I != E; ++I) {
    error_code EC = (*I)->openFileForRead(Path, Result);
    if (!EC || EC != errc::no_such_file_or_directory)
      return EC;
  }
  return error_code(errc::no_such_file_or_directory, system_category());
}
