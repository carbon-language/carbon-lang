//===- VirtualFileSystem.h - Virtual File System Layer ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// \brief Defines the virtual file system interface vfs::FileSystem.
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_VIRTUAL_FILE_SYSTEM_H
#define LLVM_CLANG_BASIC_VIRTUAL_FILE_SYSTEM_H

#include "clang/Basic/LLVM.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/SourceMgr.h"

namespace llvm {
class MemoryBuffer;
}

namespace clang {
namespace vfs {

/// \brief The result of a \p status operation.
class Status {
  std::string Name;
  llvm::sys::fs::UniqueID UID;
  llvm::sys::TimeValue MTime;
  uint32_t User;
  uint32_t Group;
  uint64_t Size;
  llvm::sys::fs::file_type Type;
  llvm::sys::fs::perms Perms;

public:
  Status() : Type(llvm::sys::fs::file_type::status_error) {}
  Status(const llvm::sys::fs::file_status &Status);
  Status(StringRef Name, StringRef RealName, llvm::sys::fs::UniqueID UID,
         llvm::sys::TimeValue MTime, uint32_t User, uint32_t Group,
         uint64_t Size, llvm::sys::fs::file_type Type,
         llvm::sys::fs::perms Perms);

  /// \brief Returns the name that should be used for this file or directory.
  StringRef getName() const { return Name; }
  void setName(StringRef N) { Name = N; }

  /// @name Status interface from llvm::sys::fs
  /// @{
  llvm::sys::fs::file_type getType() const { return Type; }
  llvm::sys::fs::perms getPermissions() const { return Perms; }
  llvm::sys::TimeValue getLastModificationTime() const { return MTime; }
  llvm::sys::fs::UniqueID getUniqueID() const { return UID; }
  uint32_t getUser() const { return User; }
  uint32_t getGroup() const { return Group; }
  uint64_t getSize() const { return Size; }
  void setType(llvm::sys::fs::file_type v) { Type = v; }
  void setPermissions(llvm::sys::fs::perms p) { Perms = p; }
  /// @}
  /// @name Status queries
  /// These are static queries in llvm::sys::fs.
  /// @{
  bool equivalent(const Status &Other) const;
  bool isDirectory() const;
  bool isRegularFile() const;
  bool isOther() const;
  bool isSymlink() const;
  bool isStatusKnown() const;
  bool exists() const;
  /// @}
};

/// \brief Represents an open file.
class File {
public:
  /// \brief Destroy the file after closing it (if open).
  /// Sub-classes should generally call close() inside their destructors.  We
  /// cannot do that from the base class, since close is virtual.
  virtual ~File();
  /// \brief Get the status of the file.
  virtual llvm::ErrorOr<Status> status() = 0;
  /// \brief Get the contents of the file as a \p MemoryBuffer.
  virtual llvm::error_code getBuffer(const Twine &Name,
                                     OwningPtr<llvm::MemoryBuffer> &Result,
                                     int64_t FileSize = -1,
                                     bool RequiresNullTerminator = true) = 0;
  /// \brief Closes the file.
  virtual llvm::error_code close() = 0;
  /// \brief Sets the name to use for this file.
  virtual void setName(StringRef Name) = 0;
};

/// \brief The virtual file system interface.
class FileSystem : public RefCountedBase<FileSystem> {
public:
  virtual ~FileSystem();

  /// \brief Get the status of the entry at \p Path, if one exists.
  virtual llvm::ErrorOr<Status> status(const Twine &Path) = 0;
  /// \brief Get a \p File object for the file at \p Path, if one exists.
  virtual llvm::error_code openFileForRead(const Twine &Path,
                                           OwningPtr<File> &Result) = 0;

  /// This is a convenience method that opens a file, gets its content and then
  /// closes the file.
  llvm::error_code getBufferForFile(const Twine &Name,
                                    OwningPtr<llvm::MemoryBuffer> &Result,
                                    int64_t FileSize = -1,
                                    bool RequiresNullTerminator = true);
};

/// \brief Gets an \p vfs::FileSystem for the 'real' file system, as seen by
/// the operating system.
IntrusiveRefCntPtr<FileSystem> getRealFileSystem();

/// \brief A file system that allows overlaying one \p AbstractFileSystem on top
/// of another.
///
/// Consists of a stack of >=1 \p FileSystem objects, which are treated as being
/// one merged file system. When there is a directory that exists in more than
/// one file system, the \p OverlayFileSystem contains a directory containing
/// the union of their contents.  The attributes (permissions, etc.) of the
/// top-most (most recently added) directory are used.  When there is a file
/// that exists in more than one file system, the file in the top-most file
/// system overrides the other(s).
class OverlayFileSystem : public FileSystem {
  typedef SmallVector<IntrusiveRefCntPtr<FileSystem>, 1> FileSystemList;
  typedef FileSystemList::reverse_iterator iterator;

  /// \brief The stack of file systems, implemented as a list in order of
  /// their addition.
  FileSystemList FSList;

  /// \brief Get an iterator pointing to the most recently added file system.
  iterator overlays_begin() { return FSList.rbegin(); }

  /// \brief Get an iterator pointing one-past the least recently added file
  /// system.
  iterator overlays_end() { return FSList.rend(); }

public:
  OverlayFileSystem(IntrusiveRefCntPtr<FileSystem> Base);
  /// \brief Pushes a file system on top of the stack.
  void pushOverlay(IntrusiveRefCntPtr<FileSystem> FS);

  llvm::ErrorOr<Status> status(const Twine &Path) override;
  llvm::error_code openFileForRead(const Twine &Path,
                                   OwningPtr<File> &Result) override;
};

/// \brief Get a globally unique ID for a virtual file or directory.
llvm::sys::fs::UniqueID getNextVirtualUniqueID();

/// \brief Gets a \p FileSystem for a virtual file system described in YAML
/// format.
///
/// Takes ownership of \p Buffer.
IntrusiveRefCntPtr<FileSystem>
getVFSFromYAML(llvm::MemoryBuffer *Buffer,
               llvm::SourceMgr::DiagHandlerTy DiagHandler,
               void *DiagContext = 0,
               IntrusiveRefCntPtr<FileSystem> ExternalFS = getRealFileSystem());

} // end namespace vfs
} // end namespace clang
#endif // LLVM_CLANG_BASIC_VIRTUAL_FILE_SYSTEM_H
