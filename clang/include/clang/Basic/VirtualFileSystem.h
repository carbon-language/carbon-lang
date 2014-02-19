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

#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ErrorOr.h"

namespace llvm {
template <typename T> class OwningPtr;
class MemoryBuffer;
}

namespace clang {
namespace vfs {

/// \brief The result of a \p status operation.
class Status {
  std::string Name;
  std::string ExternalName;
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
  Status(llvm::StringRef Name, llvm::StringRef RealName,
         llvm::sys::fs::UniqueID UID, llvm::sys::TimeValue MTime,
         uint32_t User, uint32_t Group, uint64_t Size,
         llvm::sys::fs::file_type Type, llvm::sys::fs::perms Perms);

  /// \brief Returns the name this status was looked up by.
  llvm::StringRef getName() const { return Name; }

  /// \brief Returns the name to use outside the compiler.
  ///
  /// For example, in diagnostics or debug info we should use this name.
  llvm::StringRef getExternalName() const { return ExternalName; }

  void setName(llvm::StringRef N) { Name = N; }
  void setExternalName(llvm::StringRef N) { ExternalName = N; }

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
  virtual llvm::error_code
  getBuffer(const llvm::Twine &Name,
            llvm::OwningPtr<llvm::MemoryBuffer> &Result, int64_t FileSize = -1,
            bool RequiresNullTerminator = true) = 0;
  /// \brief Closes the file.
  virtual llvm::error_code close() = 0;
};

/// \brief The virtual file system interface.
class FileSystem : public llvm::RefCountedBase<FileSystem> {
public:
  virtual ~FileSystem();

  /// \brief Get the status of the entry at \p Path, if one exists.
  virtual llvm::ErrorOr<Status> status(const llvm::Twine &Path) = 0;
  /// \brief Get a \p File object for the file at \p Path, if one exists.
  virtual llvm::error_code openFileForRead(const llvm::Twine &Path,
                                           llvm::OwningPtr<File> &Result) = 0;

  /// This is a convenience method that opens a file, gets its content and then
  /// closes the file.
  llvm::error_code getBufferForFile(const llvm::Twine &Name,
                                    llvm::OwningPtr<llvm::MemoryBuffer> &Result,
                                    int64_t FileSize = -1,
                                    bool RequiresNullTerminator = true);
};

/// \brief Gets an \p vfs::FileSystem for the 'real' file system, as seen by
/// the operating system.
llvm::IntrusiveRefCntPtr<FileSystem> getRealFileSystem();

} // end namespace vfs
} // end namespace clang
#endif // LLVM_CLANG_BASIC_VIRTUAL_FILE_SYSTEM_H
