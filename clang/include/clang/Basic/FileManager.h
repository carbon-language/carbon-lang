//===--- FileManager.h - File System Probing and Caching --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the FileManager interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FILEMANAGER_H
#define LLVM_CLANG_FILEMANAGER_H

#include "clang/Basic/FileSystemOptions.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Config/config.h" // for mode_t
// FIXME: Enhance libsystem to support inode and other fields in stat.
#include <sys/types.h>

struct stat;

namespace llvm {
class MemoryBuffer;
namespace sys { class Path; }
}

namespace clang {
class FileManager;
class FileSystemStatCache;
  
/// DirectoryEntry - Cached information about one directory on the disk.
///
class DirectoryEntry {
  const char *Name;   // Name of the directory.
  friend class FileManager;
public:
  DirectoryEntry() : Name(0) {}
  const char *getName() const { return Name; }
};

/// FileEntry - Cached information about one file on the disk.  If the 'FD'
/// member is valid, then this FileEntry has an open file descriptor for the
/// file.
///
class FileEntry {
  const char *Name;           // Name of the file.
  off_t Size;                 // File size in bytes.
  time_t ModTime;             // Modification time of file.
  const DirectoryEntry *Dir;  // Directory file lives in.
  unsigned UID;               // A unique (small) ID for the file.
  dev_t Device;               // ID for the device containing the file.
  ino_t Inode;                // Inode number for the file.
  mode_t FileMode;            // The file mode as returned by 'stat'.
  
  /// FD - The file descriptor for the file entry if it is opened and owned
  /// by the FileEntry.  If not, this is set to -1.
  mutable int FD;
  friend class FileManager;
  
public:
  FileEntry(dev_t device, ino_t inode, mode_t m)
    : Name(0), Device(device), Inode(inode), FileMode(m), FD(-1) {}
  // Add a default constructor for use with llvm::StringMap
  FileEntry() : Name(0), Device(0), Inode(0), FileMode(0), FD(-1) {}

  FileEntry(const FileEntry &FE) {
    memcpy(this, &FE, sizeof(FE));
    assert(FD == -1 && "Cannot copy a file-owning FileEntry");
  }
  
  void operator=(const FileEntry &FE) {
    memcpy(this, &FE, sizeof(FE));
    assert(FD == -1 && "Cannot assign a file-owning FileEntry");
  }

  ~FileEntry();

  const char *getName() const { return Name; }
  off_t getSize() const { return Size; }
  unsigned getUID() const { return UID; }
  ino_t getInode() const { return Inode; }
  dev_t getDevice() const { return Device; }
  time_t getModificationTime() const { return ModTime; }
  mode_t getFileMode() const { return FileMode; }

  /// getDir - Return the directory the file lives in.
  ///
  const DirectoryEntry *getDir() const { return Dir; }

  bool operator<(const FileEntry &RHS) const {
    return Device < RHS.Device || (Device == RHS.Device && Inode < RHS.Inode);
  }
};

/// FileManager - Implements support for file system lookup, file system
/// caching, and directory search management.  This also handles more advanced
/// properties, such as uniquing files based on "inode", so that a file with two
/// names (e.g. symlinked) will be treated as a single file.
///
class FileManager {
  FileSystemOptions FileSystemOpts;
  
  class UniqueDirContainer;
  class UniqueFileContainer;

  /// UniqueDirs/UniqueFiles - Cache for existing directories/files.
  ///
  UniqueDirContainer &UniqueDirs;
  UniqueFileContainer &UniqueFiles;

  /// DirEntries/FileEntries - This is a cache of directory/file entries we have
  /// looked up.  The actual Entry is owned by UniqueFiles/UniqueDirs above.
  ///
  llvm::StringMap<DirectoryEntry*, llvm::BumpPtrAllocator> DirEntries;
  llvm::StringMap<FileEntry*, llvm::BumpPtrAllocator> FileEntries;

  /// NextFileUID - Each FileEntry we create is assigned a unique ID #.
  ///
  unsigned NextFileUID;

  /// \brief The virtual files that we have allocated.
  llvm::SmallVector<FileEntry*, 4> VirtualFileEntries;

  // Statistics.
  unsigned NumDirLookups, NumFileLookups;
  unsigned NumDirCacheMisses, NumFileCacheMisses;

  // Caching.
  llvm::OwningPtr<FileSystemStatCache> StatCache;

  bool getStatValue(const char *Path, struct stat &StatBuf,
                    int *FileDescriptor);
public:
  FileManager(const FileSystemOptions &FileSystemOpts);
  ~FileManager();

  /// \brief Installs the provided FileSystemStatCache object within
  /// the FileManager. 
  ///
  /// Ownership of this object is transferred to the FileManager.
  ///
  /// \param statCache the new stat cache to install. Ownership of this
  /// object is transferred to the FileManager.
  ///
  /// \param AtBeginning whether this new stat cache must be installed at the
  /// beginning of the chain of stat caches. Otherwise, it will be added to
  /// the end of the chain.
  void addStatCache(FileSystemStatCache *statCache, bool AtBeginning = false);

  /// \brief Removes the specified FileSystemStatCache object from the manager.
  void removeStatCache(FileSystemStatCache *statCache);
  
  /// getDirectory - Lookup, cache, and verify the specified directory.  This
  /// returns null if the directory doesn't exist.
  ///
  const DirectoryEntry *getDirectory(llvm::StringRef Filename);

  /// getFile - Lookup, cache, and verify the specified file.  This returns null
  /// if the file doesn't exist.
  ///
  const FileEntry *getFile(llvm::StringRef Filename);

  /// \brief Retrieve a file entry for a "virtual" file that acts as
  /// if there were a file with the given name on disk. The file
  /// itself is not accessed.
  const FileEntry *getVirtualFile(llvm::StringRef Filename, off_t Size,
                                  time_t ModificationTime);

  /// \brief Open the specified file as a MemoryBuffer, returning a new
  /// MemoryBuffer if successful, otherwise returning null.
  llvm::MemoryBuffer *getBufferForFile(const FileEntry *Entry,
                                       std::string *ErrorStr = 0);
  llvm::MemoryBuffer *getBufferForFile(llvm::StringRef Filename,
                                       std::string *ErrorStr = 0);

  /// \brief If path is not absolute and FileSystemOptions set the working
  /// directory, the path is modified to be relative to the given
  /// working directory.
  static void FixupRelativePath(llvm::sys::Path &path,
                                const FileSystemOptions &FSOpts);
  
  void PrintStats() const;
};

}  // end namespace clang

#endif
