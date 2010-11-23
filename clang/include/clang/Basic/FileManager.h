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

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Config/config.h" // for mode_t
// FIXME: Enhance libsystem to support inode and other fields in stat.
#include <sys/types.h>
#include <sys/stat.h>

namespace llvm {
class MemoryBuffer;
namespace sys {
class Path;
}
}

namespace clang {
class FileManager;
class FileSystemOptions;

/// DirectoryEntry - Cached information about one directory on the disk.
///
class DirectoryEntry {
  const char *Name;   // Name of the directory.
  friend class FileManager;
public:
  DirectoryEntry() : Name(0) {}
  const char *getName() const { return Name; }
};

/// FileEntry - Cached information about one file on the disk.
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
  friend class FileManager;
public:
  FileEntry(dev_t device, ino_t inode, mode_t m)
    : Name(0), Device(device), Inode(inode), FileMode(m) {}
  // Add a default constructor for use with llvm::StringMap
  FileEntry() : Name(0), Device(0), Inode(0), FileMode(0) {}

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

  bool operator<(const FileEntry& RHS) const {
    return Device < RHS.Device || (Device == RHS.Device && Inode < RHS.Inode);
  }
};

/// \brief Abstract interface for introducing a FileManager cache for 'stat'
/// system calls, which is used by precompiled and pretokenized headers to
/// improve performance.
class StatSysCallCache {
protected:
  llvm::OwningPtr<StatSysCallCache> NextStatCache;
  
public:
  virtual ~StatSysCallCache() {}
  virtual int stat(const char *path, struct stat *buf) {
    if (getNextStatCache())
      return getNextStatCache()->stat(path, buf);
    
    return ::stat(path, buf);
  }
  
  /// \brief Sets the next stat call cache in the chain of stat caches.
  /// Takes ownership of the given stat cache.
  void setNextStatCache(StatSysCallCache *Cache) {
    NextStatCache.reset(Cache);
  }
  
  /// \brief Retrieve the next stat call cache in the chain.
  StatSysCallCache *getNextStatCache() { return NextStatCache.get(); }

  /// \brief Retrieve the next stat call cache in the chain, transferring
  /// ownership of this cache (and, transitively, all of the remaining caches)
  /// to the caller.
  StatSysCallCache *takeNextStatCache() { return NextStatCache.take(); }
};

/// \brief A stat "cache" that can be used by FileManager to keep
/// track of the results of stat() calls that occur throughout the
/// execution of the front end.
class MemorizeStatCalls : public StatSysCallCache {
public:
  /// \brief The result of a stat() call.
  ///
  /// The first member is the result of calling stat(). If stat()
  /// found something, the second member is a copy of the stat
  /// structure.
  typedef std::pair<int, struct stat> StatResult;

  /// \brief The set of stat() calls that have been
  llvm::StringMap<StatResult, llvm::BumpPtrAllocator> StatCalls;

  typedef llvm::StringMap<StatResult, llvm::BumpPtrAllocator>::const_iterator
    iterator;

  iterator begin() const { return StatCalls.begin(); }
  iterator end() const { return StatCalls.end(); }

  virtual int stat(const char *path, struct stat *buf);
};

/// FileManager - Implements support for file system lookup, file system
/// caching, and directory search management.  This also handles more advanced
/// properties, such as uniquing files based on "inode", so that a file with two
/// names (e.g. symlinked) will be treated as a single file.
///
class FileManager {
  const FileSystemOptions &FileSystemOpts;
  
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
  llvm::SmallVector<FileEntry *, 4> VirtualFileEntries;

  // Statistics.
  unsigned NumDirLookups, NumFileLookups;
  unsigned NumDirCacheMisses, NumFileCacheMisses;

  // Caching.
  llvm::OwningPtr<StatSysCallCache> StatCache;

  int stat_cached(const char* path, struct stat* buf);

public:
  FileManager(const FileSystemOptions &FileSystemOpts);
  ~FileManager();

  /// \brief Installs the provided StatSysCallCache object within
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
  void addStatCache(StatSysCallCache *statCache, bool AtBeginning = false);

  /// \brief Removes the provided StatSysCallCache object from the file manager.
  void removeStatCache(StatSysCallCache *statCache);
  
  /// getDirectory - Lookup, cache, and verify the specified directory.  This
  /// returns null if the directory doesn't exist.
  ///
  const DirectoryEntry *getDirectory(llvm::StringRef Filename,
                                     const FileSystemOptions &FileSystemOpts);

  /// getFile - Lookup, cache, and verify the specified file.  This returns null
  /// if the file doesn't exist.
  ///
  const FileEntry *getFile(llvm::StringRef Filename,
                           const FileSystemOptions &FileSystemOpts);

  /// \brief Retrieve a file entry for a "virtual" file that acts as
  /// if there were a file with the given name on disk. The file
  /// itself is not accessed.
  const FileEntry *getVirtualFile(llvm::StringRef Filename, off_t Size,
                                  time_t ModificationTime,
                                  const FileSystemOptions &FileSystemOpts);

  /// \brief Open the specified file as a MemoryBuffer, returning a new
  /// MemoryBuffer if successful, otherwise returning null.
  llvm::MemoryBuffer *getBufferForFile(const FileEntry *Entry,
                                       const FileSystemOptions &FileSystemOpts,
                                       std::string *ErrorStr = 0) {
    return getBufferForFile(Entry->getName(), FileSystemOpts,
                            ErrorStr, Entry->getSize());
  }
  llvm::MemoryBuffer *getBufferForFile(llvm::StringRef Filename,
                                       const FileSystemOptions &FileSystemOpts,
                                       std::string *ErrorStr = 0,
                                       int64_t FileSize = -1);

  /// \brief If path is not absolute and FileSystemOptions set the working
  /// directory, the path is modified to be relative to the given
  /// working directory.
  static void FixupRelativePath(llvm::sys::Path &path,
                                const FileSystemOptions &FSOpts);
  
  void PrintStats() const;
};

}  // end namespace clang

#endif
