//===--- FileManager.cpp - File System Probing and Caching ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the FileManager interface.
//
//===----------------------------------------------------------------------===//
//
// TODO: This should index all interesting directories with dirent calls.
//  getdirentries ?
//  opendir/readdir_r/closedir ?
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/FileManager.h"
#include "clang/Basic/FileSystemStatCache.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/System/Path.h"
#include "llvm/Config/config.h"
#include <map>
#include <set>
#include <string>
using namespace clang;

// FIXME: Enhance libsystem to support inode and other fields.
#include <sys/stat.h>

#if defined(_MSC_VER)
#define S_ISDIR(s) (_S_IFDIR & s)
#endif

/// NON_EXISTENT_DIR - A special value distinct from null that is used to
/// represent a dir name that doesn't exist on the disk.
#define NON_EXISTENT_DIR reinterpret_cast<DirectoryEntry*>((intptr_t)-1)

/// NON_EXISTENT_FILE - A special value distinct from null that is used to
/// represent a filename that doesn't exist on the disk.
#define NON_EXISTENT_FILE reinterpret_cast<FileEntry*>((intptr_t)-1)


FileEntry::~FileEntry() {
  // If this FileEntry owns an open file descriptor that never got used, close
  // it.
  if (FD != -1) ::close(FD);
}

//===----------------------------------------------------------------------===//
// Windows.
//===----------------------------------------------------------------------===//

#ifdef LLVM_ON_WIN32

#define IS_DIR_SEPARATOR_CHAR(x) ((x) == '/' || (x) == '\\')

namespace {
  static std::string GetFullPath(const char *relPath) {
    char *absPathStrPtr = _fullpath(NULL, relPath, 0);
    assert(absPathStrPtr && "_fullpath() returned NULL!");

    std::string absPath(absPathStrPtr);

    free(absPathStrPtr);
    return absPath;
  }
}

class FileManager::UniqueDirContainer {
  /// UniqueDirs - Cache from full path to existing directories/files.
  ///
  llvm::StringMap<DirectoryEntry> UniqueDirs;

public:
  DirectoryEntry &getDirectory(const char *Name, struct stat &StatBuf) {
    std::string FullPath(GetFullPath(Name));
    return UniqueDirs.GetOrCreateValue(FullPath).getValue();
  }

  size_t size() const { return UniqueDirs.size(); }
};

class FileManager::UniqueFileContainer {
  /// UniqueFiles - Cache from full path to existing directories/files.
  ///
  llvm::StringMap<FileEntry, llvm::BumpPtrAllocator> UniqueFiles;

public:
  FileEntry &getFile(const char *Name, struct stat &StatBuf) {
    std::string FullPath(GetFullPath(Name));
    
    // LowercaseString because Windows filesystem is case insensitive.
    FullPath = llvm::LowercaseString(FullPath);
    return UniqueFiles.GetOrCreateValue(FullPath).getValue();
  }

  size_t size() const { return UniqueFiles.size(); }
};

//===----------------------------------------------------------------------===//
// Unix-like Systems.
//===----------------------------------------------------------------------===//

#else

#define IS_DIR_SEPARATOR_CHAR(x) ((x) == '/')

class FileManager::UniqueDirContainer {
  /// UniqueDirs - Cache from ID's to existing directories/files.
  std::map<std::pair<dev_t, ino_t>, DirectoryEntry> UniqueDirs;

public:
  DirectoryEntry &getDirectory(const char *Name, struct stat &StatBuf) {
    return UniqueDirs[std::make_pair(StatBuf.st_dev, StatBuf.st_ino)];
  }

  size_t size() const { return UniqueDirs.size(); }
};

class FileManager::UniqueFileContainer {
  /// UniqueFiles - Cache from ID's to existing directories/files.
  std::set<FileEntry> UniqueFiles;

public:
  FileEntry &getFile(const char *Name, struct stat &StatBuf) {
    return
      const_cast<FileEntry&>(
                    *UniqueFiles.insert(FileEntry(StatBuf.st_dev,
                                                  StatBuf.st_ino,
                                                  StatBuf.st_mode)).first);
  }

  size_t size() const { return UniqueFiles.size(); }
};

#endif

//===----------------------------------------------------------------------===//
// Common logic.
//===----------------------------------------------------------------------===//

FileManager::FileManager(const FileSystemOptions &FSO)
  : FileSystemOpts(FSO),
    UniqueDirs(*new UniqueDirContainer()),
    UniqueFiles(*new UniqueFileContainer()),
    DirEntries(64), FileEntries(64), NextFileUID(0) {
  NumDirLookups = NumFileLookups = 0;
  NumDirCacheMisses = NumFileCacheMisses = 0;
}

FileManager::~FileManager() {
  delete &UniqueDirs;
  delete &UniqueFiles;
  for (unsigned i = 0, e = VirtualFileEntries.size(); i != e; ++i)
    delete VirtualFileEntries[i];
}

void FileManager::addStatCache(FileSystemStatCache *statCache,
                               bool AtBeginning) {
  assert(statCache && "No stat cache provided?");
  if (AtBeginning || StatCache.get() == 0) {
    statCache->setNextStatCache(StatCache.take());
    StatCache.reset(statCache);
    return;
  }
  
  FileSystemStatCache *LastCache = StatCache.get();
  while (LastCache->getNextStatCache())
    LastCache = LastCache->getNextStatCache();
  
  LastCache->setNextStatCache(statCache);
}

void FileManager::removeStatCache(FileSystemStatCache *statCache) {
  if (!statCache)
    return;
  
  if (StatCache.get() == statCache) {
    // This is the first stat cache.
    StatCache.reset(StatCache->takeNextStatCache());
    return;
  }
  
  // Find the stat cache in the list.
  FileSystemStatCache *PrevCache = StatCache.get();
  while (PrevCache && PrevCache->getNextStatCache() != statCache)
    PrevCache = PrevCache->getNextStatCache();
  
  assert(PrevCache && "Stat cache not found for removal");
  PrevCache->setNextStatCache(statCache->getNextStatCache());
}

/// \brief Retrieve the directory that the given file name resides in.
static const DirectoryEntry *getDirectoryFromFile(FileManager &FileMgr,
                                                  llvm::StringRef Filename) {
  // Figure out what directory it is in.   If the string contains a / in it,
  // strip off everything after it.
  // FIXME: this logic should be in sys::Path.
  size_t SlashPos = Filename.size();
  while (SlashPos != 0 && !IS_DIR_SEPARATOR_CHAR(Filename[SlashPos-1]))
    --SlashPos;

  // Use the current directory if file has no path component.
  if (SlashPos == 0)
    return FileMgr.getDirectory(".");

  if (SlashPos == Filename.size()-1)
    return 0;       // If filename ends with a /, it's a directory.

  // Ignore repeated //'s.
  while (SlashPos != 0 && IS_DIR_SEPARATOR_CHAR(Filename[SlashPos-1]))
    --SlashPos;

  return FileMgr.getDirectory(Filename.substr(0, SlashPos));
}

/// getDirectory - Lookup, cache, and verify the specified directory.  This
/// returns null if the directory doesn't exist.
///
const DirectoryEntry *FileManager::getDirectory(llvm::StringRef Filename) {
  // stat doesn't like trailing separators (at least on Windows).
  if (Filename.size() > 1 && IS_DIR_SEPARATOR_CHAR(Filename.back()))
    Filename = Filename.substr(0, Filename.size()-1);

  ++NumDirLookups;
  llvm::StringMapEntry<DirectoryEntry *> &NamedDirEnt =
    DirEntries.GetOrCreateValue(Filename);

  // See if there is already an entry in the map.
  if (NamedDirEnt.getValue())
    return NamedDirEnt.getValue() == NON_EXISTENT_DIR
              ? 0 : NamedDirEnt.getValue();

  ++NumDirCacheMisses;

  // By default, initialize it to invalid.
  NamedDirEnt.setValue(NON_EXISTENT_DIR);

  // Get the null-terminated directory name as stored as the key of the
  // DirEntries map.
  const char *InterndDirName = NamedDirEnt.getKeyData();

  // Check to see if the directory exists.
  struct stat StatBuf;
  if (getStatValue(InterndDirName, StatBuf, true))
    return 0;

  // It exists.  See if we have already opened a directory with the same inode.
  // This occurs when one dir is symlinked to another, for example.
  DirectoryEntry &UDE = UniqueDirs.getDirectory(InterndDirName, StatBuf);

  NamedDirEnt.setValue(&UDE);
  if (UDE.getName()) // Already have an entry with this inode, return it.
    return &UDE;

  // Otherwise, we don't have this directory yet, add it.  We use the string
  // key from the DirEntries map as the string.
  UDE.Name  = InterndDirName;
  return &UDE;
}

/// getFile - Lookup, cache, and verify the specified file.  This returns null
/// if the file doesn't exist.
///
const FileEntry *FileManager::getFile(llvm::StringRef Filename) {
  ++NumFileLookups;

  // See if there is already an entry in the map.
  llvm::StringMapEntry<FileEntry *> &NamedFileEnt =
    FileEntries.GetOrCreateValue(Filename);

  // See if there is already an entry in the map.
  if (NamedFileEnt.getValue())
    return NamedFileEnt.getValue() == NON_EXISTENT_FILE
                 ? 0 : NamedFileEnt.getValue();

  ++NumFileCacheMisses;

  // By default, initialize it to invalid.
  NamedFileEnt.setValue(NON_EXISTENT_FILE);


  // Get the null-terminated file name as stored as the key of the
  // FileEntries map.
  const char *InterndFileName = NamedFileEnt.getKeyData();

  
  // Look up the directory for the file.  When looking up something like
  // sys/foo.h we'll discover all of the search directories that have a 'sys'
  // subdirectory.  This will let us avoid having to waste time on known-to-fail
  // searches when we go to find sys/bar.h, because all the search directories
  // without a 'sys' subdir will get a cached failure result.
  const DirectoryEntry *DirInfo = getDirectoryFromFile(*this, Filename);
  if (DirInfo == 0)  // Directory doesn't exist, file can't exist.
    return 0;

  // FIXME: Use the directory info to prune this, before doing the stat syscall.
  // FIXME: This will reduce the # syscalls.

  // Nope, there isn't.  Check to see if the file exists.
  struct stat StatBuf;
  if (getStatValue(InterndFileName, StatBuf, false))
    return 0;

  // It exists.  See if we have already opened a file with the same inode.
  // This occurs when one dir is symlinked to another, for example.
  FileEntry &UFE = UniqueFiles.getFile(InterndFileName, StatBuf);

  NamedFileEnt.setValue(&UFE);
  if (UFE.getName())  // Already have an entry with this inode, return it.
    return &UFE;

  // Otherwise, we don't have this directory yet, add it.
  // FIXME: Change the name to be a char* that points back to the 'FileEntries'
  // key.
  UFE.Name    = InterndFileName;
  UFE.Size    = StatBuf.st_size;
  UFE.ModTime = StatBuf.st_mtime;
  UFE.Dir     = DirInfo;
  UFE.UID     = NextFileUID++;
  return &UFE;
}

const FileEntry *
FileManager::getVirtualFile(llvm::StringRef Filename, off_t Size,
                            time_t ModificationTime) {
  ++NumFileLookups;

  // See if there is already an entry in the map.
  llvm::StringMapEntry<FileEntry *> &NamedFileEnt =
    FileEntries.GetOrCreateValue(Filename);

  // See if there is already an entry in the map.
  if (NamedFileEnt.getValue())
    return NamedFileEnt.getValue() == NON_EXISTENT_FILE
                 ? 0 : NamedFileEnt.getValue();

  ++NumFileCacheMisses;

  // By default, initialize it to invalid.
  NamedFileEnt.setValue(NON_EXISTENT_FILE);

  const DirectoryEntry *DirInfo = getDirectoryFromFile(*this, Filename);
  if (DirInfo == 0)  // Directory doesn't exist, file can't exist.
    return 0;

  FileEntry *UFE = new FileEntry();
  VirtualFileEntries.push_back(UFE);
  NamedFileEnt.setValue(UFE);

  // Get the null-terminated file name as stored as the key of the
  // FileEntries map.
  const char *InterndFileName = NamedFileEnt.getKeyData();
   
  UFE->Name    = InterndFileName;
  UFE->Size    = Size;
  UFE->ModTime = ModificationTime;
  UFE->Dir     = DirInfo;
  UFE->UID     = NextFileUID++;
  
  // If this virtual file resolves to a file, also map that file to the 
  // newly-created file entry.
  struct stat StatBuf;
  if (getStatValue(InterndFileName, StatBuf, false))
    return UFE;
    
  llvm::sys::Path FilePath(UFE->Name);
  FilePath.makeAbsolute();
  FileEntries[FilePath.str()] = UFE;
  return UFE;
}

void FileManager::FixupRelativePath(llvm::sys::Path &path,
                                    const FileSystemOptions &FSOpts) {
  if (FSOpts.WorkingDir.empty() || path.isAbsolute()) return;
  
  llvm::sys::Path NewPath(FSOpts.WorkingDir);
  NewPath.appendComponent(path.str());
  path = NewPath;
}

llvm::MemoryBuffer *FileManager::
getBufferForFile(const FileEntry *Entry, std::string *ErrorStr) {
  llvm::StringRef Filename = Entry->getName();
  if (FileSystemOpts.WorkingDir.empty())
    return llvm::MemoryBuffer::getFile(Filename, ErrorStr, Entry->getSize());
  
  llvm::sys::Path FilePath(Filename);
  FixupRelativePath(FilePath, FileSystemOpts);
  return llvm::MemoryBuffer::getFile(FilePath.c_str(), ErrorStr,
                                     Entry->getSize());
}

llvm::MemoryBuffer *FileManager::
getBufferForFile(llvm::StringRef Filename, std::string *ErrorStr) {
  if (FileSystemOpts.WorkingDir.empty())
    return llvm::MemoryBuffer::getFile(Filename, ErrorStr);
  
  llvm::sys::Path FilePath(Filename);
  FixupRelativePath(FilePath, FileSystemOpts);
  return llvm::MemoryBuffer::getFile(FilePath.c_str(), ErrorStr);
}

/// getStatValue - Get the 'stat' information for the specified path, using the
/// cache to accellerate it if possible.  This returns true if the path does not
/// exist or false if it exists.
///
/// The isForDir member indicates whether this is a directory lookup or not.
/// This will return failure if the lookup isn't the expected kind.
bool FileManager::getStatValue(const char *Path, struct stat &StatBuf,
                               bool isForDir) {
  // FIXME: FileSystemOpts shouldn't be passed in here, all paths should be
  // absolute!
  if (FileSystemOpts.WorkingDir.empty())
    return FileSystemStatCache::get(Path, StatBuf, StatCache.get()) ||
           S_ISDIR(StatBuf.st_mode) != isForDir;
  
  llvm::sys::Path FilePath(Path);
  FixupRelativePath(FilePath, FileSystemOpts);

  return FileSystemStatCache::get(FilePath.c_str(), StatBuf, StatCache.get()) ||
         S_ISDIR(StatBuf.st_mode) != isForDir;
}



void FileManager::PrintStats() const {
  llvm::errs() << "\n*** File Manager Stats:\n";
  llvm::errs() << UniqueFiles.size() << " files found, "
               << UniqueDirs.size() << " dirs found.\n";
  llvm::errs() << NumDirLookups << " dir lookups, "
               << NumDirCacheMisses << " dir cache misses.\n";
  llvm::errs() << NumFileLookups << " file lookups, "
               << NumFileCacheMisses << " file cache misses.\n";

  //llvm::errs() << PagesMapped << BytesOfPagesMapped << FSLookups;
}

