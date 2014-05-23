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
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/system_error.h"
#include <map>
#include <set>
#include <string>

using namespace clang;

// FIXME: Enhance libsystem to support inode and other fields.
#include <sys/stat.h>

/// NON_EXISTENT_DIR - A special value distinct from null that is used to
/// represent a dir name that doesn't exist on the disk.
#define NON_EXISTENT_DIR reinterpret_cast<DirectoryEntry*>((intptr_t)-1)

/// NON_EXISTENT_FILE - A special value distinct from null that is used to
/// represent a filename that doesn't exist on the disk.
#define NON_EXISTENT_FILE reinterpret_cast<FileEntry*>((intptr_t)-1)

//===----------------------------------------------------------------------===//
// Common logic.
//===----------------------------------------------------------------------===//

FileManager::FileManager(const FileSystemOptions &FSO,
                         IntrusiveRefCntPtr<vfs::FileSystem> FS)
  : FS(FS), FileSystemOpts(FSO),
    SeenDirEntries(64), SeenFileEntries(64), NextFileUID(0) {
  NumDirLookups = NumFileLookups = 0;
  NumDirCacheMisses = NumFileCacheMisses = 0;

  // If the caller doesn't provide a virtual file system, just grab the real
  // file system.
  if (!FS)
    this->FS = vfs::getRealFileSystem();
}

FileManager::~FileManager() {
  for (unsigned i = 0, e = VirtualFileEntries.size(); i != e; ++i)
    delete VirtualFileEntries[i];
  for (unsigned i = 0, e = VirtualDirectoryEntries.size(); i != e; ++i)
    delete VirtualDirectoryEntries[i];
}

void FileManager::addStatCache(FileSystemStatCache *statCache,
                               bool AtBeginning) {
  assert(statCache && "No stat cache provided?");
  if (AtBeginning || !StatCache.get()) {
    statCache->setNextStatCache(StatCache.release());
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

void FileManager::clearStatCaches() {
  StatCache.reset(nullptr);
}

/// \brief Retrieve the directory that the given file name resides in.
/// Filename can point to either a real file or a virtual file.
static const DirectoryEntry *getDirectoryFromFile(FileManager &FileMgr,
                                                  StringRef Filename,
                                                  bool CacheFailure) {
  if (Filename.empty())
    return nullptr;

  if (llvm::sys::path::is_separator(Filename[Filename.size() - 1]))
    return nullptr; // If Filename is a directory.

  StringRef DirName = llvm::sys::path::parent_path(Filename);
  // Use the current directory if file has no path component.
  if (DirName.empty())
    DirName = ".";

  return FileMgr.getDirectory(DirName, CacheFailure);
}

/// Add all ancestors of the given path (pointing to either a file or
/// a directory) as virtual directories.
void FileManager::addAncestorsAsVirtualDirs(StringRef Path) {
  StringRef DirName = llvm::sys::path::parent_path(Path);
  if (DirName.empty())
    return;

  llvm::StringMapEntry<DirectoryEntry *> &NamedDirEnt =
    SeenDirEntries.GetOrCreateValue(DirName);

  // When caching a virtual directory, we always cache its ancestors
  // at the same time.  Therefore, if DirName is already in the cache,
  // we don't need to recurse as its ancestors must also already be in
  // the cache.
  if (NamedDirEnt.getValue())
    return;

  // Add the virtual directory to the cache.
  DirectoryEntry *UDE = new DirectoryEntry;
  UDE->Name = NamedDirEnt.getKeyData();
  NamedDirEnt.setValue(UDE);
  VirtualDirectoryEntries.push_back(UDE);

  // Recursively add the other ancestors.
  addAncestorsAsVirtualDirs(DirName);
}

const DirectoryEntry *FileManager::getDirectory(StringRef DirName,
                                                bool CacheFailure) {
  // stat doesn't like trailing separators except for root directory.
  // At least, on Win32 MSVCRT, stat() cannot strip trailing '/'.
  // (though it can strip '\\')
  if (DirName.size() > 1 &&
      DirName != llvm::sys::path::root_path(DirName) &&
      llvm::sys::path::is_separator(DirName.back()))
    DirName = DirName.substr(0, DirName.size()-1);
#ifdef LLVM_ON_WIN32
  // Fixing a problem with "clang C:test.c" on Windows.
  // Stat("C:") does not recognize "C:" as a valid directory
  std::string DirNameStr;
  if (DirName.size() > 1 && DirName.back() == ':' &&
      DirName.equals_lower(llvm::sys::path::root_name(DirName))) {
    DirNameStr = DirName.str() + '.';
    DirName = DirNameStr;
  }
#endif

  ++NumDirLookups;
  llvm::StringMapEntry<DirectoryEntry *> &NamedDirEnt =
    SeenDirEntries.GetOrCreateValue(DirName);

  // See if there was already an entry in the map.  Note that the map
  // contains both virtual and real directories.
  if (NamedDirEnt.getValue())
    return NamedDirEnt.getValue() == NON_EXISTENT_DIR ? nullptr
                                                      : NamedDirEnt.getValue();

  ++NumDirCacheMisses;

  // By default, initialize it to invalid.
  NamedDirEnt.setValue(NON_EXISTENT_DIR);

  // Get the null-terminated directory name as stored as the key of the
  // SeenDirEntries map.
  const char *InterndDirName = NamedDirEnt.getKeyData();

  // Check to see if the directory exists.
  FileData Data;
  if (getStatValue(InterndDirName, Data, false, nullptr /*directory lookup*/)) {
    // There's no real directory at the given path.
    if (!CacheFailure)
      SeenDirEntries.erase(DirName);
    return nullptr;
  }

  // It exists.  See if we have already opened a directory with the
  // same inode (this occurs on Unix-like systems when one dir is
  // symlinked to another, for example) or the same path (on
  // Windows).
  DirectoryEntry &UDE = UniqueRealDirs[Data.UniqueID];

  NamedDirEnt.setValue(&UDE);
  if (!UDE.getName()) {
    // We don't have this directory yet, add it.  We use the string
    // key from the SeenDirEntries map as the string.
    UDE.Name  = InterndDirName;
  }

  return &UDE;
}

const FileEntry *FileManager::getFile(StringRef Filename, bool openFile,
                                      bool CacheFailure) {
  ++NumFileLookups;

  // See if there is already an entry in the map.
  llvm::StringMapEntry<FileEntry *> &NamedFileEnt =
    SeenFileEntries.GetOrCreateValue(Filename);

  // See if there is already an entry in the map.
  if (NamedFileEnt.getValue())
    return NamedFileEnt.getValue() == NON_EXISTENT_FILE
                 ? nullptr : NamedFileEnt.getValue();

  ++NumFileCacheMisses;

  // By default, initialize it to invalid.
  NamedFileEnt.setValue(NON_EXISTENT_FILE);

  // Get the null-terminated file name as stored as the key of the
  // SeenFileEntries map.
  const char *InterndFileName = NamedFileEnt.getKeyData();

  // Look up the directory for the file.  When looking up something like
  // sys/foo.h we'll discover all of the search directories that have a 'sys'
  // subdirectory.  This will let us avoid having to waste time on known-to-fail
  // searches when we go to find sys/bar.h, because all the search directories
  // without a 'sys' subdir will get a cached failure result.
  const DirectoryEntry *DirInfo = getDirectoryFromFile(*this, Filename,
                                                       CacheFailure);
  if (DirInfo == nullptr) { // Directory doesn't exist, file can't exist.
    if (!CacheFailure)
      SeenFileEntries.erase(Filename);

    return nullptr;
  }
  
  // FIXME: Use the directory info to prune this, before doing the stat syscall.
  // FIXME: This will reduce the # syscalls.

  // Nope, there isn't.  Check to see if the file exists.
  vfs::File *F = nullptr;
  FileData Data;
  if (getStatValue(InterndFileName, Data, true, openFile ? &F : nullptr)) {
    // There's no real file at the given path.
    if (!CacheFailure)
      SeenFileEntries.erase(Filename);

    return nullptr;
  }

  assert((openFile || !F) && "undesired open file");

  // It exists.  See if we have already opened a file with the same inode.
  // This occurs when one dir is symlinked to another, for example.
  FileEntry &UFE = UniqueRealFiles[Data.UniqueID];

  NamedFileEnt.setValue(&UFE);
  if (UFE.isValid()) { // Already have an entry with this inode, return it.

    // FIXME: this hack ensures that if we look up a file by a virtual path in
    // the VFS that the getDir() will have the virtual path, even if we found
    // the file by a 'real' path first. This is required in order to find a
    // module's structure when its headers/module map are mapped in the VFS.
    // We should remove this as soon as we can properly support a file having
    // multiple names.
    if (DirInfo != UFE.Dir && Data.IsVFSMapped)
      UFE.Dir = DirInfo;

    // If the stat process opened the file, close it to avoid a FD leak.
    if (F)
      delete F;

    return &UFE;
  }

  // Otherwise, we don't have this file yet, add it.
  UFE.Name    = Data.Name;
  UFE.Size = Data.Size;
  UFE.ModTime = Data.ModTime;
  UFE.Dir     = DirInfo;
  UFE.UID     = NextFileUID++;
  UFE.UniqueID = Data.UniqueID;
  UFE.IsNamedPipe = Data.IsNamedPipe;
  UFE.InPCH = Data.InPCH;
  UFE.File.reset(F);
  UFE.IsValid = true;
  return &UFE;
}

const FileEntry *
FileManager::getVirtualFile(StringRef Filename, off_t Size,
                            time_t ModificationTime) {
  ++NumFileLookups;

  // See if there is already an entry in the map.
  llvm::StringMapEntry<FileEntry *> &NamedFileEnt =
    SeenFileEntries.GetOrCreateValue(Filename);

  // See if there is already an entry in the map.
  if (NamedFileEnt.getValue() && NamedFileEnt.getValue() != NON_EXISTENT_FILE)
    return NamedFileEnt.getValue();

  ++NumFileCacheMisses;

  // By default, initialize it to invalid.
  NamedFileEnt.setValue(NON_EXISTENT_FILE);

  addAncestorsAsVirtualDirs(Filename);
  FileEntry *UFE = nullptr;

  // Now that all ancestors of Filename are in the cache, the
  // following call is guaranteed to find the DirectoryEntry from the
  // cache.
  const DirectoryEntry *DirInfo = getDirectoryFromFile(*this, Filename,
                                                       /*CacheFailure=*/true);
  assert(DirInfo &&
         "The directory of a virtual file should already be in the cache.");

  // Check to see if the file exists. If so, drop the virtual file
  FileData Data;
  const char *InterndFileName = NamedFileEnt.getKeyData();
  if (getStatValue(InterndFileName, Data, true, nullptr) == 0) {
    Data.Size = Size;
    Data.ModTime = ModificationTime;
    UFE = &UniqueRealFiles[Data.UniqueID];

    NamedFileEnt.setValue(UFE);

    // If we had already opened this file, close it now so we don't
    // leak the descriptor. We're not going to use the file
    // descriptor anyway, since this is a virtual file.
    if (UFE->File)
      UFE->closeFile();

    // If we already have an entry with this inode, return it.
    if (UFE->isValid())
      return UFE;

    UFE->UniqueID = Data.UniqueID;
    UFE->IsNamedPipe = Data.IsNamedPipe;
    UFE->InPCH = Data.InPCH;
  }

  if (!UFE) {
    UFE = new FileEntry();
    VirtualFileEntries.push_back(UFE);
    NamedFileEnt.setValue(UFE);
  }

  UFE->Name    = InterndFileName;
  UFE->Size    = Size;
  UFE->ModTime = ModificationTime;
  UFE->Dir     = DirInfo;
  UFE->UID     = NextFileUID++;
  UFE->File.reset();
  return UFE;
}

void FileManager::FixupRelativePath(SmallVectorImpl<char> &path) const {
  StringRef pathRef(path.data(), path.size());

  if (FileSystemOpts.WorkingDir.empty() 
      || llvm::sys::path::is_absolute(pathRef))
    return;

  SmallString<128> NewPath(FileSystemOpts.WorkingDir);
  llvm::sys::path::append(NewPath, pathRef);
  path = NewPath;
}

llvm::MemoryBuffer *FileManager::
getBufferForFile(const FileEntry *Entry, std::string *ErrorStr,
                 bool isVolatile) {
  std::unique_ptr<llvm::MemoryBuffer> Result;
  llvm::error_code ec;

  uint64_t FileSize = Entry->getSize();
  // If there's a high enough chance that the file have changed since we
  // got its size, force a stat before opening it.
  if (isVolatile)
    FileSize = -1;

  const char *Filename = Entry->getName();
  // If the file is already open, use the open file descriptor.
  if (Entry->File) {
    ec = Entry->File->getBuffer(Filename, Result, FileSize,
                                /*RequiresNullTerminator=*/true, isVolatile);
    if (ErrorStr)
      *ErrorStr = ec.message();
    Entry->closeFile();
    return Result.release();
  }

  // Otherwise, open the file.

  if (FileSystemOpts.WorkingDir.empty()) {
    ec = FS->getBufferForFile(Filename, Result, FileSize,
                              /*RequiresNullTerminator=*/true, isVolatile);
    if (ec && ErrorStr)
      *ErrorStr = ec.message();
    return Result.release();
  }

  SmallString<128> FilePath(Entry->getName());
  FixupRelativePath(FilePath);
  ec = FS->getBufferForFile(FilePath.str(), Result, FileSize,
                            /*RequiresNullTerminator=*/true, isVolatile);
  if (ec && ErrorStr)
    *ErrorStr = ec.message();
  return Result.release();
}

llvm::MemoryBuffer *FileManager::
getBufferForFile(StringRef Filename, std::string *ErrorStr) {
  std::unique_ptr<llvm::MemoryBuffer> Result;
  llvm::error_code ec;
  if (FileSystemOpts.WorkingDir.empty()) {
    ec = FS->getBufferForFile(Filename, Result);
    if (ec && ErrorStr)
      *ErrorStr = ec.message();
    return Result.release();
  }

  SmallString<128> FilePath(Filename);
  FixupRelativePath(FilePath);
  ec = FS->getBufferForFile(FilePath.c_str(), Result);
  if (ec && ErrorStr)
    *ErrorStr = ec.message();
  return Result.release();
}

/// getStatValue - Get the 'stat' information for the specified path,
/// using the cache to accelerate it if possible.  This returns true
/// if the path points to a virtual file or does not exist, or returns
/// false if it's an existent real file.  If FileDescriptor is NULL,
/// do directory look-up instead of file look-up.
bool FileManager::getStatValue(const char *Path, FileData &Data, bool isFile,
                               vfs::File **F) {
  // FIXME: FileSystemOpts shouldn't be passed in here, all paths should be
  // absolute!
  if (FileSystemOpts.WorkingDir.empty())
    return FileSystemStatCache::get(Path, Data, isFile, F,StatCache.get(), *FS);

  SmallString<128> FilePath(Path);
  FixupRelativePath(FilePath);

  return FileSystemStatCache::get(FilePath.c_str(), Data, isFile, F,
                                  StatCache.get(), *FS);
}

bool FileManager::getNoncachedStatValue(StringRef Path,
                                        vfs::Status &Result) {
  SmallString<128> FilePath(Path);
  FixupRelativePath(FilePath);

  llvm::ErrorOr<vfs::Status> S = FS->status(FilePath.c_str());
  if (!S)
    return true;
  Result = *S;
  return false;
}

void FileManager::invalidateCache(const FileEntry *Entry) {
  assert(Entry && "Cannot invalidate a NULL FileEntry");

  SeenFileEntries.erase(Entry->getName());

  // FileEntry invalidation should not block future optimizations in the file
  // caches. Possible alternatives are cache truncation (invalidate last N) or
  // invalidation of the whole cache.
  UniqueRealFiles.erase(Entry->getUniqueID());
}


void FileManager::GetUniqueIDMapping(
                   SmallVectorImpl<const FileEntry *> &UIDToFiles) const {
  UIDToFiles.clear();
  UIDToFiles.resize(NextFileUID);
  
  // Map file entries
  for (llvm::StringMap<FileEntry*, llvm::BumpPtrAllocator>::const_iterator
         FE = SeenFileEntries.begin(), FEEnd = SeenFileEntries.end();
       FE != FEEnd; ++FE)
    if (FE->getValue() && FE->getValue() != NON_EXISTENT_FILE)
      UIDToFiles[FE->getValue()->getUID()] = FE->getValue();
  
  // Map virtual file entries
  for (SmallVectorImpl<FileEntry *>::const_iterator
         VFE = VirtualFileEntries.begin(), VFEEnd = VirtualFileEntries.end();
       VFE != VFEEnd; ++VFE)
    if (*VFE && *VFE != NON_EXISTENT_FILE)
      UIDToFiles[(*VFE)->getUID()] = *VFE;
}

void FileManager::modifyFileEntry(FileEntry *File,
                                  off_t Size, time_t ModificationTime) {
  File->Size = Size;
  File->ModTime = ModificationTime;
}

StringRef FileManager::getCanonicalName(const DirectoryEntry *Dir) {
  // FIXME: use llvm::sys::fs::canonical() when it gets implemented
#ifdef LLVM_ON_UNIX
  llvm::DenseMap<const DirectoryEntry *, llvm::StringRef>::iterator Known
    = CanonicalDirNames.find(Dir);
  if (Known != CanonicalDirNames.end())
    return Known->second;

  StringRef CanonicalName(Dir->getName());
  char CanonicalNameBuf[PATH_MAX];
  if (realpath(Dir->getName(), CanonicalNameBuf)) {
    unsigned Len = strlen(CanonicalNameBuf);
    char *Mem = static_cast<char *>(CanonicalNameStorage.Allocate(Len, 1));
    memcpy(Mem, CanonicalNameBuf, Len);
    CanonicalName = StringRef(Mem, Len);
  }

  CanonicalDirNames.insert(std::make_pair(Dir, CanonicalName));
  return CanonicalName;
#else
  return StringRef(Dir->getName());
#endif
}

void FileManager::PrintStats() const {
  llvm::errs() << "\n*** File Manager Stats:\n";
  llvm::errs() << UniqueRealFiles.size() << " real files found, "
               << UniqueRealDirs.size() << " real dirs found.\n";
  llvm::errs() << VirtualFileEntries.size() << " virtual files found, "
               << VirtualDirectoryEntries.size() << " virtual dirs found.\n";
  llvm::errs() << NumDirLookups << " dir lookups, "
               << NumDirCacheMisses << " dir cache misses.\n";
  llvm::errs() << NumFileLookups << " file lookups, "
               << NumFileCacheMisses << " file cache misses.\n";

  //llvm::errs() << PagesMapped << BytesOfPagesMapped << FSLookups;
}
