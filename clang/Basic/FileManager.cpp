//===--- FileManager.cpp - File System Probing and Caching ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
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
#include <iostream>
using namespace llvm;
using namespace clang;

// FIXME: Enhance libsystem to support inode and other fields.
#include <sys/stat.h>

/// getDirectory - Lookup, cache, and verify the specified directory.  This
/// returns null if the directory doesn't exist.
/// 
const DirectoryEntry *FileManager::getDirectory(const std::string &Filename) {
  ++NumDirLookups;
  // See if there is already an entry in the map.
  std::map<std::string, DirectoryEntry*>::iterator I = 
    DirEntries.lower_bound(Filename);
  if (I != DirEntries.end() && I->first == Filename)
    return I->second;
  
  ++NumDirCacheMisses;
  
  // By default, zero initialize it.
  DirectoryEntry *&Ent =
    DirEntries.insert(I, std::make_pair(Filename, (DirectoryEntry*)0))->second;
  
  // Nope, there isn't.  Check to see if the directory exists.
  struct stat StatBuf;
  if (stat(Filename.c_str(), &StatBuf) ||   // Error stat'ing.
      !S_ISDIR(StatBuf.st_mode))            // Not a directory?
    return 0;
  
  // It exists.  See if we have already opened a directory with the same inode.
  // This occurs when one dir is symlinked to another, for example.
  DirectoryEntry *&UDE = 
    UniqueDirs[std::make_pair(StatBuf.st_dev, StatBuf.st_ino)];
  
  if (UDE)  // Already have an entry with this inode, return it.
    return Ent = UDE;
  
  // Otherwise, we don't have this directory yet, add it.
  DirectoryEntry *DE = new DirectoryEntry();
  DE->Name           = Filename;
  return Ent = UDE = DE;
}

/// getFile - Lookup, cache, and verify the specified file.  This returns null
/// if the file doesn't exist.
/// 
const FileEntry *FileManager::getFile(const std::string &Filename) {
  ++NumFileLookups;
  
  // See if there is already an entry in the map.
  std::map<std::string, FileEntry*>::iterator I = 
    FileEntries.lower_bound(Filename);
  if (I != FileEntries.end() && I->first == Filename)
    return I->second;

  ++NumFileCacheMisses;

  // By default, zero initialize it.
  FileEntry *&Ent =
    FileEntries.insert(I, std::make_pair(Filename, (FileEntry*)0))->second;

  // Figure out what directory it is in.
  std::string DirName;
  
  // If the string contains a / in it, strip off everything after it.
  // FIXME: this logic should be in sys::Path.
  std::string::size_type SlashPos = Filename.find_last_of('/');
  if (SlashPos == std::string::npos)
    DirName = ".";  // Use the current directory if file has no path component.
  else if (SlashPos == Filename.size()-1)
    return 0;       // If filename ends with a /, it's a directory.
  else
    DirName = std::string(Filename.begin(), Filename.begin()+SlashPos);

  const DirectoryEntry *DirInfo = getDirectory(DirName);
  if (DirInfo == 0)  // Directory doesn't exist, file can't exist.
    return 0;
  
  // FIXME: Use the directory info to prune this, before doing the stat syscall.
  // FIXME: This will reduce the # syscalls.
  
  // Nope, there isn't.  Check to see if the file exists.
  struct stat StatBuf;
  if (stat(Filename.c_str(), &StatBuf) ||   // Error stat'ing.
      S_ISDIR(StatBuf.st_mode))             // A directory?
    return 0;
  
  // It exists.  See if we have already opened a directory with the same inode.
  // This occurs when one dir is symlinked to another, for example.
  FileEntry *&UFE = 
    UniqueFiles[std::make_pair(StatBuf.st_dev, StatBuf.st_ino)];
  
  if (UFE)  // Already have an entry with this inode, return it.
    return Ent = UFE;
  
  // Otherwise, we don't have this directory yet, add it.
  FileEntry *FE = new FileEntry();
  FE->Size      = StatBuf.st_size;
  FE->Name      = Filename;
  FE->Dir       = DirInfo;
  FE->UID       = NextFileUID++;
  return Ent = UFE = FE;
}

void FileManager::PrintStats() const {
  std::cerr << "\n*** File Manager Stats:\n";
  std::cerr << UniqueFiles.size() << " files found, "
            << UniqueDirs.size() << " dirs found.\n";
  std::cerr << NumDirLookups << " dir lookups, "
            << NumDirCacheMisses << " dir cache misses.\n";
  std::cerr << NumFileLookups << " file lookups, "
            << NumFileCacheMisses << " file cache misses.\n";
  
  //std::cerr << PagesMapped << BytesOfPagesMapped << FSLookups;
}
