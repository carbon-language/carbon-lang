//===--- FileSystemStatCache.h - Caching for 'stat' calls -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the FileSystemStatCache interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FILESYSTEMSTATCACHE_H
#define LLVM_CLANG_FILESYSTEMSTATCACHE_H

#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/StringMap.h"
#include <sys/types.h>
#include <sys/stat.h>

namespace clang {

/// \brief Abstract interface for introducing a FileManager cache for 'stat'
/// system calls, which is used by precompiled and pretokenized headers to
/// improve performance.
class FileSystemStatCache {
protected:
  llvm::OwningPtr<FileSystemStatCache> NextStatCache;
  
public:
  virtual ~FileSystemStatCache() {}
  
  enum LookupResult {
    CacheHitExists,   //< We know the file exists and its cached stat data.
    CacheHitMissing,  //< We know that the file doesn't exist.
    CacheMiss         //< We don't know anything about the file.
  };

  /// FileSystemStatCache::get - Get the 'stat' information for the specified
  /// path, using the cache to accellerate it if possible.  This returns true if
  /// the path does not exist or false if it exists.
  static bool get(const char *Path, struct stat &StatBuf,
                  FileSystemStatCache *Cache) {
    LookupResult R = CacheMiss;
    
    if (Cache)
      R = Cache->getStat(Path, StatBuf);
    
    if (R == FileSystemStatCache::CacheMiss)
      return ::stat(Path, &StatBuf);
    return R == FileSystemStatCache::CacheHitMissing;
  }
  
  /// \brief Sets the next stat call cache in the chain of stat caches.
  /// Takes ownership of the given stat cache.
  void setNextStatCache(FileSystemStatCache *Cache) {
    NextStatCache.reset(Cache);
  }
  
  /// \brief Retrieve the next stat call cache in the chain.
  FileSystemStatCache *getNextStatCache() { return NextStatCache.get(); }
  
  /// \brief Retrieve the next stat call cache in the chain, transferring
  /// ownership of this cache (and, transitively, all of the remaining caches)
  /// to the caller.
  FileSystemStatCache *takeNextStatCache() { return NextStatCache.take(); }
  
protected:
  virtual LookupResult getStat(const char *Path, struct stat &StatBuf) = 0;

  LookupResult statChained(const char *Path, struct stat &StatBuf) {
    if (FileSystemStatCache *Next = getNextStatCache())
      return Next->getStat(Path, StatBuf);
    
    return CacheMiss;
  }
};

/// \brief A stat "cache" that can be used by FileManager to keep
/// track of the results of stat() calls that occur throughout the
/// execution of the front end.
class MemorizeStatCalls : public FileSystemStatCache {
public:
  /// \brief The set of stat() calls that have been seen.
  llvm::StringMap<struct stat, llvm::BumpPtrAllocator> StatCalls;
  
  typedef llvm::StringMap<struct stat, llvm::BumpPtrAllocator>::const_iterator
  iterator;
  
  iterator begin() const { return StatCalls.begin(); }
  iterator end() const { return StatCalls.end(); }
  
  virtual LookupResult getStat(const char *Path, struct stat &StatBuf);
};

} // end namespace clang

#endif
