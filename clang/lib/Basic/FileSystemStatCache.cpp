//===--- FileSystemStatCache.cpp - Caching for 'stat' calls ---------------===//
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

#include "clang/Basic/FileSystemStatCache.h"
#include "llvm/System/Path.h"

// FIXME: This is terrible, we need this for ::close.
#if !defined(_MSC_VER) && !defined(__MINGW32__)
#include <unistd.h>
#include <sys/uio.h>
#else
#include <io.h>
#endif
using namespace clang;

#if defined(_MSC_VER)
#define S_ISDIR(s) (_S_IFDIR & s)
#endif

/// FileSystemStatCache::get - Get the 'stat' information for the specified
/// path, using the cache to accellerate it if possible.  This returns true if
/// the path does not exist or false if it exists.
///
/// If FileDescriptor is non-null, then this lookup should only return success
/// for files (not directories).  If it is null this lookup should only return
/// success for directories (not files).  On a successful file lookup, the
/// implementation can optionally fill in FileDescriptor with a valid
/// descriptor and the client guarantees that it will close it.
bool FileSystemStatCache::get(const char *Path, struct stat &StatBuf,
                              int *FileDescriptor, FileSystemStatCache *Cache) {
  LookupResult R;
  
  if (Cache)
    R = Cache->getStat(Path, StatBuf, FileDescriptor);
  else
    R = ::stat(Path, &StatBuf) != 0 ? CacheMissing : CacheExists;

  // If the path doesn't exist, return failure.
  if (R == CacheMissing) return true;
  
  // If the path exists, make sure that its "directoryness" matches the clients
  // demands.
  bool isForDir = FileDescriptor == 0;
  if (S_ISDIR(StatBuf.st_mode) != isForDir) {
    // If not, close the file if opened.
    if (FileDescriptor && *FileDescriptor != -1) {
      ::close(*FileDescriptor);
      *FileDescriptor = -1;
    }
    
    return true;
  }
  
  return false;
}


MemorizeStatCalls::LookupResult
MemorizeStatCalls::getStat(const char *Path, struct stat &StatBuf,
                           int *FileDescriptor) {
  LookupResult Result = statChained(Path, StatBuf, FileDescriptor);
  
  // Do not cache failed stats, it is easy to construct common inconsistent
  // situations if we do, and they are not important for PCH performance (which
  // currently only needs the stats to construct the initial FileManager
  // entries).
  if (Result == CacheMissing)
    return Result;
  
  // Cache file 'stat' results and directories with absolutely paths.
  if (!S_ISDIR(StatBuf.st_mode) || llvm::sys::Path(Path).isAbsolute())
    StatCalls[Path] = StatBuf;
  
  return Result;
}
