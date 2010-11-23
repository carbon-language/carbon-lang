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
using namespace clang;

MemorizeStatCalls::LookupResult
MemorizeStatCalls::getStat(const char *Path, struct stat &StatBuf) {
  LookupResult Result = statChained(Path, StatBuf);
  
  // If the chained cache didn't know anything about the file, do the stat now
  // so we can record the result.
  if (Result == CacheMiss)
    Result = ::stat(Path, &StatBuf) ? CacheHitMissing : CacheHitExists;
  
  
  // Do not cache failed stats, it is easy to construct common inconsistent
  // situations if we do, and they are not important for PCH performance (which
  // currently only needs the stats to construct the initial FileManager
  // entries).
  if (Result == CacheHitMissing)
    return Result;
  
  // Cache file 'stat' results and directories with absolutely paths.
  if (!S_ISDIR(StatBuf.st_mode) || llvm::sys::Path(Path).isAbsolute())
    StatCalls[Path] = StatBuf;
  
  return Result;
}
