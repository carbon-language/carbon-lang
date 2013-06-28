//===-- ObjectCache.h - Class definition for the ObjectCache -----C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_EXECUTIONENGINE_OBJECTCACHE_H
#define LLVM_LIB_EXECUTIONENGINE_OBJECTCACHE_H

#include "llvm/Support/MemoryBuffer.h"

namespace llvm {

class Module;

/// This is the base ObjectCache type which can be provided to an
/// ExecutionEngine for the purpose of avoiding compilation for Modules that
/// have already been compiled and an object file is available.
class ObjectCache {
public:
  ObjectCache() { }

  virtual ~ObjectCache() { }

  /// notifyObjectCompiled - Provides a pointer to compiled code for Module M.
  virtual void notifyObjectCompiled(const Module *M, const MemoryBuffer *Obj) = 0;

  /// getObjectCopy - Returns a pointer to a newly allocated MemoryBuffer that
  /// contains the object which corresponds with Module M, or 0 if an object is
  /// not available. The caller owns both the MemoryBuffer returned by this
  /// and the memory it references.
  virtual MemoryBuffer* getObject(const Module* M) = 0;
};

}

#endif
