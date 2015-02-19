//===-- sanitizer_libignore.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// LibIgnore allows to ignore all interceptors called from a particular set
// of dynamic libraries. LibIgnore can be initialized with several templates
// of names of libraries to be ignored. It finds code ranges for the libraries;
// and checks whether the provided PC value belongs to the code ranges.
//
//===----------------------------------------------------------------------===//

#ifndef SANITIZER_LIBIGNORE_H
#define SANITIZER_LIBIGNORE_H

#include "sanitizer_internal_defs.h"
#include "sanitizer_common.h"
#include "sanitizer_atomic.h"
#include "sanitizer_mutex.h"

namespace __sanitizer {

class LibIgnore {
 public:
  explicit LibIgnore(LinkerInitialized);

  // Must be called during initialization.
  void AddIgnoredLibrary(const char *name_templ);

  // Must be called after a new dynamic library is loaded.
  void OnLibraryLoaded(const char *name);

  // Must be called after a dynamic library is unloaded.
  void OnLibraryUnloaded();

  // Checks whether the provided PC belongs to one of the ignored libraries.
  bool IsIgnored(uptr pc) const;

 private:
  struct Lib {
    char *templ;
    char *name;
    char *real_name;  // target of symlink
    bool loaded;
  };

  struct LibCodeRange {
    uptr begin;
    uptr end;
  };

  static const uptr kMaxLibs = 128;

  // Hot part:
  atomic_uintptr_t loaded_count_;
  LibCodeRange code_ranges_[kMaxLibs];

  // Cold part:
  BlockingMutex mutex_;
  uptr count_;
  Lib libs_[kMaxLibs];

  // Disallow copying of LibIgnore objects.
  LibIgnore(const LibIgnore&);  // not implemented
  void operator = (const LibIgnore&);  // not implemented
};

inline bool LibIgnore::IsIgnored(uptr pc) const {
  const uptr n = atomic_load(&loaded_count_, memory_order_acquire);
  for (uptr i = 0; i < n; i++) {
    if (pc >= code_ranges_[i].begin && pc < code_ranges_[i].end)
      return true;
  }
  return false;
}

}  // namespace __sanitizer

#endif  // SANITIZER_LIBIGNORE_H
