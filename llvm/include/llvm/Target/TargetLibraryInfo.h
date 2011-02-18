//===-- llvm/Target/TargetLibraryInfo.h - Library information ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETLIBRARYINFO_H
#define LLVM_TARGET_TARGETLIBRARYINFO_H

#include "llvm/Pass.h"

namespace llvm {
  class Triple;

  namespace LibFunc {
    enum Func {
      /// void *memset(void *b, int c, size_t len);
      memset,
      
      // void *memcpy(void *s1, const void *s2, size_t n);
      memcpy,
      
      /// void memset_pattern16(void *b, const void *pattern16, size_t len);
      memset_pattern16,
      
      NumLibFuncs
    };
  }

/// TargetLibraryInfo - This immutable pass captures information about what
/// library functions are available for the current target, and allows a
/// frontend to disable optimizations through -fno-builtin etc.
class TargetLibraryInfo : public ImmutablePass {
  unsigned char AvailableArray[(LibFunc::NumLibFuncs+7)/8];
public:
  static char ID;
  TargetLibraryInfo();
  TargetLibraryInfo(const Triple &T);
  
  /// has - This function is used by optimizations that want to match on or form
  /// a given library function.
  bool has(LibFunc::Func F) const {
    return (AvailableArray[F/8] & (1 << (F&7))) != 0;
  }

  /// setUnavailable - this can be used by whatever sets up TargetLibraryInfo to
  /// ban use of specific library functions.
  void setUnavailable(LibFunc::Func F) {
    AvailableArray[F/8] &= ~(1 << (F&7));
  }

  void setAvailable(LibFunc::Func F) {
    AvailableArray[F/8] |= 1 << (F&7);
  }
  
  /// disableAllFunctions - This disables all builtins, which is used for
  /// options like -fno-builtin.
  void disableAllFunctions();
};

} // end namespace llvm

#endif
