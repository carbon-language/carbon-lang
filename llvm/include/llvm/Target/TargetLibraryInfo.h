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
#include "llvm/ADT/DenseMap.h"

namespace llvm {
  class Triple;

  namespace LibFunc {
    enum Func {
      /// void *memset(void *b, int c, size_t len);
      memset,
      
      // void *memcpy(void *s1, const void *s2, size_t n);
      memcpy,
      
      // void *memmove(void *s1, const void *s2, size_t n);
      memmove,
      
      /// void memset_pattern16(void *b, const void *pattern16, size_t len);
      memset_pattern16,
      
      /// int iprintf(const char *format, ...);
      iprintf,
      
      /// int siprintf(char *str, const char *format, ...);
      siprintf,
      
      /// int fiprintf(FILE *stream, const char *format, ...);
      fiprintf,

      // size_t fwrite(const void *ptr, size_t size, size_t nitems,
      //               FILE *stream);
      fwrite,

      // int fputs(const char *s, FILE *stream);
      fputs,

      NumLibFuncs
    };
  }

/// TargetLibraryInfo - This immutable pass captures information about what
/// library functions are available for the current target, and allows a
/// frontend to disable optimizations through -fno-builtin etc.
class TargetLibraryInfo : public ImmutablePass {
  unsigned char AvailableArray[(LibFunc::NumLibFuncs+3)/4];
  llvm::DenseMap<unsigned, std::string> CustomNames;
  static const char* StandardNames[LibFunc::NumLibFuncs];

  enum AvailabilityState {
    StandardName = 3, // (memset to all ones)
    CustomName = 1,
    Unavailable = 0  // (memset to all zeros)
  };
  void setState(LibFunc::Func F, AvailabilityState State) {
    AvailableArray[F/4] &= ~(3 << 2*(F&3));
    AvailableArray[F/4] |= State << 2*(F&3);
  }
  AvailabilityState getState(LibFunc::Func F) const {
    return static_cast<AvailabilityState>((AvailableArray[F/4] >> 2*(F&3)) & 3);
  }

public:
  static char ID;
  TargetLibraryInfo();
  TargetLibraryInfo(const Triple &T);
  explicit TargetLibraryInfo(const TargetLibraryInfo &TLI);
  
  /// has - This function is used by optimizations that want to match on or form
  /// a given library function.
  bool has(LibFunc::Func F) const {
    return getState(F) != Unavailable;
  }

  StringRef getName(LibFunc::Func F) const {
    AvailabilityState State = getState(F);
    if (State == Unavailable)
      return StringRef();
    if (State == StandardName)
      return StandardNames[F];
    assert(State == CustomName);
    return CustomNames.find(F)->second;
  }

  /// setUnavailable - this can be used by whatever sets up TargetLibraryInfo to
  /// ban use of specific library functions.
  void setUnavailable(LibFunc::Func F) {
    setState(F, Unavailable);
  }

  void setAvailable(LibFunc::Func F) {
    setState(F, StandardName);
  }

  void setAvailableWithName(LibFunc::Func F, StringRef Name) {
    if (StandardNames[F] != Name) {
      setState(F, CustomName);
      CustomNames[F] = Name;
      assert(CustomNames.find(F) != CustomNames.end());
    } else {
      setState(F, StandardName);
    }
  }

  /// disableAllFunctions - This disables all builtins, which is used for
  /// options like -fno-builtin.
  void disableAllFunctions();
};

} // end namespace llvm

#endif
