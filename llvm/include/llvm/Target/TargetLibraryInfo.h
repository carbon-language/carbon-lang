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
      /// double acos(double x);
      acos,
      /// long double acosl(long double x);
      acosl,
      /// float acosf(float x);
      acosf,
      /// double asin(double x);
      asin,
      /// long double asinl(long double x);
      asinl,
      /// float asinf(float x);
      asinf,
      /// double atan(double x);
      atan,
      /// long double atanl(long double x);
      atanl,
      /// float atanf(float x);
      atanf,
      /// double ceil(double x);
      ceil,
      /// long double ceill(long double x);
      ceill,
      /// float ceilf(float x);
      ceilf,
      /// double cos(double x);
      cos,
      /// long double cosl(long double x);
      cosl,
      /// float cosf(float x);
      cosf,
      /// double cosh(double x);
      cosh,
      /// long double coshl(long double x);
      coshl,
      /// float coshf(float x);
      coshf,
      /// double exp(double x);
      exp,
      /// long double expl(long double x);
      expl,
      /// float expf(float x);
      expf,
      /// double exp2(double x);
      exp2,
      /// long double exp2l(long double x);
      exp2l,
      /// float exp2f(float x);
      exp2f,
      /// double expm1(double x);
      expm1,
      /// long double expm1l(long double x);
      expm1l,
      /// float expm1f(float x);
      expl1f,
      /// double fabs(double x);
      fabs,
      /// long double fabsl(long double x);
      fabsl,
      /// float fabsf(float x);
      fabsf,
      /// double floor(double x);
      floor,
      /// long double floorl(long double x);
      floorl,
      /// float floorf(float x);
      floorf,
      /// int fiprintf(FILE *stream, const char *format, ...);
      fiprintf,
      /// int fputs(const char *s, FILE *stream);
      fputs,
      /// size_t fwrite(const void *ptr, size_t size, size_t nitems,
      /// FILE *stream);
      fwrite,
      /// int iprintf(const char *format, ...);
      iprintf,
      /// double log(double x);
      log,
      /// long double logl(long double x);
      logl,
      /// float logf(float x);
      logf,
      /// double log2(double x);
      log2,
      /// double long double log2l(long double x);
      log2l,
      /// float log2f(float x);
      log2f,
      /// double log10(double x);
      log10,
      /// long double log10l(long double x);
      log10l,
      /// float log10f(float x);
      log10f,
      /// double log1p(double x);
      log1p,
      /// long double log1pl(long double x);
      log1pl,
      /// float log1pf(float x);
      log1pf,
      /// void *memcpy(void *s1, const void *s2, size_t n);
      memcpy,
      /// void *memmove(void *s1, const void *s2, size_t n);
      memmove,
      /// void *memset(void *b, int c, size_t len);
      memset,
      /// void memset_pattern16(void *b, const void *pattern16, size_t len);
      memset_pattern16,
      /// double pow(double x, double y);
      pow,
      /// float powf(float x, float y);
      powf,
      /// long double powl(long double x, long double y);
      powl,
      /// int siprintf(char *str, const char *format, ...);
      siprintf,
      /// double sqrt(double x);
      sqrt,
      /// long double sqrtl(long double x);
      sqrtl,
      /// float sqrtf(float x);
      sqrtf,

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
