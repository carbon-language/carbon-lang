//===-- TargetLibraryInfo.cpp - Runtime library information ----------------==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the TargetLibraryInfo class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetLibraryInfo.h"
#include "llvm/ADT/Triple.h"
using namespace llvm;

// Register the default implementation.
INITIALIZE_PASS(TargetLibraryInfo, "targetlibinfo",
                "Target Library Information", false, true)
char TargetLibraryInfo::ID = 0;

void TargetLibraryInfo::anchor() { }

const char* TargetLibraryInfo::StandardNames[LibFunc::NumLibFuncs] =
  {
    "acos",
    "acosl",
    "acosf",
    "asin",
    "asinl",
    "asinf",
    "atan",
    "atanl",
    "atanf",
    "atan2",
    "atan2l",
    "atan2f",
    "ceil",
    "ceill",
    "ceilf",
    "copysign",
    "copysignf",
    "copysignl",
    "cos",
    "cosl",
    "cosf",
    "cosh",
    "coshl",
    "coshf",
    "exp",
    "expl",
    "expf",
    "exp2",
    "exp2l",
    "exp2f",
    "expm1",
    "expm1l",
    "expm1f",
    "fabs",
    "fabsl",
    "fabsf",
    "floor",
    "floorl",
    "floorf",
    "fiprintf",
    "fmod",
    "fmodl",
    "fmodf",
    "fputs",
    "fwrite",
    "iprintf",
    "log",
    "logl",
    "logf",
    "log2",
    "log2l",
    "log2f",
    "log10",
    "log10l",
    "log10f",
    "log1p",
    "log1pl",
    "log1pf",
    "memcpy",
    "memmove",
    "memset",
    "memset_pattern16",
    "nearbyint",
    "nearbyintf",
    "nearbyintl",
    "pow",
    "powf",
    "powl",
    "rint",
    "rintf",
    "rintl",
    "round",
    "roundf",
    "roundl",
    "sin",
    "sinl",
    "sinf",
    "sinh",
    "sinhl",
    "sinhf",
    "siprintf",
    "sqrt",
    "sqrtl",
    "sqrtf",
    "tan",
    "tanl",
    "tanf",
    "tanh",
    "tanhl",
    "tanhf",
    "trunc",
    "truncf",
    "truncl",
    "__cxa_atexit",
    "__cxa_guard_abort",
    "__cxa_guard_acquire",
    "__cxa_guard_release"
  };

/// initialize - Initialize the set of available library functions based on the
/// specified target triple.  This should be carefully written so that a missing
/// target triple gets a sane set of defaults.
static void initialize(TargetLibraryInfo &TLI, const Triple &T) {
  initializeTargetLibraryInfoPass(*PassRegistry::getPassRegistry());

  
  // memset_pattern16 is only available on iOS 3.0 and Mac OS/X 10.5 and later.
  if (T.isMacOSX()) {
    if (T.isMacOSXVersionLT(10, 5))
      TLI.setUnavailable(LibFunc::memset_pattern16);
  } else if (T.getOS() == Triple::IOS) {
    if (T.isOSVersionLT(3, 0))
      TLI.setUnavailable(LibFunc::memset_pattern16);
  } else {
    TLI.setUnavailable(LibFunc::memset_pattern16);
  }

  if (T.isMacOSX() && T.getArch() == Triple::x86 &&
      !T.isMacOSXVersionLT(10, 7)) {
    // x86-32 OSX has a scheme where fwrite and fputs (and some other functions
    // we don't care about) have two versions; on recent OSX, the one we want
    // has a $UNIX2003 suffix. The two implementations are identical except
    // for the return value in some edge cases.  However, we don't want to
    // generate code that depends on the old symbols.
    TLI.setAvailableWithName(LibFunc::fwrite, "fwrite$UNIX2003");
    TLI.setAvailableWithName(LibFunc::fputs, "fputs$UNIX2003");
  }

  // iprintf and friends are only available on XCore and TCE.
  if (T.getArch() != Triple::xcore && T.getArch() != Triple::tce) {
    TLI.setUnavailable(LibFunc::iprintf);
    TLI.setUnavailable(LibFunc::siprintf);
    TLI.setUnavailable(LibFunc::fiprintf);
  }

  if (T.getOS() == Triple::Win32) {
    // Win32 does not support long double
    TLI.setUnavailable(LibFunc::acosl);
    TLI.setUnavailable(LibFunc::asinl);
    TLI.setUnavailable(LibFunc::atanl);
    TLI.setUnavailable(LibFunc::atan2l);
    TLI.setUnavailable(LibFunc::ceill);
    TLI.setUnavailable(LibFunc::copysignl);
    TLI.setUnavailable(LibFunc::cosl);
    TLI.setUnavailable(LibFunc::coshl);
    TLI.setUnavailable(LibFunc::expl);
    TLI.setUnavailable(LibFunc::fabsf); // Win32 and Win64 both lack fabsf
    TLI.setUnavailable(LibFunc::fabsl);
    TLI.setUnavailable(LibFunc::floorl);
    TLI.setUnavailable(LibFunc::fmodl);
    TLI.setUnavailable(LibFunc::logl);
    TLI.setUnavailable(LibFunc::powl);
    TLI.setUnavailable(LibFunc::sinl);
    TLI.setUnavailable(LibFunc::sinhl);
    TLI.setUnavailable(LibFunc::sqrtl);
    TLI.setUnavailable(LibFunc::tanl);
    TLI.setUnavailable(LibFunc::tanhl);

    // Win32 only has C89 math
    TLI.setUnavailable(LibFunc::exp2);
    TLI.setUnavailable(LibFunc::exp2f);
    TLI.setUnavailable(LibFunc::exp2l);
    TLI.setUnavailable(LibFunc::expm1);
    TLI.setUnavailable(LibFunc::expm1f);
    TLI.setUnavailable(LibFunc::expm1l);
    TLI.setUnavailable(LibFunc::log2);
    TLI.setUnavailable(LibFunc::log2f);
    TLI.setUnavailable(LibFunc::log2l);
    TLI.setUnavailable(LibFunc::log1p);
    TLI.setUnavailable(LibFunc::log1pf);
    TLI.setUnavailable(LibFunc::log1pl);
    TLI.setUnavailable(LibFunc::nearbyint);
    TLI.setUnavailable(LibFunc::nearbyintf);
    TLI.setUnavailable(LibFunc::nearbyintl);
    TLI.setUnavailable(LibFunc::rint);
    TLI.setUnavailable(LibFunc::rintf);
    TLI.setUnavailable(LibFunc::rintl);
    TLI.setUnavailable(LibFunc::round);
    TLI.setUnavailable(LibFunc::roundf);
    TLI.setUnavailable(LibFunc::roundl);
    TLI.setUnavailable(LibFunc::trunc);
    TLI.setUnavailable(LibFunc::truncf);
    TLI.setUnavailable(LibFunc::truncl);

    // Win32 provides some C99 math with mangled names
    TLI.setAvailableWithName(LibFunc::copysign, "_copysign");

    if (T.getArch() == Triple::x86) {
      // Win32 on x86 implements single-precision math functions as macros
      TLI.setUnavailable(LibFunc::acosf);
      TLI.setUnavailable(LibFunc::asinf);
      TLI.setUnavailable(LibFunc::atanf);
      TLI.setUnavailable(LibFunc::atan2f);
      TLI.setUnavailable(LibFunc::ceilf);
      TLI.setUnavailable(LibFunc::copysignf);
      TLI.setUnavailable(LibFunc::cosf);
      TLI.setUnavailable(LibFunc::coshf);
      TLI.setUnavailable(LibFunc::expf);
      TLI.setUnavailable(LibFunc::floorf);
      TLI.setUnavailable(LibFunc::fmodf);
      TLI.setUnavailable(LibFunc::logf);
      TLI.setUnavailable(LibFunc::powf);
      TLI.setUnavailable(LibFunc::sinf);
      TLI.setUnavailable(LibFunc::sinhf);
      TLI.setUnavailable(LibFunc::sqrtf);
      TLI.setUnavailable(LibFunc::tanf);
      TLI.setUnavailable(LibFunc::tanhf);
    }
  }
}


TargetLibraryInfo::TargetLibraryInfo() : ImmutablePass(ID) {
  // Default to everything being available.
  memset(AvailableArray, -1, sizeof(AvailableArray));

  initialize(*this, Triple());
}

TargetLibraryInfo::TargetLibraryInfo(const Triple &T) : ImmutablePass(ID) {
  // Default to everything being available.
  memset(AvailableArray, -1, sizeof(AvailableArray));
  
  initialize(*this, T);
}

TargetLibraryInfo::TargetLibraryInfo(const TargetLibraryInfo &TLI)
  : ImmutablePass(ID) {
  memcpy(AvailableArray, TLI.AvailableArray, sizeof(AvailableArray));
  CustomNames = TLI.CustomNames;
}


/// disableAllFunctions - This disables all builtins, which is used for options
/// like -fno-builtin.
void TargetLibraryInfo::disableAllFunctions() {
  memset(AvailableArray, 0, sizeof(AvailableArray));
}
