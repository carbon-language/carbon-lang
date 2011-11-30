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
    "ceil",
    "ceill",
    "ceilf",
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
    "expl1f",
    "fabs",
    "fabsl",
    "fabsf",
    "floor",
    "floorl",
    "floorf",
    "fiprintf",
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
    "pow",
    "powf",
    "powl",
    "siprintf",
    "sqrt",
    "sqrtl",
    "sqrtf"
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
