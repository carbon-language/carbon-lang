//===--- Targets.cpp - Implement -arch option and targets -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the -arch command line option and creates a TargetInfo
// that represents them.
//
//===----------------------------------------------------------------------===//

#include "clang.h"
#include "clang/AST/Builtins.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/Support/CommandLine.h"
using namespace clang;

/// Note: a hard coded list of targets is clearly silly, these should be
/// dynamicly registered and loadable with "-load".
enum SupportedTargets {
  target_ppc, target_ppc64,
  target_i386, target_x86_64,
  target_linux_i386
};

static llvm::cl::list<SupportedTargets>
Archs("arch", llvm::cl::desc("Architectures to compile for"),
llvm::cl::values(clEnumValN(target_ppc,       "ppc",   "32-bit Darwin PowerPC"),
                 clEnumValN(target_ppc64,     "ppc64", "64-bit Darwin PowerPC"),
                 clEnumValN(target_i386,      "i386",  "32-bit Darwin X86"),
                 clEnumValN(target_x86_64,    "x86_64","64-bit Darwin X86"),
                 clEnumValN(target_linux_i386,"linux", "Linux i386"),
                 clEnumValEnd));

//===----------------------------------------------------------------------===//
//  Common code shared among targets.
//===----------------------------------------------------------------------===//

namespace {
class DarwinTargetInfo : public TargetInfoImpl {
public:
  virtual void getTargetDefines(std::vector<std::string> &Defines) const {
    Defines.push_back("__APPLE__=1");
    Defines.push_back("__MACH__=1");
    
    if (1) {// -fobjc-gc controls this.
      Defines.push_back("__weak=");
      Defines.push_back("__strong=");
    } else {
      Defines.push_back("__weak=__attribute__((objc_gc(weak)))");
      Defines.push_back("__strong=__attribute__((objc_gc(strong)))");
      Defines.push_back("__OBJC_GC__");
    }

    // darwin_constant_cfstrings controls this.
    Defines.push_back("__CONSTANT_CFSTRINGS__=1");
    
    if (0)  // darwin_pascal_strings
      Defines.push_back("__PASCAL_STRINGS__");
  }

};
} // end anonymous namespace.


/// getPowerPCDefines - Return a set of the PowerPC-specific #defines that are
/// not tied to a specific subtarget.
static void getPowerPCDefines(std::vector<std::string> &Defines, bool is64Bit) {
  // Target identification.
  Defines.push_back("__ppc__");
  Defines.push_back("_ARCH_PPC=1");
  Defines.push_back("__POWERPC__=1");
  if (is64Bit) {
    Defines.push_back("_ARCH_PPC64");
    Defines.push_back("_LP64");
    Defines.push_back("__LP64__");
    Defines.push_back("__ppc64__");
  } else {
    Defines.push_back("__ppc__=1");
  }

  // Target properties.
  Defines.push_back("_BIG_ENDIAN=1");
  Defines.push_back("__BIG_ENDIAN__=1");

  if (is64Bit) {
    Defines.push_back("__INTMAX_MAX__=9223372036854775807L");
    Defines.push_back("__INTMAX_TYPE__=long int");
    Defines.push_back("__LONG_MAX__=9223372036854775807L");
    Defines.push_back("__PTRDIFF_TYPE__=long int");
    Defines.push_back("__UINTMAX_TYPE__=long unsigned int");
  } else {
    Defines.push_back("__INTMAX_MAX__=9223372036854775807LL");
    Defines.push_back("__INTMAX_TYPE__=long long int");
    Defines.push_back("__LONG_MAX__=2147483647L");
    Defines.push_back("__PTRDIFF_TYPE__=int");
    Defines.push_back("__UINTMAX_TYPE__=long long unsigned int");
  }
  Defines.push_back("__INT_MAX__=2147483647");
  Defines.push_back("__LONG_LONG_MAX__=9223372036854775807LL");
  Defines.push_back("__CHAR_BIT__=8");
  Defines.push_back("__SCHAR_MAX__=127");
  Defines.push_back("__SHRT_MAX__=32767");
  Defines.push_back("__SIZE_TYPE__=long unsigned int");
  
  // Subtarget options.
  Defines.push_back("__USER_LABEL_PREFIX__=_");
  Defines.push_back("__NATURAL_ALIGNMENT__=1");
  Defines.push_back("__REGISTER_PREFIX__=");

  Defines.push_back("__WCHAR_MAX__=2147483647");
  Defines.push_back("__WCHAR_TYPE__=int");
  Defines.push_back("__WINT_TYPE__=int");
  
  // Float macros.
  Defines.push_back("__FLT_DENORM_MIN__=1.40129846e-45F");
  Defines.push_back("__FLT_DIG__=6");
  Defines.push_back("__FLT_EPSILON__=1.19209290e-7F");
  Defines.push_back("__FLT_EVAL_METHOD__=0");
  Defines.push_back("__FLT_HAS_INFINITY__=1");
  Defines.push_back("__FLT_HAS_QUIET_NAN__=1");
  Defines.push_back("__FLT_MANT_DIG__=24");
  Defines.push_back("__FLT_MAX_10_EXP__=38");
  Defines.push_back("__FLT_MAX_EXP__=128");
  Defines.push_back("__FLT_MAX__=3.40282347e+38F");
  Defines.push_back("__FLT_MIN_10_EXP__=(-37)");
  Defines.push_back("__FLT_MIN_EXP__=(-125)");
  Defines.push_back("__FLT_MIN__=1.17549435e-38F");
  Defines.push_back("__FLT_RADIX__=2");
  
  // double macros.
  Defines.push_back("__DBL_DENORM_MIN__=4.9406564584124654e-324");
  Defines.push_back("__DBL_DIG__=15");
  Defines.push_back("__DBL_EPSILON__=2.2204460492503131e-16");
  Defines.push_back("__DBL_HAS_INFINITY__=1");
  Defines.push_back("__DBL_HAS_QUIET_NAN__=1");
  Defines.push_back("__DBL_MANT_DIG__=53");
  Defines.push_back("__DBL_MAX_10_EXP__=308");
  Defines.push_back("__DBL_MAX_EXP__=1024");
  Defines.push_back("__DBL_MAX__=1.7976931348623157e+308");
  Defines.push_back("__DBL_MIN_10_EXP__=(-307)");
  Defines.push_back("__DBL_MIN_EXP__=(-1021)");
  Defines.push_back("__DBL_MIN__=2.2250738585072014e-308");
  Defines.push_back("__DECIMAL_DIG__=33");
  
  // 128-bit long double macros.
  Defines.push_back("__LDBL_DENORM_MIN__=4.940656458412465441765687"
                     "92868221e-324L");
  Defines.push_back("__LDBL_DIG__=31");
  Defines.push_back("__LDBL_EPSILON__=4.9406564584124654417656879286822"
                     "1e-324L");
  Defines.push_back("__LDBL_HAS_INFINITY__=1");
  Defines.push_back("__LDBL_HAS_QUIET_NAN__=1");
  Defines.push_back("__LDBL_MANT_DIG__=106");
  Defines.push_back("__LDBL_MAX_10_EXP__=308");
  Defines.push_back("__LDBL_MAX_EXP__=1024");
  Defines.push_back("__LDBL_MAX__=1.7976931348623158079372897140"
                     "5301e+308L");
  Defines.push_back("__LDBL_MIN_10_EXP__=(-291)");
  Defines.push_back("__LDBL_MIN_EXP__=(-968)");
  Defines.push_back("__LDBL_MIN__=2.004168360008972777996108051350"
                     "16e-292L");
  Defines.push_back("__LONG_DOUBLE_128__=1");
  
}

/// getX86Defines - Return a set of the X86-specific #defines that are
/// not tied to a specific subtarget.
static void getX86Defines(std::vector<std::string> &Defines, bool is64Bit) {
  // Target identification.
  if (is64Bit) {
    Defines.push_back("_LP64");
    Defines.push_back("__LP64__");
    Defines.push_back("__amd64__");
    Defines.push_back("__amd64");
    Defines.push_back("__x86_64");
    Defines.push_back("__x86_64__");
  } else {
    Defines.push_back("__i386__=1");
    Defines.push_back("__i386=1");
    Defines.push_back("i386=1");
  }

  // Target properties.
  Defines.push_back("__LITTLE_ENDIAN__=1");
  
  if (is64Bit) {
    Defines.push_back("__INTMAX_MAX__=9223372036854775807L");
    Defines.push_back("__INTMAX_TYPE__=long int");
    Defines.push_back("__LONG_MAX__=9223372036854775807L");
    Defines.push_back("__PTRDIFF_TYPE__=long int");
    Defines.push_back("__UINTMAX_TYPE__=long unsigned int");
  } else {
    Defines.push_back("__INTMAX_MAX__=9223372036854775807LL");
    Defines.push_back("__INTMAX_TYPE__=long long int");
    Defines.push_back("__LONG_MAX__=2147483647L");
    Defines.push_back("__PTRDIFF_TYPE__=int");
    Defines.push_back("__UINTMAX_TYPE__=long long unsigned int");
  }
  Defines.push_back("__CHAR_BIT__=8");
  Defines.push_back("__INT_MAX__=2147483647");
  Defines.push_back("__LONG_LONG_MAX__=9223372036854775807LL");
  Defines.push_back("__SCHAR_MAX__=127");
  Defines.push_back("__SHRT_MAX__=32767");
  Defines.push_back("__SIZE_TYPE__=long unsigned int");
  
  // Subtarget options.
  Defines.push_back("__nocona=1");
  Defines.push_back("__nocona__=1");
  Defines.push_back("__tune_nocona__=1");
  Defines.push_back("__SSE2_MATH__=1");
  Defines.push_back("__SSE2__=1");
  Defines.push_back("__SSE_MATH__=1");
  Defines.push_back("__SSE__=1");
  Defines.push_back("__MMX__=1");
  Defines.push_back("__REGISTER_PREFIX__=");

  Defines.push_back("__WCHAR_MAX__=2147483647");
  Defines.push_back("__WCHAR_TYPE__=int");
  Defines.push_back("__WINT_TYPE__=int");
  
  // Float macros.
  Defines.push_back("__FLT_DENORM_MIN__=1.40129846e-45F");
  Defines.push_back("__FLT_DIG__=6");
  Defines.push_back("__FLT_EPSILON__=1.19209290e-7F");
  Defines.push_back("__FLT_EVAL_METHOD__=0");
  Defines.push_back("__FLT_HAS_INFINITY__=1");
  Defines.push_back("__FLT_HAS_QUIET_NAN__=1");
  Defines.push_back("__FLT_MANT_DIG__=24");
  Defines.push_back("__FLT_MAX_10_EXP__=38");
  Defines.push_back("__FLT_MAX_EXP__=128");
  Defines.push_back("__FLT_MAX__=3.40282347e+38F");
  Defines.push_back("__FLT_MIN_10_EXP__=(-37)");
  Defines.push_back("__FLT_MIN_EXP__=(-125)");
  Defines.push_back("__FLT_MIN__=1.17549435e-38F");
  Defines.push_back("__FLT_RADIX__=2");
  
  // Double macros.
  Defines.push_back("__DBL_DENORM_MIN__=4.9406564584124654e-324");
  Defines.push_back("__DBL_DIG__=15");
  Defines.push_back("__DBL_EPSILON__=2.2204460492503131e-16");
  Defines.push_back("__DBL_HAS_INFINITY__=1");
  Defines.push_back("__DBL_HAS_QUIET_NAN__=1");
  Defines.push_back("__DBL_MANT_DIG__=53");
  Defines.push_back("__DBL_MAX_10_EXP__=308");
  Defines.push_back("__DBL_MAX_EXP__=1024");
  Defines.push_back("__DBL_MAX__=1.7976931348623157e+308");
  Defines.push_back("__DBL_MIN_10_EXP__=(-307)");
  Defines.push_back("__DBL_MIN_EXP__=(-1021)");
  Defines.push_back("__DBL_MIN__=2.2250738585072014e-308");
  Defines.push_back("__DECIMAL_DIG__=21");
  
  // 80-bit Long double macros.
  Defines.push_back("__LDBL_DENORM_MIN__=3.64519953188247460253e-4951L");
  Defines.push_back("__LDBL_DIG__=18");
  Defines.push_back("__LDBL_EPSILON__=1.08420217248550443401e-19L");
  Defines.push_back("__LDBL_HAS_INFINITY__=1");
  Defines.push_back("__LDBL_HAS_QUIET_NAN__=1");
  Defines.push_back("__LDBL_MANT_DIG__=64");
  Defines.push_back("__LDBL_MAX_10_EXP__=4932");
  Defines.push_back("__LDBL_MAX_EXP__=16384");
  Defines.push_back("__LDBL_MAX__=1.18973149535723176502e+4932L");
  Defines.push_back("__LDBL_MIN_10_EXP__=(-4931)");
  Defines.push_back("__LDBL_MIN_EXP__=(-16381)");
  Defines.push_back("__LDBL_MIN__=3.36210314311209350626e-4932L");

}

/// PPC builtin info.
namespace PPC {
  enum {
    LastTIBuiltin = Builtin::FirstTSBuiltin-1,
#define BUILTIN(ID, TYPE, ATTRS) BI##ID,
#include "PPCBuiltins.def"
    LastTSBuiltin
  };
  
  static const Builtin::Info BuiltinInfo[] = {
#define BUILTIN(ID, TYPE, ATTRS) { #ID, TYPE, ATTRS },
#include "PPCBuiltins.def"
  };
  
  static void getBuiltins(const Builtin::Info *&Records, unsigned &NumRecords) {
    Records = BuiltinInfo;
    NumRecords = LastTSBuiltin-Builtin::FirstTSBuiltin;
  }
} // End namespace PPC


/// X86 builtin info.
namespace X86 {
  enum {
    LastTIBuiltin = Builtin::FirstTSBuiltin-1,
#define BUILTIN(ID, TYPE, ATTRS) BI##ID,
#include "X86Builtins.def"
    LastTSBuiltin
  };

  static const Builtin::Info BuiltinInfo[] = {
#define BUILTIN(ID, TYPE, ATTRS) { #ID, TYPE, ATTRS },
#include "X86Builtins.def"
  };

  static void getBuiltins(const Builtin::Info *&Records, unsigned &NumRecords) {
    Records = BuiltinInfo;
    NumRecords = LastTSBuiltin-Builtin::FirstTSBuiltin;
  }
} // End namespace X86

//===----------------------------------------------------------------------===//
// Specific target implementations.
//===----------------------------------------------------------------------===//


namespace {
class DarwinPPCTargetInfo : public DarwinTargetInfo {
public:
  virtual void getTargetDefines(std::vector<std::string> &Defines) const {
    DarwinTargetInfo::getTargetDefines(Defines);
    getPowerPCDefines(Defines, false);
  }
  virtual void getTargetBuiltins(const Builtin::Info *&Records,
                                 unsigned &NumRecords) const {
    PPC::getBuiltins(Records, NumRecords);
  }
};
} // end anonymous namespace.

namespace {
class DarwinPPC64TargetInfo : public DarwinTargetInfo {
public:
  virtual void getTargetDefines(std::vector<std::string> &Defines) const {
    DarwinTargetInfo::getTargetDefines(Defines);
    getPowerPCDefines(Defines, true);
  }
  virtual void getTargetBuiltins(const Builtin::Info *&Records,
                                 unsigned &NumRecords) const {
    PPC::getBuiltins(Records, NumRecords);
  }
};
} // end anonymous namespace.

namespace {
class DarwinI386TargetInfo : public DarwinTargetInfo {
public:
  virtual void getTargetDefines(std::vector<std::string> &Defines) const {
    DarwinTargetInfo::getTargetDefines(Defines);
    getX86Defines(Defines, false);
  }
  virtual void getTargetBuiltins(const Builtin::Info *&Records,
                                 unsigned &NumRecords) const {
    X86::getBuiltins(Records, NumRecords);
  }
};
} // end anonymous namespace.

namespace {
class DarwinX86_64TargetInfo : public DarwinTargetInfo {
public:
  virtual void getTargetDefines(std::vector<std::string> &Defines) const {
    DarwinTargetInfo::getTargetDefines(Defines);
    getX86Defines(Defines, true);
  }
  virtual void getTargetBuiltins(const Builtin::Info *&Records,
                                 unsigned &NumRecords) const {
    X86::getBuiltins(Records, NumRecords);
  }
};
} // end anonymous namespace.

namespace {
class LinuxTargetInfo : public DarwinTargetInfo {
public:
  LinuxTargetInfo() {
    // Note: I have no idea if this is right, just for testing.
    WCharWidth = 16;
    WCharAlign = 16;
  }
  
  virtual void getTargetDefines(std::vector<std::string> &Defines) const {
    // TODO: linux-specific stuff.
    getX86Defines(Defines, false);
  }
  virtual void getTargetBuiltins(const Builtin::Info *&Records,
                                 unsigned &NumRecords) const {
    X86::getBuiltins(Records, NumRecords);
  }
};
} // end anonymous namespace.


//===----------------------------------------------------------------------===//
// Driver code
//===----------------------------------------------------------------------===//

/// CreateTarget - Create the TargetInfoImpl object for the specified target
/// enum value.
static TargetInfoImpl *CreateTarget(SupportedTargets T) {
  switch (T) {
  default: assert(0 && "Unknown target!");
  case target_ppc:        return new DarwinPPCTargetInfo();
  case target_ppc64:      return new DarwinPPC64TargetInfo();
  case target_i386:       return new DarwinI386TargetInfo();
  case target_x86_64:     return new DarwinX86_64TargetInfo();
  case target_linux_i386: return new LinuxTargetInfo();
  }
}

/// CreateTargetInfo - Return the set of target info objects as specified by
/// the -arch command line option.
TargetInfo *clang::CreateTargetInfo(Diagnostic &Diags) {
  // If the user didn't specify at least one architecture, auto-sense the
  // current host.  TODO: This is a hack. :)
  if (Archs.empty()) {
#ifndef __APPLE__
    // Assume non-apple = i386 for now.
    Archs.push_back(target_i386);
#elif (defined(__POWERPC__) || defined (__ppc__) || defined(_POWER)) && \
      defined(__ppc64__)
    Archs.push_back(target_ppc64);
#elif defined(__POWERPC__) || defined (__ppc__) || defined(_POWER)
    Archs.push_back(target_ppc);
#elif defined(__x86_64__)
    Archs.push_back(target_x86_64);
#elif defined(__i386__) || defined(i386) || defined(_M_IX86)
    Archs.push_back(target_i386);
#else
    // Don't know what this is!
    return 0;
#endif
  }

  // Create the primary target and target info.
  TargetInfo *TI = new TargetInfo(CreateTarget(Archs[0]), &Diags);
  
  // Add all secondary targets.
  for (unsigned i = 1, e = Archs.size(); i != e; ++i)
    TI->AddSecondaryTarget(CreateTarget(Archs[i]));
  return TI;
}
