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

static void Define(std::vector<char> &Buf, const char *Macro,
                   const char *Val = "1") {
  const char *Def = "#define ";
  Buf.insert(Buf.end(), Def, Def+strlen(Def));
  Buf.insert(Buf.end(), Macro, Macro+strlen(Macro));
  Buf.push_back(' ');
  Buf.insert(Buf.end(), Val, Val+strlen(Val));
  Buf.push_back('\n');
}


namespace {
class DarwinTargetInfo : public TargetInfoImpl {
public:
  virtual void getTargetDefines(std::vector<char> &Defs) const {
    Define(Defs, "__APPLE__");
    Define(Defs, "__MACH__");
    
    if (1) {// -fobjc-gc controls this.
      Define(Defs, "__weak", "");
      Define(Defs, "__strong", "");
    } else {
      Define(Defs, "__weak", "__attribute__((objc_gc(weak)))");
      Define(Defs, "__strong", "__attribute__((objc_gc(strong)))");
      Define(Defs, "__OBJC_GC__");
    }

    // darwin_constant_cfstrings controls this.
    Define(Defs, "__CONSTANT_CFSTRINGS__");
    
    if (0)  // darwin_pascal_strings
      Define(Defs, "__PASCAL_STRINGS__");
  }

};
} // end anonymous namespace.


/// getPowerPCDefines - Return a set of the PowerPC-specific #defines that are
/// not tied to a specific subtarget.
static void getPowerPCDefines(std::vector<char> &Defs, bool is64Bit) {
  // Target identification.
  Define(Defs, "__ppc__");
  Define(Defs, "_ARCH_PPC");
  Define(Defs, "__POWERPC__");
  if (is64Bit) {
    Define(Defs, "_ARCH_PPC64");
    Define(Defs, "_LP64");
    Define(Defs, "__LP64__");
    Define(Defs, "__ppc64__");
  } else {
    Define(Defs, "__ppc__");
  }

  // Target properties.
  Define(Defs, "_BIG_ENDIAN");
  Define(Defs, "__BIG_ENDIAN__");

  if (is64Bit) {
    Define(Defs, "__INTMAX_MAX__", "9223372036854775807L");
    Define(Defs, "__INTMAX_TYPE__", "long int");
    Define(Defs, "__LONG_MAX__", "9223372036854775807L");
    Define(Defs, "__PTRDIFF_TYPE__", "long int");
    Define(Defs, "__UINTMAX_TYPE__", "long unsigned int");
  } else {
    Define(Defs, "__INTMAX_MAX__", "9223372036854775807LL");
    Define(Defs, "__INTMAX_TYPE__", "long long int");
    Define(Defs, "__LONG_MAX__", "2147483647L");
    Define(Defs, "__PTRDIFF_TYPE__", "int");
    Define(Defs, "__UINTMAX_TYPE__", "long long unsigned int");
  }
  Define(Defs, "__INT_MAX__", "2147483647");
  Define(Defs, "__LONG_LONG_MAX__", "9223372036854775807LL");
  Define(Defs, "__CHAR_BIT__", "8");
  Define(Defs, "__SCHAR_MAX__", "127");
  Define(Defs, "__SHRT_MAX__", "32767");
  Define(Defs, "__SIZE_TYPE__", "long unsigned int");
  
  // Subtarget options.
  Define(Defs, "__USER_LABEL_PREFIX__", "_");
  Define(Defs, "__NATURAL_ALIGNMENT__");
  Define(Defs, "__REGISTER_PREFIX__", "");

  Define(Defs, "__WCHAR_MAX__", "2147483647");
  Define(Defs, "__WCHAR_TYPE__", "int");
  Define(Defs, "__WINT_TYPE__", "int");
  
  // Float macros.
  Define(Defs, "__FLT_DENORM_MIN__", "1.40129846e-45F");
  Define(Defs, "__FLT_DIG__", "6");
  Define(Defs, "__FLT_EPSILON__", "1.19209290e-7F");
  Define(Defs, "__FLT_EVAL_METHOD__", "0");
  Define(Defs, "__FLT_HAS_INFINITY__");
  Define(Defs, "__FLT_HAS_QUIET_NAN__");
  Define(Defs, "__FLT_MANT_DIG__", "24");
  Define(Defs, "__FLT_MAX_10_EXP__", "38");
  Define(Defs, "__FLT_MAX_EXP__", "128");
  Define(Defs, "__FLT_MAX__", "3.40282347e+38F");
  Define(Defs, "__FLT_MIN_10_EXP__", "(-37)");
  Define(Defs, "__FLT_MIN_EXP__", "(-125)");
  Define(Defs, "__FLT_MIN__", "1.17549435e-38F");
  Define(Defs, "__FLT_RADIX__", "2");
  
  // double macros.
  Define(Defs, "__DBL_DENORM_MIN__", "4.9406564584124654e-324");
  Define(Defs, "__DBL_DIG__", "15");
  Define(Defs, "__DBL_EPSILON__", "2.2204460492503131e-16");
  Define(Defs, "__DBL_HAS_INFINITY__");
  Define(Defs, "__DBL_HAS_QUIET_NAN__");
  Define(Defs, "__DBL_MANT_DIG__", "53");
  Define(Defs, "__DBL_MAX_10_EXP__", "308");
  Define(Defs, "__DBL_MAX_EXP__", "1024");
  Define(Defs, "__DBL_MAX__", "1.7976931348623157e+308");
  Define(Defs, "__DBL_MIN_10_EXP__", "(-307)");
  Define(Defs, "__DBL_MIN_EXP__", "(-1021)");
  Define(Defs, "__DBL_MIN__", "2.2250738585072014e-308");
  Define(Defs, "__DECIMAL_DIG__", "33");
  
  // 128-bit long double macros.
  Define(Defs, "__LDBL_DENORM_MIN__",
         "4.94065645841246544176568792868221e-324L");
  Define(Defs, "__LDBL_DIG__", "31");
  Define(Defs, "__LDBL_EPSILON__",
         "4.94065645841246544176568792868221e-324L");
  Define(Defs, "__LDBL_HAS_INFINITY__");
  Define(Defs, "__LDBL_HAS_QUIET_NAN__");
  Define(Defs, "__LDBL_MANT_DIG__", "106");
  Define(Defs, "__LDBL_MAX_10_EXP__", "308");
  Define(Defs, "__LDBL_MAX_EXP__", "1024");
  Define(Defs, "__LDBL_MAX__",
         "1.79769313486231580793728971405301e+308L");
  Define(Defs, "__LDBL_MIN_10_EXP__", "(-291)");
  Define(Defs, "__LDBL_MIN_EXP__", "(-968)");
  Define(Defs, "__LDBL_MIN__",
         "2.00416836000897277799610805135016e-292L");
  Define(Defs, "__LONG_DOUBLE_128__");
}

/// getX86Defines - Return a set of the X86-specific #defines that are
/// not tied to a specific subtarget.
static void getX86Defines(std::vector<char> &Defs, bool is64Bit) {
  // Target identification.
  if (is64Bit) {
    Define(Defs, "_LP64");
    Define(Defs, "__LP64__");
    Define(Defs, "__amd64__");
    Define(Defs, "__amd64");
    Define(Defs, "__x86_64");
    Define(Defs, "__x86_64__");
  } else {
    Define(Defs, "__i386__");
    Define(Defs, "__i386");
    Define(Defs, "i386");
  }

  // Target properties.
  Define(Defs, "__LITTLE_ENDIAN__");
  
  if (is64Bit) {
    Define(Defs, "__INTMAX_MAX__", "9223372036854775807L");
    Define(Defs, "__INTMAX_TYPE__", "long int");
    Define(Defs, "__LONG_MAX__", "9223372036854775807L");
    Define(Defs, "__PTRDIFF_TYPE__", "long int");
    Define(Defs, "__UINTMAX_TYPE__", "long unsigned int");
  } else {
    Define(Defs, "__INTMAX_MAX__", "9223372036854775807LL");
    Define(Defs, "__INTMAX_TYPE__", "long long int");
    Define(Defs, "__LONG_MAX__", "2147483647L");
    Define(Defs, "__PTRDIFF_TYPE__", "int");
    Define(Defs, "__UINTMAX_TYPE__", "long long unsigned int");
  }
  Define(Defs, "__CHAR_BIT__", "8");
  Define(Defs, "__INT_MAX__", "2147483647");
  Define(Defs, "__LONG_LONG_MAX__", "9223372036854775807LL");
  Define(Defs, "__SCHAR_MAX__", "127");
  Define(Defs, "__SHRT_MAX__", "32767");
  Define(Defs, "__SIZE_TYPE__", "long unsigned int");
  
  // Subtarget options.
  Define(Defs, "__nocona");
  Define(Defs, "__nocona__");
  Define(Defs, "__tune_nocona__");
  Define(Defs, "__SSE2_MATH__");
  Define(Defs, "__SSE2__");
  Define(Defs, "__SSE_MATH__");
  Define(Defs, "__SSE__");
  Define(Defs, "__MMX__");
  Define(Defs, "__REGISTER_PREFIX__", "");

  Define(Defs, "__WCHAR_MAX__", "2147483647");
  Define(Defs, "__WCHAR_TYPE__", "int");
  Define(Defs, "__WINT_TYPE__", "int");
  
  // Float macros.
  Define(Defs, "__FLT_DENORM_MIN__", "1.40129846e-45F");
  Define(Defs, "__FLT_DIG__", "6");
  Define(Defs, "__FLT_EPSILON__", "1.19209290e-7F");
  Define(Defs, "__FLT_EVAL_METHOD__", "0");
  Define(Defs, "__FLT_HAS_INFINITY__");
  Define(Defs, "__FLT_HAS_QUIET_NAN__");
  Define(Defs, "__FLT_MANT_DIG__", "24");
  Define(Defs, "__FLT_MAX_10_EXP__", "38");
  Define(Defs, "__FLT_MAX_EXP__", "128");
  Define(Defs, "__FLT_MAX__", "3.40282347e+38F");
  Define(Defs, "__FLT_MIN_10_EXP__", "(-37)");
  Define(Defs, "__FLT_MIN_EXP__", "(-125)");
  Define(Defs, "__FLT_MIN__", "1.17549435e-38F");
  Define(Defs, "__FLT_RADIX__", "2");
  
  // Double macros.
  Define(Defs, "__DBL_DENORM_MIN__", "4.9406564584124654e-324");
  Define(Defs, "__DBL_DIG__", "15");
  Define(Defs, "__DBL_EPSILON__", "2.2204460492503131e-16");
  Define(Defs, "__DBL_HAS_INFINITY__");
  Define(Defs, "__DBL_HAS_QUIET_NAN__");
  Define(Defs, "__DBL_MANT_DIG__", "53");
  Define(Defs, "__DBL_MAX_10_EXP__", "308");
  Define(Defs, "__DBL_MAX_EXP__", "1024");
  Define(Defs, "__DBL_MAX__", "1.7976931348623157e+308");
  Define(Defs, "__DBL_MIN_10_EXP__", "(-307)");
  Define(Defs, "__DBL_MIN_EXP__", "(-1021)");
  Define(Defs, "__DBL_MIN__", "2.2250738585072014e-308");
  Define(Defs, "__DECIMAL_DIG__", "21");
  
  // 80-bit Long double macros.
  Define(Defs, "__LDBL_DENORM_MIN__", "3.64519953188247460253e-4951L");
  Define(Defs, "__LDBL_DIG__", "18");
  Define(Defs, "__LDBL_EPSILON__", "1.08420217248550443401e-19L");
  Define(Defs, "__LDBL_HAS_INFINITY__");
  Define(Defs, "__LDBL_HAS_QUIET_NAN__");
  Define(Defs, "__LDBL_MANT_DIG__", "64");
  Define(Defs, "__LDBL_MAX_10_EXP__", "4932");
  Define(Defs, "__LDBL_MAX_EXP__", "16384");
  Define(Defs, "__LDBL_MAX__", "1.18973149535723176502e+4932L");
  Define(Defs, "__LDBL_MIN_10_EXP__", "(-4931)");
  Define(Defs, "__LDBL_MIN_EXP__", "(-16381)");
  Define(Defs, "__LDBL_MIN__", "3.36210314311209350626e-4932L");
}

static const char* getI386VAListDeclaration() {
  return "typedef char* __builtin_va_list;";
}

static const char* getX86_64VAListDeclaration() {
  return 
    "typedef struct __va_list_tag {"
    "  unsigned gp_offset;"
    "  unsigned fp_offset;"
    "  void* overflow_arg_area;"
    "  void* reg_save_area;"
    "} __builtin_va_list[1];";
}

static const char* getPPCVAListDeclaration() {
  return 
    "typedef struct __va_list_tag {"
    "  unsigned char gpr;"
    "  unsigned char fpr;"
    "  unsigned short reserved;"
    "  void* overflow_arg_area;"
    "  void* reg_save_area;"
    "} __builtin_va_list[1];";
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
  virtual void getTargetDefines(std::vector<char> &Defines) const {
    DarwinTargetInfo::getTargetDefines(Defines);
    getPowerPCDefines(Defines, false);
  }
  virtual void getTargetBuiltins(const Builtin::Info *&Records,
                                 unsigned &NumRecords) const {
    PPC::getBuiltins(Records, NumRecords);
  }
  virtual const char *getVAListDeclaration() const {
    return getPPCVAListDeclaration();
  }  
};
} // end anonymous namespace.

namespace {
class DarwinPPC64TargetInfo : public DarwinTargetInfo {
public:
  virtual void getTargetDefines(std::vector<char> &Defines) const {
    DarwinTargetInfo::getTargetDefines(Defines);
    getPowerPCDefines(Defines, true);
  }
  virtual void getTargetBuiltins(const Builtin::Info *&Records,
                                 unsigned &NumRecords) const {
    PPC::getBuiltins(Records, NumRecords);
  }
  virtual const char *getVAListDeclaration() const {
    return getPPCVAListDeclaration();
  }  
};
} // end anonymous namespace.

namespace {
class DarwinI386TargetInfo : public DarwinTargetInfo {
public:
  virtual void getTargetDefines(std::vector<char> &Defines) const {
    DarwinTargetInfo::getTargetDefines(Defines);
    getX86Defines(Defines, false);
  }
  virtual void getTargetBuiltins(const Builtin::Info *&Records,
                                 unsigned &NumRecords) const {
    X86::getBuiltins(Records, NumRecords);
  }
  
  virtual const char *getVAListDeclaration() const {
    return getI386VAListDeclaration();
  }
};
} // end anonymous namespace.

namespace {
class DarwinX86_64TargetInfo : public DarwinTargetInfo {
public:
  virtual void getTargetDefines(std::vector<char> &Defines) const {
    DarwinTargetInfo::getTargetDefines(Defines);
    getX86Defines(Defines, true);
  }
  virtual void getTargetBuiltins(const Builtin::Info *&Records,
                                 unsigned &NumRecords) const {
    X86::getBuiltins(Records, NumRecords);
  }
  virtual const char *getVAListDeclaration() const {
    return getX86_64VAListDeclaration();
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
  
  virtual void getTargetDefines(std::vector<char> &Defines) const {
    // TODO: linux-specific stuff.
    getX86Defines(Defines, false);
  }
  virtual void getTargetBuiltins(const Builtin::Info *&Records,
                                 unsigned &NumRecords) const {
    X86::getBuiltins(Records, NumRecords);
  }
  virtual const char *getVAListDeclaration() const {
    return getI386VAListDeclaration();
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
