//===--- Targets.cpp - Implement -arch option and targets -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements construction of a TargetInfo object from a 
// target triple.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/Builtins.h"
#include "clang/AST/TargetBuiltins.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/APFloat.h"
using namespace clang;

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

static void getSolarisDefines(std::vector<char> &Defs) {
  Define(Defs, "__SUN__");
  Define(Defs, "__SOLARIS__");
}

static void getDarwinDefines(std::vector<char> &Defs, const char *Triple) {
  Define(Defs, "__APPLE__");
  Define(Defs, "__MACH__");
  
  // Figure out which "darwin number" the target triple is.  "darwin9" -> 10.5.
  const char *Darwin = strstr(Triple, "-darwin");
  if (Darwin) {
    char DarwinStr[] = "1000";
    Darwin += strlen("-darwin");
    if (Darwin[0] >= '0' && Darwin[0] <= '9') {
      unsigned DarwinNo = Darwin[0]-'0';
      ++Darwin;
      
      // Handle "darwin11".
      if (DarwinNo == 1 && Darwin[0] >= '0' && Darwin[0] <= '9') {
        DarwinNo = 10+Darwin[0]-'0';
        ++Darwin;
      }
      
      if (DarwinNo >= 4 && DarwinNo <= 13) { // 10.0-10.9
        // darwin7 -> 1030, darwin8 -> 1040, darwin9 -> 1050, etc.
        DarwinStr[2] = '0' + DarwinNo-4;
      }
      
      // Handle minor version: 10.4.9 -> darwin8.9 -> "1049"
      if (Darwin[0] == '.' && Darwin[1] >= '0' && Darwin[1] <= '9' &&
          Darwin[2] == '\0')
        DarwinStr[3] = Darwin[1];
      
    }
    Define(Defs, "__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__", DarwinStr);
  }
}

static void getDragonFlyDefines(std::vector<char> &Defs) {
  // DragonFly defines; list based off of gcc output
  Define(Defs, "__DragonFly__");
  Define(Defs, "__DragonFly_cc_version", "100001");
  Define(Defs, "__ELF__");
  Define(Defs, "__KPRINTF_ATTRIBUTE__");
  Define(Defs, "__tune_i386__");
  Define(Defs, "unix");
  Define(Defs, "__unix");
  Define(Defs, "__unix__");
}

static void getLinuxDefines(std::vector<char> &Defs) {
  // Linux defines; list based off of gcc output
  Define(Defs, "__unix__");
  Define(Defs, "__unix");
  Define(Defs, "unix");
  Define(Defs, "__linux__");
  Define(Defs, "__linux");
  Define(Defs, "linux");
  Define(Defs, "__gnu_linux__");
}

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

  // Subtarget options.
  Define(Defs, "__NATURAL_ALIGNMENT__");
  Define(Defs, "__REGISTER_PREFIX__", "");

  // FIXME: Should be controlled by command line option.
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
}

/// getARMDefines - Return a set of the ARM-specific #defines that are
/// not tied to a specific subtarget.
static void getARMDefines(std::vector<char> &Defs) {
  // Target identification.
  Define(Defs, "__arm");
  Define(Defs, "__arm__");
  
  // Target properties.
  Define(Defs, "__LITTLE_ENDIAN__");
  
  // Subtarget options.  [hard coded to v6 for now]
  Define(Defs, "__ARM_ARCH_6K__");
  Define(Defs, "__ARMEL__");
  Define(Defs, "__ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__", "20000");
}

//===----------------------------------------------------------------------===//
// Specific target implementations.
//===----------------------------------------------------------------------===//

namespace {
// PPC abstract base class
class PPCTargetInfo : public TargetInfo {
  static const Builtin::Info BuiltinInfo[];
  static const char * const GCCRegNames[];
  static const TargetInfo::GCCRegAlias GCCRegAliases[];

public:
  PPCTargetInfo(const std::string& triple) : TargetInfo(triple) {
    CharIsSigned = false;
  }
  virtual void getTargetBuiltins(const Builtin::Info *&Records,
                                 unsigned &NumRecords) const {
    Records = BuiltinInfo;
    NumRecords = clang::PPC::LastTSBuiltin-Builtin::FirstTSBuiltin;
  }
  virtual const char *getVAListDeclaration() const {
    return "typedef struct __va_list_tag {"
           "  unsigned char gpr;"
           "  unsigned char fpr;"
           "  unsigned short reserved;"
           "  void* overflow_arg_area;"
           "  void* reg_save_area;"
           "} __builtin_va_list[1];";
  }
  virtual const char *getTargetPrefix() const {
    return "ppc";
  }
  virtual void getGCCRegNames(const char * const *&Names, 
                              unsigned &NumNames) const;
  virtual void getGCCRegAliases(const GCCRegAlias *&Aliases, 
                                unsigned &NumAliases) const;
  virtual bool validateAsmConstraint(char c,
                                     TargetInfo::ConstraintInfo &info) const {
    switch (c) {
    default: return false;
    case 'O': // Zero
      return true;
    case 'b': // Base register
    case 'f': // Floating point register
      info = (TargetInfo::ConstraintInfo)(info|TargetInfo::CI_AllowsRegister);
      return true;
    }
  }
  virtual const char *getClobbers() const {
    return "";
  }
};

const Builtin::Info PPCTargetInfo::BuiltinInfo[] = {
#define BUILTIN(ID, TYPE, ATTRS) { #ID, TYPE, ATTRS },
#include "clang/AST/PPCBuiltins.def"
};

const char * const PPCTargetInfo::GCCRegNames[] = {
  "0", "1", "2", "3", "4", "5", "6", "7",
  "8", "9", "10", "11", "12", "13", "14", "15",
  "16", "17", "18", "19", "20", "21", "22", "23",
  "24", "25", "26", "27", "28", "29", "30", "31",
  "0", "1", "2", "3", "4", "5", "6", "7",
  "8", "9", "10", "11", "12", "13", "14", "15",
  "16", "17", "18", "19", "20", "21", "22", "23",
  "24", "25", "26", "27", "28", "29", "30", "31",
  "mq", "lr", "ctr", "ap",
  "0", "1", "2", "3", "4", "5", "6", "7",
  "xer",
  "0", "1", "2", "3", "4", "5", "6", "7",
  "8", "9", "10", "11", "12", "13", "14", "15",
  "16", "17", "18", "19", "20", "21", "22", "23",
  "24", "25", "26", "27", "28", "29", "30", "31",
  "vrsave", "vscr",
  "spe_acc", "spefscr",
  "sfp"
};

void PPCTargetInfo::getGCCRegNames(const char * const *&Names, 
                                   unsigned &NumNames) const {
  Names = GCCRegNames;
  NumNames = llvm::array_lengthof(GCCRegNames);
}

const TargetInfo::GCCRegAlias PPCTargetInfo::GCCRegAliases[] = {
  // While some of these aliases do map to different registers
  // they still share the same register name.
  { { "cc", "cr0", "fr0", "r0", "v0"}, "0" }, 
  { { "cr1", "fr1", "r1", "sp", "v1"}, "1" }, 
  { { "cr2", "fr2", "r2", "toc", "v2"}, "2" }, 
  { { "cr3", "fr3", "r3", "v3"}, "3" }, 
  { { "cr4", "fr4", "r4", "v4"}, "4" }, 
  { { "cr5", "fr5", "r5", "v5"}, "5" }, 
  { { "cr6", "fr6", "r6", "v6"}, "6" }, 
  { { "cr7", "fr7", "r7", "v7"}, "7" }, 
  { { "fr8", "r8", "v8"}, "8" }, 
  { { "fr9", "r9", "v9"}, "9" }, 
  { { "fr10", "r10", "v10"}, "10" }, 
  { { "fr11", "r11", "v11"}, "11" }, 
  { { "fr12", "r12", "v12"}, "12" }, 
  { { "fr13", "r13", "v13"}, "13" }, 
  { { "fr14", "r14", "v14"}, "14" }, 
  { { "fr15", "r15", "v15"}, "15" }, 
  { { "fr16", "r16", "v16"}, "16" }, 
  { { "fr17", "r17", "v17"}, "17" }, 
  { { "fr18", "r18", "v18"}, "18" }, 
  { { "fr19", "r19", "v19"}, "19" }, 
  { { "fr20", "r20", "v20"}, "20" }, 
  { { "fr21", "r21", "v21"}, "21" }, 
  { { "fr22", "r22", "v22"}, "22" }, 
  { { "fr23", "r23", "v23"}, "23" }, 
  { { "fr24", "r24", "v24"}, "24" }, 
  { { "fr25", "r25", "v25"}, "25" }, 
  { { "fr26", "r26", "v26"}, "26" }, 
  { { "fr27", "r27", "v27"}, "27" }, 
  { { "fr28", "r28", "v28"}, "28" }, 
  { { "fr29", "r29", "v29"}, "29" }, 
  { { "fr30", "r30", "v30"}, "30" }, 
  { { "fr31", "r31", "v31"}, "31" }, 
};

void PPCTargetInfo::getGCCRegAliases(const GCCRegAlias *&Aliases, 
                                     unsigned &NumAliases) const {
  Aliases = GCCRegAliases;
  NumAliases = llvm::array_lengthof(GCCRegAliases);
}
} // end anonymous namespace.

namespace {
class PPC32TargetInfo : public PPCTargetInfo {
public:
  PPC32TargetInfo(const std::string& triple) : PPCTargetInfo(triple) {
    DescriptionString = "E-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-"
                        "i64:64:64-f32:32:32-f64:64:64-v128:128:128";
  }
  virtual void getTargetDefines(std::vector<char> &Defines) const {
    getPowerPCDefines(Defines, false);
  }
};
} // end anonymous namespace.

namespace {
class PPC64TargetInfo : public PPCTargetInfo {
public:
  PPC64TargetInfo(const std::string& triple) : PPCTargetInfo(triple) {
    LongWidth = LongAlign = PointerWidth = PointerAlign = 64;
    DescriptionString = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-"
                        "i64:64:64-f32:32:32-f64:64:64-v128:128:128";
  }
  virtual void getTargetDefines(std::vector<char> &Defines) const {
    getPowerPCDefines(Defines, true);
  }
};
} // end anonymous namespace.

namespace {
class DarwinPPCTargetInfo : public PPC32TargetInfo {
public:
  DarwinPPCTargetInfo(const std::string& triple) : PPC32TargetInfo(triple) {}
  virtual void getTargetDefines(std::vector<char> &Defines) const {
    PPC32TargetInfo::getTargetDefines(Defines);
    getDarwinDefines(Defines, getTargetTriple());
  }

  virtual bool useNeXTRuntimeAsDefault() const { return true; }
};
} // end anonymous namespace.

namespace {
class DarwinPPC64TargetInfo : public PPC64TargetInfo {
public:
  DarwinPPC64TargetInfo(const std::string& triple) : PPC64TargetInfo(triple) {}
  virtual void getTargetDefines(std::vector<char> &Defines) const {
    PPC64TargetInfo::getTargetDefines(Defines);
    getDarwinDefines(Defines, getTargetTriple());
  }

  virtual bool useNeXTRuntimeAsDefault() const { return true; }
};
} // end anonymous namespace.

namespace {
// Namespace for x86 abstract base class
const Builtin::Info BuiltinInfo[] = {
#define BUILTIN(ID, TYPE, ATTRS) { #ID, TYPE, ATTRS },
#include "clang/AST/X86Builtins.def"
};

const char *GCCRegNames[] = {
  "ax", "dx", "cx", "bx", "si", "di", "bp", "sp",
  "st", "st(1)", "st(2)", "st(3)", "st(4)", "st(5)", "st(6)", "st(7)",
  "argp", "flags", "fspr", "dirflag", "frame",
  "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7",
  "mm0", "mm1", "mm2", "mm3", "mm4", "mm5", "mm6", "mm7",
  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
  "xmm8", "xmm9", "xmm10", "xmm11", "xmm12", "xmm13", "xmm14", "xmm15"
};

const TargetInfo::GCCRegAlias GCCRegAliases[] = {
  { { "al", "ah", "eax", "rax" }, "ax" },
  { { "bl", "bh", "ebx", "rbx" }, "bx" },
  { { "cl", "ch", "ecx", "rcx" }, "cx" },
  { { "dl", "dh", "edx", "rdx" }, "dx" },
  { { "esi", "rsi" }, "si" },
  { { "edi", "rdi" }, "di" },
  { { "esp", "rsp" }, "sp" },
  { { "ebp", "rbp" }, "bp" },
};

// X86 target abstract base class; x86-32 and x86-64 are very close, so
// most of the implementation can be shared.
class X86TargetInfo : public TargetInfo {
public:
  X86TargetInfo(const std::string& triple) : TargetInfo(triple) {
    LongDoubleFormat = &llvm::APFloat::x87DoubleExtended;
  }
  virtual void getTargetBuiltins(const Builtin::Info *&Records,
                                 unsigned &NumRecords) const {
    Records = BuiltinInfo;
    NumRecords = clang::X86::LastTSBuiltin-Builtin::FirstTSBuiltin;
  }
  virtual const char *getTargetPrefix() const {
    return "x86";
  }  
  virtual void getGCCRegNames(const char * const *&Names, 
                              unsigned &NumNames) const {
    Names = GCCRegNames;
    NumNames = llvm::array_lengthof(GCCRegNames);
  }
  virtual void getGCCRegAliases(const GCCRegAlias *&Aliases, 
                                unsigned &NumAliases) const {
    Aliases = GCCRegAliases;
    NumAliases = llvm::array_lengthof(GCCRegAliases);
  }
  virtual bool validateAsmConstraint(char c,
                                     TargetInfo::ConstraintInfo &info) const;
  virtual std::string convertConstraint(const char Constraint) const;
  virtual const char *getClobbers() const {
    return "~{dirflag},~{fpsr},~{flags}";
  }
};
  
bool
X86TargetInfo::validateAsmConstraint(char c,
                                     TargetInfo::ConstraintInfo &info) const {
  switch (c) {
  default: return false;
  case 'a': // eax.
  case 'b': // ebx.
  case 'c': // ecx.
  case 'd': // edx.
  case 'S': // esi.
  case 'D': // edi.
  case 'A': // edx:eax.
  case 't': // top of floating point stack.
  case 'u': // second from top of floating point stack.
  case 'q': // Any register accessible as [r]l: a, b, c, and d.
  case 'Q': // Any register accessible as [r]h: a, b, c, and d.
  case 'Z': // 32-bit integer constant for use with zero-extending x86_64
            // instructions.
  case 'N': // unsigned 8-bit integer constant for use with in and out
            // instructions.
    info = (TargetInfo::ConstraintInfo)(info|TargetInfo::CI_AllowsRegister);
    return true;
  }
}

std::string
X86TargetInfo::convertConstraint(const char Constraint) const {
  switch (Constraint) {
  case 'a': return std::string("{ax}");
  case 'b': return std::string("{bx}");
  case 'c': return std::string("{cx}");
  case 'd': return std::string("{dx}");
  case 'S': return std::string("{si}");
  case 'D': return std::string("{di}");
  case 't': // top of floating point stack.
    return std::string("{st}");
  case 'u': // second from top of floating point stack.
    return std::string("{st(1)}"); // second from top of floating point stack.
  default:
    return std::string(1, Constraint);
  }
}
} // end anonymous namespace

namespace {
// X86-32 generic target
class X86_32TargetInfo : public X86TargetInfo {
public:
  X86_32TargetInfo(const std::string& triple) : X86TargetInfo(triple) {
    DoubleAlign = LongLongAlign = 32;
    LongDoubleWidth = 96;
    LongDoubleAlign = 32;
    DescriptionString = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-"
                        "i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-"
                        "a0:0:64-f80:32:32";
  }
  virtual const char *getVAListDeclaration() const {
    return "typedef char* __builtin_va_list;";
  }
  virtual void getTargetDefines(std::vector<char> &Defines) const {
    getX86Defines(Defines, false);
  }
};
} // end anonymous namespace

namespace {
// x86-32 Darwin (OS X) target
class DarwinI386TargetInfo : public X86_32TargetInfo {
public:
  DarwinI386TargetInfo(const std::string& triple) : X86_32TargetInfo(triple) {
    LongDoubleWidth = 128;
    LongDoubleAlign = 128;
    DescriptionString = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-"
                        "i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-"
                        "a0:0:64-f80:128:128";
  }
  virtual void getTargetDefines(std::vector<char> &Defines) const {
    X86_32TargetInfo::getTargetDefines(Defines);
    getDarwinDefines(Defines, getTargetTriple());
  }
  virtual bool useNeXTRuntimeAsDefault() const { return true; }
};
} // end anonymous namespace

namespace {
// x86-32 DragonFly target
class DragonFlyX86_32TargetInfo : public X86_32TargetInfo {
public:
  DragonFlyX86_32TargetInfo(const std::string& triple) : X86_32TargetInfo(triple) {
  }
  virtual void getTargetDefines(std::vector<char> &Defines) const {
    X86_32TargetInfo::getTargetDefines(Defines);
    getDragonFlyDefines(Defines);
  }
};
} // end anonymous namespace

namespace {
// x86-32 Linux target
class LinuxX86_32TargetInfo : public X86_32TargetInfo {
public:
  LinuxX86_32TargetInfo(const std::string& triple) : X86_32TargetInfo(triple) {
    UserLabelPrefix = "";
  }
  virtual void getTargetDefines(std::vector<char> &Defines) const {
    X86_32TargetInfo::getTargetDefines(Defines);
    getLinuxDefines(Defines);
  }
};
} // end anonymous namespace

namespace {
// x86-32 Windows target
class WindowsX86_32TargetInfo : public X86_32TargetInfo {
public:
  WindowsX86_32TargetInfo(const std::string& triple)
    : X86_32TargetInfo(triple) {
    // FIXME: Fix wchar_t.
    // FIXME: We should probably enable -fms-extensions by default for
    // this target.
  }
  virtual void getTargetDefines(std::vector<char> &Defines) const {
    X86_32TargetInfo::getTargetDefines(Defines);
    // This list is based off of the the list of things MingW defines
    Define(Defines, "__WIN32__");
    Define(Defines, "__WIN32");
    Define(Defines, "_WIN32");
    Define(Defines, "WIN32");
    Define(Defines, "__WINNT__");
    Define(Defines, "__WINNT");
    Define(Defines, "WINNT");
    Define(Defines, "_X86_");
    Define(Defines, "__MSVCRT__");
  }
};
} // end anonymous namespace

namespace {
// x86-64 generic target
class X86_64TargetInfo : public X86TargetInfo {
public:
  X86_64TargetInfo(const std::string& triple) : X86TargetInfo(triple) {
    LongWidth = LongAlign = PointerWidth = PointerAlign = 64;
    LongDoubleWidth = 128;
    LongDoubleAlign = 128;
    DescriptionString = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-"
                        "i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-"
                        "a0:0:64-f80:128:128";
  }
  virtual const char *getVAListDeclaration() const {
    return "typedef struct __va_list_tag {"
           "  unsigned gp_offset;"
           "  unsigned fp_offset;"
           "  void* overflow_arg_area;"
           "  void* reg_save_area;"
           "} __builtin_va_list[1];";
  }
  virtual void getTargetDefines(std::vector<char> &Defines) const {
    getX86Defines(Defines, true);
  }
};
} // end anonymous namespace

namespace {
// x86-64 Linux target
class LinuxX86_64TargetInfo : public X86_64TargetInfo {
public:
  LinuxX86_64TargetInfo(const std::string& triple) : X86_64TargetInfo(triple) {
    UserLabelPrefix = "";
  }
  virtual void getTargetDefines(std::vector<char> &Defines) const {
    X86_64TargetInfo::getTargetDefines(Defines);
    getLinuxDefines(Defines);
  }
};
} // end anonymous namespace

namespace {
// x86-64 Darwin (OS X) target
class DarwinX86_64TargetInfo : public X86_64TargetInfo {
public:
  DarwinX86_64TargetInfo(const std::string& triple) :
    X86_64TargetInfo(triple) {}

  virtual void getTargetDefines(std::vector<char> &Defines) const {
    X86_64TargetInfo::getTargetDefines(Defines);
    getDarwinDefines(Defines, getTargetTriple());
  }

  virtual bool useNeXTRuntimeAsDefault() const { return true; }
};
} // end anonymous namespace.

namespace {
class ARMTargetInfo : public TargetInfo {
public:
  ARMTargetInfo(const std::string& triple) : TargetInfo(triple) {
    // FIXME: Are the defaults correct for ARM?
    DescriptionString = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-"
                        "i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:64:64";
  }
  virtual void getTargetDefines(std::vector<char> &Defines) const {
    getARMDefines(Defines);
  }
  virtual void getTargetBuiltins(const Builtin::Info *&Records,
                                 unsigned &NumRecords) const {
    // FIXME: Implement.
    Records = 0;
    NumRecords = 0;
  }
  virtual const char *getVAListDeclaration() const {
    return "typedef char* __builtin_va_list;";
  }
  virtual const char *getTargetPrefix() const {
    return "arm";
  }
  virtual void getGCCRegNames(const char * const *&Names, 
                              unsigned &NumNames) const {
    // FIXME: Implement.
    Names = 0;
    NumNames = 0;
  }
  virtual void getGCCRegAliases(const GCCRegAlias *&Aliases, 
                                unsigned &NumAliases) const {
    // FIXME: Implement.
    Aliases = 0;
    NumAliases = 0;
  }
  virtual bool validateAsmConstraint(char c,
                                     TargetInfo::ConstraintInfo &info) const {
    // FIXME: Check if this is complete
    switch (c) {
    default:
    case 'l': // r0-r7
    case 'h': // r8-r15
    case 'w': // VFP Floating point register single precision
    case 'P': // VFP Floating point register double precision
      info = (TargetInfo::ConstraintInfo)(info|TargetInfo::CI_AllowsRegister);
      return true;
    }
    return false;
  }
  virtual const char *getClobbers() const {
    // FIXME: Is this really right?
    return "";
  }
};
} // end anonymous namespace.


namespace {
class DarwinARMTargetInfo : public ARMTargetInfo {
public:
  DarwinARMTargetInfo(const std::string& triple) : ARMTargetInfo(triple) {}

  virtual void getTargetDefines(std::vector<char> &Defines) const {
    ARMTargetInfo::getTargetDefines(Defines);
    getDarwinDefines(Defines, getTargetTriple());
  }
};
} // end anonymous namespace.

namespace {
class SparcV8TargetInfo : public TargetInfo {
public:
  SparcV8TargetInfo(const std::string& triple) : TargetInfo(triple) {
    // FIXME: Support Sparc quad-precision long double?
    DescriptionString = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-"
                        "i64:64:64-f32:32:32-f64:64:64-v64:64:64";
  }
  virtual void getTargetDefines(std::vector<char> &Defines) const {
    // FIXME: This is missing a lot of important defines; some of the
    // missing stuff is likely to break system headers.
    Define(Defines, "__sparc");
    Define(Defines, "__sparc__");
    Define(Defines, "__sparcv8");
  }
  virtual void getTargetBuiltins(const Builtin::Info *&Records,
                                 unsigned &NumRecords) const {
    // FIXME: Implement!
  }
  virtual const char *getVAListDeclaration() const {
    return "typedef void* __builtin_va_list;";
  }
  virtual const char *getTargetPrefix() const {
    return "sparc";
  }
  virtual void getGCCRegNames(const char * const *&Names, 
                              unsigned &NumNames) const {
    // FIXME: Implement!
    Names = 0;
    NumNames = 0;
  }
  virtual void getGCCRegAliases(const GCCRegAlias *&Aliases, 
                                unsigned &NumAliases) const {
    // FIXME: Implement!
    Aliases = 0;
    NumAliases = 0;
  }
  virtual bool validateAsmConstraint(char c,
                                     TargetInfo::ConstraintInfo &info) const {
    // FIXME: Implement!
    return false;
  }
  virtual const char *getClobbers() const {
    // FIXME: Implement!
    return "";
  }
};

} // end anonymous namespace.

namespace {
class SolarisSparcV8TargetInfo : public SparcV8TargetInfo {
public:
  SolarisSparcV8TargetInfo(const std::string& triple) :
    SparcV8TargetInfo(triple) {}

  virtual void getTargetDefines(std::vector<char> &Defines) const {
    SparcV8TargetInfo::getTargetDefines(Defines);
    getSolarisDefines(Defines);
  }
};
} // end anonymous namespace.

namespace {
  class PIC16TargetInfo : public TargetInfo{
  public:
    PIC16TargetInfo(const std::string& triple) : TargetInfo(triple) {
      // FIXME: Is IntAlign really supposed to be 16?  There seems
      // little point on a platform with 8-bit loads.
      IntWidth = IntAlign = LongAlign = LongLongAlign = PointerWidth = 16;
      LongWidth = 16;
      PointerAlign = 8;
      DescriptionString = "e-p:16:8:8-i8:8:8-i16:8:8-i32:8:8";
    }
    virtual uint64_t getPointerWidthV(unsigned AddrSpace) const { return 16; }
    virtual uint64_t getPointerAlignV(unsigned AddrSpace) const { return 8; }
    virtual void getTargetDefines(std::vector<char> &Defines) const {
      Define(Defines, "__pic16");
    }
    virtual void getTargetBuiltins(const Builtin::Info *&Records,
                                   unsigned &NumRecords) const {}
    virtual const char *getVAListDeclaration() const { return "";}
    virtual const char *getClobbers() const {return "";}
    virtual const char *getTargetPrefix() const {return "";}
    virtual void getGCCRegNames(const char * const *&Names, 
                                unsigned &NumNames) const {} 
    virtual bool validateAsmConstraint(char c, 
                                       TargetInfo::ConstraintInfo &info) const {
      return true;
    }
    virtual void getGCCRegAliases(const GCCRegAlias *&Aliases, 
                                  unsigned &NumAliases) const {}
    virtual bool useGlobalsForAutomaticVariables() const {return true;}
  };
}

//===----------------------------------------------------------------------===//
// Driver code
//===----------------------------------------------------------------------===//

static inline bool IsX86(const std::string& TT) {
  return (TT.size() >= 5 && TT[0] == 'i' && TT[2] == '8' && TT[3] == '6' &&
          TT[4] == '-' && TT[1] - '3' < 6);
}

/// CreateTargetInfo - Return the target info object for the specified target
/// triple.
TargetInfo* TargetInfo::CreateTargetInfo(const std::string &T) {
  // OS detection; this isn't really anywhere near complete.
  // Additions and corrections are welcome.
  bool isDarwin = T.find("-darwin") != std::string::npos;
  bool isDragonFly = T.find("-dragonfly") != std::string::npos;
  bool isSolaris = T.find("-solaris") != std::string::npos;
  bool isLinux = T.find("-linux") != std::string::npos;
  bool isWindows = T.find("-windows") != std::string::npos ||
                   T.find("-win32") != std::string::npos ||
                   T.find("-mingw") != std::string::npos;

  if (T.find("ppc-") == 0 || T.find("powerpc-") == 0) {
    if (isDarwin)
      return new DarwinPPCTargetInfo(T);
    return new PPC32TargetInfo(T);
  }

  if (T.find("ppc64-") == 0 || T.find("powerpc64-") == 0) {
    if (isDarwin)
      return new DarwinPPC64TargetInfo(T);
    return new PPC64TargetInfo(T);
  }

  if (T.find("armv6-") == 0 || T.find("arm-") == 0) {
    if (isDarwin)
      return new DarwinARMTargetInfo(T);
    return new ARMTargetInfo(T);
  }

  if (T.find("sparc-") == 0) {
    if (isSolaris)
      return new SolarisSparcV8TargetInfo(T);
    return new SparcV8TargetInfo(T);
  }

  if (T.find("x86_64-") == 0) {
    if (isDarwin)
      return new DarwinX86_64TargetInfo(T);
    if (isLinux)
      return new LinuxX86_64TargetInfo(T);
    return new X86_64TargetInfo(T);
  }

  if (T.find("pic16-") == 0)
    return new PIC16TargetInfo(T);

  if (IsX86(T)) {
    if (isDarwin)
      return new DarwinI386TargetInfo(T);
    if (isLinux)
      return new LinuxX86_32TargetInfo(T);
    if (isDragonFly)
      return new DragonFlyX86_32TargetInfo(T);
    if (isWindows)
      return new WindowsX86_32TargetInfo(T);
    return new X86_32TargetInfo(T);
  }

  return NULL;
}

