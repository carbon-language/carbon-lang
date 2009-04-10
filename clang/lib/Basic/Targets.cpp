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

// FIXME: Layering violation
#include "clang/AST/Builtins.h"
#include "clang/AST/TargetBuiltins.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/LangOptions.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallString.h"
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

/// DefineStd - Define a macro name and standard variants.  For example if
/// MacroName is "unix", then this will define "__unix", "__unix__", and "unix"
/// when in GNU mode.
static void DefineStd(std::vector<char> &Buf, const char *MacroName, 
                      const LangOptions &Opts) {
  assert(MacroName[0] != '_' && "Identifier should be in the user's namespace");
  
  // If in GNU mode (e.g. -std=gnu99 but not -std=c99) define the raw identifier
  // in the user's namespace.
  if (Opts.GNUMode)
    Define(Buf, MacroName);
  
  // Define __unix.
  llvm::SmallString<20> TmpStr;
  TmpStr = "__";
  TmpStr += MacroName;
  Define(Buf, TmpStr.c_str());
  
  // Define __unix__.
  TmpStr += "__";
  Define(Buf, TmpStr.c_str());
}

//===----------------------------------------------------------------------===//
// Defines specific to certain operating systems.
//===----------------------------------------------------------------------===//

static void getSolarisDefines(std::vector<char> &Defs) {
  Define(Defs, "__SUN__");
  Define(Defs, "__SOLARIS__");
}

static void getFreeBSDDefines(const LangOptions &Opts, bool is64Bit,
                              const char *Triple, std::vector<char> &Defs) {
  // FreeBSD defines; list based off of gcc output

  const char *FreeBSD = strstr(Triple, "-freebsd");
  FreeBSD += strlen("-freebsd");
  char release[] = "X";
  release[0] = FreeBSD[0];
  char version[] = "X00001";
  version[0] = FreeBSD[0];

  Define(Defs, "__FreeBSD__", release);
  Define(Defs, "__FreeBSD_cc_version", version);
  Define(Defs, "__KPRINTF_ATTRIBUTE__");
  DefineStd(Defs, "unix", Opts);
  Define(Defs, "__ELF__", "1");
  if (is64Bit) {
    Define(Defs, "__LP64__");
  }
}

static void getDragonFlyDefines(const LangOptions &Opts,
                                std::vector<char> &Defs) {
  // DragonFly defines; list based off of gcc output
  Define(Defs, "__DragonFly__");
  Define(Defs, "__DragonFly_cc_version", "100001");
  Define(Defs, "__ELF__");
  Define(Defs, "__KPRINTF_ATTRIBUTE__");
  Define(Defs, "__tune_i386__");
  DefineStd(Defs, "unix", Opts);
}

static void getLinuxDefines(const LangOptions &Opts, std::vector<char> &Defs) {
  // Linux defines; list based off of gcc output
  DefineStd(Defs, "unix", Opts);
  DefineStd(Defs, "linux", Opts);
  Define(Defs, "__gnu_linux__");
  Define(Defs, "__ELF__", "1");
}

/// getDarwinNumber - Parse the 'darwin number' out of the specific targe
/// triple.  For example, if we have darwin8.5 return 8,5,0.  If any entry is
/// not defined, return 0's.  Return true if we have -darwin in the string or
/// false otherwise.
static bool getDarwinNumber(const char *Triple, unsigned &Maj, unsigned &Min, unsigned &Revision) {
  Maj = Min = Revision = 0;
  const char *Darwin = strstr(Triple, "-darwin");
  if (Darwin == 0) return false;
  
  Darwin += strlen("-darwin");
  if (Darwin[0] < '0' || Darwin[0] > '9')
    return true;
  
  Maj = Darwin[0]-'0';
  ++Darwin;
    
  // Handle "darwin11".
  if (Maj == 1 && Darwin[0] >= '0' && Darwin[0] <= '9') {
    Maj = Maj*10 + (Darwin[0] - '0');
    ++Darwin;
  }
    
  // Handle minor version: 10.4.9 -> darwin8.9 -> "1049"
  if (Darwin[0] != '.')
    return true;
  
  ++Darwin;
  if (Darwin[0] < '0' || Darwin[0] > '9')
    return true;
  
  Min = Darwin[0]-'0';
  ++Darwin;

  // Handle 10.4.11 -> darwin8.11
  if (Min == 1 && Darwin[0] >= '0' && Darwin[0] <= '9') {
    Min = Min*10 + (Darwin[0] - '0');
    ++Darwin;
  }
  
  // Handle revision darwin8.9.1
  if (Darwin[0] != '.')
    return true;
  
  ++Darwin;
  if (Darwin[0] < '0' || Darwin[0] > '9')
    return true;
  
  Revision = Darwin[0]-'0';
  ++Darwin;
  
  if (Revision == 1 && Darwin[0] >= '0' && Darwin[0] <= '9') {
    Revision = Revision*10 + (Darwin[0] - '0');
    ++Darwin;
  }
  
  return true;
}

static void getDarwinDefines(std::vector<char> &Defs, const LangOptions &Opts) {
  Define(Defs, "__APPLE__");
  Define(Defs, "__MACH__");
  Define(Defs, "OBJC_NEW_PROPERTIES");
  
  // __weak is always defined, for use in blocks and with objc pointers.
  Define(Defs, "__weak", "__attribute__((objc_gc(weak)))");
  
  // Darwin defines __strong even in C mode (just to nothing).
  if (!Opts.ObjC1 || Opts.getGCMode() == LangOptions::NonGC)
    Define(Defs, "__strong", "");
  else
    Define(Defs, "__strong", "__attribute__((objc_gc(strong)))");
}

static void getDarwinOSXDefines(std::vector<char> &Defs, const char *Triple) {
  // Figure out which "darwin number" the target triple is.  "darwin9" -> 10.5.
  unsigned Maj, Min, Rev;
  if (getDarwinNumber(Triple, Maj, Min, Rev)) {
    char MacOSXStr[] = "1000";
    if (Maj >= 4 && Maj <= 13) { // 10.0-10.9
      // darwin7 -> 1030, darwin8 -> 1040, darwin9 -> 1050, etc.
      MacOSXStr[2] = '0' + Maj-4;
    }
        
    // Handle minor version: 10.4.9 -> darwin8.9 -> "1049"
    // Cap 10.4.11 -> darwin8.11 -> "1049"
    MacOSXStr[3] = std::min(Min, 9U)+'0';
    Define(Defs, "__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__", MacOSXStr);
  }
}

static void getDarwinIPhoneOSDefines(std::vector<char> &Defs, 
                                     const char *Triple) {
  // Figure out which "darwin number" the target triple is.  "darwin9" -> 10.5.
  unsigned Maj, Min, Rev;
  if (getDarwinNumber(Triple, Maj, Min, Rev)) {
    // When targetting iPhone OS, interpret the minor version and
    // revision as the iPhone OS version
    char iPhoneOSStr[] = "10000";
    if (Min >= 2 && Min <= 9) { // iPhone OS 2.0-9.0
      // darwin9.2.0 -> 20000, darwin9.3.0 -> 30000, etc.
      iPhoneOSStr[0] = '0' + Min;
    }
      
    // Handle minor version: 2.2 -> darwin9.2.2 -> 20200
    iPhoneOSStr[2] = std::min(Rev, 9U)+'0';
    Define(Defs, "__ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__", 
           iPhoneOSStr);
  }
}

/// GetDarwinLanguageOptions - Set the default language options for darwin.
static void GetDarwinLanguageOptions(LangOptions &Opts,
                                     const char *Triple) {
  Opts.NeXTRuntime = true;
  
  unsigned Maj, Min, Rev;
  if (!getDarwinNumber(Triple, Maj, Min, Rev))
    return;
  
  // Blocks default to on for 10.6 (darwin10) and beyond.
  // As does nonfragile-abi for 64bit mode
  if (Maj > 9) 
    Opts.Blocks = 1;

  if (Maj >= 9 && Opts.ObjC1 && !strncmp(Triple, "x86_64", 6))
    Opts.ObjCNonFragileABI = 1;
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
  
  virtual void getTargetDefines(const LangOptions &Opts,
                                std::vector<char> &Defines) const;
  
  virtual const char *getVAListDeclaration() const {
    return "typedef char* __builtin_va_list;";
    // This is the right definition for ABI/V4: System V.4/eabi.
    /*return "typedef struct __va_list_tag {"
           "  unsigned char gpr;"
           "  unsigned char fpr;"
           "  unsigned short reserved;"
           "  void* overflow_arg_area;"
           "  void* reg_save_area;"
           "} __builtin_va_list[1];";*/
  }
  virtual const char *getTargetPrefix() const {
    return "ppc";
  }
  virtual void getGCCRegNames(const char * const *&Names, 
                              unsigned &NumNames) const;
  virtual void getGCCRegAliases(const GCCRegAlias *&Aliases, 
                                unsigned &NumAliases) const;
  virtual bool validateAsmConstraint(const char *&Name,
                                     TargetInfo::ConstraintInfo &info) const {
    switch (*Name) {
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
#define BUILTIN(ID, TYPE, ATTRS) { #ID, TYPE, ATTRS, 0, false },
#define LIBBUILTIN(ID, TYPE, ATTRS, HEADER) { #ID, TYPE, ATTRS, HEADER, false },
#include "clang/AST/PPCBuiltins.def"
};
  
  
/// PPCTargetInfo::getTargetDefines - Return a set of the PowerPC-specific
/// #defines that are not tied to a specific subtarget.
void PPCTargetInfo::getTargetDefines(const LangOptions &Opts,
                                     std::vector<char> &Defs) const {
  // Target identification.
  Define(Defs, "__ppc__");
  Define(Defs, "_ARCH_PPC");
  Define(Defs, "__POWERPC__");
  if (PointerWidth == 64) {
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
};
} // end anonymous namespace.


namespace {
class DarwinPPCTargetInfo : public PPC32TargetInfo {
public:
  DarwinPPCTargetInfo(const std::string& triple) : PPC32TargetInfo(triple) {}
  virtual void getTargetDefines(const LangOptions &Opts,
                                std::vector<char> &Defines) const {
    PPC32TargetInfo::getTargetDefines(Opts, Defines);
    getDarwinDefines(Defines, Opts);
    getDarwinOSXDefines(Defines, getTargetTriple());
  }

  /// getDefaultLangOptions - Allow the target to specify default settings for
  /// various language options.  These may be overridden by command line
  /// options. 
  virtual void getDefaultLangOptions(LangOptions &Opts) {
    GetDarwinLanguageOptions(Opts, getTargetTriple());
  }
};
} // end anonymous namespace.

namespace {
class DarwinPPC64TargetInfo : public PPC64TargetInfo {
public:
  DarwinPPC64TargetInfo(const std::string& triple) : PPC64TargetInfo(triple) {}
  virtual void getTargetDefines(const LangOptions &Opts,
                                std::vector<char> &Defines) const {
    PPC64TargetInfo::getTargetDefines(Opts, Defines);
    getDarwinDefines(Defines, Opts);
    getDarwinOSXDefines(Defines, getTargetTriple());
  }

  /// getDefaultLangOptions - Allow the target to specify default settings for
  /// various language options.  These may be overridden by command line
  /// options. 
  virtual void getDefaultLangOptions(LangOptions &Opts) {
    GetDarwinLanguageOptions(Opts, getTargetTriple());
  }
};
} // end anonymous namespace.

namespace {
// Namespace for x86 abstract base class
const Builtin::Info BuiltinInfo[] = {
#define BUILTIN(ID, TYPE, ATTRS) { #ID, TYPE, ATTRS, 0, false },
#define LIBBUILTIN(ID, TYPE, ATTRS, HEADER) { #ID, TYPE, ATTRS, HEADER, false },
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
  enum X86SSEEnum {
    NoMMXSSE, MMX, SSE1, SSE2, SSE3, SSSE3, SSE41, SSE42
  } SSELevel;
public:
  X86TargetInfo(const std::string& triple) 
    : TargetInfo(triple),
      // FIXME: hard coding to SSE2 for now.  This should change to NoMMXSSE so
      // that the driver controls this.
      SSELevel(SSE2) {
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
  virtual bool validateAsmConstraint(const char *&Name,
                                     TargetInfo::ConstraintInfo &info) const;
  virtual std::string convertConstraint(const char Constraint) const;
  virtual const char *getClobbers() const {
    return "~{dirflag},~{fpsr},~{flags}";
  }
  virtual void getTargetDefines(const LangOptions &Opts,
                                std::vector<char> &Defines) const;
  
  virtual int HandleTargetFeatures(std::string *StrArray, unsigned NumStrs,
                                   std::string &ErrorReason);
};

/// HandleTargetOptions - Handle target-specific options like -msse2 and
/// friends.  An array of arguments is passed in: if they are all valid, this
/// should handle them and return -1.  If there is an error, the index of the
/// invalid argument should be returned along with an optional error string.
int X86TargetInfo::HandleTargetFeatures(std::string *StrArray, unsigned NumStrs,
                                        std::string &ErrorReason) {
  for (unsigned i = 0; i != NumStrs; ++i) {
    const std::string &Feature = StrArray[i];
    if (Feature.size() < 2) return i;
    // Ignore explicitly disabled features.
    if (Feature[0] == '-') continue;
    
    // Feature strings are of the form "+feature".
    if (Feature[0] != '+') return i;
    
    // The set of supported subtarget features is defined in
    // lib/Target/X86/X86.td.  Here we recognize and map onto our internal
    // state.
    if (Feature == "+mmx")
      SSELevel = std::max(SSELevel, MMX);
    else if (Feature == "+sse")
      SSELevel = std::max(SSELevel, SSE1);
    else if (Feature == "+sse2")
      SSELevel = std::max(SSELevel, SSE2);
    else if (Feature == "+sse3")
      SSELevel = std::max(SSELevel, SSE3);
    else if (Feature == "+ssse3")
      SSELevel = std::max(SSELevel, SSSE3);
    else if (Feature == "+sse41")
      SSELevel = std::max(SSELevel, SSE41);
    else if (Feature == "+sse42")
      SSELevel = std::max(SSELevel, SSE42);
    else if (Feature == "+64bit" || Feature == "+slow-bt-mem")
      // Ignore these features.
      continue;
    else
      return i;
  }
  return -1;
}

/// X86TargetInfo::getTargetDefines - Return a set of the X86-specific #defines
/// that are not tied to a specific subtarget.
void X86TargetInfo::getTargetDefines(const LangOptions &Opts,
                                     std::vector<char> &Defs) const {
  // Target identification.
  if (PointerWidth == 64) {
    Define(Defs, "_LP64");
    Define(Defs, "__LP64__");
    Define(Defs, "__amd64__");
    Define(Defs, "__amd64");
    Define(Defs, "__x86_64");
    Define(Defs, "__x86_64__");
    Define(Defs, "__SSE3__");
  } else {
    DefineStd(Defs, "i386", Opts);
  }
  
  // Target properties.
  Define(Defs, "__LITTLE_ENDIAN__");
  
  // Subtarget options.
  Define(Defs, "__nocona");
  Define(Defs, "__nocona__");
  Define(Defs, "__tune_nocona__");
  Define(Defs, "__REGISTER_PREFIX__", "");

  // Each case falls through to the previous one here.
  switch (SSELevel) {
  case SSE42:
    Define(Defs, "__SSE4_2__");
  case SSE41:
    Define(Defs, "__SSE4_1__");
  case SSSE3:
    Define(Defs, "__SSSE3__");
  case SSE3:
    Define(Defs, "__SSE3__");
  case SSE2:
    Define(Defs, "__SSE2__");
    Define(Defs, "__SSE2_MATH__");  // -mfp-math=sse always implied.
  case SSE1:
    Define(Defs, "__SSE__");
    Define(Defs, "__SSE_MATH__");   // -mfp-math=sse always implied.
  case MMX:
    Define(Defs, "__MMX__");
  case NoMMXSSE:
    break;
  }
}
  
  
bool
X86TargetInfo::validateAsmConstraint(const char *&Name,
                                     TargetInfo::ConstraintInfo &info) const {
  switch (*Name) {
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
  case 'y': // Any MMX register.
  case 'x': // Any SSE register.
  case 'Q': // Any register accessible as [r]h: a, b, c, and d.
  case 'e': // 32-bit signed integer constant for use with zero-extending 
            // x86_64 instructions.
  case 'Z': // 32-bit unsigned integer constant for use with zero-extending 
            // x86_64 instructions.
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
    SizeType = UnsignedInt;
    PtrDiffType = SignedInt;
    IntPtrType = SignedInt;
    RegParmMax = 3;
  }
  virtual const char *getVAListDeclaration() const {
    return "typedef char* __builtin_va_list;";
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
    SizeType = UnsignedLong;
    IntPtrType = SignedLong;
    DescriptionString = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-"
                        "i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-"
                        "a0:0:64-f80:128:128";
  }

  virtual const char *getStringSymbolPrefix(bool IsConstant) const { 
    return IsConstant ? "\01LC" : "\01lC";
  }

  virtual const char *getUnicodeStringSymbolPrefix() const { 
    return "__utf16_string_";
  }

  virtual const char *getUnicodeStringSection() const { 
    return "__TEXT,__ustring";
  }

  virtual const char *getCFStringSymbolPrefix() const { 
    return "\01LC";
  }

  virtual void getTargetDefines(const LangOptions &Opts,
                                std::vector<char> &Defines) const {
    X86_32TargetInfo::getTargetDefines(Opts, Defines);
    getDarwinDefines(Defines, Opts);
    getDarwinOSXDefines(Defines, getTargetTriple());
  }

  /// getDefaultLangOptions - Allow the target to specify default settings for
  /// various language options.  These may be overridden by command line
  /// options. 
  virtual void getDefaultLangOptions(LangOptions &Opts) {
    GetDarwinLanguageOptions(Opts, getTargetTriple());
  }
};
} // end anonymous namespace

namespace {
// x86-32 FreeBSD target
class FreeBSDX86_32TargetInfo : public X86_32TargetInfo {
public:
  FreeBSDX86_32TargetInfo(const std::string& triple) :
      X86_32TargetInfo(triple) { }
  virtual void getTargetDefines(const LangOptions &Opts,
                                std::vector<char> &Defines) const {
    X86_32TargetInfo::getTargetDefines(Opts, Defines);
    getFreeBSDDefines(Opts, 0, getTargetTriple(), Defines);
  }
};
} // end anonymous namespace

namespace {
// x86-32 DragonFly target
class DragonFlyX86_32TargetInfo : public X86_32TargetInfo {
public:
  DragonFlyX86_32TargetInfo(const std::string& triple) :
      X86_32TargetInfo(triple) { }
  virtual void getTargetDefines(const LangOptions &Opts,
                                std::vector<char> &Defines) const {
    X86_32TargetInfo::getTargetDefines(Opts, Defines);
    getDragonFlyDefines(Opts, Defines);
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
  virtual void getTargetDefines(const LangOptions &Opts,
                                std::vector<char> &Defines) const {
    X86_32TargetInfo::getTargetDefines(Opts, Defines);
    getLinuxDefines(Opts, Defines);
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
  virtual void getTargetDefines(const LangOptions &Opts,
                                std::vector<char> &Defines) const {
    X86_32TargetInfo::getTargetDefines(Opts, Defines);
    // This list is based off of the the list of things MingW defines
    Define(Defines, "_WIN32");
    DefineStd(Defines, "WIN32", Opts);
    DefineStd(Defines, "WINNT", Opts);
    Define(Defines, "_X86_");
    Define(Defines, "__MSVCRT__");
  }
};
} // end anonymous namespace

namespace {
// x86-64 generic target
class X86_64TargetInfo : public X86TargetInfo {
public:
  X86_64TargetInfo(const std::string &triple) : X86TargetInfo(triple) {
    LongWidth = LongAlign = PointerWidth = PointerAlign = 64;
    DoubleAlign = LongLongAlign = 64;
    LongDoubleWidth = 128;
    LongDoubleAlign = 128;
    IntMaxType = SignedLong;
    UIntMaxType = UnsignedLong;
    RegParmMax = 6;

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
};
} // end anonymous namespace

namespace {
// x86-64 FreeBSD target
class FreeBSDX86_64TargetInfo : public X86_64TargetInfo {
public:
  FreeBSDX86_64TargetInfo(const std::string &triple)
    : X86_64TargetInfo(triple) {}
  virtual void getTargetDefines(const LangOptions &Opts,
                                std::vector<char> &Defines) const {
    X86_64TargetInfo::getTargetDefines(Opts, Defines);
    getFreeBSDDefines(Opts, 1, getTargetTriple(), Defines);
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
  virtual void getTargetDefines(const LangOptions &Opts,
                                std::vector<char> &Defines) const {
    X86_64TargetInfo::getTargetDefines(Opts, Defines);
    getLinuxDefines(Opts, Defines);
  }
};
} // end anonymous namespace

namespace {
// x86-64 Darwin (OS X) target
class DarwinX86_64TargetInfo : public X86_64TargetInfo {
public:
  DarwinX86_64TargetInfo(const std::string& triple) :
    X86_64TargetInfo(triple) {}

  virtual const char *getStringSymbolPrefix(bool IsConstant) const { 
    return IsConstant ? "\01LC" : "\01lC";
  }

  virtual const char *getUnicodeStringSymbolPrefix() const { 
    return "__utf16_string_";
  }

  virtual const char *getUnicodeStringSection() const { 
    return "__TEXT,__ustring";
  }

  virtual const char *getCFStringSymbolPrefix() const { 
    return "\01L_unnamed_cfstring_";
  }

  virtual void getTargetDefines(const LangOptions &Opts,
                                std::vector<char> &Defines) const {
    X86_64TargetInfo::getTargetDefines(Opts, Defines);
    getDarwinDefines(Defines, Opts);
    getDarwinOSXDefines(Defines, getTargetTriple());
  }

  /// getDefaultLangOptions - Allow the target to specify default settings for
  /// various language options.  These may be overridden by command line
  /// options. 
  virtual void getDefaultLangOptions(LangOptions &Opts) {
    GetDarwinLanguageOptions(Opts, getTargetTriple());
  }
};
} // end anonymous namespace.

namespace {
class ARMTargetInfo : public TargetInfo {
  enum {
    Armv4t,
    Armv5,
    Armv6,
    XScale
  } ArmArch;
public:
  ARMTargetInfo(const std::string& triple) : TargetInfo(triple) {
    // FIXME: Are the defaults correct for ARM?
    DescriptionString = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-"
                        "i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:64:64";
    if (triple.find("arm-") || triple.find("armv6-"))
      ArmArch = Armv6;
    else if (triple.find("armv5-"))
      ArmArch = Armv5;
    else if (triple.find("armv4t-"))
      ArmArch = Armv4t;
    else if (triple.find("xscale-"))
      ArmArch = XScale;
  }
  virtual void getTargetDefines(const LangOptions &Opts,
                                std::vector<char> &Defs) const {
    // Target identification.
    Define(Defs, "__arm");
    Define(Defs, "__arm__");
    
    // Target properties.
    Define(Defs, "__LITTLE_ENDIAN__");
    
    // Subtarget options.
    if (ArmArch == Armv6) {
      Define(Defs, "__ARM_ARCH_6K__");
      Define(Defs, "__THUMB_INTERWORK__");
    } else if (ArmArch == Armv5) {
      Define(Defs, "__ARM_ARCH_5TEJ__");
      Define(Defs, "__THUMB_INTERWORK__");
      Define(Defs, "__SOFTFP__");
    } else if (ArmArch == Armv4t) {
      Define(Defs, "__ARM_ARCH_4T__");
      Define(Defs, "__SOFTFP__");
    } else if (ArmArch == XScale) {
      Define(Defs, "__ARM_ARCH_5TE__");
      Define(Defs, "__XSCALE__");
      Define(Defs, "__SOFTFP__");
    }
    Define(Defs, "__ARMEL__");
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
  virtual bool validateAsmConstraint(const char *&Name,
                                     TargetInfo::ConstraintInfo &info) const {
    // FIXME: Check if this is complete
    switch (*Name) {
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

  virtual void getTargetDefines(const LangOptions &Opts,
                                std::vector<char> &Defines) const {
    ARMTargetInfo::getTargetDefines(Opts, Defines);
    getDarwinDefines(Defines, Opts);
    getDarwinIPhoneOSDefines(Defines, getTargetTriple());
  }
};
} // end anonymous namespace.

namespace {
// arm FreeBSD target
class FreeBSDARMTargetInfo : public ARMTargetInfo {
public:
  FreeBSDARMTargetInfo(const std::string& triple) : ARMTargetInfo(triple) {}
  virtual void getTargetDefines(const LangOptions &Opts,
                                std::vector<char> &Defines) const {
    ARMTargetInfo::getTargetDefines(Opts, Defines);
    getFreeBSDDefines(Opts, 0, getTargetTriple(), Defines);
  }
};
} // end anonymous namespace

namespace {
class SparcV8TargetInfo : public TargetInfo {
  static const TargetInfo::GCCRegAlias GCCRegAliases[];
  static const char * const GCCRegNames[];
public:
  SparcV8TargetInfo(const std::string& triple) : TargetInfo(triple) {
    // FIXME: Support Sparc quad-precision long double?
    DescriptionString = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-"
                        "i64:64:64-f32:32:32-f64:64:64-v64:64:64";
  }
  virtual void getTargetDefines(const LangOptions &Opts,
                                std::vector<char> &Defines) const {
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
                              unsigned &NumNames) const;
  virtual void getGCCRegAliases(const GCCRegAlias *&Aliases, 
                                unsigned &NumAliases) const;
  virtual bool validateAsmConstraint(const char *&Name,
                                     TargetInfo::ConstraintInfo &info) const {
    // FIXME: Implement!
    return false;
  }
  virtual const char *getClobbers() const {
    // FIXME: Implement!
    return "";
  }
};

const char * const SparcV8TargetInfo::GCCRegNames[] = {
  "r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7",
  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
  "r16", "r17", "r18", "r19", "r20", "r21", "r22", "r23",
  "r24", "r25", "r26", "r27", "r28", "r29", "r30", "r31"
};

void SparcV8TargetInfo::getGCCRegNames(const char * const *&Names, 
                                       unsigned &NumNames) const {
  Names = GCCRegNames;
  NumNames = llvm::array_lengthof(GCCRegNames);
}

const TargetInfo::GCCRegAlias SparcV8TargetInfo::GCCRegAliases[] = {
  { { "g0" }, "r0" }, 
  { { "g1" }, "r1" }, 
  { { "g2" }, "r2" }, 
  { { "g3" }, "r3" }, 
  { { "g4" }, "r4" }, 
  { { "g5" }, "r5" }, 
  { { "g6" }, "r6" }, 
  { { "g7" }, "r7" }, 
  { { "o0" }, "r8" }, 
  { { "o1" }, "r9" }, 
  { { "o2" }, "r10" }, 
  { { "o3" }, "r11" }, 
  { { "o4" }, "r12" }, 
  { { "o5" }, "r13" }, 
  { { "o6", "sp" }, "r14" }, 
  { { "o7" }, "r15" }, 
  { { "l0" }, "r16" }, 
  { { "l1" }, "r17" }, 
  { { "l2" }, "r18" }, 
  { { "l3" }, "r19" }, 
  { { "l4" }, "r20" }, 
  { { "l5" }, "r21" }, 
  { { "l6" }, "r22" }, 
  { { "l7" }, "r23" }, 
  { { "i0" }, "r24" }, 
  { { "i1" }, "r25" }, 
  { { "i2" }, "r26" }, 
  { { "i3" }, "r27" }, 
  { { "i4" }, "r28" }, 
  { { "i5" }, "r29" }, 
  { { "i6", "fp" }, "r30" }, 
  { { "i7" }, "r31" }, 
};

void SparcV8TargetInfo::getGCCRegAliases(const GCCRegAlias *&Aliases, 
                                         unsigned &NumAliases) const {
  Aliases = GCCRegAliases;
  NumAliases = llvm::array_lengthof(GCCRegAliases);
}
} // end anonymous namespace.

namespace {
class SolarisSparcV8TargetInfo : public SparcV8TargetInfo {
public:
  SolarisSparcV8TargetInfo(const std::string& triple) :
      SparcV8TargetInfo(triple) {
    SizeType = UnsignedInt;
    PtrDiffType = SignedInt;
  }

  virtual void getTargetDefines(const LangOptions &Opts,
                                std::vector<char> &Defines) const {
    SparcV8TargetInfo::getTargetDefines(Opts, Defines);
    getSolarisDefines(Defines);
  }
};
} // end anonymous namespace.

namespace {
  class PIC16TargetInfo : public TargetInfo{
  public:
    PIC16TargetInfo(const std::string& triple) : TargetInfo(triple) {
      IntWidth = 16;
      LongWidth = LongLongWidth = 32;
      PointerWidth = 16;
      IntAlign = 8;
      LongAlign = LongLongAlign = 8;
      PointerAlign = 8;
      SizeType = UnsignedInt;
      IntMaxType = SignedLong;
      UIntMaxType = UnsignedLong;
      IntPtrType = SignedShort;
      PtrDiffType = SignedInt;
      DescriptionString = "e-p:16:8:8-i8:8:8-i16:8:8-i32:8:8";
    }
    virtual uint64_t getPointerWidthV(unsigned AddrSpace) const { return 16; }
    virtual uint64_t getPointerAlignV(unsigned AddrSpace) const { return 8; }
    virtual void getTargetDefines(const LangOptions &Opts,
                                  std::vector<char> &Defines) const {
      Define(Defines, "__pic16");
    }
    virtual void getTargetBuiltins(const Builtin::Info *&Records,
                                   unsigned &NumRecords) const {}
    virtual const char *getVAListDeclaration() const { return "";}
    virtual const char *getClobbers() const {return "";}
    virtual const char *getTargetPrefix() const {return "";}
    virtual void getGCCRegNames(const char * const *&Names, 
                                unsigned &NumNames) const {} 
    virtual bool validateAsmConstraint(const char *&Name, 
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
  bool isFreeBSD = T.find("-freebsd") != std::string::npos;
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

  if (T.find("armv6-") == 0 || T.find("arm-") == 0
      || T.find("armv4t") == 0 || T.find("armv5-") == 0
      || T.find("xscale") == 0) {
    if (isDarwin)
      return new DarwinARMTargetInfo(T);
    if (isFreeBSD)
      return new FreeBSDARMTargetInfo(T);
    return new ARMTargetInfo(T);
  }

  if (T.find("sparc-") == 0) {
    if (isSolaris)
      return new SolarisSparcV8TargetInfo(T);
    return new SparcV8TargetInfo(T);
  }

  if (T.find("x86_64-") == 0 || T.find("amd64-") == 0) {
    if (isDarwin)
      return new DarwinX86_64TargetInfo(T);
    if (isLinux)
      return new LinuxX86_64TargetInfo(T);
    if (isFreeBSD)
      return new FreeBSDX86_64TargetInfo(T);
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
    if (isFreeBSD)
      return new FreeBSDX86_32TargetInfo(T);
    if (isWindows)
      return new WindowsX86_32TargetInfo(T);
    return new X86_32TargetInfo(T);
  }

  return NULL;
}

