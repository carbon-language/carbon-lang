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

#include "clang/Basic/Builtins.h"
#include "clang/Basic/TargetBuiltins.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/LangOptions.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Triple.h"
#include "llvm/MC/MCSectionMachO.h"
using namespace clang;

//===----------------------------------------------------------------------===//
//  Common code shared among targets.
//===----------------------------------------------------------------------===//

static void Define(std::vector<char> &Buf, const llvm::StringRef &Macro,
                   const llvm::StringRef &Val = "1") {
  const char *Def = "#define ";
  Buf.insert(Buf.end(), Def, Def+strlen(Def));
  Buf.insert(Buf.end(), Macro.begin(), Macro.end());
  Buf.push_back(' ');
  Buf.insert(Buf.end(), Val.begin(), Val.end());
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
  Define(Buf, TmpStr.str());

  // Define __unix__.
  TmpStr += "__";
  Define(Buf, TmpStr.str());
}

//===----------------------------------------------------------------------===//
// Defines specific to certain operating systems.
//===----------------------------------------------------------------------===//

namespace {
template<typename TgtInfo>
class OSTargetInfo : public TgtInfo {
protected:
  virtual void getOSDefines(const LangOptions &Opts, const llvm::Triple &Triple,
                            std::vector<char> &Defines) const=0;
public:
  OSTargetInfo(const std::string& triple) : TgtInfo(triple) {}
  virtual void getTargetDefines(const LangOptions &Opts,
                                std::vector<char> &Defines) const {
    TgtInfo::getTargetDefines(Opts, Defines);
    getOSDefines(Opts, TgtInfo::getTriple(), Defines);
  }

};
} // end anonymous namespace


static void getDarwinDefines(std::vector<char> &Defs, const LangOptions &Opts) {
  Define(Defs, "__APPLE_CC__", "5621");
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

  if (Opts.Static)
    Define(Defs, "__STATIC__");
  else
    Define(Defs, "__DYNAMIC__");
}

static void getDarwinOSXDefines(std::vector<char> &Defs,
                                const llvm::Triple &Triple) {
  if (Triple.getOS() != llvm::Triple::Darwin)
    return;
  
  // Figure out which "darwin number" the target triple is.  "darwin9" -> 10.5.
  unsigned Maj, Min, Rev;
  Triple.getDarwinNumber(Maj, Min, Rev);
  
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

static void getDarwinIPhoneOSDefines(std::vector<char> &Defs,
                                     const llvm::Triple &Triple) {
  if (Triple.getOS() != llvm::Triple::Darwin)
    return;
  
  // Figure out which "darwin number" the target triple is.  "darwin9" -> 10.5.
  unsigned Maj, Min, Rev;
  Triple.getDarwinNumber(Maj, Min, Rev);
  
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

/// GetDarwinLanguageOptions - Set the default language options for darwin.
static void GetDarwinLanguageOptions(LangOptions &Opts,
                                     const llvm::Triple &Triple) {
  Opts.NeXTRuntime = true;
  
  if (Triple.getOS() != llvm::Triple::Darwin)
    return;

  unsigned MajorVersion = Triple.getDarwinMajorNumber();

  // Blocks and stack protectors default to on for 10.6 (darwin10) and beyond.
  if (MajorVersion > 9) {
    Opts.Blocks = 1;
    Opts.setStackProtectorMode(LangOptions::SSPOn);
  }

  // Non-fragile ABI (in 64-bit mode) default to on for 10.5 (darwin9) and
  // beyond.
  if (MajorVersion >= 9 && Opts.ObjC1 &&
      Triple.getArch() == llvm::Triple::x86_64)
    Opts.ObjCNonFragileABI = 1;
}

namespace {
template<typename Target>
class DarwinTargetInfo : public OSTargetInfo<Target> {
protected:
  virtual void getOSDefines(const LangOptions &Opts, const llvm::Triple &Triple,
                    std::vector<char> &Defines) const {
    getDarwinDefines(Defines, Opts);
    getDarwinOSXDefines(Defines, Triple);
  }
  
  /// getDefaultLangOptions - Allow the target to specify default settings for
  /// various language options.  These may be overridden by command line
  /// options.
  virtual void getDefaultLangOptions(LangOptions &Opts) {
    TargetInfo::getDefaultLangOptions(Opts);
    GetDarwinLanguageOptions(Opts, TargetInfo::getTriple());
  }
public:
  DarwinTargetInfo(const std::string& triple) :
    OSTargetInfo<Target>(triple) {
      this->TLSSupported = false;
    }

  virtual const char *getUnicodeStringSymbolPrefix() const {
    return "__utf16_string_";
  }

  virtual const char *getUnicodeStringSection() const {
    return "__TEXT,__ustring";
  }
  
  virtual std::string isValidSectionSpecifier(const llvm::StringRef &SR) const {
    // Let MCSectionMachO validate this.
    llvm::StringRef Segment, Section;
    unsigned TAA, StubSize;
    return llvm::MCSectionMachO::ParseSectionSpecifier(SR, Segment, Section,
                                                       TAA, StubSize);
  }
};


// DragonFlyBSD Target
template<typename Target>
class DragonFlyBSDTargetInfo : public OSTargetInfo<Target> {
protected:
  virtual void getOSDefines(const LangOptions &Opts, const llvm::Triple &Triple,
                    std::vector<char> &Defs) const {
    // DragonFly defines; list based off of gcc output
    Define(Defs, "__DragonFly__");
    Define(Defs, "__DragonFly_cc_version", "100001");
    Define(Defs, "__ELF__");
    Define(Defs, "__KPRINTF_ATTRIBUTE__");
    Define(Defs, "__tune_i386__");
    DefineStd(Defs, "unix", Opts);
  }
public:
  DragonFlyBSDTargetInfo(const std::string &triple) 
    : OSTargetInfo<Target>(triple) {}
};

// FreeBSD Target
template<typename Target>
class FreeBSDTargetInfo : public OSTargetInfo<Target> {
protected:
  virtual void getOSDefines(const LangOptions &Opts, const llvm::Triple &Triple,
                    std::vector<char> &Defs) const {
    // FreeBSD defines; list based off of gcc output

    // FIXME: Move version number handling to llvm::Triple.
    const char *FreeBSD = strstr(Triple.getTriple().c_str(),
                                 "-freebsd");
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
  }
public:
  FreeBSDTargetInfo(const std::string &triple) 
    : OSTargetInfo<Target>(triple) {
      this->UserLabelPrefix = "";
    }
};

// Linux target
template<typename Target>
class LinuxTargetInfo : public OSTargetInfo<Target> {
protected:
  virtual void getOSDefines(const LangOptions &Opts, const llvm::Triple &Triple,
                           std::vector<char> &Defs) const {
    // Linux defines; list based off of gcc output
    DefineStd(Defs, "unix", Opts);
    DefineStd(Defs, "linux", Opts);
    Define(Defs, "__gnu_linux__");
    Define(Defs, "__ELF__", "1");
  }
public:
  LinuxTargetInfo(const std::string& triple) 
    : OSTargetInfo<Target>(triple) {
    this->UserLabelPrefix = "";
  }
};

// NetBSD Target
template<typename Target>
class NetBSDTargetInfo : public OSTargetInfo<Target> {
protected:
  virtual void getOSDefines(const LangOptions &Opts, const llvm::Triple &Triple,
                    std::vector<char> &Defs) const {
    // NetBSD defines; list based off of gcc output
    Define(Defs, "__NetBSD__", "1");
    Define(Defs, "__unix__", "1");
    Define(Defs, "__ELF__", "1");
  }
public:
  NetBSDTargetInfo(const std::string &triple) 
    : OSTargetInfo<Target>(triple) {
      this->UserLabelPrefix = "";
    }
};

// OpenBSD Target
template<typename Target>
class OpenBSDTargetInfo : public OSTargetInfo<Target> {
protected:
  virtual void getOSDefines(const LangOptions &Opts, const llvm::Triple &Triple,
                    std::vector<char> &Defs) const {
    // OpenBSD defines; list based off of gcc output

    Define(Defs, "__OpenBSD__", "1");
    DefineStd(Defs, "unix", Opts);
    Define(Defs, "__ELF__", "1");
  }
public:
  OpenBSDTargetInfo(const std::string &triple) 
    : OSTargetInfo<Target>(triple) {}
};

// Solaris target
template<typename Target>
class SolarisTargetInfo : public OSTargetInfo<Target> {
protected:
  virtual void getOSDefines(const LangOptions &Opts, const llvm::Triple &Triple,
                                std::vector<char> &Defs) const {
    DefineStd(Defs, "sun", Opts);
    DefineStd(Defs, "unix", Opts);
    Define(Defs, "__ELF__");
    Define(Defs, "__svr4__");
    Define(Defs, "__SVR4");
  }
public:
  SolarisTargetInfo(const std::string& triple) 
    : OSTargetInfo<Target>(triple) {
    this->UserLabelPrefix = "";
    this->WCharType = this->SignedLong;
    // FIXME: WIntType should be SignedLong
  }
};
} // end anonymous namespace. 

/// GetWindowsLanguageOptions - Set the default language options for Windows.
static void GetWindowsLanguageOptions(LangOptions &Opts,
                                     const llvm::Triple &Triple) {
  Opts.Microsoft = true;
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
  PPCTargetInfo(const std::string& triple) : TargetInfo(triple) {}

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
  virtual void getGCCRegNames(const char * const *&Names,
                              unsigned &NumNames) const;
  virtual void getGCCRegAliases(const GCCRegAlias *&Aliases,
                                unsigned &NumAliases) const;
  virtual bool validateAsmConstraint(const char *&Name,
                                     TargetInfo::ConstraintInfo &Info) const {
    switch (*Name) {
    default: return false;
    case 'O': // Zero
      return true;
    case 'b': // Base register
    case 'f': // Floating point register
      Info.setAllowsRegister();
      return true;
    }
  }
  virtual void getDefaultLangOptions(LangOptions &Opts) {
    TargetInfo::getDefaultLangOptions(Opts);
    Opts.CharIsSigned = false;
  }
  virtual const char *getClobbers() const {
    return "";
  }
};

const Builtin::Info PPCTargetInfo::BuiltinInfo[] = {
#define BUILTIN(ID, TYPE, ATTRS) { #ID, TYPE, ATTRS, 0, false },
#define LIBBUILTIN(ID, TYPE, ATTRS, HEADER) { #ID, TYPE, ATTRS, HEADER, false },
#include "clang/Basic/BuiltinsPPC.def"
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
    IntMaxType = SignedLong;
    UIntMaxType = UnsignedLong;
    Int64Type = SignedLong;
    DescriptionString = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-"
                        "i64:64:64-f32:32:32-f64:64:64-v128:128:128";
  }
};
} // end anonymous namespace.

namespace {
// Namespace for x86 abstract base class
const Builtin::Info BuiltinInfo[] = {
#define BUILTIN(ID, TYPE, ATTRS) { #ID, TYPE, ATTRS, 0, false },
#define LIBBUILTIN(ID, TYPE, ATTRS, HEADER) { #ID, TYPE, ATTRS, HEADER, false },
#include "clang/Basic/BuiltinsX86.def"
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
    : TargetInfo(triple), SSELevel(NoMMXSSE) {
    LongDoubleFormat = &llvm::APFloat::x87DoubleExtended;
  }
  virtual void getTargetBuiltins(const Builtin::Info *&Records,
                                 unsigned &NumRecords) const {
    Records = BuiltinInfo;
    NumRecords = clang::X86::LastTSBuiltin-Builtin::FirstTSBuiltin;
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
  virtual bool setFeatureEnabled(llvm::StringMap<bool> &Features,
                                 const std::string &Name,
                                 bool Enabled) const;
  virtual void getDefaultFeatures(const std::string &CPU, 
                                  llvm::StringMap<bool> &Features) const;
  virtual void HandleTargetFeatures(const llvm::StringMap<bool> &Features);
};

void X86TargetInfo::getDefaultFeatures(const std::string &CPU, 
                                       llvm::StringMap<bool> &Features) const {
  // FIXME: This should not be here.
  Features["3dnow"] = false;
  Features["3dnowa"] = false;
  Features["mmx"] = false;
  Features["sse"] = false;
  Features["sse2"] = false;
  Features["sse3"] = false;
  Features["ssse3"] = false;
  Features["sse41"] = false;
  Features["sse42"] = false;

  // LLVM does not currently recognize this.
  // Features["sse4a"] = false;

  // FIXME: This *really* should not be here.

  // X86_64 always has SSE2.
  if (PointerWidth == 64)
    Features["sse2"] = Features["sse"] = Features["mmx"] = true;

  if (CPU == "generic" || CPU == "i386" || CPU == "i486" || CPU == "i586" ||
      CPU == "pentium" || CPU == "i686" || CPU == "pentiumpro")
    ;
  else if (CPU == "pentium-mmx" || CPU == "pentium2")
    setFeatureEnabled(Features, "mmx", true);
  else if (CPU == "pentium3")
    setFeatureEnabled(Features, "sse", true);
  else if (CPU == "pentium-m" || CPU == "pentium4" || CPU == "x86-64")
    setFeatureEnabled(Features, "sse2", true);
  else if (CPU == "yonah" || CPU == "prescott" || CPU == "nocona")
    setFeatureEnabled(Features, "sse3", true);
  else if (CPU == "core2")
    setFeatureEnabled(Features, "ssse3", true);
  else if (CPU == "penryn") {
    setFeatureEnabled(Features, "sse4", true);
    Features["sse42"] = false;
  } else if (CPU == "atom")
    setFeatureEnabled(Features, "sse3", true);
  else if (CPU == "corei7")
    setFeatureEnabled(Features, "sse4", true);
  else if (CPU == "k6" || CPU == "winchip-c6")
    setFeatureEnabled(Features, "mmx", true);
  else if (CPU == "k6-2" || CPU == "k6-3" || CPU == "athlon" || 
           CPU == "athlon-tbird" || CPU == "winchip2" || CPU == "c3") {
    setFeatureEnabled(Features, "mmx", true);
    setFeatureEnabled(Features, "3dnow", true);
  } else if (CPU == "athlon-4" || CPU == "athlon-xp" || CPU == "athlon-mp") {
    setFeatureEnabled(Features, "sse", true);
    setFeatureEnabled(Features, "3dnowa", true);
  } else if (CPU == "k8" || CPU == "opteron" || CPU == "athlon64" ||
           CPU == "athlon-fx") {
    setFeatureEnabled(Features, "sse2", true); 
    setFeatureEnabled(Features, "3dnowa", true);
  } else if (CPU == "c3-2")
    setFeatureEnabled(Features, "sse", true);
}

bool X86TargetInfo::setFeatureEnabled(llvm::StringMap<bool> &Features,
                                      const std::string &Name, 
                                      bool Enabled) const {
  // FIXME: This *really* should not be here.
  if (!Features.count(Name) && Name != "sse4")
    return false;

  if (Enabled) {
    if (Name == "mmx")
      Features["mmx"] = true;
    else if (Name == "sse")
      Features["mmx"] = Features["sse"] = true;
    else if (Name == "sse2")
      Features["mmx"] = Features["sse"] = Features["sse2"] = true;
    else if (Name == "sse3")
      Features["mmx"] = Features["sse"] = Features["sse2"] = 
        Features["sse3"] = true;
    else if (Name == "ssse3")
      Features["mmx"] = Features["sse"] = Features["sse2"] = Features["sse3"] = 
        Features["ssse3"] = true;
    else if (Name == "sse4")
      Features["mmx"] = Features["sse"] = Features["sse2"] = Features["sse3"] = 
        Features["ssse3"] = Features["sse41"] = Features["sse42"] = true;
    else if (Name == "3dnow")
      Features["3dnowa"] = true;
    else if (Name == "3dnowa")
      Features["3dnow"] = Features["3dnowa"] = true;
  } else {
    if (Name == "mmx")
      Features["mmx"] = Features["sse"] = Features["sse2"] = Features["sse3"] = 
        Features["ssse3"] = Features["sse41"] = Features["sse42"] = false;
    else if (Name == "sse")
      Features["sse"] = Features["sse2"] = Features["sse3"] = 
        Features["ssse3"] = Features["sse41"] = Features["sse42"] = false;
    else if (Name == "sse2")
      Features["sse2"] = Features["sse3"] = Features["ssse3"] = 
        Features["sse41"] = Features["sse42"] = false;
    else if (Name == "sse3")
      Features["sse3"] = Features["ssse3"] = Features["sse41"] = 
        Features["sse42"] = false;
    else if (Name == "ssse3")
      Features["ssse3"] = Features["sse41"] = Features["sse42"] = false;
    else if (Name == "sse4")
      Features["sse41"] = Features["sse42"] = false;
    else if (Name == "3dnow")
      Features["3dnow"] = Features["3dnowa"] = false;
    else if (Name == "3dnowa")
      Features["3dnowa"] = false;
  }

  return true;
}

/// HandleTargetOptions - Perform initialization based on the user
/// configured set of features.
void X86TargetInfo::HandleTargetFeatures(const llvm::StringMap<bool>&Features) {
  if (Features.lookup("sse42"))
    SSELevel = SSE42;
  else if (Features.lookup("sse41"))
    SSELevel = SSE41;
  else if (Features.lookup("ssse3"))
    SSELevel = SSSE3;
  else if (Features.lookup("sse3"))
    SSELevel = SSE3;
  else if (Features.lookup("sse2"))
    SSELevel = SSE2;
  else if (Features.lookup("sse"))
    SSELevel = SSE1;
  else if (Features.lookup("mmx"))
    SSELevel = MMX;
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

  // Define __NO_MATH_INLINES on linux/x86 so that we don't get inline
  // functions in glibc header files that use FP Stack inline asm which the
  // backend can't deal with (PR879).
  Define(Defs, "__NO_MATH_INLINES");

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
                                     TargetInfo::ConstraintInfo &Info) const {
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
  case 'R': // "legacy" registers: ax, bx, cx, dx, di, si, sp, bp.
    Info.setAllowsRegister();
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
class OpenBSDI386TargetInfo : public OpenBSDTargetInfo<X86_32TargetInfo> {
public:
  OpenBSDI386TargetInfo(const std::string& triple) :
    OpenBSDTargetInfo<X86_32TargetInfo>(triple) {
    SizeType = UnsignedLong;
    IntPtrType = SignedLong;
    PtrDiffType = SignedLong;
  }
};
} // end anonymous namespace

namespace {
class DarwinI386TargetInfo : public DarwinTargetInfo<X86_32TargetInfo> {
public:
  DarwinI386TargetInfo(const std::string& triple) :
    DarwinTargetInfo<X86_32TargetInfo>(triple) {
    LongDoubleWidth = 128;
    LongDoubleAlign = 128;
    SizeType = UnsignedLong;
    IntPtrType = SignedLong;
    DescriptionString = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-"
                        "i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-"
                        "a0:0:64-f80:128:128";
  }

};
} // end anonymous namespace

namespace {
// x86-32 Windows target
class WindowsX86_32TargetInfo : public X86_32TargetInfo {
public:
  WindowsX86_32TargetInfo(const std::string& triple)
    : X86_32TargetInfo(triple) {
    TLSSupported = false;
    WCharType = UnsignedShort;
    WCharWidth = WCharAlign = 16;
    DoubleAlign = LongLongAlign = 64;
    DescriptionString = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-"
                        "i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-"
                        "a0:0:64-f80:32:32";
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

  virtual void getDefaultLangOptions(LangOptions &Opts) {
    X86_32TargetInfo::getDefaultLangOptions(Opts);
    GetWindowsLanguageOptions(Opts, getTriple());
  }
};
} // end anonymous namespace

namespace {
// x86-64 generic target
class X86_64TargetInfo : public X86TargetInfo {
public:
  X86_64TargetInfo(const std::string &triple) : X86TargetInfo(triple) {
    LongWidth = LongAlign = PointerWidth = PointerAlign = 64;
    LongDoubleWidth = 128;
    LongDoubleAlign = 128;
    IntMaxType = SignedLong;
    UIntMaxType = UnsignedLong;
    Int64Type = SignedLong;
    RegParmMax = 6;

    DescriptionString = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-"
                        "i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-"
                        "a0:0:64-s0:64:64-f80:128:128";
  }
  virtual const char *getVAListDeclaration() const {
    return "typedef struct __va_list_tag {"
           "  unsigned gp_offset;"
           "  unsigned fp_offset;"
           "  void* overflow_arg_area;"
           "  void* reg_save_area;"
           "} __va_list_tag;"
           "typedef __va_list_tag __builtin_va_list[1];";
  }
};
} // end anonymous namespace

namespace {
class DarwinX86_64TargetInfo : public DarwinTargetInfo<X86_64TargetInfo> {
public:
  DarwinX86_64TargetInfo(const std::string& triple) 
      : DarwinTargetInfo<X86_64TargetInfo>(triple) {
    Int64Type = SignedLongLong;
  }
};
} // end anonymous namespace

namespace {
class OpenBSDX86_64TargetInfo : public OpenBSDTargetInfo<X86_64TargetInfo> {
public:
  OpenBSDX86_64TargetInfo(const std::string& triple) 
      : OpenBSDTargetInfo<X86_64TargetInfo>(triple) {
    IntMaxType = SignedLongLong;
    UIntMaxType = UnsignedLongLong;
    Int64Type = SignedLongLong;
  }
};
} // end anonymous namespace

namespace {
class ARMTargetInfo : public TargetInfo {
  enum {
    Armv4t,
    Armv5,
    Armv6,
    Armv7a,
    XScale
  } ArmArch;
public:
  ARMTargetInfo(const std::string& triple) : TargetInfo(triple) {
    // FIXME: Are the defaults correct for ARM?
    DescriptionString = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-"
                        "i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:64:64";
    if (triple.find("armv7-") == 0)
      ArmArch = Armv7a;
    else if (triple.find("arm-") == 0 || triple.find("armv6-") == 0)
      ArmArch = Armv6;
    else if (triple.find("armv5-") == 0)
      ArmArch = Armv5;
    else if (triple.find("armv4t-") == 0)
      ArmArch = Armv4t;
    else if (triple.find("xscale-") == 0)
      ArmArch = XScale;
    else if (triple.find("armv") == 0) {
      // FIXME: fuzzy match for other random weird arm triples.  This is useful
      // for the static analyzer and other clients, but probably should be
      // re-evaluated when codegen is brought up.
      ArmArch = Armv6;
    }
  }
  virtual void getTargetDefines(const LangOptions &Opts,
                                std::vector<char> &Defs) const {
    // Target identification.
    Define(Defs, "__arm");
    Define(Defs, "__arm__");

    // Target properties.
    Define(Defs, "__LITTLE_ENDIAN__");

    // Subtarget options.
    if (ArmArch == Armv7a) {
      Define(Defs, "__ARM_ARCH_7A__");
      Define(Defs, "__THUMB_INTERWORK__");
    } else if (ArmArch == Armv6) {
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
    Define(Defs, "__APCS_32__");
    Define(Defs, "__VFP_FP__");
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
                                     TargetInfo::ConstraintInfo &Info) const {
    // FIXME: Check if this is complete
    switch (*Name) {
    default:
    case 'l': // r0-r7
    case 'h': // r8-r15
    case 'w': // VFP Floating point register single precision
    case 'P': // VFP Floating point register double precision
      Info.setAllowsRegister();
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
class DarwinARMTargetInfo : 
  public DarwinTargetInfo<ARMTargetInfo> {
protected:
  virtual void getOSDefines(const LangOptions &Opts, const llvm::Triple &Triple,
                    std::vector<char> &Defines) const {
    getDarwinDefines(Defines, Opts);
    getDarwinIPhoneOSDefines(Defines, Triple);
  }

public:
  DarwinARMTargetInfo(const std::string& triple) 
    : DarwinTargetInfo<ARMTargetInfo>(triple) {}
};
} // end anonymous namespace.

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
    DefineStd(Defines, "sparc", Opts);
    Define(Defines, "__sparcv8");
    Define(Defines, "__REGISTER_PREFIX__", "");
  }
  virtual void getTargetBuiltins(const Builtin::Info *&Records,
                                 unsigned &NumRecords) const {
    // FIXME: Implement!
  }
  virtual const char *getVAListDeclaration() const {
    return "typedef void* __builtin_va_list;";
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
class SolarisSparcV8TargetInfo : public SolarisTargetInfo<SparcV8TargetInfo> {
public:
  SolarisSparcV8TargetInfo(const std::string& triple) :
      SolarisTargetInfo<SparcV8TargetInfo>(triple) {
    SizeType = UnsignedInt;
    PtrDiffType = SignedInt;
  }
};
} // end anonymous namespace.

namespace {
  class PIC16TargetInfo : public TargetInfo{
  public:
    PIC16TargetInfo(const std::string& triple) : TargetInfo(triple) {
      TLSSupported = false;
      IntWidth = 16;
      LongWidth = LongLongWidth = 32;
      IntMaxTWidth = 32;
      PointerWidth = 16;
      IntAlign = 8;
      LongAlign = LongLongAlign = 8;
      PointerAlign = 8;
      SizeType = UnsignedInt;
      IntMaxType = SignedLong;
      UIntMaxType = UnsignedLong;
      IntPtrType = SignedShort;
      PtrDiffType = SignedInt;
      FloatWidth = 32;
      FloatAlign = 32;
      DoubleWidth = 32;
      DoubleAlign = 32;
      LongDoubleWidth = 32;
      LongDoubleAlign = 32;
      FloatFormat = &llvm::APFloat::IEEEsingle;
      DoubleFormat = &llvm::APFloat::IEEEsingle;
      LongDoubleFormat = &llvm::APFloat::IEEEsingle;
      DescriptionString = "e-p:16:8:8-i8:8:8-i16:8:8-i32:8:8-f32:32:32";

    }
    virtual uint64_t getPointerWidthV(unsigned AddrSpace) const { return 16; }
    virtual uint64_t getPointerAlignV(unsigned AddrSpace) const { return 8; }
    virtual void getTargetDefines(const LangOptions &Opts,
                                  std::vector<char> &Defines) const {
      Define(Defines, "__pic16");
      Define(Defines, "rom", "__attribute__((address_space(1)))");
      Define(Defines, "ram", "__attribute__((address_space(0)))");
      Define(Defines, "_section(SectName)", 
             "__attribute__((section(SectName)))");
      Define(Defines, "_address(Addr)",
             "__attribute__((section(\"Address=\"#Addr)))");
      Define(Defines, "_CONFIG(conf)", "asm(\"CONFIG \"#conf)");
      Define(Defines, "_interrupt",
             "__attribute__((section(\"interrupt=0x4\"))) \
             __attribute__((used))");
    }
    virtual void getTargetBuiltins(const Builtin::Info *&Records,
                                   unsigned &NumRecords) const {}
    virtual const char *getVAListDeclaration() const { 
      return "";
    }
    virtual const char *getClobbers() const {
      return "";
    }
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

namespace {
  class MSP430TargetInfo : public TargetInfo {
    static const char * const GCCRegNames[];
  public:
    MSP430TargetInfo(const std::string& triple) : TargetInfo(triple) {
      TLSSupported = false;
      IntWidth = 16;
      LongWidth = LongLongWidth = 32;
      IntMaxTWidth = 32;
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
    virtual void getTargetDefines(const LangOptions &Opts,
                                 std::vector<char> &Defines) const {
      Define(Defines, "MSP430");
      Define(Defines, "__MSP430__");
      // FIXME: defines for different 'flavours' of MCU
    }
    virtual void getTargetBuiltins(const Builtin::Info *&Records,
                                   unsigned &NumRecords) const {
     // FIXME: Implement.
      Records = 0;
      NumRecords = 0;
    }
    virtual void getGCCRegNames(const char * const *&Names,
                                unsigned &NumNames) const;
    virtual void getGCCRegAliases(const GCCRegAlias *&Aliases,
                                  unsigned &NumAliases) const {
      // No aliases.
      Aliases = 0;
      NumAliases = 0;
    }
    virtual bool validateAsmConstraint(const char *&Name,
                                       TargetInfo::ConstraintInfo &info) const {
      // FIXME: implement
      return true;
    }
    virtual const char *getClobbers() const {
      // FIXME: Is this really right?
      return "";
    }
    virtual const char *getVAListDeclaration() const {
      // FIXME: implement
      return "typedef char* __builtin_va_list;";
   }
  };

  const char * const MSP430TargetInfo::GCCRegNames[] = {
    "r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7",
    "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"
  };

  void MSP430TargetInfo::getGCCRegNames(const char * const *&Names,
                                        unsigned &NumNames) const {
    Names = GCCRegNames;
    NumNames = llvm::array_lengthof(GCCRegNames);
  }
}


namespace {
  class SystemZTargetInfo : public TargetInfo {
    static const char * const GCCRegNames[];
  public:
    SystemZTargetInfo(const std::string& triple) : TargetInfo(triple) {
      TLSSupported = false;
      IntWidth = IntAlign = 32;
      LongWidth = LongLongWidth = LongAlign = LongLongAlign = 64;
      PointerWidth = PointerAlign = 64;
      DescriptionString = "E-p:64:64:64-i8:8:16-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-a0:16:16";
   }
    virtual void getTargetDefines(const LangOptions &Opts,
                                 std::vector<char> &Defines) const {
      Define(Defines, "__s390__");
      Define(Defines, "__s390x__");
    }
    virtual void getTargetBuiltins(const Builtin::Info *&Records,
                                   unsigned &NumRecords) const {
      // FIXME: Implement.
      Records = 0;
      NumRecords = 0;
    }

    virtual void getDefaultLangOptions(LangOptions &Opts) {
      TargetInfo::getDefaultLangOptions(Opts);
      Opts.CharIsSigned = false;
    }

    virtual void getGCCRegNames(const char * const *&Names,
                                unsigned &NumNames) const;
    virtual void getGCCRegAliases(const GCCRegAlias *&Aliases,
                                  unsigned &NumAliases) const {
      // No aliases.
      Aliases = 0;
      NumAliases = 0;
    }
    virtual bool validateAsmConstraint(const char *&Name,
                                       TargetInfo::ConstraintInfo &info) const {
      // FIXME: implement
      return true;
    }
    virtual const char *getClobbers() const {
      // FIXME: Is this really right?
      return "";
    }
    virtual const char *getVAListDeclaration() const {
      // FIXME: implement
      return "typedef char* __builtin_va_list;";
   }
  };

  const char * const SystemZTargetInfo::GCCRegNames[] = {
    "r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7",
    "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"
  };

  void SystemZTargetInfo::getGCCRegNames(const char * const *&Names,
                                         unsigned &NumNames) const {
    Names = GCCRegNames;
    NumNames = llvm::array_lengthof(GCCRegNames);
  }
}

namespace {
  class BlackfinTargetInfo : public TargetInfo {
    static const char * const GCCRegNames[];
  public:
    BlackfinTargetInfo(const std::string& triple) : TargetInfo(triple) {
      TLSSupported = false;
      DoubleAlign = 32;
      LongLongAlign = 32;
      LongDoubleAlign = 32;
      DescriptionString = "e-p:32:32-i64:32-f64:32";
    }

    virtual void getTargetDefines(const LangOptions &Opts,
                                  std::vector<char> &Defines) const {
      DefineStd(Defines, "bfin", Opts);
      DefineStd(Defines, "BFIN", Opts);
      Define(Defines, "__ADSPBLACKFIN__");
      // FIXME: This one is really dependent on -mcpu
      Define(Defines, "__ADSPLPBLACKFIN__");
      // FIXME: Add cpu-dependent defines and __SILICON_REVISION__
    }

    virtual void getTargetBuiltins(const Builtin::Info *&Records,
                                   unsigned &NumRecords) const {
      // FIXME: Implement.
      Records = 0;
      NumRecords = 0;
    }

    virtual void getGCCRegNames(const char * const *&Names,
                                unsigned &NumNames) const;

    virtual void getGCCRegAliases(const GCCRegAlias *&Aliases,
                                  unsigned &NumAliases) const {
      // No aliases.
      Aliases = 0;
      NumAliases = 0;
    }

    virtual bool validateAsmConstraint(const char *&Name,
                                       TargetInfo::ConstraintInfo &Info) const {
      if (strchr("adzDWeABbvfcCtukxywZY", Name[0])) {
        Info.setAllowsRegister();
        return true;
      }
      return false;
    }

    virtual const char *getClobbers() const {
      return "";
    }

    virtual const char *getVAListDeclaration() const {
      return "typedef char* __builtin_va_list;";
    }
  };

  const char * const BlackfinTargetInfo::GCCRegNames[] = {
    "r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7",
    "p0", "p1", "p2", "p3", "p4", "p5", "sp", "fp",
    "i0", "i1", "i2", "i3", "b0", "b1", "b2", "b3",
    "l0", "l1", "l2", "l3", "m0", "m1", "m2", "m3",
    "a0", "a1", "cc",
    "rets", "reti", "retx", "retn", "rete", "astat", "seqstat", "usp",
    "argp", "lt0", "lt1", "lc0", "lc1", "lb0", "lb1"
  };

  void BlackfinTargetInfo::getGCCRegNames(const char * const *&Names,
                                          unsigned &NumNames) const {
    Names = GCCRegNames;
    NumNames = llvm::array_lengthof(GCCRegNames);
  }
}

namespace {

  // LLVM and Clang cannot be used directly to output native binaries for 
  // target, but is used to compile C code to llvm bitcode with correct 
  // type and alignment information.
  // 
  // TCE uses the llvm bitcode as input and uses it for generating customized 
  // target processor and program binary. TCE co-design environment is 
  // publicly available in http://tce.cs.tut.fi

  class TCETargetInfo : public TargetInfo{
  public:
    TCETargetInfo(const std::string& triple) : TargetInfo(triple) {
      TLSSupported = false;
      IntWidth = 32;
      LongWidth = LongLongWidth = 32;
      IntMaxTWidth = 32;
      PointerWidth = 32;
      IntAlign = 32;
      LongAlign = LongLongAlign = 32;
      PointerAlign = 32;
      SizeType = UnsignedInt;
      IntMaxType = SignedLong;
      UIntMaxType = UnsignedLong;
      IntPtrType = SignedInt;
      PtrDiffType = SignedInt;
      FloatWidth = 32;
      FloatAlign = 32;
      DoubleWidth = 32;
      DoubleAlign = 32;
      LongDoubleWidth = 32;
      LongDoubleAlign = 32;
      FloatFormat = &llvm::APFloat::IEEEsingle;
      DoubleFormat = &llvm::APFloat::IEEEsingle;
      LongDoubleFormat = &llvm::APFloat::IEEEsingle;
      DescriptionString = "E-p:32:32:32-a0:32:32"
                          "-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64"
                          "-f32:32:32-f64:32:64";
    }

    virtual void getTargetDefines(const LangOptions &Opts,
                                  std::vector<char> &Defines) const {
      DefineStd(Defines, "tce", Opts);
      Define(Defines, "__TCE__");
      Define(Defines, "__TCE_V1__");
    }
    virtual void getTargetBuiltins(const Builtin::Info *&Records,
                                   unsigned &NumRecords) const {}
    virtual const char *getClobbers() const {
      return "";
    }
    virtual const char *getVAListDeclaration() const {
      return "typedef void* __builtin_va_list;";
    }
    virtual void getGCCRegNames(const char * const *&Names,
                                unsigned &NumNames) const {}
    virtual bool validateAsmConstraint(const char *&Name,
                                       TargetInfo::ConstraintInfo &info) const {
      return true;
    }
    virtual void getGCCRegAliases(const GCCRegAlias *&Aliases,
                                  unsigned &NumAliases) const {}
  };
}

//===----------------------------------------------------------------------===//
// Driver code
//===----------------------------------------------------------------------===//

/// CreateTargetInfo - Return the target info object for the specified target
/// triple.
TargetInfo* TargetInfo::CreateTargetInfo(const std::string &T) {
  llvm::Triple Triple(T);
  llvm::Triple::OSType os = Triple.getOS();

  switch (Triple.getArch()) {
  default:
    return NULL;

  case llvm::Triple::arm:
    switch (os) {
    case llvm::Triple::Darwin:
      return new DarwinARMTargetInfo(T);
    case llvm::Triple::FreeBSD:
      return new FreeBSDTargetInfo<ARMTargetInfo>(T);
    default:
      return new ARMTargetInfo(T);
    }

  case llvm::Triple::bfin:
    return new BlackfinTargetInfo(T);

  case llvm::Triple::msp430:
    return new MSP430TargetInfo(T);

  case llvm::Triple::pic16:
    return new PIC16TargetInfo(T);

  case llvm::Triple::ppc:
    if (os == llvm::Triple::Darwin)
      return new DarwinTargetInfo<PPCTargetInfo>(T);
    return new PPC32TargetInfo(T);

  case llvm::Triple::ppc64:
    if (os == llvm::Triple::Darwin)
      return new DarwinTargetInfo<PPC64TargetInfo>(T);
    return new PPC64TargetInfo(T);

  case llvm::Triple::sparc:
    if (os == llvm::Triple::Solaris)
      return new SolarisSparcV8TargetInfo(T);
    return new SparcV8TargetInfo(T);

  case llvm::Triple::systemz:
    return new SystemZTargetInfo(T);

  case llvm::Triple::tce:
    return new TCETargetInfo(T);

  case llvm::Triple::x86:
    switch (os) {
    case llvm::Triple::Darwin:
      return new DarwinI386TargetInfo(T);
    case llvm::Triple::Linux:
      return new LinuxTargetInfo<X86_32TargetInfo>(T);
    case llvm::Triple::DragonFly:
      return new DragonFlyBSDTargetInfo<X86_32TargetInfo>(T);
    case llvm::Triple::NetBSD:
      return new NetBSDTargetInfo<X86_32TargetInfo>(T);
    case llvm::Triple::OpenBSD:
      return new OpenBSDI386TargetInfo(T);
    case llvm::Triple::FreeBSD:
      return new FreeBSDTargetInfo<X86_32TargetInfo>(T);
    case llvm::Triple::Solaris:
      return new SolarisTargetInfo<X86_32TargetInfo>(T);
    case llvm::Triple::Cygwin:
    case llvm::Triple::MinGW32:
    case llvm::Triple::MinGW64:
    case llvm::Triple::Win32:
      return new WindowsX86_32TargetInfo(T);
    default:
      return new X86_32TargetInfo(T);
    }

  case llvm::Triple::x86_64:
    switch (os) {
    case llvm::Triple::Darwin:
      return new DarwinX86_64TargetInfo(T);
    case llvm::Triple::Linux:
      return new LinuxTargetInfo<X86_64TargetInfo>(T);
    case llvm::Triple::NetBSD:
      return new NetBSDTargetInfo<X86_64TargetInfo>(T);
    case llvm::Triple::OpenBSD:
      return new OpenBSDX86_64TargetInfo(T);
    case llvm::Triple::FreeBSD:
      return new FreeBSDTargetInfo<X86_64TargetInfo>(T);
    case llvm::Triple::Solaris:
      return new SolarisTargetInfo<X86_64TargetInfo>(T);
    default:
      return new X86_64TargetInfo(T);
    }
  }
}
