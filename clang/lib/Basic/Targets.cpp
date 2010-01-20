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

#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/MacroBuilder.h"
#include "clang/Basic/TargetBuiltins.h"
#include "clang/Basic/TargetOptions.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Triple.h"
#include "llvm/MC/MCSectionMachO.h"
#include <algorithm>
using namespace clang;

//===----------------------------------------------------------------------===//
//  Common code shared among targets.
//===----------------------------------------------------------------------===//

/// DefineStd - Define a macro name and standard variants.  For example if
/// MacroName is "unix", then this will define "__unix", "__unix__", and "unix"
/// when in GNU mode.
static void DefineStd(MacroBuilder &Builder, llvm::StringRef MacroName,
                      const LangOptions &Opts) {
  assert(MacroName[0] != '_' && "Identifier should be in the user's namespace");

  // If in GNU mode (e.g. -std=gnu99 but not -std=c99) define the raw identifier
  // in the user's namespace.
  if (Opts.GNUMode)
    Builder.defineMacro(MacroName);

  // Define __unix.
  Builder.defineMacro("__" + MacroName);

  // Define __unix__.
  Builder.defineMacro("__" + MacroName + "__");
}

//===----------------------------------------------------------------------===//
// Defines specific to certain operating systems.
//===----------------------------------------------------------------------===//

namespace {
template<typename TgtInfo>
class OSTargetInfo : public TgtInfo {
protected:
  virtual void getOSDefines(const LangOptions &Opts, const llvm::Triple &Triple,
                            MacroBuilder &Builder) const=0;
public:
  OSTargetInfo(const std::string& triple) : TgtInfo(triple) {}
  virtual void getTargetDefines(const LangOptions &Opts,
                                MacroBuilder &Builder) const {
    TgtInfo::getTargetDefines(Opts, Builder);
    getOSDefines(Opts, TgtInfo::getTriple(), Builder);
  }

};
} // end anonymous namespace


static void getDarwinDefines(MacroBuilder &Builder, const LangOptions &Opts) {
  Builder.defineMacro("__APPLE_CC__", "5621");
  Builder.defineMacro("__APPLE__");
  Builder.defineMacro("__MACH__");
  Builder.defineMacro("OBJC_NEW_PROPERTIES");

  // __weak is always defined, for use in blocks and with objc pointers.
  Builder.defineMacro("__weak", "__attribute__((objc_gc(weak)))");

  // Darwin defines __strong even in C mode (just to nothing).
  if (!Opts.ObjC1 || Opts.getGCMode() == LangOptions::NonGC)
    Builder.defineMacro("__strong", "");
  else
    Builder.defineMacro("__strong", "__attribute__((objc_gc(strong)))");

  if (Opts.Static)
    Builder.defineMacro("__STATIC__");
  else
    Builder.defineMacro("__DYNAMIC__");

  if (Opts.POSIXThreads)
    Builder.defineMacro("_REENTRANT");
}

static void getDarwinOSXDefines(MacroBuilder &Builder,
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
  Builder.defineMacro("__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__",
                      MacOSXStr);
}

static void getDarwinIPhoneOSDefines(MacroBuilder &Builder,
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
  Builder.defineMacro("__ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__",
                      iPhoneOSStr);
}

namespace {
template<typename Target>
class DarwinTargetInfo : public OSTargetInfo<Target> {
protected:
  virtual void getOSDefines(const LangOptions &Opts, const llvm::Triple &Triple,
                            MacroBuilder &Builder) const {
    getDarwinDefines(Builder, Opts);
    getDarwinOSXDefines(Builder, Triple);
  }

public:
  DarwinTargetInfo(const std::string& triple) :
    OSTargetInfo<Target>(triple) {
      this->TLSSupported = false;
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
                            MacroBuilder &Builder) const {
    // DragonFly defines; list based off of gcc output
    Builder.defineMacro("__DragonFly__");
    Builder.defineMacro("__DragonFly_cc_version", "100001");
    Builder.defineMacro("__ELF__");
    Builder.defineMacro("__KPRINTF_ATTRIBUTE__");
    Builder.defineMacro("__tune_i386__");
    DefineStd(Builder, "unix", Opts);
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
                            MacroBuilder &Builder) const {
    // FreeBSD defines; list based off of gcc output

    // FIXME: Move version number handling to llvm::Triple.
    const char *FreeBSD = strstr(Triple.getTriple().c_str(),
                                 "-freebsd");
    FreeBSD += strlen("-freebsd");
    char release[] = "X";
    release[0] = FreeBSD[0];
    char version[] = "X00001";
    version[0] = FreeBSD[0];

    Builder.defineMacro("__FreeBSD__", release);
    Builder.defineMacro("__FreeBSD_cc_version", version);
    Builder.defineMacro("__KPRINTF_ATTRIBUTE__");
    DefineStd(Builder, "unix", Opts);
    Builder.defineMacro("__ELF__");
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
                            MacroBuilder &Builder) const {
    // Linux defines; list based off of gcc output
    DefineStd(Builder, "unix", Opts);
    DefineStd(Builder, "linux", Opts);
    Builder.defineMacro("__gnu_linux__");
    Builder.defineMacro("__ELF__");
    if (Opts.POSIXThreads)
      Builder.defineMacro("_REENTRANT");
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
                            MacroBuilder &Builder) const {
    // NetBSD defines; list based off of gcc output
    Builder.defineMacro("__NetBSD__");
    Builder.defineMacro("__unix__");
    Builder.defineMacro("__ELF__");
    if (Opts.POSIXThreads)
      Builder.defineMacro("_POSIX_THREADS");
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
                            MacroBuilder &Builder) const {
    // OpenBSD defines; list based off of gcc output

    Builder.defineMacro("__OpenBSD__");
    DefineStd(Builder, "unix", Opts);
    Builder.defineMacro("__ELF__");
    if (Opts.POSIXThreads)
      Builder.defineMacro("_POSIX_THREADS");
  }
public:
  OpenBSDTargetInfo(const std::string &triple)
    : OSTargetInfo<Target>(triple) {}
};

// PSP Target
template<typename Target>
class PSPTargetInfo : public OSTargetInfo<Target> {
protected:
  virtual void getOSDefines(const LangOptions &Opts, const llvm::Triple &Triple,
                            MacroBuilder &Builder) const {
    // PSP defines; list based on the output of the pspdev gcc toolchain.
    Builder.defineMacro("PSP");
    Builder.defineMacro("_PSP");
    Builder.defineMacro("__psp__");
    Builder.defineMacro("__ELF__");
  }
public:
  PSPTargetInfo(const std::string& triple)
    : OSTargetInfo<Target>(triple) {
    this->UserLabelPrefix = "";
  }
};

// PS3 PPU Target
template<typename Target>
class PS3PPUTargetInfo : public OSTargetInfo<Target> {
protected:
  virtual void getOSDefines(const LangOptions &Opts, const llvm::Triple &Triple,
                            MacroBuilder &Builder) const {
    // PS3 PPU defines.
    Builder.defineMacro("__PPU__");
    Builder.defineMacro("__CELLOS_LV2__");
    Builder.defineMacro("__ELF__");
    Builder.defineMacro("__LP32__");
  }
public:
  PS3PPUTargetInfo(const std::string& triple)
    : OSTargetInfo<Target>(triple) {
    this->UserLabelPrefix = "";
    this->LongWidth = this->LongAlign = this->PointerWidth = this->PointerAlign = 32;
    this->SizeType = TargetInfo::UnsignedInt;
  }
};

// FIXME: Need a real SPU target.
// PS3 SPU Target
template<typename Target>
class PS3SPUTargetInfo : public OSTargetInfo<Target> {
protected:
  virtual void getOSDefines(const LangOptions &Opts, const llvm::Triple &Triple,
                            MacroBuilder &Builder) const {
    // PS3 PPU defines.
    Builder.defineMacro("__SPU__");
    Builder.defineMacro("__ELF__");
  }
public:
  PS3SPUTargetInfo(const std::string& triple)
    : OSTargetInfo<Target>(triple) {
    this->UserLabelPrefix = "";
  }
};

// AuroraUX target
template<typename Target>
class AuroraUXTargetInfo : public OSTargetInfo<Target> {
protected:
  virtual void getOSDefines(const LangOptions &Opts, const llvm::Triple &Triple,
                            MacroBuilder &Builder) const {
    DefineStd(Builder, "sun", Opts);
    DefineStd(Builder, "unix", Opts);
    Builder.defineMacro("__ELF__");
    Builder.defineMacro("__svr4__");
    Builder.defineMacro("__SVR4");
  }
public:
  AuroraUXTargetInfo(const std::string& triple)
    : OSTargetInfo<Target>(triple) {
    this->UserLabelPrefix = "";
    this->WCharType = this->SignedLong;
    // FIXME: WIntType should be SignedLong
  }
};

// Solaris target
template<typename Target>
class SolarisTargetInfo : public OSTargetInfo<Target> {
protected:
  virtual void getOSDefines(const LangOptions &Opts, const llvm::Triple &Triple,
                            MacroBuilder &Builder) const {
    DefineStd(Builder, "sun", Opts);
    DefineStd(Builder, "unix", Opts);
    Builder.defineMacro("__ELF__");
    Builder.defineMacro("__svr4__");
    Builder.defineMacro("__SVR4");
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
                                MacroBuilder &Builder) const;

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
                                     MacroBuilder &Builder) const {
  // Target identification.
  Builder.defineMacro("__ppc__");
  Builder.defineMacro("_ARCH_PPC");
  Builder.defineMacro("__POWERPC__");
  if (PointerWidth == 64) {
    Builder.defineMacro("_ARCH_PPC64");
    Builder.defineMacro("_LP64");
    Builder.defineMacro("__LP64__");
    Builder.defineMacro("__ppc64__");
  } else {
    Builder.defineMacro("__ppc__");
  }

  // Target properties.
  Builder.defineMacro("_BIG_ENDIAN");
  Builder.defineMacro("__BIG_ENDIAN__");

  // Subtarget options.
  Builder.defineMacro("__NATURAL_ALIGNMENT__");
  Builder.defineMacro("__REGISTER_PREFIX__", "");

  // FIXME: Should be controlled by command line option.
  Builder.defineMacro("__LONG_DOUBLE_128__");
  
  if (Opts.AltiVec) {
    Builder.defineMacro("__VEC__", "10206");
    Builder.defineMacro("__ALTIVEC__");
  }
}


const char * const PPCTargetInfo::GCCRegNames[] = {
  "r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7",
  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
  "r16", "r17", "r18", "r19", "r20", "r21", "r22", "r23",
  "r24", "r25", "r26", "r27", "r28", "r29", "r30", "r31",
  "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7",
  "f8", "f9", "f10", "f11", "f12", "f13", "f14", "f15",
  "f16", "f17", "f18", "f19", "f20", "f21", "f22", "f23",
  "f24", "f25", "f26", "f27", "f28", "f29", "f30", "f31",
  "mq", "lr", "ctr", "ap",
  "cr0", "cr1", "cr2", "cr3", "cr4", "cr5", "cr6", "cr7",
  "xer",
  "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
  "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
  "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
  "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31",
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
  { { "0" }, "r0" },
  { { "1"}, "r1" },
  { { "2" }, "r2" },
  { { "3" }, "r3" },
  { { "4" }, "r4" },
  { { "5" }, "r5" },
  { { "6" }, "r6" },
  { { "7" }, "r7" },
  { { "8" }, "r8" },
  { { "9" }, "r9" },
  { { "10" }, "r10" },
  { { "11" }, "r11" },
  { { "12" }, "r12" },
  { { "13" }, "r13" },
  { { "14" }, "r14" },
  { { "15" }, "r15" },
  { { "16" }, "r16" },
  { { "17" }, "r17" },
  { { "18" }, "r18" },
  { { "19" }, "r19" },
  { { "20" }, "r20" },
  { { "21" }, "r21" },
  { { "22" }, "r22" },
  { { "23" }, "r23" },
  { { "24" }, "r24" },
  { { "25" }, "r25" },
  { { "26" }, "r26" },
  { { "27" }, "r27" },
  { { "28" }, "r28" },
  { { "29" }, "r29" },
  { { "30" }, "r30" },
  { { "31" }, "r31" },
  { { "fr0" }, "f0" },
  { { "fr1" }, "f1" },
  { { "fr2" }, "f2" },
  { { "fr3" }, "f3" },
  { { "fr4" }, "f4" },
  { { "fr5" }, "f5" },
  { { "fr6" }, "f6" },
  { { "fr7" }, "f7" },
  { { "fr8" }, "f8" },
  { { "fr9" }, "f9" },
  { { "fr10" }, "f10" },
  { { "fr11" }, "f11" },
  { { "fr12" }, "f12" },
  { { "fr13" }, "f13" },
  { { "fr14" }, "f14" },
  { { "fr15" }, "f15" },
  { { "fr16" }, "f16" },
  { { "fr17" }, "f17" },
  { { "fr18" }, "f18" },
  { { "fr19" }, "f19" },
  { { "fr20" }, "f20" },
  { { "fr21" }, "f21" },
  { { "fr22" }, "f22" },
  { { "fr23" }, "f23" },
  { { "fr24" }, "f24" },
  { { "fr25" }, "f25" },
  { { "fr26" }, "f26" },
  { { "fr27" }, "f27" },
  { { "fr28" }, "f28" },
  { { "fr29" }, "f29" },
  { { "fr30" }, "f30" },
  { { "fr31" }, "f31" },
  { { "cc" }, "cr0" },
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
                        "i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32";
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
                        "i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32:64";
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

static const char* const GCCRegNames[] = {
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
                                MacroBuilder &Builder) const;
  virtual bool setFeatureEnabled(llvm::StringMap<bool> &Features,
                                 const std::string &Name,
                                 bool Enabled) const;
  virtual void getDefaultFeatures(const std::string &CPU,
                                  llvm::StringMap<bool> &Features) const;
  virtual void HandleTargetFeatures(std::vector<std::string> &Features);
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
void X86TargetInfo::HandleTargetFeatures(std::vector<std::string> &Features) {
  // Remember the maximum enabled sselevel.
  for (unsigned i = 0, e = Features.size(); i !=e; ++i) {
    // Ignore disabled features.
    if (Features[i][0] == '-')
      continue;

    assert(Features[i][0] == '+' && "Invalid target feature!");
    X86SSEEnum Level = llvm::StringSwitch<X86SSEEnum>(Features[i].substr(1))
      .Case("sse42", SSE42)
      .Case("sse41", SSE41)
      .Case("ssse3", SSSE3)
      .Case("sse2", SSE2)
      .Case("sse", SSE1)
      .Case("mmx", MMX)
      .Default(NoMMXSSE);
    SSELevel = std::max(SSELevel, Level);
  }
}

/// X86TargetInfo::getTargetDefines - Return a set of the X86-specific #defines
/// that are not tied to a specific subtarget.
void X86TargetInfo::getTargetDefines(const LangOptions &Opts,
                                     MacroBuilder &Builder) const {
  // Target identification.
  if (PointerWidth == 64) {
    Builder.defineMacro("_LP64");
    Builder.defineMacro("__LP64__");
    Builder.defineMacro("__amd64__");
    Builder.defineMacro("__amd64");
    Builder.defineMacro("__x86_64");
    Builder.defineMacro("__x86_64__");
  } else {
    DefineStd(Builder, "i386", Opts);
  }

  // Target properties.
  Builder.defineMacro("__LITTLE_ENDIAN__");

  // Subtarget options.
  Builder.defineMacro("__nocona");
  Builder.defineMacro("__nocona__");
  Builder.defineMacro("__tune_nocona__");
  Builder.defineMacro("__REGISTER_PREFIX__", "");

  // Define __NO_MATH_INLINES on linux/x86 so that we don't get inline
  // functions in glibc header files that use FP Stack inline asm which the
  // backend can't deal with (PR879).
  Builder.defineMacro("__NO_MATH_INLINES");

  // Each case falls through to the previous one here.
  switch (SSELevel) {
  case SSE42:
    Builder.defineMacro("__SSE4_2__");
  case SSE41:
    Builder.defineMacro("__SSE4_1__");
  case SSSE3:
    Builder.defineMacro("__SSSE3__");
  case SSE3:
    Builder.defineMacro("__SSE3__");
  case SSE2:
    Builder.defineMacro("__SSE2__");
    Builder.defineMacro("__SSE2_MATH__");  // -mfp-math=sse always implied.
  case SSE1:
    Builder.defineMacro("__SSE__");
    Builder.defineMacro("__SSE_MATH__");   // -mfp-math=sse always implied.
  case MMX:
    Builder.defineMacro("__MMX__");
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
                        "a0:0:64-f80:32:32-n8:16:32";
    SizeType = UnsignedInt;
    PtrDiffType = SignedInt;
    IntPtrType = SignedInt;
    RegParmMax = 3;
  }
  virtual const char *getVAListDeclaration() const {
    return "typedef char* __builtin_va_list;";
  }
  
  int getEHDataRegisterNumber(unsigned RegNo) const {
    if (RegNo == 0) return 0;
    if (RegNo == 1) return 2;
    return -1;
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
                        "a0:0:64-f80:128:128-n8:16:32";
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
    DoubleAlign = LongLongAlign = 64;
    DescriptionString = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-"
                        "i64:64:64-f32:32:32-f64:64:64-f80:128:128-v64:64:64-"
                        "v128:128:128-a0:0:64-f80:32:32-n8:16:32";
  }
  virtual void getTargetDefines(const LangOptions &Opts,
                                MacroBuilder &Builder) const {
    X86_32TargetInfo::getTargetDefines(Opts, Builder);
    // This list is based off of the the list of things MingW defines
    Builder.defineMacro("_WIN32");
    DefineStd(Builder, "WIN32", Opts);
    DefineStd(Builder, "WINNT", Opts);
    Builder.defineMacro("_X86_");
  }
};
} // end anonymous namespace

namespace {

// x86-32 Windows Visual Studio target
class VisualStudioWindowsX86_32TargetInfo : public WindowsX86_32TargetInfo {
public:
  VisualStudioWindowsX86_32TargetInfo(const std::string& triple)
    : WindowsX86_32TargetInfo(triple) {
  }
  virtual void getTargetDefines(const LangOptions &Opts,
                                MacroBuilder &Builder) const {
    WindowsX86_32TargetInfo::getTargetDefines(Opts, Builder);
    // The value of the following reflects processor type.
    // 300=386, 400=486, 500=Pentium, 600=Blend (default)
    // We lost the original triple, so we use the default.
    Builder.defineMacro("_M_IX86", "600");
  }
};
} // end anonymous namespace

namespace {
// x86-32 MinGW target
class MinGWX86_32TargetInfo : public WindowsX86_32TargetInfo {
public:
  MinGWX86_32TargetInfo(const std::string& triple)
    : WindowsX86_32TargetInfo(triple) {
  }
  virtual void getTargetDefines(const LangOptions &Opts,
                                MacroBuilder &Builder) const {
    WindowsX86_32TargetInfo::getTargetDefines(Opts, Builder);
    Builder.defineMacro("__MSVCRT__");
    Builder.defineMacro("__MINGW32__");
    Builder.defineMacro("__declspec", "__declspec");
  }
};
} // end anonymous namespace

namespace {
// x86-32 Cygwin target
class CygwinX86_32TargetInfo : public X86_32TargetInfo {
public:
  CygwinX86_32TargetInfo(const std::string& triple)
    : X86_32TargetInfo(triple) {
    TLSSupported = false;
    WCharType = UnsignedShort;
    DoubleAlign = LongLongAlign = 64;
    DescriptionString = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-"
                        "i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-"
                        "a0:0:64-f80:32:32-n8:16:32";
  }
  virtual void getTargetDefines(const LangOptions &Opts,
                                MacroBuilder &Builder) const {
    X86_32TargetInfo::getTargetDefines(Opts, Builder);
    Builder.defineMacro("__CYGWIN__");
    Builder.defineMacro("__CYGWIN32__");
    DefineStd(Builder, "unix", Opts);
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
                        "a0:0:64-s0:64:64-f80:128:128-n8:16:32:64";
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
  
  int getEHDataRegisterNumber(unsigned RegNo) const {
    if (RegNo == 0) return 0;
    if (RegNo == 1) return 1;
    return -1;
  }
};
} // end anonymous namespace

namespace {
// x86-64 Windows target
class WindowsX86_64TargetInfo : public X86_64TargetInfo {
public:
  WindowsX86_64TargetInfo(const std::string& triple)
    : X86_64TargetInfo(triple) {
    TLSSupported = false;
    WCharType = UnsignedShort;
    LongWidth = LongAlign = 32;
    DoubleAlign = LongLongAlign = 64;
  }
  virtual void getTargetDefines(const LangOptions &Opts,
                                MacroBuilder &Builder) const {
    X86_64TargetInfo::getTargetDefines(Opts, Builder);
    Builder.defineMacro("_WIN64");
    DefineStd(Builder, "WIN64", Opts);
  }
};
} // end anonymous namespace

namespace {
// x86-64 Windows Visual Studio target
class VisualStudioWindowsX86_64TargetInfo : public WindowsX86_64TargetInfo {
public:
  VisualStudioWindowsX86_64TargetInfo(const std::string& triple)
    : WindowsX86_64TargetInfo(triple) {
  }
  virtual void getTargetDefines(const LangOptions &Opts,
                                MacroBuilder &Builder) const {
    WindowsX86_64TargetInfo::getTargetDefines(Opts, Builder);
    Builder.defineMacro("_M_X64");
  }
  virtual const char *getVAListDeclaration() const {
    return "typedef char* va_list;";
  }
};
} // end anonymous namespace

namespace {
// x86-64 MinGW target
class MinGWX86_64TargetInfo : public WindowsX86_64TargetInfo {
public:
  MinGWX86_64TargetInfo(const std::string& triple)
    : WindowsX86_64TargetInfo(triple) {
  }
  virtual void getTargetDefines(const LangOptions &Opts,
                                MacroBuilder &Builder) const {
    WindowsX86_64TargetInfo::getTargetDefines(Opts, Builder);
    Builder.defineMacro("__MSVCRT__");
    Builder.defineMacro("__MINGW64__");
    Builder.defineMacro("__declspec");
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
  // Possible FPU choices.
  enum FPUMode {
    NoFPU,
    VFP2FPU,
    VFP3FPU,
    NeonFPU
  };

  static bool FPUModeIsVFP(FPUMode Mode) {
    return Mode >= VFP2FPU && Mode <= NeonFPU;
  }

  static const TargetInfo::GCCRegAlias GCCRegAliases[];
  static const char * const GCCRegNames[];

  std::string ABI, CPU;

  unsigned FPU : 3;

  unsigned IsThumb : 1;

  // Initialized via features.
  unsigned SoftFloat : 1;
  unsigned SoftFloatABI : 1;

public:
  ARMTargetInfo(const std::string &TripleStr)
    : TargetInfo(TripleStr), ABI("aapcs-linux"), CPU("arm1136j-s")
  {
    SizeType = UnsignedInt;
    PtrDiffType = SignedInt;

    // FIXME: Should we just treat this as a feature?
    IsThumb = getTriple().getArchName().startswith("thumb");
    if (IsThumb) {
      DescriptionString = ("e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-"
                           "i64:64:64-f32:32:32-f64:64:64-"
                           "v64:64:64-v128:128:128-a0:0:32-n32");
    } else {
      DescriptionString = ("e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-"
                           "i64:64:64-f32:32:32-f64:64:64-"
                           "v64:64:64-v128:128:128-a0:0:64-n32");
    }
  }
  virtual const char *getABI() const { return ABI.c_str(); }
  virtual bool setABI(const std::string &Name) {
    ABI = Name;

    // The defaults (above) are for AAPCS, check if we need to change them.
    //
    // FIXME: We need support for -meabi... we could just mangle it into the
    // name.
    if (Name == "apcs-gnu") {
      DoubleAlign = LongLongAlign = 32;
      SizeType = UnsignedLong;

      if (IsThumb) {
        DescriptionString = ("e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-"
                             "i64:32:32-f32:32:32-f64:32:32-"
                             "v64:64:64-v128:128:128-a0:0:32-n32");
      } else {
        DescriptionString = ("e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-"
                             "i64:32:32-f32:32:32-f64:32:32-"
                             "v64:64:64-v128:128:128-a0:0:64-n32");
      }

      // FIXME: Override "preferred align" for double and long long.
    } else if (Name == "aapcs") {
      // FIXME: Enumerated types are variable width in straight AAPCS.
    } else if (Name == "aapcs-linux") {
      ;
    } else
      return false;

    return true;
  }

  void getDefaultFeatures(const std::string &CPU,
                          llvm::StringMap<bool> &Features) const {
    // FIXME: This should not be here.
    Features["vfp2"] = false;
    Features["vfp3"] = false;
    Features["neon"] = false;

    if (CPU == "arm1136jf-s" || CPU == "arm1176jzf-s" || CPU == "mpcore")
      Features["vfp2"] = true;
    else if (CPU == "cortex-a8" || CPU == "cortex-a9")
      Features["neon"] = true;
  }
  
  virtual bool setFeatureEnabled(llvm::StringMap<bool> &Features,
                                 const std::string &Name,
                                 bool Enabled) const {
    if (Name == "soft-float" || Name == "soft-float-abi") {
      Features[Name] = Enabled;
    } else if (Name == "vfp2" || Name == "vfp3" || Name == "neon") {
      // These effectively are a single option, reset them when any is enabled.
      if (Enabled)
        Features["vfp2"] = Features["vfp3"] = Features["neon"] = false;
      Features[Name] = Enabled;
    } else
      return false;

    return true;
  }

  virtual void HandleTargetFeatures(std::vector<std::string> &Features) {
    FPU = NoFPU;
    SoftFloat = SoftFloatABI = false;
    for (unsigned i = 0, e = Features.size(); i != e; ++i) {
      if (Features[i] == "+soft-float")
        SoftFloat = true;
      else if (Features[i] == "+soft-float-abi")
        SoftFloatABI = true;
      else if (Features[i] == "+vfp2")
        FPU = VFP2FPU;
      else if (Features[i] == "+vfp3")
        FPU = VFP3FPU;
      else if (Features[i] == "+neon")
        FPU = NeonFPU;
    }

    // Remove front-end specific options which the backend handles differently.
    std::vector<std::string>::iterator it;
    it = std::find(Features.begin(), Features.end(), "+soft-float");
    if (it != Features.end())
      Features.erase(it);
    it = std::find(Features.begin(), Features.end(), "+soft-float-abi");
    if (it != Features.end())
      Features.erase(it);
  }

  static const char *getCPUDefineSuffix(llvm::StringRef Name) {
    return llvm::StringSwitch<const char*>(Name)
      .Cases("arm8", "arm810", "4")
      .Cases("strongarm", "strongarm110", "strongarm1100", "strongarm1110", "4")
      .Cases("arm7tdmi", "arm7tdmi-s", "arm710t", "arm720t", "arm9", "4T")
      .Cases("arm9tdmi", "arm920", "arm920t", "arm922t", "arm940t", "4T")
      .Case("ep9312", "4T")
      .Cases("arm10tdmi", "arm1020t", "5T")
      .Cases("arm9e", "arm946e-s", "arm966e-s", "arm968e-s", "5TE")
      .Case("arm926ej-s", "5TEJ")
      .Cases("arm10e", "arm1020e", "arm1022e", "5TE")
      .Cases("xscale", "iwmmxt", "5TE")
      .Case("arm1136j-s", "6J")
      .Cases("arm1176jz-s", "arm1176jzf-s", "6ZK")
      .Cases("arm1136jf-s", "mpcorenovfp", "mpcore", "6K")
      .Cases("arm1156t2-s", "arm1156t2f-s", "6T2")
      .Cases("cortex-a8", "cortex-a9", "7A")
      .Default(0);
  }
  virtual bool setCPU(const std::string &Name) {
    if (!getCPUDefineSuffix(Name))
      return false;

    CPU = Name;
    return true;
  }
  virtual void getTargetDefines(const LangOptions &Opts,
                                MacroBuilder &Builder) const {
    // Target identification.
    Builder.defineMacro("__arm");
    Builder.defineMacro("__arm__");

    // Target properties.
    Builder.defineMacro("__ARMEL__");
    Builder.defineMacro("__LITTLE_ENDIAN__");
    Builder.defineMacro("__REGISTER_PREFIX__", "");

    llvm::StringRef CPUArch = getCPUDefineSuffix(CPU);
    Builder.defineMacro("__ARM_ARCH_" + CPUArch + "__");

    // Subtarget options.

    // FIXME: It's more complicated than this and we don't really support
    // interworking.
    if ('5' <= CPUArch[0] && CPUArch[0] <= '7')
      Builder.defineMacro("__THUMB_INTERWORK__");

    if (ABI == "aapcs" || ABI == "aapcs-linux")
      Builder.defineMacro("__ARM_EABI__");

    if (SoftFloat)
      Builder.defineMacro("__SOFTFP__");

    if (CPU == "xscale")
      Builder.defineMacro("__XSCALE__");

    bool IsThumb2 = IsThumb && (CPUArch == "6T2" || CPUArch.startswith("7"));
    if (IsThumb) {
      Builder.defineMacro("__THUMBEL__");
      Builder.defineMacro("__thumb__");
      if (IsThumb2)
        Builder.defineMacro("__thumb2__");
    }

    // Note, this is always on in gcc, even though it doesn't make sense.
    Builder.defineMacro("__APCS_32__");

    if (FPUModeIsVFP((FPUMode) FPU))
      Builder.defineMacro("__VFP_FP__");

    // This only gets set when Neon instructions are actually available, unlike
    // the VFP define, hence the soft float and arch check. This is subtly
    // different from gcc, we follow the intent which was that it should be set
    // when Neon instructions are actually available.
    if (FPU == NeonFPU && !SoftFloat && IsThumb2)
      Builder.defineMacro("__ARM_NEON__");

    if (getTriple().getOS() == llvm::Triple::Darwin)
      Builder.defineMacro("__USING_SJLJ_EXCEPTIONS__");
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
                              unsigned &NumNames) const;
  virtual void getGCCRegAliases(const GCCRegAlias *&Aliases,
                                unsigned &NumAliases) const;
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

const char * const ARMTargetInfo::GCCRegNames[] = {
  "r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7",
  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"
};

void ARMTargetInfo::getGCCRegNames(const char * const *&Names,
                                       unsigned &NumNames) const {
  Names = GCCRegNames;
  NumNames = llvm::array_lengthof(GCCRegNames);
}

const TargetInfo::GCCRegAlias ARMTargetInfo::GCCRegAliases[] = {

  { { "a1" }, "r0" },
  { { "a2" }, "r1" },
  { { "a3" }, "r2" },
  { { "a4" }, "r3" },
  { { "v1" }, "r4" },
  { { "v2" }, "r5" },
  { { "v3" }, "r6" },
  { { "v4" }, "r7" },
  { { "v5" }, "r8" },
  { { "v6", "rfp" }, "r9" },
  { { "sl" }, "r10" },
  { { "fp" }, "r11" },
  { { "ip" }, "r12" },
  { { "sp" }, "r13" },
  { { "lr" }, "r14" },
  { { "pc" }, "r15" },
};

void ARMTargetInfo::getGCCRegAliases(const GCCRegAlias *&Aliases,
                                       unsigned &NumAliases) const {
  Aliases = GCCRegAliases;
  NumAliases = llvm::array_lengthof(GCCRegAliases);
}
} // end anonymous namespace.


namespace {
class DarwinARMTargetInfo :
  public DarwinTargetInfo<ARMTargetInfo> {
protected:
  virtual void getOSDefines(const LangOptions &Opts, const llvm::Triple &Triple,
                            MacroBuilder &Builder) const {
    getDarwinDefines(Builder, Opts);
    getDarwinIPhoneOSDefines(Builder, Triple);
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
                        "i64:64:64-f32:32:32-f64:64:64-v64:64:64-n32";
  }
  virtual void getTargetDefines(const LangOptions &Opts,
                                MacroBuilder &Builder) const {
    DefineStd(Builder, "sparc", Opts);
    Builder.defineMacro("__sparcv8");
    Builder.defineMacro("__REGISTER_PREFIX__", "");
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
class AuroraUXSparcV8TargetInfo : public AuroraUXTargetInfo<SparcV8TargetInfo> {
public:
  AuroraUXSparcV8TargetInfo(const std::string& triple) :
      AuroraUXTargetInfo<SparcV8TargetInfo>(triple) {
    SizeType = UnsignedInt;
    PtrDiffType = SignedInt;
  }
};
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
      PointerWidth = 16;
      IntAlign = 8;
      LongAlign = LongLongAlign = 8;
      PointerAlign = 8;
      SizeType = UnsignedInt;
      IntMaxType = SignedLong;
      UIntMaxType = UnsignedLong;
      IntPtrType = SignedShort;
      PtrDiffType = SignedInt;
      SigAtomicType = SignedLong;
      FloatWidth = 32;
      FloatAlign = 32;
      DoubleWidth = 32;
      DoubleAlign = 32;
      LongDoubleWidth = 32;
      LongDoubleAlign = 32;
      FloatFormat = &llvm::APFloat::IEEEsingle;
      DoubleFormat = &llvm::APFloat::IEEEsingle;
      LongDoubleFormat = &llvm::APFloat::IEEEsingle;
      DescriptionString = "e-p:16:8:8-i8:8:8-i16:8:8-i32:8:8-f32:32:32-n8";

    }
    virtual uint64_t getPointerWidthV(unsigned AddrSpace) const { return 16; }
    virtual uint64_t getPointerAlignV(unsigned AddrSpace) const { return 8; }
    virtual void getTargetDefines(const LangOptions &Opts,
                                MacroBuilder &Builder) const {
      Builder.defineMacro("__pic16");
      Builder.defineMacro("rom", "__attribute__((address_space(1)))");
      Builder.defineMacro("ram", "__attribute__((address_space(0)))");
      Builder.defineMacro("_section(SectName)",
             "__attribute__((section(SectName)))");
      Builder.defineMacro("near",
             "__attribute__((section(\"Address=NEAR\")))");
      Builder.defineMacro("_address(Addr)",
             "__attribute__((section(\"Address=\"#Addr)))");
      Builder.defineMacro("_CONFIG(conf)", "asm(\"CONFIG \"#conf)");
      Builder.defineMacro("_interrupt",
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
      LongWidth = 32;
      LongLongWidth = 64;
      PointerWidth = 16;
      IntAlign = 8;
      LongAlign = LongLongAlign = 8;
      PointerAlign = 8;
      SizeType = UnsignedInt;
      IntMaxType = SignedLong;
      UIntMaxType = UnsignedLong;
      IntPtrType = SignedShort;
      PtrDiffType = SignedInt;
      SigAtomicType = SignedLong;
      DescriptionString = "e-p:16:16:16-i8:8:8-i16:16:16-i32:16:32-n8:16";
   }
    virtual void getTargetDefines(const LangOptions &Opts,
                                  MacroBuilder &Builder) const {
      Builder.defineMacro("MSP430");
      Builder.defineMacro("__MSP430__");
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
      // No target constraints for now.
      return false;
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
      DescriptionString = "E-p:64:64:64-i8:8:16-i16:16:16-i32:32:32-"
      "i64:64:64-f32:32:32-f64:64:64-f128:128:128-a0:16:16-n32:64";
   }
    virtual void getTargetDefines(const LangOptions &Opts,
                                  MacroBuilder &Builder) const {
      Builder.defineMacro("__s390__");
      Builder.defineMacro("__s390x__");
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
      DescriptionString = "e-p:32:32-i64:32-f64:32-n32";
    }

    virtual void getTargetDefines(const LangOptions &Opts,
                                  MacroBuilder &Builder) const {
      DefineStd(Builder, "bfin", Opts);
      DefineStd(Builder, "BFIN", Opts);
      Builder.defineMacro("__ADSPBLACKFIN__");
      // FIXME: This one is really dependent on -mcpu
      Builder.defineMacro("__ADSPLPBLACKFIN__");
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
                          "-f32:32:32-f64:32:64-n32";
    }

    virtual void getTargetDefines(const LangOptions &Opts,
                                  MacroBuilder &Builder) const {
      DefineStd(Builder, "tce", Opts);
      Builder.defineMacro("__TCE__");
      Builder.defineMacro("__TCE_V1__");
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

namespace {
class MipsTargetInfo : public TargetInfo {
  static const TargetInfo::GCCRegAlias GCCRegAliases[];
  static const char * const GCCRegNames[];
public:
  MipsTargetInfo(const std::string& triple) : TargetInfo(triple) {
    DescriptionString = "E-p:32:32:32-i1:8:8-i8:8:32-i16:16:32-i32:32:32-"
                        "i64:32:64-f32:32:32-f64:64:64-v64:64:64-n32";
  }
  virtual void getTargetDefines(const LangOptions &Opts,
                                MacroBuilder &Builder) const {
    DefineStd(Builder, "mips", Opts);
    Builder.defineMacro("_mips");
    DefineStd(Builder, "MIPSEB", Opts);
    Builder.defineMacro("_MIPSEB");
    Builder.defineMacro("__REGISTER_PREFIX__", "");
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
                                     TargetInfo::ConstraintInfo &Info) const {
    switch (*Name) {
    default:
    case 'r': // CPU registers.
    case 'd': // Equivalent to "r" unless generating MIPS16 code.
    case 'y': // Equivalent to "r", backwards compatibility only.
    case 'f': // floating-point registers.
      Info.setAllowsRegister();
      return true;
    }
    return false;
  }

  virtual const char *getClobbers() const {
    // FIXME: Implement!
    return "";
  }
};

const char * const MipsTargetInfo::GCCRegNames[] = {
  "$0",   "$1",   "$2",   "$3",   "$4",   "$5",   "$6",   "$7", 
  "$8",   "$9",   "$10",  "$11",  "$12",  "$13",  "$14",  "$15",
  "$16",  "$17",  "$18",  "$19",  "$20",  "$21",  "$22",  "$23",
  "$24",  "$25",  "$26",  "$27",  "$28",  "$sp",  "$fp",  "$31",
  "$f0",  "$f1",  "$f2",  "$f3",  "$f4",  "$f5",  "$f6",  "$f7",
  "$f8",  "$f9",  "$f10", "$f11", "$f12", "$f13", "$f14", "$f15",
  "$f16", "$f17", "$f18", "$f19", "$f20", "$f21", "$f22", "$f23",
  "$f24", "$f25", "$f26", "$f27", "$f28", "$f29", "$f30", "$f31",
  "hi",   "lo",   "",     "$fcc0","$fcc1","$fcc2","$fcc3","$fcc4",
  "$fcc5","$fcc6","$fcc7"
};

void MipsTargetInfo::getGCCRegNames(const char * const *&Names,
                                       unsigned &NumNames) const {
  Names = GCCRegNames;
  NumNames = llvm::array_lengthof(GCCRegNames);
}

const TargetInfo::GCCRegAlias MipsTargetInfo::GCCRegAliases[] = {
  { { "at" },  "$1" },
  { { "v0" },  "$2" },
  { { "v1" },  "$3" },
  { { "a0" },  "$4" },
  { { "a1" },  "$5" },
  { { "a2" },  "$6" },
  { { "a3" },  "$7" },
  { { "t0" },  "$8" },
  { { "t1" },  "$9" },
  { { "t2" }, "$10" },
  { { "t3" }, "$11" },
  { { "t4" }, "$12" },
  { { "t5" }, "$13" },
  { { "t6" }, "$14" },
  { { "t7" }, "$15" },
  { { "s0" }, "$16" },
  { { "s1" }, "$17" },
  { { "s2" }, "$18" },
  { { "s3" }, "$19" },
  { { "s4" }, "$20" },
  { { "s5" }, "$21" },
  { { "s6" }, "$22" },
  { { "s7" }, "$23" },
  { { "t8" }, "$24" },
  { { "t9" }, "$25" },
  { { "k0" }, "$26" },
  { { "k1" }, "$27" },
  { { "gp" }, "$28" },
  { { "sp" }, "$29" },
  { { "fp" }, "$30" },
  { { "ra" }, "$31" }
};

void MipsTargetInfo::getGCCRegAliases(const GCCRegAlias *&Aliases,
                                         unsigned &NumAliases) const {
  Aliases = GCCRegAliases;
  NumAliases = llvm::array_lengthof(GCCRegAliases);
}
} // end anonymous namespace.

namespace {
class MipselTargetInfo : public MipsTargetInfo {
public:
  MipselTargetInfo(const std::string& triple) : MipsTargetInfo(triple) {
    DescriptionString = "e-p:32:32:32-i1:8:8-i8:8:32-i16:16:32-i32:32:32-"
                        "i64:32:64-f32:32:32-f64:64:64-v64:64:64-n32";
  }

  virtual void getTargetDefines(const LangOptions &Opts,
                                MacroBuilder &Builder) const;
};

void MipselTargetInfo::getTargetDefines(const LangOptions &Opts,
                                        MacroBuilder &Builder) const {
  DefineStd(Builder, "mips", Opts);
  Builder.defineMacro("_mips");
  DefineStd(Builder, "MIPSEL", Opts);
  Builder.defineMacro("_MIPSEL");
  Builder.defineMacro("__REGISTER_PREFIX__", "");
}
} // end anonymous namespace.

//===----------------------------------------------------------------------===//
// Driver code
//===----------------------------------------------------------------------===//

static TargetInfo *AllocateTarget(const std::string &T) {
  llvm::Triple Triple(T);
  llvm::Triple::OSType os = Triple.getOS();

  switch (Triple.getArch()) {
  default:
    return NULL;

  case llvm::Triple::arm:
  case llvm::Triple::thumb:
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

  case llvm::Triple::mips:
    if (os == llvm::Triple::Psp)
      return new PSPTargetInfo<MipsTargetInfo>(T);
    if (os == llvm::Triple::Linux)
      return new LinuxTargetInfo<MipsTargetInfo>(T);
    return new MipsTargetInfo(T);

  case llvm::Triple::mipsel:
    if (os == llvm::Triple::Psp)
      return new PSPTargetInfo<MipselTargetInfo>(T);
    if (os == llvm::Triple::Linux)
      return new LinuxTargetInfo<MipselTargetInfo>(T);
    return new MipselTargetInfo(T);

  case llvm::Triple::pic16:
    return new PIC16TargetInfo(T);

  case llvm::Triple::ppc:
    if (os == llvm::Triple::Darwin)
      return new DarwinTargetInfo<PPCTargetInfo>(T);
    return new PPC32TargetInfo(T);

  case llvm::Triple::ppc64:
    if (os == llvm::Triple::Darwin)
      return new DarwinTargetInfo<PPC64TargetInfo>(T);
    else if (os == llvm::Triple::Lv2)
      return new PS3PPUTargetInfo<PPC64TargetInfo>(T);
    return new PPC64TargetInfo(T);

  case llvm::Triple::sparc:
    if (os == llvm::Triple::AuroraUX)
      return new AuroraUXSparcV8TargetInfo(T);
    if (os == llvm::Triple::Solaris)
      return new SolarisSparcV8TargetInfo(T);
    return new SparcV8TargetInfo(T);

  // FIXME: Need a real SPU target.
  case llvm::Triple::cellspu:
    return new PS3SPUTargetInfo<PPC64TargetInfo>(T);

  case llvm::Triple::systemz:
    return new SystemZTargetInfo(T);

  case llvm::Triple::tce:
    return new TCETargetInfo(T);

  case llvm::Triple::x86:
    switch (os) {
    case llvm::Triple::AuroraUX:
      return new AuroraUXTargetInfo<X86_32TargetInfo>(T);
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
      return new CygwinX86_32TargetInfo(T);
    case llvm::Triple::MinGW32:
      return new MinGWX86_32TargetInfo(T);
    case llvm::Triple::Win32:
      return new VisualStudioWindowsX86_32TargetInfo(T);
    default:
      return new X86_32TargetInfo(T);
    }

  case llvm::Triple::x86_64:
    switch (os) {
    case llvm::Triple::AuroraUX:
      return new AuroraUXTargetInfo<X86_64TargetInfo>(T);
    case llvm::Triple::Darwin:
      return new DarwinX86_64TargetInfo(T);
    case llvm::Triple::Linux:
      return new LinuxTargetInfo<X86_64TargetInfo>(T);
    case llvm::Triple::DragonFly:
      return new DragonFlyBSDTargetInfo<X86_64TargetInfo>(T);
    case llvm::Triple::NetBSD:
      return new NetBSDTargetInfo<X86_64TargetInfo>(T);
    case llvm::Triple::OpenBSD:
      return new OpenBSDX86_64TargetInfo(T);
    case llvm::Triple::FreeBSD:
      return new FreeBSDTargetInfo<X86_64TargetInfo>(T);
    case llvm::Triple::Solaris:
      return new SolarisTargetInfo<X86_64TargetInfo>(T);
    case llvm::Triple::MinGW64:
      return new MinGWX86_64TargetInfo(T);
    case llvm::Triple::Win32:   // This is what Triple.h supports now.
      return new VisualStudioWindowsX86_64TargetInfo(T);
    default:
      return new X86_64TargetInfo(T);
    }
  }
}

/// CreateTargetInfo - Return the target info object for the specified target
/// triple.
TargetInfo *TargetInfo::CreateTargetInfo(Diagnostic &Diags,
                                         TargetOptions &Opts) {
  llvm::Triple Triple(Opts.Triple);

  // Construct the target
  llvm::OwningPtr<TargetInfo> Target(AllocateTarget(Triple.str()));
  if (!Target) {
    Diags.Report(diag::err_target_unknown_triple) << Triple.str();
    return 0;
  }

  // Set the target CPU if specified.
  if (!Opts.CPU.empty() && !Target->setCPU(Opts.CPU)) {
    Diags.Report(diag::err_target_unknown_cpu) << Opts.CPU;
    return 0;
  }

  // Set the target ABI if specified.
  if (!Opts.ABI.empty() && !Target->setABI(Opts.ABI)) {
    Diags.Report(diag::err_target_unknown_abi) << Opts.ABI;
    return 0;
  }

  // Compute the default target features, we need the target to handle this
  // because features may have dependencies on one another.
  llvm::StringMap<bool> Features;
  Target->getDefaultFeatures(Opts.CPU, Features);

  // Apply the user specified deltas.
  for (std::vector<std::string>::const_iterator it = Opts.Features.begin(),
         ie = Opts.Features.end(); it != ie; ++it) {
    const char *Name = it->c_str();

    // Apply the feature via the target.
    if ((Name[0] != '-' && Name[0] != '+') ||
        !Target->setFeatureEnabled(Features, Name + 1, (Name[0] == '+'))) {
      Diags.Report(diag::err_target_invalid_feature) << Name;
      return 0;
    }
  }

  // Add the features to the compile options.
  //
  // FIXME: If we are completely confident that we have the right set, we only
  // need to pass the minuses.
  Opts.Features.clear();
  for (llvm::StringMap<bool>::const_iterator it = Features.begin(),
         ie = Features.end(); it != ie; ++it)
    Opts.Features.push_back(std::string(it->second ? "+" : "-") + it->first());
  Target->HandleTargetFeatures(Opts.Features);

  return Target.take();
}
