//===--- SanitizerArgs.cpp - Arguments for sanitizer tools  ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "clang/Driver/SanitizerArgs.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/Options.h"
#include "clang/Driver/ToolChain.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SpecialCaseList.h"
#include <memory>

using namespace clang::driver;
using namespace llvm::opt;

namespace {
/// Assign ordinals to possible values of -fsanitize= flag.
/// We use the ordinal values as bit positions within \c SanitizeKind.
enum SanitizeOrdinal {
#define SANITIZER(NAME, ID) SO_##ID,
#define SANITIZER_GROUP(NAME, ID, ALIAS) SO_##ID##Group,
#include "clang/Basic/Sanitizers.def"
  SO_Count
};

/// Represents a set of sanitizer kinds. It is also used to define:
/// 1) set of sanitizers each sanitizer group expands into.
/// 2) set of sanitizers sharing a specific property (e.g.
///    all sanitizers with zero-base shadow).
enum SanitizeKind {
#define SANITIZER(NAME, ID) ID = 1 << SO_##ID,
#define SANITIZER_GROUP(NAME, ID, ALIAS)                                       \
ID = ALIAS, ID##Group = 1 << SO_##ID##Group,
#include "clang/Basic/Sanitizers.def"
  NeedsUbsanRt = Undefined | Integer,
  NotAllowedWithTrap = Vptr,
  RequiresPIE = Memory | DataFlow,
  NeedsUnwindTables = Address | Thread | Memory | DataFlow,
  SupportsCoverage = Address | Memory | Leak | Undefined | Integer,
  RecoverableByDefault = Undefined | Integer,
  Unrecoverable = Address | Unreachable | Return,
  LegacyFsanitizeRecoverMask = Undefined | Integer
};
}

/// Returns true if set of \p Sanitizers contain at least one sanitizer from
/// \p Kinds.
static bool hasOneOf(const clang::SanitizerSet &Sanitizers, unsigned Kinds) {
#define SANITIZER(NAME, ID)                                                    \
  if (Sanitizers.has(clang::SanitizerKind::ID) && (Kinds & ID))                \
    return true;
#include "clang/Basic/Sanitizers.def"
  return false;
}

/// Adds all sanitizers from \p Kinds to \p Sanitizers.
static void addAllOf(clang::SanitizerSet &Sanitizers, unsigned Kinds) {
#define SANITIZER(NAME, ID) \
  if (Kinds & ID) \
    Sanitizers.set(clang::SanitizerKind::ID, true);
#include "clang/Basic/Sanitizers.def"
}

static unsigned toSanitizeKind(clang::SanitizerKind K) {
#define SANITIZER(NAME, ID) \
  if (K == clang::SanitizerKind::ID) \
    return ID;
#include "clang/Basic/Sanitizers.def"
  llvm_unreachable("Invalid SanitizerKind!");
}

/// Parse a single value from a -fsanitize= or -fno-sanitize= value list.
/// Returns a member of the \c SanitizeKind enumeration, or \c 0
/// if \p Value is not known.
static unsigned parseValue(const char *Value);

/// Parse a -fsanitize= or -fno-sanitize= argument's values, diagnosing any
/// invalid components. Returns OR of members of \c SanitizeKind enumeration.
static unsigned parseArgValues(const Driver &D, const llvm::opt::Arg *A,
                               bool DiagnoseErrors);

/// Produce an argument string from ArgList \p Args, which shows how it
/// provides some sanitizer kind from \p Mask. For example, the argument list
/// "-fsanitize=thread,vptr -fsanitize=address" with mask \c NeedsUbsanRt
/// would produce "-fsanitize=vptr".
static std::string lastArgumentForMask(const Driver &D,
                                       const llvm::opt::ArgList &Args,
                                       unsigned Mask);

static std::string lastArgumentForKind(const Driver &D,
                                       const llvm::opt::ArgList &Args,
                                       clang::SanitizerKind K) {
  return lastArgumentForMask(D, Args, toSanitizeKind(K));
}

/// Produce an argument string from argument \p A, which shows how it provides
/// a value in \p Mask. For instance, the argument
/// "-fsanitize=address,alignment" with mask \c NeedsUbsanRt would produce
/// "-fsanitize=alignment".
static std::string describeSanitizeArg(const llvm::opt::Arg *A, unsigned Mask);

/// Produce a string containing comma-separated names of sanitizers in \p
/// Sanitizers set.
static std::string toString(const clang::SanitizerSet &Sanitizers);

/// For each sanitizer group bit set in \p Kinds, set the bits for sanitizers
/// this group enables.
static unsigned expandGroups(unsigned Kinds);

static unsigned getToolchainUnsupportedKinds(const ToolChain &TC) {
  bool IsFreeBSD = TC.getTriple().getOS() == llvm::Triple::FreeBSD;
  bool IsLinux = TC.getTriple().getOS() == llvm::Triple::Linux;
  bool IsX86 = TC.getTriple().getArch() == llvm::Triple::x86;
  bool IsX86_64 = TC.getTriple().getArch() == llvm::Triple::x86_64;
  bool IsMIPS64 = TC.getTriple().getArch() == llvm::Triple::mips64 ||
                  TC.getTriple().getArch() == llvm::Triple::mips64el;

  unsigned Unsupported = 0;
  if (!(IsLinux && (IsX86_64 || IsMIPS64))) {
    Unsupported |= Memory | DataFlow;
  }
  if (!((IsLinux || IsFreeBSD) && IsX86_64)) {
    Unsupported |= Thread;
  }
  if (!(IsLinux && (IsX86 || IsX86_64))) {
    Unsupported |= Function;
  }
  return Unsupported;
}

bool SanitizerArgs::needsUbsanRt() const {
  return !UbsanTrapOnError && hasOneOf(Sanitizers, NeedsUbsanRt);
}

bool SanitizerArgs::requiresPIE() const {
  return AsanZeroBaseShadow || hasOneOf(Sanitizers, RequiresPIE);
}

bool SanitizerArgs::needsUnwindTables() const {
  return hasOneOf(Sanitizers, NeedsUnwindTables);
}

void SanitizerArgs::clear() {
  Sanitizers.clear();
  RecoverableSanitizers.clear();
  BlacklistFile = "";
  SanitizeCoverage = 0;
  MsanTrackOrigins = 0;
  AsanFieldPadding = 0;
  AsanZeroBaseShadow = false;
  UbsanTrapOnError = false;
  AsanSharedRuntime = false;
  LinkCXXRuntimes = false;
}

SanitizerArgs::SanitizerArgs(const ToolChain &TC,
                             const llvm::opt::ArgList &Args) {
  clear();
  unsigned AllRemove = 0;  // During the loop below, the accumulated set of
                           // sanitizers disabled by the current sanitizer
                           // argument or any argument after it.
  unsigned DiagnosedKinds = 0;  // All Kinds we have diagnosed up to now.
                                // Used to deduplicate diagnostics.
  unsigned Kinds = 0;
  unsigned NotSupported = getToolchainUnsupportedKinds(TC);
  const Driver &D = TC.getDriver();
  for (ArgList::const_reverse_iterator I = Args.rbegin(), E = Args.rend();
       I != E; ++I) {
    const auto *Arg = *I;
    if (Arg->getOption().matches(options::OPT_fsanitize_EQ)) {
      Arg->claim();
      unsigned Add = parseArgValues(D, Arg, true);

      // Avoid diagnosing any sanitizer which is disabled later.
      Add &= ~AllRemove;
      // At this point we have not expanded groups, so any unsupported
      // sanitizers in Add are those which have been explicitly enabled.
      // Diagnose them.
      if (unsigned KindsToDiagnose = Add & NotSupported & ~DiagnosedKinds) {
        // Only diagnose the new kinds.
        std::string Desc = describeSanitizeArg(*I, KindsToDiagnose);
        D.Diag(diag::err_drv_unsupported_opt_for_target)
            << Desc << TC.getTriple().str();
        DiagnosedKinds |= KindsToDiagnose;
      }
      Add &= ~NotSupported;

      Add = expandGroups(Add);
      // Group expansion may have enabled a sanitizer which is disabled later.
      Add &= ~AllRemove;
      // Silently discard any unsupported sanitizers implicitly enabled through
      // group expansion.
      Add &= ~NotSupported;

      Kinds |= Add;
    } else if (Arg->getOption().matches(options::OPT_fno_sanitize_EQ)) {
      Arg->claim();
      unsigned Remove = parseArgValues(D, Arg, true);
      AllRemove |= expandGroups(Remove);
    }
  }
  addAllOf(Sanitizers, Kinds);

  // Parse -f(no-)?sanitize-recover flags.
  unsigned RecoverableKinds = RecoverableByDefault;
  unsigned DiagnosedUnrecoverableKinds = 0;
  for (const auto *Arg : Args) {
    if (Arg->getOption().matches(options::OPT_fsanitize_recover)) {
      // FIXME: Add deprecation notice, and then remove this flag.
      RecoverableKinds |= expandGroups(LegacyFsanitizeRecoverMask);
      Arg->claim();
    } else if (Arg->getOption().matches(options::OPT_fno_sanitize_recover)) {
      // FIXME: Add deprecation notice, and then remove this flag.
      RecoverableKinds &= ~expandGroups(LegacyFsanitizeRecoverMask);
      Arg->claim();
    } else if (Arg->getOption().matches(options::OPT_fsanitize_recover_EQ)) {
      unsigned Add = parseArgValues(D, Arg, true);
      // Report error if user explicitly tries to recover from unrecoverable
      // sanitizer.
      if (unsigned KindsToDiagnose =
              Add & Unrecoverable & ~DiagnosedUnrecoverableKinds) {
        SanitizerSet SetToDiagnose;
        addAllOf(SetToDiagnose, KindsToDiagnose);
        D.Diag(diag::err_drv_unsupported_option_argument)
            << Arg->getOption().getName() << toString(SetToDiagnose);
        DiagnosedUnrecoverableKinds |= KindsToDiagnose;
      }
      RecoverableKinds |= expandGroups(Add);
      Arg->claim();
    } else if (Arg->getOption().matches(options::OPT_fno_sanitize_recover_EQ)) {
      RecoverableKinds &= ~expandGroups(parseArgValues(D, Arg, true));
      Arg->claim();
    }
  }
  RecoverableKinds &= Kinds;
  RecoverableKinds &= ~Unrecoverable;
  addAllOf(RecoverableSanitizers, RecoverableKinds);

  UbsanTrapOnError =
    Args.hasFlag(options::OPT_fsanitize_undefined_trap_on_error,
                 options::OPT_fno_sanitize_undefined_trap_on_error, false);

  // Warn about undefined sanitizer options that require runtime support.
  if (UbsanTrapOnError && hasOneOf(Sanitizers, NotAllowedWithTrap)) {
    D.Diag(clang::diag::err_drv_argument_not_allowed_with)
      << lastArgumentForMask(D, Args, NotAllowedWithTrap)
      << "-fsanitize-undefined-trap-on-error";
  }

  // Check for incompatible sanitizers.
  bool NeedsAsan = Sanitizers.has(SanitizerKind::Address);
  bool NeedsTsan = Sanitizers.has(SanitizerKind::Thread);
  bool NeedsMsan = Sanitizers.has(SanitizerKind::Memory);
  bool NeedsLsan = Sanitizers.has(SanitizerKind::Leak);
  if (NeedsAsan && NeedsTsan)
    D.Diag(clang::diag::err_drv_argument_not_allowed_with)
      << lastArgumentForKind(D, Args, SanitizerKind::Address)
      << lastArgumentForKind(D, Args, SanitizerKind::Thread);
  if (NeedsAsan && NeedsMsan)
    D.Diag(clang::diag::err_drv_argument_not_allowed_with)
      << lastArgumentForKind(D, Args, SanitizerKind::Address)
      << lastArgumentForKind(D, Args, SanitizerKind::Memory);
  if (NeedsTsan && NeedsMsan)
    D.Diag(clang::diag::err_drv_argument_not_allowed_with)
      << lastArgumentForKind(D, Args, SanitizerKind::Thread)
      << lastArgumentForKind(D, Args, SanitizerKind::Memory);
  if (NeedsLsan && NeedsTsan)
    D.Diag(clang::diag::err_drv_argument_not_allowed_with)
      << lastArgumentForKind(D, Args, SanitizerKind::Leak)
      << lastArgumentForKind(D, Args, SanitizerKind::Thread);
  if (NeedsLsan && NeedsMsan)
    D.Diag(clang::diag::err_drv_argument_not_allowed_with)
      << lastArgumentForKind(D, Args, SanitizerKind::Leak)
      << lastArgumentForKind(D, Args, SanitizerKind::Memory);
  // FIXME: Currently -fsanitize=leak is silently ignored in the presence of
  // -fsanitize=address. Perhaps it should print an error, or perhaps
  // -f(-no)sanitize=leak should change whether leak detection is enabled by
  // default in ASan?

  // Parse -f(no-)sanitize-blacklist options.
  if (Arg *BLArg = Args.getLastArg(options::OPT_fsanitize_blacklist,
                                   options::OPT_fno_sanitize_blacklist)) {
    if (BLArg->getOption().matches(options::OPT_fsanitize_blacklist)) {
      std::string BLPath = BLArg->getValue();
      if (llvm::sys::fs::exists(BLPath)) {
        // Validate the blacklist format.
        std::string BLError;
        std::unique_ptr<llvm::SpecialCaseList> SCL(
            llvm::SpecialCaseList::create(BLPath, BLError));
        if (!SCL.get())
          D.Diag(clang::diag::err_drv_malformed_sanitizer_blacklist) << BLError;
        else
          BlacklistFile = BLPath;
      } else {
        D.Diag(clang::diag::err_drv_no_such_file) << BLPath;
      }
    }
  } else {
    // If no -fsanitize-blacklist option is specified, try to look up for
    // blacklist in the resource directory.
    std::string BLPath;
    if (getDefaultBlacklist(D, BLPath) && llvm::sys::fs::exists(BLPath))
      BlacklistFile = BLPath;
  }

  // Parse -f[no-]sanitize-memory-track-origins[=level] options.
  if (NeedsMsan) {
    if (Arg *A =
            Args.getLastArg(options::OPT_fsanitize_memory_track_origins_EQ,
                            options::OPT_fsanitize_memory_track_origins,
                            options::OPT_fno_sanitize_memory_track_origins)) {
      if (A->getOption().matches(options::OPT_fsanitize_memory_track_origins)) {
        MsanTrackOrigins = 1;
      } else if (A->getOption().matches(
                     options::OPT_fno_sanitize_memory_track_origins)) {
        MsanTrackOrigins = 0;
      } else {
        StringRef S = A->getValue();
        if (S.getAsInteger(0, MsanTrackOrigins) || MsanTrackOrigins < 0 ||
            MsanTrackOrigins > 2) {
          D.Diag(clang::diag::err_drv_invalid_value) << A->getAsString(Args) << S;
        }
      }
    }
  }

  // Parse -fsanitize-coverage=N. Currently one of asan/msan/lsan is required.
  if (hasOneOf(Sanitizers, SupportsCoverage)) {
    if (Arg *A = Args.getLastArg(options::OPT_fsanitize_coverage)) {
      StringRef S = A->getValue();
      // Legal values are 0..4.
      if (S.getAsInteger(0, SanitizeCoverage) || SanitizeCoverage < 0 ||
          SanitizeCoverage > 4)
        D.Diag(clang::diag::err_drv_invalid_value) << A->getAsString(Args) << S;
    }
  }

  if (NeedsAsan) {
    AsanSharedRuntime =
        Args.hasArg(options::OPT_shared_libasan) ||
        (TC.getTriple().getEnvironment() == llvm::Triple::Android);
    AsanZeroBaseShadow =
        (TC.getTriple().getEnvironment() == llvm::Triple::Android);
    if (Arg *A =
            Args.getLastArg(options::OPT_fsanitize_address_field_padding)) {
        StringRef S = A->getValue();
        // Legal values are 0 and 1, 2, but in future we may add more levels.
        if (S.getAsInteger(0, AsanFieldPadding) || AsanFieldPadding < 0 ||
            AsanFieldPadding > 2) {
          D.Diag(clang::diag::err_drv_invalid_value) << A->getAsString(Args) << S;
        }
    }

    if (Arg *WindowsDebugRTArg =
            Args.getLastArg(options::OPT__SLASH_MTd, options::OPT__SLASH_MT,
                            options::OPT__SLASH_MDd, options::OPT__SLASH_MD,
                            options::OPT__SLASH_LDd, options::OPT__SLASH_LD)) {
      switch (WindowsDebugRTArg->getOption().getID()) {
      case options::OPT__SLASH_MTd:
      case options::OPT__SLASH_MDd:
      case options::OPT__SLASH_LDd:
        D.Diag(clang::diag::err_drv_argument_not_allowed_with)
            << WindowsDebugRTArg->getAsString(Args)
            << lastArgumentForKind(D, Args, SanitizerKind::Address);
        D.Diag(clang::diag::note_drv_address_sanitizer_debug_runtime);
      }
    }
  }

  // Parse -link-cxx-sanitizer flag.
  LinkCXXRuntimes =
      Args.hasArg(options::OPT_fsanitize_link_cxx_runtime) || D.CCCIsCXX();
}

static std::string toString(const clang::SanitizerSet &Sanitizers) {
  std::string Res;
#define SANITIZER(NAME, ID)                                                    \
  if (Sanitizers.has(clang::SanitizerKind::ID)) {                              \
    if (!Res.empty())                                                          \
      Res += ",";                                                              \
    Res += NAME;                                                               \
  }
#include "clang/Basic/Sanitizers.def"
  return Res;
}

void SanitizerArgs::addArgs(const llvm::opt::ArgList &Args,
                            llvm::opt::ArgStringList &CmdArgs) const {
  if (Sanitizers.empty())
    return;
  CmdArgs.push_back(Args.MakeArgString("-fsanitize=" + toString(Sanitizers)));

  if (!RecoverableSanitizers.empty())
    CmdArgs.push_back(Args.MakeArgString("-fsanitize-recover=" +
                                         toString(RecoverableSanitizers)));

  if (UbsanTrapOnError)
    CmdArgs.push_back("-fsanitize-undefined-trap-on-error");

  if (!BlacklistFile.empty()) {
    SmallString<64> BlacklistOpt("-fsanitize-blacklist=");
    BlacklistOpt += BlacklistFile;
    CmdArgs.push_back(Args.MakeArgString(BlacklistOpt));
  }

  if (MsanTrackOrigins)
    CmdArgs.push_back(Args.MakeArgString("-fsanitize-memory-track-origins=" +
                                         llvm::utostr(MsanTrackOrigins)));
  if (AsanFieldPadding)
    CmdArgs.push_back(Args.MakeArgString("-fsanitize-address-field-padding=" +
                                         llvm::utostr(AsanFieldPadding)));
  if (SanitizeCoverage)
    CmdArgs.push_back(Args.MakeArgString("-fsanitize-coverage=" +
                                         llvm::utostr(SanitizeCoverage)));
  // Workaround for PR16386.
  if (Sanitizers.has(SanitizerKind::Memory))
    CmdArgs.push_back(Args.MakeArgString("-fno-assume-sane-operator-new"));
}

bool SanitizerArgs::getDefaultBlacklist(const Driver &D, std::string &BLPath) {
  const char *BlacklistFile = nullptr;
  if (Sanitizers.has(SanitizerKind::Address))
    BlacklistFile = "asan_blacklist.txt";
  else if (Sanitizers.has(SanitizerKind::Memory))
    BlacklistFile = "msan_blacklist.txt";
  else if (Sanitizers.has(SanitizerKind::Thread))
    BlacklistFile = "tsan_blacklist.txt";
  else if (Sanitizers.has(SanitizerKind::DataFlow))
    BlacklistFile = "dfsan_abilist.txt";

  if (BlacklistFile) {
    SmallString<64> Path(D.ResourceDir);
    llvm::sys::path::append(Path, BlacklistFile);
    BLPath = Path.str();
    return true;
  }
  return false;
}

unsigned parseValue(const char *Value) {
  unsigned ParsedKind = llvm::StringSwitch<SanitizeKind>(Value)
#define SANITIZER(NAME, ID) .Case(NAME, ID)
#define SANITIZER_GROUP(NAME, ID, ALIAS) .Case(NAME, ID##Group)
#include "clang/Basic/Sanitizers.def"
    .Default(SanitizeKind());
  return ParsedKind;
}

unsigned expandGroups(unsigned Kinds) {
#define SANITIZER(NAME, ID)
#define SANITIZER_GROUP(NAME, ID, ALIAS) if (Kinds & ID##Group) Kinds |= ID;
#include "clang/Basic/Sanitizers.def"
  return Kinds;
}

unsigned parseArgValues(const Driver &D, const llvm::opt::Arg *A,
                        bool DiagnoseErrors) {
  assert((A->getOption().matches(options::OPT_fsanitize_EQ) ||
          A->getOption().matches(options::OPT_fno_sanitize_EQ) ||
          A->getOption().matches(options::OPT_fsanitize_recover_EQ) ||
          A->getOption().matches(options::OPT_fno_sanitize_recover_EQ)) &&
         "Invalid argument in parseArgValues!");
  unsigned Kinds = 0;
  for (unsigned I = 0, N = A->getNumValues(); I != N; ++I) {
    const char *Value = A->getValue(I);
    unsigned Kind;
    // Special case: don't accept -fsanitize=all.
    if (A->getOption().matches(options::OPT_fsanitize_EQ) &&
        0 == strcmp("all", Value))
      Kind = 0;
    else
      Kind = parseValue(Value);

    if (Kind)
      Kinds |= Kind;
    else if (DiagnoseErrors)
      D.Diag(clang::diag::err_drv_unsupported_option_argument)
          << A->getOption().getName() << Value;
  }
  return Kinds;
}

std::string lastArgumentForMask(const Driver &D, const llvm::opt::ArgList &Args,
                                unsigned Mask) {
  for (llvm::opt::ArgList::const_reverse_iterator I = Args.rbegin(),
                                                  E = Args.rend();
       I != E; ++I) {
    const auto *Arg = *I;
    if (Arg->getOption().matches(options::OPT_fsanitize_EQ)) {
      unsigned AddKinds = expandGroups(parseArgValues(D, Arg, false));
      if (AddKinds & Mask)
        return describeSanitizeArg(Arg, Mask);
    } else if (Arg->getOption().matches(options::OPT_fno_sanitize_EQ)) {
      unsigned RemoveKinds = expandGroups(parseArgValues(D, Arg, false));
      Mask &= ~RemoveKinds;
    }
  }
  llvm_unreachable("arg list didn't provide expected value");
}

std::string describeSanitizeArg(const llvm::opt::Arg *A, unsigned Mask) {
  assert(A->getOption().matches(options::OPT_fsanitize_EQ)
         && "Invalid argument in describeSanitizerArg!");

  std::string Sanitizers;
  for (unsigned I = 0, N = A->getNumValues(); I != N; ++I) {
    if (expandGroups(parseValue(A->getValue(I))) & Mask) {
      if (!Sanitizers.empty())
        Sanitizers += ",";
      Sanitizers += A->getValue(I);
    }
  }

  assert(!Sanitizers.empty() && "arg didn't provide expected value");
  return "-fsanitize=" + Sanitizers;
}
