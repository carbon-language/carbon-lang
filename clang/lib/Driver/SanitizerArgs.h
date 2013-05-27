//===--- SanitizerArgs.h - Arguments for sanitizer tools  -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef CLANG_LIB_DRIVER_SANITIZERARGS_H_
#define CLANG_LIB_DRIVER_SANITIZERARGS_H_

#include "clang/Driver/Arg.h"
#include "clang/Driver/ArgList.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/Options.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Path.h"

namespace clang {
namespace driver {

class SanitizerArgs {
  /// Assign ordinals to sanitizer flags. We'll use the ordinal values as
  /// bit positions within \c Kind.
  enum SanitizeOrdinal {
#define SANITIZER(NAME, ID) SO_##ID,
#include "clang/Basic/Sanitizers.def"
    SO_Count
  };

  /// Bugs to catch at runtime.
  enum SanitizeKind {
#define SANITIZER(NAME, ID) ID = 1 << SO_##ID,
#define SANITIZER_GROUP(NAME, ID, ALIAS) ID = ALIAS,
#include "clang/Basic/Sanitizers.def"
    NeedsAsanRt = Address,
    NeedsTsanRt = Thread,
    NeedsMsanRt = Memory,
    NeedsLeakDetection = Leak,
    NeedsUbsanRt = Undefined | Integer,
    NotAllowedWithTrap = Vptr,
    HasZeroBaseShadow = Thread | Memory
  };
  unsigned Kind;
  std::string BlacklistFile;
  bool MsanTrackOrigins;
  bool AsanZeroBaseShadow;
  bool UbsanTrapOnError;

 public:
  SanitizerArgs() : Kind(0), BlacklistFile(""), MsanTrackOrigins(false),
                    AsanZeroBaseShadow(false), UbsanTrapOnError(false) {}
  /// Parses the sanitizer arguments from an argument list.
  SanitizerArgs(const ToolChain &TC, const ArgList &Args);

  bool needsAsanRt() const { return Kind & NeedsAsanRt; }
  bool needsTsanRt() const { return Kind & NeedsTsanRt; }
  bool needsMsanRt() const { return Kind & NeedsMsanRt; }
  bool needsLeakDetection() const { return Kind & NeedsLeakDetection; }
  bool needsLsanRt() const {
    return needsLeakDetection() && !needsAsanRt();
  }
  bool needsUbsanRt() const {
    if (UbsanTrapOnError)
      return false;
    return Kind & NeedsUbsanRt;
  }

  bool sanitizesVptr() const { return Kind & Vptr; }
  bool notAllowedWithTrap() const { return Kind & NotAllowedWithTrap; }
  bool hasZeroBaseShadow() const {
    return (Kind & HasZeroBaseShadow) || AsanZeroBaseShadow;
  }

  void addArgs(const ArgList &Args, ArgStringList &CmdArgs) const {
    if (!Kind)
      return;
    SmallString<256> SanitizeOpt("-fsanitize=");
#define SANITIZER(NAME, ID) \
    if (Kind & ID) \
      SanitizeOpt += NAME ",";
#include "clang/Basic/Sanitizers.def"
    SanitizeOpt.pop_back();
    CmdArgs.push_back(Args.MakeArgString(SanitizeOpt));
    if (!BlacklistFile.empty()) {
      SmallString<64> BlacklistOpt("-fsanitize-blacklist=");
      BlacklistOpt += BlacklistFile;
      CmdArgs.push_back(Args.MakeArgString(BlacklistOpt));
    }

    if (MsanTrackOrigins)
      CmdArgs.push_back(Args.MakeArgString("-fsanitize-memory-track-origins"));

    if (AsanZeroBaseShadow)
      CmdArgs.push_back(Args.MakeArgString(
          "-fsanitize-address-zero-base-shadow"));
  }

 private:
  /// Parse a single value from a -fsanitize= or -fno-sanitize= value list.
  /// Returns OR of members of the \c SanitizeKind enumeration, or \c 0
  /// if \p Value is not known.
  static unsigned parse(const char *Value) {
    unsigned ParsedKind = llvm::StringSwitch<SanitizeKind>(Value)
#define SANITIZER(NAME, ID) .Case(NAME, ID)
#define SANITIZER_GROUP(NAME, ID, ALIAS) .Case(NAME, ID)
#include "clang/Basic/Sanitizers.def"
      .Default(SanitizeKind());
    // Assume -fsanitize=address implies -fsanitize=init-order.
    // FIXME: This should be either specified in Sanitizers.def, or go away when
    // we get rid of "-fsanitize=init-order" flag at all.
    if (ParsedKind & Address)
      ParsedKind |= InitOrder;
    return ParsedKind;
  }

  /// Parse a -fsanitize= or -fno-sanitize= argument's values, diagnosing any
  /// invalid components.
  static unsigned parse(const Driver &D, const Arg *A, bool DiagnoseErrors) {
    unsigned Kind = 0;
    for (unsigned I = 0, N = A->getNumValues(); I != N; ++I) {
      if (unsigned K = parse(A->getValue(I)))
        Kind |= K;
      else if (DiagnoseErrors)
        D.Diag(diag::err_drv_unsupported_option_argument)
          << A->getOption().getName() << A->getValue(I);
    }
    return Kind;
  }

  /// Parse a single flag of the form -f[no]sanitize=, or
  /// -f*-sanitizer. Sets the masks defining required change of Kind value.
  /// Returns true if the flag was parsed successfully.
  static bool parse(const Driver &D, const ArgList &Args, const Arg *A,
                    unsigned &Add, unsigned &Remove, bool DiagnoseErrors) {
    Add = 0;
    Remove = 0;
    const char *DeprecatedReplacement = 0;
    if (A->getOption().matches(options::OPT_faddress_sanitizer)) {
      Add = Address;
      DeprecatedReplacement = "-fsanitize=address";
    } else if (A->getOption().matches(options::OPT_fno_address_sanitizer)) {
      Remove = Address;
      DeprecatedReplacement = "-fno-sanitize=address";
    } else if (A->getOption().matches(options::OPT_fthread_sanitizer)) {
      Add = Thread;
      DeprecatedReplacement = "-fsanitize=thread";
    } else if (A->getOption().matches(options::OPT_fno_thread_sanitizer)) {
      Remove = Thread;
      DeprecatedReplacement = "-fno-sanitize=thread";
    } else if (A->getOption().matches(options::OPT_fcatch_undefined_behavior)) {
      Add = UndefinedTrap;
      DeprecatedReplacement = 
        "-fsanitize=undefined-trap -fsanitize-undefined-trap-on-error";
    } else if (A->getOption().matches(options::OPT_fbounds_checking) ||
               A->getOption().matches(options::OPT_fbounds_checking_EQ)) {
      Add = Bounds;
      DeprecatedReplacement = "-fsanitize=bounds";
    } else if (A->getOption().matches(options::OPT_fsanitize_EQ)) {
      Add = parse(D, A, DiagnoseErrors);
    } else if (A->getOption().matches(options::OPT_fno_sanitize_EQ)) {
      Remove = parse(D, A, DiagnoseErrors);
    } else {
      // Flag is not relevant to sanitizers.
      return false;
    }
    // If this is a deprecated synonym, produce a warning directing users
    // towards the new spelling.
    if (DeprecatedReplacement && DiagnoseErrors)
      D.Diag(diag::warn_drv_deprecated_arg)
        << A->getAsString(Args) << DeprecatedReplacement;
    return true;
  }

  /// Produce an argument string from ArgList \p Args, which shows how it
  /// provides a sanitizer kind in \p Mask. For example, the argument list
  /// "-fsanitize=thread,vptr -faddress-sanitizer" with mask \c NeedsUbsanRt
  /// would produce "-fsanitize=vptr".
  static std::string lastArgumentForKind(const Driver &D, const ArgList &Args,
                                         unsigned Kind) {
    for (ArgList::const_reverse_iterator I = Args.rbegin(), E = Args.rend();
         I != E; ++I) {
      unsigned Add, Remove;
      if (parse(D, Args, *I, Add, Remove, false) &&
          (Add & Kind))
        return describeSanitizeArg(Args, *I, Kind);
      Kind &= ~Remove;
    }
    llvm_unreachable("arg list didn't provide expected value");
  }

  /// Produce an argument string from argument \p A, which shows how it provides
  /// a value in \p Mask. For instance, the argument
  /// "-fsanitize=address,alignment" with mask \c NeedsUbsanRt would produce
  /// "-fsanitize=alignment".
  static std::string describeSanitizeArg(const ArgList &Args, const Arg *A,
                                         unsigned Mask) {
    if (!A->getOption().matches(options::OPT_fsanitize_EQ))
      return A->getAsString(Args);

    for (unsigned I = 0, N = A->getNumValues(); I != N; ++I)
      if (parse(A->getValue(I)) & Mask)
        return std::string("-fsanitize=") + A->getValue(I);

    llvm_unreachable("arg didn't provide expected value");
  }

  static bool getDefaultBlacklistForKind(const Driver &D, unsigned Kind,
                                         std::string &BLPath) {
    const char *BlacklistFile = 0;
    if (Kind & NeedsAsanRt)
      BlacklistFile = "asan_blacklist.txt";
    else if (Kind & NeedsMsanRt)
      BlacklistFile = "msan_blacklist.txt";
    else if (Kind & NeedsTsanRt)
      BlacklistFile = "tsan_blacklist.txt";
    if (BlacklistFile) {
      SmallString<64> Path(D.ResourceDir);
      llvm::sys::path::append(Path, BlacklistFile);
      BLPath = Path.str();
      return true;
    }
    return false;
  }
};

}  // namespace driver
}  // namespace clang

#endif // CLANG_LIB_DRIVER_SANITIZERARGS_H_
