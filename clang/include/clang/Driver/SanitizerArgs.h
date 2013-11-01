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

#include <string>

#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"

namespace clang {
namespace driver {

class Driver;
class ToolChain;

class SanitizerArgs {
  /// Assign ordinals to sanitizer flags. We'll use the ordinal values as
  /// bit positions within \c Kind.
  enum SanitizeOrdinal {
#define SANITIZER(NAME, ID) SO_##ID,
#define SANITIZER_GROUP(NAME, ID, ALIAS) SO_##ID##Group,
#include "clang/Basic/Sanitizers.def"
    SO_Count
  };

  /// Bugs to catch at runtime.
  enum SanitizeKind {
#define SANITIZER(NAME, ID) ID = 1 << SO_##ID,
#define SANITIZER_GROUP(NAME, ID, ALIAS)                                       \
  ID = ALIAS, ID##Group = 1 << SO_##ID##Group,
#include "clang/Basic/Sanitizers.def"
    NeedsAsanRt = Address,
    NeedsTsanRt = Thread,
    NeedsMsanRt = Memory,
    NeedsDfsanRt = DataFlow,
    NeedsLeakDetection = Leak,
    NeedsUbsanRt = Undefined | Integer,
    NotAllowedWithTrap = Vptr,
    HasZeroBaseShadow = Thread | Memory | DataFlow
  };
  unsigned Kind;

  std::string BlacklistFile;
  bool MsanTrackOrigins;
  bool AsanZeroBaseShadow;
  bool UbsanTrapOnError;

 public:
  SanitizerArgs();
  /// Parses the sanitizer arguments from an argument list.
  SanitizerArgs(const ToolChain &TC, const llvm::opt::ArgList &Args);

  bool needsAsanRt() const { return Kind & NeedsAsanRt; }
  bool needsTsanRt() const { return Kind & NeedsTsanRt; }
  bool needsMsanRt() const { return Kind & NeedsMsanRt; }
  bool needsLeakDetection() const { return Kind & NeedsLeakDetection; }
  bool needsLsanRt() const {
    return needsLeakDetection() && !needsAsanRt();
  }
  bool needsUbsanRt() const {
    return !UbsanTrapOnError && (Kind & NeedsUbsanRt);
  }
  bool needsDfsanRt() const { return Kind & NeedsDfsanRt; }

  bool sanitizesVptr() const { return Kind & Vptr; }
  bool notAllowedWithTrap() const { return Kind & NotAllowedWithTrap; }
  bool hasZeroBaseShadow() const {
    return (Kind & HasZeroBaseShadow) || AsanZeroBaseShadow;
  }
  void addArgs(const llvm::opt::ArgList &Args,
               llvm::opt::ArgStringList &CmdArgs) const;

 private:
  void clear();

  /// Parse a single value from a -fsanitize= or -fno-sanitize= value list.
  /// Returns OR of members of the \c SanitizeKind enumeration, or \c 0
  /// if \p Value is not known.
  static unsigned parse(const char *Value);

  /// Parse a -fsanitize= or -fno-sanitize= argument's values, diagnosing any
  /// invalid components.
  static unsigned parse(const Driver &D, const llvm::opt::Arg *A,
                        bool DiagnoseErrors);

  /// Parse a single flag of the form -f[no]sanitize=, or
  /// -f*-sanitizer. Sets the masks defining required change of Kind value.
  /// Returns true if the flag was parsed successfully.
  static bool parse(const Driver &D, const llvm::opt::ArgList &Args,
                    const llvm::opt::Arg *A, unsigned &Add, unsigned &Remove,
                    bool DiagnoseErrors);

  /// Produce an argument string from ArgList \p Args, which shows how it
  /// provides a sanitizer kind in \p Mask. For example, the argument list
  /// "-fsanitize=thread,vptr -faddress-sanitizer" with mask \c NeedsUbsanRt
  /// would produce "-fsanitize=vptr".
  static std::string lastArgumentForKind(const Driver &D,
                                         const llvm::opt::ArgList &Args,
                                         unsigned Kind);

  /// Produce an argument string from argument \p A, which shows how it provides
  /// a value in \p Mask. For instance, the argument
  /// "-fsanitize=address,alignment" with mask \c NeedsUbsanRt would produce
  /// "-fsanitize=alignment".
  static std::string describeSanitizeArg(const llvm::opt::ArgList &Args,
                                         const llvm::opt::Arg *A,
                                         unsigned Mask);

  static bool getDefaultBlacklistForKind(const Driver &D, unsigned Kind,
                                         std::string &BLPath);

  /// Return the smallest superset of sanitizer set \p Kinds such that each
  /// member of each group whose flag is set in \p Kinds has its flag set in the
  /// result.
  static unsigned expandGroups(unsigned Kinds);

  /// Return the subset of \p Kinds supported by toolchain \p TC.  If
  /// \p DiagnoseErrors is true, produce an error diagnostic for each sanitizer
  /// removed from \p Kinds.
  static unsigned filterUnsupportedKinds(const ToolChain &TC, unsigned Kinds,
                                         const llvm::opt::ArgList &Args,
                                         const llvm::opt::Arg *A,
                                         bool DiagnoseErrors,
                                         unsigned &DiagnosedKinds);

  /// The flags in \p Mask are unsupported by \p TC.  If present in \p Kinds,
  /// remove them and produce an error diagnostic referring to \p A if
  /// \p DiagnoseErrors is true.
  static void filterUnsupportedMask(const ToolChain &TC, unsigned &Kinds,
                                    unsigned Mask,
                                    const llvm::opt::ArgList &Args,
                                    const llvm::opt::Arg *A,
                                    bool DiagnoseErrors,
                                    unsigned &DiagnosedKinds);
};

}  // namespace driver
}  // namespace clang

#endif // CLANG_LIB_DRIVER_SANITIZERARGS_H_
