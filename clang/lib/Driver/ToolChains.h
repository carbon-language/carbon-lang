//===--- ToolChains.h - ToolChain Implementations ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_LIB_DRIVER_TOOLCHAINS_H_
#define CLANG_LIB_DRIVER_TOOLCHAINS_H_

#include "clang/Driver/Action.h"
#include "clang/Driver/ToolChain.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Compiler.h"

#include "Tools.h"

namespace clang {
namespace driver {
namespace toolchains {

/// Generic_GCC - A tool chain using the 'gcc' command to perform
/// all subcommands; this relies on gcc translating the majority of
/// command line options.
class VISIBILITY_HIDDEN Generic_GCC : public ToolChain {
protected:
  mutable llvm::DenseMap<unsigned, Tool*> Tools;

public:
  Generic_GCC(const HostInfo &Host, const llvm::Triple& Triple);
  ~Generic_GCC();

  virtual DerivedArgList *TranslateArgs(InputArgList &Args,
                                        const char *BoundArch) const;

  virtual Tool &SelectTool(const Compilation &C, const JobAction &JA) const;

  virtual bool IsUnwindTablesDefault() const;
  virtual const char *GetDefaultRelocationModel() const;
  virtual const char *GetForcedPicModel() const;
};

/// Darwin - The base Darwin tool chain.
class VISIBILITY_HIDDEN Darwin : public ToolChain {
  mutable llvm::DenseMap<unsigned, Tool*> Tools;

  /// Darwin version of tool chain.
  unsigned DarwinVersion[3];

  /// Whether this is this an iPhoneOS toolchain.
  //
  // FIXME: This should go away, such differences should be completely
  // determined by the target triple.
  bool IsIPhoneOS;

  /// The default macosx-version-min of this tool chain; empty until
  /// initialized.
  mutable std::string MacosxVersionMin;

  /// The default iphoneos-version-min of this tool chain.
  std::string IPhoneOSVersionMin;

  const char *getMacosxVersionMin() const;

public:
  Darwin(const HostInfo &Host, const llvm::Triple& Triple,
         const unsigned (&DarwinVersion)[3], bool IsIPhoneOS);
  ~Darwin();

  /// @name Darwin Specific Toolchain API
  /// {

  void getDarwinVersion(unsigned (&Res)[3]) const {
    Res[0] = DarwinVersion[0];
    Res[1] = DarwinVersion[1];
    Res[2] = DarwinVersion[2];
  }

  void getMacosxVersion(unsigned (&Res)[3]) const {
    Res[0] = 10;
    Res[1] = DarwinVersion[0] - 4;
    Res[2] = DarwinVersion[1];
  }

  /// getMacosxVersionMin - Get the effective -mmacosx-version-min, which is
  /// either the -mmacosx-version-min, or the current version if unspecified.
  void getMacosxVersionMin(const ArgList &Args, unsigned (&Res)[3]) const;

  static bool isMacosxVersionLT(unsigned (&A)[3], unsigned (&B)[3]) {
    for (unsigned i=0; i < 3; ++i) {
      if (A[i] > B[i]) return false;
      if (A[i] < B[i]) return true;
    }
    return false;
  }

  static bool isMacosxVersionLT(unsigned (&A)[3],
                                unsigned V0, unsigned V1=0, unsigned V2=0) {
    unsigned B[3] = { V0, V1, V2 };
    return isMacosxVersionLT(A, B);
  }

  const char *getMacosxVersionStr() const {
    return MacosxVersionMin.c_str();
  }

  const char *getIPhoneOSVersionStr() const {
    return IPhoneOSVersionMin.c_str();
  }

  /// AddLinkSearchPathArgs - Add the linker search paths to \arg CmdArgs.
  ///
  /// \param Args - The input argument list.
  /// \param CmdArgs [out] - The command argument list to append the paths
  /// (prefixed by -L) to.
  virtual void AddLinkSearchPathArgs(const ArgList &Args,
                                     ArgStringList &CmdArgs) const = 0;

  /// AddLinkRuntimeLibArgs - Add the linker arguments to link the compiler
  /// runtime library.
  virtual void AddLinkRuntimeLibArgs(const ArgList &Args,
                                     ArgStringList &CmdArgs) const = 0;

  bool isIPhoneOS() const { return IsIPhoneOS; }

  /// }
  /// @name ToolChain Implementation
  /// {

  virtual DerivedArgList *TranslateArgs(InputArgList &Args,
                                        const char *BoundArch) const;

  virtual Tool &SelectTool(const Compilation &C, const JobAction &JA) const;

  virtual bool IsBlocksDefault() const {
    // Blocks default to on for 10.6 (darwin10) and beyond.
    return (DarwinVersion[0] > 9);
  }
  virtual bool IsObjCNonFragileABIDefault() const {
    // Non-fragile ABI default to on for 10.5 (darwin9) and beyond on x86-64.
    return (DarwinVersion[0] >= 9 &&
            getTriple().getArch() == llvm::Triple::x86_64);
  }
  virtual bool IsUnwindTablesDefault() const;
  virtual unsigned GetDefaultStackProtectorLevel() const {
    // Stack protectors default to on for 10.6 (darwin10) and beyond.
    return (DarwinVersion[0] > 9) ? 1 : 0;
  }
  virtual const char *GetDefaultRelocationModel() const;
  virtual const char *GetForcedPicModel() const;

  virtual bool UseDwarfDebugFlags() const;

  /// }
};

/// DarwinClang - The Darwin toolchain used by Clang.
class VISIBILITY_HIDDEN DarwinClang : public Darwin {
public:
  DarwinClang(const HostInfo &Host, const llvm::Triple& Triple,
              const unsigned (&DarwinVersion)[3], bool IsIPhoneOS);

  /// @name Darwin ToolChain Implementation
  /// {

  virtual void AddLinkSearchPathArgs(const ArgList &Args,
                                    ArgStringList &CmdArgs) const;

  virtual void AddLinkRuntimeLibArgs(const ArgList &Args,
                                     ArgStringList &CmdArgs) const;

  /// }
};

/// DarwinGCC - The Darwin toolchain used by GCC.
class VISIBILITY_HIDDEN DarwinGCC : public Darwin {
  /// GCC version to use.
  unsigned GCCVersion[3];

  /// The directory suffix for this tool chain.
  std::string ToolChainDir;

public:
  DarwinGCC(const HostInfo &Host, const llvm::Triple& Triple,
            const unsigned (&DarwinVersion)[3], const unsigned (&GCCVersion)[3],
            bool IsIPhoneOS);

  /// @name Darwin ToolChain Implementation
  /// {

  virtual void AddLinkSearchPathArgs(const ArgList &Args,
                                    ArgStringList &CmdArgs) const;

  virtual void AddLinkRuntimeLibArgs(const ArgList &Args,
                                     ArgStringList &CmdArgs) const;

  /// }
};

/// Darwin_Generic_GCC - Generic Darwin tool chain using gcc.
class VISIBILITY_HIDDEN Darwin_Generic_GCC : public Generic_GCC {
public:
  Darwin_Generic_GCC(const HostInfo &Host, const llvm::Triple& Triple)
    : Generic_GCC(Host, Triple) {}

  virtual const char *GetDefaultRelocationModel() const { return "pic"; }
};

class VISIBILITY_HIDDEN AuroraUX : public Generic_GCC {
public:
  AuroraUX(const HostInfo &Host, const llvm::Triple& Triple);

  virtual Tool &SelectTool(const Compilation &C, const JobAction &JA) const;
};

class VISIBILITY_HIDDEN OpenBSD : public Generic_GCC {
public:
  OpenBSD(const HostInfo &Host, const llvm::Triple& Triple);

  virtual Tool &SelectTool(const Compilation &C, const JobAction &JA) const;
};

class VISIBILITY_HIDDEN FreeBSD : public Generic_GCC {
public:
  FreeBSD(const HostInfo &Host, const llvm::Triple& Triple, bool Lib32);

  virtual Tool &SelectTool(const Compilation &C, const JobAction &JA) const;
};

class VISIBILITY_HIDDEN DragonFly : public Generic_GCC {
public:
  DragonFly(const HostInfo &Host, const llvm::Triple& Triple);

  virtual Tool &SelectTool(const Compilation &C, const JobAction &JA) const;
};

class VISIBILITY_HIDDEN Linux : public Generic_GCC {
public:
  Linux(const HostInfo &Host, const llvm::Triple& Triple);
};


} // end namespace toolchains
} // end namespace driver
} // end namespace clang

#endif
