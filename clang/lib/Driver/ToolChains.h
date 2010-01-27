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

  /// Whether the information on the target has been initialized.
  //
  // FIXME: This should be eliminated. What we want to do is make this part of
  // the "default target for arguments" selection process, once we get out of
  // the argument translation business.
  mutable bool TargetInitialized;

  /// Whether we are targetting iPhoneOS target.
  mutable bool TargetIsIPhoneOS;
  
  /// The OS version we are targetting.
  mutable unsigned TargetVersion[3];

  /// The default macosx-version-min of this tool chain; empty until
  /// initialized.
  std::string MacosxVersionMin;

public:
  Darwin(const HostInfo &Host, const llvm::Triple& Triple,
         const unsigned (&DarwinVersion)[3]);
  ~Darwin();

  /// @name Darwin Specific Toolchain API
  /// {

  // FIXME: Eliminate these ...Target functions and derive separate tool chains
  // for these targets and put version in constructor.
  void setTarget(bool isIPhoneOS, unsigned Major, unsigned Minor,
                 unsigned Micro) const {
    // FIXME: For now, allow reinitialization as long as values don't
    // change. This will go away when we move away from argument translation.
    if (TargetInitialized && TargetIsIPhoneOS == isIPhoneOS &&
        TargetVersion[0] == Major && TargetVersion[1] == Minor &&
        TargetVersion[2] == Micro)
      return;

    assert(!TargetInitialized && "Target already initialized!");
    TargetInitialized = true;
    TargetIsIPhoneOS = isIPhoneOS;
    TargetVersion[0] = Major;
    TargetVersion[1] = Minor;
    TargetVersion[2] = Micro;
  }

  bool isTargetIPhoneOS() const {
    assert(TargetInitialized && "Target not initialized!");
    return TargetIsIPhoneOS;
  }

  void getTargetVersion(unsigned (&Res)[3]) const {
    assert(TargetInitialized && "Target not initialized!");
    Res[0] = TargetVersion[0];
    Res[1] = TargetVersion[1];
    Res[2] = TargetVersion[2];
  }

  void getDarwinVersion(unsigned (&Res)[3]) const {
    Res[0] = DarwinVersion[0];
    Res[1] = DarwinVersion[1];
    Res[2] = DarwinVersion[2];
  }

  /// getDarwinArchName - Get the "Darwin" arch name for a particular compiler
  /// invocation. For example, Darwin treats different ARM variations as
  /// distinct architectures.
  llvm::StringRef getDarwinArchName(const ArgList &Args) const;

  static bool isVersionLT(unsigned (&A)[3], unsigned (&B)[3]) {
    for (unsigned i=0; i < 3; ++i) {
      if (A[i] > B[i]) return false;
      if (A[i] < B[i]) return true;
    }
    return false;
  }

  bool isIPhoneOSVersionLT(unsigned V0, unsigned V1=0, unsigned V2=0) const {
    assert(isTargetIPhoneOS() && "Unexpected call for OS X target!");
    unsigned B[3] = { V0, V1, V2 };
    return isVersionLT(TargetVersion, B);
  }

  bool isMacosxVersionLT(unsigned V0, unsigned V1=0, unsigned V2=0) const {
    assert(!isTargetIPhoneOS() && "Unexpected call for iPhoneOS target!");
    unsigned B[3] = { V0, V1, V2 };
    return isVersionLT(TargetVersion, B);
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
              const unsigned (&DarwinVersion)[3]);

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
            const unsigned (&DarwinVersion)[3],
            const unsigned (&GCCVersion)[3]);

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
