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
class LLVM_LIBRARY_VISIBILITY Generic_GCC : public ToolChain {
protected:
  mutable llvm::DenseMap<unsigned, Tool*> Tools;

public:
  Generic_GCC(const HostInfo &Host, const llvm::Triple& Triple);
  ~Generic_GCC();

  virtual Tool &SelectTool(const Compilation &C, const JobAction &JA) const;

  virtual bool IsUnwindTablesDefault() const;
  virtual const char *GetDefaultRelocationModel() const;
  virtual const char *GetForcedPicModel() const;
};

/// Darwin - The base Darwin tool chain.
class LLVM_LIBRARY_VISIBILITY Darwin : public ToolChain {
public:
  /// The host version.
  unsigned DarwinVersion[3];

private:
  mutable llvm::DenseMap<unsigned, Tool*> Tools;

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

private:
  void AddDeploymentTarget(DerivedArgList &Args) const;

public:
  Darwin(const HostInfo &Host, const llvm::Triple& Triple);
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

  bool isTargetInitialized() const { return TargetInitialized; }

  void getTargetVersion(unsigned (&Res)[3]) const {
    assert(TargetInitialized && "Target not initialized!");
    Res[0] = TargetVersion[0];
    Res[1] = TargetVersion[1];
    Res[2] = TargetVersion[2];
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

  virtual types::ID LookupTypeForExtension(const char *Ext) const;

  virtual DerivedArgList *TranslateArgs(const DerivedArgList &Args,
                                        const char *BoundArch) const;

  virtual Tool &SelectTool(const Compilation &C, const JobAction &JA) const;

  virtual bool IsBlocksDefault() const {
    // Always allow blocks on Darwin; users interested in versioning are
    // expected to use /usr/include/Blocks.h.
    return true;
  }
  virtual bool IsIntegratedAssemblerDefault() const {
#ifdef DISABLE_DEFAULT_INTEGRATED_ASSEMBLER
    return false;
#else
    // Default integrated assembler to on for x86.
    return (getTriple().getArch() == llvm::Triple::x86 ||
            getTriple().getArch() == llvm::Triple::x86_64);
#endif
  }
  virtual bool IsObjCNonFragileABIDefault() const {
    // Non-fragile ABI is default for everything but i386.
    return getTriple().getArch() != llvm::Triple::x86;
  }
  virtual bool IsObjCLegacyDispatchDefault() const {
    // This is only used with the non-fragile ABI.

    // Legacy dispatch is used everywhere except on x86_64.
    return getTriple().getArch() != llvm::Triple::x86_64;
  }
  virtual bool UseObjCMixedDispatch() const {
    // This is only used with the non-fragile ABI and non-legacy dispatch.

    // Mixed dispatch is used everywhere except OS X before 10.6.
    return !(!isTargetIPhoneOS() && isMacosxVersionLT(10, 6));
  }
  virtual bool IsUnwindTablesDefault() const;
  virtual unsigned GetDefaultStackProtectorLevel() const {
    // Stack protectors default to on for 10.6 and beyond.
    return !isTargetIPhoneOS() && !isMacosxVersionLT(10, 6);
  }
  virtual const char *GetDefaultRelocationModel() const;
  virtual const char *GetForcedPicModel() const;

  virtual bool SupportsObjCGC() const;

  virtual bool UseDwarfDebugFlags() const;

  virtual bool UseSjLjExceptions() const;

  /// }
};

/// DarwinClang - The Darwin toolchain used by Clang.
class LLVM_LIBRARY_VISIBILITY DarwinClang : public Darwin {
public:
  DarwinClang(const HostInfo &Host, const llvm::Triple& Triple);

  /// @name Darwin ToolChain Implementation
  /// {

  virtual void AddLinkSearchPathArgs(const ArgList &Args,
                                    ArgStringList &CmdArgs) const;

  virtual void AddLinkRuntimeLibArgs(const ArgList &Args,
                                     ArgStringList &CmdArgs) const;

  /// }
};

/// DarwinGCC - The Darwin toolchain used by GCC.
class LLVM_LIBRARY_VISIBILITY DarwinGCC : public Darwin {
  /// GCC version to use.
  unsigned GCCVersion[3];

  /// The directory suffix for this tool chain.
  std::string ToolChainDir;

public:
  DarwinGCC(const HostInfo &Host, const llvm::Triple& Triple);

  /// @name Darwin ToolChain Implementation
  /// {

  virtual void AddLinkSearchPathArgs(const ArgList &Args,
                                    ArgStringList &CmdArgs) const;

  virtual void AddLinkRuntimeLibArgs(const ArgList &Args,
                                     ArgStringList &CmdArgs) const;

  /// }
};

/// Darwin_Generic_GCC - Generic Darwin tool chain using gcc.
class LLVM_LIBRARY_VISIBILITY Darwin_Generic_GCC : public Generic_GCC {
public:
  Darwin_Generic_GCC(const HostInfo &Host, const llvm::Triple& Triple)
    : Generic_GCC(Host, Triple) {}

  virtual const char *GetDefaultRelocationModel() const { return "pic"; }
};

class LLVM_LIBRARY_VISIBILITY AuroraUX : public Generic_GCC {
public:
  AuroraUX(const HostInfo &Host, const llvm::Triple& Triple);

  virtual Tool &SelectTool(const Compilation &C, const JobAction &JA) const;
};

class LLVM_LIBRARY_VISIBILITY OpenBSD : public Generic_GCC {
public:
  OpenBSD(const HostInfo &Host, const llvm::Triple& Triple);

  virtual Tool &SelectTool(const Compilation &C, const JobAction &JA) const;
};

class LLVM_LIBRARY_VISIBILITY FreeBSD : public Generic_GCC {
public:
  FreeBSD(const HostInfo &Host, const llvm::Triple& Triple);

  virtual Tool &SelectTool(const Compilation &C, const JobAction &JA) const;
};

class LLVM_LIBRARY_VISIBILITY Minix : public Generic_GCC {
public:
  Minix(const HostInfo &Host, const llvm::Triple& Triple);

  virtual Tool &SelectTool(const Compilation &C, const JobAction &JA) const;
};

class LLVM_LIBRARY_VISIBILITY DragonFly : public Generic_GCC {
public:
  DragonFly(const HostInfo &Host, const llvm::Triple& Triple);

  virtual Tool &SelectTool(const Compilation &C, const JobAction &JA) const;
};

class LLVM_LIBRARY_VISIBILITY Linux : public Generic_GCC {
public:
  Linux(const HostInfo &Host, const llvm::Triple& Triple);

  virtual Tool &SelectTool(const Compilation &C, const JobAction &JA) const;
};


/// TCEToolChain - A tool chain using the llvm bitcode tools to perform
/// all subcommands. See http://tce.cs.tut.fi for our peculiar target.
class LLVM_LIBRARY_VISIBILITY TCEToolChain : public ToolChain {
public:
  TCEToolChain(const HostInfo &Host, const llvm::Triple& Triple);
  ~TCEToolChain();

  virtual Tool &SelectTool(const Compilation &C, const JobAction &JA) const;
  bool IsMathErrnoDefault() const;
  bool IsUnwindTablesDefault() const;
  const char* GetDefaultRelocationModel() const;
  const char* GetForcedPicModel() const;

private:
  mutable llvm::DenseMap<unsigned, Tool*> Tools;

};

} // end namespace toolchains
} // end namespace driver
} // end namespace clang

#endif
