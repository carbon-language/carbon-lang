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

#include "Tools.h"
#include "clang/Basic/VersionTuple.h"
#include "clang/Driver/Action.h"
#include "clang/Driver/Multilib.h"
#include "clang/Driver/ToolChain.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Support/Compiler.h"
#include <set>
#include <vector>

namespace clang {
namespace driver {
namespace toolchains {

/// Generic_GCC - A tool chain using the 'gcc' command to perform
/// all subcommands; this relies on gcc translating the majority of
/// command line options.
class LLVM_LIBRARY_VISIBILITY Generic_GCC : public ToolChain {
protected:
  /// \brief Struct to store and manipulate GCC versions.
  ///
  /// We rely on assumptions about the form and structure of GCC version
  /// numbers: they consist of at most three '.'-separated components, and each
  /// component is a non-negative integer except for the last component. For
  /// the last component we are very flexible in order to tolerate release
  /// candidates or 'x' wildcards.
  ///
  /// Note that the ordering established among GCCVersions is based on the
  /// preferred version string to use. For example we prefer versions without
  /// a hard-coded patch number to those with a hard coded patch number.
  ///
  /// Currently this doesn't provide any logic for textual suffixes to patches
  /// in the way that (for example) Debian's version format does. If that ever
  /// becomes necessary, it can be added.
  struct GCCVersion {
    /// \brief The unparsed text of the version.
    std::string Text;

    /// \brief The parsed major, minor, and patch numbers.
    int Major, Minor, Patch;

    /// \brief The text of the parsed major, and major+minor versions.
    std::string MajorStr, MinorStr;

    /// \brief Any textual suffix on the patch number.
    std::string PatchSuffix;

    static GCCVersion Parse(StringRef VersionText);
    bool isOlderThan(int RHSMajor, int RHSMinor, int RHSPatch,
                     StringRef RHSPatchSuffix = StringRef()) const;
    bool operator<(const GCCVersion &RHS) const {
      return isOlderThan(RHS.Major, RHS.Minor, RHS.Patch, RHS.PatchSuffix);
    }
    bool operator>(const GCCVersion &RHS) const { return RHS < *this; }
    bool operator<=(const GCCVersion &RHS) const { return !(*this > RHS); }
    bool operator>=(const GCCVersion &RHS) const { return !(*this < RHS); }
  };

  /// \brief This is a class to find a viable GCC installation for Clang to
  /// use.
  ///
  /// This class tries to find a GCC installation on the system, and report
  /// information about it. It starts from the host information provided to the
  /// Driver, and has logic for fuzzing that where appropriate.
  class GCCInstallationDetector {
    bool IsValid;
    llvm::Triple GCCTriple;

    // FIXME: These might be better as path objects.
    std::string GCCInstallPath;
    std::string GCCParentLibPath;

    /// The primary multilib appropriate for the given flags.
    Multilib SelectedMultilib;
    /// On Biarch systems, this corresponds to the default multilib when
    /// targeting the non-default multilib. Otherwise, it is empty.
    llvm::Optional<Multilib> BiarchSibling;

    GCCVersion Version;

    // We retain the list of install paths that were considered and rejected in
    // order to print out detailed information in verbose mode.
    std::set<std::string> CandidateGCCInstallPaths;

    /// The set of multilibs that the detected installation supports.
    MultilibSet Multilibs;

  public:
    GCCInstallationDetector() : IsValid(false) {}
    void init(const Driver &D, const llvm::Triple &TargetTriple,
                            const llvm::opt::ArgList &Args);

    /// \brief Check whether we detected a valid GCC install.
    bool isValid() const { return IsValid; }

    /// \brief Get the GCC triple for the detected install.
    const llvm::Triple &getTriple() const { return GCCTriple; }

    /// \brief Get the detected GCC installation path.
    StringRef getInstallPath() const { return GCCInstallPath; }

    /// \brief Get the detected GCC parent lib path.
    StringRef getParentLibPath() const { return GCCParentLibPath; }

    /// \brief Get the detected Multilib
    const Multilib &getMultilib() const { return SelectedMultilib; }

    /// \brief Get the whole MultilibSet
    const MultilibSet &getMultilibs() const { return Multilibs; }

    /// Get the biarch sibling multilib (if it exists).
    /// \return true iff such a sibling exists
    bool getBiarchSibling(Multilib &M) const;

    /// \brief Get the detected GCC version string.
    const GCCVersion &getVersion() const { return Version; }

    /// \brief Print information about the detected GCC installation.
    void print(raw_ostream &OS) const;

  private:
    static void
    CollectLibDirsAndTriples(const llvm::Triple &TargetTriple,
                             const llvm::Triple &BiarchTriple,
                             SmallVectorImpl<StringRef> &LibDirs,
                             SmallVectorImpl<StringRef> &TripleAliases,
                             SmallVectorImpl<StringRef> &BiarchLibDirs,
                             SmallVectorImpl<StringRef> &BiarchTripleAliases);

    void ScanLibDirForGCCTriple(const llvm::Triple &TargetArch,
                                const llvm::opt::ArgList &Args,
                                const std::string &LibDir,
                                StringRef CandidateTriple,
                                bool NeedsBiarchSuffix = false);

    bool findMIPSMultilibs(const llvm::Triple &TargetArch, StringRef Path,
                           const llvm::opt::ArgList &Args);

    bool findBiarchMultilibs(const llvm::Triple &TargetArch, StringRef Path,
                             const llvm::opt::ArgList &Args,
                             bool NeedsBiarchSuffix);
  };

  GCCInstallationDetector GCCInstallation;

public:
  Generic_GCC(const Driver &D, const llvm::Triple &Triple,
              const llvm::opt::ArgList &Args);
  ~Generic_GCC();

  virtual void printVerboseInfo(raw_ostream &OS) const;

  virtual bool IsUnwindTablesDefault() const;
  virtual bool isPICDefault() const;
  virtual bool isPIEDefault() const;
  virtual bool isPICDefaultForced() const;
  virtual bool IsIntegratedAssemblerDefault() const;

protected:
  virtual Tool *getTool(Action::ActionClass AC) const;
  virtual Tool *buildAssembler() const;
  virtual Tool *buildLinker() const;

  /// \name ToolChain Implementation Helper Functions
  /// @{

  /// \brief Check whether the target triple's architecture is 64-bits.
  bool isTarget64Bit() const { return getTriple().isArch64Bit(); }

  /// \brief Check whether the target triple's architecture is 32-bits.
  bool isTarget32Bit() const { return getTriple().isArch32Bit(); }

  /// @}

private:
  mutable OwningPtr<tools::gcc::Preprocess> Preprocess;
  mutable OwningPtr<tools::gcc::Compile> Compile;
};

class LLVM_LIBRARY_VISIBILITY MachO : public ToolChain {
protected:
  virtual Tool *buildAssembler() const;
  virtual Tool *buildLinker() const;
  virtual Tool *getTool(Action::ActionClass AC) const;
private:
  mutable OwningPtr<tools::darwin::Lipo> Lipo;
  mutable OwningPtr<tools::darwin::Dsymutil> Dsymutil;
  mutable OwningPtr<tools::darwin::VerifyDebug> VerifyDebug;

public:
  MachO(const Driver &D, const llvm::Triple &Triple,
             const llvm::opt::ArgList &Args);
  ~MachO();

  /// @name MachO specific toolchain API
  /// {

  /// Get the "MachO" arch name for a particular compiler invocation. For
  /// example, Apple treats different ARM variations as distinct architectures.
  StringRef getMachOArchName(const llvm::opt::ArgList &Args) const;


  /// Add the linker arguments to link the ARC runtime library.
  virtual void AddLinkARCArgs(const llvm::opt::ArgList &Args,
                              llvm::opt::ArgStringList &CmdArgs) const {}

  /// Add the linker arguments to link the compiler runtime library.
  virtual void AddLinkRuntimeLibArgs(const llvm::opt::ArgList &Args,
                                     llvm::opt::ArgStringList &CmdArgs) const;

  virtual void
  addStartObjectFileArgs(const llvm::opt::ArgList &Args,
                         llvm::opt::ArgStringList &CmdArgs) const {}

  virtual void addMinVersionArgs(const llvm::opt::ArgList &Args,
                                 llvm::opt::ArgStringList &CmdArgs) const {}

  /// On some iOS platforms, kernel and kernel modules were built statically. Is
  /// this such a target?
  virtual bool isKernelStatic() const {
    return false;
  }

  /// Is the target either iOS or an iOS simulator?
  bool isTargetIOSBased() const {
    return false;
  }

  void AddLinkRuntimeLib(const llvm::opt::ArgList &Args,
                         llvm::opt::ArgStringList &CmdArgs,
                         StringRef DarwinStaticLib,
                         bool AlwaysLink = false,
                         bool IsEmbedded = false) const;

  /// }
  /// @name ToolChain Implementation
  /// {

  std::string ComputeEffectiveClangTriple(const llvm::opt::ArgList &Args,
                                          types::ID InputType) const;

  virtual types::ID LookupTypeForExtension(const char *Ext) const;

  virtual bool HasNativeLLVMSupport() const;

  virtual llvm::opt::DerivedArgList *
  TranslateArgs(const llvm::opt::DerivedArgList &Args,
                const char *BoundArch) const;

  virtual bool IsBlocksDefault() const {
    // Always allow blocks on Apple; users interested in versioning are
    // expected to use /usr/include/Blocks.h.
    return true;
  }
  virtual bool IsIntegratedAssemblerDefault() const {
    // Default integrated assembler to on for Apple's MachO targets.
    return true;
  }

  virtual bool IsMathErrnoDefault() const {
    return false;
  }

  virtual bool IsEncodeExtendedBlockSignatureDefault() const {
    return true;
  }

  virtual bool IsObjCNonFragileABIDefault() const {
    // Non-fragile ABI is default for everything but i386.
    return getTriple().getArch() != llvm::Triple::x86;
  }

  virtual bool UseObjCMixedDispatch() const {
    return true;
  }

  virtual bool IsUnwindTablesDefault() const;

  virtual RuntimeLibType GetDefaultRuntimeLibType() const {
    return ToolChain::RLT_CompilerRT;
  }

  virtual bool isPICDefault() const;
  virtual bool isPIEDefault() const;
  virtual bool isPICDefaultForced() const;

  virtual bool SupportsProfiling() const;

  virtual bool SupportsObjCGC() const {
    return false;
  }

  virtual bool UseDwarfDebugFlags() const;

  virtual bool UseSjLjExceptions() const {
    return false;
  }

  /// }
};

  /// Darwin - The base Darwin tool chain.
class LLVM_LIBRARY_VISIBILITY Darwin : public MachO {
public:
  /// The host version.
  unsigned DarwinVersion[3];

  /// Whether the information on the target has been initialized.
  //
  // FIXME: This should be eliminated. What we want to do is make this part of
  // the "default target for arguments" selection process, once we get out of
  // the argument translation business.
  mutable bool TargetInitialized;

  enum DarwinPlatformKind {
    MacOS,
    IPhoneOS,
    IPhoneOSSimulator
  };

  mutable DarwinPlatformKind TargetPlatform;

  /// The OS version we are targeting.
  mutable VersionTuple TargetVersion;

private:
  /// The default macosx-version-min of this tool chain; empty until
  /// initialized.
  std::string MacosxVersionMin;

  /// The default ios-version-min of this tool chain; empty until
  /// initialized.
  std::string iOSVersionMin;

private:
  void AddDeploymentTarget(llvm::opt::DerivedArgList &Args) const;

public:
  Darwin(const Driver &D, const llvm::Triple &Triple,
         const llvm::opt::ArgList &Args);
  ~Darwin();

  std::string ComputeEffectiveClangTriple(const llvm::opt::ArgList &Args,
                                          types::ID InputType) const;

  /// @name Apple Specific Toolchain Implementation
  /// {

  virtual void
  addMinVersionArgs(const llvm::opt::ArgList &Args,
                    llvm::opt::ArgStringList &CmdArgs) const override;

  virtual void
  addStartObjectFileArgs(const llvm::opt::ArgList &Args,
                         llvm::opt::ArgStringList &CmdArgs) const override;

  virtual bool isKernelStatic() const {
    return !isTargetIPhoneOS() || isIPhoneOSVersionLT(6, 0);
  }

protected:
  /// }
  /// @name Darwin specific Toolchain functions
  /// {

  // FIXME: Eliminate these ...Target functions and derive separate tool chains
  // for these targets and put version in constructor.
  void setTarget(DarwinPlatformKind Platform, unsigned Major, unsigned Minor,
                 unsigned Micro) const {
    // FIXME: For now, allow reinitialization as long as values don't
    // change. This will go away when we move away from argument translation.
    if (TargetInitialized && TargetPlatform == Platform &&
        TargetVersion == VersionTuple(Major, Minor, Micro))
      return;

    assert(!TargetInitialized && "Target already initialized!");
    TargetInitialized = true;
    TargetPlatform = Platform;
    TargetVersion = VersionTuple(Major, Minor, Micro);
  }

  bool isTargetIPhoneOS() const {
    assert(TargetInitialized && "Target not initialized!");
    return TargetPlatform == IPhoneOS;
  }

  bool isTargetIOSSimulator() const {
    assert(TargetInitialized && "Target not initialized!");
    return TargetPlatform == IPhoneOSSimulator;
  }

  bool isTargetIOSBased() const {
    assert(TargetInitialized && "Target not initialized!");
    return isTargetIPhoneOS() || isTargetIOSSimulator();
  }

  bool isTargetMacOS() const {
    return TargetPlatform == MacOS;
  }

  bool isTargetInitialized() const { return TargetInitialized; }

  VersionTuple getTargetVersion() const {
    assert(TargetInitialized && "Target not initialized!");
    return TargetVersion;
  }

  bool isIPhoneOSVersionLT(unsigned V0, unsigned V1=0, unsigned V2=0) const {
    assert(isTargetIOSBased() && "Unexpected call for non iOS target!");
    return TargetVersion < VersionTuple(V0, V1, V2);
  }

  bool isMacosxVersionLT(unsigned V0, unsigned V1=0, unsigned V2=0) const {
    assert(isTargetMacOS() && "Unexpected call for non OS X target!");
    return TargetVersion < VersionTuple(V0, V1, V2);
  }

public:
  /// }
  /// @name ToolChain Implementation
  /// {

  virtual llvm::opt::DerivedArgList *
  TranslateArgs(const llvm::opt::DerivedArgList &Args,
                const char *BoundArch) const;

  virtual ObjCRuntime getDefaultObjCRuntime(bool isNonFragile) const;
  virtual bool hasBlocksRuntime() const;

  virtual bool UseObjCMixedDispatch() const {
    // This is only used with the non-fragile ABI and non-legacy dispatch.

    // Mixed dispatch is used everywhere except OS X before 10.6.
    return !(isTargetMacOS() && isMacosxVersionLT(10, 6));
  }

  virtual unsigned GetDefaultStackProtectorLevel(bool KernelOrKext) const {
    // Stack protectors default to on for user code on 10.5,
    // and for everything in 10.6 and beyond
    if (isTargetIOSBased())
      return 1;
    else if (isTargetMacOS() && !isMacosxVersionLT(10, 6))
      return 1;
    else if (isTargetMacOS() && !isMacosxVersionLT(10, 5) && !KernelOrKext)
      return 1;

    return 0;
  }

  virtual bool SupportsObjCGC() const;

  virtual void CheckObjCARC() const;

  virtual bool UseSjLjExceptions() const;
};

/// DarwinClang - The Darwin toolchain used by Clang.
class LLVM_LIBRARY_VISIBILITY DarwinClang : public Darwin {
public:
  DarwinClang(const Driver &D, const llvm::Triple &Triple,
              const llvm::opt::ArgList &Args);

  /// @name Apple ToolChain Implementation
  /// {

  virtual void
  AddLinkRuntimeLibArgs(const llvm::opt::ArgList &Args,
                        llvm::opt::ArgStringList &CmdArgs) const override;

  virtual void
  AddCXXStdlibLibArgs(const llvm::opt::ArgList &Args,
                      llvm::opt::ArgStringList &CmdArgs) const override;

  virtual void
  AddCCKextLibArgs(const llvm::opt::ArgList &Args,
                   llvm::opt::ArgStringList &CmdArgs) const override;

  virtual void
  AddLinkARCArgs(const llvm::opt::ArgList &Args,
                 llvm::opt::ArgStringList &CmdArgs) const override;
  /// }
};

class LLVM_LIBRARY_VISIBILITY Generic_ELF : public Generic_GCC {
  virtual void anchor();
public:
  Generic_ELF(const Driver &D, const llvm::Triple &Triple,
              const llvm::opt::ArgList &Args)
      : Generic_GCC(D, Triple, Args) {}

  virtual void addClangTargetOptions(const llvm::opt::ArgList &DriverArgs,
                                     llvm::opt::ArgStringList &CC1Args) const;
};

class LLVM_LIBRARY_VISIBILITY AuroraUX : public Generic_GCC {
public:
  AuroraUX(const Driver &D, const llvm::Triple &Triple,
           const llvm::opt::ArgList &Args);

protected:
  virtual Tool *buildAssembler() const;
  virtual Tool *buildLinker() const;
};

class LLVM_LIBRARY_VISIBILITY Solaris : public Generic_GCC {
public:
  Solaris(const Driver &D, const llvm::Triple &Triple,
          const llvm::opt::ArgList &Args);

  virtual bool IsIntegratedAssemblerDefault() const { return true; }
protected:
  virtual Tool *buildAssembler() const;
  virtual Tool *buildLinker() const;

};


class LLVM_LIBRARY_VISIBILITY OpenBSD : public Generic_ELF {
public:
  OpenBSD(const Driver &D, const llvm::Triple &Triple,
          const llvm::opt::ArgList &Args);

  virtual bool IsMathErrnoDefault() const { return false; }
  virtual bool IsObjCNonFragileABIDefault() const { return true; }
  virtual bool isPIEDefault() const { return true; }

  virtual unsigned GetDefaultStackProtectorLevel(bool KernelOrKext) const {
    return 1;
  }

protected:
  virtual Tool *buildAssembler() const;
  virtual Tool *buildLinker() const;
};

class LLVM_LIBRARY_VISIBILITY Bitrig : public Generic_ELF {
public:
  Bitrig(const Driver &D, const llvm::Triple &Triple,
         const llvm::opt::ArgList &Args);

  virtual bool IsMathErrnoDefault() const { return false; }
  virtual bool IsObjCNonFragileABIDefault() const { return true; }
  virtual bool IsObjCLegacyDispatchDefault() const { return false; }

  virtual void
  AddClangCXXStdlibIncludeArgs(const llvm::opt::ArgList &DriverArgs,
                               llvm::opt::ArgStringList &CC1Args) const;
  virtual void AddCXXStdlibLibArgs(const llvm::opt::ArgList &Args,
                                   llvm::opt::ArgStringList &CmdArgs) const;
  virtual unsigned GetDefaultStackProtectorLevel(bool KernelOrKext) const {
     return 1;
  }

protected:
  virtual Tool *buildAssembler() const;
  virtual Tool *buildLinker() const;
};

class LLVM_LIBRARY_VISIBILITY FreeBSD : public Generic_ELF {
public:
  FreeBSD(const Driver &D, const llvm::Triple &Triple,
          const llvm::opt::ArgList &Args);
  virtual bool HasNativeLLVMSupport() const;

  virtual bool IsMathErrnoDefault() const { return false; }
  virtual bool IsObjCNonFragileABIDefault() const { return true; }

  virtual CXXStdlibType GetCXXStdlibType(const llvm::opt::ArgList &Args) const;
  virtual void
  AddClangCXXStdlibIncludeArgs(const llvm::opt::ArgList &DriverArgs,
                               llvm::opt::ArgStringList &CC1Args) const;
  virtual bool IsIntegratedAssemblerDefault() const {
    if (getTriple().getArch() == llvm::Triple::ppc ||
        getTriple().getArch() == llvm::Triple::ppc64)
      return true;
    return Generic_ELF::IsIntegratedAssemblerDefault();
  }

  virtual bool UseSjLjExceptions() const;
  virtual bool isPIEDefault() const;
protected:
  virtual Tool *buildAssembler() const;
  virtual Tool *buildLinker() const;
};

class LLVM_LIBRARY_VISIBILITY NetBSD : public Generic_ELF {
public:
  NetBSD(const Driver &D, const llvm::Triple &Triple,
         const llvm::opt::ArgList &Args);

  virtual bool IsMathErrnoDefault() const { return false; }
  virtual bool IsObjCNonFragileABIDefault() const { return true; }

  virtual CXXStdlibType GetCXXStdlibType(const llvm::opt::ArgList &Args) const;

  virtual void
  AddClangCXXStdlibIncludeArgs(const llvm::opt::ArgList &DriverArgs,
                               llvm::opt::ArgStringList &CC1Args) const;
  virtual bool IsUnwindTablesDefault() const {
    return true;
  }
  virtual bool IsIntegratedAssemblerDefault() const {
    if (getTriple().getArch() == llvm::Triple::ppc)
      return true;
    return Generic_ELF::IsIntegratedAssemblerDefault();
  }

protected:
  virtual Tool *buildAssembler() const;
  virtual Tool *buildLinker() const;
};

class LLVM_LIBRARY_VISIBILITY Minix : public Generic_ELF {
public:
  Minix(const Driver &D, const llvm::Triple &Triple,
        const llvm::opt::ArgList &Args);

protected:
  virtual Tool *buildAssembler() const;
  virtual Tool *buildLinker() const;
};

class LLVM_LIBRARY_VISIBILITY DragonFly : public Generic_ELF {
public:
  DragonFly(const Driver &D, const llvm::Triple &Triple,
            const llvm::opt::ArgList &Args);

  virtual bool IsMathErrnoDefault() const { return false; }

protected:
  virtual Tool *buildAssembler() const;
  virtual Tool *buildLinker() const;
};

class LLVM_LIBRARY_VISIBILITY Linux : public Generic_ELF {
public:
  Linux(const Driver &D, const llvm::Triple &Triple,
        const llvm::opt::ArgList &Args);

  virtual bool HasNativeLLVMSupport() const;

  virtual void
  AddClangSystemIncludeArgs(const llvm::opt::ArgList &DriverArgs,
                            llvm::opt::ArgStringList &CC1Args) const;
  virtual void
  AddClangCXXStdlibIncludeArgs(const llvm::opt::ArgList &DriverArgs,
                               llvm::opt::ArgStringList &CC1Args) const;
  virtual bool isPIEDefault() const;

  std::string Linker;
  std::vector<std::string> ExtraOpts;

protected:
  virtual Tool *buildAssembler() const;
  virtual Tool *buildLinker() const;

private:
  static bool addLibStdCXXIncludePaths(Twine Base, Twine Suffix,
                                       Twine TargetArchDir, Twine IncludeSuffix,
                                       const llvm::opt::ArgList &DriverArgs,
                                       llvm::opt::ArgStringList &CC1Args);
  static bool addLibStdCXXIncludePaths(Twine Base, Twine TargetArchDir,
                                       const llvm::opt::ArgList &DriverArgs,
                                       llvm::opt::ArgStringList &CC1Args);

  std::string computeSysRoot() const;
};

class LLVM_LIBRARY_VISIBILITY Hexagon_TC : public Linux {
protected:
  GCCVersion GCCLibAndIncVersion;
  virtual Tool *buildAssembler() const;
  virtual Tool *buildLinker() const;

public:
  Hexagon_TC(const Driver &D, const llvm::Triple &Triple,
             const llvm::opt::ArgList &Args);
  ~Hexagon_TC();

  virtual void
  AddClangSystemIncludeArgs(const llvm::opt::ArgList &DriverArgs,
                            llvm::opt::ArgStringList &CC1Args) const;
  virtual void
  AddClangCXXStdlibIncludeArgs(const llvm::opt::ArgList &DriverArgs,
                               llvm::opt::ArgStringList &CC1Args) const;
  virtual CXXStdlibType GetCXXStdlibType(const llvm::opt::ArgList &Args) const;

  StringRef GetGCCLibAndIncVersion() const { return GCCLibAndIncVersion.Text; }

  static std::string GetGnuDir(const std::string &InstalledDir);

  static StringRef GetTargetCPU(const llvm::opt::ArgList &Args);
};

/// TCEToolChain - A tool chain using the llvm bitcode tools to perform
/// all subcommands. See http://tce.cs.tut.fi for our peculiar target.
class LLVM_LIBRARY_VISIBILITY TCEToolChain : public ToolChain {
public:
  TCEToolChain(const Driver &D, const llvm::Triple &Triple,
               const llvm::opt::ArgList &Args);
  ~TCEToolChain();

  bool IsMathErrnoDefault() const;
  bool isPICDefault() const;
  bool isPIEDefault() const;
  bool isPICDefaultForced() const;
};

class LLVM_LIBRARY_VISIBILITY Windows : public ToolChain {
public:
  Windows(const Driver &D, const llvm::Triple &Triple,
          const llvm::opt::ArgList &Args);

  virtual bool IsIntegratedAssemblerDefault() const;
  virtual bool IsUnwindTablesDefault() const;
  virtual bool isPICDefault() const;
  virtual bool isPIEDefault() const;
  virtual bool isPICDefaultForced() const;

  virtual void
  AddClangSystemIncludeArgs(const llvm::opt::ArgList &DriverArgs,
                            llvm::opt::ArgStringList &CC1Args) const;
  virtual void
  AddClangCXXStdlibIncludeArgs(const llvm::opt::ArgList &DriverArgs,
                               llvm::opt::ArgStringList &CC1Args) const;

protected:
  virtual Tool *buildLinker() const;
  virtual Tool *buildAssembler() const;
};


class LLVM_LIBRARY_VISIBILITY XCore : public ToolChain {
public:
  XCore(const Driver &D, const llvm::Triple &Triple,
          const llvm::opt::ArgList &Args);
protected:
  virtual Tool *buildAssembler() const;
  virtual Tool *buildLinker() const;
public:
  virtual bool isPICDefault() const;
  virtual bool isPIEDefault() const;
  virtual bool isPICDefaultForced() const;
  virtual bool SupportsProfiling() const;
  virtual bool hasBlocksRuntime() const;
  virtual void AddClangSystemIncludeArgs(const llvm::opt::ArgList &DriverArgs,
                            llvm::opt::ArgStringList &CC1Args) const;
  virtual void addClangTargetOptions(const llvm::opt::ArgList &DriverArgs,
                                     llvm::opt::ArgStringList &CC1Args) const;
  virtual void AddClangCXXStdlibIncludeArgs(const llvm::opt::ArgList &DriverArgs,
                               llvm::opt::ArgStringList &CC1Args) const;
  virtual void AddCXXStdlibLibArgs(const llvm::opt::ArgList &Args,
                                   llvm::opt::ArgStringList &CmdArgs) const;
};

} // end namespace toolchains
} // end namespace driver
} // end namespace clang

#endif
