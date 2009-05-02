//===--- ToolChains.cpp - ToolChain Implementations ---------------------*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ToolChains.h"

#include "clang/Driver/Arg.h"
#include "clang/Driver/ArgList.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/HostInfo.h"
#include "clang/Driver/Option.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/System/Path.h"

#include <cstdlib> // ::getenv

using namespace clang::driver;
using namespace clang::driver::toolchains;

/// Darwin_X86 - Darwin tool chain for i386 and x86_64.

Darwin_X86::Darwin_X86(const HostInfo &Host, const char *Arch, 
                       const char *Platform, const char *OS, 
                       const unsigned (&_DarwinVersion)[3],
                       const unsigned (&_GCCVersion)[3])
  : ToolChain(Host, Arch, Platform, OS) {
  DarwinVersion[0] = _DarwinVersion[0];
  DarwinVersion[1] = _DarwinVersion[1];
  DarwinVersion[2] = _DarwinVersion[2];
  GCCVersion[0] = _GCCVersion[0];
  GCCVersion[1] = _GCCVersion[1];
  GCCVersion[2] = _GCCVersion[2];

  llvm::raw_string_ostream(MacosxVersionMin)
    << "10." << DarwinVersion[0] - 4 << '.' << DarwinVersion[1];

  ToolChainDir = "i686-apple-darwin";
  ToolChainDir += llvm::utostr(DarwinVersion[0]);
  ToolChainDir += "/";
  ToolChainDir += llvm::utostr(GCCVersion[0]);
  ToolChainDir += '.';
  ToolChainDir += llvm::utostr(GCCVersion[1]);
  ToolChainDir += '.';
  ToolChainDir += llvm::utostr(GCCVersion[2]);

  std::string Path;
  if (getArchName() == "x86_64") {
    Path = getHost().getDriver().Dir;
    Path += "/../lib/gcc/";
    Path += getToolChainDir();
    Path += "/x86_64";
    getFilePaths().push_back(Path);

    Path = "/usr/lib/gcc/";
    Path += getToolChainDir();
    Path += "/x86_64";
    getFilePaths().push_back(Path);
  }
  
  Path = getHost().getDriver().Dir;
  Path += "/../lib/gcc/";
  Path += getToolChainDir();
  getFilePaths().push_back(Path);

  Path = "/usr/lib/gcc/";
  Path += getToolChainDir();
  getFilePaths().push_back(Path);

  Path = getHost().getDriver().Dir;
  Path += "/../libexec/gcc/";
  Path += getToolChainDir();
  getProgramPaths().push_back(Path);

  Path = "/usr/libexec/gcc/";
  Path += getToolChainDir();
  getProgramPaths().push_back(Path);

  Path = getHost().getDriver().Dir;
  Path += "/../libexec";
  getProgramPaths().push_back(Path);

  getProgramPaths().push_back(getHost().getDriver().Dir);
}

Darwin_X86::~Darwin_X86() {
  // Free tool implementations.
  for (llvm::DenseMap<unsigned, Tool*>::iterator
         it = Tools.begin(), ie = Tools.end(); it != ie; ++it)
    delete it->second;
}

Tool &Darwin_X86::SelectTool(const Compilation &C, 
                              const JobAction &JA) const {
  Action::ActionClass Key;
  if (getHost().getDriver().ShouldUseClangCompiler(C, JA, getArchName()))
    Key = Action::AnalyzeJobClass;
  else
    Key = JA.getKind();

  Tool *&T = Tools[Key];
  if (!T) {
    switch (Key) {
    case Action::InputClass:
    case Action::BindArchClass:
      assert(0 && "Invalid tool kind.");
    case Action::PreprocessJobClass:
      T = new tools::darwin::Preprocess(*this); break;
    case Action::AnalyzeJobClass:
      T = new tools::Clang(*this); break;
    case Action::PrecompileJobClass:
    case Action::CompileJobClass:
      T = new tools::darwin::Compile(*this); break;
    case Action::AssembleJobClass:
      T = new tools::darwin::Assemble(*this); break;
    case Action::LinkJobClass:
      T = new tools::darwin::Link(*this, MacosxVersionMin.c_str()); break;
    case Action::LipoJobClass:
      T = new tools::darwin::Lipo(*this); break;
    }
  }

  return *T;
}

DerivedArgList *Darwin_X86::TranslateArgs(InputArgList &Args) const { 
  DerivedArgList *DAL = new DerivedArgList(Args, false);
  const OptTable &Opts = getHost().getDriver().getOpts();

  // FIXME: We really want to get out of the tool chain level argument
  // translation business, as it makes the driver functionality much
  // more opaque. For now, we follow gcc closely solely for the
  // purpose of easily achieving feature parity & testability. Once we
  // have something that works, we should reevaluate each translation
  // and try to push it down into tool specific logic.  

  Arg *OSXVersion = 
    Args.getLastArg(options::OPT_mmacosx_version_min_EQ, false);
  Arg *iPhoneVersion =
    Args.getLastArg(options::OPT_miphoneos_version_min_EQ, false);  
  if (OSXVersion && iPhoneVersion) {
    getHost().getDriver().Diag(clang::diag::err_drv_argument_not_allowed_with)
          << OSXVersion->getAsString(Args)
          << iPhoneVersion->getAsString(Args); 
  } else if (!OSXVersion && !iPhoneVersion) {
    // Chose the default version based on the arch.
    //
    // FIXME: This will need to be fixed when we merge in arm support.

    // Look for MACOSX_DEPLOYMENT_TARGET, otherwise use the version
    // from the host.
    const char *Version = ::getenv("MACOSX_DEPLOYMENT_TARGET");
    if (!Version)
      Version = MacosxVersionMin.c_str();
    const Option *O = Opts.getOption(options::OPT_mmacosx_version_min_EQ);
    DAL->append(DAL->MakeJoinedArg(0, O, Version));
  }
  
  for (ArgList::iterator it = Args.begin(), ie = Args.end(); it != ie; ++it) {
    Arg *A = *it;

    if (A->getOption().matches(options::OPT_Xarch__)) {
      // FIXME: Canonicalize name.
      if (getArchName() != A->getValue(Args, 0))
        continue;

      // FIXME: The arg is leaked here, and we should have a nicer
      // interface for this.
      unsigned Prev, Index = Prev = A->getIndex() + 1;
      Arg *XarchArg = Opts.ParseOneArg(Args, Index);
      
      // If the argument parsing failed or more than one argument was
      // consumed, the -Xarch_ argument's parameter tried to consume
      // extra arguments. Emit an error and ignore.
      //
      // We also want to disallow any options which would alter the
      // driver behavior; that isn't going to work in our model. We
      // use isDriverOption() as an approximation, although things
      // like -O4 are going to slip through.
      if (!XarchArg || Index > Prev + 1 || 
          XarchArg->getOption().isDriverOption()) {
       getHost().getDriver().Diag(clang::diag::err_drv_invalid_Xarch_argument)
          << A->getAsString(Args);
        continue;
      }

      XarchArg->setBaseArg(A);
      A = XarchArg;
    } 

    // Sob. These is strictly gcc compatible for the time being. Apple
    // gcc translates options twice, which means that self-expanding
    // options add duplicates.
    options::ID id = A->getOption().getId();
    switch (id) {
    default:
      DAL->append(A);
      break;

    case options::OPT_mkernel:
    case options::OPT_fapple_kext:
      DAL->append(A);
      DAL->append(DAL->MakeFlagArg(A, Opts.getOption(options::OPT_static)));
      DAL->append(DAL->MakeFlagArg(A, Opts.getOption(options::OPT_static)));
      break;
      
    case options::OPT_dependency_file:
      DAL->append(DAL->MakeSeparateArg(A, Opts.getOption(options::OPT_MF),
                                       A->getValue(Args)));
      break;

    case options::OPT_gfull:
      DAL->append(DAL->MakeFlagArg(A, Opts.getOption(options::OPT_g_Flag)));
      DAL->append(DAL->MakeFlagArg(A,
             Opts.getOption(options::OPT_fno_eliminate_unused_debug_symbols)));
      break;

    case options::OPT_gused:
      DAL->append(DAL->MakeFlagArg(A, Opts.getOption(options::OPT_g_Flag)));
      DAL->append(DAL->MakeFlagArg(A,
             Opts.getOption(options::OPT_feliminate_unused_debug_symbols)));
      break;

    case options::OPT_fterminated_vtables:
    case options::OPT_findirect_virtual_calls:
      DAL->append(DAL->MakeFlagArg(A,
                                   Opts.getOption(options::OPT_fapple_kext)));
      DAL->append(DAL->MakeFlagArg(A, Opts.getOption(options::OPT_static)));
      break;

    case options::OPT_shared:
      DAL->append(DAL->MakeFlagArg(A, Opts.getOption(options::OPT_dynamiclib)));
      break;

    case options::OPT_fconstant_cfstrings:
      DAL->append(DAL->MakeFlagArg(A,
                             Opts.getOption(options::OPT_mconstant_cfstrings)));
      break;

    case options::OPT_fno_constant_cfstrings:
      DAL->append(DAL->MakeFlagArg(A,
                          Opts.getOption(options::OPT_mno_constant_cfstrings)));
      break;

    case options::OPT_Wnonportable_cfstrings:
      DAL->append(DAL->MakeFlagArg(A,
                     Opts.getOption(options::OPT_mwarn_nonportable_cfstrings)));
      break;

    case options::OPT_Wno_nonportable_cfstrings:
      DAL->append(DAL->MakeFlagArg(A,
                  Opts.getOption(options::OPT_mno_warn_nonportable_cfstrings)));
      break;

    case options::OPT_fpascal_strings:
      DAL->append(DAL->MakeFlagArg(A,
                                 Opts.getOption(options::OPT_mpascal_strings)));
      break;

    case options::OPT_fno_pascal_strings:
      DAL->append(DAL->MakeFlagArg(A,
                              Opts.getOption(options::OPT_mno_pascal_strings)));
      break;
    }
  }

  // FIXME: Actually, gcc always adds this, but it is filtered for
  // duplicates somewhere. This also changes the order of things, so
  // look it up.
  if (getArchName() == "x86_64")
    if (!Args.hasArg(options::OPT_m64, false))
      DAL->append(DAL->MakeFlagArg(0, Opts.getOption(options::OPT_m64)));

  if (!Args.hasArg(options::OPT_mtune_EQ, false))
    DAL->append(DAL->MakeJoinedArg(0, Opts.getOption(options::OPT_mtune_EQ),
                                    "core2"));

  return DAL;
} 

bool Darwin_X86::IsMathErrnoDefault() const { 
  return false; 
}

bool Darwin_X86::IsUnwindTablesDefault() const {
  // FIXME: Gross; we should probably have some separate target
  // definition, possibly even reusing the one in clang.
  return getArchName() == "x86_64";
}

const char *Darwin_X86::GetDefaultRelocationModel() const {
  return "pic";
}

const char *Darwin_X86::GetForcedPicModel() const {
  if (getArchName() == "x86_64")
    return "pic";
  return 0;
}

/// Generic_GCC - A tool chain using the 'gcc' command to perform
/// all subcommands; this relies on gcc translating the majority of
/// command line options.

Generic_GCC::Generic_GCC(const HostInfo &Host, const char *Arch, 
                         const char *Platform, const char *OS)
  : ToolChain(Host, Arch, Platform, OS) 
{
  std::string Path(getHost().getDriver().Dir);
  Path += "/../libexec";
  getProgramPaths().push_back(Path);

  getProgramPaths().push_back(getHost().getDriver().Dir);  
}

Generic_GCC::~Generic_GCC() {
  // Free tool implementations.
  for (llvm::DenseMap<unsigned, Tool*>::iterator
         it = Tools.begin(), ie = Tools.end(); it != ie; ++it)
    delete it->second;
}

Tool &Generic_GCC::SelectTool(const Compilation &C, 
                              const JobAction &JA) const {
  Action::ActionClass Key;
  if (getHost().getDriver().ShouldUseClangCompiler(C, JA, getArchName()))
    Key = Action::AnalyzeJobClass;
  else
    Key = JA.getKind();

  Tool *&T = Tools[Key];
  if (!T) {
    switch (Key) {
    case Action::InputClass:
    case Action::BindArchClass:
      assert(0 && "Invalid tool kind.");
    case Action::PreprocessJobClass:
      T = new tools::gcc::Preprocess(*this); break;
    case Action::PrecompileJobClass:
      T = new tools::gcc::Precompile(*this); break;
    case Action::AnalyzeJobClass:
      T = new tools::Clang(*this); break;
    case Action::CompileJobClass:
      T = new tools::gcc::Compile(*this); break;
    case Action::AssembleJobClass:
      T = new tools::gcc::Assemble(*this); break;
    case Action::LinkJobClass:
      T = new tools::gcc::Link(*this); break;
      
      // This is a bit ungeneric, but the only platform using a driver
      // driver is Darwin.
    case Action::LipoJobClass:
      T = new tools::darwin::Lipo(*this); break;
    }
  }

  return *T;
}

bool Generic_GCC::IsMathErrnoDefault() const { 
  return true; 
}

bool Generic_GCC::IsUnwindTablesDefault() const {
  // FIXME: Gross; we should probably have some separate target
  // definition, possibly even reusing the one in clang.
  return getArchName() == "x86_64";
}

const char *Generic_GCC::GetDefaultRelocationModel() const {
  return "static";
}

const char *Generic_GCC::GetForcedPicModel() const {
  return 0;
}

DerivedArgList *Generic_GCC::TranslateArgs(InputArgList &Args) const {
  return new DerivedArgList(Args, true);
}

/// FreeBSD - FreeBSD tool chain which can call as(1) and ld(1) directly.

FreeBSD::FreeBSD(const HostInfo &Host, const char *Arch, 
                 const char *Platform, const char *OS, bool Lib32)
  : Generic_GCC(Host, Arch, Platform, OS) {
  if (Lib32) {
    getFilePaths().push_back(getHost().getDriver().Dir + "/../lib32");
    getFilePaths().push_back("/usr/lib32");
  } else {
    getFilePaths().push_back(getHost().getDriver().Dir + "/../lib");
    getFilePaths().push_back("/usr/lib");
  }
}

Tool &FreeBSD::SelectTool(const Compilation &C, const JobAction &JA) const {
  Action::ActionClass Key;
  if (getHost().getDriver().ShouldUseClangCompiler(C, JA, getArchName()))
    Key = Action::AnalyzeJobClass;
  else
    Key = JA.getKind();

  Tool *&T = Tools[Key];
  if (!T) {
    switch (Key) {
    case Action::AssembleJobClass:
      T = new tools::freebsd::Assemble(*this); break;
    case Action::LinkJobClass:
      T = new tools::freebsd::Link(*this); break;
    default:
      T = &Generic_GCC::SelectTool(C, JA);
    }
  }

  return *T;
}

/// DragonFly - DragonFly tool chain which can call as(1) and ld(1) directly.

DragonFly::DragonFly(const HostInfo &Host, const char *Arch, 
                 const char *Platform, const char *OS)
  : Generic_GCC(Host, Arch, Platform, OS) {

  // Path mangling to find libexec
  std::string Path(getHost().getDriver().Dir);

  Path += "/../libexec";
  getProgramPaths().push_back(Path);
  getProgramPaths().push_back(getHost().getDriver().Dir);  

  getFilePaths().push_back(getHost().getDriver().Dir + "/../lib");
  getFilePaths().push_back("/usr/lib");
  getFilePaths().push_back("/usr/lib/gcc41");
}

Tool &DragonFly::SelectTool(const Compilation &C, const JobAction &JA) const {
  Action::ActionClass Key;
  if (getHost().getDriver().ShouldUseClangCompiler(C, JA, getArchName()))
    Key = Action::AnalyzeJobClass;
  else
    Key = JA.getKind();

  Tool *&T = Tools[Key];
  if (!T) {
    switch (Key) {
    case Action::AssembleJobClass:
      T = new tools::dragonfly::Assemble(*this); break;
    case Action::LinkJobClass:
      T = new tools::dragonfly::Link(*this); break;
    default:
      T = &Generic_GCC::SelectTool(C, JA);
    }
  }

  return *T;
}
