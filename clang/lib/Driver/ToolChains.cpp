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
#include "llvm/System/Path.h"

using namespace clang::driver;
using namespace clang::driver::toolchains;

/// Darwin_X86 - Darwin tool chain for i386 and x86_64.

Darwin_X86::Darwin_X86(const HostInfo &Host, const char *Arch, 
                       const char *Platform, const char *OS, 
                       const unsigned (&_DarwinVersion)[3],
                       const unsigned (&_GCCVersion)[3])
  : ToolChain(Host, Arch, Platform, OS) 
{
  DarwinVersion[0] = _DarwinVersion[0];
  DarwinVersion[1] = _DarwinVersion[1];
  DarwinVersion[2] = _DarwinVersion[2];
  GCCVersion[0] = _GCCVersion[0];
  GCCVersion[1] = _GCCVersion[1];
  GCCVersion[2] = _GCCVersion[2];

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
      T = new tools::gcc::Preprocess(*this); break;
    case Action::PrecompileJobClass:
      T = new tools::gcc::Precompile(*this); break;
    case Action::AnalyzeJobClass:
      T = new tools::Clang(*this); break;
    case Action::CompileJobClass:
      T = new tools::gcc::Compile(*this); break;
    case Action::AssembleJobClass:
      T = new tools::darwin::Assemble(*this); break;
    case Action::LinkJobClass:
      T = new tools::gcc::Link(*this); break;
    case Action::LipoJobClass:
      T = new tools::darwin::Lipo(*this); break;
    }
  }

  return *T;
}

DerivedArgList *Darwin_X86::TranslateArgs(InputArgList &Args) const { 
  DerivedArgList *DAL = new DerivedArgList(Args, false);
  
  for (ArgList::iterator it = Args.begin(), ie = Args.end(); it != ie; ++it) {
    Arg *A = *it;

    if (A->getOption().matches(options::OPT_Xarch__)) {
      // FIXME: Canonicalize name.
      if (getArchName() != A->getValue(Args, 0))
        continue;

      // FIXME: The arg is leaked here, and we should have a nicer
      // interface for this.
      const Driver &D = getHost().getDriver();
      unsigned Prev, Index = Prev = A->getIndex() + 1;
      Arg *XarchArg = D.getOpts().ParseOneArg(Args, Index);
      
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
        D.Diag(clang::diag::err_drv_invalid_Xarch_argument)
          << A->getAsString(Args);
        continue;
      }

      A = XarchArg;
    } 

    // FIXME: Translate.
    DAL->append(A);
  }

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
