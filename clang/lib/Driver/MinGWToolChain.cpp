//===--- MinGWToolChain.cpp - MinGWToolChain Implementation ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ToolChains.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Options.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

using namespace clang::diag;
using namespace clang::driver;
using namespace clang::driver::toolchains;
using namespace clang;
using namespace llvm::opt;

MinGW::MinGW(const Driver &D, const llvm::Triple &Triple, const ArgList &Args)
    : ToolChain(D, Triple, Args) {
  getProgramPaths().push_back(getDriver().getInstalledDir());

  llvm::SmallString<1024> LibDir;

#ifdef LLVM_ON_WIN32
  if (getDriver().SysRoot.size())
    Base = getDriver().SysRoot;
  else if (llvm::ErrorOr<std::string> GPPName =
               llvm::sys::findProgramByName("gcc"))
    Base = llvm::sys::path::parent_path(
        llvm::sys::path::parent_path(GPPName.get()));
  else
    Base = llvm::sys::path::parent_path(getDriver().getInstalledDir());
  Base += llvm::sys::path::get_separator();
  LibDir = Base;
  llvm::sys::path::append(LibDir, "lib", "gcc");
#else
  if (getDriver().SysRoot.size())
    Base = getDriver().SysRoot;
  else
    Base = "/usr/";
  LibDir = Base;
  llvm::sys::path::append(LibDir, "lib64", "gcc");
#endif

  LibDir += llvm::sys::path::get_separator();

  // First look for mingw-w64.
  Arch = getTriple().getArchName();
  Arch += "-w64-mingw32";
  std::error_code EC;
  llvm::sys::fs::directory_iterator MingW64Entry(LibDir + Arch, EC);
  if (!EC) {
    GccLibDir = MingW64Entry->path();
    Ver = llvm::sys::path::filename(GccLibDir);
  } else {
    // If mingw-w64 not found, try looking for mingw.org.
    Arch = "mingw32";
    llvm::sys::fs::directory_iterator MingwOrgEntry(LibDir + Arch, EC);
    if (!EC)
      GccLibDir = MingwOrgEntry->path();
  }
  Arch += llvm::sys::path::get_separator();
  // GccLibDir must precede Base/lib so that the
  // correct crtbegin.o ,cetend.o would be found.
  getFilePaths().push_back(GccLibDir);
  getFilePaths().push_back(Base + "lib");
  getFilePaths().push_back(Base + Arch + "lib");
#ifdef LLVM_ON_UNIX
  // For openSUSE.
  getFilePaths().push_back(Base + Arch + "sys-root/mingw/lib");
#endif
}

bool MinGW::IsIntegratedAssemblerDefault() const { return true; }

Tool *MinGW::getTool(Action::ActionClass AC) const {
  switch (AC) {
  case Action::PreprocessJobClass:
    if (!Preprocessor)
      Preprocessor.reset(new tools::gcc::Preprocessor(*this));
    return Preprocessor.get();
  case Action::CompileJobClass:
    if (!Compiler)
      Compiler.reset(new tools::gcc::Compiler(*this));
    return Compiler.get();
  default:
    return ToolChain::getTool(AC);
  }
}

Tool *MinGW::buildAssembler() const {
  return new tools::MinGW::Assembler(*this);
}

Tool *MinGW::buildLinker() const { return new tools::MinGW::Linker(*this); }

bool MinGW::IsUnwindTablesDefault() const {
  return getArch() == llvm::Triple::x86_64;
}

bool MinGW::isPICDefault() const { return getArch() == llvm::Triple::x86_64; }

bool MinGW::isPIEDefault() const { return false; }

bool MinGW::isPICDefaultForced() const {
  return getArch() == llvm::Triple::x86_64;
}

bool MinGW::UseSEHExceptions() const {
  return getArch() == llvm::Triple::x86_64;
}

void MinGW::AddClangSystemIncludeArgs(const ArgList &DriverArgs,
                                      ArgStringList &CC1Args) const {
  if (DriverArgs.hasArg(options::OPT_nostdinc))
    return;

  if (!DriverArgs.hasArg(options::OPT_nobuiltininc)) {
    SmallString<1024> P(getDriver().ResourceDir);
    llvm::sys::path::append(P, "include");
    addSystemInclude(DriverArgs, CC1Args, P.str());
  }

  if (DriverArgs.hasArg(options::OPT_nostdlibinc))
    return;

  llvm::SmallString<1024> IncludeDir(GccLibDir);
  llvm::sys::path::append(IncludeDir, "include");
  addSystemInclude(DriverArgs, CC1Args, IncludeDir.c_str());
  IncludeDir += "-fixed";
#ifdef LLVM_ON_UNIX
  // For openSUSE.
  addSystemInclude(DriverArgs, CC1Args,
                   "/usr/x86_64-w64-mingw32/sys-root/mingw/include");
#endif
  addSystemInclude(DriverArgs, CC1Args, IncludeDir.c_str());
  addSystemInclude(DriverArgs, CC1Args, Base + Arch + "include");
  addSystemInclude(DriverArgs, CC1Args, Base + "include");
}

void MinGW::AddClangCXXStdlibIncludeArgs(const ArgList &DriverArgs,
                                         ArgStringList &CC1Args) const {
  if (DriverArgs.hasArg(options::OPT_nostdlibinc) ||
      DriverArgs.hasArg(options::OPT_nostdincxx))
    return;

  // C++ includes may be found in several locations depending on distribution.
  // Windows
  // -------
  // mingw-w64 mingw-builds: $sysroot/i686-w64-mingw32/include/c++.
  // mingw-w64 msys2:        $sysroot/include/c++/4.9.2
  // mingw.org:              GccLibDir/include/c++
  //
  // Linux
  // -----
  // openSUSE:               GccLibDir/include/c++
  llvm::SmallVector<llvm::SmallString<1024>, 3> CppIncludeBases;
  CppIncludeBases.emplace_back(Base);
  llvm::sys::path::append(CppIncludeBases[0], Arch, "include", "c++");
  CppIncludeBases.emplace_back(Base);
  llvm::sys::path::append(CppIncludeBases[1], "include", "c++", Ver);
  CppIncludeBases.emplace_back(GccLibDir);
  llvm::sys::path::append(CppIncludeBases[2], "include", "c++");
  for (auto &CppIncludeBase : CppIncludeBases) {
    CppIncludeBase += llvm::sys::path::get_separator();
    addSystemInclude(DriverArgs, CC1Args, CppIncludeBase);
    addSystemInclude(DriverArgs, CC1Args, CppIncludeBase + Arch);
    addSystemInclude(DriverArgs, CC1Args, CppIncludeBase + "backward");
  }
}
