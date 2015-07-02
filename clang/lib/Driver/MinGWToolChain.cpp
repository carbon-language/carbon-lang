//===--- MinGWToolChain.cpp - MinGWToolChain Implementation
//-----------------------===//
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

  if (getDriver().SysRoot.size())
    Base = getDriver().SysRoot;
  else if (llvm::ErrorOr<std::string> GPPName =
               llvm::sys::findProgramByName("gcc"))
    Base = llvm::sys::path::parent_path(
        llvm::sys::path::parent_path(GPPName.get()));
  else
    Base = llvm::sys::path::parent_path(getDriver().getInstalledDir());
  Base += llvm::sys::path::get_separator();
  llvm::SmallString<1024> LibDir(Base);
  llvm::sys::path::append(LibDir, "lib", "gcc");
  LibDir += llvm::sys::path::get_separator();

  // First look for mingw-w64.
  Arch = getTriple().getArchName();
  Arch += "-w64-mingw32";
  std::error_code EC;
  llvm::sys::fs::directory_iterator MingW64Entry(LibDir + Arch, EC);
  if (!EC) {
    GccLibDir = MingW64Entry->path();
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
  addSystemInclude(DriverArgs, CC1Args, IncludeDir.c_str());
  addSystemInclude(DriverArgs, CC1Args, Base + Arch + "include");
  addSystemInclude(DriverArgs, CC1Args, Base + "include");
}

void MinGW::AddClangCXXStdlibIncludeArgs(const ArgList &DriverArgs,
                                         ArgStringList &CC1Args) const {
  if (DriverArgs.hasArg(options::OPT_nostdlibinc) ||
      DriverArgs.hasArg(options::OPT_nostdincxx))
    return;

  llvm::SmallString<1024> IncludeDir;
  for (bool MingW64 : {true, false}) {
    if (MingW64)
      IncludeDir = Base + Arch;
    else
      IncludeDir = GccLibDir;
    llvm::sys::path::append(IncludeDir, "include", "c++");
    addSystemInclude(DriverArgs, CC1Args, IncludeDir.str());
    IncludeDir += llvm::sys::path::get_separator();
    addSystemInclude(DriverArgs, CC1Args, IncludeDir.str() + Arch);
    addSystemInclude(DriverArgs, CC1Args, IncludeDir.str() + "backward");
  }
}
