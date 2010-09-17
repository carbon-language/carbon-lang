//===--- ToolChain.cpp - Collections of tools for one platform ----------*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/ToolChain.h"

#include "clang/Driver/Action.h"
#include "clang/Driver/Arg.h"
#include "clang/Driver/ArgList.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/HostInfo.h"
#include "clang/Driver/Options.h"

using namespace clang::driver;

ToolChain::ToolChain(const HostInfo &_Host, const llvm::Triple &_Triple)
  : Host(_Host), Triple(_Triple) {
}

ToolChain::~ToolChain() {
}

const Driver &ToolChain::getDriver() const {
 return Host.getDriver();
}

std::string ToolChain::GetFilePath(const char *Name) const {
  return Host.getDriver().GetFilePath(Name, *this);

}

std::string ToolChain::GetProgramPath(const char *Name, bool WantFile) const {
  return Host.getDriver().GetProgramPath(Name, *this, WantFile);
}

types::ID ToolChain::LookupTypeForExtension(const char *Ext) const {
  return types::lookupTypeForExtension(Ext);
}

bool ToolChain::HasNativeLLVMSupport() const {
  return false;
}

/// getARMTargetCPU - Get the (LLVM) name of the ARM cpu we are targetting.
//
// FIXME: tblgen this.
static const char *getARMTargetCPU(const ArgList &Args,
                                   const llvm::Triple &Triple) {
  // FIXME: Warn on inconsistent use of -mcpu and -march.

  // If we have -mcpu=, use that.
  if (Arg *A = Args.getLastArg(options::OPT_mcpu_EQ))
    return A->getValue(Args);

  llvm::StringRef MArch;
  if (Arg *A = Args.getLastArg(options::OPT_march_EQ)) {
    // Otherwise, if we have -march= choose the base CPU for that arch.
    MArch = A->getValue(Args);
  } else {
    // Otherwise, use the Arch from the triple.
    MArch = Triple.getArchName();
  }

  if (MArch == "armv2" || MArch == "armv2a")
    return "arm2";
  if (MArch == "armv3")
    return "arm6";
  if (MArch == "armv3m")
    return "arm7m";
  if (MArch == "armv4" || MArch == "armv4t")
    return "arm7tdmi";
  if (MArch == "armv5" || MArch == "armv5t")
    return "arm10tdmi";
  if (MArch == "armv5e" || MArch == "armv5te")
    return "arm1026ejs";
  if (MArch == "armv5tej")
    return "arm926ej-s";
  if (MArch == "armv6" || MArch == "armv6k")
    return "arm1136jf-s";
  if (MArch == "armv6j")
    return "arm1136j-s";
  if (MArch == "armv6z" || MArch == "armv6zk")
    return "arm1176jzf-s";
  if (MArch == "armv6t2")
    return "arm1156t2-s";
  if (MArch == "armv7" || MArch == "armv7a" || MArch == "armv7-a")
    return "cortex-a8";
  if (MArch == "armv7r" || MArch == "armv7-r")
    return "cortex-r4";
  if (MArch == "armv7m" || MArch == "armv7-m")
    return "cortex-m3";
  if (MArch == "ep9312")
    return "ep9312";
  if (MArch == "iwmmxt")
    return "iwmmxt";
  if (MArch == "xscale")
    return "xscale";

  // If all else failed, return the most base CPU LLVM supports.
  return "arm7tdmi";
}

/// getLLVMArchSuffixForARM - Get the LLVM arch name to use for a particular
/// CPU.
//
// FIXME: This is redundant with -mcpu, why does LLVM use this.
// FIXME: tblgen this, or kill it!
static const char *getLLVMArchSuffixForARM(llvm::StringRef CPU) {
  if (CPU == "arm7tdmi" || CPU == "arm7tdmi-s" || CPU == "arm710t" ||
      CPU == "arm720t" || CPU == "arm9" || CPU == "arm9tdmi" ||
      CPU == "arm920" || CPU == "arm920t" || CPU == "arm922t" ||
      CPU == "arm940t" || CPU == "ep9312")
    return "v4t";

  if (CPU == "arm10tdmi" || CPU == "arm1020t")
    return "v5";

  if (CPU == "arm9e" || CPU == "arm926ej-s" || CPU == "arm946e-s" ||
      CPU == "arm966e-s" || CPU == "arm968e-s" || CPU == "arm10e" ||
      CPU == "arm1020e" || CPU == "arm1022e" || CPU == "xscale" ||
      CPU == "iwmmxt")
    return "v5e";

  if (CPU == "arm1136j-s" || CPU == "arm1136jf-s" || CPU == "arm1176jz-s" ||
      CPU == "arm1176jzf-s" || CPU == "mpcorenovfp" || CPU == "mpcore")
    return "v6";

  if (CPU == "arm1156t2-s" || CPU == "arm1156t2f-s")
    return "v6t2";

  if (CPU == "cortex-a8" || CPU == "cortex-a9")
    return "v7";

  return "";
}

std::string ToolChain::ComputeLLVMTriple(const ArgList &Args) const {
  switch (getTriple().getArch()) {
  default:
    return getTripleString();

  case llvm::Triple::arm:
  case llvm::Triple::thumb: {
    // FIXME: Factor into subclasses.
    llvm::Triple Triple = getTriple();

    // Thumb2 is the default for V7 on Darwin.
    //
    // FIXME: Thumb should just be another -target-feaure, not in the triple.
    llvm::StringRef Suffix =
      getLLVMArchSuffixForARM(getARMTargetCPU(Args, Triple));
    bool ThumbDefault =
      (Suffix == "v7" && getTriple().getOS() == llvm::Triple::Darwin);
    std::string ArchName = "arm";
    if (Args.hasFlag(options::OPT_mthumb, options::OPT_mno_thumb, ThumbDefault))
      ArchName = "thumb";
    Triple.setArchName(ArchName + Suffix.str());

    return Triple.getTriple();
  }
  }
}

std::string ToolChain::ComputeEffectiveClangTriple(const ArgList &Args) const {
  // Diagnose use of -mmacosx-version-min and -miphoneos-version-min on
  // non-Darwin.
  if (Arg *A = Args.getLastArg(options::OPT_mmacosx_version_min_EQ,
                               options::OPT_miphoneos_version_min_EQ))
    getDriver().Diag(clang::diag::err_drv_clang_unsupported)
      << A->getAsString(Args);

  return ComputeLLVMTriple(Args);
}

ToolChain::CXXStdlibType ToolChain::GetCXXStdlibType(const ArgList &Args) const{
  if (Arg *A = Args.getLastArg(options::OPT_stdlib_EQ)) {
    llvm::StringRef Value = A->getValue(Args);
    if (Value == "libc++")
      return ToolChain::CST_Libcxx;
    if (Value == "libstdc++")
      return ToolChain::CST_Libstdcxx;
    getDriver().Diag(clang::diag::err_drv_invalid_stdlib_name)
      << A->getAsString(Args);
  }

  return ToolChain::CST_Libstdcxx;
}

void ToolChain::AddClangCXXStdlibIncludeArgs(const ArgList &Args,
                                             ArgStringList &CmdArgs) const {
  CXXStdlibType Type = GetCXXStdlibType(Args);

  switch (Type) {
  case ToolChain::CST_Libcxx:
    CmdArgs.push_back("-cxx-system-include");
    CmdArgs.push_back("/usr/include/c++/v1");
    break;

  case ToolChain::CST_Libstdcxx:
    // Currently handled by the mass of goop in InitHeaderSearch.
    break;
  }
}

void ToolChain::AddCXXStdlibLibArgs(const ArgList &Args,
                                    ArgStringList &CmdArgs) const {
  CXXStdlibType Type = GetCXXStdlibType(Args);

  switch (Type) {
  case ToolChain::CST_Libcxx:
    CmdArgs.push_back("-lc++");
    break;

  case ToolChain::CST_Libstdcxx:
    CmdArgs.push_back("-lstdc++");
    break;
  }
}

void ToolChain::AddCCKextLibArgs(const ArgList &Args,
                                 ArgStringList &CmdArgs) const {
  CmdArgs.push_back("-lcc_kext");
}
