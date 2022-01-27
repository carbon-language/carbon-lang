//===-- PPCLinux.cpp - PowerPC ToolChain Implementations --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PPCLinux.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/Options.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

using namespace clang::driver::toolchains;
using namespace llvm::opt;
using namespace llvm::sys;

// Glibc older than 2.32 doesn't fully support IEEE float128. Here we check
// glibc version by looking at dynamic linker name.
static bool GlibcSupportsFloat128(const std::string &Linker) {
  llvm::SmallVector<char, 16> Path;

  // Resolve potential symlinks to linker.
  if (fs::real_path(Linker, Path))
    return false;
  llvm::StringRef LinkerName =
      path::filename(llvm::StringRef(Path.data(), Path.size()));

  // Since glibc 2.34, the installed .so file is not symlink anymore. But we can
  // still safely assume it's newer than 2.32.
  if (LinkerName.startswith("ld64.so"))
    return true;

  if (!LinkerName.startswith("ld-2."))
    return false;
  unsigned Minor = (LinkerName[5] - '0') * 10 + (LinkerName[6] - '0');
  if (Minor < 32)
    return false;

  return true;
}

PPCLinuxToolChain::PPCLinuxToolChain(const Driver &D,
                                     const llvm::Triple &Triple,
                                     const llvm::opt::ArgList &Args)
    : Linux(D, Triple, Args) {
  if (Arg *A = Args.getLastArg(options::OPT_mabi_EQ)) {
    StringRef ABIName = A->getValue();
    if (ABIName == "ieeelongdouble" && !SupportIEEEFloat128(D, Triple, Args))
      D.Diag(diag::warn_drv_unsupported_float_abi_by_lib) << ABIName;
  }
}

void PPCLinuxToolChain::AddClangSystemIncludeArgs(const ArgList &DriverArgs,
                                                  ArgStringList &CC1Args) const {
  if (!DriverArgs.hasArg(clang::driver::options::OPT_nostdinc) &&
      !DriverArgs.hasArg(options::OPT_nobuiltininc)) {
    const Driver &D = getDriver();
    SmallString<128> P(D.ResourceDir);
    llvm::sys::path::append(P, "include", "ppc_wrappers");
    addSystemInclude(DriverArgs, CC1Args, P);
  }

  Linux::AddClangSystemIncludeArgs(DriverArgs, CC1Args);
}

bool PPCLinuxToolChain::SupportIEEEFloat128(
    const Driver &D, const llvm::Triple &Triple,
    const llvm::opt::ArgList &Args) const {
  if (!Triple.isLittleEndian() || !Triple.isPPC64())
    return false;

  if (Args.hasArg(options::OPT_nostdlib, options::OPT_nostdlibxx))
    return true;

  bool HasUnsupportedCXXLib =
      ToolChain::GetCXXStdlibType(Args) == CST_Libcxx &&
      GCCInstallation.getVersion().isOlderThan(12, 1, 0);

  return GlibcSupportsFloat128(Linux::getDynamicLinker(Args)) &&
         !(D.CCCIsCXX() && HasUnsupportedCXXLib);
}
