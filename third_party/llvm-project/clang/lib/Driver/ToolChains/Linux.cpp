//===--- Linux.h - Linux ToolChain Implementations --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Linux.h"
#include "Arch/ARM.h"
#include "Arch/Mips.h"
#include "Arch/PPC.h"
#include "Arch/RISCV.h"
#include "CommonArgs.h"
#include "clang/Config/config.h"
#include "clang/Driver/Distro.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Options.h"
#include "clang/Driver/SanitizerArgs.h"
#include "llvm/Option/ArgList.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/VirtualFileSystem.h"
#include <system_error>

using namespace clang::driver;
using namespace clang::driver::toolchains;
using namespace clang;
using namespace llvm::opt;

using tools::addPathIfExists;

/// Get our best guess at the multiarch triple for a target.
///
/// Debian-based systems are starting to use a multiarch setup where they use
/// a target-triple directory in the library and header search paths.
/// Unfortunately, this triple does not align with the vanilla target triple,
/// so we provide a rough mapping here.
std::string Linux::getMultiarchTriple(const Driver &D,
                                      const llvm::Triple &TargetTriple,
                                      StringRef SysRoot) const {
  llvm::Triple::EnvironmentType TargetEnvironment =
      TargetTriple.getEnvironment();
  bool IsAndroid = TargetTriple.isAndroid();
  bool IsMipsR6 = TargetTriple.getSubArch() == llvm::Triple::MipsSubArch_r6;
  bool IsMipsN32Abi = TargetTriple.getEnvironment() == llvm::Triple::GNUABIN32;

  // For most architectures, just use whatever we have rather than trying to be
  // clever.
  switch (TargetTriple.getArch()) {
  default:
    break;

  // We use the existence of '/lib/<triple>' as a directory to detect some
  // common linux triples that don't quite match the Clang triple for both
  // 32-bit and 64-bit targets. Multiarch fixes its install triples to these
  // regardless of what the actual target triple is.
  case llvm::Triple::arm:
  case llvm::Triple::thumb:
    if (IsAndroid)
      return "arm-linux-androideabi";
    if (TargetEnvironment == llvm::Triple::GNUEABIHF)
      return "arm-linux-gnueabihf";
    return "arm-linux-gnueabi";
  case llvm::Triple::armeb:
  case llvm::Triple::thumbeb:
    if (TargetEnvironment == llvm::Triple::GNUEABIHF)
      return "armeb-linux-gnueabihf";
    return "armeb-linux-gnueabi";
  case llvm::Triple::x86:
    if (IsAndroid)
      return "i686-linux-android";
    return "i386-linux-gnu";
  case llvm::Triple::x86_64:
    if (IsAndroid)
      return "x86_64-linux-android";
    if (TargetEnvironment == llvm::Triple::GNUX32)
      return "x86_64-linux-gnux32";
    return "x86_64-linux-gnu";
  case llvm::Triple::aarch64:
    if (IsAndroid)
      return "aarch64-linux-android";
    return "aarch64-linux-gnu";
  case llvm::Triple::aarch64_be:
    return "aarch64_be-linux-gnu";

  case llvm::Triple::m68k:
    return "m68k-linux-gnu";

  case llvm::Triple::mips:
    return IsMipsR6 ? "mipsisa32r6-linux-gnu" : "mips-linux-gnu";
  case llvm::Triple::mipsel:
    if (IsAndroid)
      return "mipsel-linux-android";
    return IsMipsR6 ? "mipsisa32r6el-linux-gnu" : "mipsel-linux-gnu";
  case llvm::Triple::mips64: {
    std::string MT = std::string(IsMipsR6 ? "mipsisa64r6" : "mips64") +
                     "-linux-" + (IsMipsN32Abi ? "gnuabin32" : "gnuabi64");
    if (D.getVFS().exists(SysRoot + "/lib/" + MT))
      return MT;
    if (D.getVFS().exists(SysRoot + "/lib/mips64-linux-gnu"))
      return "mips64-linux-gnu";
    break;
  }
  case llvm::Triple::mips64el: {
    if (IsAndroid)
      return "mips64el-linux-android";
    std::string MT = std::string(IsMipsR6 ? "mipsisa64r6el" : "mips64el") +
                     "-linux-" + (IsMipsN32Abi ? "gnuabin32" : "gnuabi64");
    if (D.getVFS().exists(SysRoot + "/lib/" + MT))
      return MT;
    if (D.getVFS().exists(SysRoot + "/lib/mips64el-linux-gnu"))
      return "mips64el-linux-gnu";
    break;
  }
  case llvm::Triple::ppc:
    if (D.getVFS().exists(SysRoot + "/lib/powerpc-linux-gnuspe"))
      return "powerpc-linux-gnuspe";
    return "powerpc-linux-gnu";
  case llvm::Triple::ppcle:
    return "powerpcle-linux-gnu";
  case llvm::Triple::ppc64:
    return "powerpc64-linux-gnu";
  case llvm::Triple::ppc64le:
    return "powerpc64le-linux-gnu";
  case llvm::Triple::sparc:
    return "sparc-linux-gnu";
  case llvm::Triple::sparcv9:
    return "sparc64-linux-gnu";
  case llvm::Triple::systemz:
    return "s390x-linux-gnu";
  }
  return TargetTriple.str();
}

static StringRef getOSLibDir(const llvm::Triple &Triple, const ArgList &Args) {
  if (Triple.isMIPS()) {
    if (Triple.isAndroid()) {
      StringRef CPUName;
      StringRef ABIName;
      tools::mips::getMipsCPUAndABI(Args, Triple, CPUName, ABIName);
      if (CPUName == "mips32r6")
        return "libr6";
      if (CPUName == "mips32r2")
        return "libr2";
    }
    // lib32 directory has a special meaning on MIPS targets.
    // It contains N32 ABI binaries. Use this folder if produce
    // code for N32 ABI only.
    if (tools::mips::hasMipsAbiArg(Args, "n32"))
      return "lib32";
    return Triple.isArch32Bit() ? "lib" : "lib64";
  }

  // It happens that only x86, PPC and SPARC use the 'lib32' variant of
  // oslibdir, and using that variant while targeting other architectures causes
  // problems because the libraries are laid out in shared system roots that
  // can't cope with a 'lib32' library search path being considered. So we only
  // enable them when we know we may need it.
  //
  // FIXME: This is a bit of a hack. We should really unify this code for
  // reasoning about oslibdir spellings with the lib dir spellings in the
  // GCCInstallationDetector, but that is a more significant refactoring.
  if (Triple.getArch() == llvm::Triple::x86 || Triple.isPPC32() ||
      Triple.getArch() == llvm::Triple::sparc)
    return "lib32";

  if (Triple.getArch() == llvm::Triple::x86_64 && Triple.isX32())
    return "libx32";

  if (Triple.getArch() == llvm::Triple::riscv32)
    return "lib32";

  return Triple.isArch32Bit() ? "lib" : "lib64";
}

Linux::Linux(const Driver &D, const llvm::Triple &Triple, const ArgList &Args)
    : Generic_ELF(D, Triple, Args) {
  GCCInstallation.init(Triple, Args);
  Multilibs = GCCInstallation.getMultilibs();
  SelectedMultilib = GCCInstallation.getMultilib();
  llvm::Triple::ArchType Arch = Triple.getArch();
  std::string SysRoot = computeSysRoot();
  ToolChain::path_list &PPaths = getProgramPaths();

  Generic_GCC::PushPPaths(PPaths);

  Distro Distro(D.getVFS(), Triple);

  if (Distro.IsAlpineLinux() || Triple.isAndroid()) {
    ExtraOpts.push_back("-z");
    ExtraOpts.push_back("now");
  }

  if (Distro.IsOpenSUSE() || Distro.IsUbuntu() || Distro.IsAlpineLinux() ||
      Triple.isAndroid()) {
    ExtraOpts.push_back("-z");
    ExtraOpts.push_back("relro");
  }

  // Android ARM/AArch64 use max-page-size=4096 to reduce VMA usage. Note, lld
  // from 11 onwards default max-page-size to 65536 for both ARM and AArch64.
  if ((Triple.isARM() || Triple.isAArch64()) && Triple.isAndroid()) {
    ExtraOpts.push_back("-z");
    ExtraOpts.push_back("max-page-size=4096");
  }

  if (GCCInstallation.getParentLibPath().contains("opt/rh/"))
    // With devtoolset on RHEL, we want to add a bin directory that is relative
    // to the detected gcc install, because if we are using devtoolset gcc then
    // we want to use other tools from devtoolset (e.g. ld) instead of the
    // standard system tools.
    PPaths.push_back(Twine(GCCInstallation.getParentLibPath() +
                     "/../bin").str());

  if (Arch == llvm::Triple::arm || Arch == llvm::Triple::thumb)
    ExtraOpts.push_back("-X");

  const bool IsAndroid = Triple.isAndroid();
  const bool IsMips = Triple.isMIPS();
  const bool IsHexagon = Arch == llvm::Triple::hexagon;
  const bool IsRISCV = Triple.isRISCV();

  if (IsMips && !SysRoot.empty())
    ExtraOpts.push_back("--sysroot=" + SysRoot);

  // Do not use 'gnu' hash style for Mips targets because .gnu.hash
  // and the MIPS ABI require .dynsym to be sorted in different ways.
  // .gnu.hash needs symbols to be grouped by hash code whereas the MIPS
  // ABI requires a mapping between the GOT and the symbol table.
  // Android loader does not support .gnu.hash until API 23.
  // Hexagon linker/loader does not support .gnu.hash
  if (!IsMips && !IsHexagon) {
    if (Distro.IsRedhat() || Distro.IsOpenSUSE() || Distro.IsAlpineLinux() ||
        (Distro.IsUbuntu() && Distro >= Distro::UbuntuMaverick) ||
        (IsAndroid && !Triple.isAndroidVersionLT(23)))
      ExtraOpts.push_back("--hash-style=gnu");

    if (Distro.IsDebian() || Distro.IsOpenSUSE() ||
        Distro == Distro::UbuntuLucid || Distro == Distro::UbuntuJaunty ||
        Distro == Distro::UbuntuKarmic ||
        (IsAndroid && Triple.isAndroidVersionLT(23)))
      ExtraOpts.push_back("--hash-style=both");
  }

#ifdef ENABLE_LINKER_BUILD_ID
  ExtraOpts.push_back("--build-id");
#endif

  if (IsAndroid || Distro.IsOpenSUSE())
    ExtraOpts.push_back("--enable-new-dtags");

  // The selection of paths to try here is designed to match the patterns which
  // the GCC driver itself uses, as this is part of the GCC-compatible driver.
  // This was determined by running GCC in a fake filesystem, creating all
  // possible permutations of these directories, and seeing which ones it added
  // to the link paths.
  path_list &Paths = getFilePaths();

  const std::string OSLibDir = std::string(getOSLibDir(Triple, Args));
  const std::string MultiarchTriple = getMultiarchTriple(D, Triple, SysRoot);

  // mips32: Debian multilib, we use /libo32, while in other case, /lib is
  // used. We need add both libo32 and /lib.
  if (Arch == llvm::Triple::mips || Arch == llvm::Triple::mipsel) {
    Generic_GCC::AddMultilibPaths(D, SysRoot, "libo32", MultiarchTriple, Paths);
    addPathIfExists(D, SysRoot + "/libo32", Paths);
    addPathIfExists(D, SysRoot + "/usr/libo32", Paths);
  }
  Generic_GCC::AddMultilibPaths(D, SysRoot, OSLibDir, MultiarchTriple, Paths);

  addPathIfExists(D, SysRoot + "/lib/" + MultiarchTriple, Paths);
  addPathIfExists(D, SysRoot + "/lib/../" + OSLibDir, Paths);

  if (IsAndroid) {
    // Android sysroots contain a library directory for each supported OS
    // version as well as some unversioned libraries in the usual multiarch
    // directory.
    addPathIfExists(
        D,
        SysRoot + "/usr/lib/" + MultiarchTriple + "/" +
            llvm::to_string(Triple.getEnvironmentVersion().getMajor()),
        Paths);
  }

  addPathIfExists(D, SysRoot + "/usr/lib/" + MultiarchTriple, Paths);
  // 64-bit OpenEmbedded sysroots may not have a /usr/lib dir. So they cannot
  // find /usr/lib64 as it is referenced as /usr/lib/../lib64. So we handle
  // this here.
  if (Triple.getVendor() == llvm::Triple::OpenEmbedded &&
      Triple.isArch64Bit())
    addPathIfExists(D, SysRoot + "/usr/" + OSLibDir, Paths);
  else
    addPathIfExists(D, SysRoot + "/usr/lib/../" + OSLibDir, Paths);
  if (IsRISCV) {
    StringRef ABIName = tools::riscv::getRISCVABI(Args, Triple);
    addPathIfExists(D, SysRoot + "/" + OSLibDir + "/" + ABIName, Paths);
    addPathIfExists(D, SysRoot + "/usr/" + OSLibDir + "/" + ABIName, Paths);
  }

  Generic_GCC::AddMultiarchPaths(D, SysRoot, OSLibDir, Paths);

  // Similar to the logic for GCC above, if we are currently running Clang
  // inside of the requested system root, add its parent library path to those
  // searched.
  // FIXME: It's not clear whether we should use the driver's installed
  // directory ('Dir' below) or the ResourceDir.
  if (StringRef(D.Dir).startswith(SysRoot)) {
    // Even if OSLibDir != "lib", this is needed for Clang in the build
    // directory (not installed) to find libc++.
    addPathIfExists(D, D.Dir + "/../lib", Paths);
    if (OSLibDir != "lib")
      addPathIfExists(D, D.Dir + "/../" + OSLibDir, Paths);
  }

  addPathIfExists(D, SysRoot + "/lib", Paths);
  addPathIfExists(D, SysRoot + "/usr/lib", Paths);
}

ToolChain::RuntimeLibType Linux::GetDefaultRuntimeLibType() const {
  if (getTriple().isAndroid())
    return ToolChain::RLT_CompilerRT;
  return Generic_ELF::GetDefaultRuntimeLibType();
}

unsigned Linux::GetDefaultDwarfVersion() const {
  if (getTriple().isAndroid())
    return 4;
  return ToolChain::GetDefaultDwarfVersion();
}

ToolChain::CXXStdlibType Linux::GetDefaultCXXStdlibType() const {
  if (getTriple().isAndroid())
    return ToolChain::CST_Libcxx;
  return ToolChain::CST_Libstdcxx;
}

bool Linux::HasNativeLLVMSupport() const { return true; }

Tool *Linux::buildLinker() const { return new tools::gnutools::Linker(*this); }

Tool *Linux::buildStaticLibTool() const {
  return new tools::gnutools::StaticLibTool(*this);
}

Tool *Linux::buildAssembler() const {
  return new tools::gnutools::Assembler(*this);
}

std::string Linux::computeSysRoot() const {
  if (!getDriver().SysRoot.empty())
    return getDriver().SysRoot;

  if (getTriple().isAndroid()) {
    // Android toolchains typically include a sysroot at ../sysroot relative to
    // the clang binary.
    const StringRef ClangDir = getDriver().getInstalledDir();
    std::string AndroidSysRootPath = (ClangDir + "/../sysroot").str();
    if (getVFS().exists(AndroidSysRootPath))
      return AndroidSysRootPath;
  }

  if (!GCCInstallation.isValid() || !getTriple().isMIPS())
    return std::string();

  // Standalone MIPS toolchains use different names for sysroot folder
  // and put it into different places. Here we try to check some known
  // variants.

  const StringRef InstallDir = GCCInstallation.getInstallPath();
  const StringRef TripleStr = GCCInstallation.getTriple().str();
  const Multilib &Multilib = GCCInstallation.getMultilib();

  std::string Path =
      (InstallDir + "/../../../../" + TripleStr + "/libc" + Multilib.osSuffix())
          .str();

  if (getVFS().exists(Path))
    return Path;

  Path = (InstallDir + "/../../../../sysroot" + Multilib.osSuffix()).str();

  if (getVFS().exists(Path))
    return Path;

  return std::string();
}

std::string Linux::getDynamicLinker(const ArgList &Args) const {
  const llvm::Triple::ArchType Arch = getArch();
  const llvm::Triple &Triple = getTriple();

  const Distro Distro(getDriver().getVFS(), Triple);

  if (Triple.isAndroid())
    return Triple.isArch64Bit() ? "/system/bin/linker64" : "/system/bin/linker";

  if (Triple.isMusl()) {
    std::string ArchName;
    bool IsArm = false;

    switch (Arch) {
    case llvm::Triple::arm:
    case llvm::Triple::thumb:
      ArchName = "arm";
      IsArm = true;
      break;
    case llvm::Triple::armeb:
    case llvm::Triple::thumbeb:
      ArchName = "armeb";
      IsArm = true;
      break;
    case llvm::Triple::x86:
      ArchName = "i386";
      break;
    case llvm::Triple::x86_64:
      ArchName = Triple.isX32() ? "x32" : Triple.getArchName().str();
      break;
    default:
      ArchName = Triple.getArchName().str();
    }
    if (IsArm &&
        (Triple.getEnvironment() == llvm::Triple::MuslEABIHF ||
         tools::arm::getARMFloatABI(*this, Args) == tools::arm::FloatABI::Hard))
      ArchName += "hf";
    if (Arch == llvm::Triple::ppc &&
        Triple.getSubArch() == llvm::Triple::PPCSubArch_spe)
      ArchName = "powerpc-sf";

    return "/lib/ld-musl-" + ArchName + ".so.1";
  }

  std::string LibDir;
  std::string Loader;

  switch (Arch) {
  default:
    llvm_unreachable("unsupported architecture");

  case llvm::Triple::aarch64:
    LibDir = "lib";
    Loader = "ld-linux-aarch64.so.1";
    break;
  case llvm::Triple::aarch64_be:
    LibDir = "lib";
    Loader = "ld-linux-aarch64_be.so.1";
    break;
  case llvm::Triple::arm:
  case llvm::Triple::thumb:
  case llvm::Triple::armeb:
  case llvm::Triple::thumbeb: {
    const bool HF =
        Triple.getEnvironment() == llvm::Triple::GNUEABIHF ||
        tools::arm::getARMFloatABI(*this, Args) == tools::arm::FloatABI::Hard;

    LibDir = "lib";
    Loader = HF ? "ld-linux-armhf.so.3" : "ld-linux.so.3";
    break;
  }
  case llvm::Triple::m68k:
    LibDir = "lib";
    Loader = "ld.so.1";
    break;
  case llvm::Triple::mips:
  case llvm::Triple::mipsel:
  case llvm::Triple::mips64:
  case llvm::Triple::mips64el: {
    bool IsNaN2008 = tools::mips::isNaN2008(getDriver(), Args, Triple);

    LibDir = "lib" + tools::mips::getMipsABILibSuffix(Args, Triple);

    if (tools::mips::isUCLibc(Args))
      Loader = IsNaN2008 ? "ld-uClibc-mipsn8.so.0" : "ld-uClibc.so.0";
    else if (!Triple.hasEnvironment() &&
             Triple.getVendor() == llvm::Triple::VendorType::MipsTechnologies)
      Loader =
          Triple.isLittleEndian() ? "ld-musl-mipsel.so.1" : "ld-musl-mips.so.1";
    else
      Loader = IsNaN2008 ? "ld-linux-mipsn8.so.1" : "ld.so.1";

    break;
  }
  case llvm::Triple::ppc:
    LibDir = "lib";
    Loader = "ld.so.1";
    break;
  case llvm::Triple::ppcle:
    LibDir = "lib";
    Loader = "ld.so.1";
    break;
  case llvm::Triple::ppc64:
    LibDir = "lib64";
    Loader =
        (tools::ppc::hasPPCAbiArg(Args, "elfv2")) ? "ld64.so.2" : "ld64.so.1";
    break;
  case llvm::Triple::ppc64le:
    LibDir = "lib64";
    Loader =
        (tools::ppc::hasPPCAbiArg(Args, "elfv1")) ? "ld64.so.1" : "ld64.so.2";
    break;
  case llvm::Triple::riscv32: {
    StringRef ABIName = tools::riscv::getRISCVABI(Args, Triple);
    LibDir = "lib";
    Loader = ("ld-linux-riscv32-" + ABIName + ".so.1").str();
    break;
  }
  case llvm::Triple::riscv64: {
    StringRef ABIName = tools::riscv::getRISCVABI(Args, Triple);
    LibDir = "lib";
    Loader = ("ld-linux-riscv64-" + ABIName + ".so.1").str();
    break;
  }
  case llvm::Triple::sparc:
  case llvm::Triple::sparcel:
    LibDir = "lib";
    Loader = "ld-linux.so.2";
    break;
  case llvm::Triple::sparcv9:
    LibDir = "lib64";
    Loader = "ld-linux.so.2";
    break;
  case llvm::Triple::systemz:
    LibDir = "lib";
    Loader = "ld64.so.1";
    break;
  case llvm::Triple::x86:
    LibDir = "lib";
    Loader = "ld-linux.so.2";
    break;
  case llvm::Triple::x86_64: {
    bool X32 = Triple.isX32();

    LibDir = X32 ? "libx32" : "lib64";
    Loader = X32 ? "ld-linux-x32.so.2" : "ld-linux-x86-64.so.2";
    break;
  }
  case llvm::Triple::ve:
    return "/opt/nec/ve/lib/ld-linux-ve.so.1";
  }

  if (Distro == Distro::Exherbo &&
      (Triple.getVendor() == llvm::Triple::UnknownVendor ||
       Triple.getVendor() == llvm::Triple::PC))
    return "/usr/" + Triple.str() + "/lib/" + Loader;
  return "/" + LibDir + "/" + Loader;
}

void Linux::AddClangSystemIncludeArgs(const ArgList &DriverArgs,
                                      ArgStringList &CC1Args) const {
  const Driver &D = getDriver();
  std::string SysRoot = computeSysRoot();

  if (DriverArgs.hasArg(clang::driver::options::OPT_nostdinc))
    return;

  // Add 'include' in the resource directory, which is similar to
  // GCC_INCLUDE_DIR (private headers) in GCC. Note: the include directory
  // contains some files conflicting with system /usr/include. musl systems
  // prefer the /usr/include copies which are more relevant.
  SmallString<128> ResourceDirInclude(D.ResourceDir);
  llvm::sys::path::append(ResourceDirInclude, "include");
  if (!DriverArgs.hasArg(options::OPT_nobuiltininc) &&
      (!getTriple().isMusl() || DriverArgs.hasArg(options::OPT_nostdlibinc)))
    addSystemInclude(DriverArgs, CC1Args, ResourceDirInclude);

  if (DriverArgs.hasArg(options::OPT_nostdlibinc))
    return;

  // LOCAL_INCLUDE_DIR
  addSystemInclude(DriverArgs, CC1Args, SysRoot + "/usr/local/include");
  // TOOL_INCLUDE_DIR
  AddMultilibIncludeArgs(DriverArgs, CC1Args);

  // Check for configure-time C include directories.
  StringRef CIncludeDirs(C_INCLUDE_DIRS);
  if (CIncludeDirs != "") {
    SmallVector<StringRef, 5> dirs;
    CIncludeDirs.split(dirs, ":");
    for (StringRef dir : dirs) {
      StringRef Prefix =
          llvm::sys::path::is_absolute(dir) ? "" : StringRef(SysRoot);
      addExternCSystemInclude(DriverArgs, CC1Args, Prefix + dir);
    }
    return;
  }

  // On systems using multiarch and Android, add /usr/include/$triple before
  // /usr/include.
  std::string MultiarchIncludeDir = getMultiarchTriple(D, getTriple(), SysRoot);
  if (!MultiarchIncludeDir.empty() &&
      D.getVFS().exists(SysRoot + "/usr/include/" + MultiarchIncludeDir))
    addExternCSystemInclude(DriverArgs, CC1Args,
                            SysRoot + "/usr/include/" + MultiarchIncludeDir);

  if (getTriple().getOS() == llvm::Triple::RTEMS)
    return;

  // Add an include of '/include' directly. This isn't provided by default by
  // system GCCs, but is often used with cross-compiling GCCs, and harmless to
  // add even when Clang is acting as-if it were a system compiler.
  addExternCSystemInclude(DriverArgs, CC1Args, SysRoot + "/include");

  addExternCSystemInclude(DriverArgs, CC1Args, SysRoot + "/usr/include");

  if (!DriverArgs.hasArg(options::OPT_nobuiltininc) && getTriple().isMusl())
    addSystemInclude(DriverArgs, CC1Args, ResourceDirInclude);
}

void Linux::addLibStdCxxIncludePaths(const llvm::opt::ArgList &DriverArgs,
                                     llvm::opt::ArgStringList &CC1Args) const {
  // We need a detected GCC installation on Linux to provide libstdc++'s
  // headers in odd Linuxish places.
  if (!GCCInstallation.isValid())
    return;

  // Detect Debian g++-multiarch-incdir.diff.
  StringRef TripleStr = GCCInstallation.getTriple().str();
  StringRef DebianMultiarch =
      GCCInstallation.getTriple().getArch() == llvm::Triple::x86
          ? "i386-linux-gnu"
          : TripleStr;

  // Try generic GCC detection first.
  if (Generic_GCC::addGCCLibStdCxxIncludePaths(DriverArgs, CC1Args,
                                               DebianMultiarch))
    return;

  StringRef LibDir = GCCInstallation.getParentLibPath();
  const Multilib &Multilib = GCCInstallation.getMultilib();
  const GCCVersion &Version = GCCInstallation.getVersion();

  const std::string LibStdCXXIncludePathCandidates[] = {
      // Android standalone toolchain has C++ headers in yet another place.
      LibDir.str() + "/../" + TripleStr.str() + "/include/c++/" + Version.Text,
      // Freescale SDK C++ headers are directly in <sysroot>/usr/include/c++,
      // without a subdirectory corresponding to the gcc version.
      LibDir.str() + "/../include/c++",
      // Cray's gcc installation puts headers under "g++" without a
      // version suffix.
      LibDir.str() + "/../include/g++",
  };

  for (const auto &IncludePath : LibStdCXXIncludePathCandidates) {
    if (addLibStdCXXIncludePaths(IncludePath, TripleStr,
                                 Multilib.includeSuffix(), DriverArgs, CC1Args))
      break;
  }
}

void Linux::AddCudaIncludeArgs(const ArgList &DriverArgs,
                               ArgStringList &CC1Args) const {
  CudaInstallation.AddCudaIncludeArgs(DriverArgs, CC1Args);
}

void Linux::AddHIPIncludeArgs(const ArgList &DriverArgs,
                              ArgStringList &CC1Args) const {
  RocmInstallation.AddHIPIncludeArgs(DriverArgs, CC1Args);
}

void Linux::AddIAMCUIncludeArgs(const ArgList &DriverArgs,
                                ArgStringList &CC1Args) const {
  if (GCCInstallation.isValid()) {
    CC1Args.push_back("-isystem");
    CC1Args.push_back(DriverArgs.MakeArgString(
        GCCInstallation.getParentLibPath() + "/../" +
        GCCInstallation.getTriple().str() + "/include"));
  }
}

bool Linux::isPIEDefault(const llvm::opt::ArgList &Args) const {
  return CLANG_DEFAULT_PIE_ON_LINUX || getTriple().isAndroid() ||
         getTriple().isMusl() || getSanitizerArgs(Args).requiresPIE();
}

bool Linux::IsAArch64OutlineAtomicsDefault(const ArgList &Args) const {
  // Outline atomics for AArch64 are supported by compiler-rt
  // and libgcc since 9.3.1
  assert(getTriple().isAArch64() && "expected AArch64 target!");
  ToolChain::RuntimeLibType RtLib = GetRuntimeLibType(Args);
  if (RtLib == ToolChain::RLT_CompilerRT)
    return true;
  assert(RtLib == ToolChain::RLT_Libgcc && "unexpected runtime library type!");
  if (GCCInstallation.getVersion().isOlderThan(9, 3, 1))
    return false;
  return true;
}

bool Linux::IsMathErrnoDefault() const {
  if (getTriple().isAndroid() || getTriple().isMusl())
    return false;
  return Generic_ELF::IsMathErrnoDefault();
}

SanitizerMask Linux::getSupportedSanitizers() const {
  const bool IsX86 = getTriple().getArch() == llvm::Triple::x86;
  const bool IsX86_64 = getTriple().getArch() == llvm::Triple::x86_64;
  const bool IsMIPS = getTriple().isMIPS32();
  const bool IsMIPS64 = getTriple().isMIPS64();
  const bool IsPowerPC64 = getTriple().getArch() == llvm::Triple::ppc64 ||
                           getTriple().getArch() == llvm::Triple::ppc64le;
  const bool IsAArch64 = getTriple().getArch() == llvm::Triple::aarch64 ||
                         getTriple().getArch() == llvm::Triple::aarch64_be;
  const bool IsArmArch = getTriple().getArch() == llvm::Triple::arm ||
                         getTriple().getArch() == llvm::Triple::thumb ||
                         getTriple().getArch() == llvm::Triple::armeb ||
                         getTriple().getArch() == llvm::Triple::thumbeb;
  const bool IsRISCV64 = getTriple().getArch() == llvm::Triple::riscv64;
  const bool IsSystemZ = getTriple().getArch() == llvm::Triple::systemz;
  const bool IsHexagon = getTriple().getArch() == llvm::Triple::hexagon;
  SanitizerMask Res = ToolChain::getSupportedSanitizers();
  Res |= SanitizerKind::Address;
  Res |= SanitizerKind::PointerCompare;
  Res |= SanitizerKind::PointerSubtract;
  Res |= SanitizerKind::Fuzzer;
  Res |= SanitizerKind::FuzzerNoLink;
  Res |= SanitizerKind::KernelAddress;
  Res |= SanitizerKind::Memory;
  Res |= SanitizerKind::Vptr;
  Res |= SanitizerKind::SafeStack;
  if (IsX86_64 || IsMIPS64 || IsAArch64)
    Res |= SanitizerKind::DataFlow;
  if (IsX86_64 || IsMIPS64 || IsAArch64 || IsX86 || IsArmArch || IsPowerPC64 ||
      IsRISCV64 || IsSystemZ || IsHexagon)
    Res |= SanitizerKind::Leak;
  if (IsX86_64 || IsMIPS64 || IsAArch64 || IsPowerPC64 || IsSystemZ)
    Res |= SanitizerKind::Thread;
  if (IsX86_64)
    Res |= SanitizerKind::KernelMemory;
  if (IsX86 || IsX86_64)
    Res |= SanitizerKind::Function;
  if (IsX86_64 || IsMIPS64 || IsAArch64 || IsX86 || IsMIPS || IsArmArch ||
      IsPowerPC64 || IsHexagon)
    Res |= SanitizerKind::Scudo;
  if (IsX86_64 || IsAArch64) {
    Res |= SanitizerKind::HWAddress;
    Res |= SanitizerKind::KernelHWAddress;
  }
  return Res;
}

void Linux::addProfileRTLibs(const llvm::opt::ArgList &Args,
                             llvm::opt::ArgStringList &CmdArgs) const {
  // Add linker option -u__llvm_profile_runtime to cause runtime
  // initialization module to be linked in.
  if (needsProfileRT(Args))
    CmdArgs.push_back(Args.MakeArgString(
        Twine("-u", llvm::getInstrProfRuntimeHookVarName())));
  ToolChain::addProfileRTLibs(Args, CmdArgs);
}

llvm::DenormalMode
Linux::getDefaultDenormalModeForType(const llvm::opt::ArgList &DriverArgs,
                                     const JobAction &JA,
                                     const llvm::fltSemantics *FPType) const {
  switch (getTriple().getArch()) {
  case llvm::Triple::x86:
  case llvm::Triple::x86_64: {
    std::string Unused;
    // DAZ and FTZ are turned on in crtfastmath.o
    if (!DriverArgs.hasArg(options::OPT_nostdlib, options::OPT_nostartfiles) &&
        isFastMathRuntimeAvailable(DriverArgs, Unused))
      return llvm::DenormalMode::getPreserveSign();
    return llvm::DenormalMode::getIEEE();
  }
  default:
    return llvm::DenormalMode::getIEEE();
  }
}

void Linux::addExtraOpts(llvm::opt::ArgStringList &CmdArgs) const {
  for (const auto &Opt : ExtraOpts)
    CmdArgs.push_back(Opt.c_str());
}
