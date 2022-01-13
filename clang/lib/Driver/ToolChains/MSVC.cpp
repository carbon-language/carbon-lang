//===-- MSVC.cpp - MSVC ToolChain Implementations -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MSVC.h"
#include "CommonArgs.h"
#include "Darwin.h"
#include "clang/Basic/CharInfo.h"
#include "clang/Basic/Version.h"
#include "clang/Config/config.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/Options.h"
#include "clang/Driver/SanitizerArgs.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/VirtualFileSystem.h"
#include <cstdio>

#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #define NOGDI
  #ifndef NOMINMAX
    #define NOMINMAX
  #endif
  #include <windows.h>
#endif

#ifdef _MSC_VER
// Don't support SetupApi on MinGW.
#define USE_MSVC_SETUP_API

// Make sure this comes before MSVCSetupApi.h
#include <comdef.h>

#include "MSVCSetupApi.h"
#include "llvm/Support/COM.h"
_COM_SMARTPTR_TYPEDEF(ISetupConfiguration, __uuidof(ISetupConfiguration));
_COM_SMARTPTR_TYPEDEF(ISetupConfiguration2, __uuidof(ISetupConfiguration2));
_COM_SMARTPTR_TYPEDEF(ISetupHelper, __uuidof(ISetupHelper));
_COM_SMARTPTR_TYPEDEF(IEnumSetupInstances, __uuidof(IEnumSetupInstances));
_COM_SMARTPTR_TYPEDEF(ISetupInstance, __uuidof(ISetupInstance));
_COM_SMARTPTR_TYPEDEF(ISetupInstance2, __uuidof(ISetupInstance2));
#endif

using namespace clang::driver;
using namespace clang::driver::toolchains;
using namespace clang::driver::tools;
using namespace clang;
using namespace llvm::opt;

// Windows SDKs and VC Toolchains group their contents into subdirectories based
// on the target architecture. This function converts an llvm::Triple::ArchType
// to the corresponding subdirectory name.
static const char *llvmArchToWindowsSDKArch(llvm::Triple::ArchType Arch) {
  using ArchType = llvm::Triple::ArchType;
  switch (Arch) {
  case ArchType::x86:
    return "x86";
  case ArchType::x86_64:
    return "x64";
  case ArchType::arm:
    return "arm";
  case ArchType::aarch64:
    return "arm64";
  default:
    return "";
  }
}

// Similar to the above function, but for Visual Studios before VS2017.
static const char *llvmArchToLegacyVCArch(llvm::Triple::ArchType Arch) {
  using ArchType = llvm::Triple::ArchType;
  switch (Arch) {
  case ArchType::x86:
    // x86 is default in legacy VC toolchains.
    // e.g. x86 libs are directly in /lib as opposed to /lib/x86.
    return "";
  case ArchType::x86_64:
    return "amd64";
  case ArchType::arm:
    return "arm";
  case ArchType::aarch64:
    return "arm64";
  default:
    return "";
  }
}

// Similar to the above function, but for DevDiv internal builds.
static const char *llvmArchToDevDivInternalArch(llvm::Triple::ArchType Arch) {
  using ArchType = llvm::Triple::ArchType;
  switch (Arch) {
  case ArchType::x86:
    return "i386";
  case ArchType::x86_64:
    return "amd64";
  case ArchType::arm:
    return "arm";
  case ArchType::aarch64:
    return "arm64";
  default:
    return "";
  }
}

static bool canExecute(llvm::vfs::FileSystem &VFS, StringRef Path) {
  auto Status = VFS.status(Path);
  if (!Status)
    return false;
  return (Status->getPermissions() & llvm::sys::fs::perms::all_exe) != 0;
}

// Defined below.
// Forward declare this so there aren't too many things above the constructor.
static bool getSystemRegistryString(const char *keyPath, const char *valueName,
                                    std::string &value, std::string *phValue);

static std::string getHighestNumericTupleInDirectory(llvm::vfs::FileSystem &VFS,
                                                     StringRef Directory) {
  std::string Highest;
  llvm::VersionTuple HighestTuple;

  std::error_code EC;
  for (llvm::vfs::directory_iterator DirIt = VFS.dir_begin(Directory, EC),
                                     DirEnd;
       !EC && DirIt != DirEnd; DirIt.increment(EC)) {
    auto Status = VFS.status(DirIt->path());
    if (!Status || !Status->isDirectory())
      continue;
    StringRef CandidateName = llvm::sys::path::filename(DirIt->path());
    llvm::VersionTuple Tuple;
    if (Tuple.tryParse(CandidateName)) // tryParse() returns true on error.
      continue;
    if (Tuple > HighestTuple) {
      HighestTuple = Tuple;
      Highest = CandidateName.str();
    }
  }

  return Highest;
}

// Check command line arguments to try and find a toolchain.
static bool
findVCToolChainViaCommandLine(llvm::vfs::FileSystem &VFS, const ArgList &Args,
                              std::string &Path,
                              MSVCToolChain::ToolsetLayout &VSLayout) {
  // Don't validate the input; trust the value supplied by the user.
  // The primary motivation is to prevent unnecessary file and registry access.
  if (Arg *A = Args.getLastArg(options::OPT__SLASH_vctoolsdir,
                               options::OPT__SLASH_winsysroot)) {
    if (A->getOption().getID() == options::OPT__SLASH_winsysroot) {
      llvm::SmallString<128> ToolsPath(A->getValue());
      llvm::sys::path::append(ToolsPath, "VC", "Tools", "MSVC");
      std::string VCToolsVersion;
      if (Arg *A = Args.getLastArg(options::OPT__SLASH_vctoolsversion))
        VCToolsVersion = A->getValue();
      else
        VCToolsVersion = getHighestNumericTupleInDirectory(VFS, ToolsPath);
      llvm::sys::path::append(ToolsPath, VCToolsVersion);
      Path = std::string(ToolsPath.str());
    } else {
      Path = A->getValue();
    }
    VSLayout = MSVCToolChain::ToolsetLayout::VS2017OrNewer;
    return true;
  }
  return false;
}

// Check various environment variables to try and find a toolchain.
static bool
findVCToolChainViaEnvironment(llvm::vfs::FileSystem &VFS, std::string &Path,
                              MSVCToolChain::ToolsetLayout &VSLayout) {
  // These variables are typically set by vcvarsall.bat
  // when launching a developer command prompt.
  if (llvm::Optional<std::string> VCToolsInstallDir =
          llvm::sys::Process::GetEnv("VCToolsInstallDir")) {
    // This is only set by newer Visual Studios, and it leads straight to
    // the toolchain directory.
    Path = std::move(*VCToolsInstallDir);
    VSLayout = MSVCToolChain::ToolsetLayout::VS2017OrNewer;
    return true;
  }
  if (llvm::Optional<std::string> VCInstallDir =
          llvm::sys::Process::GetEnv("VCINSTALLDIR")) {
    // If the previous variable isn't set but this one is, then we've found
    // an older Visual Studio. This variable is set by newer Visual Studios too,
    // so this check has to appear second.
    // In older Visual Studios, the VC directory is the toolchain.
    Path = std::move(*VCInstallDir);
    VSLayout = MSVCToolChain::ToolsetLayout::OlderVS;
    return true;
  }

  // We couldn't find any VC environment variables. Let's walk through PATH and
  // see if it leads us to a VC toolchain bin directory. If it does, pick the
  // first one that we find.
  if (llvm::Optional<std::string> PathEnv =
          llvm::sys::Process::GetEnv("PATH")) {
    llvm::SmallVector<llvm::StringRef, 8> PathEntries;
    llvm::StringRef(*PathEnv).split(PathEntries, llvm::sys::EnvPathSeparator);
    for (llvm::StringRef PathEntry : PathEntries) {
      if (PathEntry.empty())
        continue;

      llvm::SmallString<256> ExeTestPath;

      // If cl.exe doesn't exist, then this definitely isn't a VC toolchain.
      ExeTestPath = PathEntry;
      llvm::sys::path::append(ExeTestPath, "cl.exe");
      if (!VFS.exists(ExeTestPath))
        continue;

      // cl.exe existing isn't a conclusive test for a VC toolchain; clang also
      // has a cl.exe. So let's check for link.exe too.
      ExeTestPath = PathEntry;
      llvm::sys::path::append(ExeTestPath, "link.exe");
      if (!VFS.exists(ExeTestPath))
        continue;

      // whatever/VC/bin --> old toolchain, VC dir is toolchain dir.
      llvm::StringRef TestPath = PathEntry;
      bool IsBin =
          llvm::sys::path::filename(TestPath).equals_insensitive("bin");
      if (!IsBin) {
        // Strip any architecture subdir like "amd64".
        TestPath = llvm::sys::path::parent_path(TestPath);
        IsBin = llvm::sys::path::filename(TestPath).equals_insensitive("bin");
      }
      if (IsBin) {
        llvm::StringRef ParentPath = llvm::sys::path::parent_path(TestPath);
        llvm::StringRef ParentFilename = llvm::sys::path::filename(ParentPath);
        if (ParentFilename.equals_insensitive("VC")) {
          Path = std::string(ParentPath);
          VSLayout = MSVCToolChain::ToolsetLayout::OlderVS;
          return true;
        }
        if (ParentFilename.equals_insensitive("x86ret") ||
            ParentFilename.equals_insensitive("x86chk") ||
            ParentFilename.equals_insensitive("amd64ret") ||
            ParentFilename.equals_insensitive("amd64chk")) {
          Path = std::string(ParentPath);
          VSLayout = MSVCToolChain::ToolsetLayout::DevDivInternal;
          return true;
        }

      } else {
        // This could be a new (>=VS2017) toolchain. If it is, we should find
        // path components with these prefixes when walking backwards through
        // the path.
        // Note: empty strings match anything.
        llvm::StringRef ExpectedPrefixes[] = {"",     "Host",  "bin", "",
                                              "MSVC", "Tools", "VC"};

        auto It = llvm::sys::path::rbegin(PathEntry);
        auto End = llvm::sys::path::rend(PathEntry);
        for (llvm::StringRef Prefix : ExpectedPrefixes) {
          if (It == End)
            goto NotAToolChain;
          if (!It->startswith_insensitive(Prefix))
            goto NotAToolChain;
          ++It;
        }

        // We've found a new toolchain!
        // Back up 3 times (/bin/Host/arch) to get the root path.
        llvm::StringRef ToolChainPath(PathEntry);
        for (int i = 0; i < 3; ++i)
          ToolChainPath = llvm::sys::path::parent_path(ToolChainPath);

        Path = std::string(ToolChainPath);
        VSLayout = MSVCToolChain::ToolsetLayout::VS2017OrNewer;
        return true;
      }

    NotAToolChain:
      continue;
    }
  }
  return false;
}

// Query the Setup Config server for installs, then pick the newest version
// and find its default VC toolchain.
// This is the preferred way to discover new Visual Studios, as they're no
// longer listed in the registry.
static bool
findVCToolChainViaSetupConfig(llvm::vfs::FileSystem &VFS, std::string &Path,
                              MSVCToolChain::ToolsetLayout &VSLayout) {
#if !defined(USE_MSVC_SETUP_API)
  return false;
#else
  // FIXME: This really should be done once in the top-level program's main
  // function, as it may have already been initialized with a different
  // threading model otherwise.
  llvm::sys::InitializeCOMRAII COM(llvm::sys::COMThreadingMode::SingleThreaded);
  HRESULT HR;

  // _com_ptr_t will throw a _com_error if a COM calls fail.
  // The LLVM coding standards forbid exception handling, so we'll have to
  // stop them from being thrown in the first place.
  // The destructor will put the regular error handler back when we leave
  // this scope.
  struct SuppressCOMErrorsRAII {
    static void __stdcall handler(HRESULT hr, IErrorInfo *perrinfo) {}

    SuppressCOMErrorsRAII() { _set_com_error_handler(handler); }

    ~SuppressCOMErrorsRAII() { _set_com_error_handler(_com_raise_error); }

  } COMErrorSuppressor;

  ISetupConfigurationPtr Query;
  HR = Query.CreateInstance(__uuidof(SetupConfiguration));
  if (FAILED(HR))
    return false;

  IEnumSetupInstancesPtr EnumInstances;
  HR = ISetupConfiguration2Ptr(Query)->EnumAllInstances(&EnumInstances);
  if (FAILED(HR))
    return false;

  ISetupInstancePtr Instance;
  HR = EnumInstances->Next(1, &Instance, nullptr);
  if (HR != S_OK)
    return false;

  ISetupInstancePtr NewestInstance;
  Optional<uint64_t> NewestVersionNum;
  do {
    bstr_t VersionString;
    uint64_t VersionNum;
    HR = Instance->GetInstallationVersion(VersionString.GetAddress());
    if (FAILED(HR))
      continue;
    HR = ISetupHelperPtr(Query)->ParseVersion(VersionString, &VersionNum);
    if (FAILED(HR))
      continue;
    if (!NewestVersionNum || (VersionNum > NewestVersionNum)) {
      NewestInstance = Instance;
      NewestVersionNum = VersionNum;
    }
  } while ((HR = EnumInstances->Next(1, &Instance, nullptr)) == S_OK);

  if (!NewestInstance)
    return false;

  bstr_t VCPathWide;
  HR = NewestInstance->ResolvePath(L"VC", VCPathWide.GetAddress());
  if (FAILED(HR))
    return false;

  std::string VCRootPath;
  llvm::convertWideToUTF8(std::wstring(VCPathWide), VCRootPath);

  llvm::SmallString<256> ToolsVersionFilePath(VCRootPath);
  llvm::sys::path::append(ToolsVersionFilePath, "Auxiliary", "Build",
                          "Microsoft.VCToolsVersion.default.txt");

  auto ToolsVersionFile = llvm::MemoryBuffer::getFile(ToolsVersionFilePath);
  if (!ToolsVersionFile)
    return false;

  llvm::SmallString<256> ToolchainPath(VCRootPath);
  llvm::sys::path::append(ToolchainPath, "Tools", "MSVC",
                          ToolsVersionFile->get()->getBuffer().rtrim());
  auto Status = VFS.status(ToolchainPath);
  if (!Status || !Status->isDirectory())
    return false;

  Path = std::string(ToolchainPath.str());
  VSLayout = MSVCToolChain::ToolsetLayout::VS2017OrNewer;
  return true;
#endif
}

// Look in the registry for Visual Studio installs, and use that to get
// a toolchain path. VS2017 and newer don't get added to the registry.
// So if we find something here, we know that it's an older version.
static bool findVCToolChainViaRegistry(std::string &Path,
                                       MSVCToolChain::ToolsetLayout &VSLayout) {
  std::string VSInstallPath;
  if (getSystemRegistryString(R"(SOFTWARE\Microsoft\VisualStudio\$VERSION)",
                              "InstallDir", VSInstallPath, nullptr) ||
      getSystemRegistryString(R"(SOFTWARE\Microsoft\VCExpress\$VERSION)",
                              "InstallDir", VSInstallPath, nullptr)) {
    if (!VSInstallPath.empty()) {
      llvm::SmallString<256> VCPath(llvm::StringRef(
          VSInstallPath.c_str(), VSInstallPath.find(R"(\Common7\IDE)")));
      llvm::sys::path::append(VCPath, "VC");

      Path = std::string(VCPath.str());
      VSLayout = MSVCToolChain::ToolsetLayout::OlderVS;
      return true;
    }
  }
  return false;
}

// Try to find Exe from a Visual Studio distribution.  This first tries to find
// an installed copy of Visual Studio and, failing that, looks in the PATH,
// making sure that whatever executable that's found is not a same-named exe
// from clang itself to prevent clang from falling back to itself.
static std::string FindVisualStudioExecutable(const ToolChain &TC,
                                              const char *Exe) {
  const auto &MSVC = static_cast<const toolchains::MSVCToolChain &>(TC);
  SmallString<128> FilePath(MSVC.getSubDirectoryPath(
      toolchains::MSVCToolChain::SubDirectoryType::Bin));
  llvm::sys::path::append(FilePath, Exe);
  return std::string(canExecute(TC.getVFS(), FilePath) ? FilePath.str() : Exe);
}

void visualstudio::Linker::ConstructJob(Compilation &C, const JobAction &JA,
                                        const InputInfo &Output,
                                        const InputInfoList &Inputs,
                                        const ArgList &Args,
                                        const char *LinkingOutput) const {
  ArgStringList CmdArgs;

  auto &TC = static_cast<const toolchains::MSVCToolChain &>(getToolChain());

  assert((Output.isFilename() || Output.isNothing()) && "invalid output");
  if (Output.isFilename())
    CmdArgs.push_back(
        Args.MakeArgString(std::string("-out:") + Output.getFilename()));

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nostartfiles) &&
      !C.getDriver().IsCLMode()) {
    CmdArgs.push_back("-defaultlib:libcmt");
    CmdArgs.push_back("-defaultlib:oldnames");
  }

  // If the VC environment hasn't been configured (perhaps because the user
  // did not run vcvarsall), try to build a consistent link environment.  If
  // the environment variable is set however, assume the user knows what
  // they're doing. If the user passes /vctoolsdir or /winsdkdir, trust that
  // over env vars.
  if (const Arg *A = Args.getLastArg(options::OPT__SLASH_diasdkdir,
                                     options::OPT__SLASH_winsysroot)) {
    // cl.exe doesn't find the DIA SDK automatically, so this too requires
    // explicit flags and doesn't automatically look in "DIA SDK" relative
    // to the path we found for VCToolChainPath.
    llvm::SmallString<128> DIAPath(A->getValue());
    if (A->getOption().getID() == options::OPT__SLASH_winsysroot)
      llvm::sys::path::append(DIAPath, "DIA SDK");

    // The DIA SDK always uses the legacy vc arch, even in new MSVC versions.
    llvm::sys::path::append(DIAPath, "lib",
                            llvmArchToLegacyVCArch(TC.getArch()));
    CmdArgs.push_back(Args.MakeArgString(Twine("-libpath:") + DIAPath));
  }
  if (!llvm::sys::Process::GetEnv("LIB") ||
      Args.getLastArg(options::OPT__SLASH_vctoolsdir,
                      options::OPT__SLASH_winsysroot)) {
    CmdArgs.push_back(Args.MakeArgString(
        Twine("-libpath:") +
        TC.getSubDirectoryPath(
            toolchains::MSVCToolChain::SubDirectoryType::Lib)));
    CmdArgs.push_back(Args.MakeArgString(
        Twine("-libpath:") +
        TC.getSubDirectoryPath(toolchains::MSVCToolChain::SubDirectoryType::Lib,
                               "atlmfc")));
  }
  if (!llvm::sys::Process::GetEnv("LIB") ||
      Args.getLastArg(options::OPT__SLASH_winsdkdir,
                      options::OPT__SLASH_winsysroot)) {
    if (TC.useUniversalCRT()) {
      std::string UniversalCRTLibPath;
      if (TC.getUniversalCRTLibraryPath(Args, UniversalCRTLibPath))
        CmdArgs.push_back(
            Args.MakeArgString(Twine("-libpath:") + UniversalCRTLibPath));
    }
    std::string WindowsSdkLibPath;
    if (TC.getWindowsSDKLibraryPath(Args, WindowsSdkLibPath))
      CmdArgs.push_back(
          Args.MakeArgString(std::string("-libpath:") + WindowsSdkLibPath));
  }

  // Add the compiler-rt library directories to libpath if they exist to help
  // the linker find the various sanitizer, builtin, and profiling runtimes.
  for (const auto &LibPath : TC.getLibraryPaths()) {
    if (TC.getVFS().exists(LibPath))
      CmdArgs.push_back(Args.MakeArgString("-libpath:" + LibPath));
  }
  auto CRTPath = TC.getCompilerRTPath();
  if (TC.getVFS().exists(CRTPath))
    CmdArgs.push_back(Args.MakeArgString("-libpath:" + CRTPath));

  if (!C.getDriver().IsCLMode() && Args.hasArg(options::OPT_L))
    for (const auto &LibPath : Args.getAllArgValues(options::OPT_L))
      CmdArgs.push_back(Args.MakeArgString("-libpath:" + LibPath));

  CmdArgs.push_back("-nologo");

  if (Args.hasArg(options::OPT_g_Group, options::OPT__SLASH_Z7))
    CmdArgs.push_back("-debug");

  // Pass on /Brepro if it was passed to the compiler.
  // Note that /Brepro maps to -mno-incremental-linker-compatible.
  bool DefaultIncrementalLinkerCompatible =
      C.getDefaultToolChain().getTriple().isWindowsMSVCEnvironment();
  if (!Args.hasFlag(options::OPT_mincremental_linker_compatible,
                    options::OPT_mno_incremental_linker_compatible,
                    DefaultIncrementalLinkerCompatible))
    CmdArgs.push_back("-Brepro");

  bool DLL = Args.hasArg(options::OPT__SLASH_LD, options::OPT__SLASH_LDd,
                         options::OPT_shared);
  if (DLL) {
    CmdArgs.push_back(Args.MakeArgString("-dll"));

    SmallString<128> ImplibName(Output.getFilename());
    llvm::sys::path::replace_extension(ImplibName, "lib");
    CmdArgs.push_back(Args.MakeArgString(std::string("-implib:") + ImplibName));
  }

  if (TC.getSanitizerArgs(Args).needsFuzzer()) {
    if (!Args.hasArg(options::OPT_shared))
      CmdArgs.push_back(
          Args.MakeArgString(std::string("-wholearchive:") +
                             TC.getCompilerRTArgString(Args, "fuzzer")));
    CmdArgs.push_back(Args.MakeArgString("-debug"));
    // Prevent the linker from padding sections we use for instrumentation
    // arrays.
    CmdArgs.push_back(Args.MakeArgString("-incremental:no"));
  }

  if (TC.getSanitizerArgs(Args).needsAsanRt()) {
    CmdArgs.push_back(Args.MakeArgString("-debug"));
    CmdArgs.push_back(Args.MakeArgString("-incremental:no"));
    if (TC.getSanitizerArgs(Args).needsSharedRt() ||
        Args.hasArg(options::OPT__SLASH_MD, options::OPT__SLASH_MDd)) {
      for (const auto &Lib : {"asan_dynamic", "asan_dynamic_runtime_thunk"})
        CmdArgs.push_back(TC.getCompilerRTArgString(Args, Lib));
      // Make sure the dynamic runtime thunk is not optimized out at link time
      // to ensure proper SEH handling.
      CmdArgs.push_back(Args.MakeArgString(
          TC.getArch() == llvm::Triple::x86
              ? "-include:___asan_seh_interceptor"
              : "-include:__asan_seh_interceptor"));
      // Make sure the linker consider all object files from the dynamic runtime
      // thunk.
      CmdArgs.push_back(Args.MakeArgString(std::string("-wholearchive:") +
          TC.getCompilerRT(Args, "asan_dynamic_runtime_thunk")));
    } else if (DLL) {
      CmdArgs.push_back(TC.getCompilerRTArgString(Args, "asan_dll_thunk"));
    } else {
      for (const auto &Lib : {"asan", "asan_cxx"}) {
        CmdArgs.push_back(TC.getCompilerRTArgString(Args, Lib));
        // Make sure the linker consider all object files from the static lib.
        // This is necessary because instrumented dlls need access to all the
        // interface exported by the static lib in the main executable.
        CmdArgs.push_back(Args.MakeArgString(std::string("-wholearchive:") +
            TC.getCompilerRT(Args, Lib)));
      }
    }
  }

  Args.AddAllArgValues(CmdArgs, options::OPT__SLASH_link);

  // Control Flow Guard checks
  if (Arg *A = Args.getLastArg(options::OPT__SLASH_guard)) {
    StringRef GuardArgs = A->getValue();
    if (GuardArgs.equals_insensitive("cf") ||
        GuardArgs.equals_insensitive("cf,nochecks")) {
      // MSVC doesn't yet support the "nochecks" modifier.
      CmdArgs.push_back("-guard:cf");
    } else if (GuardArgs.equals_insensitive("cf-")) {
      CmdArgs.push_back("-guard:cf-");
    } else if (GuardArgs.equals_insensitive("ehcont")) {
      CmdArgs.push_back("-guard:ehcont");
    } else if (GuardArgs.equals_insensitive("ehcont-")) {
      CmdArgs.push_back("-guard:ehcont-");
    }
  }

  if (Args.hasFlag(options::OPT_fopenmp, options::OPT_fopenmp_EQ,
                   options::OPT_fno_openmp, false)) {
    CmdArgs.push_back("-nodefaultlib:vcomp.lib");
    CmdArgs.push_back("-nodefaultlib:vcompd.lib");
    CmdArgs.push_back(Args.MakeArgString(std::string("-libpath:") +
                                         TC.getDriver().Dir + "/../lib"));
    switch (TC.getDriver().getOpenMPRuntime(Args)) {
    case Driver::OMPRT_OMP:
      CmdArgs.push_back("-defaultlib:libomp.lib");
      break;
    case Driver::OMPRT_IOMP5:
      CmdArgs.push_back("-defaultlib:libiomp5md.lib");
      break;
    case Driver::OMPRT_GOMP:
      break;
    case Driver::OMPRT_Unknown:
      // Already diagnosed.
      break;
    }
  }

  // Add compiler-rt lib in case if it was explicitly
  // specified as an argument for --rtlib option.
  if (!Args.hasArg(options::OPT_nostdlib)) {
    AddRunTimeLibs(TC, TC.getDriver(), CmdArgs, Args);
  }

  // Add filenames, libraries, and other linker inputs.
  for (const auto &Input : Inputs) {
    if (Input.isFilename()) {
      CmdArgs.push_back(Input.getFilename());
      continue;
    }

    const Arg &A = Input.getInputArg();

    // Render -l options differently for the MSVC linker.
    if (A.getOption().matches(options::OPT_l)) {
      StringRef Lib = A.getValue();
      const char *LinkLibArg;
      if (Lib.endswith(".lib"))
        LinkLibArg = Args.MakeArgString(Lib);
      else
        LinkLibArg = Args.MakeArgString(Lib + ".lib");
      CmdArgs.push_back(LinkLibArg);
      continue;
    }

    // Otherwise, this is some other kind of linker input option like -Wl, -z,
    // or -L. Render it, even if MSVC doesn't understand it.
    A.renderAsInput(Args, CmdArgs);
  }

  TC.addProfileRTLibs(Args, CmdArgs);

  std::vector<const char *> Environment;

  // We need to special case some linker paths.  In the case of lld, we need to
  // translate 'lld' into 'lld-link', and in the case of the regular msvc
  // linker, we need to use a special search algorithm.
  llvm::SmallString<128> linkPath;
  StringRef Linker
    = Args.getLastArgValue(options::OPT_fuse_ld_EQ, CLANG_DEFAULT_LINKER);
  if (Linker.empty())
    Linker = "link";
  if (Linker.equals_insensitive("lld"))
    Linker = "lld-link";

  if (Linker.equals_insensitive("link")) {
    // If we're using the MSVC linker, it's not sufficient to just use link
    // from the program PATH, because other environments like GnuWin32 install
    // their own link.exe which may come first.
    linkPath = FindVisualStudioExecutable(TC, "link.exe");

    if (!TC.FoundMSVCInstall() && !canExecute(TC.getVFS(), linkPath)) {
      llvm::SmallString<128> ClPath;
      ClPath = TC.GetProgramPath("cl.exe");
      if (canExecute(TC.getVFS(), ClPath)) {
        linkPath = llvm::sys::path::parent_path(ClPath);
        llvm::sys::path::append(linkPath, "link.exe");
        if (!canExecute(TC.getVFS(), linkPath))
          C.getDriver().Diag(clang::diag::warn_drv_msvc_not_found);
      } else {
        C.getDriver().Diag(clang::diag::warn_drv_msvc_not_found);
      }
    }

#ifdef _WIN32
    // When cross-compiling with VS2017 or newer, link.exe expects to have
    // its containing bin directory at the top of PATH, followed by the
    // native target bin directory.
    // e.g. when compiling for x86 on an x64 host, PATH should start with:
    // /bin/Hostx64/x86;/bin/Hostx64/x64
    // This doesn't attempt to handle ToolsetLayout::DevDivInternal.
    if (TC.getIsVS2017OrNewer() &&
        llvm::Triple(llvm::sys::getProcessTriple()).getArch() != TC.getArch()) {
      auto HostArch = llvm::Triple(llvm::sys::getProcessTriple()).getArch();

      auto EnvBlockWide =
          std::unique_ptr<wchar_t[], decltype(&FreeEnvironmentStringsW)>(
              GetEnvironmentStringsW(), FreeEnvironmentStringsW);
      if (!EnvBlockWide)
        goto SkipSettingEnvironment;

      size_t EnvCount = 0;
      size_t EnvBlockLen = 0;
      while (EnvBlockWide[EnvBlockLen] != L'\0') {
        ++EnvCount;
        EnvBlockLen += std::wcslen(&EnvBlockWide[EnvBlockLen]) +
                       1 /*string null-terminator*/;
      }
      ++EnvBlockLen; // add the block null-terminator

      std::string EnvBlock;
      if (!llvm::convertUTF16ToUTF8String(
              llvm::ArrayRef<char>(reinterpret_cast<char *>(EnvBlockWide.get()),
                                   EnvBlockLen * sizeof(EnvBlockWide[0])),
              EnvBlock))
        goto SkipSettingEnvironment;

      Environment.reserve(EnvCount);

      // Now loop over each string in the block and copy them into the
      // environment vector, adjusting the PATH variable as needed when we
      // find it.
      for (const char *Cursor = EnvBlock.data(); *Cursor != '\0';) {
        llvm::StringRef EnvVar(Cursor);
        if (EnvVar.startswith_insensitive("path=")) {
          using SubDirectoryType = toolchains::MSVCToolChain::SubDirectoryType;
          constexpr size_t PrefixLen = 5; // strlen("path=")
          Environment.push_back(Args.MakeArgString(
              EnvVar.substr(0, PrefixLen) +
              TC.getSubDirectoryPath(SubDirectoryType::Bin) +
              llvm::Twine(llvm::sys::EnvPathSeparator) +
              TC.getSubDirectoryPath(SubDirectoryType::Bin, "", HostArch) +
              (EnvVar.size() > PrefixLen
                   ? llvm::Twine(llvm::sys::EnvPathSeparator) +
                         EnvVar.substr(PrefixLen)
                   : "")));
        } else {
          Environment.push_back(Args.MakeArgString(EnvVar));
        }
        Cursor += EnvVar.size() + 1 /*null-terminator*/;
      }
    }
  SkipSettingEnvironment:;
#endif
  } else {
    linkPath = TC.GetProgramPath(Linker.str().c_str());
  }

  auto LinkCmd = std::make_unique<Command>(
      JA, *this, ResponseFileSupport::AtFileUTF16(),
      Args.MakeArgString(linkPath), CmdArgs, Inputs, Output);
  if (!Environment.empty())
    LinkCmd->setEnvironment(Environment);
  C.addCommand(std::move(LinkCmd));
}

MSVCToolChain::MSVCToolChain(const Driver &D, const llvm::Triple &Triple,
                             const ArgList &Args)
    : ToolChain(D, Triple, Args), CudaInstallation(D, Triple, Args),
      RocmInstallation(D, Triple, Args) {
  getProgramPaths().push_back(getDriver().getInstalledDir());
  if (getDriver().getInstalledDir() != getDriver().Dir)
    getProgramPaths().push_back(getDriver().Dir);

  // Check the command line first, that's the user explicitly telling us what to
  // use. Check the environment next, in case we're being invoked from a VS
  // command prompt. Failing that, just try to find the newest Visual Studio
  // version we can and use its default VC toolchain.
  findVCToolChainViaCommandLine(getVFS(), Args, VCToolChainPath, VSLayout) ||
      findVCToolChainViaEnvironment(getVFS(), VCToolChainPath, VSLayout) ||
      findVCToolChainViaSetupConfig(getVFS(), VCToolChainPath, VSLayout) ||
      findVCToolChainViaRegistry(VCToolChainPath, VSLayout);
}

Tool *MSVCToolChain::buildLinker() const {
  return new tools::visualstudio::Linker(*this);
}

Tool *MSVCToolChain::buildAssembler() const {
  if (getTriple().isOSBinFormatMachO())
    return new tools::darwin::Assembler(*this);
  getDriver().Diag(clang::diag::err_no_external_assembler);
  return nullptr;
}

bool MSVCToolChain::IsIntegratedAssemblerDefault() const {
  return true;
}

bool MSVCToolChain::IsUnwindTablesDefault(const ArgList &Args) const {
  // Don't emit unwind tables by default for MachO targets.
  if (getTriple().isOSBinFormatMachO())
    return false;

  // All non-x86_32 Windows targets require unwind tables. However, LLVM
  // doesn't know how to generate them for all targets, so only enable
  // the ones that are actually implemented.
  return getArch() == llvm::Triple::x86_64 ||
         getArch() == llvm::Triple::aarch64;
}

bool MSVCToolChain::isPICDefault() const {
  return getArch() == llvm::Triple::x86_64 ||
         getArch() == llvm::Triple::aarch64;
}

bool MSVCToolChain::isPIEDefault(const llvm::opt::ArgList &Args) const {
  return false;
}

bool MSVCToolChain::isPICDefaultForced() const {
  return getArch() == llvm::Triple::x86_64 ||
         getArch() == llvm::Triple::aarch64;
}

void MSVCToolChain::AddCudaIncludeArgs(const ArgList &DriverArgs,
                                       ArgStringList &CC1Args) const {
  CudaInstallation.AddCudaIncludeArgs(DriverArgs, CC1Args);
}

void MSVCToolChain::AddHIPIncludeArgs(const ArgList &DriverArgs,
                                      ArgStringList &CC1Args) const {
  RocmInstallation.AddHIPIncludeArgs(DriverArgs, CC1Args);
}

void MSVCToolChain::printVerboseInfo(raw_ostream &OS) const {
  CudaInstallation.print(OS);
  RocmInstallation.print(OS);
}

// Get the path to a specific subdirectory in the current toolchain for
// a given target architecture.
// VS2017 changed the VC toolchain layout, so this should be used instead
// of hardcoding paths.
std::string
MSVCToolChain::getSubDirectoryPath(SubDirectoryType Type,
                                   llvm::StringRef SubdirParent,
                                   llvm::Triple::ArchType TargetArch) const {
  const char *SubdirName;
  const char *IncludeName;
  switch (VSLayout) {
  case ToolsetLayout::OlderVS:
    SubdirName = llvmArchToLegacyVCArch(TargetArch);
    IncludeName = "include";
    break;
  case ToolsetLayout::VS2017OrNewer:
    SubdirName = llvmArchToWindowsSDKArch(TargetArch);
    IncludeName = "include";
    break;
  case ToolsetLayout::DevDivInternal:
    SubdirName = llvmArchToDevDivInternalArch(TargetArch);
    IncludeName = "inc";
    break;
  }

  llvm::SmallString<256> Path(VCToolChainPath);
  if (!SubdirParent.empty())
    llvm::sys::path::append(Path, SubdirParent);

  switch (Type) {
  case SubDirectoryType::Bin:
    if (VSLayout == ToolsetLayout::VS2017OrNewer) {
      const bool HostIsX64 =
          llvm::Triple(llvm::sys::getProcessTriple()).isArch64Bit();
      const char *const HostName = HostIsX64 ? "Hostx64" : "Hostx86";
      llvm::sys::path::append(Path, "bin", HostName, SubdirName);
    } else { // OlderVS or DevDivInternal
      llvm::sys::path::append(Path, "bin", SubdirName);
    }
    break;
  case SubDirectoryType::Include:
    llvm::sys::path::append(Path, IncludeName);
    break;
  case SubDirectoryType::Lib:
    llvm::sys::path::append(Path, "lib", SubdirName);
    break;
  }
  return std::string(Path.str());
}

#ifdef _WIN32
static bool readFullStringValue(HKEY hkey, const char *valueName,
                                std::string &value) {
  std::wstring WideValueName;
  if (!llvm::ConvertUTF8toWide(valueName, WideValueName))
    return false;

  DWORD result = 0;
  DWORD valueSize = 0;
  DWORD type = 0;
  // First just query for the required size.
  result = RegQueryValueExW(hkey, WideValueName.c_str(), NULL, &type, NULL,
                            &valueSize);
  if (result != ERROR_SUCCESS || type != REG_SZ || !valueSize)
    return false;
  std::vector<BYTE> buffer(valueSize);
  result = RegQueryValueExW(hkey, WideValueName.c_str(), NULL, NULL, &buffer[0],
                            &valueSize);
  if (result == ERROR_SUCCESS) {
    std::wstring WideValue(reinterpret_cast<const wchar_t *>(buffer.data()),
                           valueSize / sizeof(wchar_t));
    if (valueSize && WideValue.back() == L'\0') {
      WideValue.pop_back();
    }
    // The destination buffer must be empty as an invariant of the conversion
    // function; but this function is sometimes called in a loop that passes in
    // the same buffer, however. Simply clear it out so we can overwrite it.
    value.clear();
    return llvm::convertWideToUTF8(WideValue, value);
  }
  return false;
}
#endif

/// Read registry string.
/// This also supports a means to look for high-versioned keys by use
/// of a $VERSION placeholder in the key path.
/// $VERSION in the key path is a placeholder for the version number,
/// causing the highest value path to be searched for and used.
/// I.e. "SOFTWARE\\Microsoft\\VisualStudio\\$VERSION".
/// There can be additional characters in the component.  Only the numeric
/// characters are compared.  This function only searches HKLM.
static bool getSystemRegistryString(const char *keyPath, const char *valueName,
                                    std::string &value, std::string *phValue) {
#ifndef _WIN32
  return false;
#else
  HKEY hRootKey = HKEY_LOCAL_MACHINE;
  HKEY hKey = NULL;
  long lResult;
  bool returnValue = false;

  const char *placeHolder = strstr(keyPath, "$VERSION");
  std::string bestName;
  // If we have a $VERSION placeholder, do the highest-version search.
  if (placeHolder) {
    const char *keyEnd = placeHolder - 1;
    const char *nextKey = placeHolder;
    // Find end of previous key.
    while ((keyEnd > keyPath) && (*keyEnd != '\\'))
      keyEnd--;
    // Find end of key containing $VERSION.
    while (*nextKey && (*nextKey != '\\'))
      nextKey++;
    size_t partialKeyLength = keyEnd - keyPath;
    char partialKey[256];
    if (partialKeyLength >= sizeof(partialKey))
      partialKeyLength = sizeof(partialKey) - 1;
    strncpy(partialKey, keyPath, partialKeyLength);
    partialKey[partialKeyLength] = '\0';
    HKEY hTopKey = NULL;
    lResult = RegOpenKeyExA(hRootKey, partialKey, 0, KEY_READ | KEY_WOW64_32KEY,
                            &hTopKey);
    if (lResult == ERROR_SUCCESS) {
      char keyName[256];
      double bestValue = 0.0;
      DWORD index, size = sizeof(keyName) - 1;
      for (index = 0; RegEnumKeyExA(hTopKey, index, keyName, &size, NULL, NULL,
                                    NULL, NULL) == ERROR_SUCCESS;
           index++) {
        const char *sp = keyName;
        while (*sp && !isDigit(*sp))
          sp++;
        if (!*sp)
          continue;
        const char *ep = sp + 1;
        while (*ep && (isDigit(*ep) || (*ep == '.')))
          ep++;
        char numBuf[32];
        strncpy(numBuf, sp, sizeof(numBuf) - 1);
        numBuf[sizeof(numBuf) - 1] = '\0';
        double dvalue = strtod(numBuf, NULL);
        if (dvalue > bestValue) {
          // Test that InstallDir is indeed there before keeping this index.
          // Open the chosen key path remainder.
          bestName = keyName;
          // Append rest of key.
          bestName.append(nextKey);
          lResult = RegOpenKeyExA(hTopKey, bestName.c_str(), 0,
                                  KEY_READ | KEY_WOW64_32KEY, &hKey);
          if (lResult == ERROR_SUCCESS) {
            if (readFullStringValue(hKey, valueName, value)) {
              bestValue = dvalue;
              if (phValue)
                *phValue = bestName;
              returnValue = true;
            }
            RegCloseKey(hKey);
          }
        }
        size = sizeof(keyName) - 1;
      }
      RegCloseKey(hTopKey);
    }
  } else {
    lResult =
        RegOpenKeyExA(hRootKey, keyPath, 0, KEY_READ | KEY_WOW64_32KEY, &hKey);
    if (lResult == ERROR_SUCCESS) {
      if (readFullStringValue(hKey, valueName, value))
        returnValue = true;
      if (phValue)
        phValue->clear();
      RegCloseKey(hKey);
    }
  }
  return returnValue;
#endif // _WIN32
}

// Find the most recent version of Universal CRT or Windows 10 SDK.
// vcvarsqueryregistry.bat from Visual Studio 2015 sorts entries in the include
// directory by name and uses the last one of the list.
// So we compare entry names lexicographically to find the greatest one.
static bool getWindows10SDKVersionFromPath(llvm::vfs::FileSystem &VFS,
                                           const std::string &SDKPath,
                                           std::string &SDKVersion) {
  llvm::SmallString<128> IncludePath(SDKPath);
  llvm::sys::path::append(IncludePath, "Include");
  SDKVersion = getHighestNumericTupleInDirectory(VFS, IncludePath);
  return !SDKVersion.empty();
}

static bool getWindowsSDKDirViaCommandLine(llvm::vfs::FileSystem &VFS,
                                           const ArgList &Args,
                                           std::string &Path, int &Major,
                                           std::string &Version) {
  if (Arg *A = Args.getLastArg(options::OPT__SLASH_winsdkdir,
                               options::OPT__SLASH_winsysroot)) {
    // Don't validate the input; trust the value supplied by the user.
    // The motivation is to prevent unnecessary file and registry access.
    llvm::VersionTuple SDKVersion;
    if (Arg *A = Args.getLastArg(options::OPT__SLASH_winsdkversion))
      SDKVersion.tryParse(A->getValue());

    if (A->getOption().getID() == options::OPT__SLASH_winsysroot) {
      llvm::SmallString<128> SDKPath(A->getValue());
      llvm::sys::path::append(SDKPath, "Windows Kits");
      if (!SDKVersion.empty())
        llvm::sys::path::append(SDKPath, Twine(SDKVersion.getMajor()));
      else
        llvm::sys::path::append(
            SDKPath, getHighestNumericTupleInDirectory(VFS, SDKPath));
      Path = std::string(SDKPath.str());
    } else {
      Path = A->getValue();
    }

    if (!SDKVersion.empty()) {
      Major = SDKVersion.getMajor();
      Version = SDKVersion.getAsString();
    } else if (getWindows10SDKVersionFromPath(VFS, Path, Version)) {
      Major = 10;
    }
    return true;
  }
  return false;
}

/// Get Windows SDK installation directory.
static bool getWindowsSDKDir(llvm::vfs::FileSystem &VFS, const ArgList &Args,
                             std::string &Path, int &Major,
                             std::string &WindowsSDKIncludeVersion,
                             std::string &WindowsSDKLibVersion) {
  // Trust /winsdkdir and /winsdkversion if present.
  if (getWindowsSDKDirViaCommandLine(VFS, Args, Path, Major,
                                     WindowsSDKIncludeVersion)) {
    WindowsSDKLibVersion = WindowsSDKIncludeVersion;
    return true;
  }

  // FIXME: Try env vars (%WindowsSdkDir%, %UCRTVersion%) before going to registry.

  // Try the Windows registry.
  std::string RegistrySDKVersion;
  if (!getSystemRegistryString(
          "SOFTWARE\\Microsoft\\Microsoft SDKs\\Windows\\$VERSION",
          "InstallationFolder", Path, &RegistrySDKVersion))
    return false;
  if (Path.empty() || RegistrySDKVersion.empty())
    return false;

  WindowsSDKIncludeVersion.clear();
  WindowsSDKLibVersion.clear();
  Major = 0;
  std::sscanf(RegistrySDKVersion.c_str(), "v%d.", &Major);
  if (Major <= 7)
    return true;
  if (Major == 8) {
    // Windows SDK 8.x installs libraries in a folder whose names depend on the
    // version of the OS you're targeting.  By default choose the newest, which
    // usually corresponds to the version of the OS you've installed the SDK on.
    const char *Tests[] = {"winv6.3", "win8", "win7"};
    for (const char *Test : Tests) {
      llvm::SmallString<128> TestPath(Path);
      llvm::sys::path::append(TestPath, "Lib", Test);
      if (VFS.exists(TestPath)) {
        WindowsSDKLibVersion = Test;
        break;
      }
    }
    return !WindowsSDKLibVersion.empty();
  }
  if (Major == 10) {
    if (!getWindows10SDKVersionFromPath(VFS, Path, WindowsSDKIncludeVersion))
      return false;
    WindowsSDKLibVersion = WindowsSDKIncludeVersion;
    return true;
  }
  // Unsupported SDK version
  return false;
}

// Gets the library path required to link against the Windows SDK.
bool MSVCToolChain::getWindowsSDKLibraryPath(
    const ArgList &Args, std::string &path) const {
  std::string sdkPath;
  int sdkMajor = 0;
  std::string windowsSDKIncludeVersion;
  std::string windowsSDKLibVersion;

  path.clear();
  if (!getWindowsSDKDir(getVFS(), Args, sdkPath, sdkMajor,
                        windowsSDKIncludeVersion, windowsSDKLibVersion))
    return false;

  llvm::SmallString<128> libPath(sdkPath);
  llvm::sys::path::append(libPath, "Lib");
  if (sdkMajor >= 8) {
    llvm::sys::path::append(libPath, windowsSDKLibVersion, "um",
                            llvmArchToWindowsSDKArch(getArch()));
  } else {
    switch (getArch()) {
    // In Windows SDK 7.x, x86 libraries are directly in the Lib folder.
    case llvm::Triple::x86:
      break;
    case llvm::Triple::x86_64:
      llvm::sys::path::append(libPath, "x64");
      break;
    case llvm::Triple::arm:
      // It is not necessary to link against Windows SDK 7.x when targeting ARM.
      return false;
    default:
      return false;
    }
  }

  path = std::string(libPath.str());
  return true;
}

// Check if the Include path of a specified version of Visual Studio contains
// specific header files. If not, they are probably shipped with Universal CRT.
bool MSVCToolChain::useUniversalCRT() const {
  llvm::SmallString<128> TestPath(
      getSubDirectoryPath(SubDirectoryType::Include));
  llvm::sys::path::append(TestPath, "stdlib.h");
  return !getVFS().exists(TestPath);
}

static bool getUniversalCRTSdkDir(llvm::vfs::FileSystem &VFS,
                                  const ArgList &Args, std::string &Path,
                                  std::string &UCRTVersion) {
  // If /winsdkdir is passed, use it as location for the UCRT too.
  // FIXME: Should there be a dedicated /ucrtdir to override /winsdkdir?
  int Major;
  if (getWindowsSDKDirViaCommandLine(VFS, Args, Path, Major, UCRTVersion))
    return true;

  // FIXME: Try env vars (%UniversalCRTSdkDir%, %UCRTVersion%) before going to
  // registry.

  // vcvarsqueryregistry.bat for Visual Studio 2015 queries the registry
  // for the specific key "KitsRoot10". So do we.
  if (!getSystemRegistryString(
          "SOFTWARE\\Microsoft\\Windows Kits\\Installed Roots", "KitsRoot10",
          Path, nullptr))
    return false;

  return getWindows10SDKVersionFromPath(VFS, Path, UCRTVersion);
}

bool MSVCToolChain::getUniversalCRTLibraryPath(const ArgList &Args,
                                               std::string &Path) const {
  std::string UniversalCRTSdkPath;
  std::string UCRTVersion;

  Path.clear();
  if (!getUniversalCRTSdkDir(getVFS(), Args, UniversalCRTSdkPath, UCRTVersion))
    return false;

  StringRef ArchName = llvmArchToWindowsSDKArch(getArch());
  if (ArchName.empty())
    return false;

  llvm::SmallString<128> LibPath(UniversalCRTSdkPath);
  llvm::sys::path::append(LibPath, "Lib", UCRTVersion, "ucrt", ArchName);

  Path = std::string(LibPath.str());
  return true;
}

static VersionTuple getMSVCVersionFromExe(const std::string &BinDir) {
  VersionTuple Version;
#ifdef _WIN32
  SmallString<128> ClExe(BinDir);
  llvm::sys::path::append(ClExe, "cl.exe");

  std::wstring ClExeWide;
  if (!llvm::ConvertUTF8toWide(ClExe.c_str(), ClExeWide))
    return Version;

  const DWORD VersionSize = ::GetFileVersionInfoSizeW(ClExeWide.c_str(),
                                                      nullptr);
  if (VersionSize == 0)
    return Version;

  SmallVector<uint8_t, 4 * 1024> VersionBlock(VersionSize);
  if (!::GetFileVersionInfoW(ClExeWide.c_str(), 0, VersionSize,
                             VersionBlock.data()))
    return Version;

  VS_FIXEDFILEINFO *FileInfo = nullptr;
  UINT FileInfoSize = 0;
  if (!::VerQueryValueW(VersionBlock.data(), L"\\",
                        reinterpret_cast<LPVOID *>(&FileInfo), &FileInfoSize) ||
      FileInfoSize < sizeof(*FileInfo))
    return Version;

  const unsigned Major = (FileInfo->dwFileVersionMS >> 16) & 0xFFFF;
  const unsigned Minor = (FileInfo->dwFileVersionMS      ) & 0xFFFF;
  const unsigned Micro = (FileInfo->dwFileVersionLS >> 16) & 0xFFFF;

  Version = VersionTuple(Major, Minor, Micro);
#endif
  return Version;
}

void MSVCToolChain::AddSystemIncludeWithSubfolder(
    const ArgList &DriverArgs, ArgStringList &CC1Args,
    const std::string &folder, const Twine &subfolder1, const Twine &subfolder2,
    const Twine &subfolder3) const {
  llvm::SmallString<128> path(folder);
  llvm::sys::path::append(path, subfolder1, subfolder2, subfolder3);
  addSystemInclude(DriverArgs, CC1Args, path);
}

void MSVCToolChain::AddClangSystemIncludeArgs(const ArgList &DriverArgs,
                                              ArgStringList &CC1Args) const {
  if (DriverArgs.hasArg(options::OPT_nostdinc))
    return;

  if (!DriverArgs.hasArg(options::OPT_nobuiltininc)) {
    AddSystemIncludeWithSubfolder(DriverArgs, CC1Args, getDriver().ResourceDir,
                                  "include");
  }

  // Add %INCLUDE%-like directories from the -imsvc flag.
  for (const auto &Path : DriverArgs.getAllArgValues(options::OPT__SLASH_imsvc))
    addSystemInclude(DriverArgs, CC1Args, Path);

  auto AddSystemIncludesFromEnv = [&](StringRef Var) -> bool {
    if (auto Val = llvm::sys::Process::GetEnv(Var)) {
      SmallVector<StringRef, 8> Dirs;
      StringRef(*Val).split(Dirs, ";", /*MaxSplit=*/-1, /*KeepEmpty=*/false);
      if (!Dirs.empty()) {
        addSystemIncludes(DriverArgs, CC1Args, Dirs);
        return true;
      }
    }
    return false;
  };

  // Add %INCLUDE%-like dirs via /external:env: flags.
  for (const auto &Var :
       DriverArgs.getAllArgValues(options::OPT__SLASH_external_env)) {
    AddSystemIncludesFromEnv(Var);
  }

  // Add DIA SDK include if requested.
  if (const Arg *A = DriverArgs.getLastArg(options::OPT__SLASH_diasdkdir,
                                           options::OPT__SLASH_winsysroot)) {
    // cl.exe doesn't find the DIA SDK automatically, so this too requires
    // explicit flags and doesn't automatically look in "DIA SDK" relative
    // to the path we found for VCToolChainPath.
    llvm::SmallString<128> DIASDKPath(A->getValue());
    if (A->getOption().getID() == options::OPT__SLASH_winsysroot)
      llvm::sys::path::append(DIASDKPath, "DIA SDK");
    AddSystemIncludeWithSubfolder(DriverArgs, CC1Args, std::string(DIASDKPath),
                                  "include");
  }

  if (DriverArgs.hasArg(options::OPT_nostdlibinc))
    return;

  // Honor %INCLUDE% and %EXTERNAL_INCLUDE%. It should have essential search
  // paths set by vcvarsall.bat. Skip if the user expressly set a vctoolsdir.
  if (!DriverArgs.getLastArg(options::OPT__SLASH_vctoolsdir,
                             options::OPT__SLASH_winsysroot)) {
    bool Found = AddSystemIncludesFromEnv("INCLUDE");
    Found |= AddSystemIncludesFromEnv("EXTERNAL_INCLUDE");
    if (Found)
      return;
  }

  // When built with access to the proper Windows APIs, try to actually find
  // the correct include paths first.
  if (!VCToolChainPath.empty()) {
    addSystemInclude(DriverArgs, CC1Args,
                     getSubDirectoryPath(SubDirectoryType::Include));
    addSystemInclude(DriverArgs, CC1Args,
                     getSubDirectoryPath(SubDirectoryType::Include, "atlmfc"));

    if (useUniversalCRT()) {
      std::string UniversalCRTSdkPath;
      std::string UCRTVersion;
      if (getUniversalCRTSdkDir(getVFS(), DriverArgs, UniversalCRTSdkPath,
                                UCRTVersion)) {
        AddSystemIncludeWithSubfolder(DriverArgs, CC1Args, UniversalCRTSdkPath,
                                      "Include", UCRTVersion, "ucrt");
      }
    }

    std::string WindowsSDKDir;
    int major = 0;
    std::string windowsSDKIncludeVersion;
    std::string windowsSDKLibVersion;
    if (getWindowsSDKDir(getVFS(), DriverArgs, WindowsSDKDir, major,
                         windowsSDKIncludeVersion, windowsSDKLibVersion)) {
      if (major >= 8) {
        // Note: windowsSDKIncludeVersion is empty for SDKs prior to v10.
        // Anyway, llvm::sys::path::append is able to manage it.
        AddSystemIncludeWithSubfolder(DriverArgs, CC1Args, WindowsSDKDir,
                                      "Include", windowsSDKIncludeVersion,
                                      "shared");
        AddSystemIncludeWithSubfolder(DriverArgs, CC1Args, WindowsSDKDir,
                                      "Include", windowsSDKIncludeVersion,
                                      "um");
        AddSystemIncludeWithSubfolder(DriverArgs, CC1Args, WindowsSDKDir,
                                      "Include", windowsSDKIncludeVersion,
                                      "winrt");
      } else {
        AddSystemIncludeWithSubfolder(DriverArgs, CC1Args, WindowsSDKDir,
                                      "Include");
      }
    }

    return;
  }

#if defined(_WIN32)
  // As a fallback, select default install paths.
  // FIXME: Don't guess drives and paths like this on Windows.
  const StringRef Paths[] = {
    "C:/Program Files/Microsoft Visual Studio 10.0/VC/include",
    "C:/Program Files/Microsoft Visual Studio 9.0/VC/include",
    "C:/Program Files/Microsoft Visual Studio 9.0/VC/PlatformSDK/Include",
    "C:/Program Files/Microsoft Visual Studio 8/VC/include",
    "C:/Program Files/Microsoft Visual Studio 8/VC/PlatformSDK/Include"
  };
  addSystemIncludes(DriverArgs, CC1Args, Paths);
#endif
}

void MSVCToolChain::AddClangCXXStdlibIncludeArgs(const ArgList &DriverArgs,
                                                 ArgStringList &CC1Args) const {
  // FIXME: There should probably be logic here to find libc++ on Windows.
}

VersionTuple MSVCToolChain::computeMSVCVersion(const Driver *D,
                                               const ArgList &Args) const {
  bool IsWindowsMSVC = getTriple().isWindowsMSVCEnvironment();
  VersionTuple MSVT = ToolChain::computeMSVCVersion(D, Args);
  if (MSVT.empty())
    MSVT = getTriple().getEnvironmentVersion();
  if (MSVT.empty() && IsWindowsMSVC)
    MSVT = getMSVCVersionFromExe(getSubDirectoryPath(SubDirectoryType::Bin));
  if (MSVT.empty() &&
      Args.hasFlag(options::OPT_fms_extensions, options::OPT_fno_ms_extensions,
                   IsWindowsMSVC)) {
    // -fms-compatibility-version=19.14 is default, aka 2017, 15.7
    MSVT = VersionTuple(19, 14);
  }
  return MSVT;
}

std::string
MSVCToolChain::ComputeEffectiveClangTriple(const ArgList &Args,
                                           types::ID InputType) const {
  // The MSVC version doesn't care about the architecture, even though it
  // may look at the triple internally.
  VersionTuple MSVT = computeMSVCVersion(/*D=*/nullptr, Args);
  MSVT = VersionTuple(MSVT.getMajor(), MSVT.getMinor().getValueOr(0),
                      MSVT.getSubminor().getValueOr(0));

  // For the rest of the triple, however, a computed architecture name may
  // be needed.
  llvm::Triple Triple(ToolChain::ComputeEffectiveClangTriple(Args, InputType));
  if (Triple.getEnvironment() == llvm::Triple::MSVC) {
    StringRef ObjFmt = Triple.getEnvironmentName().split('-').second;
    if (ObjFmt.empty())
      Triple.setEnvironmentName((Twine("msvc") + MSVT.getAsString()).str());
    else
      Triple.setEnvironmentName(
          (Twine("msvc") + MSVT.getAsString() + Twine('-') + ObjFmt).str());
  }
  return Triple.getTriple();
}

SanitizerMask MSVCToolChain::getSupportedSanitizers() const {
  SanitizerMask Res = ToolChain::getSupportedSanitizers();
  Res |= SanitizerKind::Address;
  Res |= SanitizerKind::PointerCompare;
  Res |= SanitizerKind::PointerSubtract;
  Res |= SanitizerKind::Fuzzer;
  Res |= SanitizerKind::FuzzerNoLink;
  Res &= ~SanitizerKind::CFIMFCall;
  return Res;
}

static void TranslateOptArg(Arg *A, llvm::opt::DerivedArgList &DAL,
                            bool SupportsForcingFramePointer,
                            const char *ExpandChar, const OptTable &Opts) {
  assert(A->getOption().matches(options::OPT__SLASH_O));

  StringRef OptStr = A->getValue();
  for (size_t I = 0, E = OptStr.size(); I != E; ++I) {
    const char &OptChar = *(OptStr.data() + I);
    switch (OptChar) {
    default:
      break;
    case '1':
    case '2':
    case 'x':
    case 'd':
      // Ignore /O[12xd] flags that aren't the last one on the command line.
      // Only the last one gets expanded.
      if (&OptChar != ExpandChar) {
        A->claim();
        break;
      }
      if (OptChar == 'd') {
        DAL.AddFlagArg(A, Opts.getOption(options::OPT_O0));
      } else {
        if (OptChar == '1') {
          DAL.AddJoinedArg(A, Opts.getOption(options::OPT_O), "s");
        } else if (OptChar == '2' || OptChar == 'x') {
          DAL.AddFlagArg(A, Opts.getOption(options::OPT_fbuiltin));
          DAL.AddJoinedArg(A, Opts.getOption(options::OPT_O), "2");
        }
        if (SupportsForcingFramePointer &&
            !DAL.hasArgNoClaim(options::OPT_fno_omit_frame_pointer))
          DAL.AddFlagArg(A, Opts.getOption(options::OPT_fomit_frame_pointer));
        if (OptChar == '1' || OptChar == '2')
          DAL.AddFlagArg(A, Opts.getOption(options::OPT_ffunction_sections));
      }
      break;
    case 'b':
      if (I + 1 != E && isdigit(OptStr[I + 1])) {
        switch (OptStr[I + 1]) {
        case '0':
          DAL.AddFlagArg(A, Opts.getOption(options::OPT_fno_inline));
          break;
        case '1':
          DAL.AddFlagArg(A, Opts.getOption(options::OPT_finline_hint_functions));
          break;
        case '2':
          DAL.AddFlagArg(A, Opts.getOption(options::OPT_finline_functions));
          break;
        }
        ++I;
      }
      break;
    case 'g':
      A->claim();
      break;
    case 'i':
      if (I + 1 != E && OptStr[I + 1] == '-') {
        ++I;
        DAL.AddFlagArg(A, Opts.getOption(options::OPT_fno_builtin));
      } else {
        DAL.AddFlagArg(A, Opts.getOption(options::OPT_fbuiltin));
      }
      break;
    case 's':
      DAL.AddJoinedArg(A, Opts.getOption(options::OPT_O), "s");
      break;
    case 't':
      DAL.AddJoinedArg(A, Opts.getOption(options::OPT_O), "2");
      break;
    case 'y': {
      bool OmitFramePointer = true;
      if (I + 1 != E && OptStr[I + 1] == '-') {
        OmitFramePointer = false;
        ++I;
      }
      if (SupportsForcingFramePointer) {
        if (OmitFramePointer)
          DAL.AddFlagArg(A,
                         Opts.getOption(options::OPT_fomit_frame_pointer));
        else
          DAL.AddFlagArg(
              A, Opts.getOption(options::OPT_fno_omit_frame_pointer));
      } else {
        // Don't warn about /Oy- in x86-64 builds (where
        // SupportsForcingFramePointer is false).  The flag having no effect
        // there is a compiler-internal optimization, and people shouldn't have
        // to special-case their build files for x86-64 clang-cl.
        A->claim();
      }
      break;
    }
    }
  }
}

static void TranslateDArg(Arg *A, llvm::opt::DerivedArgList &DAL,
                          const OptTable &Opts) {
  assert(A->getOption().matches(options::OPT_D));

  StringRef Val = A->getValue();
  size_t Hash = Val.find('#');
  if (Hash == StringRef::npos || Hash > Val.find('=')) {
    DAL.append(A);
    return;
  }

  std::string NewVal = std::string(Val);
  NewVal[Hash] = '=';
  DAL.AddJoinedArg(A, Opts.getOption(options::OPT_D), NewVal);
}

static void TranslatePermissive(Arg *A, llvm::opt::DerivedArgList &DAL,
                                const OptTable &Opts) {
  DAL.AddFlagArg(A, Opts.getOption(options::OPT__SLASH_Zc_twoPhase_));
  DAL.AddFlagArg(A, Opts.getOption(options::OPT_fno_operator_names));
}

static void TranslatePermissiveMinus(Arg *A, llvm::opt::DerivedArgList &DAL,
                                     const OptTable &Opts) {
  DAL.AddFlagArg(A, Opts.getOption(options::OPT__SLASH_Zc_twoPhase));
  DAL.AddFlagArg(A, Opts.getOption(options::OPT_foperator_names));
}

llvm::opt::DerivedArgList *
MSVCToolChain::TranslateArgs(const llvm::opt::DerivedArgList &Args,
                             StringRef BoundArch,
                             Action::OffloadKind OFK) const {
  DerivedArgList *DAL = new DerivedArgList(Args.getBaseArgs());
  const OptTable &Opts = getDriver().getOpts();

  // /Oy and /Oy- don't have an effect on X86-64
  bool SupportsForcingFramePointer = getArch() != llvm::Triple::x86_64;

  // The -O[12xd] flag actually expands to several flags.  We must desugar the
  // flags so that options embedded can be negated.  For example, the '-O2' flag
  // enables '-Oy'.  Expanding '-O2' into its constituent flags allows us to
  // correctly handle '-O2 -Oy-' where the trailing '-Oy-' disables a single
  // aspect of '-O2'.
  //
  // Note that this expansion logic only applies to the *last* of '[12xd]'.

  // First step is to search for the character we'd like to expand.
  const char *ExpandChar = nullptr;
  for (Arg *A : Args.filtered(options::OPT__SLASH_O)) {
    StringRef OptStr = A->getValue();
    for (size_t I = 0, E = OptStr.size(); I != E; ++I) {
      char OptChar = OptStr[I];
      char PrevChar = I > 0 ? OptStr[I - 1] : '0';
      if (PrevChar == 'b') {
        // OptChar does not expand; it's an argument to the previous char.
        continue;
      }
      if (OptChar == '1' || OptChar == '2' || OptChar == 'x' || OptChar == 'd')
        ExpandChar = OptStr.data() + I;
    }
  }

  for (Arg *A : Args) {
    if (A->getOption().matches(options::OPT__SLASH_O)) {
      // The -O flag actually takes an amalgam of other options.  For example,
      // '/Ogyb2' is equivalent to '/Og' '/Oy' '/Ob2'.
      TranslateOptArg(A, *DAL, SupportsForcingFramePointer, ExpandChar, Opts);
    } else if (A->getOption().matches(options::OPT_D)) {
      // Translate -Dfoo#bar into -Dfoo=bar.
      TranslateDArg(A, *DAL, Opts);
    } else if (A->getOption().matches(options::OPT__SLASH_permissive)) {
      // Expand /permissive
      TranslatePermissive(A, *DAL, Opts);
    } else if (A->getOption().matches(options::OPT__SLASH_permissive_)) {
      // Expand /permissive-
      TranslatePermissiveMinus(A, *DAL, Opts);
    } else if (OFK != Action::OFK_HIP) {
      // HIP Toolchain translates input args by itself.
      DAL->append(A);
    }
  }

  return DAL;
}

void MSVCToolChain::addClangTargetOptions(
    const ArgList &DriverArgs, ArgStringList &CC1Args,
    Action::OffloadKind DeviceOffloadKind) const {
  // MSVC STL kindly allows removing all usages of typeid by defining
  // _HAS_STATIC_RTTI to 0. Do so, when compiling with -fno-rtti
  if (DriverArgs.hasArg(options::OPT_fno_rtti, options::OPT_frtti,
                        /*Default=*/false))
    CC1Args.push_back("-D_HAS_STATIC_RTTI=0");
}
