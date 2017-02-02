//===--- ToolChains.cpp - ToolChain Implementations -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ToolChains.h"
#include "Tools.h"
#include "clang/Basic/CharInfo.h"
#include "clang/Basic/Version.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/Options.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include <cstdio>

#if defined(LLVM_ON_WIN32)
  #define USE_WIN32

  // FIXME: Make this configurable with cmake when the final version of the API
  //        has been released.
  #if 0
    #define USE_VS_SETUP_CONFIG
  #endif
#endif

// Include the necessary headers to interface with the Windows registry and
// environment.
#ifdef USE_WIN32
  #define WIN32_LEAN_AND_MEAN
  #define NOGDI
  #ifndef NOMINMAX
    #define NOMINMAX
  #endif
  #include <windows.h>
#endif

// Include the headers needed for the setup config COM stuff and define
// smart pointers for the interfaces we need.
#ifdef USE_VS_SETUP_CONFIG
  #include "clang/Basic/VirtualFileSystem.h"
  #include "llvm/Support/COM.h"
  #include <comdef.h>
  #include <Setup.Configuration.h>
  _COM_SMARTPTR_TYPEDEF(ISetupConfiguration,  __uuidof(ISetupConfiguration));
  _COM_SMARTPTR_TYPEDEF(ISetupConfiguration2, __uuidof(ISetupConfiguration2));
  _COM_SMARTPTR_TYPEDEF(ISetupHelper,         __uuidof(ISetupHelper));
  _COM_SMARTPTR_TYPEDEF(IEnumSetupInstances,  __uuidof(IEnumSetupInstances));
  _COM_SMARTPTR_TYPEDEF(ISetupInstance,       __uuidof(ISetupInstance));
  _COM_SMARTPTR_TYPEDEF(ISetupInstance2,      __uuidof(ISetupInstance2));
#endif

using namespace clang::driver;
using namespace clang::driver::toolchains;
using namespace clang;
using namespace llvm::opt;

// Defined below.
// Forward declare this so there aren't too many things above the constructor.
static bool getSystemRegistryString(const char *keyPath, const char *valueName,
                                    std::string &value, std::string *phValue);

// Check various environment variables to try and find a toolchain.
static bool findVCToolChainViaEnvironment(std::string &Path,
                                          bool &IsVS2017OrNewer) {
  // These variables are typically set by vcvarsall.bat
  // when launching a developer command prompt.
  if (llvm::Optional<std::string> VCToolsInstallDir =
          llvm::sys::Process::GetEnv("VCToolsInstallDir")) {
    // This is only set by newer Visual Studios, and it leads straight to
    // the toolchain directory.
    Path = std::move(*VCToolsInstallDir);
    IsVS2017OrNewer = true;
    return true;
  }
  if (llvm::Optional<std::string> VCInstallDir =
          llvm::sys::Process::GetEnv("VCINSTALLDIR")) {
    // If the previous variable isn't set but this one is, then we've found
    // an older Visual Studio. This variable is set by newer Visual Studios too,
    // so this check has to appear second.
    // In older Visual Studios, the VC directory is the toolchain.
    Path = std::move(*VCInstallDir);
    IsVS2017OrNewer = false;
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
      if (!llvm::sys::fs::exists(ExeTestPath))
        continue;

      // cl.exe existing isn't a conclusive test for a VC toolchain; clang also
      // has a cl.exe. So let's check for link.exe too.
      ExeTestPath = PathEntry;
      llvm::sys::path::append(ExeTestPath, "link.exe");
      if (!llvm::sys::fs::exists(ExeTestPath))
        continue;

      // whatever/VC/bin --> old toolchain, VC dir is toolchain dir.
      if (llvm::sys::path::filename(PathEntry) == "bin") {
        llvm::StringRef ParentPath = llvm::sys::path::parent_path(PathEntry);
        if (llvm::sys::path::filename(ParentPath) == "VC") {
          Path = ParentPath;
          IsVS2017OrNewer = false;
          return true;
        }
      } else {
        // This could be a new (>=VS2017) toolchain. If it is, we should find
        // path components with these prefixes when walking backwards through
        // the path.
        // Note: empty strings match anything.
        llvm::StringRef ExpectedPrefixes[] =
        { "", "Host", "bin", "", "MSVC", "Tools", "VC" };

        llvm::sys::path::reverse_iterator
            It = llvm::sys::path::rbegin(PathEntry),
            End = llvm::sys::path::rend(PathEntry);
        for (llvm::StringRef Prefix : ExpectedPrefixes) {
          if (It == End) goto NotAToolChain;
          if (!It->startswith(Prefix)) goto NotAToolChain;
          ++It;
        }

        // We've found a new toolchain!
        // Back up 3 times (/bin/Host/arch) to get the root path.
        llvm::StringRef ToolChainPath(PathEntry);
        for (int i = 0; i < 3; ++i)
          ToolChainPath = llvm::sys::path::parent_path(ToolChainPath);

        Path = ToolChainPath;
        IsVS2017OrNewer = true;
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
static bool findVCToolChainViaSetupConfig(std::string &Path,
                                          bool &IsVS2017OrNewer) {
#ifndef USE_VS_SETUP_CONFIG
  return false;
#else
  llvm::sys::InitializeCOMRAII COM(llvm::sys::COMThreadingMode::SingleThreaded);
  HRESULT HR;

  // _com_ptr_t will throw a _com_error if a COM calls fail.
  // The LLVM coding standards forbid exception handling, so we'll have to
  // stop them from being thrown in the first place.
  // The destructor will put the regular error handler back when we leave
  // this scope.
  struct SuppressCOMErrorsRAII {
    SuppressCOMErrorsRAII() {
      _set_com_error_handler([](HRESULT, IErrorInfo *) { });
    }
    ~SuppressCOMErrorsRAII() {
      _set_com_error_handler(_com_raise_error);
    }
  } COMErrorSuppressor;

  ISetupConfigurationPtr Query;
  HR = Query.CreateInstance(__uuidof(SetupConfiguration));
  if (FAILED(HR)) return false;

  IEnumSetupInstancesPtr EnumInstances;
  HR = ISetupConfiguration2Ptr(Query)->EnumAllInstances(&EnumInstances);
  if (FAILED(HR)) return false;

  ISetupInstancePtr Instance;
  HR = EnumInstances->Next(1, &Instance, nullptr);
  if (HR != S_OK) return false;

  ISetupInstancePtr NewestInstance(Instance);
  uint64_t NewestVersionNum;
  {
    bstr_t VersionString;
    HR = NewestInstance->GetInstallationVersion(VersionString.GetAddress());
    if (FAILED(HR)) return false;
    HR = ISetupHelperPtr(Query)->ParseVersion(VersionString,
                                              &NewestVersionNum);
    if (FAILED(HR)) return false;
  }

  while ((HR = EnumInstances->Next(1, &Instance, nullptr)) == S_OK) {
    bstr_t VersionString;
    uint64_t VersionNum;
    HR = Instance->GetInstallationVersion(VersionString.GetAddress());
    if (FAILED(HR)) continue;
    HR = ISetupHelperPtr(Query)->ParseVersion(VersionString,
                                              &VersionNum);
    if (FAILED(HR)) continue;
    if (VersionNum > NewestVersionNum) {
      NewestInstance = Instance;
      NewestVersionNum = VersionNum;
    }
  }

  bstr_t VCPathWide;
  HR = NewestInstance->ResolvePath(L"VC",
                                   VCPathWide.GetAddress());
  if (FAILED(HR)) return false;

  std::string VCRootPath;
  llvm::convertWideToUTF8(std::wstring(VCPathWide), VCRootPath);

  llvm::SmallString<256> ToolsVersionFilePath(VCRootPath);
  llvm::sys::path::append(ToolsVersionFilePath,
                          "Auxiliary",
                          "Build",
                          "Microsoft.VCToolsVersion.default.txt");

  auto ToolsVersionFile =
      clang::vfs::getRealFileSystem()->getBufferForFile(ToolsVersionFilePath);
  if (!ToolsVersionFile)
    return false;

  llvm::SmallString<256> ToolchainPath(VCRootPath);
  llvm::sys::path::append(ToolchainPath,
                          "Tools",
                          "MSVC",
                          ToolsVersionFile->get()->getBuffer().rtrim());
  if (!llvm::sys::fs::is_directory(ToolchainPath))
    return false;

  Path = ToolchainPath.str();
  IsVS2017OrNewer = true;
  return true;
#endif /*USE_VS_SETUP_CONFIG*/
}

// Look in the registry for Visual Studio installs, and use that to get
// a toolchain path. VS2017 and newer don't get added to the registry.
// So if we find something here, we know that it's an older version.
static bool findVCToolChainViaRegistry(std::string &Path,
                                       bool &IsVS2017OrNewer) {
  std::string VSInstallPath;
  if (getSystemRegistryString(R"(SOFTWARE\Microsoft\VisualStudio\$VERSION)",
                              "InstallDir", VSInstallPath, nullptr) ||
      getSystemRegistryString(R"(SOFTWARE\Microsoft\VCExpress\$VERSION)",
                              "InstallDir", VSInstallPath, nullptr)) {
    if (!VSInstallPath.empty()) {
      llvm::SmallString<256>
          VCPath(llvm::StringRef(VSInstallPath.c_str(),
                                 VSInstallPath.find(R"(\Common7\IDE)")));
      llvm::sys::path::append(VCPath, "VC");

      Path = VCPath.str();
      IsVS2017OrNewer = false;
      return true;
    }
  }
  return false;
}

MSVCToolChain::MSVCToolChain(const Driver &D, const llvm::Triple& Triple,
                             const ArgList &Args)
    : ToolChain(D, Triple, Args), CudaInstallation(D, Triple, Args) {
  getProgramPaths().push_back(getDriver().getInstalledDir());
  if (getDriver().getInstalledDir() != getDriver().Dir)
    getProgramPaths().push_back(getDriver().Dir);

  // Check the environment first, since that's probably the user telling us
  // what they want to use.
  // Failing that, just try to find the newest Visual Studio version we can
  // and use its default VC toolchain.
  findVCToolChainViaEnvironment(VCToolChainPath, IsVS2017OrNewer)
    || findVCToolChainViaSetupConfig(VCToolChainPath, IsVS2017OrNewer)
         || findVCToolChainViaRegistry(VCToolChainPath, IsVS2017OrNewer);
}

Tool *MSVCToolChain::buildLinker() const {
  if (VCToolChainPath.empty()) {
    getDriver().Diag(clang::diag::err_drv_msvc_not_found);
    return nullptr;
  }
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

bool MSVCToolChain::IsUnwindTablesDefault() const {
  // Emit unwind tables by default on Win64. All non-x86_32 Windows platforms
  // such as ARM and PPC actually require unwind tables, but LLVM doesn't know
  // how to generate them yet.

  // Don't emit unwind tables by default for MachO targets.
  if (getTriple().isOSBinFormatMachO())
    return false;

  return getArch() == llvm::Triple::x86_64;
}

bool MSVCToolChain::isPICDefault() const {
  return getArch() == llvm::Triple::x86_64;
}

bool MSVCToolChain::isPIEDefault() const {
  return false;
}

bool MSVCToolChain::isPICDefaultForced() const {
  return getArch() == llvm::Triple::x86_64;
}

void MSVCToolChain::AddCudaIncludeArgs(const ArgList &DriverArgs,
                                       ArgStringList &CC1Args) const {
  CudaInstallation.AddCudaIncludeArgs(DriverArgs, CC1Args);
}

void MSVCToolChain::printVerboseInfo(raw_ostream &OS) const {
  CudaInstallation.print(OS);
}

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
  default:
    return "";
  }
}

// Get the path to a specific subdirectory in the current toolchain for
// a given target architecture.
// VS2017 changed the VC toolchain layout, so this should be used instead
// of hardcoding paths.
std::string
    MSVCToolChain::getSubDirectoryPath(SubDirectoryType Type,
                                       llvm::Triple::ArchType TargetArch) const {
  llvm::SmallString<256> Path(VCToolChainPath);
  switch (Type) {
  case SubDirectoryType::Bin:
    if (IsVS2017OrNewer) {
      bool HostIsX64 = llvm::Triple(llvm::sys::getProcessTriple()).isArch64Bit();
      llvm::sys::path::append(Path,
        "bin",
        (HostIsX64 ? "HostX64" : "HostX86"),
        llvmArchToWindowsSDKArch(TargetArch));
    }
    else {
      llvm::sys::path::append(Path,
        "bin",
        llvmArchToLegacyVCArch(TargetArch));
    }
    break;
  case SubDirectoryType::Include:
    llvm::sys::path::append(Path, "include");
    break;
  case SubDirectoryType::Lib:
    llvm::sys::path::append(Path,
      "lib",
      IsVS2017OrNewer
        ? llvmArchToWindowsSDKArch(TargetArch)
        : llvmArchToLegacyVCArch(TargetArch));
    break;
  }
  return Path.str();
}

#ifdef USE_WIN32
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

/// \brief Read registry string.
/// This also supports a means to look for high-versioned keys by use
/// of a $VERSION placeholder in the key path.
/// $VERSION in the key path is a placeholder for the version number,
/// causing the highest value path to be searched for and used.
/// I.e. "SOFTWARE\\Microsoft\\VisualStudio\\$VERSION".
/// There can be additional characters in the component.  Only the numeric
/// characters are compared.  This function only searches HKLM.
static bool getSystemRegistryString(const char *keyPath, const char *valueName,
                                    std::string &value, std::string *phValue) {
#ifndef USE_WIN32
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
#endif // USE_WIN32
}

// Find the most recent version of Universal CRT or Windows 10 SDK.
// vcvarsqueryregistry.bat from Visual Studio 2015 sorts entries in the include
// directory by name and uses the last one of the list.
// So we compare entry names lexicographically to find the greatest one.
static bool getWindows10SDKVersionFromPath(const std::string &SDKPath,
                                           std::string &SDKVersion) {
  SDKVersion.clear();

  std::error_code EC;
  llvm::SmallString<128> IncludePath(SDKPath);
  llvm::sys::path::append(IncludePath, "Include");
  for (llvm::sys::fs::directory_iterator DirIt(IncludePath, EC), DirEnd;
       DirIt != DirEnd && !EC; DirIt.increment(EC)) {
    if (!llvm::sys::fs::is_directory(DirIt->path()))
      continue;
    StringRef CandidateName = llvm::sys::path::filename(DirIt->path());
    // If WDK is installed, there could be subfolders like "wdf" in the
    // "Include" directory.
    // Allow only directories which names start with "10.".
    if (!CandidateName.startswith("10."))
      continue;
    if (CandidateName > SDKVersion)
      SDKVersion = CandidateName;
  }

  return !SDKVersion.empty();
}

/// \brief Get Windows SDK installation directory.
static bool getWindowsSDKDir(std::string &Path, int &Major,
                             std::string &WindowsSDKIncludeVersion,
                             std::string &WindowsSDKLibVersion) {
  std::string RegistrySDKVersion;
  // Try the Windows registry.
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
      if (llvm::sys::fs::exists(TestPath.c_str())) {
        WindowsSDKLibVersion = Test;
        break;
      }
    }
    return !WindowsSDKLibVersion.empty();
  }
  if (Major == 10) {
    if (!getWindows10SDKVersionFromPath(Path, WindowsSDKIncludeVersion))
      return false;
    WindowsSDKLibVersion = WindowsSDKIncludeVersion;
    return true;
  }
  // Unsupported SDK version
  return false;
}

// Gets the library path required to link against the Windows SDK.
bool MSVCToolChain::getWindowsSDKLibraryPath(std::string &path) const {
  std::string sdkPath;
  int sdkMajor = 0;
  std::string windowsSDKIncludeVersion;
  std::string windowsSDKLibVersion;

  path.clear();
  if (!getWindowsSDKDir(sdkPath, sdkMajor, windowsSDKIncludeVersion,
                        windowsSDKLibVersion))
    return false;

  llvm::SmallString<128> libPath(sdkPath);
  llvm::sys::path::append(libPath, "Lib");
  if (sdkMajor >= 8) {
    llvm::sys::path::append(libPath,
      windowsSDKLibVersion,
      "um",
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

  path = libPath.str();
  return true;
}

// Check if the Include path of a specified version of Visual Studio contains
// specific header files. If not, they are probably shipped with Universal CRT.
bool MSVCToolChain::useUniversalCRT() const {
  llvm::SmallString<128> TestPath(getSubDirectoryPath(SubDirectoryType::Include));
  llvm::sys::path::append(TestPath, "stdlib.h");
  return !llvm::sys::fs::exists(TestPath);
}

static bool getUniversalCRTSdkDir(std::string &Path,
                                  std::string &UCRTVersion) {
  // vcvarsqueryregistry.bat for Visual Studio 2015 queries the registry
  // for the specific key "KitsRoot10". So do we.
  if (!getSystemRegistryString(
          "SOFTWARE\\Microsoft\\Windows Kits\\Installed Roots",
          "KitsRoot10", Path, nullptr))
    return false;

  return getWindows10SDKVersionFromPath(Path, UCRTVersion);
}

bool MSVCToolChain::getUniversalCRTLibraryPath(std::string &Path) const {
  std::string UniversalCRTSdkPath;
  std::string UCRTVersion;

  Path.clear();
  if (!getUniversalCRTSdkDir(UniversalCRTSdkPath, UCRTVersion))
    return false;

  StringRef ArchName = llvmArchToWindowsSDKArch(getArch());
  if (ArchName.empty())
    return false;

  llvm::SmallString<128> LibPath(UniversalCRTSdkPath);
  llvm::sys::path::append(LibPath, "Lib", UCRTVersion, "ucrt", ArchName);

  Path = LibPath.str();
  return true;
}

static VersionTuple getMSVCVersionFromTriple(const llvm::Triple &Triple) {
  unsigned Major, Minor, Micro;
  Triple.getEnvironmentVersion(Major, Minor, Micro);
  if (Major || Minor || Micro)
    return VersionTuple(Major, Minor, Micro);
  return VersionTuple();
}

static VersionTuple getMSVCVersionFromExe(const std::string &BinDir) {
  VersionTuple Version;
#ifdef USE_WIN32
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

  if (DriverArgs.hasArg(options::OPT_nostdlibinc))
    return;

  // Honor %INCLUDE%. It should know essential search paths with vcvarsall.bat.
  if (llvm::Optional<std::string> cl_include_dir =
          llvm::sys::Process::GetEnv("INCLUDE")) {
    SmallVector<StringRef, 8> Dirs;
    StringRef(*cl_include_dir)
        .split(Dirs, ";", /*MaxSplit=*/-1, /*KeepEmpty=*/false);
    for (StringRef Dir : Dirs)
      addSystemInclude(DriverArgs, CC1Args, Dir);
    if (!Dirs.empty())
      return;
  }

  // When built with access to the proper Windows APIs, try to actually find
  // the correct include paths first.
  if (!VCToolChainPath.empty()) {
    addSystemInclude(DriverArgs,
                     CC1Args,
                     getSubDirectoryPath(SubDirectoryType::Include));

    if (useUniversalCRT()) {
      std::string UniversalCRTSdkPath;
      std::string UCRTVersion;
      if (getUniversalCRTSdkDir(UniversalCRTSdkPath, UCRTVersion)) {
        AddSystemIncludeWithSubfolder(DriverArgs, CC1Args, UniversalCRTSdkPath,
                                      "Include", UCRTVersion, "ucrt");
      }
    }

    std::string WindowsSDKDir;
    int major;
    std::string windowsSDKIncludeVersion;
    std::string windowsSDKLibVersion;
    if (getWindowsSDKDir(WindowsSDKDir, major, windowsSDKIncludeVersion,
                         windowsSDKLibVersion)) {
      if (major >= 8) {
        // Note: windowsSDKIncludeVersion is empty for SDKs prior to v10.
        // Anyway, llvm::sys::path::append is able to manage it.
        AddSystemIncludeWithSubfolder(DriverArgs, CC1Args, WindowsSDKDir,
                                      "include", windowsSDKIncludeVersion,
                                      "shared");
        AddSystemIncludeWithSubfolder(DriverArgs, CC1Args, WindowsSDKDir,
                                      "include", windowsSDKIncludeVersion,
                                      "um");
        AddSystemIncludeWithSubfolder(DriverArgs, CC1Args, WindowsSDKDir,
                                      "include", windowsSDKIncludeVersion,
                                      "winrt");
      } else {
        AddSystemIncludeWithSubfolder(DriverArgs, CC1Args, WindowsSDKDir,
                                      "include");
      }
    }

    return;
  }

#if defined(LLVM_ON_WIN32)
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
    MSVT = getMSVCVersionFromTriple(getTriple());
  if (MSVT.empty() && IsWindowsMSVC)
    MSVT = getMSVCVersionFromExe(getSubDirectoryPath(SubDirectoryType::Bin));
  if (MSVT.empty() &&
      Args.hasFlag(options::OPT_fms_extensions, options::OPT_fno_ms_extensions,
                   IsWindowsMSVC)) {
    // -fms-compatibility-version=18.00 is default.
    // FIXME: Consider bumping this to 19 (MSVC2015) soon.
    MSVT = VersionTuple(18);
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
      if (&OptChar == ExpandChar) {
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
            DAL.AddFlagArg(A,
                           Opts.getOption(options::OPT_fomit_frame_pointer));
          if (OptChar == '1' || OptChar == '2')
            DAL.AddFlagArg(A,
                           Opts.getOption(options::OPT_ffunction_sections));
        }
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
        // Don't warn about /Oy- in 64-bit builds (where
        // SupportsForcingFramePointer is false).  The flag having no effect
        // there is a compiler-internal optimization, and people shouldn't have
        // to special-case their build files for 64-bit clang-cl.
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

  std::string NewVal = Val;
  NewVal[Hash] = '=';
  DAL.AddJoinedArg(A, Opts.getOption(options::OPT_D), NewVal);
}

llvm::opt::DerivedArgList *
MSVCToolChain::TranslateArgs(const llvm::opt::DerivedArgList &Args,
                             StringRef BoundArch, Action::OffloadKind) const {
  DerivedArgList *DAL = new DerivedArgList(Args.getBaseArgs());
  const OptTable &Opts = getDriver().getOpts();

  // /Oy and /Oy- only has an effect under X86-32.
  bool SupportsForcingFramePointer = getArch() == llvm::Triple::x86;

  // The -O[12xd] flag actually expands to several flags.  We must desugar the
  // flags so that options embedded can be negated.  For example, the '-O2' flag
  // enables '-Oy'.  Expanding '-O2' into its constituent flags allows us to
  // correctly handle '-O2 -Oy-' where the trailing '-Oy-' disables a single
  // aspect of '-O2'.
  //
  // Note that this expansion logic only applies to the *last* of '[12xd]'.

  // First step is to search for the character we'd like to expand.
  const char *ExpandChar = nullptr;
  for (Arg *A : Args) {
    if (!A->getOption().matches(options::OPT__SLASH_O))
      continue;
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
    } else {
      DAL->append(A);
    }
  }

  return DAL;
}
