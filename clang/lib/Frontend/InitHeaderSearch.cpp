//===--- InitHeaderSearch.cpp - Initialize header search paths ----------*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the InitHeaderSearch class.
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/Utils.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Frontend/HeaderSearchOptions.h"
#include "clang/Lex/HeaderSearch.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Triple.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/System/Path.h"
#include "llvm/Config/config.h"
#ifdef _MSC_VER
  #define WIN32_LEAN_AND_MEAN 1
  #include <windows.h>
#endif
using namespace clang;
using namespace clang::frontend;

namespace {

/// InitHeaderSearch - This class makes it easier to set the search paths of
///  a HeaderSearch object. InitHeaderSearch stores several search path lists
///  internally, which can be sent to a HeaderSearch object in one swoop.
class InitHeaderSearch {
  std::vector<DirectoryLookup> IncludeGroup[4];
  HeaderSearch& Headers;
  bool Verbose;
  std::string isysroot;

public:

  InitHeaderSearch(HeaderSearch &HS,
      bool verbose = false, const std::string &iSysroot = "")
    : Headers(HS), Verbose(verbose), isysroot(iSysroot) {}

  /// AddPath - Add the specified path to the specified group list.
  void AddPath(const llvm::Twine &Path, IncludeDirGroup Group,
               bool isCXXAware, bool isUserSupplied,
               bool isFramework, bool IgnoreSysRoot = false);

  /// AddGnuCPlusPlusIncludePaths - Add the necessary paths to suport a gnu
  ///  libstdc++.
  void AddGnuCPlusPlusIncludePaths(llvm::StringRef Base,
                                   llvm::StringRef ArchDir,
                                   llvm::StringRef Dir32,
                                   llvm::StringRef Dir64,
                                   const llvm::Triple &triple);

  /// AddMinGWCPlusPlusIncludePaths - Add the necessary paths to suport a MinGW
  ///  libstdc++.
  void AddMinGWCPlusPlusIncludePaths(llvm::StringRef Base,
                                     llvm::StringRef Arch,
                                     llvm::StringRef Version);

  /// AddDelimitedPaths - Add a list of paths delimited by the system PATH
  /// separator. The processing follows that of the CPATH variable for gcc.
  void AddDelimitedPaths(llvm::StringRef String);

  // AddDefaultCIncludePaths - Add paths that should always be searched.
  void AddDefaultCIncludePaths(const llvm::Triple &triple);

  // AddDefaultCPlusPlusIncludePaths -  Add paths that should be searched when
  //  compiling c++.
  void AddDefaultCPlusPlusIncludePaths(const llvm::Triple &triple);

  /// AddDefaultSystemIncludePaths - Adds the default system include paths so
  ///  that e.g. stdio.h is found.
  void AddDefaultSystemIncludePaths(const LangOptions &Lang,
                                    const llvm::Triple &triple);

  /// Realize - Merges all search path lists into one list and send it to
  /// HeaderSearch.
  void Realize();
};

}

void InitHeaderSearch::AddPath(const llvm::Twine &Path,
                               IncludeDirGroup Group, bool isCXXAware,
                               bool isUserSupplied, bool isFramework,
                               bool IgnoreSysRoot) {
  assert(!Path.isTriviallyEmpty() && "can't handle empty path here");
  FileManager &FM = Headers.getFileMgr();

  // Compute the actual path, taking into consideration -isysroot.
  llvm::SmallString<256> MappedPathStr;
  llvm::raw_svector_ostream MappedPath(MappedPathStr);

  // Handle isysroot.
  if (Group == System && !IgnoreSysRoot) {
    // FIXME: Portability.  This should be a sys::Path interface, this doesn't
    // handle things like C:\ right, nor win32 \\network\device\blah.
    if (isysroot.size() != 1 || isysroot[0] != '/') // Add isysroot if present.
      MappedPath << isysroot;
  }

  Path.print(MappedPath);

  // Compute the DirectoryLookup type.
  SrcMgr::CharacteristicKind Type;
  if (Group == Quoted || Group == Angled)
    Type = SrcMgr::C_User;
  else if (isCXXAware)
    Type = SrcMgr::C_System;
  else
    Type = SrcMgr::C_ExternCSystem;


  // If the directory exists, add it.
  if (const DirectoryEntry *DE = FM.getDirectory(MappedPath.str())) {
    IncludeGroup[Group].push_back(DirectoryLookup(DE, Type, isUserSupplied,
                                                  isFramework));
    return;
  }

  // Check to see if this is an apple-style headermap (which are not allowed to
  // be frameworks).
  if (!isFramework) {
    if (const FileEntry *FE = FM.getFile(MappedPath.str())) {
      if (const HeaderMap *HM = Headers.CreateHeaderMap(FE)) {
        // It is a headermap, add it to the search path.
        IncludeGroup[Group].push_back(DirectoryLookup(HM, Type,isUserSupplied));
        return;
      }
    }
  }

  if (Verbose)
    llvm::errs() << "ignoring nonexistent directory \""
                 << MappedPath.str() << "\"\n";
}


void InitHeaderSearch::AddDelimitedPaths(llvm::StringRef at) {
  if (at.empty()) // Empty string should not add '.' path.
    return;

  llvm::StringRef::size_type delim;
  while ((delim = at.find(llvm::sys::PathSeparator)) != llvm::StringRef::npos) {
    if (delim == 0)
      AddPath(".", Angled, false, true, false);
    else
      AddPath(at.substr(0, delim), Angled, false, true, false);
    at = at.substr(delim + 1);
  }

  if (at.empty())
    AddPath(".", Angled, false, true, false);
  else
    AddPath(at, Angled, false, true, false);
}

void InitHeaderSearch::AddGnuCPlusPlusIncludePaths(llvm::StringRef Base,
                                                   llvm::StringRef ArchDir,
                                                   llvm::StringRef Dir32,
                                                   llvm::StringRef Dir64,
                                                   const llvm::Triple &triple) {
  // Add the base dir
  AddPath(Base, System, true, false, false);

  // Add the multilib dirs
  llvm::Triple::ArchType arch = triple.getArch();
  bool is64bit = arch == llvm::Triple::ppc64 || arch == llvm::Triple::x86_64;
  if (is64bit)
    AddPath(Base + "/" + ArchDir + "/" + Dir64, System, true, false, false);
  else
    AddPath(Base + "/" + ArchDir + "/" + Dir32, System, true, false, false);

  // Add the backward dir
  AddPath(Base + "/backward", System, true, false, false);
}

void InitHeaderSearch::AddMinGWCPlusPlusIncludePaths(llvm::StringRef Base,
                                                     llvm::StringRef Arch,
                                                     llvm::StringRef Version) {
  llvm::Twine localBase = Base + "/" + Arch + "/" + Version + "/include";
  AddPath(localBase, System, true, false, false);
  AddPath(localBase + "/c++", System, true, false, false);
  AddPath(localBase + "/c++/backward", System, true, false, false);
}

  // FIXME: This probably should goto to some platform utils place.
#ifdef _MSC_VER

  // Read registry string.
  // This also supports a means to look for high-versioned keys by use
  // of a $VERSION placeholder in the key path.
  // $VERSION in the key path is a placeholder for the version number,
  // causing the highest value path to be searched for and used.
  // I.e. "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\VisualStudio\\$VERSION".
  // There can be additional characters in the component.  Only the numberic
  // characters are compared.
bool getSystemRegistryString(const char *keyPath, const char *valueName,
                       char *value, size_t maxLength) {
  HKEY hRootKey = NULL;
  HKEY hKey = NULL;
  const char* subKey = NULL;
  DWORD valueType;
  DWORD valueSize = maxLength - 1;
  long lResult;
  bool returnValue = false;
  if (strncmp(keyPath, "HKEY_CLASSES_ROOT\\", 18) == 0) {
    hRootKey = HKEY_CLASSES_ROOT;
    subKey = keyPath + 18;
  }
  else if (strncmp(keyPath, "HKEY_USERS\\", 11) == 0) {
    hRootKey = HKEY_USERS;
    subKey = keyPath + 11;
  }
  else if (strncmp(keyPath, "HKEY_LOCAL_MACHINE\\", 19) == 0) {
    hRootKey = HKEY_LOCAL_MACHINE;
    subKey = keyPath + 19;
  }
  else if (strncmp(keyPath, "HKEY_CURRENT_USER\\", 18) == 0) {
    hRootKey = HKEY_CURRENT_USER;
    subKey = keyPath + 18;
  }
  else
    return(false);
  const char *placeHolder = strstr(subKey, "$VERSION");
  char bestName[256];
  bestName[0] = '\0';
  // If we have a $VERSION placeholder, do the highest-version search.
  if (placeHolder) {
    const char *keyEnd = placeHolder - 1;
    const char *nextKey = placeHolder;
    // Find end of previous key.
    while ((keyEnd > subKey) && (*keyEnd != '\\'))
      keyEnd--;
    // Find end of key containing $VERSION.
    while (*nextKey && (*nextKey != '\\'))
      nextKey++;
    size_t partialKeyLength = keyEnd - subKey;
    char partialKey[256];
    if (partialKeyLength > sizeof(partialKey))
      partialKeyLength = sizeof(partialKey);
    strncpy(partialKey, subKey, partialKeyLength);
    partialKey[partialKeyLength] = '\0';
    HKEY hTopKey = NULL;
    lResult = RegOpenKeyEx(hRootKey, partialKey, 0, KEY_READ, &hTopKey);
    if (lResult == ERROR_SUCCESS) {
      char keyName[256];
      int bestIndex = -1;
      double bestValue = 0.0;
      DWORD index, size = sizeof(keyName) - 1;
      for (index = 0; RegEnumKeyEx(hTopKey, index, keyName, &size, NULL,
          NULL, NULL, NULL) == ERROR_SUCCESS; index++) {
        const char *sp = keyName;
        while (*sp && !isdigit(*sp))
          sp++;
        if (!*sp)
          continue;
        const char *ep = sp + 1;
        while (*ep && (isdigit(*ep) || (*ep == '.')))
          ep++;
        char numBuf[32];
        strncpy(numBuf, sp, sizeof(numBuf) - 1);
        numBuf[sizeof(numBuf) - 1] = '\0';
        double value = strtod(numBuf, NULL);
        if (value > bestValue) {
          bestIndex = (int)index;
          bestValue = value;
          strcpy(bestName, keyName);
        }
        size = sizeof(keyName) - 1;
      }
      // If we found the highest versioned key, open the key and get the value.
      if (bestIndex != -1) {
        // Append rest of key.
        strncat(bestName, nextKey, sizeof(bestName) - 1);
        bestName[sizeof(bestName) - 1] = '\0';
        // Open the chosen key path remainder.
        lResult = RegOpenKeyEx(hTopKey, bestName, 0, KEY_READ, &hKey);
        if (lResult == ERROR_SUCCESS) {
          lResult = RegQueryValueEx(hKey, valueName, NULL, &valueType,
            (LPBYTE)value, &valueSize);
          if (lResult == ERROR_SUCCESS)
            returnValue = true;
          RegCloseKey(hKey);
        }
      }
      RegCloseKey(hTopKey);
    }
  }
  else {
    lResult = RegOpenKeyEx(hRootKey, subKey, 0, KEY_READ, &hKey);
    if (lResult == ERROR_SUCCESS) {
      lResult = RegQueryValueEx(hKey, valueName, NULL, &valueType,
        (LPBYTE)value, &valueSize);
      if (lResult == ERROR_SUCCESS)
        returnValue = true;
      RegCloseKey(hKey);
    }
  }
  return(returnValue);
}
#else // _MSC_VER
  // Read registry string.
bool getSystemRegistryString(const char *, const char *, char *, size_t) {
  return(false);
}
#endif // _MSC_VER

  // Get Visual Studio installation directory.
bool getVisualStudioDir(std::string &path) {
  char vsIDEInstallDir[256];
  // Try the Windows registry first.
  bool hasVCDir = getSystemRegistryString(
    "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\VisualStudio\\$VERSION",
    "InstallDir", vsIDEInstallDir, sizeof(vsIDEInstallDir) - 1);
    // If we have both vc80 and vc90, pick version we were compiled with. 
  if (hasVCDir && vsIDEInstallDir[0]) {
    char *p = (char*)strstr(vsIDEInstallDir, "\\Common7\\IDE");
    if (p)
      *p = '\0';
    path = vsIDEInstallDir;
    return(true);
  }
  else {
    // Try the environment.
    const char* vs90comntools = getenv("VS90COMNTOOLS");
    const char* vs80comntools = getenv("VS80COMNTOOLS");
    const char* vscomntools = NULL;
      // If we have both vc80 and vc90, pick version we were compiled with. 
    if (vs90comntools && vs80comntools) {
      #if (_MSC_VER >= 1500)  // VC90
          vscomntools = vs90comntools;
      #elif (_MSC_VER == 1400) // VC80
          vscomntools = vs80comntools;
      #else
          vscomntools = vs90comntools;
      #endif
    }
    else if (vs90comntools)
      vscomntools = vs90comntools;
    else if (vs80comntools)
      vscomntools = vs80comntools;
    if (vscomntools && *vscomntools) {
      char *p = (char*)strstr(vscomntools, "\\Common7\\Tools");
      if (p)
        *p = '\0';
      path = vscomntools;
      return(true);
    }
    else
      return(false);
  }
  return(false);
}

  // Get Windows SDK installation directory.
bool getWindowsSDKDir(std::string &path) {
  char windowsSDKInstallDir[256];
  // Try the Windows registry.
  bool hasSDKDir = getSystemRegistryString(
   "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Microsoft SDKs\\Windows\\$VERSION",
    "InstallationFolder", windowsSDKInstallDir, sizeof(windowsSDKInstallDir) - 1);
    // If we have both vc80 and vc90, pick version we were compiled with. 
  if (hasSDKDir && windowsSDKInstallDir[0]) {
    path = windowsSDKInstallDir;
    return(true);
  }
  return(false);
}

void InitHeaderSearch::AddDefaultCIncludePaths(const llvm::Triple &triple) {
  // FIXME: temporary hack: hard-coded paths.
  llvm::StringRef CIncludeDirs(C_INCLUDE_DIRS);
  if (CIncludeDirs != "") {
    llvm::SmallVector<llvm::StringRef, 5> dirs;
    CIncludeDirs.split(dirs, ":");
    for (llvm::SmallVectorImpl<llvm::StringRef>::iterator i = dirs.begin();
         i != dirs.end();
         ++i) 
      AddPath(*i, System, false, false, false);
    return;
  }
  llvm::Triple::OSType os = triple.getOS();
  switch (os) {
  case llvm::Triple::Win32:
    {
      std::string VSDir;
      std::string WindowsSDKDir;
      if (getVisualStudioDir(VSDir)) {
        AddPath(VSDir + "\\VC\\include", System, false, false, false);
        if (getWindowsSDKDir(WindowsSDKDir))
          AddPath(WindowsSDKDir, System, false, false, false);
        else
          AddPath(VSDir + "\\VC\\PlatformSDK\\Include",
            System, false, false, false);
      }
      else {
          // Default install paths.
        AddPath("C:/Program Files/Microsoft Visual Studio 9.0/VC/include",
          System, false, false, false);
        AddPath(
        "C:/Program Files/Microsoft Visual Studio 9.0/VC/PlatformSDK/Include",
          System, false, false, false);
        AddPath("C:/Program Files/Microsoft Visual Studio 8/VC/include",
          System, false, false, false);
        AddPath(
        "C:/Program Files/Microsoft Visual Studio 8/VC/PlatformSDK/Include",
          System, false, false, false);
          // For some clang developers.
        AddPath("G:/Program Files/Microsoft Visual Studio 9.0/VC/include",
          System, false, false, false);
        AddPath(
        "G:/Program Files/Microsoft Visual Studio 9.0/VC/PlatformSDK/Include",
          System, false, false, false);
      }
    }
    break;
  case llvm::Triple::MinGW64:
  case llvm::Triple::MinGW32:
    AddPath("c:/mingw/include", System, true, false, false);
    break;
  default:
    break;
  }

  AddPath("/usr/local/include", System, false, false, false);
  AddPath("/usr/include", System, false, false, false);
}

void InitHeaderSearch::AddDefaultCPlusPlusIncludePaths(const llvm::Triple &triple) {
  llvm::Triple::OSType os = triple.getOS();
  llvm::StringRef CxxIncludeRoot(CXX_INCLUDE_ROOT);
  if (CxxIncludeRoot != "") {
    llvm::StringRef CxxIncludeArch(CXX_INCLUDE_ARCH);
    if (CxxIncludeArch == "")
      AddGnuCPlusPlusIncludePaths(CxxIncludeRoot, triple.str().c_str(),
                                  CXX_INCLUDE_32BIT_DIR, CXX_INCLUDE_64BIT_DIR, triple);
    else
      AddGnuCPlusPlusIncludePaths(CxxIncludeRoot, CXX_INCLUDE_ARCH,
                                  CXX_INCLUDE_32BIT_DIR, CXX_INCLUDE_64BIT_DIR, triple);
    return;
  }
  // FIXME: temporary hack: hard-coded paths.
  switch (os) {
  case llvm::Triple::Cygwin:
    AddPath("/lib/gcc/i686-pc-cygwin/3.4.4/include",
        System, true, false, false);
    AddPath("/lib/gcc/i686-pc-cygwin/3.4.4/include/c++",
        System, true, false, false);
    break;
  case llvm::Triple::MinGW64:
    // Try gcc 4.4.0
    AddMinGWCPlusPlusIncludePaths("c:/MinGW/lib/gcc", "mingw64", "4.4.0");
    // Try gcc 4.3.0
    AddMinGWCPlusPlusIncludePaths("c:/MinGW/lib/gcc", "mingw64", "4.3.0");
    // Fall through.
  case llvm::Triple::MinGW32:
    // Try gcc 4.4.0
    AddMinGWCPlusPlusIncludePaths("c:/MinGW/lib/gcc", "mingw32", "4.4.0");
    // Try gcc 4.3.0
    AddMinGWCPlusPlusIncludePaths("c:/MinGW/lib/gcc", "mingw32", "4.3.0");
    break;
  case llvm::Triple::Darwin:
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.2.1",
                                "i686-apple-darwin10", "", "x86_64", triple);
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.0.0",
                                "i686-apple-darwin8", "", "", triple);
    break;
  case llvm::Triple::Linux:
    // Ubuntu 7.10 - Gutsy Gibbon
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.1.3",
                                "i486-linux-gnu", "", "", triple);
    // Ubuntu 9.04
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.3.3",
                                "x86_64-linux-gnu","32", "", triple);
    // Ubuntu 9.10
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.4.1",
                                "x86_64-linux-gnu", "32", "", triple);
    // Fedora 8
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.1.2",
                                "i386-redhat-linux", "", "", triple);
    // Fedora 9
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.3.0",
                                "i386-redhat-linux", "", "", triple);
    // Fedora 10
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.3.2",
                                "i386-redhat-linux","", "", triple);

    // Fedora 11
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.4.1",
                                "i586-redhat-linux","", "", triple);

    // Fedora 12
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.4.2",
                                "i686-redhat-linux","", "", triple);

    // openSUSE 11.1 32 bit
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.3",
                                "i586-suse-linux", "", "", triple);
    // openSUSE 11.1 64 bit
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.3",
                                "x86_64-suse-linux", "32", "", triple);
    // openSUSE 11.2
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.4",
                                "i586-suse-linux", "", "", triple);
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.4",
                                "x86_64-suse-linux", "", "", triple);
    // Arch Linux 2008-06-24
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.3.1",
                                "i686-pc-linux-gnu", "", "", triple);
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.3.1",
                                "x86_64-unknown-linux-gnu", "", "", triple);
    // Gentoo x86 2009.1 stable
    AddGnuCPlusPlusIncludePaths(
      "/usr/lib/gcc/i686-pc-linux-gnu/4.3.4/include/g++-v4",
      "i686-pc-linux-gnu", "", "", triple);
    // Gentoo x86 2009.0 stable
    AddGnuCPlusPlusIncludePaths(
      "/usr/lib/gcc/i686-pc-linux-gnu/4.3.2/include/g++-v4",
      "i686-pc-linux-gnu", "", "", triple);
    // Gentoo x86 2008.0 stable
    AddGnuCPlusPlusIncludePaths(
      "/usr/lib/gcc/i686-pc-linux-gnu/4.1.2/include/g++-v4",
      "i686-pc-linux-gnu", "", "", triple);
    // Ubuntu 8.10
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.3",
                                "i486-pc-linux-gnu", "", "", triple);
    // Ubuntu 9.04
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.3",
                                "i486-linux-gnu","", "", triple);
    // Gentoo amd64 stable
    AddGnuCPlusPlusIncludePaths(
        "/usr/lib/gcc/x86_64-pc-linux-gnu/4.1.2/include/g++-v4",
        "i686-pc-linux-gnu", "", "", triple);
    // Exherbo (2009-10-26)
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.4.2",
                                "x86_64-pc-linux-gnu", "32", "", triple);
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.4.2",
                                "i686-pc-linux-gnu", "", "", triple);
    break;
  case llvm::Triple::FreeBSD:
    // DragonFly
    AddPath("/usr/include/c++/4.1", System, true, false, false);
    // FreeBSD
    AddPath("/usr/include/c++/4.2", System, true, false, false);
    break;
  case llvm::Triple::Solaris:
    // Solaris - Fall though..
  case llvm::Triple::AuroraUX:
    // AuroraUX
    AddGnuCPlusPlusIncludePaths("/opt/gcc4/include/c++/4.2.4",
                                "i386-pc-solaris2.11", "", "", triple);
    break;
  default:
    break;
  }
}

void InitHeaderSearch::AddDefaultSystemIncludePaths(const LangOptions &Lang,
                                                    const llvm::Triple &triple) {
  if (Lang.CPlusPlus)
    AddDefaultCPlusPlusIncludePaths(triple);

  AddDefaultCIncludePaths(triple);

  // Add the default framework include paths on Darwin.
  if (triple.getOS() == llvm::Triple::Darwin) {
    AddPath("/System/Library/Frameworks", System, true, false, true);
    AddPath("/Library/Frameworks", System, true, false, true);
  }
}

/// RemoveDuplicates - If there are duplicate directory entries in the specified
/// search list, remove the later (dead) ones.
static void RemoveDuplicates(std::vector<DirectoryLookup> &SearchList,
                             bool Verbose) {
  llvm::SmallPtrSet<const DirectoryEntry *, 8> SeenDirs;
  llvm::SmallPtrSet<const DirectoryEntry *, 8> SeenFrameworkDirs;
  llvm::SmallPtrSet<const HeaderMap *, 8> SeenHeaderMaps;
  for (unsigned i = 0; i != SearchList.size(); ++i) {
    unsigned DirToRemove = i;

    const DirectoryLookup &CurEntry = SearchList[i];

    if (CurEntry.isNormalDir()) {
      // If this isn't the first time we've seen this dir, remove it.
      if (SeenDirs.insert(CurEntry.getDir()))
        continue;
    } else if (CurEntry.isFramework()) {
      // If this isn't the first time we've seen this framework dir, remove it.
      if (SeenFrameworkDirs.insert(CurEntry.getFrameworkDir()))
        continue;
    } else {
      assert(CurEntry.isHeaderMap() && "Not a headermap or normal dir?");
      // If this isn't the first time we've seen this headermap, remove it.
      if (SeenHeaderMaps.insert(CurEntry.getHeaderMap()))
        continue;
    }

    // If we have a normal #include dir/framework/headermap that is shadowed
    // later in the chain by a system include location, we actually want to
    // ignore the user's request and drop the user dir... keeping the system
    // dir.  This is weird, but required to emulate GCC's search path correctly.
    //
    // Since dupes of system dirs are rare, just rescan to find the original
    // that we're nuking instead of using a DenseMap.
    if (CurEntry.getDirCharacteristic() != SrcMgr::C_User) {
      // Find the dir that this is the same of.
      unsigned FirstDir;
      for (FirstDir = 0; ; ++FirstDir) {
        assert(FirstDir != i && "Didn't find dupe?");

        const DirectoryLookup &SearchEntry = SearchList[FirstDir];

        // If these are different lookup types, then they can't be the dupe.
        if (SearchEntry.getLookupType() != CurEntry.getLookupType())
          continue;

        bool isSame;
        if (CurEntry.isNormalDir())
          isSame = SearchEntry.getDir() == CurEntry.getDir();
        else if (CurEntry.isFramework())
          isSame = SearchEntry.getFrameworkDir() == CurEntry.getFrameworkDir();
        else {
          assert(CurEntry.isHeaderMap() && "Not a headermap or normal dir?");
          isSame = SearchEntry.getHeaderMap() == CurEntry.getHeaderMap();
        }

        if (isSame)
          break;
      }

      // If the first dir in the search path is a non-system dir, zap it
      // instead of the system one.
      if (SearchList[FirstDir].getDirCharacteristic() == SrcMgr::C_User)
        DirToRemove = FirstDir;
    }

    if (Verbose) {
      llvm::errs() << "ignoring duplicate directory \""
                   << CurEntry.getName() << "\"\n";
      if (DirToRemove != i)
        llvm::errs() << "  as it is a non-system directory that duplicates "
                     << "a system directory\n";
    }

    // This is reached if the current entry is a duplicate.  Remove the
    // DirToRemove (usually the current dir).
    SearchList.erase(SearchList.begin()+DirToRemove);
    --i;
  }
}


void InitHeaderSearch::Realize() {
  // Concatenate ANGLE+SYSTEM+AFTER chains together into SearchList.
  std::vector<DirectoryLookup> SearchList;
  SearchList = IncludeGroup[Angled];
  SearchList.insert(SearchList.end(), IncludeGroup[System].begin(),
                    IncludeGroup[System].end());
  SearchList.insert(SearchList.end(), IncludeGroup[After].begin(),
                    IncludeGroup[After].end());
  RemoveDuplicates(SearchList, Verbose);
  RemoveDuplicates(IncludeGroup[Quoted], Verbose);

  // Prepend QUOTED list on the search list.
  SearchList.insert(SearchList.begin(), IncludeGroup[Quoted].begin(),
                    IncludeGroup[Quoted].end());


  bool DontSearchCurDir = false;  // TODO: set to true if -I- is set?
  Headers.SetSearchPaths(SearchList, IncludeGroup[Quoted].size(),
                         DontSearchCurDir);

  // If verbose, print the list of directories that will be searched.
  if (Verbose) {
    llvm::errs() << "#include \"...\" search starts here:\n";
    unsigned QuotedIdx = IncludeGroup[Quoted].size();
    for (unsigned i = 0, e = SearchList.size(); i != e; ++i) {
      if (i == QuotedIdx)
        llvm::errs() << "#include <...> search starts here:\n";
      const char *Name = SearchList[i].getName();
      const char *Suffix;
      if (SearchList[i].isNormalDir())
        Suffix = "";
      else if (SearchList[i].isFramework())
        Suffix = " (framework directory)";
      else {
        assert(SearchList[i].isHeaderMap() && "Unknown DirectoryLookup");
        Suffix = " (headermap)";
      }
      llvm::errs() << " " << Name << Suffix << "\n";
    }
    llvm::errs() << "End of search list.\n";
  }
}

void clang::ApplyHeaderSearchOptions(HeaderSearch &HS,
                                     const HeaderSearchOptions &HSOpts,
                                     const LangOptions &Lang,
                                     const llvm::Triple &Triple) {
  InitHeaderSearch Init(HS, HSOpts.Verbose, HSOpts.Sysroot);

  // Add the user defined entries.
  for (unsigned i = 0, e = HSOpts.UserEntries.size(); i != e; ++i) {
    const HeaderSearchOptions::Entry &E = HSOpts.UserEntries[i];
    Init.AddPath(E.Path, E.Group, false, E.IsUserSupplied, E.IsFramework,
                 false);
  }

  // Add entries from CPATH and friends.
  Init.AddDelimitedPaths(HSOpts.EnvIncPath);
  if (Lang.CPlusPlus && Lang.ObjC1)
    Init.AddDelimitedPaths(HSOpts.ObjCXXEnvIncPath);
  else if (Lang.CPlusPlus)
    Init.AddDelimitedPaths(HSOpts.CXXEnvIncPath);
  else if (Lang.ObjC1)
    Init.AddDelimitedPaths(HSOpts.ObjCEnvIncPath);
  else
    Init.AddDelimitedPaths(HSOpts.CEnvIncPath);

  if (HSOpts.UseBuiltinIncludes) {
    // Ignore the sys root, we *always* look for clang headers relative to
    // supplied path.
    llvm::sys::Path P(HSOpts.ResourceDir);
    P.appendComponent("include");
    Init.AddPath(P.str(), System, false, false, false, /*IgnoreSysRoot=*/ true);
  }

  if (HSOpts.UseStandardIncludes)
    Init.AddDefaultSystemIncludePaths(Lang, Triple);

  Init.Realize();
}
