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

#include "clang/Frontend/InitHeaderSearch.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/LangOptions.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/System/Path.h"
#include "llvm/Config/config.h"
#include <cstdio>
using namespace clang;

void InitHeaderSearch::AddPath(const llvm::StringRef &Path,
                               IncludeDirGroup Group, bool isCXXAware,
                               bool isUserSupplied, bool isFramework,
                               bool IgnoreSysRoot) {
  assert(!Path.empty() && "can't handle empty path here");
  FileManager &FM = Headers.getFileMgr();

  // Compute the actual path, taking into consideration -isysroot.
  llvm::SmallString<256> MappedPath;

  // Handle isysroot.
  if (Group == System && !IgnoreSysRoot) {
    // FIXME: Portability.  This should be a sys::Path interface, this doesn't
    // handle things like C:\ right, nor win32 \\network\device\blah.
    if (isysroot.size() != 1 || isysroot[0] != '/') // Add isysroot if present.
      MappedPath.append(isysroot.begin(), isysroot.end());
  }

  MappedPath.append(Path.begin(), Path.end());

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


void InitHeaderSearch::AddEnvVarPaths(const char *Name) {
  const char* at = getenv(Name);
  if (!at || *at == 0) // Empty string should not add '.' path.
    return;

  const char* delim = strchr(at, llvm::sys::PathSeparator);
  while (delim != 0) {
    if (delim-at == 0)
      AddPath(".", Angled, false, true, false);
    else
      AddPath(llvm::StringRef(at, delim-at), Angled, false, true, false);
    at = delim + 1;
    delim = strchr(at, llvm::sys::PathSeparator);
  }
  if (*at == 0)
    AddPath(".", Angled, false, true, false);
  else
    AddPath(at, Angled, false, true, false);
}

void InitHeaderSearch::AddGnuCPlusPlusIncludePaths(std::string base,
                                                   std::string arch) {
    AddPath(base, System, true, false, false);
    AddPath(base + "/" + arch, System, true, false, false);
    AddPath(base + "/backward", System, true, false, false);
}

#if defined(LLVM_ON_WIN32)

#if 0 // Yikes!  Can't include windows.h.
  #if LLVM_ON_WIN32
    #define WIN32_LEAN_AND_MEAN 1
    #include <windows.h>
  #endif

  // Read Windows registry string.
bool getWindowsRegistryString(const char *keyPath, const char *valueName,
                       char *value, size_t maxLength) {
  HKEY hRootKey = NULL;
  HKEY hKey = NULL;
  const char* subKey = NULL;
  DWORD valueType;
  DWORD valueSize = maxLength - 1;
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
  long lResult = RegOpenKeyEx(hRootKey, subKey, 0, KEY_READ, &hKey);
  if (lResult == ERROR_SUCCESS) {
    lResult = RegQueryValueEx(hKey, valueName, NULL, &valueType, (LPBYTE)value,
                              &valueSize);
    if (lResult == ERROR_SUCCESS)
      returnValue = true;
    RegCloseKey(kKey);
  }
  return(returnValue);
}

  // Get Visual Studio installation directory.
bool getVisualStudioDir(std::string &path) {
  char vs80comntools[256];
  char vs90comntools[256];
  const char* vscomntools = NULL;
  bool has80 = getWindowsRegistryString(
    "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\VisualStudio\\8.0",
    "InstallDir", vs80comntools, sizeof(vs80comntools) - 1);
  bool has90 = getWindowsRegistryString(
    "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\VisualStudio\\9.0",
    "InstallDir", vs90comntools, sizeof(vs90comntools) - 1);
    // If we have both vc80 and vc90, pick version we were compiled with. 
  if (has80 && has90) {
    #ifdef _MSC_VER
      #if (_MSC_VER >= 1500)  // VC90
          vscomntools = vs90comntools;
      #elif (_MSC_VER == 1400) // VC80
          vscomntools = vs80comntools;
      #else
          vscomntools = vs90comntools;
      #endif
    #else
      vscomntools = vs90comntools;
    #endif
  }
  else if (has90)
    vscomntools = vs90comntools;
  else if (has80)
    vscomntools = vs80comntools;
  else
    return(false);
  char *p = strstr(vscomntools, "\\Common7\\ide");
  if (p)
    *p = '\0';
  path = vscomntools;
  return(true);
}
#else

  // Get Visual Studio installation directory.
bool getVisualStudioDir(std::string &path) {
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
  else
    return(false);
  char *p = (char*)strstr(vscomntools, "\\Common7\\Tools");
  if (p)
    *p = '\0';
  path = vscomntools;
  return(true);
}
#endif

#endif // LLVM_ON_WIN32

void InitHeaderSearch::AddDefaultSystemIncludePaths(const LangOptions &Lang,
                                               const llvm::Triple &triple) {
  // FIXME: temporary hack: hard-coded paths.
  llvm::Triple::OSType os = triple.getOS();

  switch (os) {
  case llvm::Triple::Win32:
    {
      #if defined(_MSC_VER)
        std::string VSDir;
        if (getVisualStudioDir(VSDir)) {
          VSDir += "\\VC\\include";
          AddPath(VSDir, System, false, false, false);
        }
        else {
            // Default install paths.
          AddPath("C:\\Program Files\\Microsoft Visual Studio 9.0\\VC\\include",
            System, false, false, false);
          AddPath("C:\\Program Files\\Microsoft Visual Studio 8\\VC\\include",
            System, false, false, false);
            // For some clang developers.
          AddPath("G:\\Program Files\\Microsoft Visual Studio 9.0\\VC\\include",
            System, false, false, false);
        }
      #else
          // Default install paths.
        AddPath("/Program Files/Microsoft Visual Studio 9.0/VC/include",
          System, false, false, false);
        AddPath("/Program Files/Microsoft Visual Studio 8/VC/include",
          System, false, false, false);
      #endif
    }
    break;
  case llvm::Triple::Cygwin:
    if (Lang.CPlusPlus) {
      AddPath("/lib/gcc/i686-pc-cygwin/3.4.4/include", System, false, false,
              false);
      AddPath("/lib/gcc/i686-pc-cygwin/3.4.4/include/c++", System, false, false,
              false);
    }
    AddPath("/usr/include", System, false, false, false);
    break;
  case llvm::Triple::MinGW32:
  case llvm::Triple::MinGW64:
    if (Lang.CPlusPlus) {
      // Try gcc 4.4.0
      // FIXME: This can just use AddGnuCPlusPlusIncludePaths, right?
      AddPath("c:/mingw/lib/gcc/mingw32/4.4.0/include/c++",
              System, true, false, false);
      AddPath("c:/mingw/lib/gcc/mingw32/4.4.0/include/c++/mingw32",
              System, true, false, false);
      AddPath("c:/mingw/lib/gcc/mingw32/4.4.0/include/c++/backward",
              System, true, false, false);
      // Try gcc 4.3.0
      // FIXME: This can just use AddGnuCPlusPlusIncludePaths, right?
      AddPath("c:/mingw/lib/gcc/mingw32/4.3.0/include/c++",
              System, true, false, false);
      AddPath("c:/mingw/lib/gcc/mingw32/4.3.0/include/c++/mingw32",
              System, true, false, false);
      AddPath("c:/mingw/lib/gcc/mingw32/4.3.0/include/c++/backward",
              System, true, false, false);
    }
    AddPath("c:/mingw/include", System, true, false, false);
    break;
  default:
    if (Lang.CPlusPlus) {
      switch (os) {
        case llvm::Triple::Darwin:
          AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.2.1",
                                      "i686-apple-darwin10");
          AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.0.0",
                                      "i686-apple-darwin8");
          break;
        case llvm::Triple::Linux:
          // Ubuntu 7.10 - Gutsy Gibbon
          AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.1.3",
                                      "i486-linux-gnu");
          // Ubuntu 9.04
          AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.3.3",
                                      "x86_64-linux-gnu");
          // Fedora 8
          AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.1.2",
                                      "i386-redhat-linux");
          // Fedora 9
          AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.3.0",
                                      "i386-redhat-linux");
          // Fedora 10
          AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.3.2",
                                      "i386-redhat-linux");
          // openSUSE 11.1
          AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.3",
                                      "i586-suse-linux");
          AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.3",
                                      "x86_64-suse-linux");
          // openSUSE 11.2
          AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.4",
                                      "i586-suse-linux");
          AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.4",
                                      "x86_64-suse-linux");
          // Arch Linux 2008-06-24
          AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.3.1",
                                      "i686-pc-linux-gnu");
          AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.3.1",
                                      "x86_64-unknown-linux-gnu");
          // Gentoo x86 2009.0 stable
          AddGnuCPlusPlusIncludePaths(
             "/usr/lib/gcc/i686-pc-linux-gnu/4.3.2/include/g++-v4",
             "i686-pc-linux-gnu");
          // Gentoo x86 2008.0 stable
          AddGnuCPlusPlusIncludePaths(
             "/usr/lib/gcc/i686-pc-linux-gnu/4.1.2/include/g++-v4",
             "i686-pc-linux-gnu");
          // Ubuntu 8.10
          AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.3",
                                      "i486-pc-linux-gnu");
          // Gentoo amd64 stable
          AddGnuCPlusPlusIncludePaths(
             "/usr/lib/gcc/x86_64-pc-linux-gnu/4.1.2/include/g++-v4",
             "i686-pc-linux-gnu");
          break;
        case llvm::Triple::FreeBSD:
          // DragonFly
          AddPath("/usr/include/c++/4.1", System, true, false, false);
          // FreeBSD
          AddPath("/usr/include/c++/4.2", System, true, false, false);
          break;
        case llvm::Triple::Solaris:
          // AuroraUX
          AddGnuCPlusPlusIncludePaths("/Opt/gcc4/include/c++/4.2.4",
                                      "i386-pc-solaris2.11");
          break;
        default:
          break;
      }
    }

    AddPath("/usr/local/include", System, false, false, false);

    AddPath("/usr/include", System, false, false, false);
    AddPath("/System/Library/Frameworks", System, true, false, true);
    AddPath("/Library/Frameworks", System, true, false, true);
    break;
  }
}

void InitHeaderSearch::AddDefaultEnvVarPaths(const LangOptions &Lang) {
  AddEnvVarPaths("CPATH");
  if (Lang.CPlusPlus && Lang.ObjC1)
    AddEnvVarPaths("OBJCPLUS_INCLUDE_PATH");
  else if (Lang.CPlusPlus)
    AddEnvVarPaths("CPLUS_INCLUDE_PATH");
  else if (Lang.ObjC1)
    AddEnvVarPaths("OBJC_INCLUDE_PATH");
  else
    AddEnvVarPaths("C_INCLUDE_PATH");
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
      fprintf(stderr, "ignoring duplicate directory \"%s\"\n",
              CurEntry.getName());
      if (DirToRemove != i)
        fprintf(stderr, "  as it is a non-system directory that duplicates"
                " a system directory\n");
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
    fprintf(stderr, "#include \"...\" search starts here:\n");
    unsigned QuotedIdx = IncludeGroup[Quoted].size();
    for (unsigned i = 0, e = SearchList.size(); i != e; ++i) {
      if (i == QuotedIdx)
        fprintf(stderr, "#include <...> search starts here:\n");
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
      fprintf(stderr, " %s%s\n", Name, Suffix);
    }
    fprintf(stderr, "End of search list.\n");
  }
}
