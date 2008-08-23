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

#include "clang/Driver/InitHeaderSearch.h"

#include "clang/Lex/HeaderSearch.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/LangOptions.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/System/Path.h"

#include <vector>

using namespace clang;

void InitHeaderSearch::AddPath(const std::string &Path, IncludeDirGroup Group,
                               bool isCXXAware, bool isUserSupplied,
                               bool isFramework) {
  assert(!Path.empty() && "can't handle empty path here");
  FileManager &FM = Headers.getFileMgr();
  
  // Compute the actual path, taking into consideration -isysroot.
  llvm::SmallString<256> MappedPath;
  
  // Handle isysroot.
  if (Group == System) {
    // FIXME: Portability.  This should be a sys::Path interface, this doesn't
    // handle things like C:\ right, nor win32 \\network\device\blah.
    if (isysroot.size() != 1 || isysroot[0] != '/') // Add isysroot if present.
      MappedPath.append(isysroot.begin(), isysroot.end());
  }
  
  MappedPath.append(Path.begin(), Path.end());

  // Compute the DirectoryLookup type.
  DirectoryLookup::DirType Type;
  if (Group == Quoted || Group == Angled)
    Type = DirectoryLookup::NormalHeaderDir;
  else if (isCXXAware)
    Type = DirectoryLookup::SystemHeaderDir;
  else
    Type = DirectoryLookup::ExternCSystemHeaderDir;
  
  
  // If the directory exists, add it.
  if (const DirectoryEntry *DE = FM.getDirectory(&MappedPath[0], 
                                                 &MappedPath[0]+
                                                 MappedPath.size())) {
    IncludeGroup[Group].push_back(DirectoryLookup(DE, Type, isUserSupplied,
                                                  isFramework));
    return;
  }
  
  // Check to see if this is an apple-style headermap (which are not allowed to
  // be frameworks).
  if (!isFramework) {
    if (const FileEntry *FE = FM.getFile(&MappedPath[0], 
                                         &MappedPath[0]+MappedPath.size())) {
      if (const HeaderMap *HM = Headers.CreateHeaderMap(FE)) {
        // It is a headermap, add it to the search path.
        IncludeGroup[Group].push_back(DirectoryLookup(HM, Type,isUserSupplied));
        return;
      }
    }
  }
  
  if (Verbose)
    fprintf(stderr, "ignoring nonexistent directory \"%s\"\n", Path.c_str());
}


void InitHeaderSearch::AddEnvVarPaths(const char *Name) {
  const char* at = getenv(Name);
  if (!at)
    return;

  const char* delim = strchr(at, llvm::sys::PathSeparator);
  while (delim != 0) {
    if (delim-at == 0)
      AddPath(".", Angled, false, true, false);
    else
      AddPath(std::string(at, std::string::size_type(delim-at)), Angled, false,
            true, false);
    at = delim + 1;
    delim = strchr(at, llvm::sys::PathSeparator);
  }
  if (*at == 0)
    AddPath(".", Angled, false, true, false);
  else
    AddPath(at, Angled, false, true, false);
}


void InitHeaderSearch::AddDefaultSystemIncludePaths(const LangOptions &Lang) {
  // FIXME: temporary hack: hard-coded paths.
  // FIXME: get these from the target?
  if (Lang.CPlusPlus) {
    AddPath("/usr/include/c++/4.2.1", System, true, false, false);
    AddPath("/usr/include/c++/4.2.1/i686-apple-darwin10", System, true, false,
        false);
    AddPath("/usr/include/c++/4.2.1/backward", System, true, false, false);

    AddPath("/usr/include/c++/4.0.0", System, true, false, false);
    AddPath("/usr/include/c++/4.0.0/i686-apple-darwin8", System, true, false,
        false);
    AddPath("/usr/include/c++/4.0.0/backward", System, true, false, false);

    // Ubuntu 7.10 - Gutsy Gibbon
    AddPath("/usr/include/c++/4.1.3", System, true, false, false);
    AddPath("/usr/include/c++/4.1.3/i486-linux-gnu", System, true, false,
        false);
    AddPath("/usr/include/c++/4.1.3/backward", System, true, false, false);

    // Fedora 8
    AddPath("/usr/include/c++/4.1.2", System, true, false, false);
    AddPath("/usr/include/c++/4.1.2/i386-redhat-linux", System, true, false,
        false);
    AddPath("/usr/include/c++/4.1.2/backward", System, true, false, false);

    // Fedora 9
    AddPath("/usr/include/c++/4.3.0", System, true, false, false);
    AddPath("/usr/include/c++/4.3.0/i386-redhat-linux", System, true, false,
        false);
    AddPath("/usr/include/c++/4.3.0/backward", System, true, false, false);

    // Arch Linux 2008-06-24
    AddPath("/usr/include/c++/4.3.1", System, true, false, false);
    AddPath("/usr/include/c++/4.3.1/i686-pc-linux-gnu", System, true, false,
        false);
    AddPath("/usr/include/c++/4.3.1/backward", System, true, false, false);
    AddPath("/usr/include/c++/4.3.1/x86_64-unknown-linux-gnu", System, true,
        false, false);

    // DragonFly
    AddPath("/usr/include/c++/4.1", System, true, false, false);
  }

  AddPath("/usr/local/include", System, false, false, false);

  AddPath("/usr/lib/gcc/i686-apple-darwin10/4.2.1/include", System, 
      false, false, false);
  AddPath("/usr/lib/gcc/powerpc-apple-darwin10/4.2.1/include", 
      System, false, false, false);

  // leopard
  AddPath("/usr/lib/gcc/i686-apple-darwin9/4.0.1/include", System, 
      false, false, false);
  AddPath("/usr/lib/gcc/powerpc-apple-darwin9/4.0.1/include", 
      System, false, false, false);
  AddPath("/usr/lib/gcc/powerpc-apple-darwin9/"
      "4.0.1/../../../../powerpc-apple-darwin0/include", 
      System, false, false, false);

  // tiger
  AddPath("/usr/lib/gcc/i686-apple-darwin8/4.0.1/include", System, 
      false, false, false);
  AddPath("/usr/lib/gcc/powerpc-apple-darwin8/4.0.1/include", 
      System, false, false, false);
  AddPath("/usr/lib/gcc/powerpc-apple-darwin8/"
      "4.0.1/../../../../powerpc-apple-darwin8/include", 
      System, false, false, false);

  // Ubuntu 7.10 - Gutsy Gibbon
  AddPath("/usr/lib/gcc/i486-linux-gnu/4.1.3/include", System,
      false, false, false);

  // Fedora 8
  AddPath("/usr/lib/gcc/i386-redhat-linux/4.1.2/include", System,
      false, false, false);

  // Fedora 9
  AddPath("/usr/lib/gcc/i386-redhat-linux/4.3.0/include", System,
      false, false, false);

  //Debian testing/lenny x86
  AddPath("/usr/lib/gcc/i486-linux-gnu/4.2.3/include", System,
      false, false, false);

  //Debian testing/lenny amd64
  AddPath("/usr/lib/gcc/x86_64-linux-gnu/4.2.3/include", System,
      false, false, false);

  // Arch Linux 2008-06-24
  AddPath("/usr/lib/gcc/i686-pc-linux-gnu/4.3.1/include", System,
      false, false, false);
  AddPath("/usr/lib/gcc/i686-pc-linux-gnu/4.3.1/include-fixed", System,
      false, false, false);
  AddPath("/usr/lib/gcc/x86_64-unknown-linux-gnu/4.3.1/include", System,
      false, false, false);
  AddPath("/usr/lib/gcc/x86_64-unknown-linux-gnu/4.3.1/include-fixed",
      System, false, false, false);

  // Debian testing/lenny ppc32
  AddPath("/usr/lib/gcc/powerpc-linux-gnu/4.2.3/include", System,
      false, false, false);

  // Gentoo x86 stable
  AddPath("/usr/lib/gcc/i686-pc-linux-gnu/4.1.2/include", System,
      false, false, false);

  // DragonFly
  AddPath("/usr/libdata/gcc41", System, true, false, false);

  AddPath("/usr/include", System, false, false, false);
  AddPath("/System/Library/Frameworks", System, true, false, true);
  AddPath("/Library/Frameworks", System, true, false, true);
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
    if (SearchList[i].isNormalDir()) {
      // If this isn't the first time we've seen this dir, remove it.
      if (SeenDirs.insert(SearchList[i].getDir()))
        continue;
      
      if (Verbose)
        fprintf(stderr, "ignoring duplicate directory \"%s\"\n",
                SearchList[i].getDir()->getName());
    } else if (SearchList[i].isFramework()) {
      // If this isn't the first time we've seen this framework dir, remove it.
      if (SeenFrameworkDirs.insert(SearchList[i].getFrameworkDir()))
        continue;
      
      if (Verbose)
        fprintf(stderr, "ignoring duplicate framework \"%s\"\n",
                SearchList[i].getFrameworkDir()->getName());
      
    } else {
      assert(SearchList[i].isHeaderMap() && "Not a headermap or normal dir?");
      // If this isn't the first time we've seen this headermap, remove it.
      if (SeenHeaderMaps.insert(SearchList[i].getHeaderMap()))
        continue;
      
      if (Verbose)
        fprintf(stderr, "ignoring duplicate directory \"%s\"\n",
                SearchList[i].getDir()->getName());
    }
    
    // This is reached if the current entry is a duplicate.
    SearchList.erase(SearchList.begin()+i);
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

