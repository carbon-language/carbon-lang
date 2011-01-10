//===--- HeaderSearch.cpp - Resolve Header File Locations ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the DirectoryLookup and HeaderSearch interfaces.
//
//===----------------------------------------------------------------------===//

#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/HeaderMap.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/IdentifierTable.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/ADT/SmallString.h"
#include <cstdio>
using namespace clang;

const IdentifierInfo *
HeaderFileInfo::getControllingMacro(ExternalIdentifierLookup *External) {
  if (ControllingMacro)
    return ControllingMacro;

  if (!ControllingMacroID || !External)
    return 0;

  ControllingMacro = External->GetIdentifier(ControllingMacroID);
  return ControllingMacro;
}

HeaderSearch::HeaderSearch(FileManager &FM)
    : FileMgr(FM), FrameworkMap(64) {
  SystemDirIdx = 0;
  NoCurDirSearch = false;

  ExternalLookup = 0;
  NumIncluded = 0;
  NumMultiIncludeFileOptzn = 0;
  NumFrameworkLookups = NumSubFrameworkLookups = 0;
}

HeaderSearch::~HeaderSearch() {
  // Delete headermaps.
  for (unsigned i = 0, e = HeaderMaps.size(); i != e; ++i)
    delete HeaderMaps[i].second;
}

void HeaderSearch::PrintStats() {
  fprintf(stderr, "\n*** HeaderSearch Stats:\n");
  fprintf(stderr, "%d files tracked.\n", (int)FileInfo.size());
  unsigned NumOnceOnlyFiles = 0, MaxNumIncludes = 0, NumSingleIncludedFiles = 0;
  for (unsigned i = 0, e = FileInfo.size(); i != e; ++i) {
    NumOnceOnlyFiles += FileInfo[i].isImport;
    if (MaxNumIncludes < FileInfo[i].NumIncludes)
      MaxNumIncludes = FileInfo[i].NumIncludes;
    NumSingleIncludedFiles += FileInfo[i].NumIncludes == 1;
  }
  fprintf(stderr, "  %d #import/#pragma once files.\n", NumOnceOnlyFiles);
  fprintf(stderr, "  %d included exactly once.\n", NumSingleIncludedFiles);
  fprintf(stderr, "  %d max times a file is included.\n", MaxNumIncludes);

  fprintf(stderr, "  %d #include/#include_next/#import.\n", NumIncluded);
  fprintf(stderr, "    %d #includes skipped due to"
          " the multi-include optimization.\n", NumMultiIncludeFileOptzn);

  fprintf(stderr, "%d framework lookups.\n", NumFrameworkLookups);
  fprintf(stderr, "%d subframework lookups.\n", NumSubFrameworkLookups);
}

/// CreateHeaderMap - This method returns a HeaderMap for the specified
/// FileEntry, uniquing them through the the 'HeaderMaps' datastructure.
const HeaderMap *HeaderSearch::CreateHeaderMap(const FileEntry *FE) {
  // We expect the number of headermaps to be small, and almost always empty.
  // If it ever grows, use of a linear search should be re-evaluated.
  if (!HeaderMaps.empty()) {
    for (unsigned i = 0, e = HeaderMaps.size(); i != e; ++i)
      // Pointer equality comparison of FileEntries works because they are
      // already uniqued by inode.
      if (HeaderMaps[i].first == FE)
        return HeaderMaps[i].second;
  }

  if (const HeaderMap *HM = HeaderMap::Create(FE, FileMgr)) {
    HeaderMaps.push_back(std::make_pair(FE, HM));
    return HM;
  }

  return 0;
}

//===----------------------------------------------------------------------===//
// File lookup within a DirectoryLookup scope
//===----------------------------------------------------------------------===//

/// getName - Return the directory or filename corresponding to this lookup
/// object.
const char *DirectoryLookup::getName() const {
  if (isNormalDir())
    return getDir()->getName();
  if (isFramework())
    return getFrameworkDir()->getName();
  assert(isHeaderMap() && "Unknown DirectoryLookup");
  return getHeaderMap()->getFileName();
}


/// LookupFile - Lookup the specified file in this search path, returning it
/// if it exists or returning null if not.
const FileEntry *DirectoryLookup::LookupFile(llvm::StringRef Filename,
                                             HeaderSearch &HS) const {
  llvm::SmallString<1024> TmpDir;
  if (isNormalDir()) {
    // Concatenate the requested file onto the directory.
    // FIXME: Portability.  Filename concatenation should be in sys::Path.
    TmpDir += getDir()->getName();
    TmpDir.push_back('/');
    TmpDir.append(Filename.begin(), Filename.end());
    return HS.getFileMgr().getFile(TmpDir.str());
  }

  if (isFramework())
    return DoFrameworkLookup(Filename, HS);

  assert(isHeaderMap() && "Unknown directory lookup");
  return getHeaderMap()->LookupFile(Filename, HS.getFileMgr());
}


/// DoFrameworkLookup - Do a lookup of the specified file in the current
/// DirectoryLookup, which is a framework directory.
const FileEntry *DirectoryLookup::DoFrameworkLookup(llvm::StringRef Filename,
                                                    HeaderSearch &HS) const {
  FileManager &FileMgr = HS.getFileMgr();

  // Framework names must have a '/' in the filename.
  size_t SlashPos = Filename.find('/');
  if (SlashPos == llvm::StringRef::npos) return 0;

  // Find out if this is the home for the specified framework, by checking
  // HeaderSearch.  Possible answer are yes/no and unknown.
  const DirectoryEntry *&FrameworkDirCache =
    HS.LookupFrameworkCache(Filename.substr(0, SlashPos));

  // If it is known and in some other directory, fail.
  if (FrameworkDirCache && FrameworkDirCache != getFrameworkDir())
    return 0;

  // Otherwise, construct the path to this framework dir.

  // FrameworkName = "/System/Library/Frameworks/"
  llvm::SmallString<1024> FrameworkName;
  FrameworkName += getFrameworkDir()->getName();
  if (FrameworkName.empty() || FrameworkName.back() != '/')
    FrameworkName.push_back('/');

  // FrameworkName = "/System/Library/Frameworks/Cocoa"
  FrameworkName.append(Filename.begin(), Filename.begin()+SlashPos);

  // FrameworkName = "/System/Library/Frameworks/Cocoa.framework/"
  FrameworkName += ".framework/";

  // If the cache entry is still unresolved, query to see if the cache entry is
  // still unresolved.  If so, check its existence now.
  if (FrameworkDirCache == 0) {
    HS.IncrementFrameworkLookupCount();

    // If the framework dir doesn't exist, we fail.
    // FIXME: It's probably more efficient to query this with FileMgr.getDir.
    bool Exists;
    if (llvm::sys::fs::exists(FrameworkName.str(), Exists) || !Exists)
      return 0;

    // Otherwise, if it does, remember that this is the right direntry for this
    // framework.
    FrameworkDirCache = getFrameworkDir();
  }

  // Check "/System/Library/Frameworks/Cocoa.framework/Headers/file.h"
  unsigned OrigSize = FrameworkName.size();

  FrameworkName += "Headers/";
  FrameworkName.append(Filename.begin()+SlashPos+1, Filename.end());
  if (const FileEntry *FE = FileMgr.getFile(FrameworkName.str()))
    return FE;

  // Check "/System/Library/Frameworks/Cocoa.framework/PrivateHeaders/file.h"
  const char *Private = "Private";
  FrameworkName.insert(FrameworkName.begin()+OrigSize, Private,
                       Private+strlen(Private));
  return FileMgr.getFile(FrameworkName.str());
}


//===----------------------------------------------------------------------===//
// Header File Location.
//===----------------------------------------------------------------------===//


/// LookupFile - Given a "foo" or <foo> reference, look up the indicated file,
/// return null on failure.  isAngled indicates whether the file reference is
/// for system #include's or not (i.e. using <> instead of "").  CurFileEnt, if
/// non-null, indicates where the #including file is, in case a relative search
/// is needed.
const FileEntry *HeaderSearch::LookupFile(llvm::StringRef Filename,
                                          bool isAngled,
                                          const DirectoryLookup *FromDir,
                                          const DirectoryLookup *&CurDir,
                                          const FileEntry *CurFileEnt) {
  // If 'Filename' is absolute, check to see if it exists and no searching.
  if (llvm::sys::path::is_absolute(Filename)) {
    CurDir = 0;

    // If this was an #include_next "/absolute/file", fail.
    if (FromDir) return 0;

    // Otherwise, just return the file.
    return FileMgr.getFile(Filename);
  }

  // Step #0, unless disabled, check to see if the file is in the #includer's
  // directory.  This has to be based on CurFileEnt, not CurDir, because
  // CurFileEnt could be a #include of a subdirectory (#include "foo/bar.h") and
  // a subsequent include of "baz.h" should resolve to "whatever/foo/baz.h".
  // This search is not done for <> headers.
  if (CurFileEnt && !isAngled && !NoCurDirSearch) {
    llvm::SmallString<1024> TmpDir;
    // Concatenate the requested file onto the directory.
    // FIXME: Portability.  Filename concatenation should be in sys::Path.
    TmpDir += CurFileEnt->getDir()->getName();
    TmpDir.push_back('/');
    TmpDir.append(Filename.begin(), Filename.end());
    if (const FileEntry *FE = FileMgr.getFile(TmpDir.str())) {
      // Leave CurDir unset.
      // This file is a system header or C++ unfriendly if the old file is.
      //
      // Note that the temporary 'DirInfo' is required here, as either call to
      // getFileInfo could resize the vector and we don't want to rely on order
      // of evaluation.
      unsigned DirInfo = getFileInfo(CurFileEnt).DirInfo;
      getFileInfo(FE).DirInfo = DirInfo;
      return FE;
    }
  }

  CurDir = 0;

  // If this is a system #include, ignore the user #include locs.
  unsigned i = isAngled ? SystemDirIdx : 0;

  // If this is a #include_next request, start searching after the directory the
  // file was found in.
  if (FromDir)
    i = FromDir-&SearchDirs[0];

  // Cache all of the lookups performed by this method.  Many headers are
  // multiply included, and the "pragma once" optimization prevents them from
  // being relex/pp'd, but they would still have to search through a
  // (potentially huge) series of SearchDirs to find it.
  std::pair<unsigned, unsigned> &CacheLookup =
    LookupFileCache.GetOrCreateValue(Filename).getValue();

  // If the entry has been previously looked up, the first value will be
  // non-zero.  If the value is equal to i (the start point of our search), then
  // this is a matching hit.
  if (CacheLookup.first == i+1) {
    // Skip querying potentially lots of directories for this lookup.
    i = CacheLookup.second;
  } else {
    // Otherwise, this is the first query, or the previous query didn't match
    // our search start.  We will fill in our found location below, so prime the
    // start point value.
    CacheLookup.first = i+1;
  }

  // Check each directory in sequence to see if it contains this file.
  for (; i != SearchDirs.size(); ++i) {
    const FileEntry *FE =
      SearchDirs[i].LookupFile(Filename, *this);
    if (!FE) continue;

    CurDir = &SearchDirs[i];

    // This file is a system header or C++ unfriendly if the dir is.
    getFileInfo(FE).DirInfo = CurDir->getDirCharacteristic();

    // Remember this location for the next lookup we do.
    CacheLookup.second = i;
    return FE;
  }

  // Otherwise, didn't find it. Remember we didn't find this.
  CacheLookup.second = SearchDirs.size();
  return 0;
}

/// LookupSubframeworkHeader - Look up a subframework for the specified
/// #include file.  For example, if #include'ing <HIToolbox/HIToolbox.h> from
/// within ".../Carbon.framework/Headers/Carbon.h", check to see if HIToolbox
/// is a subframework within Carbon.framework.  If so, return the FileEntry
/// for the designated file, otherwise return null.
const FileEntry *HeaderSearch::
LookupSubframeworkHeader(llvm::StringRef Filename,
                         const FileEntry *ContextFileEnt) {
  assert(ContextFileEnt && "No context file?");

  // Framework names must have a '/' in the filename.  Find it.
  size_t SlashPos = Filename.find('/');
  if (SlashPos == llvm::StringRef::npos) return 0;

  // Look up the base framework name of the ContextFileEnt.
  const char *ContextName = ContextFileEnt->getName();

  // If the context info wasn't a framework, couldn't be a subframework.
  const char *FrameworkPos = strstr(ContextName, ".framework/");
  if (FrameworkPos == 0)
    return 0;

  llvm::SmallString<1024> FrameworkName(ContextName,
                                        FrameworkPos+strlen(".framework/"));

  // Append Frameworks/HIToolbox.framework/
  FrameworkName += "Frameworks/";
  FrameworkName.append(Filename.begin(), Filename.begin()+SlashPos);
  FrameworkName += ".framework/";

  llvm::StringMapEntry<const DirectoryEntry *> &CacheLookup =
    FrameworkMap.GetOrCreateValue(Filename.substr(0, SlashPos));

  // Some other location?
  if (CacheLookup.getValue() &&
      CacheLookup.getKeyLength() == FrameworkName.size() &&
      memcmp(CacheLookup.getKeyData(), &FrameworkName[0],
             CacheLookup.getKeyLength()) != 0)
    return 0;

  // Cache subframework.
  if (CacheLookup.getValue() == 0) {
    ++NumSubFrameworkLookups;

    // If the framework dir doesn't exist, we fail.
    const DirectoryEntry *Dir = FileMgr.getDirectory(FrameworkName.str());
    if (Dir == 0) return 0;

    // Otherwise, if it does, remember that this is the right direntry for this
    // framework.
    CacheLookup.setValue(Dir);
  }

  const FileEntry *FE = 0;

  // Check ".../Frameworks/HIToolbox.framework/Headers/HIToolbox.h"
  llvm::SmallString<1024> HeadersFilename(FrameworkName);
  HeadersFilename += "Headers/";
  HeadersFilename.append(Filename.begin()+SlashPos+1, Filename.end());
  if (!(FE = FileMgr.getFile(HeadersFilename.str()))) {

    // Check ".../Frameworks/HIToolbox.framework/PrivateHeaders/HIToolbox.h"
    HeadersFilename = FrameworkName;
    HeadersFilename += "PrivateHeaders/";
    HeadersFilename.append(Filename.begin()+SlashPos+1, Filename.end());
    if (!(FE = FileMgr.getFile(HeadersFilename.str())))
      return 0;
  }

  // This file is a system header or C++ unfriendly if the old file is.
  //
  // Note that the temporary 'DirInfo' is required here, as either call to
  // getFileInfo could resize the vector and we don't want to rely on order
  // of evaluation.
  unsigned DirInfo = getFileInfo(ContextFileEnt).DirInfo;
  getFileInfo(FE).DirInfo = DirInfo;
  return FE;
}

//===----------------------------------------------------------------------===//
// File Info Management.
//===----------------------------------------------------------------------===//


/// getFileInfo - Return the HeaderFileInfo structure for the specified
/// FileEntry.
HeaderFileInfo &HeaderSearch::getFileInfo(const FileEntry *FE) {
  if (FE->getUID() >= FileInfo.size())
    FileInfo.resize(FE->getUID()+1);
  return FileInfo[FE->getUID()];
}

void HeaderSearch::setHeaderFileInfoForUID(HeaderFileInfo HFI, unsigned UID) {
  if (UID >= FileInfo.size())
    FileInfo.resize(UID+1);
  FileInfo[UID] = HFI;
}

/// ShouldEnterIncludeFile - Mark the specified file as a target of of a
/// #include, #include_next, or #import directive.  Return false if #including
/// the file will have no effect or true if we should include it.
bool HeaderSearch::ShouldEnterIncludeFile(const FileEntry *File, bool isImport){
  ++NumIncluded; // Count # of attempted #includes.

  // Get information about this file.
  HeaderFileInfo &FileInfo = getFileInfo(File);

  // If this is a #import directive, check that we have not already imported
  // this header.
  if (isImport) {
    // If this has already been imported, don't import it again.
    FileInfo.isImport = true;

    // Has this already been #import'ed or #include'd?
    if (FileInfo.NumIncludes) return false;
  } else {
    // Otherwise, if this is a #include of a file that was previously #import'd
    // or if this is the second #include of a #pragma once file, ignore it.
    if (FileInfo.isImport)
      return false;
  }

  // Next, check to see if the file is wrapped with #ifndef guards.  If so, and
  // if the macro that guards it is defined, we know the #include has no effect.
  if (const IdentifierInfo *ControllingMacro
      = FileInfo.getControllingMacro(ExternalLookup))
    if (ControllingMacro->hasMacroDefinition()) {
      ++NumMultiIncludeFileOptzn;
      return false;
    }

  // Increment the number of times this file has been included.
  ++FileInfo.NumIncludes;

  return true;
}


