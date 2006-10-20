//===--- HeaderSearch.cpp - Resolve Header File Locations ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the DirectoryLookup and HeaderSearch interfaces.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/FileManager.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/IdentifierTable.h"
#include "llvm/System/Path.h"
#include <iostream>
using namespace llvm;
using namespace clang;

HeaderSearch::HeaderSearch(FileManager &FM) : FileMgr(FM) {
  SystemDirIdx = 0;
  NoCurDirSearch = false;
  
  NumIncluded = 0;
  NumMultiIncludeFileOptzn = 0;
  NumFrameworkLookups = NumSubFrameworkLookups = 0;
}

void HeaderSearch::PrintStats() {
  std::cerr << "\n*** HeaderSearch Stats:\n";
  std::cerr << FileInfo.size() << " files tracked.\n";
  unsigned NumOnceOnlyFiles = 0, MaxNumIncludes = 0, NumSingleIncludedFiles = 0;
  for (unsigned i = 0, e = FileInfo.size(); i != e; ++i) {
    NumOnceOnlyFiles += FileInfo[i].isImport;
    if (MaxNumIncludes < FileInfo[i].NumIncludes)
      MaxNumIncludes = FileInfo[i].NumIncludes;
    NumSingleIncludedFiles += FileInfo[i].NumIncludes == 1;
  }
  std::cerr << "  " << NumOnceOnlyFiles << " #import/#pragma once files.\n";
  std::cerr << "  " << NumSingleIncludedFiles << " included exactly once.\n";
  std::cerr << "  " << MaxNumIncludes << " max times a file is included.\n";
  
  std::cerr << "  " << NumIncluded << " #include/#include_next/#import.\n";
  std::cerr << "    " << NumMultiIncludeFileOptzn << " #includes skipped due to"
            << " the multi-include optimization.\n";
  
  std::cerr << NumFrameworkLookups << " framework lookups.\n";
  std::cerr << NumSubFrameworkLookups << " subframework lookups.\n";
}

//===----------------------------------------------------------------------===//
// Header File Location.
//===----------------------------------------------------------------------===//

const FileEntry *HeaderSearch::DoFrameworkLookup(const DirectoryEntry *Dir,
                                                 const std::string &Filename) {
  // Framework names must have a '/' in the filename.
  std::string::size_type SlashPos = Filename.find('/');
  if (SlashPos == std::string::npos) return 0;
  
  // TODO: caching.
  ++NumFrameworkLookups;
  
  // FrameworkName = "/System/Library/Frameworks/"
  std::string FrameworkName = Dir->getName();
  if (FrameworkName.empty() || FrameworkName[FrameworkName.size()-1] != '/')
    FrameworkName += '/';
  
  // FrameworkName = "/System/Library/Frameworks/Cocoa"
  FrameworkName += std::string(Filename.begin(), Filename.begin()+SlashPos);
  
  // FrameworkName = "/System/Library/Frameworks/Cocoa.framework/"
  FrameworkName += ".framework/";
  
  // Check "/System/Library/Frameworks/Cocoa.framework/Headers/file.h"
  std::string HeadersFilename = FrameworkName + "Headers/" +
    std::string(Filename.begin()+SlashPos+1, Filename.end());
  if (const FileEntry *FE = FileMgr.getFile(HeadersFilename))
    return FE;
  
  // Check "/System/Library/Frameworks/Cocoa.framework/PrivateHeaders/file.h"
  std::string PrivateHeadersFilename = FrameworkName + "PrivateHeaders/" +
    std::string(Filename.begin()+SlashPos+1, Filename.end());
  return FileMgr.getFile(PrivateHeadersFilename);
}

/// LookupFile - Given a "foo" or <foo> reference, look up the indicated file,
/// return null on failure.  isAngled indicates whether the file reference is
/// for system #include's or not (i.e. using <> instead of "").  CurFileEnt, if
/// non-null, indicates where the #including file is, in case a relative search
/// is needed.
const FileEntry *HeaderSearch::LookupFile(const std::string &Filename, 
                                          bool isAngled,
                                          const DirectoryLookup *FromDir,
                                          const DirectoryLookup *&CurDir,
                                          const FileEntry *CurFileEnt) {
  // If 'Filename' is absolute, check to see if it exists and no searching.
  // FIXME: Portability.  This should be a sys::Path interface, this doesn't
  // handle things like C:\foo.txt right, nor win32 \\network\device\blah.
  if (Filename[0] == '/') {
    CurDir = 0;

    // If this was an #include_next "/absolute/file", fail.
    if (FromDir) return 0;
    
    // Otherwise, just return the file.
    return FileMgr.getFile(Filename);
  }
  
  // Step #0, unless disabled, check to see if the file is in the #includer's
  // directory.  This search is not done for <> headers.
  if (CurFileEnt && !isAngled && !NoCurDirSearch) {
    // Concatenate the requested file onto the directory.
    // FIXME: Portability.  Filename concatenation should be in sys::Path.
    if (const FileEntry *FE = 
          FileMgr.getFile(CurFileEnt->getDir()->getName()+"/"+Filename)) {
      // Leave CurDir unset.
      
      // This file is a system header or C++ unfriendly if the old file is.
      getFileInfo(FE).DirInfo = getFileInfo(CurFileEnt).DirInfo;
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
  
  // Check each directory in sequence to see if it contains this file.
  for (; i != SearchDirs.size(); ++i) {
    // Concatenate the requested file onto the directory.
    std::string SearchDir;
    
    const FileEntry *FE = 0;
    if (!SearchDirs[i].isFramework()) {
      // FIXME: Portability.  Adding file to dir should be in sys::Path.
      FE = FileMgr.getFile(SearchDirs[i].getDir()->getName()+"/"+Filename);
    } else {
      FE = DoFrameworkLookup(SearchDirs[i].getDir(), Filename);
    }
    
    if (FE) {
      CurDir = &SearchDirs[i];
      
      // This file is a system header or C++ unfriendly if the dir is.
      getFileInfo(FE).DirInfo = CurDir->getDirCharacteristic();
      return FE;
    }
  }
  
  // Otherwise, didn't find it.
  return 0;
}

/// LookupSubframeworkHeader - Look up a subframework for the specified
/// #include file.  For example, if #include'ing <HIToolbox/HIToolbox.h> from
/// within ".../Carbon.framework/Headers/Carbon.h", check to see if HIToolbox
/// is a subframework within Carbon.framework.  If so, return the FileEntry
/// for the designated file, otherwise return null.
const FileEntry *HeaderSearch::
LookupSubframeworkHeader(const std::string &Filename,
                         const FileEntry *ContextFileEnt) {
  // Framework names must have a '/' in the filename.  Find it.
  std::string::size_type SlashPos = Filename.find('/');
  if (SlashPos == std::string::npos) return 0;
  
  // TODO: Cache subframework.
  ++NumSubFrameworkLookups;
  
  // Look up the base framework name of the ContextFileEnt.
  const std::string &ContextName = ContextFileEnt->getName();
  std::string::size_type FrameworkPos = ContextName.find(".framework/");
  // If the context info wasn't a framework, couldn't be a subframework.
  if (FrameworkPos == std::string::npos)
    return 0;
  
  std::string FrameworkName(ContextName.begin(),
                        ContextName.begin()+FrameworkPos+strlen(".framework/"));
  // Append Frameworks/HIToolbox.framework/
  FrameworkName += "Frameworks/";
  FrameworkName += std::string(Filename.begin(), Filename.begin()+SlashPos);
  FrameworkName += ".framework/";

  const FileEntry *FE = 0;

  // Check ".../Frameworks/HIToolbox.framework/Headers/HIToolbox.h"
  std::string HeadersFilename = FrameworkName + "Headers/" +
    std::string(Filename.begin()+SlashPos+1, Filename.end());
  if (!(FE = FileMgr.getFile(HeadersFilename))) {
    
    // Check ".../Frameworks/HIToolbox.framework/PrivateHeaders/HIToolbox.h"
    std::string PrivateHeadersFilename = FrameworkName + "PrivateHeaders/" +
      std::string(Filename.begin()+SlashPos+1, Filename.end());
    if (!(FE = FileMgr.getFile(PrivateHeadersFilename)))
      return 0;
  }
  
  // This file is a system header or C++ unfriendly if the old file is.
  getFileInfo(FE).DirInfo = getFileInfo(ContextFileEnt).DirInfo;
  return FE;
}

//===----------------------------------------------------------------------===//
// File Info Management.
//===----------------------------------------------------------------------===//


/// getFileInfo - Return the PerFileInfo structure for the specified
/// FileEntry.
HeaderSearch::PerFileInfo &HeaderSearch::getFileInfo(const FileEntry *FE) {
  if (FE->getUID() >= FileInfo.size())
    FileInfo.resize(FE->getUID()+1);
  return FileInfo[FE->getUID()];
}  

/// ShouldEnterIncludeFile - Mark the specified file as a target of of a
/// #include, #include_next, or #import directive.  Return false if #including
/// the file will have no effect or true if we should include it.
bool HeaderSearch::ShouldEnterIncludeFile(const FileEntry *File, bool isImport){
  ++NumIncluded; // Count # of attempted #includes.

  // Get information about this file.
  PerFileInfo &FileInfo = getFileInfo(File);
  
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
  if (FileInfo.ControllingMacro && FileInfo.ControllingMacro->getMacroInfo()) {
    ++NumMultiIncludeFileOptzn;
    return false;
  }
  
  // Increment the number of times this file has been included.
  ++FileInfo.NumIncludes;
  
  return true;
}


