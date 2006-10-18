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

}

//===----------------------------------------------------------------------===//
// Header File Location.
//===----------------------------------------------------------------------===//

static std::string DoFrameworkLookup(const DirectoryEntry *Dir,
                                     const std::string &Filename) {
  // TODO: caching.
  
  // Framework names must have a '/' in the filename.
  std::string::size_type SlashPos = Filename.find('/');
  if (SlashPos == std::string::npos) return "";
  
  // FrameworkName = "/System/Library/Frameworks/"
  std::string FrameworkName = Dir->getName();
  if (FrameworkName.empty() || FrameworkName[FrameworkName.size()-1] != '/')
    FrameworkName += '/';
  
  // FrameworkName = "/System/Library/Frameworks/Cocoa"
  FrameworkName += std::string(Filename.begin(), Filename.begin()+SlashPos);
  
  // FrameworkName = "/System/Library/Frameworks/Cocoa.framework/"
  FrameworkName += ".framework/";
  
  // If the dir doesn't exist, give up.
  if (!sys::Path(FrameworkName).exists()) return "";
  
  // Check "/System/Library/Frameworks/Cocoa.framework/Headers/file.h"
  std::string HeadersFilename = FrameworkName + "Headers/" +
    std::string(Filename.begin()+SlashPos+1, Filename.end());
  if (sys::Path(HeadersFilename).exists()) return HeadersFilename;
  
  // Check "/System/Library/Frameworks/Cocoa.framework/PrivateHeaders/file.h"
  std::string PrivateHeadersFilename = FrameworkName + "PrivateHeaders/" +
    std::string(Filename.begin()+SlashPos+1, Filename.end());
  if (sys::Path(PrivateHeadersFilename).exists()) return HeadersFilename;
  
  return "";
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
  if (CurFileEnt && !NoCurDirSearch) {
    // Concatenate the requested file onto the directory.
    // FIXME: Portability.  Filename concatenation should be in sys::Path.
    if (const FileEntry *FE = 
          FileMgr.getFile(CurFileEnt->getDir()->getName()+"/"+Filename)) {
      // Leave CurDir unset.
      
      // This file is a system header or C++ unfriendly if the old file is.
      getFileInfo(CurFileEnt).DirInfo = getFileInfo(CurFileEnt).DirInfo;
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
    
    if (!SearchDirs[i].isFramework()) {
      // FIXME: Portability.  Adding file to dir should be in sys::Path.
      SearchDir = SearchDirs[i].getDir()->getName()+"/"+Filename;
    } else {
      SearchDir = DoFrameworkLookup(SearchDirs[i].getDir(), Filename);
      if (SearchDir.empty()) continue;
    }
    
    if (const FileEntry *FE = FileMgr.getFile(SearchDir)) {
      CurDir = &SearchDirs[i];
      
      // This file is a system header or C++ unfriendly if the dir is.
      getFileInfo(FE).DirInfo = CurDir->getDirCharacteristic();
      return FE;
    }
  }
  
  // Otherwise, didn't find it.
  return 0;
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


