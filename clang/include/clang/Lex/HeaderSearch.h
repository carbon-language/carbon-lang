//===--- HeaderSearch.h - Resolve Header File Locations ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the HeaderSearch interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LEX_HEADERSEARCH_H
#define LLVM_CLANG_LEX_HEADERSEARCH_H

#include "clang/Lex/DirectoryLookup.h"
#include "llvm/ADT/StringMap.h"
#include <vector>

namespace clang {

class ExternalIdentifierLookup;
class FileEntry;
class FileManager;
class IdentifierInfo;

/// HeaderFileInfo - The preprocessor keeps track of this information for each
/// file that is #included.
struct HeaderFileInfo {
  /// isImport - True if this is a #import'd or #pragma once file.
  bool isImport : 1;
  
  /// DirInfo - Keep track of whether this is a system header, and if so,
  /// whether it is C++ clean or not.  This can be set by the include paths or
  /// by #pragma gcc system_header.  This is an instance of
  /// SrcMgr::CharacteristicKind.
  unsigned DirInfo : 2;
  
  /// NumIncludes - This is the number of times the file has been included
  /// already.
  unsigned short NumIncludes;
  
  /// ControllingMacro - If this file has a #ifndef XXX (or equivalent) guard
  /// that protects the entire contents of the file, this is the identifier
  /// for the macro that controls whether or not it has any effect.
  ///
  /// Note: Most clients should use getControllingMacro() to access
  /// the controlling macro of this header, since
  /// getControllingMacro() is able to load a controlling macro from
  /// external storage.
  const IdentifierInfo *ControllingMacro;

  /// \brief The ID number of the controlling macro. 
  ///
  /// This ID number will be non-zero when there is a controlling
  /// macro whose IdentifierInfo may not yet have been loaded from
  /// external storage.
  unsigned ControllingMacroID;

  HeaderFileInfo() 
    : isImport(false), DirInfo(SrcMgr::C_User),
      NumIncludes(0), ControllingMacro(0), ControllingMacroID(0) {}

  /// \brief Retrieve the controlling macro for this header file, if
  /// any.
  const IdentifierInfo *getControllingMacro(ExternalIdentifierLookup *External);
};

/// HeaderSearch - This class encapsulates the information needed to find the
/// file referenced by a #include or #include_next, (sub-)framework lookup, etc.
class HeaderSearch {
  FileManager &FileMgr;
  
  /// #include search path information.  Requests for #include "x" search the
  /// directory of the #including file first, then each directory in SearchDirs
  /// consequtively. Requests for <x> search the current dir first, then each
  /// directory in SearchDirs, starting at SystemDirIdx, consequtively.  If
  /// NoCurDirSearch is true, then the check for the file in the current
  /// directory is supressed.
  std::vector<DirectoryLookup> SearchDirs;
  unsigned SystemDirIdx;
  bool NoCurDirSearch;
  
  /// FileInfo - This contains all of the preprocessor-specific data about files
  /// that are included.  The vector is indexed by the FileEntry's UID.
  ///
  std::vector<HeaderFileInfo> FileInfo;

  /// LookupFileCache - This is keeps track of each lookup performed by
  /// LookupFile.  The first part of the value is the starting index in
  /// SearchDirs that the cached search was performed from.  If there is a hit
  /// and this value doesn't match the current query, the cache has to be
  /// ignored.  The second value is the entry in SearchDirs that satisfied the
  /// query.
  llvm::StringMap<std::pair<unsigned, unsigned> > LookupFileCache;
  
  
  /// FrameworkMap - This is a collection mapping a framework or subframework
  /// name like "Carbon" to the Carbon.framework directory.
  llvm::StringMap<const DirectoryEntry *> FrameworkMap;

  /// HeaderMaps - This is a mapping from FileEntry -> HeaderMap, uniquing 
  /// headermaps.  This vector owns the headermap.
  std::vector<std::pair<const FileEntry*, const HeaderMap*> > HeaderMaps;

  /// \brief Entity used to resolve the identifier IDs of controlling
  /// macros into IdentifierInfo pointers, as needed.
  ExternalIdentifierLookup *ExternalLookup;

  // Various statistics we track for performance analysis.
  unsigned NumIncluded;
  unsigned NumMultiIncludeFileOptzn;
  unsigned NumFrameworkLookups, NumSubFrameworkLookups;

  // HeaderSearch doesn't support default or copy construction.
  explicit HeaderSearch();  
  explicit HeaderSearch(const HeaderSearch&);
  void operator=(const HeaderSearch&);
public:
  HeaderSearch(FileManager &FM);
  ~HeaderSearch();

  FileManager &getFileMgr() const { return FileMgr; }

  /// SetSearchPaths - Interface for setting the file search paths.
  ///
  void SetSearchPaths(const std::vector<DirectoryLookup> &dirs,
                      unsigned systemDirIdx, bool noCurDirSearch) {
    SearchDirs = dirs;
    SystemDirIdx = systemDirIdx;
    NoCurDirSearch = noCurDirSearch;
    //LookupFileCache.clear();
  }
  
  /// ClearFileInfo - Forget everything we know about headers so far.
  void ClearFileInfo() {
    FileInfo.clear();
  }
  
  void SetExternalLookup(ExternalIdentifierLookup *EIL) {
    ExternalLookup = EIL;
  }

  /// LookupFile - Given a "foo" or <foo> reference, look up the indicated file,
  /// return null on failure.  isAngled indicates whether the file reference is
  /// a <> reference.  If successful, this returns 'UsedDir', the
  /// DirectoryLookup member the file was found in, or null if not applicable.
  /// If CurDir is non-null, the file was found in the specified directory
  /// search location.  This is used to implement #include_next.  CurFileEnt, if
  /// non-null, indicates where the #including file is, in case a relative
  /// search is needed.
  const FileEntry *LookupFile(const char *FilenameStart,
                              const char *FilenameEnd, bool isAngled,
                              const DirectoryLookup *FromDir,
                              const DirectoryLookup *&CurDir,
                              const FileEntry *CurFileEnt);
  
  /// LookupSubframeworkHeader - Look up a subframework for the specified
  /// #include file.  For example, if #include'ing <HIToolbox/HIToolbox.h> from
  /// within ".../Carbon.framework/Headers/Carbon.h", check to see if HIToolbox
  /// is a subframework within Carbon.framework.  If so, return the FileEntry
  /// for the designated file, otherwise return null.
  const FileEntry *LookupSubframeworkHeader(const char *FilenameStart,
                                            const char *FilenameEnd,
                                            const FileEntry *RelativeFileEnt);
  
  /// LookupFrameworkCache - Look up the specified framework name in our
  /// framework cache, returning the DirectoryEntry it is in if we know,
  /// otherwise, return null.
  const DirectoryEntry *&LookupFrameworkCache(const char *FWNameStart,
                                              const char *FWNameEnd) {
    return FrameworkMap.GetOrCreateValue(FWNameStart, FWNameEnd).getValue();
  }
  
  /// ShouldEnterIncludeFile - Mark the specified file as a target of of a
  /// #include, #include_next, or #import directive.  Return false if #including
  /// the file will have no effect or true if we should include it.
  bool ShouldEnterIncludeFile(const FileEntry *File, bool isImport);
  
  
  /// getFileDirFlavor - Return whether the specified file is a normal header,
  /// a system header, or a C++ friendly system header.
  SrcMgr::CharacteristicKind getFileDirFlavor(const FileEntry *File) {
    return (SrcMgr::CharacteristicKind)getFileInfo(File).DirInfo;
  }
    
  /// MarkFileIncludeOnce - Mark the specified file as a "once only" file, e.g.
  /// due to #pragma once.
  void MarkFileIncludeOnce(const FileEntry *File) {
    getFileInfo(File).isImport = true;
  }

  /// MarkFileSystemHeader - Mark the specified file as a system header, e.g.
  /// due to #pragma GCC system_header.
  void MarkFileSystemHeader(const FileEntry *File) {
    getFileInfo(File).DirInfo = SrcMgr::C_System;
  }
  
  /// IncrementIncludeCount - Increment the count for the number of times the
  /// specified FileEntry has been entered.
  void IncrementIncludeCount(const FileEntry *File) {
    ++getFileInfo(File).NumIncludes;
  }
  
  /// SetFileControllingMacro - Mark the specified file as having a controlling
  /// macro.  This is used by the multiple-include optimization to eliminate
  /// no-op #includes.
  void SetFileControllingMacro(const FileEntry *File,
                               const IdentifierInfo *ControllingMacro) {
    getFileInfo(File).ControllingMacro = ControllingMacro;
  }
  
  /// CreateHeaderMap - This method returns a HeaderMap for the specified
  /// FileEntry, uniquing them through the the 'HeaderMaps' datastructure.
  const HeaderMap *CreateHeaderMap(const FileEntry *FE);
  
  void IncrementFrameworkLookupCount() { ++NumFrameworkLookups; }

  typedef std::vector<HeaderFileInfo>::iterator header_file_iterator;
  header_file_iterator header_file_begin() { return FileInfo.begin(); }
  header_file_iterator header_file_end() { return FileInfo.end(); }

  // Used by PCHReader.
  void setHeaderFileInfoForUID(HeaderFileInfo HFI, unsigned UID);
  
  void PrintStats();
private:
      
  /// getFileInfo - Return the HeaderFileInfo structure for the specified
  /// FileEntry.
  HeaderFileInfo &getFileInfo(const FileEntry *FE);
};

}  // end namespace clang

#endif
