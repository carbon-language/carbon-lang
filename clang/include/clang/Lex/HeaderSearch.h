//===--- HeaderSearch.h - Resolve Header File Locations ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the HeaderSearch interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LEX_HEADERSEARCH_H
#define LLVM_CLANG_LEX_HEADERSEARCH_H

#include "clang/Lex/DirectoryLookup.h"
#include "llvm/ADT/CStringMap.h"
#include <vector>
#include <string>

namespace llvm {
namespace clang {
class FileEntry;
class FileManager;
class IdentifierInfo;

  
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
  
  /// PreFileInfo - The preprocessor keeps track of this information for each
  /// file that is #included.
  struct PerFileInfo {
    /// isImport - True if this is a #import'd or #pragma once file.
    bool isImport : 1;
    
    /// DirInfo - Keep track of whether this is a system header, and if so,
    /// whether it is C++ clean or not.  This can be set by the include paths or
    /// by #pragma gcc system_header.
    DirectoryLookup::DirType DirInfo : 2;
    
    /// NumIncludes - This is the number of times the file has been included
    /// already.
    unsigned short NumIncludes;
    
    /// ControllingMacro - If this file has a #ifndef XXX (or equivalent) guard
    /// that protects the entire contents of the file, this is the identifier
    /// for the macro that controls whether or not it has any effect.
    const IdentifierInfo *ControllingMacro;
    
    PerFileInfo() : isImport(false), DirInfo(DirectoryLookup::NormalHeaderDir),
      NumIncludes(0), ControllingMacro(0) {}
  };
  
  /// FileInfo - This contains all of the preprocessor-specific data about files
  /// that are included.  The vector is indexed by the FileEntry's UID.
  ///
  std::vector<PerFileInfo> FileInfo;

  /// FrameworkMap - This is a collection mapping a framework or subframework
  /// name like "Carbon" to the Carbon.framework directory.
  CStringMap<const DirectoryEntry *> FrameworkMap;

  // Various statistics we track for performance analysis.
  unsigned NumIncluded;
  unsigned NumMultiIncludeFileOptzn;
  unsigned NumFrameworkLookups, NumSubFrameworkLookups;
public:
  HeaderSearch(FileManager &FM);
    
  /// SetSearchPaths - Interface for setting the file search paths.
  ///
  void SetSearchPaths(const std::vector<DirectoryLookup> &dirs,
                      unsigned systemDirIdx, bool noCurDirSearch) {
    SearchDirs = dirs;
    SystemDirIdx = systemDirIdx;
    NoCurDirSearch = noCurDirSearch;
  }
  
  /// LookupFile - Given a "foo" or <foo> reference, look up the indicated file,
  /// return null on failure.  isAngled indicates whether the file reference is
  /// a <> reference.  If successful, this returns 'UsedDir', the
  /// DirectoryLookup member the file was found in, or null if not applicable.
  /// If CurDir is non-null, the file was found in the specified directory
  /// search location.  This is used to implement #include_next.  CurFileEnt, if
  /// non-null, indicates where the #including file is, in case a relative
  /// search is needed.
  const FileEntry *LookupFile(const std::string &Filename, bool isAngled,
                              const DirectoryLookup *FromDir,
                              const DirectoryLookup *&CurDir,
                              const FileEntry *CurFileEnt);
  
  /// LookupSubframeworkHeader - Look up a subframework for the specified
  /// #include file.  For example, if #include'ing <HIToolbox/HIToolbox.h> from
  /// within ".../Carbon.framework/Headers/Carbon.h", check to see if HIToolbox
  /// is a subframework within Carbon.framework.  If so, return the FileEntry
  /// for the designated file, otherwise return null.
  const FileEntry *LookupSubframeworkHeader(const std::string &Filename,
                                            const FileEntry *RelativeFileEnt);
  
  /// ShouldEnterIncludeFile - Mark the specified file as a target of of a
  /// #include, #include_next, or #import directive.  Return false if #including
  /// the file will have no effect or true if we should include it.
  bool ShouldEnterIncludeFile(const FileEntry *File, bool isImport);
  
  
  /// getFileDirFlavor - Return whether the specified file is a normal header,
  /// a system header, or a C++ friendly system header.
  DirectoryLookup::DirType getFileDirFlavor(const FileEntry *File) {
    return getFileInfo(File).DirInfo;
  }
    
  /// MarkFileIncludeOnce - Mark the specified file as a "once only" file, e.g.
  /// due to #pragma once.
  void MarkFileIncludeOnce(const FileEntry *File) {
    getFileInfo(File).isImport = true;
  }

  /// MarkFileSystemHeader - Mark the specified fiel as a system header, e.g.
  /// due to #pragma GCC system_header.
  void MarkFileSystemHeader(const FileEntry *File) {
    getFileInfo(File).DirInfo = DirectoryLookup::SystemHeaderDir;
  }
  
  /// SetFileControllingMacro - Mark the specified file as having a controlling
  /// macro.  This is used by the multiple-include optimization to eliminate
  /// no-op #includes.
  void SetFileControllingMacro(const FileEntry *File,
                               const IdentifierInfo *ControllingMacro) {
    getFileInfo(File).ControllingMacro = ControllingMacro;
  }
  
  void PrintStats();
private:
  const FileEntry *DoFrameworkLookup(const DirectoryEntry *Dir,
                                     const std::string &Filename);
      
  /// getFileInfo - Return the PerFileInfo structure for the specified
  /// FileEntry.
  PerFileInfo &getFileInfo(const FileEntry *FE);
};

}  // end namespace llvm
}  // end namespace clang

#endif
