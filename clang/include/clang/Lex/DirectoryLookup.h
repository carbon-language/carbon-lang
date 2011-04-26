//===--- DirectoryLookup.h - Info for searching for headers -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the DirectoryLookup interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LEX_DIRECTORYLOOKUP_H
#define LLVM_CLANG_LEX_DIRECTORYLOOKUP_H

#include "clang/Basic/SourceManager.h"

namespace llvm {
  class StringRef;
  template <typename T> class SmallVectorImpl;
}
namespace clang {
class HeaderMap;
class DirectoryEntry;
class FileEntry;
class HeaderSearch;

/// DirectoryLookup - This class represents one entry in the search list that
/// specifies the search order for directories in #include directives.  It
/// represents either a directory, a framework, or a headermap.
///
class DirectoryLookup {
public:
  enum LookupType_t {
    LT_NormalDir,
    LT_Framework,
    LT_HeaderMap
  };
private:
  union {  // This union is discriminated by isHeaderMap.
    /// Dir - This is the actual directory that we're referring to for a normal
    /// directory or a framework.
    const DirectoryEntry *Dir;

    /// Map - This is the HeaderMap if this is a headermap lookup.
    ///
    const HeaderMap *Map;
  } u;

  /// DirCharacteristic - The type of directory this is: this is an instance of
  /// SrcMgr::CharacteristicKind.
  unsigned DirCharacteristic : 2;

  /// UserSupplied - True if this is a user-supplied directory.
  ///
  bool UserSupplied : 1;

  /// LookupType - This indicates whether this DirectoryLookup object is a
  /// normal directory, a framework, or a headermap.
  unsigned LookupType : 2;
public:
  /// DirectoryLookup ctor - Note that this ctor *does not take ownership* of
  /// 'dir'.
  DirectoryLookup(const DirectoryEntry *dir, SrcMgr::CharacteristicKind DT,
                  bool isUser, bool isFramework)
    : DirCharacteristic(DT), UserSupplied(isUser),
     LookupType(isFramework ? LT_Framework : LT_NormalDir) {
    u.Dir = dir;
  }

  /// DirectoryLookup ctor - Note that this ctor *does not take ownership* of
  /// 'map'.
  DirectoryLookup(const HeaderMap *map, SrcMgr::CharacteristicKind DT,
                  bool isUser)
    : DirCharacteristic(DT), UserSupplied(isUser), LookupType(LT_HeaderMap) {
    u.Map = map;
  }

  /// getLookupType - Return the kind of directory lookup that this is: either a
  /// normal directory, a framework path, or a HeaderMap.
  LookupType_t getLookupType() const { return (LookupType_t)LookupType; }

  /// getName - Return the directory or filename corresponding to this lookup
  /// object.
  const char *getName() const;

  /// getDir - Return the directory that this entry refers to.
  ///
  const DirectoryEntry *getDir() const { return isNormalDir() ? u.Dir : 0; }

  /// getFrameworkDir - Return the directory that this framework refers to.
  ///
  const DirectoryEntry *getFrameworkDir() const {
    return isFramework() ? u.Dir : 0;
  }

  /// getHeaderMap - Return the directory that this entry refers to.
  ///
  const HeaderMap *getHeaderMap() const { return isHeaderMap() ? u.Map : 0; }

  /// isNormalDir - Return true if this is a normal directory, not a header map.
  bool isNormalDir() const { return getLookupType() == LT_NormalDir; }

  /// isFramework - True if this is a framework directory.
  ///
  bool isFramework() const { return getLookupType() == LT_Framework; }

  /// isHeaderMap - Return true if this is a header map, not a normal directory.
  bool isHeaderMap() const { return getLookupType() == LT_HeaderMap; }

  /// DirCharacteristic - The type of directory this is, one of the DirType enum
  /// values.
  SrcMgr::CharacteristicKind getDirCharacteristic() const {
    return (SrcMgr::CharacteristicKind)DirCharacteristic;
  }

  /// isUserSupplied - True if this is a user-supplied directory.
  ///
  bool isUserSupplied() const { return UserSupplied; }


  /// LookupFile - Lookup the specified file in this search path, returning it
  /// if it exists or returning null if not.
  ///
  /// \param Filename The file to look up relative to the search paths.
  ///
  /// \param HS The header search instance to search with.
  ///
  /// \param SearchPath If not NULL, will be set to the search path relative
  /// to which the file was found.
  ///
  /// \param RelativePath If not NULL, will be set to the path relative to
  /// SearchPath at which the file was found. This only differs from the
  /// Filename for framework includes.
  const FileEntry *LookupFile(llvm::StringRef Filename, HeaderSearch &HS,
                              llvm::SmallVectorImpl<char> *SearchPath,
                              llvm::SmallVectorImpl<char> *RelativePath) const;

private:
  const FileEntry *DoFrameworkLookup(
      llvm::StringRef Filename, HeaderSearch &HS,
      llvm::SmallVectorImpl<char> *SearchPath,
      llvm::SmallVectorImpl<char> *RelativePath) const;

};

}  // end namespace clang

#endif
