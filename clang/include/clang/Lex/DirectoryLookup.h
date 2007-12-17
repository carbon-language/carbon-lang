//===--- DirectoryLookup.h - Info for searching for headers -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the DirectoryLookup interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LEX_DIRECTORYLOOKUP_H
#define LLVM_CLANG_LEX_DIRECTORYLOOKUP_H

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
  enum DirType {
    NormalHeaderDir,
    SystemHeaderDir,
    ExternCSystemHeaderDir
  };
  
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
  
  /// DirCharacteristic - The type of directory this is, one of the DirType enum
  /// values.
  DirType DirCharacteristic : 2;
  
  /// UserSupplied - True if this is a user-supplied directory.
  ///
  bool UserSupplied : 1;
  
  /// LookupType - This indicates whether this DirectoryLookup object is a
  /// normal directory, a framework, or a headermap.
  unsigned LookupType : 2;
public:
  /// DirectoryLookup ctor - Note that this ctor *does not take ownership* of
  /// 'dir'.
  DirectoryLookup(const DirectoryEntry *dir, DirType DT, bool isUser,
                  bool isFramework)
    : DirCharacteristic(DT), UserSupplied(isUser),
     LookupType(isFramework ? LT_Framework : LT_NormalDir) {
    u.Dir = dir; 
  }
  
  /// DirectoryLookup ctor - Note that this ctor *does not take ownership* of
  /// 'map'.
  DirectoryLookup(const HeaderMap *map, DirType DT, bool isUser)
    : DirCharacteristic(DT), UserSupplied(isUser), LookupType(LT_HeaderMap) {
    u.Map = map; 
  }
  
  /// LookupFile - Lookup the specified file in this search path, returning it
  /// if it exists or returning null if not.
  const FileEntry *LookupFile(const char *FilenameStart,
                              const char *FilenameEnd, HeaderSearch &HS) const;
  
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

  LookupType_t getLookupType() const { return (LookupType_t)LookupType; }
  
  /// isNormalDir - Return true if this is a normal directory, not a header map.
  bool isNormalDir() const { return getLookupType() == LT_NormalDir; }
  
  /// isFramework - True if this is a framework directory.
  ///
  bool isFramework() const { return getLookupType() == LT_Framework; }
  
  /// isHeaderMap - Return true if this is a header map, not a normal directory.
  bool isHeaderMap() const { return getLookupType() == LT_HeaderMap; }
  
  /// DirCharacteristic - The type of directory this is, one of the DirType enum
  /// values.
  DirType getDirCharacteristic() const { return DirCharacteristic; }
  
  /// isUserSupplied - True if this is a user-supplied directory.
  ///
  bool isUserSupplied() const { return UserSupplied; }
  
private:
  const FileEntry *DoFrameworkLookup(const char *FilenameStart,
                                     const char *FilenameEnd, 
                                     HeaderSearch &HS) const;
  
};

}  // end namespace clang

#endif
