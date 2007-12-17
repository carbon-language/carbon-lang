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
class DirectoryEntry;
class HeaderMap;

/// DirectoryLookup - This class represents one entry in the search list that
/// specifies the search order for directories in #include directives.  It
/// represents either a directory or a 'headermap'.  A headermap is just like a
/// directory, but it remaps its contents through an indirection table instead
/// of indexing a directory.
class DirectoryLookup {
public:
  enum DirType {
    NormalHeaderDir,
    SystemHeaderDir,
    ExternCSystemHeaderDir
  };
private:
  union {  // This union is discriminated by isHeaderMap.
    /// Dir - This is the actual directory that we're referring to.
    ///
    const DirectoryEntry *Dir;
  
    /// Map - This is the HeaderMap corresponding if the isHeaderMap field is
    /// true.
    const HeaderMap *Map;
  } u;
  
  /// DirCharacteristic - The type of directory this is, one of the DirType enum
  /// values.
  DirType DirCharacteristic : 2;
  
  /// UserSupplied - True if this is a user-supplied directory.
  ///
  bool UserSupplied : 1;
  
  /// Framework - True if this is a framework directory search-path.
  ///
  bool Framework : 1;
  
  /// isHeaderMap - True if the HeaderMap field is valid, false if the Dir field
  /// is valid.
  bool isHeaderMap : 1;
public:
  /// DirectoryLookup ctor - Note that this ctor *does not take ownership* of
  /// 'dir'.
  DirectoryLookup(const DirectoryEntry *dir, DirType DT, bool isUser,
                  bool isFramework)
    : DirCharacteristic(DT), UserSupplied(isUser),
      Framework(isFramework), isHeaderMap(false) {
    u.Dir = dir; 
  }
  
  /// DirectoryLookup ctor - Note that this ctor *does not take ownership* of
  /// 'map'.
  DirectoryLookup(const HeaderMap *map, DirType DT, bool isUser, bool isFWork)
    : DirCharacteristic(DT), UserSupplied(isUser), Framework(isFWork), 
      isHeaderMap(true) {
    u.Map = map; 
  }
  
  /// getDir - Return the directory that this entry refers to.
  ///
  const DirectoryEntry *getDir() const { return !isHeaderMap ? u.Dir : 0; }
  
  /// getHeaderMap - Return the directory that this entry refers to.
  ///
  const HeaderMap *getHeaderMap() const { return isHeaderMap ? u.Map : 0; }
  
  /// DirCharacteristic - The type of directory this is, one of the DirType enum
  /// values.
  DirType getDirCharacteristic() const { return DirCharacteristic; }
  
  /// isUserSupplied - True if this is a user-supplied directory.
  ///
  bool isUserSupplied() const { return UserSupplied; }
  
  /// isFramework - True if this is a framework directory.
  ///
  bool isFramework() const { return Framework; }
};

}  // end namespace clang

#endif
