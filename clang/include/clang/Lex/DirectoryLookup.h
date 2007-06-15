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

/// DirectoryLookup - This class is used to specify the search order for
/// directories in #include directives.
class DirectoryLookup {
public:
  enum DirType {
    NormalHeaderDir,
    SystemHeaderDir,
    ExternCSystemHeaderDir
  };
private:  
  /// Dir - This is the actual directory that we're referring to.
  ///
  const DirectoryEntry *Dir;
  
  /// DirCharacteristic - The type of directory this is, one of the DirType enum
  /// values.
  DirType DirCharacteristic : 2;
  
  /// UserSupplied - True if this is a user-supplied directory.
  ///
  bool UserSupplied : 1;
  
  /// Framework - True if this is a framework directory search-path.
  ///
  bool Framework : 1;
public:
  DirectoryLookup(const DirectoryEntry *dir, DirType DT, bool isUser,
                  bool isFramework)
    : Dir(dir), DirCharacteristic(DT), UserSupplied(isUser),
      Framework(isFramework) {}
  
  /// getDir - Return the directory that this entry refers to.
  ///
  const DirectoryEntry *getDir() const { return Dir; }
  
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
