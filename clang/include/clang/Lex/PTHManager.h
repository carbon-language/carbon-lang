//===--- PTHManager.h - Manager object for PTH processing -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the PTHManager interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_PTHMANAGER_H
#define LLVM_CLANG_PTHMANAGER_H

#include "clang/Lex/PTHLexer.h"
#include "clang/Basic/LangOptions.h"
#include <string>

namespace llvm {
  class MemoryBuffer;
}

namespace clang {

class FileEntry;
class IdentifierInfo;
class IdentifierTable;
class PTHLexer;
  
class PTHManager {
  
  friend class PTHLexer;
  
  /// The memory mapped PTH file.
  const llvm::MemoryBuffer* Buf;
  
  /// IdMap - A lazily generated cache mapping from persistent identifiers to
  ///  IdentifierInfo*.
  IdentifierInfo** PerIDCache;
  
  /// FileLookup - Abstract data structure used for mapping between files
  ///  and token data in the PTH file.
  void* FileLookup;
  
  /// IdDataTable - Array representing the mapping from persistent IDs to the
  ///  data offset within the PTH file containing the information to
  ///  reconsitute an IdentifierInfo.
  const char* IdDataTable;
  
  /// ITable - The IdentifierTable used for the translation unit being lexed.
  IdentifierTable& ITable;

  /// PP - The Preprocessor object that will use this PTHManager to create
  ///  PTHLexer objects.
  Preprocessor& PP;
  
  /// This constructor is intended to only be called by the static 'Create'
  /// method.
  PTHManager(const llvm::MemoryBuffer* buf, void* fileLookup,
             const char* idDataTable, IdentifierInfo** perIDCache,
             Preprocessor& pp);

  // Do not implement.
  PTHManager();
  void operator=(const PTHManager&);
  
  /// GetIdentifierInfo - Used by PTHManager to reconstruct IdentifierInfo
  ///  objects from the PTH file.
  IdentifierInfo* GetIdentifierInfo(unsigned);
  
  /// GetSpelling - Used by PTHLexer classes to get the cached spelling
  ///  for a token.
  unsigned GetSpelling(unsigned PTHOffset, const char*& Buffer);

public:
  
  ~PTHManager();
  
  /// Create - This method creates PTHManager objects.  The 'file' argument
  ///  is the name of the PTH file.  This method returns NULL upon failure.
  static PTHManager* Create(const std::string& file, Preprocessor& PP);
  
  /// CreateLexer - Return a PTHLexer that "lexes" the cached tokens for the
  ///  specified file.  This method returns NULL if no cached tokens exist.
  ///  It is the responsibility of the caller to 'delete' the returned object.
  PTHLexer* CreateLexer(unsigned FileID, const FileEntry* FE);  
};
  
}  // end namespace clang

#endif
