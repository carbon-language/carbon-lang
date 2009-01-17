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
#include "clang/Basic/IdentifierTable.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Allocator.h"
#include <string>

namespace llvm {
  class MemoryBuffer;
}

namespace clang {

class FileEntry;
class PTHLexer;
class PTHManager;

class PTHSpellingSearch {
  PTHManager& PTHMgr;
  
  const char* const TableBeg;
  const char* const TableEnd;
  
  const unsigned NumSpellings;
  const char* LinearItr;
  
public:
  enum { SpellingEntrySize = 4*2 };
  
  unsigned getSpellingBinarySearch(unsigned fpos, const char *&Buffer);
  unsigned getSpellingLinearSearch(unsigned fpos, const char *&Buffer);
  
  PTHSpellingSearch(PTHManager& pm, unsigned numSpellings, const char* tableBeg)
    : PTHMgr(pm),
      TableBeg(tableBeg),
      TableEnd(tableBeg + numSpellings*SpellingEntrySize),
      NumSpellings(numSpellings),
      LinearItr(tableBeg) {}
};  
  
class PTHManager : public IdentifierInfoLookup {
  friend class PTHLexer;
  friend class PTHSpellingSearch;
  
  /// The memory mapped PTH file.
  const llvm::MemoryBuffer* Buf;
  
  /// A map from FileIDs to SpellingSearch objects.
  llvm::DenseMap<FileID, PTHSpellingSearch*> SpellingMap;
  
  /// Alloc - Allocator used for IdentifierInfo objects.
  llvm::BumpPtrAllocator Alloc;
  
  /// IdMap - A lazily generated cache mapping from persistent identifiers to
  ///  IdentifierInfo*.
  IdentifierInfo** PerIDCache;
  
  /// FileLookup - Abstract data structure used for mapping between files
  ///  and token data in the PTH file.
  void* FileLookup;
  
  /// IdDataTable - Array representing the mapping from persistent IDs to the
  ///  data offset within the PTH file containing the information to
  ///  reconsitute an IdentifierInfo.
  const char* const IdDataTable;
  
  /// SortedIdTable - Array ordering persistent identifier IDs by the lexical
  ///  order of their corresponding strings.  This is used by get().
  const char* const SortedIdTable;
  
  /// NumIds - The number of identifiers in the PTH file.
  const unsigned NumIds;

  /// PP - The Preprocessor object that will use this PTHManager to create
  ///  PTHLexer objects.
  Preprocessor* PP;
  
  /// This constructor is intended to only be called by the static 'Create'
  /// method.
  PTHManager(const llvm::MemoryBuffer* buf, void* fileLookup,
             const char* idDataTable, IdentifierInfo** perIDCache,
             const char* sortedIdTable, unsigned numIds);

  // Do not implement.
  PTHManager();
  void operator=(const PTHManager&);
  
  /// getSpellingAtPTHOffset - Used by PTHLexer classes to get the cached 
  ///  spelling for a token.
  unsigned getSpellingAtPTHOffset(unsigned PTHOffset, const char*& Buffer);
  
  
  /// GetIdentifierInfo - Used to reconstruct IdentifierInfo objects from the
  ///  PTH file.
  IdentifierInfo* GetIdentifierInfo(unsigned);
  
public:  
  ~PTHManager();
  
  /// get - Return the identifier token info for the specified named identifier.
  ///  Unlike the version in IdentifierTable, this returns a pointer instead
  ///  of a reference.  If the pointer is NULL then the IdentifierInfo cannot
  ///  be found.
  IdentifierInfo *get(const char *NameStart, const char *NameEnd);
  
  /// Create - This method creates PTHManager objects.  The 'file' argument
  ///  is the name of the PTH file.  This method returns NULL upon failure.
  static PTHManager *Create(const std::string& file);

  void setPreprocessor(Preprocessor *pp) { PP = pp; }    
  
  /// CreateLexer - Return a PTHLexer that "lexes" the cached tokens for the
  ///  specified file.  This method returns NULL if no cached tokens exist.
  ///  It is the responsibility of the caller to 'delete' the returned object.
  PTHLexer *CreateLexer(FileID FID, const FileEntry *FE);
  
  unsigned getSpelling(SourceLocation Loc, const char *&Buffer);  
private:
  unsigned getSpelling(FileID FID, unsigned fpos, const char *& Buffer);  
};
  
}  // end namespace clang

#endif
