//===--- PTHManager.cpp - Manager object for PTH processing -----*- C++ -*-===//
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

#include "clang/Lex/PTHManager.h"
#include "clang/Lex/Token.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/IdentifierTable.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/DenseMap.h"

using namespace clang;

//===----------------------------------------------------------------------===//
// Utility methods for reading from the mmap'ed PTH file.
//===----------------------------------------------------------------------===//

static uint8_t Read8(const char*& data) {
  return (uint8_t) *(data++);
}

static uint32_t Read32(const char*& data) {
  uint32_t V = (uint32_t) Read8(data);
  V |= (((uint32_t) Read8(data)) << 8);
  V |= (((uint32_t) Read8(data)) << 16);
  V |= (((uint32_t) Read8(data)) << 24);
  return V;
}

//===----------------------------------------------------------------------===//
// Internal Data Structures.
//===----------------------------------------------------------------------===//

typedef llvm::DenseMap<uint32_t, IdentifierInfo*> IDCache;

/// PTHFileLookup - This internal data structure is used by the PTHManager
///  to map from FileEntry objects managed by FileManager to offsets within
///  the PTH file.
namespace {
class VISIBILITY_HIDDEN PTHFileLookup {
public:
  class Val {
    uint32_t v;
    
  public:
    Val() : v(~0) {}
    Val(uint32_t x) : v(x) {}

    operator uint32_t() const {
      assert(v != ~((uint32_t)0) && "PTHFileLookup entry initialized.");
      return v;
    }
    
    Val& operator=(uint32_t x) { v = x; return *this; }
    bool isValid() const { return v != ~((uint32_t)0); }
  };
    
private:
  llvm::StringMap<Val> FileMap;

public:
  PTHFileLookup() {};
  
  Val Lookup(const FileEntry* FE) {
    const char* s = FE->getName();
    unsigned size = strlen(s);
    return FileMap.GetOrCreateValue(s, s+size).getValue();
  }
  
  void ReadTable(const char* D) {    
    uint32_t N = Read32(D);     // Read the length of the table.
    
    for ( ; N > 0; --N) {       // The rest of the data is the table itself.
      uint32_t len = Read32(D);
      const char* s = D;
      D += len;
      FileMap.GetOrCreateValue(s, s+len).getValue() = Read32(D);
    }
  }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// PTHManager methods.
//===----------------------------------------------------------------------===//

PTHManager::PTHManager(const llvm::MemoryBuffer* buf, void* fileLookup,
                       const char* idDataTable, Preprocessor& pp)
  : Buf(buf), PersistentIDCache(0), FileLookup(fileLookup),
    IdDataTable(idDataTable), ITable(pp.getIdentifierTable()), PP(pp) {}

PTHManager::~PTHManager() {
  delete Buf;
  delete (PTHFileLookup*) FileLookup;
  delete (IDCache*) PersistentIDCache;
}

PTHManager* PTHManager::Create(const std::string& file, Preprocessor& PP) {

  // Memory map the PTH file.
  llvm::OwningPtr<llvm::MemoryBuffer>
    File(llvm::MemoryBuffer::getFile(file.c_str()));

  if (!File)
    return 0;

  // Get the buffer ranges and check if there are at least three 32-bit
  // words at the end of the file.
  const char* BufBeg = File->getBufferStart();
  const char* BufEnd = File->getBufferEnd();
  
  if(!(BufEnd > BufBeg + sizeof(uint32_t)*3)) {
    assert(false && "Invalid PTH file.");
    return 0; // FIXME: Proper error diagnostic?
  }

  // Compute the address of the index table at the end of the PTH file.
  // This table contains the offset of the file lookup table, the
  // persistent ID -> identifer data table.
  const char* EndTable = BufEnd - sizeof(uint32_t)*3;

  // Construct the file lookup table.  This will be used for mapping from
  // FileEntry*'s to cached tokens.
  const char* FileTableOffset = EndTable + sizeof(uint32_t)*2;
  const char* FileTable = BufBeg + Read32(FileTableOffset);
  
  if (!(FileTable > BufBeg && FileTable < BufEnd)) {
    assert(false && "Invalid PTH file.");
    return 0; // FIXME: Proper error diagnostic?
  }
  
  llvm::OwningPtr<PTHFileLookup> FL(new PTHFileLookup());
  FL->ReadTable(FileTable);

  // Get the location of the table mapping from persistent ids to the
  // data needed to reconstruct identifiers.
  const char* IDTableOffset = EndTable + sizeof(uint32_t)*1;
  const char* IData = BufBeg + Read32(IDTableOffset);
  if (!(IData > BufBeg && IData < BufEnd)) {
    assert(false && "Invalid PTH file.");
    return 0; // FIXME: Proper error diagnostic?
  }

  return new PTHManager(File.take(), FL.take(), IData, PP);
}

IdentifierInfo* PTHManager::ReadIdentifierInfo(const char*& D) {
  // Read the persistent ID from the PTH file.
  uint32_t persistentID = Read32(D);

  // A persistent ID of '0' always maps to NULL.
  if (!persistentID)
    return 0;
  
  // Adjust the persistent ID by subtracting '1' so that it can be used
  // as an index within a table in the PTH file.
  --persistentID;
  
  // Check if the IdentifierInfo has already been resolved.
  if (!PersistentIDCache)
    PersistentIDCache = new IDCache();

  // FIXME: We can make this an array, but what is the performance tradeoff?
  IdentifierInfo*& II = (*((IDCache*) PersistentIDCache))[persistentID];
  if (II) return II;

  // Look in the PTH file for the string data for the IdentifierInfo object.
  const char* TableEntry = IdDataTable + sizeof(uint32_t) * persistentID;
  const char* IDData = Buf->getBufferStart() + Read32(TableEntry);
  assert(IDData < Buf->getBufferEnd());

  // Read the length of the string.
  uint32_t len = Read32(IDData);  
  
  // Get the IdentifierInfo* with the specified string.
  II = &ITable.get(IDData, IDData+len);
  return II;
}

void PTHManager::ReadToken(const char*& D, unsigned FileID, Token& T) {
  // Clear the token.
  // FIXME: Setting the flags directly should obviate this step.
  T.startToken();
  
  // Read the type of the token.
  T.setKind((tok::TokenKind) Read8(D));

  // Set flags.  This is gross, since we are really setting multiple flags.
  T.setFlag((Token::TokenFlags) Read8(D));
  
  // Set the IdentifierInfo* (if any).
  T.setIdentifierInfo(ReadIdentifierInfo(D));
  
  // Set the SourceLocation.  Since all tokens are constructed using a
  // raw lexer, they will all be offseted from the same FileID.
  T.setLocation(SourceLocation::getFileLoc(FileID, Read32(D)));

  // Finally, read and set the length of the token.
  T.setLength(Read32(D));
}

PTHLexer* PTHManager::CreateLexer(unsigned FileID, const FileEntry* FE) {
  
  if (!FE)
    return 0;
  
  // Lookup the FileEntry object in our file lookup data structure.  It will
  // return a variant that indicates whether or not there is an offset within
  // the PTH file that contains cached tokens.
  PTHFileLookup::Val Off = ((PTHFileLookup*) FileLookup)->Lookup(FE);

  if (!Off.isValid()) // No tokens available.
    return 0;

  // Compute the offset of the token data within the buffer.
  const char* data = Buf->getBufferStart() + Off;
  assert(data < Buf->getBufferEnd());
  
  // First cut: read the tokens from the file into a vector.
  // Later, stream them.  
  SourceLocation Loc = SourceLocation::getFileLoc(FileID, 0);
  llvm::OwningPtr<PTHLexer> L(new PTHLexer(PP, Loc));
  std::vector<Token>& Tokens = L->getTokens();
  
  Token T;
  do {
    ReadToken(data, FileID, T);
    Tokens.push_back(T);
  }
  while (T.isNot(tok::eof));

  // Return the lexer to the client.  The client assumes ownership of this
  // PTHLexer object.
  return L.take();
}
