//===--- PTHLexer.cpp - Lex from a token stream ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the PTHLexer interface.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/TokenKinds.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Lex/PTHLexer.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/PTHManager.h"
#include "clang/Lex/Token.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/OwningPtr.h"

using namespace clang;

PTHLexer::PTHLexer(Preprocessor& pp, SourceLocation fileloc, const char* D,
                   PTHManager& PM)
  : PreprocessorLexer(&pp, fileloc), TokBuf(D), CurTokenIdx(0), PTHMgr(PM), 
    NeedsFetching(true) {
    // Make sure the EofToken is completely clean.
    EofToken.startToken();
  }

Token PTHLexer::GetToken() {
  // Read the next token, or if we haven't advanced yet, get the last
  // token read.
  if (NeedsFetching) {
    NeedsFetching = false;
    ReadToken(LastFetched);
  }
  
  Token Tok = LastFetched;
  
  // If we are in raw mode, zero out identifier pointers.  This is
  // needed for 'pragma poison'.  Note that this requires that the Preprocessor
  // can go back to the original source when it calls getSpelling().
  if (LexingRawMode && Tok.is(tok::identifier))
    Tok.setIdentifierInfo(0);

  return Tok;
}

void PTHLexer::Lex(Token& Tok) {
LexNextToken:
  Tok = GetToken();
  
  if (AtLastToken()) {
    Preprocessor *PPCache = PP;

    if (LexEndOfFile(Tok))
      return;

    assert(PPCache && "Raw buffer::LexEndOfFile should return a token");
    return PPCache->Lex(Tok);
  }
  
  // Don't advance to the next token yet.  Check if we are at the
  // start of a new line and we're processing a directive.  If so, we
  // consume this token twice, once as an tok::eom.
  if (Tok.isAtStartOfLine() && ParsingPreprocessorDirective) {
    ParsingPreprocessorDirective = false;
    Tok.setKind(tok::eom);
    MIOpt.ReadToken();
    return;
  }
  
  // Advance to the next token.
  AdvanceToken();
    
  if (Tok.is(tok::hash)) {    
    if (Tok.isAtStartOfLine() && !LexingRawMode) {
      PP->HandleDirective(Tok);

      if (PP->isCurrentLexer(this))
        goto LexNextToken;

      return PP->Lex(Tok);
    }
  }

  MIOpt.ReadToken();
  
  if (Tok.is(tok::identifier)) {
    if (LexingRawMode) return;
    return PP->HandleIdentifier(Tok);
  }  
}

bool PTHLexer::LexEndOfFile(Token &Tok) {
  
  if (ParsingPreprocessorDirective) {
    ParsingPreprocessorDirective = false;
    Tok.setKind(tok::eom);
    MIOpt.ReadToken();
    return true; // Have a token.
  }
  
  if (LexingRawMode) {
    MIOpt.ReadToken();
    return true;  // Have an eof token.
  }
  
  // FIXME: Issue diagnostics similar to Lexer.
  return PP->HandleEndOfFile(Tok, false);
}

void PTHLexer::setEOF(Token& Tok) {
  assert(!EofToken.is(tok::eof));
  Tok = EofToken;
}

void PTHLexer::DiscardToEndOfLine() {
  assert(ParsingPreprocessorDirective && ParsingFilename == false &&
         "Must be in a preprocessing directive!");

  // Already at end-of-file?
  if (AtLastToken())
    return;

  // Find the first token that is not the start of the *current* line.
  Token T;
  for (Lex(T); !AtLastToken(); Lex(T))
    if (GetToken().isAtStartOfLine())
      return;
}

//===----------------------------------------------------------------------===//
// Utility methods for reading from the mmap'ed PTH file.
//===----------------------------------------------------------------------===//

static inline uint8_t Read8(const char*& data) {
  return (uint8_t) *(data++);
}

static inline uint32_t Read32(const char*& data) {
  uint32_t V = (uint32_t) Read8(data);
  V |= (((uint32_t) Read8(data)) << 8);
  V |= (((uint32_t) Read8(data)) << 16);
  V |= (((uint32_t) Read8(data)) << 24);
  return V;
}

//===----------------------------------------------------------------------===//
// Token reconstruction from the PTH file.
//===----------------------------------------------------------------------===//

void PTHLexer::ReadToken(Token& T) {
  // Clear the token.
  // FIXME: Setting the flags directly should obviate this step.
  T.startToken();
  
  // Read the type of the token.
  T.setKind((tok::TokenKind) Read8(TokBuf));
  
  // Set flags.  This is gross, since we are really setting multiple flags.
  T.setFlag((Token::TokenFlags) Read8(TokBuf));
  
  // Set the IdentifierInfo* (if any).
  T.setIdentifierInfo(PTHMgr.ReadIdentifierInfo(TokBuf));
  
  // Set the SourceLocation.  Since all tokens are constructed using a
  // raw lexer, they will all be offseted from the same FileID.
  T.setLocation(SourceLocation::getFileLoc(FileID, Read32(TokBuf)));
  
  // Finally, read and set the length of the token.
  T.setLength(Read32(TokBuf));
}

//===----------------------------------------------------------------------===//
// Internal Data Structures for PTH file lookup and resolving identifiers.
//===----------------------------------------------------------------------===//


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
                       const char* idDataTable, void* perIDCache, 
                       Preprocessor& pp)
: Buf(buf), PerIDCache(perIDCache), FileLookup(fileLookup),
  IdDataTable(idDataTable), ITable(pp.getIdentifierTable()), PP(pp) {}

PTHManager::~PTHManager() {
  delete Buf;
  delete (PTHFileLookup*) FileLookup;
  free(PerIDCache);
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
  
  // Get the number of IdentifierInfos and pre-allocate the identifier cache.
  uint32_t NumIds = Read32(IData);

  // Pre-allocate the peristent ID -> IdentifierInfo* cache.  We use calloc()
  // so that we in the best case only zero out memory once when the OS returns
  // us new pages.
  IdentifierInfo** PerIDCache =
    (IdentifierInfo**) calloc(NumIds, sizeof(*PerIDCache));
  
  if (!PerIDCache) {
    assert(false && "Could not allocate Persistent ID cache.");
    return 0;
  }
  
  // Create the new lexer.
  return new PTHManager(File.take(), FL.take(), IData, PerIDCache, PP);
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
  IdentifierInfo*& II = ((IdentifierInfo**) PerIDCache)[persistentID];
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
  return new PTHLexer(PP, SourceLocation::getFileLoc(FileID, 0), data, *this); 
}
