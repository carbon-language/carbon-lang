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

#define DISK_TOKEN_SIZE (2+3*4)

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
// PTHLexer methods.
//===----------------------------------------------------------------------===//

PTHLexer::PTHLexer(Preprocessor& pp, SourceLocation fileloc, const char* D,
                   const char* ppcond, PTHManager& PM)
  : PreprocessorLexer(&pp, fileloc), TokBuf(D), CurPtr(D), LastHashTokPtr(0),
    PPCond(ppcond), CurPPCondPtr(ppcond), PTHMgr(PM) {}

void PTHLexer::Lex(Token& Tok) {
LexNextToken:
  
  // Read the token.
  // FIXME: Setting the flags directly should obviate this step.
  Tok.startToken();
  
  // Shadow CurPtr into an automatic variable so that Read8 doesn't load and
  // store back into the instance variable.
  const char *CurPtrShadow = CurPtr;
  
  // Read the type of the token.
  Tok.setKind((tok::TokenKind) Read8(CurPtrShadow));
  
  // Set flags.  This is gross, since we are really setting multiple flags.
  Tok.setFlag((Token::TokenFlags) Read8(CurPtrShadow));
  
  // Set the IdentifierInfo* (if any).
  Tok.setIdentifierInfo(PTHMgr.ReadIdentifierInfo(CurPtrShadow));
  
  // Set the SourceLocation.  Since all tokens are constructed using a
  // raw lexer, they will all be offseted from the same FileID.
  Tok.setLocation(SourceLocation::getFileLoc(FileID, Read32(CurPtrShadow)));
  
  // Finally, read and set the length of the token.
  Tok.setLength(Read32(CurPtrShadow));

  CurPtr = CurPtrShadow;

  if (Tok.is(tok::eof)) {
    // Save the end-of-file token.
    EofToken = Tok;
    
    Preprocessor *PPCache = PP;

    if (LexEndOfFile(Tok))
      return;

    assert(PPCache && "Raw buffer::LexEndOfFile should return a token");
    return PPCache->Lex(Tok);
  }

  MIOpt.ReadToken();
  
  if (Tok.is(tok::eom)) {
    ParsingPreprocessorDirective = false;
    return;
  }
  
#if 0
  SourceManager& SM = PP->getSourceManager();
  SourceLocation L = Tok.getLocation();
  
  static const char* last = 0;
  const char* next = SM.getContentCacheForLoc(L)->Entry->getName();
  if (next != last) {
    last = next;
    llvm::cerr << next << '\n';
  }

  llvm::cerr << "line " << SM.getLogicalLineNumber(L) << " col " <<
  SM.getLogicalColumnNumber(L) << '\n';
#endif
    
  if (Tok.is(tok::hash)) {    
    if (Tok.isAtStartOfLine()) {
      LastHashTokPtr = CurPtr - DISK_TOKEN_SIZE;
      if (!LexingRawMode) {
        PP->HandleDirective(Tok);

        if (PP->isCurrentLexer(this))
          goto LexNextToken;
        
        return PP->Lex(Tok);
      }
    }
  }
  
  if (Tok.is(tok::identifier)) {
    if (LexingRawMode) {
      Tok.setIdentifierInfo(0);
      return;
    }
    
    return PP->HandleIdentifier(Tok);
  }

  
  assert(!Tok.is(tok::eom) || ParsingPreprocessorDirective);
}

// FIXME: This method can just be inlined into Lex().
bool PTHLexer::LexEndOfFile(Token &Tok) {
  assert(!ParsingPreprocessorDirective);
  assert(!LexingRawMode);
  
  // FIXME: Issue diagnostics similar to Lexer.
  return PP->HandleEndOfFile(Tok, false);
}

// FIXME: We can just grab the last token instead of storing a copy
// into EofToken.
void PTHLexer::setEOF(Token& Tok) {
  assert(!EofToken.is(tok::eof));
  Tok = EofToken;
}

void PTHLexer::DiscardToEndOfLine() {
  assert(ParsingPreprocessorDirective && ParsingFilename == false &&
         "Must be in a preprocessing directive!");

  // We assume that if the preprocessor wishes to discard to the end of
  // the line that it also means to end the current preprocessor directive.
  ParsingPreprocessorDirective = false;
  
  // Skip tokens by only peeking at their token kind and the flags.
  // We don't need to actually reconstruct full tokens from the token buffer.
  // This saves some copies and it also reduces IdentifierInfo* lookup.
  const char* p = CurPtr;
  while (1) {
    // Read the token kind.  Are we at the end of the file?
    tok::TokenKind x = (tok::TokenKind) (uint8_t) *p;
    if (x == tok::eof) break;
    
    // Read the token flags.  Are we at the start of the next line?
    Token::TokenFlags y = (Token::TokenFlags) (uint8_t) p[1];
    if (y & Token::StartOfLine) break;

    // Skip to the next token.
    p += DISK_TOKEN_SIZE;
  }
  
  CurPtr = p;
}

/// SkipBlock - Used by Preprocessor to skip the current conditional block.
bool PTHLexer::SkipBlock() {
  assert(CurPPCondPtr && "No cached PP conditional information.");
  assert(LastHashTokPtr && "No known '#' token.");
  
  const char* HashEntryI = 0;
  uint32_t Offset; 
  uint32_t TableIdx;
  
  do {
    // Read the token offset from the side-table.
    Offset = Read32(CurPPCondPtr);
    
    // Read the target table index from the side-table.    
    TableIdx = Read32(CurPPCondPtr);
    
    // Compute the actual memory address of the '#' token data for this entry.
    HashEntryI = TokBuf + Offset;

    // Optmization: "Sibling jumping".  #if...#else...#endif blocks can
    //  contain nested blocks.  In the side-table we can jump over these
    //  nested blocks instead of doing a linear search if the next "sibling"
    //  entry is not at a location greater than LastHashTokPtr.
    if (HashEntryI < LastHashTokPtr && TableIdx) {
      // In the side-table we are still at an entry for a '#' token that
      // is earlier than the last one we saw.  Check if the location we would
      // stride gets us closer.
      const char* NextPPCondPtr = PPCond + TableIdx*(sizeof(uint32_t)*2);
      assert(NextPPCondPtr >= CurPPCondPtr);
      // Read where we should jump to.
      uint32_t TmpOffset = Read32(NextPPCondPtr);
      const char* HashEntryJ = TokBuf + TmpOffset;
      
      if (HashEntryJ <= LastHashTokPtr) {
        // Jump directly to the next entry in the side table.
        HashEntryI = HashEntryJ;
        Offset = TmpOffset;
        TableIdx = Read32(NextPPCondPtr);
        CurPPCondPtr = NextPPCondPtr;
      }
    }
  }
  while (HashEntryI < LastHashTokPtr);  
  assert(HashEntryI == LastHashTokPtr && "No PP-cond entry found for '#'");
  assert(TableIdx && "No jumping from #endifs.");
  
  // Update our side-table iterator.
  const char* NextPPCondPtr = PPCond + TableIdx*(sizeof(uint32_t)*2);
  assert(NextPPCondPtr >= CurPPCondPtr);
  CurPPCondPtr = NextPPCondPtr;
  
  // Read where we should jump to.
  HashEntryI = TokBuf + Read32(NextPPCondPtr);
  uint32_t NextIdx = Read32(NextPPCondPtr);
  
  // By construction NextIdx will be zero if this is a #endif.  This is useful
  // to know to obviate lexing another token.
  bool isEndif = NextIdx == 0;
  
  // This case can occur when we see something like this:
  //
  //  #if ...
  //   /* a comment or nothing */
  //  #elif
  //
  // If we are skipping the first #if block it will be the case that CurPtr
  // already points 'elif'.  Just return.
  
  if (CurPtr > HashEntryI) {
    assert(CurPtr == HashEntryI + DISK_TOKEN_SIZE);
    // Did we reach a #endif?  If so, go ahead and consume that token as well.
    if (isEndif)
      CurPtr += DISK_TOKEN_SIZE*2;
    else
      LastHashTokPtr = HashEntryI;
    
    return isEndif;
  }

  // Otherwise, we need to advance.  Update CurPtr to point to the '#' token.
  CurPtr = HashEntryI;
  
  // Update the location of the last observed '#'.  This is useful if we
  // are skipping multiple blocks.
  LastHashTokPtr = CurPtr;

  // Skip the '#' token.
  assert(((tok::TokenKind) (unsigned char) *CurPtr) == tok::hash);
  CurPtr += DISK_TOKEN_SIZE;
  
  // Did we reach a #endif?  If so, go ahead and consume that token as well.
  if (isEndif) { CurPtr += DISK_TOKEN_SIZE*2; }

  return isEndif;
}

SourceLocation PTHLexer::getSourceLocation() {
  // getLocation is not on the hot path.  It is used to get the location of
  // the next token when transitioning back to this lexer when done
  // handling a #included file.  Just read the necessary data from the token
  // data buffer to construct the SourceLocation object.
  // NOTE: This is a virtual function; hence it is defined out-of-line.
  const char* p = CurPtr + (1 + 1 + 4);
  uint32_t offset = 
       ((uint32_t) ((uint8_t) p[0]))
    | (((uint32_t) ((uint8_t) p[1])) << 8)
    | (((uint32_t) ((uint8_t) p[2])) << 16)
    | (((uint32_t) ((uint8_t) p[3])) << 24);
  return SourceLocation::getFileLoc(FileID, offset);
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
    uint32_t TokenOff;
    uint32_t PPCondOff;
    
  public:
    Val() : TokenOff(~0) {}
    Val(uint32_t toff, uint32_t poff) : TokenOff(toff), PPCondOff(poff) {}
    
    uint32_t getTokenOffset() const {
      assert(TokenOff != ~((uint32_t)0) && "PTHFileLookup entry initialized.");
      return TokenOff;
    }
    
    uint32_t gettPPCondOffset() const {
      assert(TokenOff != ~((uint32_t)0) && "PTHFileLookup entry initialized.");
      return PPCondOff;
    }
    
    bool isValid() const { return TokenOff != ~((uint32_t)0); }
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
      uint32_t TokenOff = Read32(D);
      FileMap.GetOrCreateValue(s, s+len).getValue() = Val(TokenOff, Read32(D));      
    }
  }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// PTHManager methods.
//===----------------------------------------------------------------------===//

PTHManager::PTHManager(const llvm::MemoryBuffer* buf, void* fileLookup,
                       const char* idDataTable, IdentifierInfo** perIDCache, 
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
  IdentifierInfo*& II = PerIDCache[persistentID];
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
  PTHFileLookup::Val FileData = ((PTHFileLookup*) FileLookup)->Lookup(FE);
  
  if (!FileData.isValid()) // No tokens available.
    return 0;
  
  // Compute the offset of the token data within the buffer.
  const char* data = Buf->getBufferStart() + FileData.getTokenOffset();

  // Get the location of pp-conditional table.
  const char* ppcond = Buf->getBufferStart() + FileData.gettPPCondOffset();
  uint32_t len = Read32(ppcond);  
  if (len == 0) ppcond = 0;
  
  assert(data < Buf->getBufferEnd());
  return new PTHLexer(PP, SourceLocation::getFileLoc(FileID, 0), data, ppcond,
                      *this); 
}
