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

#define DISK_TOKEN_SIZE (1+1+3+4+2)

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
                   const char* ppcond,
                   PTHSpellingSearch& mySpellingSrch,
                   PTHManager& PM)
  : PreprocessorLexer(&pp, fileloc), TokBuf(D), CurPtr(D), LastHashTokPtr(0),
    PPCond(ppcond), CurPPCondPtr(ppcond), MySpellingSrch(mySpellingSrch),
    PTHMgr(PM)
{      
  FileID = fileloc.getFileID();
}

void PTHLexer::Lex(Token& Tok) {
LexNextToken:

  //===--------------------------------------==//
  // Read the raw token data.
  //===--------------------------------------==//
  
  // Shadow CurPtr into an automatic variable.
  const unsigned char *CurPtrShadow = (const unsigned char*) CurPtr;  

  // Read in the data for the token.  14 bytes in total.
  tok::TokenKind k = (tok::TokenKind) CurPtrShadow[0];
  Token::TokenFlags flags = (Token::TokenFlags) CurPtrShadow[1];
    
  uint32_t perID = ((uint32_t) CurPtrShadow[2])
      | (((uint32_t) CurPtrShadow[3]) << 8)
      | (((uint32_t) CurPtrShadow[4]) << 16);
  
  uint32_t FileOffset = ((uint32_t) CurPtrShadow[5])
      | (((uint32_t) CurPtrShadow[6]) << 8)
      | (((uint32_t) CurPtrShadow[7]) << 16)
      | (((uint32_t) CurPtrShadow[8]) << 24);
  
  uint32_t Len = ((uint32_t) CurPtrShadow[9])
      | (((uint32_t) CurPtrShadow[10]) << 8);
  
  CurPtr = (const char*) (CurPtrShadow + DISK_TOKEN_SIZE);
  
  //===--------------------------------------==//
  // Construct the token itself.
  //===--------------------------------------==//
  
  Tok.startToken();
  Tok.setKind(k);
  Tok.setFlag(flags);
  assert(!LexingRawMode);
  Tok.setIdentifierInfo(perID ? PTHMgr.GetIdentifierInfo(perID-1) : 0);
  Tok.setLocation(SourceLocation::getFileLoc(FileID, FileOffset));
  Tok.setLength(Len);

  //===--------------------------------------==//
  // Process the token.
  //===--------------------------------------==//
#if 0  
  SourceManager& SM = PP->getSourceManager();
  llvm::cerr << SM.getFileEntryForID(FileID)->getName()
    << ':' << SM.getLogicalLineNumber(Tok.getLocation())
    << ':' << SM.getLogicalColumnNumber(Tok.getLocation())
    << '\n';
#endif  

  if (k == tok::identifier) {
    MIOpt.ReadToken();
    return PP->HandleIdentifier(Tok);
  }
  
  if (k == tok::eof) {
    // Save the end-of-file token.
    EofToken = Tok;
    
    Preprocessor *PPCache = PP;
    
    assert(!ParsingPreprocessorDirective);
    assert(!LexingRawMode);
    
    // FIXME: Issue diagnostics similar to Lexer.
    if (PP->HandleEndOfFile(Tok, false))
      return;
    
    assert(PPCache && "Raw buffer::LexEndOfFile should return a token");
    return PPCache->Lex(Tok);
  }
  
  if (k == tok::hash && Tok.isAtStartOfLine()) {
    LastHashTokPtr = CurPtr - DISK_TOKEN_SIZE;
    assert(!LexingRawMode);
    PP->HandleDirective(Tok);
    
    if (PP->isCurrentLexer(this))
      goto LexNextToken;
    
    return PP->Lex(Tok);
  }
  
  if (k == tok::eom) {
    assert(ParsingPreprocessorDirective);
    ParsingPreprocessorDirective = false;
    return;
  }

  MIOpt.ReadToken();
}

// FIXME: We can just grab the last token instead of storing a copy
// into EofToken.
void PTHLexer::getEOF(Token& Tok) {
  assert(EofToken.is(tok::eof));
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
  const char* p = CurPtr + (1 + 1 + 3);
  uint32_t offset = 
       ((uint32_t) ((uint8_t) p[0]))
    | (((uint32_t) ((uint8_t) p[1])) << 8)
    | (((uint32_t) ((uint8_t) p[2])) << 16)
    | (((uint32_t) ((uint8_t) p[3])) << 24);
  return SourceLocation::getFileLoc(FileID, offset);
}

//===----------------------------------------------------------------------===//
// getSpelling() - Use cached data in PTH files for getSpelling().
//===----------------------------------------------------------------------===//

unsigned PTHManager::getSpelling(unsigned FileID, unsigned fpos,
                                 const char *& Buffer) {
  
  llvm::DenseMap<unsigned,PTHSpellingSearch*>::iterator I =
    SpellingMap.find(FileID);

  if (I == SpellingMap.end())
      return 0;

  return I->second->getSpellingBinarySearch(fpos, Buffer);
}

unsigned PTHManager::getSpellingAtPTHOffset(unsigned PTHOffset,
                                            const char *& Buffer) {

  const char* p = Buf->getBufferStart() + PTHOffset;
  assert(p < Buf->getBufferEnd());
  
  // The string is prefixed by 16 bits for its length, followed by the string
  // itself.
  unsigned len = ((unsigned) ((uint8_t) p[0]))
    | (((unsigned) ((uint8_t) p[1])) << 8);

  Buffer = p + 2;
  return len;
}

unsigned PTHSpellingSearch::getSpellingLinearSearch(unsigned fpos,
                                                    const char *&Buffer) {
  const char* p = LinearItr;
  unsigned len = 0;
  
  if (p == TableEnd)
    return getSpellingBinarySearch(fpos, Buffer);
  
  do {
    uint32_t TokOffset = 
      ((uint32_t) ((uint8_t) p[0]))
      | (((uint32_t) ((uint8_t) p[1])) << 8)
      | (((uint32_t) ((uint8_t) p[2])) << 16)
      | (((uint32_t) ((uint8_t) p[3])) << 24);
    
    if (TokOffset > fpos)
      return getSpellingBinarySearch(fpos, Buffer);
    
    // Did we find a matching token offset for this spelling?
    if (TokOffset == fpos) {
      uint32_t SpellingPTHOffset = 
        ((uint32_t) ((uint8_t) p[4]))
        | (((uint32_t) ((uint8_t) p[5])) << 8)
        | (((uint32_t) ((uint8_t) p[6])) << 16)
        | (((uint32_t) ((uint8_t) p[7])) << 24);
      
      p += SpellingEntrySize;
      len = PTHMgr.getSpellingAtPTHOffset(SpellingPTHOffset, Buffer);
      break;
    }

    // No match.  Keep on looking.
    p += SpellingEntrySize;
  }
  while (p != TableEnd);

  LinearItr = p;
  return len;
}

unsigned PTHSpellingSearch::getSpellingBinarySearch(unsigned fpos,
                                                    const char *& Buffer) {
  
  assert((TableEnd - TableBeg) % SpellingEntrySize == 0);
  
  if (TableEnd == TableBeg)
    return 0;
  
  assert(TableEnd > TableBeg);
  
  unsigned min = 0;
  const char* tb = TableBeg;
  unsigned max = NumSpellings;

  do {
    unsigned i = (max - min) / 2 + min;
    const char* p = tb + (i * SpellingEntrySize);
    
    uint32_t TokOffset = 
      ((uint32_t) ((uint8_t) p[0]))
      | (((uint32_t) ((uint8_t) p[1])) << 8)
      | (((uint32_t) ((uint8_t) p[2])) << 16)
      | (((uint32_t) ((uint8_t) p[3])) << 24);
    
    if (TokOffset > fpos) {
      max = i;
      assert(!(max == min) || (min == i));
      continue;
    }
    
    if (TokOffset < fpos) {
      if (i == min)
        break;
      
      min = i;
      continue;
    }
    
    uint32_t SpellingPTHOffset = 
        ((uint32_t) ((uint8_t) p[4]))
        | (((uint32_t) ((uint8_t) p[5])) << 8)
        | (((uint32_t) ((uint8_t) p[6])) << 16)
        | (((uint32_t) ((uint8_t) p[7])) << 24);
    
    return PTHMgr.getSpellingAtPTHOffset(SpellingPTHOffset, Buffer);
  }
  while (min != max);
  
  return 0;
}

unsigned PTHLexer::getSpelling(SourceLocation sloc, const char *&Buffer) {
  SourceManager& SM = PP->getSourceManager();
  sloc = SM.getPhysicalLoc(sloc);
  unsigned fid = SM.getCanonicalFileID(sloc);
  unsigned fpos = SM.getFullFilePos(sloc);
  
  return (fid == FileID ) ? MySpellingSrch.getSpellingLinearSearch(fpos, Buffer)
                          : PTHMgr.getSpelling(fid, fpos, Buffer);  
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
    uint32_t SpellingOff;
    
  public:
    Val() : TokenOff(~0) {}
    Val(uint32_t toff, uint32_t poff, uint32_t soff)
      : TokenOff(toff), PPCondOff(poff), SpellingOff(soff) {}
    
    uint32_t getTokenOffset() const {
      assert(TokenOff != ~((uint32_t)0) && "PTHFileLookup entry initialized.");
      return TokenOff;
    }
    
    uint32_t getPPCondOffset() const {
      assert(TokenOff != ~((uint32_t)0) && "PTHFileLookup entry initialized.");
      return PPCondOff;
    }
    
    uint32_t getSpellingOffset() const {
      assert(TokenOff != ~((uint32_t)0) && "PTHFileLookup entry initialized.");
      return SpellingOff;
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
      uint32_t PPCondOff = Read32(D);
      uint32_t SpellingOff = Read32(D);

      FileMap.GetOrCreateValue(s, s+len).getValue() =
        Val(TokenOff, PPCondOff, SpellingOff);      
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
  // FIXME: We should just embed this offset in the PTH file.
  const char* EndTable = BufEnd - sizeof(uint32_t)*4;
  
  // Construct the file lookup table.  This will be used for mapping from
  // FileEntry*'s to cached tokens.
  const char* FileTableOffset = EndTable + sizeof(uint32_t)*3;
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

IdentifierInfo* PTHManager::GetIdentifierInfo(unsigned persistentID) {
    
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
  const char* ppcond = Buf->getBufferStart() + FileData.getPPCondOffset();
  uint32_t len = Read32(ppcond);
  if (len == 0) ppcond = 0;
  
  // Get the location of the spelling table.
  const char* spellingTable = Buf->getBufferStart() +
                              FileData.getSpellingOffset();
  
  len = Read32(spellingTable);
  if (len == 0) spellingTable = 0;

  assert(data < Buf->getBufferEnd());
  
  // Create the SpellingSearch object for this FileID.
  PTHSpellingSearch* ss = new PTHSpellingSearch(*this, len, spellingTable);
  SpellingMap[FileID] = ss;
  
  return new PTHLexer(PP, SourceLocation::getFileLoc(FileID, 0), data, ppcond,
                      *ss, *this); 
}
