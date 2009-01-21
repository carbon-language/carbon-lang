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

#define DISK_TOKEN_SIZE (1+1+2+4+4)

//===----------------------------------------------------------------------===//
// Utility methods for reading from the mmap'ed PTH file.
//===----------------------------------------------------------------------===//

static inline uint8_t Read8(const unsigned char *&Data) {
  uint8_t V = Data[0];
  Data += 1;
  return V;
}

static inline uint16_t Read16(const unsigned char *&Data) {
// Targets that directly support unaligned little-endian 16-bit loads can just
// use them.
#if defined(__i386__) || defined(__x86_64__)
  uint16_t V = *((uint16_t*)Data);
#else
  uint16_t V = ((uint16_t)Data[0] <<  0) |
               ((uint16_t)Data[1] <<  8);
#endif
  Data += 2;
  return V;
}

static inline uint32_t Read32(const unsigned char *&Data) {
// Targets that directly support unaligned little-endian 32-bit loads can just
// use them.
#if defined(__i386__) || defined(__x86_64__)
  uint32_t V = *((uint32_t*)Data);
#else
  uint32_t V = ((uint32_t)Data[0] <<  0) |
               ((uint32_t)Data[1] <<  8) |
               ((uint32_t)Data[2] << 16) |
               ((uint32_t)Data[3] << 24);
#endif
  Data += 4;
  return V;
}


//===----------------------------------------------------------------------===//
// PTHLexer methods.
//===----------------------------------------------------------------------===//

PTHLexer::PTHLexer(Preprocessor &PP, FileID FID, const unsigned char *D,
                   const unsigned char *ppcond,
                   PTHSpellingSearch &mySpellingSrch, PTHManager &PM)
  : PreprocessorLexer(&PP, FID), TokBuf(D), CurPtr(D), LastHashTokPtr(0),
    PPCond(ppcond), CurPPCondPtr(ppcond), MySpellingSrch(mySpellingSrch),
    PTHMgr(PM) {
      
  FileStartLoc = PP.getSourceManager().getLocForStartOfFile(FID);
}

void PTHLexer::Lex(Token& Tok) {
LexNextToken:

  //===--------------------------------------==//
  // Read the raw token data.
  //===--------------------------------------==//
  
  // Shadow CurPtr into an automatic variable.
  const unsigned *CurPtrShadow = (const unsigned *)CurPtr;  

  // Read in the data for the token.
  unsigned Word0 = CurPtrShadow[0];
  unsigned IdentifierID = CurPtrShadow[1];
  unsigned FileOffset = CurPtrShadow[2];
  
  tok::TokenKind TKind = (tok::TokenKind) (Word0 & 0xFF);
  Token::TokenFlags TFlags = (Token::TokenFlags) ((Word0 >> 8) & 0xFF);
  unsigned Len = Word0 >> 16;

  CurPtr = (const unsigned char*)(CurPtrShadow+3);
  
  //===--------------------------------------==//
  // Construct the token itself.
  //===--------------------------------------==//
  
  Tok.startToken();
  Tok.setKind(TKind);
  Tok.setFlag(TFlags);
  assert(!LexingRawMode);
  if (IdentifierID)
    Tok.setIdentifierInfo(PTHMgr.GetIdentifierInfo(IdentifierID-1));
  Tok.setLocation(FileStartLoc.getFileLocWithOffset(FileOffset));
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

  if (TKind == tok::identifier) {
    MIOpt.ReadToken();
    return PP->HandleIdentifier(Tok);
  }
  
  if (TKind == tok::eof) {
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
  
  if (TKind == tok::hash && Tok.isAtStartOfLine()) {
    LastHashTokPtr = CurPtr - DISK_TOKEN_SIZE;
    assert(!LexingRawMode);
    PP->HandleDirective(Tok);
    
    if (PP->isCurrentLexer(this))
      goto LexNextToken;
    
    return PP->Lex(Tok);
  }
  
  if (TKind == tok::eom) {
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
  const unsigned char* p = CurPtr;
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
  
  const unsigned char* HashEntryI = 0;
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
      const unsigned char* NextPPCondPtr =
        PPCond + TableIdx*(sizeof(uint32_t)*2);
      assert(NextPPCondPtr >= CurPPCondPtr);
      // Read where we should jump to.
      uint32_t TmpOffset = Read32(NextPPCondPtr);
      const unsigned char* HashEntryJ = TokBuf + TmpOffset;
      
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
  const unsigned char* NextPPCondPtr = PPCond + TableIdx*(sizeof(uint32_t)*2);
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
  assert(((tok::TokenKind)*CurPtr) == tok::hash);
  CurPtr += DISK_TOKEN_SIZE;
  
  // Did we reach a #endif?  If so, go ahead and consume that token as well.
  if (isEndif) { CurPtr += DISK_TOKEN_SIZE*2; }

  return isEndif;
}

SourceLocation PTHLexer::getSourceLocation() {
  // getSourceLocation is not on the hot path.  It is used to get the location
  // of the next token when transitioning back to this lexer when done
  // handling a #included file.  Just read the necessary data from the token
  // data buffer to construct the SourceLocation object.
  // NOTE: This is a virtual function; hence it is defined out-of-line.
  const unsigned char *OffsetPtr = CurPtr + (1 + 1 + 3);
  uint32_t Offset = Read32(OffsetPtr);
  return FileStartLoc.getFileLocWithOffset(Offset);
}

//===----------------------------------------------------------------------===//
// getSpelling() - Use cached data in PTH files for getSpelling().
//===----------------------------------------------------------------------===//

unsigned PTHManager::getSpelling(FileID FID, unsigned FPos,
                                 const char *&Buffer) {
  llvm::DenseMap<FileID, PTHSpellingSearch*>::iterator I =SpellingMap.find(FID);

  if (I == SpellingMap.end())
    return 0;

  return I->second->getSpellingBinarySearch(FPos, Buffer);
}

unsigned PTHManager::getSpelling(SourceLocation Loc, const char *&Buffer) {
  SourceManager &SM = PP->getSourceManager();
  Loc = SM.getSpellingLoc(Loc);
  std::pair<FileID, unsigned> LocInfo = SM.getDecomposedFileLoc(Loc);
  return getSpelling(LocInfo.first, LocInfo.second, Buffer);
}

unsigned PTHManager::getSpellingAtPTHOffset(unsigned PTHOffset,
                                            const char *&Buffer) {
  assert(PTHOffset < Buf->getBufferSize());
  const unsigned char* Ptr =
    (const unsigned char*)Buf->getBufferStart() + PTHOffset;
  
  // The string is prefixed by 16 bits for its length, followed by the string
  // itself.
  unsigned Len = Read16(Ptr);
  Buffer = (const char *)Ptr;
  return Len;
}

unsigned PTHSpellingSearch::getSpellingLinearSearch(unsigned FPos,
                                                    const char *&Buffer) {
  const unsigned char *Ptr = LinearItr;
  unsigned Len = 0;
  
  if (Ptr == TableEnd)
    return getSpellingBinarySearch(FPos, Buffer);
  
  do {
    uint32_t TokOffset = Read32(Ptr);
    
    if (TokOffset > FPos)
      return getSpellingBinarySearch(FPos, Buffer);
    
    // Did we find a matching token offset for this spelling?
    if (TokOffset == FPos) {
      uint32_t SpellingPTHOffset = Read32(Ptr);
      Len = PTHMgr.getSpellingAtPTHOffset(SpellingPTHOffset, Buffer);
      break;
    }
  } while (Ptr != TableEnd);

  LinearItr = Ptr;
  return Len;
}


unsigned PTHSpellingSearch::getSpellingBinarySearch(unsigned FPos,
                                                    const char *&Buffer) {
  
  assert((TableEnd - TableBeg) % SpellingEntrySize == 0);
  assert(TableEnd >= TableBeg);
  
  if (TableEnd == TableBeg)
    return 0;
  
  unsigned min = 0;
  const unsigned char *tb = TableBeg;
  unsigned max = NumSpellings;

  do {
    unsigned i = (max - min) / 2 + min;
    const unsigned char *Ptr = tb + (i * SpellingEntrySize);
    
    uint32_t TokOffset = Read32(Ptr);
    if (TokOffset > FPos) {
      max = i;
      assert(!(max == min) || (min == i));
      continue;
    }
    
    if (TokOffset < FPos) {
      if (i == min)
        break;
      
      min = i;
      continue;
    }
    
    uint32_t SpellingPTHOffset = Read32(Ptr);
    return PTHMgr.getSpellingAtPTHOffset(SpellingPTHOffset, Buffer);
  }
  while (min != max);
  
  return 0;
}

unsigned PTHLexer::getSpelling(SourceLocation Loc, const char *&Buffer) {
  SourceManager &SM = PP->getSourceManager();
  Loc = SM.getSpellingLoc(Loc);
  std::pair<FileID, unsigned> LocInfo = SM.getDecomposedFileLoc(Loc);

  FileID FID = LocInfo.first;
  unsigned FPos = LocInfo.second;
  
  if (FID == getFileID())
    return MySpellingSrch.getSpellingLinearSearch(FPos, Buffer);
  return PTHMgr.getSpelling(FID, FPos, Buffer);  
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
    
    bool isValid() const { return TokenOff != ~((uint32_t)0); }

    uint32_t getTokenOffset() const {
      assert(isValid() && "PTHFileLookup entry initialized.");
      return TokenOff;
    }
    
    uint32_t getPPCondOffset() const {
      assert(isValid() && "PTHFileLookup entry initialized.");
      return PPCondOff;
    }
    
    uint32_t getSpellingOffset() const {
      assert(isValid() && "PTHFileLookup entry initialized.");
      return SpellingOff;
    }
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
  
  void ReadTable(const unsigned char* D) {    
    uint32_t N = Read32(D);     // Read the length of the table.
    
    for ( ; N > 0; --N) {       // The rest of the data is the table itself.
      uint32_t Len = Read32(D);
      const char* s = (const char *)D;
      D += Len;

      uint32_t TokenOff = Read32(D);
      uint32_t PPCondOff = Read32(D);
      uint32_t SpellingOff = Read32(D);

      FileMap.GetOrCreateValue(s, s+Len).getValue() =
        Val(TokenOff, PPCondOff, SpellingOff);      
    }
  }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// PTHManager methods.
//===----------------------------------------------------------------------===//

PTHManager::PTHManager(const llvm::MemoryBuffer* buf, void* fileLookup,
                       const unsigned char* idDataTable,
                       IdentifierInfo** perIDCache, 
                       const unsigned char* sortedIdTable, unsigned numIds)
: Buf(buf), PerIDCache(perIDCache), FileLookup(fileLookup),
  IdDataTable(idDataTable), SortedIdTable(sortedIdTable),
  NumIds(numIds), PP(0) {}

PTHManager::~PTHManager() {
  delete Buf;
  delete (PTHFileLookup*) FileLookup;
  free(PerIDCache);
}

PTHManager* PTHManager::Create(const std::string& file) {
  // Memory map the PTH file.
  llvm::OwningPtr<llvm::MemoryBuffer>
  File(llvm::MemoryBuffer::getFile(file.c_str()));
  
  if (!File)
    return 0;
  
  // Get the buffer ranges and check if there are at least three 32-bit
  // words at the end of the file.
  const unsigned char* BufBeg = (unsigned char*)File->getBufferStart();
  const unsigned char* BufEnd = (unsigned char*)File->getBufferEnd();
  
  if(!(BufEnd > BufBeg + sizeof(uint32_t)*3)) {
    assert(false && "Invalid PTH file.");
    return 0; // FIXME: Proper error diagnostic?
  }
  
  // Compute the address of the index table at the end of the PTH file.
  // This table contains the offset of the file lookup table, the
  // persistent ID -> identifer data table.
  // FIXME: We should just embed this offset in the PTH file.
  const unsigned char* EndTable = BufEnd - sizeof(uint32_t)*4;
  
  // Construct the file lookup table.  This will be used for mapping from
  // FileEntry*'s to cached tokens.
  const unsigned char* FileTableOffset = EndTable + sizeof(uint32_t)*3;
  const unsigned char* FileTable = BufBeg + Read32(FileTableOffset);
  
  if (!(FileTable > BufBeg && FileTable < BufEnd)) {
    assert(false && "Invalid PTH file.");
    return 0; // FIXME: Proper error diagnostic?
  }
  
  llvm::OwningPtr<PTHFileLookup> FL(new PTHFileLookup());
  FL->ReadTable(FileTable);
  
  // Get the location of the table mapping from persistent ids to the
  // data needed to reconstruct identifiers.
  const unsigned char* IDTableOffset = EndTable + sizeof(uint32_t)*1;
  const unsigned char* IData = BufBeg + Read32(IDTableOffset);
  if (!(IData > BufBeg && IData < BufEnd)) {
    assert(false && "Invalid PTH file.");
    return 0; // FIXME: Proper error diagnostic?
  }
  
  // Get the location of the lexigraphically-sorted table of persistent IDs.
  const unsigned char* SortedIdTableOffset = EndTable + sizeof(uint32_t)*2;
  const unsigned char* SortedIdTable = BufBeg + Read32(SortedIdTableOffset);
  if (!(SortedIdTable > BufBeg && SortedIdTable < BufEnd)) {
    assert(false && "Invalid PTH file.");
    return 0; // FIXME: Proper error diagnostic?
  }
  
  // Get the number of IdentifierInfos and pre-allocate the identifier cache.
  uint32_t NumIds = Read32(IData);

  // Pre-allocate the peristent ID -> IdentifierInfo* cache.  We use calloc()
  // so that we in the best case only zero out memory once when the OS returns
  // us new pages.
  IdentifierInfo** PerIDCache =
    (IdentifierInfo**)calloc(NumIds, sizeof(*PerIDCache));
  
  if (!PerIDCache) {
    assert(false && "Could not allocate Persistent ID cache.");
    return 0;
  }
  
  // Create the new PTHManager.
  return new PTHManager(File.take(), FL.take(), IData, PerIDCache,
                        SortedIdTable, NumIds);
}
IdentifierInfo* PTHManager::LazilyCreateIdentifierInfo(unsigned PersistentID) {
  // Look in the PTH file for the string data for the IdentifierInfo object.
  const unsigned char* TableEntry = IdDataTable + sizeof(uint32_t)*PersistentID;
  const unsigned char* IDData =
    (const unsigned char*)Buf->getBufferStart() + Read32(TableEntry);
  assert(IDData < (const unsigned char*)Buf->getBufferEnd());
  
  // Allocate the object.
  std::pair<IdentifierInfo,const unsigned char*> *Mem =
    Alloc.Allocate<std::pair<IdentifierInfo,const unsigned char*> >();

  Mem->second = IDData;
  IdentifierInfo *II = new ((void*) Mem) IdentifierInfo();
  
  // Store the new IdentifierInfo in the cache.
  PerIDCache[PersistentID] = II;
  return II;
}

IdentifierInfo* PTHManager::get(const char *NameStart, const char *NameEnd) {
  unsigned min = 0;
  unsigned max = NumIds;
  unsigned Len = NameEnd - NameStart;
  
  do {
    unsigned i = (max - min) / 2 + min;
    const unsigned char *Ptr = SortedIdTable + (i * 4);
    
    // Read the persistentID.
    unsigned perID = Read32(Ptr);
    
    // Get the IdentifierInfo.
    IdentifierInfo* II = GetIdentifierInfo(perID);
    
    // First compare the lengths.
    unsigned IILen = II->getLength();
    if (Len < IILen) goto IsLess;
    if (Len > IILen) goto IsGreater;
    
    // Now compare the strings!
    {
      signed comp = strncmp(NameStart, II->getName(), Len);
      if (comp < 0) goto IsLess;
      if (comp > 0) goto IsGreater;
    }    
    // We found a match!
    return II;
    
  IsGreater:
    if (i == min) break;
    min = i;
    continue;
    
  IsLess:
    max = i;
    assert(!(max == min) || (min == i));
  }
  while (min != max);
  
  return 0;
}


PTHLexer *PTHManager::CreateLexer(FileID FID) {
  const FileEntry *FE = PP->getSourceManager().getFileEntryForID(FID);
  if (!FE)
    return 0;
  
  // Lookup the FileEntry object in our file lookup data structure.  It will
  // return a variant that indicates whether or not there is an offset within
  // the PTH file that contains cached tokens.
  PTHFileLookup::Val FileData = ((PTHFileLookup*)FileLookup)->Lookup(FE);
  
  if (!FileData.isValid()) // No tokens available.
    return 0;
  
  const unsigned char *BufStart = (const unsigned char *)Buf->getBufferStart();
  // Compute the offset of the token data within the buffer.
  const unsigned char* data = BufStart + FileData.getTokenOffset();

  // Get the location of pp-conditional table.
  const unsigned char* ppcond = BufStart + FileData.getPPCondOffset();
  uint32_t Len = Read32(ppcond);
  if (Len == 0) ppcond = 0;
  
  // Get the location of the spelling table.
  const unsigned char* spellingTable = BufStart + FileData.getSpellingOffset();
  
  Len = Read32(spellingTable);
  if (Len == 0) spellingTable = 0;

  assert(data < (const unsigned char*)Buf->getBufferEnd());
  
  // Create the SpellingSearch object for this FileID.
  PTHSpellingSearch* ss = new PTHSpellingSearch(*this, Len, spellingTable);
  SpellingMap[FID] = ss;
  
  assert(PP && "No preprocessor set yet!");
  return new PTHLexer(*PP, FID, data, ppcond, *ss, *this); 
}
