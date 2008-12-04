//===--- CacheTokens.cpp - Caching of lexer tokens for PCH support --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This provides a possible implementation of PCH support for Clang that is
// based on caching lexed tokens and identifiers.
//
//===----------------------------------------------------------------------===//

#include "clang.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/System/Path.h"

using namespace clang;

typedef uint32_t Offset;

typedef llvm::DenseMap<const FileEntry*,Offset> PCHMap;
typedef llvm::DenseMap<const IdentifierInfo*,uint32_t> IDMap;

static void Emit32(llvm::raw_ostream& Out, uint32_t V) {
  Out << (unsigned char)(V);
  Out << (unsigned char)(V >>  8);
  Out << (unsigned char)(V >> 16);
  Out << (unsigned char)(V >> 24);
}

static void Emit8(llvm::raw_ostream& Out, uint32_t V) {
  Out << (unsigned char)(V);
}

static void EmitBuf(llvm::raw_ostream& Out, const char* I, const char* E) {
  for ( ; I != E ; ++I) Out << *I;
}

static uint32_t ResolveID(IDMap& IM, uint32_t& idx, const IdentifierInfo* II) {
  
  // Null IdentifierInfo's map to the persistent ID 0.
  if (!II)
    return 0;
  
  IDMap::iterator I = IM.find(II);

  if (I == IM.end()) {
    IM[II] = ++idx; // Pre-increment since '0' is reserved for NULL.
    return idx;
  }
  
  return I->second; // We've already added 1.
}

static void EmitToken(llvm::raw_ostream& Out, const Token& T,
                      const SourceManager& SMgr,
                      uint32_t& idcount, IDMap& IM) {
  
  Emit8(Out, T.getKind());
  Emit8(Out, T.getFlags());
  Emit32(Out, ResolveID(IM, idcount, T.getIdentifierInfo()));
  Emit32(Out, SMgr.getFullFilePos(T.getLocation()));
  Emit32(Out, T.getLength());
}

struct IDData {
  const IdentifierInfo* II;
  uint32_t FileOffset;
  const IdentifierTable::const_iterator::value_type* Str;
};

static std::pair<Offset,Offset>
EmitIdentifierTable(llvm::raw_fd_ostream& Out, uint32_t max,
                    const IdentifierTable& T, const IDMap& IM) {

  // Build an inverse map from persistent IDs -> IdentifierInfo*.
  typedef std::vector<IDData> InverseIDMap;
  InverseIDMap IIDMap;
  IIDMap.resize(max);
  
  // Generate mapping from persistent IDs -> IdentifierInfo*.
  for (IDMap::const_iterator I=IM.begin(), E=IM.end(); I!=E; ++I) {
    // Decrement by 1 because we are using a vector for the lookup and
    // 0 is reserved for NULL.
    assert(I->second > 0);
    assert(I->second-1 < IIDMap.size());
    IIDMap[I->second-1].II = I->first;
  }

  // Get the string data associated with the IdentifierInfo.
  for (IdentifierTable::const_iterator I=T.begin(), E=T.end(); I!=E; ++I) {
    IDMap::const_iterator IDI = IM.find(&(I->getValue()));
    if (IDI == IM.end()) continue;
    IIDMap[IDI->second-1].Str = &(*I);
  }
  
  Offset DataOff = Out.tell();
    
  for (InverseIDMap::iterator I=IIDMap.begin(), E=IIDMap.end(); I!=E; ++I) {
    // Record the location for this data.
    I->FileOffset = Out.tell();
    // Write out the keyword.
    unsigned len = I->Str->getKeyLength();  
    Emit32(Out, len);
    const char* buf = I->Str->getKeyData();    
    EmitBuf(Out, buf, buf+len);  
  }
  
  // Now emit the table mapping from persistent IDs to PTH file offsets.  
  Offset IDOff = Out.tell();
  
  // Emit the number of identifiers.
  Emit32(Out, max);

  for (InverseIDMap::iterator I=IIDMap.begin(), E=IIDMap.end(); I!=E; ++I)
    Emit32(Out, I->FileOffset);

  return std::make_pair(DataOff, IDOff);
}

Offset EmitFileTable(llvm::raw_fd_ostream& Out, SourceManager& SM, PCHMap& PM) {
  
  Offset off = (Offset) Out.tell();
  
  // Output the size of the table.
  Emit32(Out, PM.size());

  for (PCHMap::iterator I=PM.begin(), E=PM.end(); I!=E; ++I) {
    const FileEntry* FE = I->first;
    const char* Name = FE->getName();
    unsigned size = strlen(Name);
    Emit32(Out, size);
    EmitBuf(Out, Name, Name+size);
    Emit32(Out, I->second);    
  }

  return off;
}

static Offset LexTokens(llvm::raw_fd_ostream& Out, Lexer& L, Preprocessor& PP,
                        uint32_t& idcount, IDMap& IM) {
  
  // Record the location within the token file.
  Offset off = (Offset) Out.tell();
  SourceManager& SMgr = PP.getSourceManager();

  Token Tok;
  
  do {
    L.LexFromRawLexer(Tok);
    
    if (Tok.is(tok::identifier)) {
      Tok.setIdentifierInfo(PP.LookUpIdentifierInfo(Tok));
    }
    else if (Tok.is(tok::hash) && Tok.isAtStartOfLine()) {
      // Special processing for #include.  Store the '#' token and lex
      // the next token.
      EmitToken(Out, Tok, SMgr, idcount, IM);
      L.LexFromRawLexer(Tok);
      
      // Did we see 'include'/'import'/'include_next'?
      if (!Tok.is(tok::identifier))
        continue;
      
      IdentifierInfo* II = PP.LookUpIdentifierInfo(Tok);
      Tok.setIdentifierInfo(II);
      tok::PPKeywordKind K = II->getPPKeywordID();
      
      if (K == tok::pp_include || K == tok::pp_import || 
          K == tok::pp_include_next) {
        
        // Save the 'include' token.
        EmitToken(Out, Tok, SMgr, idcount, IM);
        
        // Lex the next token as an include string.
        L.setParsingPreprocessorDirective(true);
        L.LexIncludeFilename(Tok); 
        L.setParsingPreprocessorDirective(false);
        
        if (Tok.is(tok::identifier))
          Tok.setIdentifierInfo(PP.LookUpIdentifierInfo(Tok));
      }
    }    
  }
  while (EmitToken(Out, Tok, SMgr, idcount, IM), Tok.isNot(tok::eof));

  return off;
}

void clang::CacheTokens(Preprocessor& PP, const std::string& OutFile) {
  // Lex through the entire file.  This will populate SourceManager with
  // all of the header information.
  Token Tok;
  PP.EnterMainSourceFile();
  do { PP.Lex(Tok); } while (Tok.isNot(tok::eof));
  
  // Iterate over all the files in SourceManager.  Create a lexer
  // for each file and cache the tokens.
  SourceManager& SM = PP.getSourceManager();
  const LangOptions& LOpts = PP.getLangOptions();
  llvm::raw_ostream& os = llvm::errs();  

  PCHMap PM;
  IDMap  IM;
  uint32_t idcount = 0;
  
  std::string ErrMsg;
  llvm::raw_fd_ostream Out(OutFile.c_str(), true, ErrMsg);
  
  if (!ErrMsg.empty()) {
    os << "PTH error: " << ErrMsg << "\n";
    return;
  }
  
  for (SourceManager::fileid_iterator I=SM.fileid_begin(), E=SM.fileid_end();
       I!=E; ++I) {
    
    const SrcMgr::ContentCache* C = I.getFileIDInfo().getContentCache();
    if (!C) continue;

    const FileEntry* FE = C->Entry;    // Does this entry correspond to a file?    
    if (!FE) continue;
    
    // FIXME: Handle files with non-absolute paths.
    llvm::sys::Path P(FE->getName());
    if (!P.isAbsolute())
      continue;

    PCHMap::iterator PI = PM.find(FE); // Have we already processed this file?
    if (PI != PM.end()) continue;
    
    const llvm::MemoryBuffer* B = C->Buffer;    
    if (!B) continue;
    
    Lexer L(SourceLocation::getFileLoc(I.getFileID(), 0), LOpts,
            B->getBufferStart(), B->getBufferEnd(), B);
    
    PM[FE] = LexTokens(Out, L, PP, idcount, IM);
  }

  // Write out the identifier table.
  std::pair<Offset,Offset> IdTableOff =
    EmitIdentifierTable(Out, idcount, PP.getIdentifierTable(), IM);
  
  // Write out the file table.
  Offset FileTableOff = EmitFileTable(Out, SM, PM);  
  
  // Finally, write out the offset table at the end.
  Emit32(Out, IdTableOff.first);
  Emit32(Out, IdTableOff.second);
  Emit32(Out, FileTableOff);
}
