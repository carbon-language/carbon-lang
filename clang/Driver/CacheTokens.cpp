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

using namespace clang;

typedef llvm::DenseMap<const FileEntry*,uint64_t> PCHMap;
typedef llvm::DenseMap<const IdentifierInfo*,uint64_t> IDMap;

static void Emit32(llvm::raw_ostream& Out, uint32_t V) {
#if 0
  Out << (unsigned char)(V);
  Out << (unsigned char)(V >>  8);
  Out << (unsigned char)(V >> 16);
  Out << (unsigned char)(V >> 24);
#else
  Out << V;
#endif
}

static void EmitOffset(llvm::raw_ostream& Out, uint64_t V) {
  assert(((uint32_t) V) == V && "Offset exceeds 32 bits.");
  Emit32(Out, (uint32_t) V);
}

static void Emit8(llvm::raw_ostream& Out, uint32_t V) {
  Out << (unsigned char)(V);
}

static void EmitBuf(llvm::raw_ostream& Out, const char* I, const char* E) {
  for ( ; I != E ; ++I) Out << *I;
}

static uint32_t ResolveID(IDMap& IM, uint32_t& idx, const IdentifierInfo* II) {
  IDMap::iterator I = IM.find(II);

  if (I == IM.end()) {
    IM[II] = idx;
    return idx++;
  }
  
  return I->second;
}

static void EmitToken(llvm::raw_ostream& Out, const Token& T,
                      uint32_t& idcount, IDMap& IM) {
  Emit8(Out, T.getKind());
  Emit8(Out, T.getFlags());
  Emit32(Out, ResolveID(IM, idcount, T.getIdentifierInfo()));
  Emit32(Out, T.getLocation().getRawEncoding());
  Emit32(Out, T.getLength());
}


static void EmitIdentifier(llvm::raw_ostream& Out, const IdentifierInfo& II) {
  uint32_t X = (uint32_t) II.getTokenID() << 19;
  X |= (uint32_t) II.getBuiltinID() << 9;
  X |= (uint32_t) II.getObjCKeywordID() << 4;
  if (II.hasMacroDefinition()) X |= 0x8;
  if (II.isExtensionToken()) X |= 0x4;
  if (II.isPoisoned()) X |= 0x2;
  if (II.isCPlusPlusOperatorKeyword()) X |= 0x1;

  Emit32(Out, X);
}

static uint64_t EmitIdentifierTable(llvm::raw_fd_ostream& Out,
                                    const IdentifierTable& T, const IDMap& IM) {

  // Record the location within the PTH file.
  uint64_t Off = Out.tell();
  
  for (IdentifierTable::const_iterator I=T.begin(), E=T.end(); I!=E; ++I) {
    const IdentifierInfo& II = I->getValue();

    // Write out the persistent identifier.
    IDMap::const_iterator IItr = IM.find(&II);
    if (IItr == IM.end()) continue;
    Emit32(Out, IItr->second);
    EmitIdentifier(Out, II);
    
    // Write out the keyword.    
    unsigned len = I->getKeyLength();
    Emit32(Out, len);
    const char* buf = I->getKeyData();    
    EmitBuf(Out, buf, buf+len);  
  }
  
  return Off;
}

static uint64_t EmitFileTable(llvm::raw_fd_ostream& Out, SourceManager& SM,
                              PCHMap& PM) {
  
  uint64_t off = Out.tell();
  assert (0 && "Write out the table.");
  return off;
}

static uint64_t LexTokens(llvm::raw_fd_ostream& Out, Lexer& L, Preprocessor& PP,
                          uint32_t& idcount, IDMap& IM) {
  
  // Record the location within the token file.
  uint64_t off = Out.tell();

  Token Tok;
  
  do {
    L.LexFromRawLexer(Tok);
    
    if (Tok.is(tok::identifier)) {
      Tok.setIdentifierInfo(PP.LookUpIdentifierInfo(Tok));
    }
    else if (Tok.is(tok::hash) && Tok.isAtStartOfLine()) {
      // Special processing for #include.  Store the '#' token and lex
      // the next token.
      EmitToken(Out, Tok, idcount, IM);
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
        EmitToken(Out, Tok, idcount, IM);
        
        // Lex the next token as an include string.
        L.setParsingPreprocessorDirective(true);
        L.LexIncludeFilename(Tok); 
        L.setParsingPreprocessorDirective(false);
        
        if (Tok.is(tok::identifier))
          Tok.setIdentifierInfo(PP.LookUpIdentifierInfo(Tok));
      }
    }    
  }
  while (EmitToken(Out, Tok, idcount, IM), Tok.isNot(tok::eof));
  
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
    os << "PCH error: " << ErrMsg << "\n";
    return;
  }
  
  for (SourceManager::fileid_iterator I=SM.fileid_begin(), E=SM.fileid_end();
       I!=E; ++I) {
    
    const SrcMgr::ContentCache* C = I.getFileIDInfo().getContentCache();
    if (!C) continue;

    const FileEntry* FE = C->Entry;    // Does this entry correspond to a file?    
    if (!FE) continue;

    PCHMap::iterator PI = PM.find(FE); // Have we already processed this file?
    if (PI != PM.end()) continue;
    
    const llvm::MemoryBuffer* B = C->Buffer;    
    if (!B) continue;
    
    Lexer L(SourceLocation::getFileLoc(I.getFileID(), 0), LOpts,
            B->getBufferStart(), B->getBufferEnd(), B);
    
    PM[FE] = LexTokens(Out, L, PP, idcount, IM);
  }

  // Write out the identifier table.
  uint64_t IdTableOff = EmitIdentifierTable(Out, PP.getIdentifierTable(), IM);
  
  // Write out the file table.
  uint64_t FileTableOff = EmitFileTable(Out, SM, PM);  
  
  // Finally, write out the offset table at the end.
  EmitOffset(Out, IdTableOff);
  EmitOffset(Out, FileTableOff);
}
