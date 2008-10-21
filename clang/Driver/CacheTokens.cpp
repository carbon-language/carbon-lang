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

static void EmitIdentifierTable(llvm::raw_ostream& Out,
                                const IdentifierTable& T, const IDMap& IM) {
  
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
  uint64_t tokIdx = 0;
  uint32_t idcount = 0;
  
  std::string ErrMsg;
  llvm::raw_fd_ostream Out(OutFile.c_str(), ErrMsg);
  
  if (!ErrMsg.empty()) {
    os << "PCH error: " << ErrMsg << "\n";
    return;
  }
  
  for (SourceManager::fileid_iterator I=SM.fileid_begin(), E=SM.fileid_end();
       I!=E; ++I) {
    
    const SrcMgr::ContentCache* C = I.getFileIDInfo().getContentCache();

    if (!C)
      continue;
    
    const FileEntry* FE = C->Entry;
    
    if (!FE)
      continue;
    
    PCHMap::iterator PI = PM.find(FE);
    if (PI != PM.end()) continue;    
    PM[FE] = tokIdx;
    
    // os << "Processing: " << FE->getName() << " : " << tokIdx << "\n";
    
    const llvm::MemoryBuffer* B = C->Buffer;

    if (!B)
      continue;

    // Create a raw lexer.
    Lexer L(SourceLocation::getFileLoc(I.getFileID(), 0), LOpts,
            B->getBufferStart(), B->getBufferEnd(), B);

    // Ignore whitespace.
    L.SetKeepWhitespaceMode(false);
    L.SetCommentRetentionState(false);
    
    // Lex the file, populating our data structures.
    Token Tok;
    L.LexFromRawLexer(Tok);

    while (Tok.isNot(tok::eof)) {
      ++tokIdx;

      if (Tok.is(tok::identifier))
        Tok.setIdentifierInfo(PP.LookUpIdentifierInfo(Tok));

      // Write the token to disk.
      EmitToken(Out, Tok, idcount, IM);

      // Lex the next token.
      L.LexFromRawLexer(Tok);
    }
  }
  
  // Now, write out the identifier table.
  EmitIdentifierTable(Out, PP.getIdentifierTable(), IM);
}
