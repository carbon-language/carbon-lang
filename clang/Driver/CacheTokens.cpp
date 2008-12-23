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

typedef llvm::DenseMap<const FileEntry*,std::pair<Offset,Offset> > PCHMap;
typedef llvm::DenseMap<const IdentifierInfo*,uint32_t> IDMap;

static void Emit8(llvm::raw_ostream& Out, uint32_t V) {
  Out << (unsigned char)(V);
}

static void Emit32(llvm::raw_ostream& Out, uint32_t V) {
  Out << (unsigned char)(V);
  Out << (unsigned char)(V >>  8);
  Out << (unsigned char)(V >> 16);
  Out << (unsigned char)(V >> 24);
}

static void Emit16(llvm::raw_ostream& Out, uint32_t V) {
  Out << (unsigned char)(V);
  Out << (unsigned char)(V >>  8);
  assert((V >> 16) == 0);
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
  Emit16(Out, T.getLength());
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
    Emit32(Out, I->second.first);
    Emit32(Out, I->second.second);
  }

  return off;
}

static std::pair<Offset,Offset>
LexTokens(llvm::raw_fd_ostream& Out, Lexer& L, Preprocessor& PP,
          uint32_t& idcount, IDMap& IM) {
  
  // Record the location within the token file.
  Offset off = (Offset) Out.tell();
  SourceManager& SMgr = PP.getSourceManager();
  
  // Keep track of matching '#if' ... '#endif'.
  typedef std::vector<std::pair<Offset, unsigned> > PPCondTable;
  PPCondTable PPCond;
  std::vector<unsigned> PPStartCond;
  bool ParsingPreprocessorDirective = false;

  Token Tok;
  
  do {
    L.LexFromRawLexer(Tok);
    
    if ((Tok.isAtStartOfLine() || Tok.is(tok::eof)) &&
        ParsingPreprocessorDirective) {
      // Insert an eom token into the token cache.  It has the same
      // position as the next token that is not on the same line as the
      // preprocessor directive.  Observe that we continue processing
      // 'Tok' when we exit this branch.
      Token Tmp = Tok;
      Tmp.setKind(tok::eom);
      Tmp.clearFlag(Token::StartOfLine);
      Tmp.setIdentifierInfo(0);
      EmitToken(Out, Tmp, SMgr, idcount, IM);
      ParsingPreprocessorDirective = false;
    }
    
    if (Tok.is(tok::identifier)) {
      Tok.setIdentifierInfo(PP.LookUpIdentifierInfo(Tok));
      continue;
    }

    if (Tok.is(tok::hash) && Tok.isAtStartOfLine()) {
      // Special processing for #include.  Store the '#' token and lex
      // the next token.
      assert(!ParsingPreprocessorDirective);
      Offset HashOff = (Offset) Out.tell();
      EmitToken(Out, Tok, SMgr, idcount, IM);

      // Get the next token.
      L.LexFromRawLexer(Tok);
            
      assert(!Tok.isAtStartOfLine());
      
      // Did we see 'include'/'import'/'include_next'?
      if (!Tok.is(tok::identifier))
        continue;
      
      IdentifierInfo* II = PP.LookUpIdentifierInfo(Tok);
      Tok.setIdentifierInfo(II);
      tok::PPKeywordKind K = II->getPPKeywordID();
      
      assert(K != tok::pp_not_keyword);
      ParsingPreprocessorDirective = true;
      
      switch (K) {
      default:
        break;
      case tok::pp_include:
      case tok::pp_import:
      case tok::pp_include_next: {        
        // Save the 'include' token.
        EmitToken(Out, Tok, SMgr, idcount, IM);
        // Lex the next token as an include string.
        L.setParsingPreprocessorDirective(true);
        L.LexIncludeFilename(Tok); 
        L.setParsingPreprocessorDirective(false);
        assert(!Tok.isAtStartOfLine());
        if (Tok.is(tok::identifier))
          Tok.setIdentifierInfo(PP.LookUpIdentifierInfo(Tok));
        
        break;
      }
      case tok::pp_if:
      case tok::pp_ifdef:
      case tok::pp_ifndef: {
        // Ad an entry for '#if' and friends.  We initially set the target index
        // to 0.  This will get backpatched when we hit #endif.
        PPStartCond.push_back(PPCond.size());
        PPCond.push_back(std::make_pair(HashOff, 0U));
        break;
      }
      case tok::pp_endif: {
        // Add an entry for '#endif'.  We set the target table index to itself.
        // This will later be set to zero when emitting to the PTH file.  We
        // use 0 for uninitialized indices because that is easier to debug.
        unsigned index = PPCond.size();
        // Backpatch the opening '#if' entry.
        assert(!PPStartCond.empty());
        assert(PPCond.size() > PPStartCond.back());
        assert(PPCond[PPStartCond.back()].second == 0);
        PPCond[PPStartCond.back()].second = index;
        PPStartCond.pop_back();        
        // Add the new entry to PPCond.      
        PPCond.push_back(std::make_pair(HashOff, index));
        break;
      }
      case tok::pp_elif:
      case tok::pp_else: {
        // Add an entry for #elif or #else.
        // This serves as both a closing and opening of a conditional block.
        // This means that its entry will get backpatched later.
        unsigned index = PPCond.size();
        // Backpatch the previous '#if' entry.
        assert(!PPStartCond.empty());
        assert(PPCond.size() > PPStartCond.back());
        assert(PPCond[PPStartCond.back()].second == 0);
        PPCond[PPStartCond.back()].second = index;
        PPStartCond.pop_back();
        // Now add '#elif' as a new block opening.
        PPCond.push_back(std::make_pair(HashOff, 0U));
        PPStartCond.push_back(index);
        break;
      }
      }
    }    
  }
  while (EmitToken(Out, Tok, SMgr, idcount, IM), Tok.isNot(tok::eof));
  
  assert(PPStartCond.empty() && "Error: imblanced preprocessor conditionals.");
  
  // Next write out PPCond.
  Offset PPCondOff = (Offset) Out.tell();

  // Write out the size of PPCond so that clients can identifer empty tables.
  Emit32(Out, PPCond.size());

  for (unsigned i = 0, e = PPCond.size(); i!=e; ++i) {
    Emit32(Out, PPCond[i].first - off);
    uint32_t x = PPCond[i].second;
    assert(x != 0 && "PPCond entry not backpatched.");
    // Emit zero for #endifs.  This allows us to do checking when
    // we read the PTH file back in.
    Emit32(Out, x == i ? 0 : x);
  }

  return std::make_pair(off,PPCondOff);
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
