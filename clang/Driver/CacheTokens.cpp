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
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/System/Path.h"
#include "llvm/Support/Compiler.h"

using namespace clang;

typedef uint32_t Offset;

typedef std::vector<std::pair<Offset, llvm::StringMapEntry<Offset>*> >
  SpellMapTy;

namespace {
class VISIBILITY_HIDDEN PCHEntry {
  Offset TokenData, PPCondData;  
  union { Offset SpellingOff; SpellMapTy* Spellings; };

public:  
  PCHEntry() {}

  PCHEntry(Offset td, Offset ppcd, SpellMapTy* sp)
    : TokenData(td), PPCondData(ppcd), Spellings(sp) {}
  
  Offset getTokenOffset() const { return TokenData; }
  Offset getPPCondTableOffset() const { return PPCondData; }
  SpellMapTy& getSpellings() const { return *Spellings; }
  
  void setSpellingTableOffset(Offset off) { SpellingOff = off; }
  Offset getSpellingTableOffset() const { return SpellingOff; }
  
};
} // end anonymous namespace

typedef llvm::DenseMap<const FileEntry*, PCHEntry> PCHMap;
typedef llvm::DenseMap<const IdentifierInfo*,uint32_t> IDMap;
typedef llvm::StringMap<Offset, llvm::BumpPtrAllocator> CachedStrsTy;

namespace {
class VISIBILITY_HIDDEN PTHWriter {
  IDMap IM;
  llvm::raw_fd_ostream& Out;
  Preprocessor& PP;
  uint32_t idcount;
  PCHMap PM;
  CachedStrsTy CachedStrs;
  
  SpellMapTy* CurSpellMap;

  //// Get the persistent id for the given IdentifierInfo*.
  uint32_t ResolveID(const IdentifierInfo* II);
  
  /// Emit a token to the PTH file.
  void EmitToken(const Token& T);

  void Emit8(uint32_t V) {
    Out << (unsigned char)(V);
  }
    
  void Emit16(uint32_t V) {
    Out << (unsigned char)(V);
    Out << (unsigned char)(V >>  8);
    assert((V >> 16) == 0);
  }
  
  void Emit24(uint32_t V) {
    Out << (unsigned char)(V);
    Out << (unsigned char)(V >>  8);
    Out << (unsigned char)(V >> 16);
    assert((V >> 24) == 0);
  }

  void Emit32(uint32_t V) {
    Out << (unsigned char)(V);
    Out << (unsigned char)(V >>  8);
    Out << (unsigned char)(V >> 16);
    Out << (unsigned char)(V >> 24);
  }
  
  void EmitBuf(const char* I, const char* E) {
    for ( ; I != E ; ++I) Out << *I;
  }
  
  std::pair<Offset,Offset> EmitIdentifierTable();
  Offset EmitFileTable();
  PCHEntry LexTokens(Lexer& L);
  void EmitCachedSpellings();
  
public:
  PTHWriter(llvm::raw_fd_ostream& out, Preprocessor& pp) 
    : Out(out), PP(pp), idcount(0) {}
    
  void GeneratePTH();
};
} // end anonymous namespace
  
uint32_t PTHWriter::ResolveID(const IdentifierInfo* II) {  
  // Null IdentifierInfo's map to the persistent ID 0.
  if (!II)
    return 0;
  
  IDMap::iterator I = IM.find(II);

  if (I == IM.end()) {
    IM[II] = ++idcount; // Pre-increment since '0' is reserved for NULL.
    return idcount;
  }
  
  return I->second; // We've already added 1.
}

void PTHWriter::EmitToken(const Token& T) {
  uint32_t fpos = PP.getSourceManager().getFullFilePos(T.getLocation());
  Emit8(T.getKind());
  Emit8(T.getFlags());
  Emit24(ResolveID(T.getIdentifierInfo()));
  Emit32(fpos);
  Emit16(T.getLength());

  // For specific tokens we cache their spelling.
  if (T.getIdentifierInfo())
    return;

  switch (T.getKind()) {
    default:
      break;
    case tok::string_literal:     
    case tok::wide_string_literal:
    case tok::angle_string_literal:
    case tok::numeric_constant:
    case tok::char_constant: {
      // FIXME: This uses the slow getSpelling().  Perhaps we do better
      // in the future?  This only slows down PTH generation.
      const std::string& spelling = PP.getSpelling(T);
      const char* s = spelling.c_str();
      
      // Get the string entry.
      llvm::StringMapEntry<Offset> *E =
        &CachedStrs.GetOrCreateValue(s, s+spelling.size());

      // Store the address of the string entry in our spelling map.
      (*CurSpellMap).push_back(std::make_pair(fpos, E));

      break;
    }
  }
}

namespace {
struct VISIBILITY_HIDDEN IDData {
  const IdentifierInfo* II;
  uint32_t FileOffset;
  const IdentifierTable::const_iterator::value_type* Str;
};
}

std::pair<Offset,Offset> PTHWriter::EmitIdentifierTable() {
  
  const IdentifierTable& T = PP.getIdentifierTable();

  // Build an inverse map from persistent IDs -> IdentifierInfo*.
  typedef std::vector<IDData> InverseIDMap;
  InverseIDMap IIDMap;
  IIDMap.resize(idcount);
  
  // Generate mapping from persistent IDs -> IdentifierInfo*.
  for (IDMap::iterator I=IM.begin(), E=IM.end(); I!=E; ++I) {
    // Decrement by 1 because we are using a vector for the lookup and
    // 0 is reserved for NULL.
    assert(I->second > 0);
    assert(I->second-1 < IIDMap.size());
    IIDMap[I->second-1].II = I->first;
  }

  // Get the string data associated with the IdentifierInfo.
  for (IdentifierTable::const_iterator I=T.begin(), E=T.end(); I!=E; ++I) {
    IDMap::iterator IDI = IM.find(&(I->getValue()));
    if (IDI == IM.end()) continue;
    IIDMap[IDI->second-1].Str = &(*I);
  }
  
  Offset DataOff = Out.tell();
    
  for (InverseIDMap::iterator I=IIDMap.begin(), E=IIDMap.end(); I!=E; ++I) {
    // Record the location for this data.
    I->FileOffset = Out.tell();
    // Write out the keyword.
    unsigned len = I->Str->getKeyLength();  
    Emit32(len);
    const char* buf = I->Str->getKeyData();    
    EmitBuf(buf, buf+len);  
  }
  
  // Now emit the table mapping from persistent IDs to PTH file offsets.  
  Offset IDOff = Out.tell();
  
  // Emit the number of identifiers.
  Emit32(idcount);

  for (InverseIDMap::iterator I=IIDMap.begin(), E=IIDMap.end(); I!=E; ++I)
    Emit32(I->FileOffset);

  return std::make_pair(DataOff, IDOff);
}

Offset PTHWriter::EmitFileTable() {
  // Determine the offset where this table appears in the PTH file.
  Offset off = (Offset) Out.tell();

  // Output the size of the table.
  Emit32(PM.size());

  for (PCHMap::iterator I=PM.begin(), E=PM.end(); I!=E; ++I) {
    const FileEntry* FE = I->first;
    const char* Name = FE->getName();
    unsigned size = strlen(Name);
    Emit32(size);
    EmitBuf(Name, Name+size);
    Emit32(I->second.getTokenOffset());
    Emit32(I->second.getPPCondTableOffset());
    Emit32(I->second.getSpellingTableOffset());
  }

  return off;
}

PCHEntry PTHWriter::LexTokens(Lexer& L) {

  // Record the location within the token file.
  Offset off = (Offset) Out.tell();
  
  // Keep track of matching '#if' ... '#endif'.
  typedef std::vector<std::pair<Offset, unsigned> > PPCondTable;
  PPCondTable PPCond;
  std::vector<unsigned> PPStartCond;
  bool ParsingPreprocessorDirective = false;

  // Allocate a spelling map for this source file.
  llvm::OwningPtr<SpellMapTy> Spellings(new SpellMapTy());
  CurSpellMap = Spellings.get();

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
      EmitToken(Tmp);
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
      EmitToken(Tok);

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
        EmitToken(Tok);
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
  while (EmitToken(Tok), Tok.isNot(tok::eof));

  assert(PPStartCond.empty() && "Error: imblanced preprocessor conditionals.");

  // Next write out PPCond.
  Offset PPCondOff = (Offset) Out.tell();

  // Write out the size of PPCond so that clients can identifer empty tables.
  Emit32(PPCond.size());

  for (unsigned i = 0, e = PPCond.size(); i!=e; ++i) {
    Emit32(PPCond[i].first - off);
    uint32_t x = PPCond[i].second;
    assert(x != 0 && "PPCond entry not backpatched.");
    // Emit zero for #endifs.  This allows us to do checking when
    // we read the PTH file back in.
    Emit32(x == i ? 0 : x);
  }

  return PCHEntry(off, PPCondOff, Spellings.take());
}

void PTHWriter::EmitCachedSpellings() {
  // Write each cached string to the PTH file and update the
  // the string map entry to contain the relevant offset.
  //
  // FIXME: We can write the strings out in order of their frequency.  This
  //  may result in better locality.
  //
  for (CachedStrsTy::iterator I = CachedStrs.begin(), E = CachedStrs.end();
       I!=E; ++I) {
    
    Offset off = Out.tell();

    // Write out the length of the string before the string itself.
    unsigned len = I->getKeyLength();
    Emit16(len);

    // Write out the string data.
    const char* data = I->getKeyData();
    EmitBuf(data, data+len);
    
    // Now patch the offset of the string in the PTH file into the string map.
    I->setValue(off);
  }
  
  // Now emit the spelling tables.
  for (PCHMap::iterator I=PM.begin(), E=PM.end(); I!=E; ++I) {
    SpellMapTy& spellings = I->second.getSpellings();
    I->second.setSpellingTableOffset(Out.tell());
    
    // Write out the number of spellings.
    unsigned n = spellings.size();
    Emit32(n);
    
    for (unsigned i = 0; i < n; ++i) {
      ++count;
      // Write out the offset of the token within the source file.
      Emit32(spellings[i].first);
      
      // Write out the offset of the spelling data within the PTH file.
      Emit32(spellings[i].second->getValue());
    }
    
    // Delete the spelling map for this source file.
    delete &spellings;
  }
}

void PTHWriter::GeneratePTH() {
  // Iterate over all the files in SourceManager.  Create a lexer
  // for each file and cache the tokens.
  SourceManager& SM = PP.getSourceManager();
  const LangOptions& LOpts = PP.getLangOptions();
  
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
    
    const llvm::MemoryBuffer* B = C->getBuffer();
    if (!B) continue;
    
    Lexer L(SourceLocation::getFileLoc(I.getFileID(), 0), LOpts,
            B->getBufferStart(), B->getBufferEnd(), B);

    PM[FE] = LexTokens(L);
  }

  // Write out the identifier table.
  std::pair<Offset,Offset> IdTableOff = EmitIdentifierTable();
  
  // Write out the cached strings table.
  EmitCachedSpellings();
  
  // Write out the file table.
  Offset FileTableOff = EmitFileTable();  
  
  // Finally, write out the offset table at the end.
  Emit32(IdTableOff.first);
  Emit32(IdTableOff.second);
  Emit32(FileTableOff);
}

void clang::CacheTokens(Preprocessor& PP, const std::string& OutFile) {
  // Lex through the entire file.  This will populate SourceManager with
  // all of the header information.
  Token Tok;
  PP.EnterMainSourceFile();
  do { PP.Lex(Tok); } while (Tok.isNot(tok::eof));
  
  // Open up the PTH file.
  std::string ErrMsg;
  llvm::raw_fd_ostream Out(OutFile.c_str(), true, ErrMsg);
  
  if (!ErrMsg.empty()) {
    llvm::errs() << "PTH error: " << ErrMsg << "\n";
    return;
  }
  
  // Create the PTHWriter and generate the PTH file.
  PTHWriter PW(Out, PP);
  PW.GeneratePTH();
}
