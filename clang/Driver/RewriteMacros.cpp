//===--- RewriteMacros.cpp - Rewrite macros into their expansions ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This code rewrites macro invocations into their expansions.  This gives you
// a macro expanded file that retains comments and #includes.
//
//===----------------------------------------------------------------------===//

#include "clang.h"
#include "clang/Rewrite/Rewriter.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/Support/Streams.h"
#include "llvm/System/Path.h"
#include <fstream>
using namespace clang;

/// isSameToken - Return true if the two specified tokens start have the same
/// content.
static bool isSameToken(Token &RawTok, Token &PPTok) {
  // If two tokens have the same kind and the same identifier info, they are
  // obviously the same.
  if (PPTok.getKind() == RawTok.getKind() &&
      PPTok.getIdentifierInfo() == RawTok.getIdentifierInfo())
    return true;
  
  // Otherwise, if they are different but have the same identifier info, they
  // are also considered to be the same.  This allows keywords and raw lexed
  // identifiers with the same name to be treated the same.
  if (PPTok.getIdentifierInfo() &&
      PPTok.getIdentifierInfo() == RawTok.getIdentifierInfo())
    return true;
  
  return false;
}

static void GetNextRawTok(Lexer &RawLex, Token &RawTok, Preprocessor &PP) {
  RawLex.LexRawToken(RawTok);
  
  // If we have an identifier with no identifier info for our raw token, look
  // up the indentifier info.
  if (RawTok.is(tok::identifier) && !RawTok.getIdentifierInfo())
    RawTok.setIdentifierInfo(PP.LookUpIdentifierInfo(RawTok));
}

/// RewriteMacrosInInput - Implement -rewrite-macros mode.
void clang::RewriteMacrosInInput(Preprocessor &PP,const std::string &InFileName,
                                 const std::string &OutFileName) {
  SourceManager &SM = PP.getSourceManager();
  
  Rewriter Rewrite;
  Rewrite.setSourceMgr(SM);
  RewriteBuffer &RB = Rewrite.getEditBuffer(SM.getMainFileID());

  const SourceManager &SourceMgr = PP.getSourceManager();
  std::pair<const char*, const char*> File =
    SourceMgr.getBufferData(SM.getMainFileID());
  
  // Create a lexer to lex all the tokens of the main file in raw mode.  Even
  // though it is in raw mode, it will not return comments.
  Lexer RawLex(SourceLocation::getFileLoc(SM.getMainFileID(), 0),
               PP.getLangOptions(), File.first, File.second);
  Token RawTok;
  GetNextRawTok(RawLex, RawTok, PP);
  
  // Get the first preprocessing token.
  PP.EnterMainSourceFile();
  Token PPTok;
  PP.Lex(PPTok);
  
  // Preprocess the input file in parallel with raw lexing the main file. Ignore
  // all tokens that are preprocessed from a file other than the main file (e.g.
  // a header).  If we see tokens that are in the preprocessed file bug not the
  // lexed file, we have a macro expansion.  If we see tokens in the lexed file
  // that aren't in the preprocessed view, we have macros that expand to no
  // tokens, or macro arguments etc.
  while (RawTok.isNot(tok::eof) || PPTok.isNot(tok::eof)) {
    SourceLocation PPLoc = SM.getLogicalLoc(PPTok.getLocation());

    // If PPTok is from a different source file, ignore it.
    if (!SM.isFromMainFile(PPLoc)) {
      PP.Lex(PPTok);
      continue;
    }
    
    // If the raw file hits a preprocessor directive, they will be extra tokens
    // in the input file, but we don't want to treat them as such... just ignore
    // them.
    if (RawTok.is(tok::hash) && RawTok.isAtStartOfLine()) {
      GetNextRawTok(RawLex, RawTok, PP);
      while (!RawTok.isAtStartOfLine() && RawTok.isNot(tok::eof))
        GetNextRawTok(RawLex, RawTok, PP);
      continue;
    }
    
    // Okay, both tokens are from the same file.  Get their offsets from the
    // start of the file.
    unsigned PPOffs = SM.getFullFilePos(PPLoc);
    unsigned RawOffs = SM.getFullFilePos(RawTok.getLocation());

    // If the offsets are the same and the token kind is the same, ignore them.
    if (PPOffs == RawOffs && isSameToken(RawTok, PPTok)) {
      GetNextRawTok(RawLex, RawTok, PP);
      PP.Lex(PPTok);
      continue;
    }

    // If the PP token is farther along than the raw token, something was
    // deleted.  Comment out the raw token.
    if (RawOffs <= PPOffs) {
      // Comment out a whole run of tokens instead of bracketing each one with
      // comments.  Add a leading space if RawTok didn't have one.
      bool HasSpace = RawTok.hasLeadingSpace();
      RB.InsertTextAfter(RawOffs, " /*"+HasSpace, 2+!HasSpace);
      unsigned EndPos;

      // Switch on comment lexing.  If we get a comment, we don't want to
      // include it as part of our run of tokens, because we don't want to
      // nest /* */ comments.
      RawLex.SetCommentRetentionState(true);
      
      do {
        EndPos = RawOffs+RawTok.getLength();

        GetNextRawTok(RawLex, RawTok, PP);
        RawOffs = SM.getFullFilePos(RawTok.getLocation());
        
        if (RawTok.is(tok::comment)) {
          RawLex.SetCommentRetentionState(false);
          // Skip past the comment.
          GetNextRawTok(RawLex, RawTok, PP);
          break;
        }
        
      } while (RawOffs <= PPOffs && !RawTok.isAtStartOfLine() &&
               (PPOffs != RawOffs || !isSameToken(RawTok, PPTok)));
      
      RawLex.SetCommentRetentionState(false);

      RB.InsertTextBefore(EndPos, "*/", 2);
      continue;
    }
    
    // Otherwise, there was a replacement an expansion.  Insert the new token
    // in the output buffer.  Insert the whole run of new tokens at once to get
    // them in the right order.
    unsigned InsertPos = PPOffs;
    std::string Expansion;
    while (PPOffs < RawOffs) {
      Expansion += ' ' + PP.getSpelling(PPTok);
      PP.Lex(PPTok);
      PPLoc = SM.getLogicalLoc(PPTok.getLocation());
      PPOffs = SM.getFullFilePos(PPLoc);
    }
    Expansion += ' ';
    RB.InsertTextBefore(InsertPos, &Expansion[0], Expansion.size());
  }
  
  // Create the output file.
  std::ostream *OutFile;
  if (OutFileName == "-") {
    OutFile = llvm::cout.stream();
  } else if (!OutFileName.empty()) {
    OutFile = new std::ofstream(OutFileName.c_str(), 
                                std::ios_base::binary|std::ios_base::out);
  } else if (InFileName == "-") {
    OutFile = llvm::cout.stream();
  } else {
    llvm::sys::Path Path(InFileName);
    Path.eraseSuffix();
    Path.appendSuffix("cpp");
    OutFile = new std::ofstream(Path.toString().c_str(), 
                                std::ios_base::binary|std::ios_base::out);
  }

  // Get the buffer corresponding to MainFileID.  If we haven't changed it, then
  // we are done.
  if (const RewriteBuffer *RewriteBuf = 
      Rewrite.getRewriteBufferFor(SM.getMainFileID())) {
    //printf("Changed:\n");
    *OutFile << std::string(RewriteBuf->begin(), RewriteBuf->end());
  } else {
    fprintf(stderr, "No changes\n");
  }
}
