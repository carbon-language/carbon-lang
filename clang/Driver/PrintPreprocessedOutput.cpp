//===--- PrintPreprocessedOutput.cpp - Implement the -E mode --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This code simply runs the preprocessor on the input file and prints out the
// result.  This is the traditional behavior of the -E option.
//
//===----------------------------------------------------------------------===//

#include "clang.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/Pragma.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Config/config.h"
#include <cstdio>
using namespace llvm;
using namespace clang;

//===----------------------------------------------------------------------===//
// Simple buffered I/O
//===----------------------------------------------------------------------===//
//
// Empirically, iostream is over 30% slower than stdio for this workload, and
// stdio itself isn't very well suited.  The problem with stdio is use of
// putchar_unlocked.  We have many newline characters that need to be emitted,
// but stdio needs to do extra checks to handle line buffering mode.  These
// extra checks make putchar_unlocked fall off its inlined code path, hitting
// slow system code.  In practice, using 'write' directly makes 'clang -E -P'
// about 10% faster than using the stdio path on darwin.

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#else
#define USE_STDIO 1
#endif

static char *OutBufStart = 0, *OutBufEnd, *OutBufCur;

/// InitOutputBuffer - Initialize our output buffer.
///
static void InitOutputBuffer() {
#ifndef USE_STDIO
  OutBufStart = new char[64*1024];
  OutBufEnd = OutBufStart+64*1024;
  OutBufCur = OutBufStart;
#endif
}

/// FlushBuffer - Write the accumulated bytes to the output stream.
///
static void FlushBuffer() {
#ifndef USE_STDIO
  write(STDOUT_FILENO, OutBufStart, OutBufCur-OutBufStart);
  OutBufCur = OutBufStart;
#endif
}

/// CleanupOutputBuffer - Finish up output.
///
static void CleanupOutputBuffer() {
#ifndef USE_STDIO
  FlushBuffer();
  delete [] OutBufStart;
#endif
}

static void OutputChar(char c) {
#ifdef USE_STDIO
  putchar_unlocked(c);
#else
  if (OutBufCur >= OutBufEnd)
    FlushBuffer();
  *OutBufCur++ = c;
#endif
}

static void OutputString(const char *Ptr, unsigned Size) {
#ifdef USE_STDIO
  fwrite(Ptr, Size, 1, stdout);
#else
  if (OutBufCur+Size >= OutBufEnd)
    FlushBuffer();
  memcpy(OutBufCur, Ptr, Size);
  OutBufCur += Size;
#endif
}


//===----------------------------------------------------------------------===//
// Preprocessed token printer
//===----------------------------------------------------------------------===//

static cl::opt<bool>
DisableLineMarkers("P", cl::desc("Disable linemarker output in -E mode"));
static cl::opt<bool>
EnableCommentOutput("C", cl::desc("Enable comment output in -E mode"));
static cl::opt<bool>
EnableMacroCommentOutput("CC", cl::desc("Enable comment output in -E mode, "
                                        "even from macro expansions"));

static unsigned EModeCurLine;
static std::string EModeCurFilename;
static Preprocessor *EModePP;
static bool EmodeEmittedTokensOnThisLine;
static DirectoryLookup::DirType EmodeFileType =DirectoryLookup::NormalHeaderDir;

/// MoveToLine - Move the output to the source line specified by the location
/// object.  We can do this by emitting some number of \n's, or be emitting a
/// #line directive.
static void MoveToLine(SourceLocation Loc) {
  if (DisableLineMarkers) {
    if (EmodeEmittedTokensOnThisLine) {
      OutputChar('\n');
      EmodeEmittedTokensOnThisLine = false;
    }
    return;
  }

  unsigned LineNo = EModePP->getSourceManager().getLineNumber(Loc);
  
  // If this line is "close enough" to the original line, just print newlines,
  // otherwise print a #line directive.
  if (LineNo-EModeCurLine < 8) {
    unsigned CurLine = EModeCurLine;
    for (; CurLine != LineNo; ++CurLine)
      OutputChar('\n');
    EModeCurLine = CurLine;
  } else {
    if (EmodeEmittedTokensOnThisLine) {
      OutputChar('\n');
      EmodeEmittedTokensOnThisLine = false;
    }
    
    EModeCurLine = LineNo;
    
    OutputChar('#');
    OutputChar(' ');
    std::string Num = utostr_32(LineNo);
    OutputString(&Num[0], Num.size());
    OutputChar(' ');
    OutputString(&EModeCurFilename[0], EModeCurFilename.size());
    
    if (EmodeFileType == DirectoryLookup::SystemHeaderDir)
      OutputString(" 3", 2);
    else if (EmodeFileType == DirectoryLookup::ExternCSystemHeaderDir)
      OutputString(" 3 4", 4);
    OutputChar('\n');
  } 
}

/// HandleFileChange - Whenever the preprocessor enters or exits a #include file
/// it invokes this handler.  Update our conception of the current 
static void HandleFileChange(SourceLocation Loc,
                             Preprocessor::FileChangeReason Reason,
                             DirectoryLookup::DirType FileType) {
  if (DisableLineMarkers) return;

  // Unless we are exiting a #include, make sure to skip ahead to the line the
  // #include directive was at.
  SourceManager &SourceMgr = EModePP->getSourceManager();
  if (Reason == Preprocessor::EnterFile) {
    MoveToLine(SourceMgr.getIncludeLoc(Loc.getFileID()));
  } else if (Reason == Preprocessor::SystemHeaderPragma) {
    MoveToLine(Loc);
    
    // TODO GCC emits the # directive for this directive on the line AFTER the
    // directive and emits a bunch of spaces that aren't needed.  Emulate this
    // strange behavior.
  }
  
  EModeCurLine = SourceMgr.getLineNumber(Loc);
  EModeCurFilename = '"' + Lexer::Stringify(SourceMgr.getSourceName(Loc)) + '"';
  EmodeFileType = FileType;
  
  if (EmodeEmittedTokensOnThisLine) {
    OutputChar('\n');
    EmodeEmittedTokensOnThisLine = false;
  }
  
  if (DisableLineMarkers) return;
  
  OutputChar('#');
  OutputChar(' ');
  std::string Num = utostr_32(EModeCurLine);
  OutputString(&Num[0], Num.size());
  OutputChar(' ');
  OutputString(&EModeCurFilename[0], EModeCurFilename.size());
  
  switch (Reason) {
  case Preprocessor::EnterFile:
    OutputString(" 1", 2);
    break;
  case Preprocessor::ExitFile:
    OutputString(" 2", 2);
    break;
  case Preprocessor::SystemHeaderPragma: break;
  case Preprocessor::RenameFile: break;
  }
  
  if (FileType == DirectoryLookup::SystemHeaderDir)
    OutputString(" 3", 2);
  else if (FileType == DirectoryLookup::ExternCSystemHeaderDir)
    OutputString(" 3 4", 4);
  
  OutputChar('\n');
}

/// HandleIdent - Handle #ident directives when read by the preprocessor.
///
static void HandleIdent(SourceLocation Loc, const std::string &Val) {
  MoveToLine(Loc);
  
  OutputString("#ident ", strlen("#ident "));
  OutputString(&Val[0], Val.size());
  EmodeEmittedTokensOnThisLine = true;
}

/// HandleFirstTokOnLine - When emitting a preprocessed file in -E mode, this
/// is called for the first token on each new line.
static void HandleFirstTokOnLine(LexerToken &Tok, Preprocessor &PP) {
  // Figure out what line we went to and insert the appropriate number of
  // newline characters.
  MoveToLine(Tok.getLocation());
  
  // Print out space characters so that the first token on a line is
  // indented for easy reading.
  unsigned ColNo = 
    PP.getSourceManager().getColumnNumber(Tok.getLocation());
  
  // This hack prevents stuff like:
  // #define HASH #
  // HASH define foo bar
  // From having the # character end up at column 1, which makes it so it
  // is not handled as a #define next time through the preprocessor if in
  // -fpreprocessed mode.
  if (ColNo <= 1 && Tok.getKind() == tok::hash)
    OutputChar(' ');
  
  // Otherwise, indent the appropriate number of spaces.
  for (; ColNo > 1; --ColNo)
    OutputChar(' ');
}

namespace {
struct UnknownPragmaHandler : public PragmaHandler {
  const char *Prefix;
  UnknownPragmaHandler(const char *prefix) : PragmaHandler(0), Prefix(prefix) {}
  virtual void HandlePragma(Preprocessor &PP, LexerToken &PragmaTok) {
    // Figure out what line we went to and insert the appropriate number of
    // newline characters.
    MoveToLine(PragmaTok.getLocation());
    OutputString(Prefix, strlen(Prefix));
    
    // Read and print all of the pragma tokens.
    while (PragmaTok.getKind() != tok::eom) {
      if (PragmaTok.hasLeadingSpace())
        OutputChar(' ');
      std::string TokSpell = PP.getSpelling(PragmaTok);
      OutputString(&TokSpell[0], TokSpell.size());
      PP.LexUnexpandedToken(PragmaTok);
    }
    OutputChar('\n');
  }
};
} // end anonymous namespace

/// AvoidConcat - If printing PrevTok immediately followed by Tok would cause
/// the two individual tokens to be lexed as a single token, return true (which
/// causes a space to be printed between them).  This allows the output of -E
/// mode to be lexed to the same token stream as lexing the input directly
/// would.
///
/// This code must conservatively return true if it doesn't want to be 100%
/// accurate.  This will cause the output to include extra space characters, but
/// the resulting output won't have incorrect concatenations going on.  Examples
/// include "..", which we print with a space between, because we don't want to
/// track enough to tell "x.." from "...".
static bool AvoidConcat(const LexerToken &PrevTok, const LexerToken &Tok,
                        Preprocessor &PP) {
  char Buffer[256];
  
  // If we haven't emitted a token on this line yet, PrevTok isn't useful to
  // look at and no concatenation could happen anyway.
  if (!EmodeEmittedTokensOnThisLine)
    return false;

  // Basic algorithm: we look at the first character of the second token, and
  // determine whether it, if appended to the first token, would form (or would
  // contribute) to a larger token if concatenated.
  char FirstChar;
  if (IdentifierInfo *II = Tok.getIdentifierInfo()) {
    // Avoid spelling identifiers, the most common form of token.
    FirstChar = II->getName()[0];
  } else if (Tok.getLength() < 256) {
    FirstChar = Buffer[0];
  } else {
    FirstChar = PP.getSpelling(Tok)[0];
  }
  
  tok::TokenKind PrevKind = PrevTok.getKind();
  if (PrevTok.getIdentifierInfo())  // Language keyword or named operator.
    PrevKind = tok::identifier;
  
  switch (PrevKind) {
  default: return false;
  case tok::identifier:   // id+id or id+number or id+L"foo".
    return isalnum(FirstChar) || FirstChar == '_';
  case tok::numeric_constant:
    return isalnum(FirstChar) || Tok.getKind() == tok::numeric_constant ||
           FirstChar == '+' || FirstChar == '-' || FirstChar == '.';
  case tok::period:          // ..., .*, .1234
    return FirstChar == '.' || FirstChar == '*' || isdigit(FirstChar);
  case tok::amp:             // &&, &=
    return FirstChar == '&' || FirstChar == '=';
  case tok::plus:            // ++, +=
    return FirstChar == '+' || FirstChar == '=';
  case tok::minus:           // --, ->, -=, ->*
    return FirstChar == '-' || FirstChar == '>' || FirstChar == '=';
  case tok::slash:           // /=, /*, //
    return FirstChar == '=' || FirstChar == '*' || FirstChar == '/';
  case tok::less:            // <<, <<=, <=, <?=, <?, <:, <%
    return FirstChar == '<' || FirstChar == '?' || FirstChar == '=' ||
           FirstChar == ':' || FirstChar == '%';
  case tok::greater:         // >>, >=, >>=, >?=, >?, ->*
    return FirstChar == '>' || FirstChar == '?' || FirstChar == '=' || 
           FirstChar == '*';
  case tok::pipe:            // ||, |=
    return FirstChar == '|' || FirstChar == '=';
  case tok::percent:         // %=, %>, %:
    return FirstChar == '=' || FirstChar == '>' || FirstChar == ':';
  case tok::colon:           // ::, :>
    return FirstChar == ':' || FirstChar == '>';
  case tok::hash:            // ##, #@, %:%:
    return FirstChar == '#' || FirstChar == '@' || FirstChar == '%';
  case tok::arrow:           // ->*
    return FirstChar == '*';
    
  case tok::star:            // *=
  case tok::exclaim:         // !=
  case tok::lessless:        // <<=
  case tok::greaterequal:    // >>=
  case tok::caret:           // ^=
  case tok::equal:           // ==
  case tok::lessquestion:    // <?=
  case tok::greaterquestion: // >?=
  case tok::question:        // <?=, >?=, check for <?= in case <? is disabled.
    // Cases that concatenate only if the next char is =.
    return FirstChar == '=';
  }
}

/// DoPrintPreprocessedInput - This implements -E mode.
///
void clang::DoPrintPreprocessedInput(unsigned MainFileID, Preprocessor &PP,
                                     LangOptions &Options) {
  if (EnableCommentOutput)          // -C specified?
    Options.KeepComments = 1;
  if (EnableMacroCommentOutput)     // -CC specified?
    Options.KeepComments = Options.KeepMacroComments = 1;
  
  InitOutputBuffer();
  
  LexerToken Tok, PrevTok;
  char Buffer[256];
  EModeCurLine = 0;
  EModeCurFilename = "\"<uninit>\"";
  PP.setFileChangeHandler(HandleFileChange);
  PP.setIdentHandler(HandleIdent);
  EModePP = &PP;
  EmodeEmittedTokensOnThisLine = false;
  
  PP.AddPragmaHandler(0, new UnknownPragmaHandler("#pragma"));
  PP.AddPragmaHandler("GCC", new UnknownPragmaHandler("#pragma GCC"));

  // After we have configured the preprocessor, enter the main file.
  
  // Start parsing the specified input file.
  PP.EnterSourceFile(MainFileID, 0, true);
  
  do {
    PrevTok = Tok;
    PP.Lex(Tok);
    
    // If this token is at the start of a line, emit newlines if needed.
    if (Tok.isAtStartOfLine()) {
      HandleFirstTokOnLine(Tok, PP);
    } else if (Tok.hasLeadingSpace() || 
               // Don't print "-" next to "-", it would form "--".
               AvoidConcat(PrevTok, Tok, PP)) {
      OutputChar(' ');
    }
    
    if (Tok.getLength() < 256) {
      const char *TokPtr = Buffer;
      unsigned Len = PP.getSpelling(Tok, TokPtr);
      OutputString(TokPtr, Len);
    } else {
      std::string S = PP.getSpelling(Tok);
      OutputString(&S[0], S.size());
    }
    EmodeEmittedTokensOnThisLine = true;
  } while (Tok.getKind() != tok::eof);
  OutputChar('\n');
  
  CleanupOutputBuffer();
}

