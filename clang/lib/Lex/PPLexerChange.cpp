//===--- PPLexerChange.cpp - Handle changing lexers in the preprocessor ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements pieces of the Preprocessor interface that manage the
// current lexer stack.
//
//===----------------------------------------------------------------------===//

#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/MacroInfo.h"
#include "clang/Lex/LexDiagnostic.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/Support/MemoryBuffer.h"
using namespace clang;

PPCallbacks::~PPCallbacks() {}

//===----------------------------------------------------------------------===//
// Miscellaneous Methods.
//===----------------------------------------------------------------------===//

/// isInPrimaryFile - Return true if we're in the top-level file, not in a
/// #include.  This looks through macro expansions and active _Pragma lexers.
bool Preprocessor::isInPrimaryFile() const {
  if (IsFileLexer())
    return IncludeMacroStack.empty();

  // If there are any stacked lexers, we're in a #include.
  assert(IsFileLexer(IncludeMacroStack[0]) &&
         "Top level include stack isn't our primary lexer?");
  for (unsigned i = 1, e = IncludeMacroStack.size(); i != e; ++i)
    if (IsFileLexer(IncludeMacroStack[i]))
      return false;
  return true;
}

/// getCurrentLexer - Return the current file lexer being lexed from.  Note
/// that this ignores any potentially active macro expansions and _Pragma
/// expansions going on at the time.
PreprocessorLexer *Preprocessor::getCurrentFileLexer() const {
  if (IsFileLexer())
    return CurPPLexer;

  // Look for a stacked lexer.
  for (unsigned i = IncludeMacroStack.size(); i != 0; --i) {
    const IncludeStackInfo& ISI = IncludeMacroStack[i-1];
    if (IsFileLexer(ISI))
      return ISI.ThePPLexer;
  }
  return 0;
}


//===----------------------------------------------------------------------===//
// Methods for Entering and Callbacks for leaving various contexts
//===----------------------------------------------------------------------===//

/// EnterSourceFile - Add a source file to the top of the include stack and
/// start lexing tokens from it instead of the current buffer.
void Preprocessor::EnterSourceFile(FileID FID, const DirectoryLookup *CurDir,
                                   SourceLocation Loc) {
  assert(CurTokenLexer == 0 && "Cannot #include a file inside a macro!");
  ++NumEnteredSourceFiles;

  if (MaxIncludeStackDepth < IncludeMacroStack.size())
    MaxIncludeStackDepth = IncludeMacroStack.size();

  if (PTH) {
    if (PTHLexer *PL = PTH->CreateLexer(FID)) {
      EnterSourceFileWithPTH(PL, CurDir);
      return;
    }
  }
  
  // Get the MemoryBuffer for this FID, if it fails, we fail.
  bool Invalid = false;
  const llvm::MemoryBuffer *InputFile = 
    getSourceManager().getBuffer(FID, Loc, &Invalid);
  if (Invalid) {
    SourceLocation FileStart = SourceMgr.getLocForStartOfFile(FID);
    Diag(Loc, diag::err_pp_error_opening_file)
      << std::string(SourceMgr.getBufferName(FileStart)) << "";
    return;
  }
  
  EnterSourceFileWithLexer(new Lexer(FID, InputFile, *this), CurDir);
  return;
}

/// EnterSourceFileWithLexer - Add a source file to the top of the include stack
///  and start lexing tokens from it instead of the current buffer.
void Preprocessor::EnterSourceFileWithLexer(Lexer *TheLexer,
                                            const DirectoryLookup *CurDir) {

  // Add the current lexer to the include stack.
  if (CurPPLexer || CurTokenLexer)
    PushIncludeMacroStack();

  CurLexer.reset(TheLexer);
  CurPPLexer = TheLexer;
  CurDirLookup = CurDir;

  // Notify the client, if desired, that we are in a new source file.
  if (Callbacks && !CurLexer->Is_PragmaLexer) {
    SrcMgr::CharacteristicKind FileType =
       SourceMgr.getFileCharacteristic(CurLexer->getFileLoc());

    Callbacks->FileChanged(CurLexer->getFileLoc(),
                           PPCallbacks::EnterFile, FileType);
  }
}

/// EnterSourceFileWithPTH - Add a source file to the top of the include stack
/// and start getting tokens from it using the PTH cache.
void Preprocessor::EnterSourceFileWithPTH(PTHLexer *PL,
                                          const DirectoryLookup *CurDir) {

  if (CurPPLexer || CurTokenLexer)
    PushIncludeMacroStack();

  CurDirLookup = CurDir;
  CurPTHLexer.reset(PL);
  CurPPLexer = CurPTHLexer.get();

  // Notify the client, if desired, that we are in a new source file.
  if (Callbacks) {
    FileID FID = CurPPLexer->getFileID();
    SourceLocation EnterLoc = SourceMgr.getLocForStartOfFile(FID);
    SrcMgr::CharacteristicKind FileType =
      SourceMgr.getFileCharacteristic(EnterLoc);
    Callbacks->FileChanged(EnterLoc, PPCallbacks::EnterFile, FileType);
  }
}

/// EnterMacro - Add a Macro to the top of the include stack and start lexing
/// tokens from it instead of the current buffer.
void Preprocessor::EnterMacro(Token &Tok, SourceLocation ILEnd,
                              MacroArgs *Args) {
  PushIncludeMacroStack();
  CurDirLookup = 0;

  if (NumCachedTokenLexers == 0) {
    CurTokenLexer.reset(new TokenLexer(Tok, ILEnd, Args, *this));
  } else {
    CurTokenLexer.reset(TokenLexerCache[--NumCachedTokenLexers]);
    CurTokenLexer->Init(Tok, ILEnd, Args);
  }
}

/// EnterTokenStream - Add a "macro" context to the top of the include stack,
/// which will cause the lexer to start returning the specified tokens.
///
/// If DisableMacroExpansion is true, tokens lexed from the token stream will
/// not be subject to further macro expansion.  Otherwise, these tokens will
/// be re-macro-expanded when/if expansion is enabled.
///
/// If OwnsTokens is false, this method assumes that the specified stream of
/// tokens has a permanent owner somewhere, so they do not need to be copied.
/// If it is true, it assumes the array of tokens is allocated with new[] and
/// must be freed.
///
void Preprocessor::EnterTokenStream(const Token *Toks, unsigned NumToks,
                                    bool DisableMacroExpansion,
                                    bool OwnsTokens) {
  // Save our current state.
  PushIncludeMacroStack();
  CurDirLookup = 0;

  // Create a macro expander to expand from the specified token stream.
  if (NumCachedTokenLexers == 0) {
    CurTokenLexer.reset(new TokenLexer(Toks, NumToks, DisableMacroExpansion,
                                       OwnsTokens, *this));
  } else {
    CurTokenLexer.reset(TokenLexerCache[--NumCachedTokenLexers]);
    CurTokenLexer->Init(Toks, NumToks, DisableMacroExpansion, OwnsTokens);
  }
}

/// HandleEndOfFile - This callback is invoked when the lexer hits the end of
/// the current file.  This either returns the EOF token or pops a level off
/// the include stack and keeps going.
bool Preprocessor::HandleEndOfFile(Token &Result, bool isEndOfMacro) {
  assert(!CurTokenLexer &&
         "Ending a file when currently in a macro!");

  // See if this file had a controlling macro.
  if (CurPPLexer) {  // Not ending a macro, ignore it.
    if (const IdentifierInfo *ControllingMacro =
          CurPPLexer->MIOpt.GetControllingMacroAtEndOfFile()) {
      // Okay, this has a controlling macro, remember in HeaderFileInfo.
      if (const FileEntry *FE =
            SourceMgr.getFileEntryForID(CurPPLexer->getFileID()))
        HeaderInfo.SetFileControllingMacro(FE, ControllingMacro);
    }
  }

  // If this is a #include'd file, pop it off the include stack and continue
  // lexing the #includer file.
  if (!IncludeMacroStack.empty()) {
    // We're done with the #included file.
    RemoveTopOfLexerStack();

    // Notify the client, if desired, that we are in a new source file.
    if (Callbacks && !isEndOfMacro && CurPPLexer) {
      SrcMgr::CharacteristicKind FileType =
        SourceMgr.getFileCharacteristic(CurPPLexer->getSourceLocation());
      Callbacks->FileChanged(CurPPLexer->getSourceLocation(),
                             PPCallbacks::ExitFile, FileType);
    }

    // Client should lex another token.
    return false;
  }

  // If the file ends with a newline, form the EOF token on the newline itself,
  // rather than "on the line following it", which doesn't exist.  This makes
  // diagnostics relating to the end of file include the last file that the user
  // actually typed, which is goodness.
  if (CurLexer) {
    const char *EndPos = CurLexer->BufferEnd;
    if (EndPos != CurLexer->BufferStart &&
        (EndPos[-1] == '\n' || EndPos[-1] == '\r')) {
      --EndPos;

      // Handle \n\r and \r\n:
      if (EndPos != CurLexer->BufferStart &&
          (EndPos[-1] == '\n' || EndPos[-1] == '\r') &&
          EndPos[-1] != EndPos[0])
        --EndPos;
    }

    Result.startToken();
    CurLexer->BufferPtr = EndPos;
    CurLexer->FormTokenWithChars(Result, EndPos, tok::eof);

    // We're done with the #included file.
    CurLexer.reset();
  } else {
    assert(CurPTHLexer && "Got EOF but no current lexer set!");
    CurPTHLexer->getEOF(Result);
    CurPTHLexer.reset();
  }

  CurPPLexer = 0;

  // This is the end of the top-level file. 'WarnUnusedMacroLocs' has collected
  // all macro locations that we need to warn because they are not used.
  for (WarnUnusedMacroLocsTy::iterator
         I=WarnUnusedMacroLocs.begin(), E=WarnUnusedMacroLocs.end(); I!=E; ++I)
    Diag(*I, diag::pp_macro_not_used);

  return true;
}

/// HandleEndOfTokenLexer - This callback is invoked when the current TokenLexer
/// hits the end of its token stream.
bool Preprocessor::HandleEndOfTokenLexer(Token &Result) {
  assert(CurTokenLexer && !CurPPLexer &&
         "Ending a macro when currently in a #include file!");

  if (!MacroExpandingLexersStack.empty() &&
      MacroExpandingLexersStack.back().first == CurTokenLexer.get())
    removeCachedMacroExpandedTokensOfLastLexer();

  // Delete or cache the now-dead macro expander.
  if (NumCachedTokenLexers == TokenLexerCacheSize)
    CurTokenLexer.reset();
  else
    TokenLexerCache[NumCachedTokenLexers++] = CurTokenLexer.take();

  // Handle this like a #include file being popped off the stack.
  return HandleEndOfFile(Result, true);
}

/// RemoveTopOfLexerStack - Pop the current lexer/macro exp off the top of the
/// lexer stack.  This should only be used in situations where the current
/// state of the top-of-stack lexer is unknown.
void Preprocessor::RemoveTopOfLexerStack() {
  assert(!IncludeMacroStack.empty() && "Ran out of stack entries to load");

  if (CurTokenLexer) {
    // Delete or cache the now-dead macro expander.
    if (NumCachedTokenLexers == TokenLexerCacheSize)
      CurTokenLexer.reset();
    else
      TokenLexerCache[NumCachedTokenLexers++] = CurTokenLexer.take();
  }

  PopIncludeMacroStack();
}

/// HandleMicrosoftCommentPaste - When the macro expander pastes together a
/// comment (/##/) in microsoft mode, this method handles updating the current
/// state, returning the token on the next source line.
void Preprocessor::HandleMicrosoftCommentPaste(Token &Tok) {
  assert(CurTokenLexer && !CurPPLexer &&
         "Pasted comment can only be formed from macro");

  // We handle this by scanning for the closest real lexer, switching it to
  // raw mode and preprocessor mode.  This will cause it to return \n as an
  // explicit EOD token.
  PreprocessorLexer *FoundLexer = 0;
  bool LexerWasInPPMode = false;
  for (unsigned i = 0, e = IncludeMacroStack.size(); i != e; ++i) {
    IncludeStackInfo &ISI = *(IncludeMacroStack.end()-i-1);
    if (ISI.ThePPLexer == 0) continue;  // Scan for a real lexer.

    // Once we find a real lexer, mark it as raw mode (disabling macro
    // expansions) and preprocessor mode (return EOD).  We know that the lexer
    // was *not* in raw mode before, because the macro that the comment came
    // from was expanded.  However, it could have already been in preprocessor
    // mode (#if COMMENT) in which case we have to return it to that mode and
    // return EOD.
    FoundLexer = ISI.ThePPLexer;
    FoundLexer->LexingRawMode = true;
    LexerWasInPPMode = FoundLexer->ParsingPreprocessorDirective;
    FoundLexer->ParsingPreprocessorDirective = true;
    break;
  }

  // Okay, we either found and switched over the lexer, or we didn't find a
  // lexer.  In either case, finish off the macro the comment came from, getting
  // the next token.
  if (!HandleEndOfTokenLexer(Tok)) Lex(Tok);

  // Discarding comments as long as we don't have EOF or EOD.  This 'comments
  // out' the rest of the line, including any tokens that came from other macros
  // that were active, as in:
  //  #define submacro a COMMENT b
  //    submacro c
  // which should lex to 'a' only: 'b' and 'c' should be removed.
  while (Tok.isNot(tok::eod) && Tok.isNot(tok::eof))
    Lex(Tok);

  // If we got an eod token, then we successfully found the end of the line.
  if (Tok.is(tok::eod)) {
    assert(FoundLexer && "Can't get end of line without an active lexer");
    // Restore the lexer back to normal mode instead of raw mode.
    FoundLexer->LexingRawMode = false;

    // If the lexer was already in preprocessor mode, just return the EOD token
    // to finish the preprocessor line.
    if (LexerWasInPPMode) return;

    // Otherwise, switch out of PP mode and return the next lexed token.
    FoundLexer->ParsingPreprocessorDirective = false;
    return Lex(Tok);
  }

  // If we got an EOF token, then we reached the end of the token stream but
  // didn't find an explicit \n.  This can only happen if there was no lexer
  // active (an active lexer would return EOD at EOF if there was no \n in
  // preprocessor directive mode), so just return EOF as our token.
  assert(!FoundLexer && "Lexer should return EOD before EOF in PP mode");
}
