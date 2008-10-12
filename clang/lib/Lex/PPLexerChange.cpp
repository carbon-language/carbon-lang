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
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceManager.h"
using namespace clang;

PPCallbacks::~PPCallbacks() {
}


//===----------------------------------------------------------------------===//
// Miscellaneous Methods.
//===----------------------------------------------------------------------===//

/// isInPrimaryFile - Return true if we're in the top-level file, not in a
/// #include.  This looks through macro expansions and active _Pragma lexers.
bool Preprocessor::isInPrimaryFile() const {
  if (CurLexer && !CurLexer->Is_PragmaLexer)
    return IncludeMacroStack.empty();
  
  // If there are any stacked lexers, we're in a #include.
  assert(IncludeMacroStack[0].TheLexer &&
         !IncludeMacroStack[0].TheLexer->Is_PragmaLexer &&
         "Top level include stack isn't our primary lexer?");
  for (unsigned i = 1, e = IncludeMacroStack.size(); i != e; ++i)
    if (IncludeMacroStack[i].TheLexer &&
        !IncludeMacroStack[i].TheLexer->Is_PragmaLexer)
      return false;
  return true;
}

/// getCurrentLexer - Return the current file lexer being lexed from.  Note
/// that this ignores any potentially active macro expansions and _Pragma
/// expansions going on at the time.
Lexer *Preprocessor::getCurrentFileLexer() const {
  if (CurLexer && !CurLexer->Is_PragmaLexer) return CurLexer;
  
  // Look for a stacked lexer.
  for (unsigned i = IncludeMacroStack.size(); i != 0; --i) {
    Lexer *L = IncludeMacroStack[i-1].TheLexer;
    if (L && !L->Is_PragmaLexer) // Ignore macro & _Pragma expansions.
      return L;
  }
  return 0;
}


//===----------------------------------------------------------------------===//
// Methods for Entering and Callbacks for leaving various contexts
//===----------------------------------------------------------------------===//

/// EnterSourceFile - Add a source file to the top of the include stack and
/// start lexing tokens from it instead of the current buffer.  Return true
/// on failure.
void Preprocessor::EnterSourceFile(unsigned FileID,
                                   const DirectoryLookup *CurDir) {
  assert(CurTokenLexer == 0 && "Cannot #include a file inside a macro!");
  ++NumEnteredSourceFiles;
  
  if (MaxIncludeStackDepth < IncludeMacroStack.size())
    MaxIncludeStackDepth = IncludeMacroStack.size();

  Lexer *TheLexer = new Lexer(SourceLocation::getFileLoc(FileID, 0), *this);
  EnterSourceFileWithLexer(TheLexer, CurDir);
}  

/// EnterSourceFile - Add a source file to the top of the include stack and
/// start lexing tokens from it instead of the current buffer.
void Preprocessor::EnterSourceFileWithLexer(Lexer *TheLexer, 
                                            const DirectoryLookup *CurDir) {
    
  // Add the current lexer to the include stack.
  if (CurLexer || CurTokenLexer)
    IncludeMacroStack.push_back(IncludeStackInfo(CurLexer, CurDirLookup,
                                                 CurTokenLexer));
  
  CurLexer = TheLexer;
  CurDirLookup = CurDir;
  CurTokenLexer = 0;
  
  // Notify the client, if desired, that we are in a new source file.
  if (Callbacks && !CurLexer->Is_PragmaLexer) {
    SrcMgr::Characteristic_t FileType =
       SourceMgr.getFileCharacteristic(CurLexer->getFileLoc());
    
    Callbacks->FileChanged(CurLexer->getFileLoc(),
                           PPCallbacks::EnterFile, FileType);
  }
}



/// EnterMacro - Add a Macro to the top of the include stack and start lexing
/// tokens from it instead of the current buffer.
void Preprocessor::EnterMacro(Token &Tok, MacroArgs *Args) {
  IncludeMacroStack.push_back(IncludeStackInfo(CurLexer, CurDirLookup,
                                               CurTokenLexer));
  CurLexer     = 0;
  CurDirLookup = 0;
  
  if (NumCachedTokenLexers == 0) {
    CurTokenLexer = new TokenLexer(Tok, Args, *this);
  } else {
    CurTokenLexer = TokenLexerCache[--NumCachedTokenLexers];
    CurTokenLexer->Init(Tok, Args);
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
  IncludeMacroStack.push_back(IncludeStackInfo(CurLexer, CurDirLookup,
                                               CurTokenLexer));
  CurLexer     = 0;
  CurDirLookup = 0;

  // Create a macro expander to expand from the specified token stream.
  if (NumCachedTokenLexers == 0) {
    CurTokenLexer = new TokenLexer(Toks, NumToks, DisableMacroExpansion,
                                   OwnsTokens, *this);
  } else {
    CurTokenLexer = TokenLexerCache[--NumCachedTokenLexers];
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
  if (CurLexer) {  // Not ending a macro, ignore it.
    if (const IdentifierInfo *ControllingMacro = 
          CurLexer->MIOpt.GetControllingMacroAtEndOfFile()) {
      // Okay, this has a controlling macro, remember in PerFileInfo.
      if (const FileEntry *FE = 
            SourceMgr.getFileEntryForLoc(CurLexer->getFileLoc()))
        HeaderInfo.SetFileControllingMacro(FE, ControllingMacro);
    }
  }
  
  // If this is a #include'd file, pop it off the include stack and continue
  // lexing the #includer file.
  if (!IncludeMacroStack.empty()) {
    // We're done with the #included file.
    RemoveTopOfLexerStack();

    // Notify the client, if desired, that we are in a new source file.
    if (Callbacks && !isEndOfMacro && CurLexer) {
      SrcMgr::Characteristic_t FileType =
        SourceMgr.getFileCharacteristic(CurLexer->getFileLoc());
      
      Callbacks->FileChanged(CurLexer->getSourceLocation(CurLexer->BufferPtr),
                             PPCallbacks::ExitFile, FileType);
    }

    // Client should lex another token.
    return false;
  }

  // If the file ends with a newline, form the EOF token on the newline itself,
  // rather than "on the line following it", which doesn't exist.  This makes
  // diagnostics relating to the end of file include the last file that the user
  // actually typed, which is goodness.
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
  delete CurLexer;
  CurLexer = 0;

  // This is the end of the top-level file.  If the diag::pp_macro_not_used
  // diagnostic is enabled, look for macros that have not been used.
  if (Diags.getDiagnosticLevel(diag::pp_macro_not_used) != Diagnostic::Ignored){
    for (llvm::DenseMap<IdentifierInfo*, MacroInfo*>::iterator I =
         Macros.begin(), E = Macros.end(); I != E; ++I) {
      if (!I->second->isUsed())
        Diag(I->second->getDefinitionLoc(), diag::pp_macro_not_used);
    }
  }
  return true;
}

/// HandleEndOfTokenLexer - This callback is invoked when the current TokenLexer
/// hits the end of its token stream.
bool Preprocessor::HandleEndOfTokenLexer(Token &Result) {
  assert(CurTokenLexer && !CurLexer &&
         "Ending a macro when currently in a #include file!");

  // Delete or cache the now-dead macro expander.
  if (NumCachedTokenLexers == TokenLexerCacheSize)
    delete CurTokenLexer;
  else
    TokenLexerCache[NumCachedTokenLexers++] = CurTokenLexer;

  // Handle this like a #include file being popped off the stack.
  CurTokenLexer = 0;
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
      delete CurTokenLexer;
    else
      TokenLexerCache[NumCachedTokenLexers++] = CurTokenLexer;
  } else {
    delete CurLexer;
  }
  CurLexer      = IncludeMacroStack.back().TheLexer;
  CurDirLookup  = IncludeMacroStack.back().TheDirLookup;
  CurTokenLexer = IncludeMacroStack.back().TheTokenLexer;
  IncludeMacroStack.pop_back();
}

/// HandleMicrosoftCommentPaste - When the macro expander pastes together a
/// comment (/##/) in microsoft mode, this method handles updating the current
/// state, returning the token on the next source line.
void Preprocessor::HandleMicrosoftCommentPaste(Token &Tok) {
  assert(CurTokenLexer && !CurLexer &&
         "Pasted comment can only be formed from macro");
  
  // We handle this by scanning for the closest real lexer, switching it to
  // raw mode and preprocessor mode.  This will cause it to return \n as an
  // explicit EOM token.
  Lexer *FoundLexer = 0;
  bool LexerWasInPPMode = false;
  for (unsigned i = 0, e = IncludeMacroStack.size(); i != e; ++i) {
    IncludeStackInfo &ISI = *(IncludeMacroStack.end()-i-1);
    if (ISI.TheLexer == 0) continue;  // Scan for a real lexer.
    
    // Once we find a real lexer, mark it as raw mode (disabling macro
    // expansions) and preprocessor mode (return EOM).  We know that the lexer
    // was *not* in raw mode before, because the macro that the comment came
    // from was expanded.  However, it could have already been in preprocessor
    // mode (#if COMMENT) in which case we have to return it to that mode and
    // return EOM.
    FoundLexer = ISI.TheLexer;
    FoundLexer->LexingRawMode = true;
    LexerWasInPPMode = FoundLexer->ParsingPreprocessorDirective;
    FoundLexer->ParsingPreprocessorDirective = true;
    break;
  }
  
  // Okay, we either found and switched over the lexer, or we didn't find a
  // lexer.  In either case, finish off the macro the comment came from, getting
  // the next token.
  if (!HandleEndOfTokenLexer(Tok)) Lex(Tok);
  
  // Discarding comments as long as we don't have EOF or EOM.  This 'comments
  // out' the rest of the line, including any tokens that came from other macros
  // that were active, as in:
  //  #define submacro a COMMENT b
  //    submacro c
  // which should lex to 'a' only: 'b' and 'c' should be removed.
  while (Tok.isNot(tok::eom) && Tok.isNot(tok::eof))
    Lex(Tok);
  
  // If we got an eom token, then we successfully found the end of the line.
  if (Tok.is(tok::eom)) {
    assert(FoundLexer && "Can't get end of line without an active lexer");
    // Restore the lexer back to normal mode instead of raw mode.
    FoundLexer->LexingRawMode = false;
    
    // If the lexer was already in preprocessor mode, just return the EOM token
    // to finish the preprocessor line.
    if (LexerWasInPPMode) return;
    
    // Otherwise, switch out of PP mode and return the next lexed token.
    FoundLexer->ParsingPreprocessorDirective = false;
    return Lex(Tok);
  }
  
  // If we got an EOF token, then we reached the end of the token stream but
  // didn't find an explicit \n.  This can only happen if there was no lexer
  // active (an active lexer would return EOM at EOF if there was no \n in
  // preprocessor directive mode), so just return EOF as our token.
  assert(!FoundLexer && "Lexer should return EOM before EOF in PP mode");
}
