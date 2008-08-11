//===--- PPDirectives.cpp - Directive Handling for Preprocessor -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements # directive processing for the Preprocessor.
//
//===----------------------------------------------------------------------===//

#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/MacroInfo.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceManager.h"
using namespace clang;

//===----------------------------------------------------------------------===//
// Utility Methods for Preprocessor Directive Handling.
//===----------------------------------------------------------------------===//

/// DiscardUntilEndOfDirective - Read and discard all tokens remaining on the
/// current line until the tok::eom token is found.
void Preprocessor::DiscardUntilEndOfDirective() {
  Token Tmp;
  do {
    LexUnexpandedToken(Tmp);
  } while (Tmp.isNot(tok::eom));
}

/// isCXXNamedOperator - Returns "true" if the token is a named operator in C++.
static bool isCXXNamedOperator(const std::string &Spelling) {
  return Spelling == "and" || Spelling == "bitand" || Spelling == "bitor" ||
    Spelling == "compl" || Spelling == "not" || Spelling == "not_eq" ||
    Spelling == "or" || Spelling == "xor";
}

/// ReadMacroName - Lex and validate a macro name, which occurs after a
/// #define or #undef.  This sets the token kind to eom and discards the rest
/// of the macro line if the macro name is invalid.  isDefineUndef is 1 if
/// this is due to a a #define, 2 if #undef directive, 0 if it is something
/// else (e.g. #ifdef).
void Preprocessor::ReadMacroName(Token &MacroNameTok, char isDefineUndef) {
  // Read the token, don't allow macro expansion on it.
  LexUnexpandedToken(MacroNameTok);
  
  // Missing macro name?
  if (MacroNameTok.is(tok::eom))
    return Diag(MacroNameTok, diag::err_pp_missing_macro_name);
  
  IdentifierInfo *II = MacroNameTok.getIdentifierInfo();
  if (II == 0) {
    std::string Spelling = getSpelling(MacroNameTok);
    if (isCXXNamedOperator(Spelling))
      // C++ 2.5p2: Alternative tokens behave the same as its primary token
      // except for their spellings.
      Diag(MacroNameTok, diag::err_pp_operator_used_as_macro_name, Spelling);
    else
      Diag(MacroNameTok, diag::err_pp_macro_not_identifier);
    // Fall through on error.
  } else if (isDefineUndef && II->getPPKeywordID() == tok::pp_defined) {
    // Error if defining "defined": C99 6.10.8.4.
    Diag(MacroNameTok, diag::err_defined_macro_name);
  } else if (isDefineUndef && II->hasMacroDefinition() &&
             getMacroInfo(II)->isBuiltinMacro()) {
    // Error if defining "__LINE__" and other builtins: C99 6.10.8.4.
    if (isDefineUndef == 1)
      Diag(MacroNameTok, diag::pp_redef_builtin_macro);
    else
      Diag(MacroNameTok, diag::pp_undef_builtin_macro);
  } else {
    // Okay, we got a good identifier node.  Return it.
    return;
  }
  
  // Invalid macro name, read and discard the rest of the line.  Then set the
  // token kind to tok::eom.
  MacroNameTok.setKind(tok::eom);
  return DiscardUntilEndOfDirective();
}

/// CheckEndOfDirective - Ensure that the next token is a tok::eom token.  If
/// not, emit a diagnostic and consume up until the eom.
void Preprocessor::CheckEndOfDirective(const char *DirType) {
  Token Tmp;
  // Lex unexpanded tokens: macros might expand to zero tokens, causing us to
  // miss diagnosing invalid lines.
  LexUnexpandedToken(Tmp);
  
  // There should be no tokens after the directive, but we allow them as an
  // extension.
  while (Tmp.is(tok::comment))  // Skip comments in -C mode.
    LexUnexpandedToken(Tmp);
  
  if (Tmp.isNot(tok::eom)) {
    Diag(Tmp, diag::ext_pp_extra_tokens_at_eol, DirType);
    DiscardUntilEndOfDirective();
  }
}



/// SkipExcludedConditionalBlock - We just read a #if or related directive and
/// decided that the subsequent tokens are in the #if'd out portion of the
/// file.  Lex the rest of the file, until we see an #endif.  If
/// FoundNonSkipPortion is true, then we have already emitted code for part of
/// this #if directive, so #else/#elif blocks should never be entered. If ElseOk
/// is true, then #else directives are ok, if not, then we have already seen one
/// so a #else directive is a duplicate.  When this returns, the caller can lex
/// the first valid token.
void Preprocessor::SkipExcludedConditionalBlock(SourceLocation IfTokenLoc,
                                                bool FoundNonSkipPortion,
                                                bool FoundElse) {
  ++NumSkipped;
  assert(CurTokenLexer == 0 && CurLexer &&
         "Lexing a macro, not a file?");

  CurLexer->pushConditionalLevel(IfTokenLoc, /*isSkipping*/false,
                                 FoundNonSkipPortion, FoundElse);
  
  // Enter raw mode to disable identifier lookup (and thus macro expansion),
  // disabling warnings, etc.
  CurLexer->LexingRawMode = true;
  Token Tok;
  while (1) {
    CurLexer->Lex(Tok);
    
    // If this is the end of the buffer, we have an error.
    if (Tok.is(tok::eof)) {
      // Emit errors for each unterminated conditional on the stack, including
      // the current one.
      while (!CurLexer->ConditionalStack.empty()) {
        Diag(CurLexer->ConditionalStack.back().IfLoc,
             diag::err_pp_unterminated_conditional);
        CurLexer->ConditionalStack.pop_back();
      }  
      
      // Just return and let the caller lex after this #include.
      break;
    }
    
    // If this token is not a preprocessor directive, just skip it.
    if (Tok.isNot(tok::hash) || !Tok.isAtStartOfLine())
      continue;
      
    // We just parsed a # character at the start of a line, so we're in
    // directive mode.  Tell the lexer this so any newlines we see will be
    // converted into an EOM token (this terminates the macro).
    CurLexer->ParsingPreprocessorDirective = true;
    CurLexer->KeepCommentMode = false;

    
    // Read the next token, the directive flavor.
    LexUnexpandedToken(Tok);
    
    // If this isn't an identifier directive (e.g. is "# 1\n" or "#\n", or
    // something bogus), skip it.
    if (Tok.isNot(tok::identifier)) {
      CurLexer->ParsingPreprocessorDirective = false;
      // Restore comment saving mode.
      CurLexer->KeepCommentMode = KeepComments;
      continue;
    }

    // If the first letter isn't i or e, it isn't intesting to us.  We know that
    // this is safe in the face of spelling differences, because there is no way
    // to spell an i/e in a strange way that is another letter.  Skipping this
    // allows us to avoid looking up the identifier info for #define/#undef and
    // other common directives.
    const char *RawCharData = SourceMgr.getCharacterData(Tok.getLocation());
    char FirstChar = RawCharData[0];
    if (FirstChar >= 'a' && FirstChar <= 'z' && 
        FirstChar != 'i' && FirstChar != 'e') {
      CurLexer->ParsingPreprocessorDirective = false;
      // Restore comment saving mode.
      CurLexer->KeepCommentMode = KeepComments;
      continue;
    }
    
    // Get the identifier name without trigraphs or embedded newlines.  Note
    // that we can't use Tok.getIdentifierInfo() because its lookup is disabled
    // when skipping.
    // TODO: could do this with zero copies in the no-clean case by using
    // strncmp below.
    char Directive[20];
    unsigned IdLen;
    if (!Tok.needsCleaning() && Tok.getLength() < 20) {
      IdLen = Tok.getLength();
      memcpy(Directive, RawCharData, IdLen);
      Directive[IdLen] = 0;
    } else {
      std::string DirectiveStr = getSpelling(Tok);
      IdLen = DirectiveStr.size();
      if (IdLen >= 20) {
        CurLexer->ParsingPreprocessorDirective = false;
        // Restore comment saving mode.
        CurLexer->KeepCommentMode = KeepComments;
        continue;
      }
      memcpy(Directive, &DirectiveStr[0], IdLen);
      Directive[IdLen] = 0;
    }
    
    if (FirstChar == 'i' && Directive[1] == 'f') {
      if ((IdLen == 2) ||   // "if"
          (IdLen == 5 && !strcmp(Directive+2, "def")) ||   // "ifdef"
          (IdLen == 6 && !strcmp(Directive+2, "ndef"))) {  // "ifndef"
        // We know the entire #if/#ifdef/#ifndef block will be skipped, don't
        // bother parsing the condition.
        DiscardUntilEndOfDirective();
        CurLexer->pushConditionalLevel(Tok.getLocation(), /*wasskipping*/true,
                                       /*foundnonskip*/false,
                                       /*fnddelse*/false);
      }
    } else if (FirstChar == 'e') {
      if (IdLen == 5 && !strcmp(Directive+1, "ndif")) {  // "endif"
        CheckEndOfDirective("#endif");
        PPConditionalInfo CondInfo;
        CondInfo.WasSkipping = true; // Silence bogus warning.
        bool InCond = CurLexer->popConditionalLevel(CondInfo);
        InCond = InCond;  // Silence warning in no-asserts mode.
        assert(!InCond && "Can't be skipping if not in a conditional!");
        
        // If we popped the outermost skipping block, we're done skipping!
        if (!CondInfo.WasSkipping)
          break;
      } else if (IdLen == 4 && !strcmp(Directive+1, "lse")) { // "else".
        // #else directive in a skipping conditional.  If not in some other
        // skipping conditional, and if #else hasn't already been seen, enter it
        // as a non-skipping conditional.
        CheckEndOfDirective("#else");
        PPConditionalInfo &CondInfo = CurLexer->peekConditionalLevel();
        
        // If this is a #else with a #else before it, report the error.
        if (CondInfo.FoundElse) Diag(Tok, diag::pp_err_else_after_else);
        
        // Note that we've seen a #else in this conditional.
        CondInfo.FoundElse = true;
        
        // If the conditional is at the top level, and the #if block wasn't
        // entered, enter the #else block now.
        if (!CondInfo.WasSkipping && !CondInfo.FoundNonSkip) {
          CondInfo.FoundNonSkip = true;
          break;
        }
      } else if (IdLen == 4 && !strcmp(Directive+1, "lif")) {  // "elif".
        PPConditionalInfo &CondInfo = CurLexer->peekConditionalLevel();

        bool ShouldEnter;
        // If this is in a skipping block or if we're already handled this #if
        // block, don't bother parsing the condition.
        if (CondInfo.WasSkipping || CondInfo.FoundNonSkip) {
          DiscardUntilEndOfDirective();
          ShouldEnter = false;
        } else {
          // Restore the value of LexingRawMode so that identifiers are
          // looked up, etc, inside the #elif expression.
          assert(CurLexer->LexingRawMode && "We have to be skipping here!");
          CurLexer->LexingRawMode = false;
          IdentifierInfo *IfNDefMacro = 0;
          ShouldEnter = EvaluateDirectiveExpression(IfNDefMacro);
          CurLexer->LexingRawMode = true;
        }
        
        // If this is a #elif with a #else before it, report the error.
        if (CondInfo.FoundElse) Diag(Tok, diag::pp_err_elif_after_else);
        
        // If this condition is true, enter it!
        if (ShouldEnter) {
          CondInfo.FoundNonSkip = true;
          break;
        }
      }
    }
    
    CurLexer->ParsingPreprocessorDirective = false;
    // Restore comment saving mode.
    CurLexer->KeepCommentMode = KeepComments;
  }

  // Finally, if we are out of the conditional (saw an #endif or ran off the end
  // of the file, just stop skipping and return to lexing whatever came after
  // the #if block.
  CurLexer->LexingRawMode = false;
}

/// LookupFile - Given a "foo" or <foo> reference, look up the indicated file,
/// return null on failure.  isAngled indicates whether the file reference is
/// for system #include's or not (i.e. using <> instead of "").
const FileEntry *Preprocessor::LookupFile(const char *FilenameStart,
                                          const char *FilenameEnd,
                                          bool isAngled,
                                          const DirectoryLookup *FromDir,
                                          const DirectoryLookup *&CurDir) {
  // If the header lookup mechanism may be relative to the current file, pass in
  // info about where the current file is.
  const FileEntry *CurFileEnt = 0;
  if (!FromDir) {
    SourceLocation FileLoc = getCurrentFileLexer()->getFileLoc();
    CurFileEnt = SourceMgr.getFileEntryForLoc(FileLoc);
  }
  
  // Do a standard file entry lookup.
  CurDir = CurDirLookup;
  const FileEntry *FE =
  HeaderInfo.LookupFile(FilenameStart, FilenameEnd,
                        isAngled, FromDir, CurDir, CurFileEnt);
  if (FE) return FE;
  
  // Otherwise, see if this is a subframework header.  If so, this is relative
  // to one of the headers on the #include stack.  Walk the list of the current
  // headers on the #include stack and pass them to HeaderInfo.
  if (CurLexer && !CurLexer->Is_PragmaLexer) {
    if ((CurFileEnt = SourceMgr.getFileEntryForLoc(CurLexer->getFileLoc())))
      if ((FE = HeaderInfo.LookupSubframeworkHeader(FilenameStart, FilenameEnd,
                                                    CurFileEnt)))
        return FE;
  }
  
  for (unsigned i = 0, e = IncludeMacroStack.size(); i != e; ++i) {
    IncludeStackInfo &ISEntry = IncludeMacroStack[e-i-1];
    if (ISEntry.TheLexer && !ISEntry.TheLexer->Is_PragmaLexer) {
      if ((CurFileEnt = 
           SourceMgr.getFileEntryForLoc(ISEntry.TheLexer->getFileLoc())))
        if ((FE = HeaderInfo.LookupSubframeworkHeader(FilenameStart,
                                                      FilenameEnd, CurFileEnt)))
          return FE;
    }
  }
  
  // Otherwise, we really couldn't find the file.
  return 0;
}


//===----------------------------------------------------------------------===//
// Preprocessor Directive Handling.
//===----------------------------------------------------------------------===//

/// HandleDirective - This callback is invoked when the lexer sees a # token
/// at the start of a line.  This consumes the directive, modifies the 
/// lexer/preprocessor state, and advances the lexer(s) so that the next token
/// read is the correct one.
void Preprocessor::HandleDirective(Token &Result) {
  // FIXME: Traditional: # with whitespace before it not recognized by K&R?
  
  // We just parsed a # character at the start of a line, so we're in directive
  // mode.  Tell the lexer this so any newlines we see will be converted into an
  // EOM token (which terminates the directive).
  CurLexer->ParsingPreprocessorDirective = true;
  
  ++NumDirectives;
  
  // We are about to read a token.  For the multiple-include optimization FA to
  // work, we have to remember if we had read any tokens *before* this 
  // pp-directive.
  bool ReadAnyTokensBeforeDirective = CurLexer->MIOpt.getHasReadAnyTokensVal();
  
  // Read the next token, the directive flavor.  This isn't expanded due to
  // C99 6.10.3p8.
  LexUnexpandedToken(Result);
  
  // C99 6.10.3p11: Is this preprocessor directive in macro invocation?  e.g.:
  //   #define A(x) #x
  //   A(abc
  //     #warning blah
  //   def)
  // If so, the user is relying on non-portable behavior, emit a diagnostic.
  if (InMacroArgs)
    Diag(Result, diag::ext_embedded_directive);
  
TryAgain:
  switch (Result.getKind()) {
  case tok::eom:
    return;   // null directive.
  case tok::comment:
    // Handle stuff like "# /*foo*/ define X" in -E -C mode.
    LexUnexpandedToken(Result);
    goto TryAgain;

  case tok::numeric_constant:
    // FIXME: implement # 7 line numbers!
    DiscardUntilEndOfDirective();
    return;
  default:
    IdentifierInfo *II = Result.getIdentifierInfo();
    if (II == 0) break;  // Not an identifier.
      
    // Ask what the preprocessor keyword ID is.
    switch (II->getPPKeywordID()) {
    default: break;
    // C99 6.10.1 - Conditional Inclusion.
    case tok::pp_if:
      return HandleIfDirective(Result, ReadAnyTokensBeforeDirective);
    case tok::pp_ifdef:
      return HandleIfdefDirective(Result, false, true/*not valid for miopt*/);
    case tok::pp_ifndef:
      return HandleIfdefDirective(Result, true, ReadAnyTokensBeforeDirective);
    case tok::pp_elif:
      return HandleElifDirective(Result);
    case tok::pp_else:
      return HandleElseDirective(Result);
    case tok::pp_endif:
      return HandleEndifDirective(Result);
      
    // C99 6.10.2 - Source File Inclusion.
    case tok::pp_include:
      return HandleIncludeDirective(Result);            // Handle #include.

    // C99 6.10.3 - Macro Replacement.
    case tok::pp_define:
      return HandleDefineDirective(Result);
    case tok::pp_undef:
      return HandleUndefDirective(Result);

    // C99 6.10.4 - Line Control.
    case tok::pp_line:
      // FIXME: implement #line
      DiscardUntilEndOfDirective();
      return;
      
    // C99 6.10.5 - Error Directive.
    case tok::pp_error:
      return HandleUserDiagnosticDirective(Result, false);
      
    // C99 6.10.6 - Pragma Directive.
    case tok::pp_pragma:
      return HandlePragmaDirective();
      
    // GNU Extensions.
    case tok::pp_import:
      return HandleImportDirective(Result);
    case tok::pp_include_next:
      return HandleIncludeNextDirective(Result);
      
    case tok::pp_warning:
      Diag(Result, diag::ext_pp_warning_directive);
      return HandleUserDiagnosticDirective(Result, true);
    case tok::pp_ident:
      return HandleIdentSCCSDirective(Result);
    case tok::pp_sccs:
      return HandleIdentSCCSDirective(Result);
    case tok::pp_assert:
      //isExtension = true;  // FIXME: implement #assert
      break;
    case tok::pp_unassert:
      //isExtension = true;  // FIXME: implement #unassert
      break;
    }
    break;
  }
  
  // If we reached here, the preprocessing token is not valid!
  Diag(Result, diag::err_pp_invalid_directive);
  
  // Read the rest of the PP line.
  DiscardUntilEndOfDirective();
  
  // Okay, we're done parsing the directive.
}

void Preprocessor::HandleUserDiagnosticDirective(Token &Tok, 
                                                 bool isWarning) {
  // Read the rest of the line raw.  We do this because we don't want macros
  // to be expanded and we don't require that the tokens be valid preprocessing
  // tokens.  For example, this is allowed: "#warning `   'foo".  GCC does
  // collapse multiple consequtive white space between tokens, but this isn't
  // specified by the standard.
  std::string Message = CurLexer->ReadToEndOfLine();

  unsigned DiagID = isWarning ? diag::pp_hash_warning : diag::err_pp_hash_error;
  return Diag(Tok, DiagID, Message);
}

/// HandleIdentSCCSDirective - Handle a #ident/#sccs directive.
///
void Preprocessor::HandleIdentSCCSDirective(Token &Tok) {
  // Yes, this directive is an extension.
  Diag(Tok, diag::ext_pp_ident_directive);
  
  // Read the string argument.
  Token StrTok;
  Lex(StrTok);
  
  // If the token kind isn't a string, it's a malformed directive.
  if (StrTok.isNot(tok::string_literal) &&
      StrTok.isNot(tok::wide_string_literal))
    return Diag(StrTok, diag::err_pp_malformed_ident);
  
  // Verify that there is nothing after the string, other than EOM.
  CheckEndOfDirective("#ident");

  if (Callbacks)
    Callbacks->Ident(Tok.getLocation(), getSpelling(StrTok));
}

//===----------------------------------------------------------------------===//
// Preprocessor Include Directive Handling.
//===----------------------------------------------------------------------===//

/// GetIncludeFilenameSpelling - Turn the specified lexer token into a fully
/// checked and spelled filename, e.g. as an operand of #include. This returns
/// true if the input filename was in <>'s or false if it were in ""'s.  The
/// caller is expected to provide a buffer that is large enough to hold the
/// spelling of the filename, but is also expected to handle the case when
/// this method decides to use a different buffer.
bool Preprocessor::GetIncludeFilenameSpelling(SourceLocation Loc,
                                              const char *&BufStart,
                                              const char *&BufEnd) {
  // Get the text form of the filename.
  assert(BufStart != BufEnd && "Can't have tokens with empty spellings!");
  
  // Make sure the filename is <x> or "x".
  bool isAngled;
  if (BufStart[0] == '<') {
    if (BufEnd[-1] != '>') {
      Diag(Loc, diag::err_pp_expects_filename);
      BufStart = 0;
      return true;
    }
    isAngled = true;
  } else if (BufStart[0] == '"') {
    if (BufEnd[-1] != '"') {
      Diag(Loc, diag::err_pp_expects_filename);
      BufStart = 0;
      return true;
    }
    isAngled = false;
  } else {
    Diag(Loc, diag::err_pp_expects_filename);
    BufStart = 0;
    return true;
  }
  
  // Diagnose #include "" as invalid.
  if (BufEnd-BufStart <= 2) {
    Diag(Loc, diag::err_pp_empty_filename);
    BufStart = 0;
    return "";
  }
  
  // Skip the brackets.
  ++BufStart;
  --BufEnd;
  return isAngled;
}

/// ConcatenateIncludeName - Handle cases where the #include name is expanded
/// from a macro as multiple tokens, which need to be glued together.  This
/// occurs for code like:
///    #define FOO <a/b.h>
///    #include FOO
/// because in this case, "<a/b.h>" is returned as 7 tokens, not one.
///
/// This code concatenates and consumes tokens up to the '>' token.  It returns
/// false if the > was found, otherwise it returns true if it finds and consumes
/// the EOM marker.
static bool ConcatenateIncludeName(llvm::SmallVector<char, 128> &FilenameBuffer,
                                   Preprocessor &PP) {
  Token CurTok;
  
  PP.Lex(CurTok);
  while (CurTok.isNot(tok::eom)) {
    // Append the spelling of this token to the buffer. If there was a space
    // before it, add it now.
    if (CurTok.hasLeadingSpace())
      FilenameBuffer.push_back(' ');
    
    // Get the spelling of the token, directly into FilenameBuffer if possible.
    unsigned PreAppendSize = FilenameBuffer.size();
    FilenameBuffer.resize(PreAppendSize+CurTok.getLength());
    
    const char *BufPtr = &FilenameBuffer[PreAppendSize];
    unsigned ActualLen = PP.getSpelling(CurTok, BufPtr);
    
    // If the token was spelled somewhere else, copy it into FilenameBuffer.
    if (BufPtr != &FilenameBuffer[PreAppendSize])
      memcpy(&FilenameBuffer[PreAppendSize], BufPtr, ActualLen);
    
    // Resize FilenameBuffer to the correct size.
    if (CurTok.getLength() != ActualLen)
      FilenameBuffer.resize(PreAppendSize+ActualLen);
    
    // If we found the '>' marker, return success.
    if (CurTok.is(tok::greater))
      return false;
    
    PP.Lex(CurTok);
  }

  // If we hit the eom marker, emit an error and return true so that the caller
  // knows the EOM has been read.
  PP.Diag(CurTok.getLocation(), diag::err_pp_expects_filename);
  return true;
}

/// HandleIncludeDirective - The "#include" tokens have just been read, read the
/// file to be included from the lexer, then include it!  This is a common
/// routine with functionality shared between #include, #include_next and
/// #import.
void Preprocessor::HandleIncludeDirective(Token &IncludeTok,
                                          const DirectoryLookup *LookupFrom,
                                          bool isImport) {

  Token FilenameTok;
  CurLexer->LexIncludeFilename(FilenameTok);
  
  // Reserve a buffer to get the spelling.
  llvm::SmallVector<char, 128> FilenameBuffer;
  const char *FilenameStart, *FilenameEnd;

  switch (FilenameTok.getKind()) {
  case tok::eom:
    // If the token kind is EOM, the error has already been diagnosed.
    return;
  
  case tok::angle_string_literal:
  case tok::string_literal: {
    FilenameBuffer.resize(FilenameTok.getLength());
    FilenameStart = &FilenameBuffer[0];
    unsigned Len = getSpelling(FilenameTok, FilenameStart);
    FilenameEnd = FilenameStart+Len;
    break;
  }
    
  case tok::less:
    // This could be a <foo/bar.h> file coming from a macro expansion.  In this
    // case, glue the tokens together into FilenameBuffer and interpret those.
    FilenameBuffer.push_back('<');
    if (ConcatenateIncludeName(FilenameBuffer, *this))
      return;   // Found <eom> but no ">"?  Diagnostic already emitted.
    FilenameStart = &FilenameBuffer[0];
    FilenameEnd = &FilenameBuffer[FilenameBuffer.size()];
    break;
  default:
    Diag(FilenameTok.getLocation(), diag::err_pp_expects_filename);
    DiscardUntilEndOfDirective();
    return;
  }
  
  bool isAngled = GetIncludeFilenameSpelling(FilenameTok.getLocation(),
                                             FilenameStart, FilenameEnd);
  // If GetIncludeFilenameSpelling set the start ptr to null, there was an
  // error.
  if (FilenameStart == 0) {
    DiscardUntilEndOfDirective();
    return;
  }
  
  // Verify that there is nothing after the filename, other than EOM.  Use the
  // preprocessor to lex this in case lexing the filename entered a macro.
  CheckEndOfDirective("#include");

  // Check that we don't have infinite #include recursion.
  if (IncludeMacroStack.size() == MaxAllowedIncludeStackDepth-1)
    return Diag(FilenameTok, diag::err_pp_include_too_deep);
  
  // Search include directories.
  const DirectoryLookup *CurDir;
  const FileEntry *File = LookupFile(FilenameStart, FilenameEnd,
                                     isAngled, LookupFrom, CurDir);
  if (File == 0)
    return Diag(FilenameTok, diag::err_pp_file_not_found,
                std::string(FilenameStart, FilenameEnd));
  
  // Ask HeaderInfo if we should enter this #include file.
  if (!HeaderInfo.ShouldEnterIncludeFile(File, isImport)) {
    // If it returns true, #including this file will have no effect.
    return;
  }

  // Look up the file, create a File ID for it.
  unsigned FileID = SourceMgr.createFileID(File, FilenameTok.getLocation(),
                                           isSystemHeader(File));
  if (FileID == 0)
    return Diag(FilenameTok, diag::err_pp_file_not_found,
                std::string(FilenameStart, FilenameEnd));

  // Finally, if all is good, enter the new file!
  EnterSourceFile(FileID, CurDir);
}

/// HandleIncludeNextDirective - Implements #include_next.
///
void Preprocessor::HandleIncludeNextDirective(Token &IncludeNextTok) {
  Diag(IncludeNextTok, diag::ext_pp_include_next_directive);
  
  // #include_next is like #include, except that we start searching after
  // the current found directory.  If we can't do this, issue a
  // diagnostic.
  const DirectoryLookup *Lookup = CurDirLookup;
  if (isInPrimaryFile()) {
    Lookup = 0;
    Diag(IncludeNextTok, diag::pp_include_next_in_primary);
  } else if (Lookup == 0) {
    Diag(IncludeNextTok, diag::pp_include_next_absolute_path);
  } else {
    // Start looking up in the next directory.
    ++Lookup;
  }
  
  return HandleIncludeDirective(IncludeNextTok, Lookup);
}

/// HandleImportDirective - Implements #import.
///
void Preprocessor::HandleImportDirective(Token &ImportTok) {
  Diag(ImportTok, diag::ext_pp_import_directive);
  
  return HandleIncludeDirective(ImportTok, 0, true);
}

//===----------------------------------------------------------------------===//
// Preprocessor Macro Directive Handling.
//===----------------------------------------------------------------------===//

/// ReadMacroDefinitionArgList - The ( starting an argument list of a macro
/// definition has just been read.  Lex the rest of the arguments and the
/// closing ), updating MI with what we learn.  Return true if an error occurs
/// parsing the arg list.
bool Preprocessor::ReadMacroDefinitionArgList(MacroInfo *MI) {
  llvm::SmallVector<IdentifierInfo*, 32> Arguments;
  
  Token Tok;
  while (1) {
    LexUnexpandedToken(Tok);
    switch (Tok.getKind()) {
    case tok::r_paren:
      // Found the end of the argument list.
      if (Arguments.empty()) {  // #define FOO()
        MI->setArgumentList(Arguments.begin(), Arguments.end());
        return false;
      }
      // Otherwise we have #define FOO(A,)
      Diag(Tok, diag::err_pp_expected_ident_in_arg_list);
      return true;
    case tok::ellipsis:  // #define X(... -> C99 varargs
      // Warn if use of C99 feature in non-C99 mode.
      if (!Features.C99) Diag(Tok, diag::ext_variadic_macro);

      // Lex the token after the identifier.
      LexUnexpandedToken(Tok);
      if (Tok.isNot(tok::r_paren)) {
        Diag(Tok, diag::err_pp_missing_rparen_in_macro_def);
        return true;
      }
      // Add the __VA_ARGS__ identifier as an argument.
      Arguments.push_back(Ident__VA_ARGS__);
      MI->setIsC99Varargs();
      MI->setArgumentList(Arguments.begin(), Arguments.end());
      return false;
    case tok::eom:  // #define X(
      Diag(Tok, diag::err_pp_missing_rparen_in_macro_def);
      return true;
    default:
      // Handle keywords and identifiers here to accept things like
      // #define Foo(for) for.
      IdentifierInfo *II = Tok.getIdentifierInfo();
      if (II == 0) {
        // #define X(1
        Diag(Tok, diag::err_pp_invalid_tok_in_arg_list);
        return true;
      }

      // If this is already used as an argument, it is used multiple times (e.g.
      // #define X(A,A.
      if (std::find(Arguments.begin(), Arguments.end(), II) != 
          Arguments.end()) {  // C99 6.10.3p6
        Diag(Tok, diag::err_pp_duplicate_name_in_arg_list, II->getName());
        return true;
      }
        
      // Add the argument to the macro info.
      Arguments.push_back(II);
      
      // Lex the token after the identifier.
      LexUnexpandedToken(Tok);
      
      switch (Tok.getKind()) {
      default:          // #define X(A B
        Diag(Tok, diag::err_pp_expected_comma_in_arg_list);
        return true;
      case tok::r_paren: // #define X(A)
        MI->setArgumentList(Arguments.begin(), Arguments.end());
        return false;
      case tok::comma:  // #define X(A,
        break;
      case tok::ellipsis:  // #define X(A... -> GCC extension
        // Diagnose extension.
        Diag(Tok, diag::ext_named_variadic_macro);
        
        // Lex the token after the identifier.
        LexUnexpandedToken(Tok);
        if (Tok.isNot(tok::r_paren)) {
          Diag(Tok, diag::err_pp_missing_rparen_in_macro_def);
          return true;
        }
          
        MI->setIsGNUVarargs();
        MI->setArgumentList(Arguments.begin(), Arguments.end());
        return false;
      }
    }
  }
}

/// HandleDefineDirective - Implements #define.  This consumes the entire macro
/// line then lets the caller lex the next real token.
void Preprocessor::HandleDefineDirective(Token &DefineTok) {
  ++NumDefined;

  Token MacroNameTok;
  ReadMacroName(MacroNameTok, 1);
  
  // Error reading macro name?  If so, diagnostic already issued.
  if (MacroNameTok.is(tok::eom))
    return;

  // If we are supposed to keep comments in #defines, reenable comment saving
  // mode.
  CurLexer->KeepCommentMode = KeepMacroComments;
  
  // Create the new macro.
  MacroInfo *MI = new MacroInfo(MacroNameTok.getLocation());
  
  Token Tok;
  LexUnexpandedToken(Tok);
  
  // If this is a function-like macro definition, parse the argument list,
  // marking each of the identifiers as being used as macro arguments.  Also,
  // check other constraints on the first token of the macro body.
  if (Tok.is(tok::eom)) {
    // If there is no body to this macro, we have no special handling here.
  } else if (Tok.is(tok::l_paren) && !Tok.hasLeadingSpace()) {
    // This is a function-like macro definition.  Read the argument list.
    MI->setIsFunctionLike();
    if (ReadMacroDefinitionArgList(MI)) {
      // Forget about MI.
      delete MI;
      // Throw away the rest of the line.
      if (CurLexer->ParsingPreprocessorDirective)
        DiscardUntilEndOfDirective();
      return;
    }

    // Read the first token after the arg list for down below.
    LexUnexpandedToken(Tok);
  } else if (!Tok.hasLeadingSpace()) {
    // C99 requires whitespace between the macro definition and the body.  Emit
    // a diagnostic for something like "#define X+".
    if (Features.C99) {
      Diag(Tok, diag::ext_c99_whitespace_required_after_macro_name);
    } else {
      // FIXME: C90/C++ do not get this diagnostic, but it does get a similar
      // one in some cases!
    }
  } else {
    // This is a normal token with leading space.  Clear the leading space
    // marker on the first token to get proper expansion.
    Tok.clearFlag(Token::LeadingSpace);
  }
  
  // If this is a definition of a variadic C99 function-like macro, not using
  // the GNU named varargs extension, enabled __VA_ARGS__.
  
  // "Poison" __VA_ARGS__, which can only appear in the expansion of a macro.
  // This gets unpoisoned where it is allowed.
  assert(Ident__VA_ARGS__->isPoisoned() && "__VA_ARGS__ should be poisoned!");
  if (MI->isC99Varargs())
    Ident__VA_ARGS__->setIsPoisoned(false);
  
  // Read the rest of the macro body.
  if (MI->isObjectLike()) {
    // Object-like macros are very simple, just read their body.
    while (Tok.isNot(tok::eom)) {
      MI->AddTokenToBody(Tok);
      // Get the next token of the macro.
      LexUnexpandedToken(Tok);
    }
    
  } else {
    // Otherwise, read the body of a function-like macro.  This has to validate
    // the # (stringize) operator.
    while (Tok.isNot(tok::eom)) {
      MI->AddTokenToBody(Tok);

      // Check C99 6.10.3.2p1: ensure that # operators are followed by macro
      // parameters in function-like macro expansions.
      if (Tok.isNot(tok::hash)) {
        // Get the next token of the macro.
        LexUnexpandedToken(Tok);
        continue;
      }
      
      // Get the next token of the macro.
      LexUnexpandedToken(Tok);
     
      // Not a macro arg identifier?
      if (!Tok.getIdentifierInfo() ||
          MI->getArgumentNum(Tok.getIdentifierInfo()) == -1) {
        Diag(Tok, diag::err_pp_stringize_not_parameter);
        delete MI;
        
        // Disable __VA_ARGS__ again.
        Ident__VA_ARGS__->setIsPoisoned(true);
        return;
      }
      
      // Things look ok, add the param name token to the macro.
      MI->AddTokenToBody(Tok);

      // Get the next token of the macro.
      LexUnexpandedToken(Tok);
    }
  }
  
  
  // Disable __VA_ARGS__ again.
  Ident__VA_ARGS__->setIsPoisoned(true);

  // Check that there is no paste (##) operator at the begining or end of the
  // replacement list.
  unsigned NumTokens = MI->getNumTokens();
  if (NumTokens != 0) {
    if (MI->getReplacementToken(0).is(tok::hashhash)) {
      Diag(MI->getReplacementToken(0), diag::err_paste_at_start);
      delete MI;
      return;
    }
    if (MI->getReplacementToken(NumTokens-1).is(tok::hashhash)) {
      Diag(MI->getReplacementToken(NumTokens-1), diag::err_paste_at_end);
      delete MI;
      return;
    }
  }
  
  // If this is the primary source file, remember that this macro hasn't been
  // used yet.
  if (isInPrimaryFile())
    MI->setIsUsed(false);
  
  // Finally, if this identifier already had a macro defined for it, verify that
  // the macro bodies are identical and free the old definition.
  if (MacroInfo *OtherMI = getMacroInfo(MacroNameTok.getIdentifierInfo())) {
    if (!OtherMI->isUsed())
      Diag(OtherMI->getDefinitionLoc(), diag::pp_macro_not_used);

    // Macros must be identical.  This means all tokes and whitespace separation
    // must be the same.  C99 6.10.3.2.
    if (!MI->isIdenticalTo(*OtherMI, *this)) {
      Diag(MI->getDefinitionLoc(), diag::ext_pp_macro_redef,
           MacroNameTok.getIdentifierInfo()->getName());
      Diag(OtherMI->getDefinitionLoc(), diag::ext_pp_macro_redef2);
    }
    delete OtherMI;
  }
  
  setMacroInfo(MacroNameTok.getIdentifierInfo(), MI);
}

/// HandleUndefDirective - Implements #undef.
///
void Preprocessor::HandleUndefDirective(Token &UndefTok) {
  ++NumUndefined;

  Token MacroNameTok;
  ReadMacroName(MacroNameTok, 2);
  
  // Error reading macro name?  If so, diagnostic already issued.
  if (MacroNameTok.is(tok::eom))
    return;
  
  // Check to see if this is the last token on the #undef line.
  CheckEndOfDirective("#undef");
  
  // Okay, we finally have a valid identifier to undef.
  MacroInfo *MI = getMacroInfo(MacroNameTok.getIdentifierInfo());
  
  // If the macro is not defined, this is a noop undef, just return.
  if (MI == 0) return;

  if (!MI->isUsed())
    Diag(MI->getDefinitionLoc(), diag::pp_macro_not_used);
  
  // Free macro definition.
  delete MI;
  setMacroInfo(MacroNameTok.getIdentifierInfo(), 0);
}


//===----------------------------------------------------------------------===//
// Preprocessor Conditional Directive Handling.
//===----------------------------------------------------------------------===//

/// HandleIfdefDirective - Implements the #ifdef/#ifndef directive.  isIfndef is
/// true when this is a #ifndef directive.  ReadAnyTokensBeforeDirective is true
/// if any tokens have been returned or pp-directives activated before this
/// #ifndef has been lexed.
///
void Preprocessor::HandleIfdefDirective(Token &Result, bool isIfndef,
                                        bool ReadAnyTokensBeforeDirective) {
  ++NumIf;
  Token DirectiveTok = Result;

  Token MacroNameTok;
  ReadMacroName(MacroNameTok);
  
  // Error reading macro name?  If so, diagnostic already issued.
  if (MacroNameTok.is(tok::eom)) {
    // Skip code until we get to #endif.  This helps with recovery by not
    // emitting an error when the #endif is reached.
    SkipExcludedConditionalBlock(DirectiveTok.getLocation(),
                                 /*Foundnonskip*/false, /*FoundElse*/false);
    return;
  }
  
  // Check to see if this is the last token on the #if[n]def line.
  CheckEndOfDirective(isIfndef ? "#ifndef" : "#ifdef");

  if (CurLexer->getConditionalStackDepth() == 0) {
    // If the start of a top-level #ifdef, inform MIOpt.
    if (!ReadAnyTokensBeforeDirective) {
      assert(isIfndef && "#ifdef shouldn't reach here");
      CurLexer->MIOpt.EnterTopLevelIFNDEF(MacroNameTok.getIdentifierInfo());
    } else
      CurLexer->MIOpt.EnterTopLevelConditional();
  }

  IdentifierInfo *MII = MacroNameTok.getIdentifierInfo();
  MacroInfo *MI = getMacroInfo(MII);

  // If there is a macro, process it.
  if (MI)  // Mark it used.
    MI->setIsUsed(true);
  
  // Should we include the stuff contained by this directive?
  if (!MI == isIfndef) {
    // Yes, remember that we are inside a conditional, then lex the next token.
    CurLexer->pushConditionalLevel(DirectiveTok.getLocation(), /*wasskip*/false,
                                   /*foundnonskip*/true, /*foundelse*/false);
  } else {
    // No, skip the contents of this block and return the first token after it.
    SkipExcludedConditionalBlock(DirectiveTok.getLocation(),
                                 /*Foundnonskip*/false, 
                                 /*FoundElse*/false);
  }
}

/// HandleIfDirective - Implements the #if directive.
///
void Preprocessor::HandleIfDirective(Token &IfToken,
                                     bool ReadAnyTokensBeforeDirective) {
  ++NumIf;
  
  // Parse and evaluation the conditional expression.
  IdentifierInfo *IfNDefMacro = 0;
  bool ConditionalTrue = EvaluateDirectiveExpression(IfNDefMacro);
  

  // If this condition is equivalent to #ifndef X, and if this is the first
  // directive seen, handle it for the multiple-include optimization.
  if (CurLexer->getConditionalStackDepth() == 0) {
    if (!ReadAnyTokensBeforeDirective && IfNDefMacro)
      CurLexer->MIOpt.EnterTopLevelIFNDEF(IfNDefMacro);
    else
      CurLexer->MIOpt.EnterTopLevelConditional();
  }

  // Should we include the stuff contained by this directive?
  if (ConditionalTrue) {
    // Yes, remember that we are inside a conditional, then lex the next token.
    CurLexer->pushConditionalLevel(IfToken.getLocation(), /*wasskip*/false,
                                   /*foundnonskip*/true, /*foundelse*/false);
  } else {
    // No, skip the contents of this block and return the first token after it.
    SkipExcludedConditionalBlock(IfToken.getLocation(), /*Foundnonskip*/false, 
                                 /*FoundElse*/false);
  }
}

/// HandleEndifDirective - Implements the #endif directive.
///
void Preprocessor::HandleEndifDirective(Token &EndifToken) {
  ++NumEndif;
  
  // Check that this is the whole directive.
  CheckEndOfDirective("#endif");
  
  PPConditionalInfo CondInfo;
  if (CurLexer->popConditionalLevel(CondInfo)) {
    // No conditionals on the stack: this is an #endif without an #if.
    return Diag(EndifToken, diag::err_pp_endif_without_if);
  }
  
  // If this the end of a top-level #endif, inform MIOpt.
  if (CurLexer->getConditionalStackDepth() == 0)
    CurLexer->MIOpt.ExitTopLevelConditional();
  
  assert(!CondInfo.WasSkipping && !CurLexer->LexingRawMode &&
         "This code should only be reachable in the non-skipping case!");
}


void Preprocessor::HandleElseDirective(Token &Result) {
  ++NumElse;
  
  // #else directive in a non-skipping conditional... start skipping.
  CheckEndOfDirective("#else");
  
  PPConditionalInfo CI;
  if (CurLexer->popConditionalLevel(CI))
    return Diag(Result, diag::pp_err_else_without_if);
  
  // If this is a top-level #else, inform the MIOpt.
  if (CurLexer->getConditionalStackDepth() == 0)
    CurLexer->MIOpt.EnterTopLevelConditional();

  // If this is a #else with a #else before it, report the error.
  if (CI.FoundElse) Diag(Result, diag::pp_err_else_after_else);
  
  // Finally, skip the rest of the contents of this block and return the first
  // token after it.
  return SkipExcludedConditionalBlock(CI.IfLoc, /*Foundnonskip*/true,
                                      /*FoundElse*/true);
}

void Preprocessor::HandleElifDirective(Token &ElifToken) {
  ++NumElse;
  
  // #elif directive in a non-skipping conditional... start skipping.
  // We don't care what the condition is, because we will always skip it (since
  // the block immediately before it was included).
  DiscardUntilEndOfDirective();

  PPConditionalInfo CI;
  if (CurLexer->popConditionalLevel(CI))
    return Diag(ElifToken, diag::pp_err_elif_without_if);
  
  // If this is a top-level #elif, inform the MIOpt.
  if (CurLexer->getConditionalStackDepth() == 0)
    CurLexer->MIOpt.EnterTopLevelConditional();
  
  // If this is a #elif with a #else before it, report the error.
  if (CI.FoundElse) Diag(ElifToken, diag::pp_err_elif_after_else);

  // Finally, skip the rest of the contents of this block and return the first
  // token after it.
  return SkipExcludedConditionalBlock(CI.IfLoc, /*Foundnonskip*/true,
                                      /*FoundElse*/CI.FoundElse);
}

