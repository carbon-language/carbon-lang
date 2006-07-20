//===--- MacroExpander.cpp - Lex from a macro expansion -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the MacroExpander interface.
//
//===----------------------------------------------------------------------===//

#include "clang/Lex/MacroExpander.h"
#include "clang/Lex/MacroInfo.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/Diagnostic.h"
#include "llvm/Config/Alloca.h"
using namespace llvm;
using namespace clang;

//===----------------------------------------------------------------------===//
// MacroArgs Implementation
//===----------------------------------------------------------------------===//

MacroArgs::MacroArgs(const MacroInfo *MI) {
  assert(MI->isFunctionLike() &&
         "Can't have args for an object-like macro!");
  // Reserve space for arguments to avoid reallocation.
  unsigned NumArgs = MI->getNumArgs();
  if (MI->isC99Varargs() || MI->isGNUVarargs())
    NumArgs += 3;    // Varargs can have more than this, just some guess.
  
  UnexpArgTokens.reserve(NumArgs);
}

/// addArgument - Add an argument for this invocation.  This method destroys
/// the vector passed in to avoid extraneous memory copies.  This adds the EOF
/// token to the end of the argument list as a marker.  'Loc' specifies a
/// location at the end of the argument, e.g. the ',' token or the ')'.
void MacroArgs::addArgument(std::vector<LexerToken> &ArgToks,
                            SourceLocation Loc) {
  UnexpArgTokens.push_back(std::vector<LexerToken>());
  UnexpArgTokens.back().swap(ArgToks);
  
  // Add a marker EOF token to the end of the argument list, useful for handling
  // empty arguments and macro pre-expansion.
  LexerToken EOFTok;
  EOFTok.StartToken();
  EOFTok.SetKind(tok::eof);
  EOFTok.SetLocation(Loc);
  EOFTok.SetLength(0);
  UnexpArgTokens.back().push_back(EOFTok);
}

/// ArgNeedsPreexpansion - If we can prove that the argument won't be affected
/// by pre-expansion, return false.  Otherwise, conservatively return true.
bool MacroArgs::ArgNeedsPreexpansion(unsigned ArgNo) const {
  const std::vector<LexerToken> &ArgTokens = getUnexpArgument(ArgNo);
  
  // If there are no identifiers in the argument list, or if the identifiers are
  // known to not be macros, pre-expansion won't modify it.
  for (unsigned i = 0, e = ArgTokens.size()-1; i != e; ++i)
    if (IdentifierInfo *II = ArgTokens[i].getIdentifierInfo()) {
      if (II->getMacroInfo() && II->getMacroInfo()->isEnabled())
        // Return true even though the macro could be a function-like macro
        // without a following '(' token.
        return true;
    }
  return false;
}

/// getPreExpArgument - Return the pre-expanded form of the specified
/// argument.
const std::vector<LexerToken> &
MacroArgs::getPreExpArgument(unsigned Arg, Preprocessor &PP) {
  assert(Arg < UnexpArgTokens.size() && "Invalid argument number!");
  
  // If we have already computed this, return it.
  if (PreExpArgTokens.empty())
    PreExpArgTokens.resize(UnexpArgTokens.size());

  std::vector<LexerToken> &Result = PreExpArgTokens[Arg];
  if (!Result.empty()) return Result;

  // Otherwise, we have to pre-expand this argument, populating Result.  To do
  // this, we set up a fake MacroExpander to lex from the unexpanded argument
  // list.  With this installed, we lex expanded tokens until we hit the EOF
  // token at the end of the unexp list.
  PP.EnterTokenStream(UnexpArgTokens[Arg]);

  // Lex all of the macro-expanded tokens into Result.
  do {
    Result.push_back(LexerToken());
    PP.Lex(Result.back());
  } while (Result.back().getKind() != tok::eof);
  
  // Pop the token stream off the top of the stack.  We know that the internal
  // pointer inside of it is to the "end" of the token stream, but the stack
  // will not otherwise be popped until the next token is lexed.  The problem is
  // that the token may be lexed sometime after the vector of tokens itself is
  // destroyed, which would be badness.
  PP.RemoveTopOfLexerStack();
  return Result;
}


/// StringifyArgument - Implement C99 6.10.3.2p2, converting a sequence of
/// tokens into the literal string token that should be produced by the C #
/// preprocessor operator.
///
static LexerToken StringifyArgument(const std::vector<LexerToken> &Toks,
                                    Preprocessor &PP, bool Charify = false) {
  LexerToken Tok;
  Tok.StartToken();
  Tok.SetKind(tok::string_literal);

  // Stringify all the tokens.
  std::string Result = "\"";
  // FIXME: Optimize this loop to not use std::strings.
  for (unsigned i = 0, e = Toks.size()-1 /*no eof*/; i != e; ++i) {
    const LexerToken &Tok = Toks[i];
    if (i != 0 && Tok.hasLeadingSpace())
      Result += ' ';
    
    // If this is a string or character constant, escape the token as specified
    // by 6.10.3.2p2.
    if (Tok.getKind() == tok::string_literal ||  // "foo" and L"foo".
        Tok.getKind() == tok::char_constant) {   // 'x' and L'x'.
      Result += Lexer::Stringify(PP.getSpelling(Tok));
    } else {
      // Otherwise, just append the token.
      Result += PP.getSpelling(Tok);
    }
  }
  
  // If the last character of the string is a \, and if it isn't escaped, this
  // is an invalid string literal, diagnose it as specified in C99.
  if (Result[Result.size()-1] == '\\') {
    // Count the number of consequtive \ characters.  If even, then they are
    // just escaped backslashes, otherwise it's an error.
    unsigned FirstNonSlash = Result.size()-2;
    // Guaranteed to find the starting " if nothing else.
    while (Result[FirstNonSlash] == '\\')
      --FirstNonSlash;
    if ((Result.size()-1-FirstNonSlash) & 1) {
      // Diagnose errors for things like: #define F(X) #X   /   F(\)
      PP.Diag(Toks.back(), diag::pp_invalid_string_literal);
      Result.erase(Result.end()-1);  // remove one of the \'s.
    }
  }
  Result += '"';
  
  // If this is the charify operation and the result is not a legal character
  // constant, diagnose it.
  if (Charify) {
    // First step, turn double quotes into single quotes:
    Result[0] = '\'';
    Result[Result.size()-1] = '\'';
    
    // Check for bogus character.
    bool isBad = false;
    if (Result.size() == 3) {
      isBad = Result[1] == '\'';   // ''' is not legal. '\' already fixed above.
    } else {
      isBad = (Result.size() != 4 || Result[1] != '\\');  // Not '\x'
    }
    
    if (isBad) {
      assert(!Toks.empty() && "No eof token at least?");
      PP.Diag(Toks[0], diag::err_invalid_character_to_charify);
      Result = "' '";  // Use something arbitrary, but legal.
    }
  }
  
  Tok.SetLength(Result.size());
  Tok.SetLocation(PP.CreateString(&Result[0], Result.size()));
  return Tok;
}

/// getStringifiedArgument - Compute, cache, and return the specified argument
/// that has been 'stringified' as required by the # operator.
const LexerToken &MacroArgs::getStringifiedArgument(unsigned ArgNo,
                                                    Preprocessor &PP) {
  assert(ArgNo < UnexpArgTokens.size() && "Invalid argument number!");
  if (StringifiedArgs.empty()) {
    StringifiedArgs.resize(getNumArguments());
    memset(&StringifiedArgs[0], 0,
           sizeof(StringifiedArgs[0])*getNumArguments());
  }
  if (StringifiedArgs[ArgNo].getKind() != tok::string_literal)
    StringifiedArgs[ArgNo] = StringifyArgument(UnexpArgTokens[ArgNo], PP);
  return StringifiedArgs[ArgNo];
}

//===----------------------------------------------------------------------===//
// MacroExpander Implementation
//===----------------------------------------------------------------------===//

/// Create a macro expander for the specified macro with the specified actual
/// arguments.  Note that this ctor takes ownership of the ActualArgs pointer.
MacroExpander::MacroExpander(LexerToken &Tok, MacroArgs *Actuals,
                             Preprocessor &pp)
  : Macro(Tok.getIdentifierInfo()->getMacroInfo()),
    ActualArgs(Actuals), PP(pp), CurToken(0),
    InstantiateLoc(Tok.getLocation()),
    AtStartOfLine(Tok.isAtStartOfLine()),
    HasLeadingSpace(Tok.hasLeadingSpace()) {
  MacroTokens = &Macro->getReplacementTokens();

  // If this is a function-like macro, expand the arguments and change
  // MacroTokens to point to the expanded tokens.
  if (Macro->isFunctionLike() && Macro->getNumArgs())
    ExpandFunctionArguments();
  
  // Mark the macro as currently disabled, so that it is not recursively
  // expanded.  The macro must be disabled only after argument pre-expansion of
  // function-like macro arguments occurs.
  Macro->DisableMacro();
}

/// Create a macro expander for the specified token stream.  This does not
/// take ownership of the specified token vector.
MacroExpander::MacroExpander(const std::vector<LexerToken> &TokStream, 
                             Preprocessor &pp)
  : Macro(0), ActualArgs(0), PP(pp), MacroTokens(&TokStream), CurToken(0),
    InstantiateLoc(SourceLocation()), AtStartOfLine(false), 
    HasLeadingSpace(false) {
      
  // Set HasLeadingSpace/AtStartOfLine so that the first token will be
  // returned unmodified.
  if (!TokStream.empty()) {
    AtStartOfLine   = TokStream[0].isAtStartOfLine();
    HasLeadingSpace = TokStream[0].hasLeadingSpace();
  }
}


MacroExpander::~MacroExpander() {
  // If this was a function-like macro that actually uses its arguments, delete
  // the expanded tokens.
  if (Macro && MacroTokens != &Macro->getReplacementTokens())
    delete MacroTokens;
  
  // MacroExpander owns its formal arguments.
  delete ActualArgs;
}

/// Expand the arguments of a function-like macro so that we can quickly
/// return preexpanded tokens from MacroTokens.
void MacroExpander::ExpandFunctionArguments() {
  std::vector<LexerToken> ResultToks;
  
  // Loop through the MacroTokens tokens, expanding them into ResultToks.  Keep
  // track of whether we change anything.  If not, no need to keep them.  If so,
  // we install the newly expanded sequence as MacroTokens.
  bool MadeChange = false;
  for (unsigned i = 0, e = MacroTokens->size(); i != e; ++i) {
    // If we found the stringify operator, get the argument stringified.  The
    // preprocessor already verified that the following token is a macro name
    // when the #define was parsed.
    const LexerToken &CurTok = (*MacroTokens)[i];
    if (CurTok.getKind() == tok::hash || CurTok.getKind() == tok::hashat) {
      int ArgNo =Macro->getArgumentNum((*MacroTokens)[i+1].getIdentifierInfo());
      assert(ArgNo != -1 && "Token following # is not an argument?");
      
      if (CurTok.getKind() == tok::hash)  // Stringify
        ResultToks.push_back(ActualArgs->getStringifiedArgument(ArgNo, PP));
      else {
        // 'charify': don't bother caching these.
        ResultToks.push_back(StringifyArgument(
                               ActualArgs->getUnexpArgument(ArgNo), PP, true));
      }
      
      // The stringified/charified string leading space flag gets set to match
      // the #/#@ operator.
      if (CurTok.hasLeadingSpace())
        ResultToks.back().SetFlag(LexerToken::LeadingSpace);
      
      MadeChange = true;
      ++i;  // Skip arg name.
    } else {
      // Otherwise, if this is not an argument token, just add the token to the
      // output buffer.
      IdentifierInfo *II = CurTok.getIdentifierInfo();
      int ArgNo = II ? Macro->getArgumentNum(II) : -1;
      if (ArgNo == -1) {
        ResultToks.push_back(CurTok);
        continue;
      }
      
      // An argument is expanded somehow, the result is different than the
      // input.
      MadeChange = true;

      // Otherwise, this is a use of the argument.  Find out if there is a paste
      // (##) operator before or after the argument.
      bool PasteBefore = 
        !ResultToks.empty() && ResultToks.back().getKind() == tok::hashhash;
      bool PasteAfter =
        i+1 != e && (*MacroTokens)[i+1].getKind() == tok::hashhash;
      
      // If it is not the LHS/RHS of a ## operator, we must pre-expand the
      // argument and substitute the expanded tokens into the result.  This is
      // C99 6.10.3.1p1.
      if (!PasteBefore && !PasteAfter) {
        const std::vector<LexerToken> *ArgToks;
        // Only preexpand the argument if it could possibly need it.  This
        // avoids some work in common cases.
        if (ActualArgs->ArgNeedsPreexpansion(ArgNo))
          ArgToks = &ActualArgs->getPreExpArgument(ArgNo, PP);
        else
          ArgToks = &ActualArgs->getUnexpArgument(ArgNo);
        
        unsigned FirstTok = ResultToks.size();
        ResultToks.insert(ResultToks.end(), ArgToks->begin(), ArgToks->end()-1);
        
        // If any tokens were substituted from the argument, the whitespace
        // before the first token should match the whitespace of the arg
        // identifier.
        if (FirstTok != ResultToks.size())
          ResultToks[FirstTok].SetFlagValue(LexerToken::LeadingSpace,
                                            CurTok.hasLeadingSpace());
        continue;
      }
      
      // Okay, we have a token that is either the LHS or RHS of a paste (##)
      // argument.  It gets substituted as its non-pre-expanded tokens.
      const std::vector<LexerToken> &ArgToks =
        ActualArgs->getUnexpArgument(ArgNo);
      assert(ArgToks.back().getKind() == tok::eof && "Bad argument!");

      if (ArgToks.size() != 1) {  // Not just an EOF token?
        ResultToks.insert(ResultToks.end(), ArgToks.begin(), ArgToks.end()-1);
        continue;
      }
      
      // FIXME: Handle comma swallowing GNU extension.
      // FIXME: Handle 'placemarker' stuff.
      assert(0 && "FIXME: handle empty arguments!");
      //ResultToks.push_back(CurTok);
    }
  }
  
  // If anything changed, install this as the new MacroTokens list.
  if (MadeChange) {
    // This is deleted in the dtor.
    std::vector<LexerToken> *Res = new std::vector<LexerToken>();
    Res->swap(ResultToks);
    MacroTokens = Res;
  }
}

/// Lex - Lex and return a token from this macro stream.
///
void MacroExpander::Lex(LexerToken &Tok) {
  // Lexing off the end of the macro, pop this macro off the expansion stack.
  if (isAtEnd()) {
    // If this is a macro (not a token stream), mark the macro enabled now
    // that it is no longer being expanded.
    if (Macro) Macro->EnableMacro();

    // Pop this context off the preprocessors lexer stack and get the next
    // token.  This will delete "this" so remember the PP instance var.
    Preprocessor &PPCache = PP;
    if (PP.HandleEndOfMacro(Tok))
      return;

    // HandleEndOfMacro may not return a token.  If it doesn't, lex whatever is
    // next.
    return PPCache.Lex(Tok);
  }
  
  // If this is the first token of the expanded result, we inherit spacing
  // properties later.
  bool isFirstToken = CurToken == 0;
  
  // Get the next token to return.
  Tok = (*MacroTokens)[CurToken++];
  
  // If this token is followed by a token paste (##) operator, paste the tokens!
  if (!isAtEnd() && (*MacroTokens)[CurToken].getKind() == tok::hashhash)
    PasteTokens(Tok);

  // The token's current location indicate where the token was lexed from.  We
  // need this information to compute the spelling of the token, but any
  // diagnostics for the expanded token should appear as if they came from
  // InstantiationLoc.  Pull this information together into a new SourceLocation
  // that captures all of this.
  if (InstantiateLoc.isValid()) {   // Don't do this for token streams.
    SourceManager &SrcMgr = PP.getSourceManager();
    // The token could have come from a prior macro expansion.  In that case,
    // ignore the macro expand part to get to the physloc.  This happens for
    // stuff like:  #define A(X) X    A(A(X))    A(1)
    SourceLocation PhysLoc = SrcMgr.getPhysicalLoc(Tok.getLocation());
    Tok.SetLocation(SrcMgr.getInstantiationLoc(PhysLoc, InstantiateLoc));
  }
  
  // If this is the first token, set the lexical properties of the token to
  // match the lexical properties of the macro identifier.
  if (isFirstToken) {
    Tok.SetFlagValue(LexerToken::StartOfLine , AtStartOfLine);
    Tok.SetFlagValue(LexerToken::LeadingSpace, HasLeadingSpace);
  }
  
  // Handle recursive expansion!
  if (Tok.getIdentifierInfo())
    return PP.HandleIdentifier(Tok);

  // Otherwise, return a normal token.
}

/// PasteTokens - Tok is the LHS of a ## operator, and CurToken is the ##
/// operator.  Read the ## and RHS, and paste the LHS/RHS together.  If there
/// are is another ## after it, chomp it iteratively.  Return the result as Tok.
void MacroExpander::PasteTokens(LexerToken &Tok) {
  do {
    // Consume the ## operator.
    SourceLocation PasteOpLoc = (*MacroTokens)[CurToken].getLocation();
    ++CurToken;
    assert(!isAtEnd() && "No token on the RHS of a paste operator!");
  
    // Get the RHS token.
    const LexerToken &RHS = (*MacroTokens)[CurToken];
  
    bool isInvalid = false;

    // Allocate space for the result token.  This is guaranteed to be enough for
    // the two tokens and a null terminator.
    char *Buffer = (char*)alloca(Tok.getLength() + RHS.getLength() + 1);
    
    // Get the spelling of the LHS token in Buffer.
    const char *BufPtr = Buffer;
    unsigned LHSLen = PP.getSpelling(Tok, BufPtr);
    if (BufPtr != Buffer)   // Really, we want the chars in Buffer!
      memcpy(Buffer, BufPtr, LHSLen);
    
    BufPtr = Buffer+LHSLen;
    unsigned RHSLen = PP.getSpelling(RHS, BufPtr);
    if (BufPtr != Buffer+LHSLen)   // Really, we want the chars in Buffer!
      memcpy(Buffer+LHSLen, BufPtr, RHSLen);
    
    // Add null terminator.
    Buffer[LHSLen+RHSLen] = '\0';
    
    // Plop the pasted result (including the trailing newline and null) into a
    // scratch buffer where we can lex it.
    SourceLocation ResultTokLoc = PP.CreateString(Buffer, LHSLen+RHSLen+1);
    
    // Lex the resultant pasted token into Result.
    LexerToken Result;
    
    // Avoid testing /*, as the lexer would think it is the start of a comment
    // and emit an error that it is unterminated.
    if (Tok.getKind() == tok::slash && RHS.getKind() == tok::star) {
      isInvalid = true;
    } else {
      // FIXME: Handle common cases: ident+ident, ident+simplenumber here.

      // Make a lexer to lex this string from.
      SourceManager &SourceMgr = PP.getSourceManager();
      const char *ResultStrData = SourceMgr.getCharacterData(ResultTokLoc);
      
      unsigned FileID = ResultTokLoc.getFileID();
      assert(FileID && "Could not get FileID for paste?");
      
      // Make a lexer object so that we lex and expand the paste result.
      Lexer *TL = new Lexer(SourceMgr.getBuffer(FileID), FileID, PP,
                            ResultStrData, 
                            ResultStrData+LHSLen+RHSLen /*don't include null*/);
      
      // Lex a token in raw mode.  This way it won't look up identifiers
      // automatically, lexing off the end will return an eof token, and
      // warnings are disabled.  This returns true if the result token is the
      // entire buffer.
      bool IsComplete = TL->LexRawToken(Result);
      
      // If we got an EOF token, we didn't form even ONE token.  For example, we
      // did "/ ## /" to get "//".
      IsComplete &= Result.getKind() != tok::eof;
      isInvalid = !IsComplete;
      
      // We're now done with the temporary lexer.
      delete TL;
    }
    
    // If pasting the two tokens didn't form a full new token, this is an error.
    // This occurs with "x ## +"  and other stuff.  Return with Tok unmodified
    // and with RHS as the next token to lex.
    if (isInvalid) {
      // If not in assembler language mode.
      PP.Diag(PasteOpLoc, diag::err_pp_bad_paste, 
              std::string(Buffer, Buffer+LHSLen+RHSLen));
      return;
    }
    
    // Turn ## into 'other' to avoid # ## # from looking like a paste operator.
    if (Result.getKind() == tok::hashhash)
      Result.SetKind(tok::unknown);
    // FIXME: Turn __VARRGS__ into "not a token"?
    
    // Transfer properties of the LHS over the the Result.
    Result.SetFlagValue(LexerToken::StartOfLine , Tok.isAtStartOfLine());
    Result.SetFlagValue(LexerToken::LeadingSpace, Tok.hasLeadingSpace());
    
    // Finally, replace LHS with the result, consume the RHS, and iterate.
    ++CurToken;
    Tok = Result;
  } while (!isAtEnd() && (*MacroTokens)[CurToken].getKind() == tok::hashhash);
  
  // Now that we got the result token, it will be subject to expansion.  Since
  // token pasting re-lexes the result token in raw mode, identifier information
  // isn't looked up.  As such, if the result is an identifier, look up id info.
  if (Tok.getKind() == tok::identifier) {
    // Look up the identifier info for the token.  We disabled identifier lookup
    // by saying we're skipping contents, so we need to do this manually.
    Tok.SetIdentifierInfo(PP.LookUpIdentifierInfo(Tok));
  }
}

/// isNextTokenLParen - If the next token lexed will pop this macro off the
/// expansion stack, return 2.  If the next unexpanded token is a '(', return
/// 1, otherwise return 0.
unsigned MacroExpander::isNextTokenLParen() const {
  // Out of tokens?
  if (isAtEnd())
    return 2;
  return (*MacroTokens)[CurToken].getKind() == tok::l_paren;
}
