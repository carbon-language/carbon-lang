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
  for (unsigned i = 0, e = Toks.size()-1 /*no eof*/; i != e; ++i) {
    const LexerToken &Tok = Toks[i];
    // FIXME: Optimize this.
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

MacroExpander::MacroExpander(LexerToken &Tok, MacroArgs *Actuals,
                             Preprocessor &pp)
  : Macro(*Tok.getIdentifierInfo()->getMacroInfo()),
    ActualArgs(Actuals), PP(pp), CurToken(0),
    InstantiateLoc(Tok.getLocation()),
    AtStartOfLine(Tok.isAtStartOfLine()),
    HasLeadingSpace(Tok.hasLeadingSpace()) {
  MacroTokens = &Macro.getReplacementTokens();

  // If this is a function-like macro, expand the arguments and change
  // MacroTokens to point to the expanded tokens.
  if (Macro.isFunctionLike() && Macro.getNumArgs())
    ExpandFunctionArguments();
}

MacroExpander::~MacroExpander() {
  // If this was a function-like macro that actually uses its arguments, delete
  // the expanded tokens.
  if (MacroTokens != &Macro.getReplacementTokens())
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
      int ArgNo = Macro.getArgumentNum((*MacroTokens)[i+1].getIdentifierInfo());
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
      int ArgNo = II ? Macro.getArgumentNum(II) : -1;
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
        if (ActualArgs->ArgNeedsPreexpansion(ArgNo)) {
          // FIXME: WRONG
          ArgToks = &ActualArgs->getUnexpArgument(ArgNo);
        } else {
          // If we don't need to pre-expand the argument, just substitute in the
          // unexpanded tokens.
          ArgToks = &ActualArgs->getUnexpArgument(ArgNo);
        }
        
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
      
      // FIXME: handle pasted args.      
      ResultToks.push_back(CurTok);
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
  if (isAtEnd())
    return PP.HandleEndOfMacro(Tok);
  
  // Get the next token to return.
  Tok = (*MacroTokens)[CurToken++];

  // The token's current location indicate where the token was lexed from.  We
  // need this information to compute the spelling of the token, but any
  // diagnostics for the expanded token should appear as if they came from
  // InstantiationLoc.  Pull this information together into a new SourceLocation
  // that captures all of this.
  Tok.SetLocation(PP.getSourceManager().getInstantiationLoc(Tok.getLocation(),
                                                            InstantiateLoc));

  // If this is the first token, set the lexical properties of the token to
  // match the lexical properties of the macro identifier.
  if (CurToken == 1) {
    Tok.SetFlagValue(LexerToken::StartOfLine , AtStartOfLine);
    Tok.SetFlagValue(LexerToken::LeadingSpace, HasLeadingSpace);
  }
  
  // Handle recursive expansion!
  if (Tok.getIdentifierInfo())
    return PP.HandleIdentifier(Tok);

  // Otherwise, return a normal token.
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
