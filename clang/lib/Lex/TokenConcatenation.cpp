//===--- TokenConcatenation.cpp - Token Concatenation Avoidance -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the TokenConcatenation class.
//
//===----------------------------------------------------------------------===//

#include "clang/Lex/TokenConcatenation.h"
#include "clang/Lex/Preprocessor.h"
using namespace clang; 


/// StartsWithL - Return true if the spelling of this token starts with 'L'.
bool TokenConcatenation::StartsWithL(const Token &Tok) const {
  if (!Tok.needsCleaning()) {
    SourceManager &SM = PP.getSourceManager();
    return *SM.getCharacterData(SM.getSpellingLoc(Tok.getLocation())) == 'L';
  }
  
  if (Tok.getLength() < 256) {
    char Buffer[256];
    const char *TokPtr = Buffer;
    PP.getSpelling(Tok, TokPtr);
    return TokPtr[0] == 'L';
  }
  
  return PP.getSpelling(Tok)[0] == 'L';
}

/// IsIdentifierL - Return true if the spelling of this token is literally
/// 'L'.
bool TokenConcatenation::IsIdentifierL(const Token &Tok) const {
  if (!Tok.needsCleaning()) {
    if (Tok.getLength() != 1)
      return false;
    SourceManager &SM = PP.getSourceManager();
    return *SM.getCharacterData(SM.getSpellingLoc(Tok.getLocation())) == 'L';
  }
  
  if (Tok.getLength() < 256) {
    char Buffer[256];
    const char *TokPtr = Buffer;
    if (PP.getSpelling(Tok, TokPtr) != 1) 
      return false;
    return TokPtr[0] == 'L';
  }
  
  return PP.getSpelling(Tok) == "L";
}

TokenConcatenation::TokenConcatenation(Preprocessor &pp) : PP(pp) {
  memset(TokenInfo, 0, sizeof(TokenInfo));
  
  // These tokens have custom code in AvoidConcat.
  TokenInfo[tok::identifier      ] |= aci_custom;
  TokenInfo[tok::numeric_constant] |= aci_custom_firstchar;
  TokenInfo[tok::period          ] |= aci_custom_firstchar;
  TokenInfo[tok::amp             ] |= aci_custom_firstchar;
  TokenInfo[tok::plus            ] |= aci_custom_firstchar;
  TokenInfo[tok::minus           ] |= aci_custom_firstchar;
  TokenInfo[tok::slash           ] |= aci_custom_firstchar;
  TokenInfo[tok::less            ] |= aci_custom_firstchar;
  TokenInfo[tok::greater         ] |= aci_custom_firstchar;
  TokenInfo[tok::pipe            ] |= aci_custom_firstchar;
  TokenInfo[tok::percent         ] |= aci_custom_firstchar;
  TokenInfo[tok::colon           ] |= aci_custom_firstchar;
  TokenInfo[tok::hash            ] |= aci_custom_firstchar;
  TokenInfo[tok::arrow           ] |= aci_custom_firstchar;
  
  // These tokens change behavior if followed by an '='.
  TokenInfo[tok::amp         ] |= aci_avoid_equal;           // &=
  TokenInfo[tok::plus        ] |= aci_avoid_equal;           // +=
  TokenInfo[tok::minus       ] |= aci_avoid_equal;           // -=
  TokenInfo[tok::slash       ] |= aci_avoid_equal;           // /=
  TokenInfo[tok::less        ] |= aci_avoid_equal;           // <=
  TokenInfo[tok::greater     ] |= aci_avoid_equal;           // >=
  TokenInfo[tok::pipe        ] |= aci_avoid_equal;           // |=
  TokenInfo[tok::percent     ] |= aci_avoid_equal;           // %=
  TokenInfo[tok::star        ] |= aci_avoid_equal;           // *=
  TokenInfo[tok::exclaim     ] |= aci_avoid_equal;           // !=
  TokenInfo[tok::lessless    ] |= aci_avoid_equal;           // <<=
  TokenInfo[tok::greaterequal] |= aci_avoid_equal;           // >>=
  TokenInfo[tok::caret       ] |= aci_avoid_equal;           // ^=
  TokenInfo[tok::equal       ] |= aci_avoid_equal;           // ==
}

/// GetFirstChar - Get the first character of the token \arg Tok,
/// avoiding calls to getSpelling where possible.
static char GetFirstChar(Preprocessor &PP, const Token &Tok) {
  if (IdentifierInfo *II = Tok.getIdentifierInfo()) {
    // Avoid spelling identifiers, the most common form of token.
    return II->getName()[0];
  } else if (!Tok.needsCleaning()) {
    if (Tok.isLiteral() && Tok.getLiteralData()) {
      return *Tok.getLiteralData();
    } else {
      SourceManager &SM = PP.getSourceManager();
      return *SM.getCharacterData(SM.getSpellingLoc(Tok.getLocation()));
    }
  } else if (Tok.getLength() < 256) {
    char Buffer[256];
    const char *TokPtr = Buffer;
    PP.getSpelling(Tok, TokPtr);
    return TokPtr[0];
  } else {
    return PP.getSpelling(Tok)[0];
  }
}

/// AvoidConcat - If printing PrevTok immediately followed by Tok would cause
/// the two individual tokens to be lexed as a single token, return true
/// (which causes a space to be printed between them).  This allows the output
/// of -E mode to be lexed to the same token stream as lexing the input
/// directly would.
///
/// This code must conservatively return true if it doesn't want to be 100%
/// accurate.  This will cause the output to include extra space characters,
/// but the resulting output won't have incorrect concatenations going on.
/// Examples include "..", which we print with a space between, because we
/// don't want to track enough to tell "x.." from "...".
bool TokenConcatenation::AvoidConcat(const Token &PrevTok,
                                     const Token &Tok) const {
  // First, check to see if the tokens were directly adjacent in the original
  // source.  If they were, it must be okay to stick them together: if there
  // were an issue, the tokens would have been lexed differently.
  if (PrevTok.getLocation().isFileID() && Tok.getLocation().isFileID() &&
      PrevTok.getLocation().getFileLocWithOffset(PrevTok.getLength()) == 
        Tok.getLocation())
    return false;
  
  tok::TokenKind PrevKind = PrevTok.getKind();
  if (PrevTok.getIdentifierInfo())  // Language keyword or named operator.
    PrevKind = tok::identifier;
  
  // Look up information on when we should avoid concatenation with prevtok.
  unsigned ConcatInfo = TokenInfo[PrevKind];
  
  // If prevtok never causes a problem for anything after it, return quickly.
  if (ConcatInfo == 0) return false;
  
  if (ConcatInfo & aci_avoid_equal) {
    // If the next token is '=' or '==', avoid concatenation.
    if (Tok.is(tok::equal) || Tok.is(tok::equalequal))
      return true;
    ConcatInfo &= ~aci_avoid_equal;
  }
  
  if (ConcatInfo == 0) return false;
  
  // Basic algorithm: we look at the first character of the second token, and
  // determine whether it, if appended to the first token, would form (or
  // would contribute) to a larger token if concatenated.
  char FirstChar = 0;
  if (ConcatInfo & aci_custom) {
    // If the token does not need to know the first character, don't get it.
  } else {
    FirstChar = GetFirstChar(PP, Tok);
  }
    
  switch (PrevKind) {
  default: assert(0 && "InitAvoidConcatTokenInfo built wrong");
  case tok::identifier:   // id+id or id+number or id+L"foo".    
    // id+'.'... will not append.
    if (Tok.is(tok::numeric_constant))
      return GetFirstChar(PP, Tok) != '.';

    if (Tok.getIdentifierInfo() || Tok.is(tok::wide_string_literal) /* ||
     Tok.is(tok::wide_char_literal)*/)
      return true;
    
    // If this isn't identifier + string, we're done.
    if (Tok.isNot(tok::char_constant) && Tok.isNot(tok::string_literal))
      return false;
    
    // FIXME: need a wide_char_constant!
    
    // If the string was a wide string L"foo" or wide char L'f', it would
    // concat with the previous identifier into fooL"bar".  Avoid this.
    if (StartsWithL(Tok))
      return true;
    
    // Otherwise, this is a narrow character or string.  If the *identifier*
    // is a literal 'L', avoid pasting L "foo" -> L"foo".
    return IsIdentifierL(PrevTok);
  case tok::numeric_constant:
    return isalnum(FirstChar) || Tok.is(tok::numeric_constant) ||
    FirstChar == '+' || FirstChar == '-' || FirstChar == '.';
  case tok::period:          // ..., .*, .1234
    return FirstChar == '.' || isdigit(FirstChar) ||
    (FirstChar == '*' && PP.getLangOptions().CPlusPlus);
  case tok::amp:             // &&
    return FirstChar == '&';
  case tok::plus:            // ++
    return FirstChar == '+';
  case tok::minus:           // --, ->, ->*
    return FirstChar == '-' || FirstChar == '>';
  case tok::slash:           //, /*, //
    return FirstChar == '*' || FirstChar == '/';
  case tok::less:            // <<, <<=, <:, <%
    return FirstChar == '<' || FirstChar == ':' || FirstChar == '%';
  case tok::greater:         // >>, >>=
    return FirstChar == '>';
  case tok::pipe:            // ||
    return FirstChar == '|';
  case tok::percent:         // %>, %:
    return (FirstChar == '>' || FirstChar == ':') &&
    PP.getLangOptions().Digraphs;
  case tok::colon:           // ::, :>
    return (FirstChar == ':' && PP.getLangOptions().CPlusPlus) ||
    (FirstChar == '>' && PP.getLangOptions().Digraphs);
  case tok::hash:            // ##, #@, %:%:
    return FirstChar == '#' || FirstChar == '@' || FirstChar == '%';
  case tok::arrow:           // ->*
    return FirstChar == '*';
  }
}
