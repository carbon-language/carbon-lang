//===--- ParseCXXInlineMethods.cpp - C++ class inline methods parsing------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements parsing for C++ class inline methods.
//
//===----------------------------------------------------------------------===//

#include "clang/Parse/Parser.h"
#include "clang/Parse/DeclSpec.h"
#include "clang/Parse/Scope.h"
using namespace clang;

/// ParseInlineCXXMethodDef - We parsed and verified that the specified
/// Declarator is a well formed C++ inline method definition. Now lex its body
/// and store its tokens for parsing after the C++ class is complete.
Parser::DeclTy *
Parser::ParseCXXInlineMethodDef(AccessSpecifier AS, Declarator &D) {
  assert(D.getTypeObject(0).Kind == DeclaratorChunk::Function &&
         "This isn't a function declarator!");
  assert(Tok.is(tok::l_brace) && "Current token not a '{'!");

  DeclTy *FnD = Actions.ActOnCXXMemberDeclarator(CurScope, AS, D, 0, 0, 0);

  // Consume the tokens and store them for later parsing.

  getCurTopClassStack().push(LexedMethod(FnD));
  TokensTy &Toks = getCurTopClassStack().top().Toks;

  // Begin by storing the '{' token.
  Toks.push_back(Tok);
  ConsumeBrace();
  ConsumeAndStoreUntil(tok::r_brace, Toks);

  return FnD;
}

/// ParseLexedMethodDefs - We finished parsing the member specification of a top
/// (non-nested) C++ class. Now go over the stack of lexed methods that were
/// collected during its parsing and parse them all.
void Parser::ParseLexedMethodDefs() {
  while (!getCurTopClassStack().empty()) {
    LexedMethod &LM = getCurTopClassStack().top();

    assert(!LM.Toks.empty() && "Empty body!");
    // Append the current token at the end of the new token stream so that it
    // doesn't get lost.
    LM.Toks.push_back(Tok);
    PP.EnterTokenStream(&LM.Toks.front(), LM.Toks.size(), true, false);

    // Consume the previously pushed token.
    ConsumeAnyToken();
    assert(Tok.is(tok::l_brace) && "Inline method not starting with '{'");

    // Parse the method body. Function body parsing code is similar enough
    // to be re-used for method bodies as well.
    EnterScope(Scope::FnScope|Scope::DeclScope);
    Actions.ActOnStartOfFunctionDef(CurScope, LM.D);

    ParseFunctionStatementBody(LM.D, Tok.getLocation(), Tok.getLocation());

    getCurTopClassStack().pop();
  }
}

/// ConsumeAndStoreUntil - Consume and store the token at the passed token
/// container until the token 'T' is reached (which gets consumed/stored too).
/// Returns true if token 'T' was found.
/// NOTE: This is a specialized version of Parser::SkipUntil.
bool Parser::ConsumeAndStoreUntil(tok::TokenKind T, TokensTy &Toks) {
  // We always want this function to consume at least one token if the first
  // token isn't T and if not at EOF.
  bool isFirstTokenConsumed = true;
  while (1) {
    // If we found one of the tokens, stop and return true.
    if (Tok.is(T)) {
      Toks.push_back(Tok);
      ConsumeAnyToken();
      return true;
    }

    switch (Tok.getKind()) {
    case tok::eof:
      // Ran out of tokens.
      return false;

    case tok::l_paren:
      // Recursively consume properly-nested parens.
      Toks.push_back(Tok);
      ConsumeParen();
      ConsumeAndStoreUntil(tok::r_paren, Toks);
      break;
    case tok::l_square:
      // Recursively consume properly-nested square brackets.
      Toks.push_back(Tok);
      ConsumeBracket();
      ConsumeAndStoreUntil(tok::r_square, Toks);
      break;
    case tok::l_brace:
      // Recursively consume properly-nested braces.
      Toks.push_back(Tok);
      ConsumeBrace();
      ConsumeAndStoreUntil(tok::r_brace, Toks);
      break;

    // Okay, we found a ']' or '}' or ')', which we think should be balanced.
    // Since the user wasn't looking for this token (if they were, it would
    // already be handled), this isn't balanced.  If there is a LHS token at a
    // higher level, we will assume that this matches the unbalanced token
    // and return it.  Otherwise, this is a spurious RHS token, which we skip.
    case tok::r_paren:
      if (ParenCount && !isFirstTokenConsumed)
        return false;  // Matches something.
      Toks.push_back(Tok);
      ConsumeParen();
      break;
    case tok::r_square:
      if (BracketCount && !isFirstTokenConsumed)
        return false;  // Matches something.
      Toks.push_back(Tok);
      ConsumeBracket();
      break;
    case tok::r_brace:
      if (BraceCount && !isFirstTokenConsumed)
        return false;  // Matches something.
      Toks.push_back(Tok);
      ConsumeBrace();
      break;

    case tok::string_literal:
    case tok::wide_string_literal:
      Toks.push_back(Tok);
      ConsumeStringToken();
      break;
    default:
      // consume this token.
      Toks.push_back(Tok);
      ConsumeToken();
      break;
    }
    isFirstTokenConsumed = false;
  }
}
