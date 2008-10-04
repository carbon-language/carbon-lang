//===--- ParsePragma.cpp - Language specific pragma parsing ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the language specific #pragma handlers.
//
//===----------------------------------------------------------------------===//

#include "ParsePragma.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Parse/Action.h"
using namespace clang;

// #pragma pack(...) comes in the following delicious flavors:
//   pack '(' [integer] ')'
//   pack '(' 'show' ')'
//   pack '(' ('push' | 'pop') [',' identifier] [, integer] ')'
void PragmaPackHandler::HandlePragma(Preprocessor &PP, Token &PackTok) {
  // FIXME: Should we be expanding macros here? My guess is no.
  SourceLocation PackLoc = PackTok.getLocation();

  Token Tok;
  PP.Lex(Tok);
  if (Tok.isNot(tok::l_paren)) {
    PP.Diag(Tok.getLocation(), diag::warn_pragma_pack_expected_lparen);
    return;
  }

  Action::PragmaPackKind Kind = Action::PPK_Default;
  IdentifierInfo *Name = 0;
  Action::ExprResult Alignment;
  SourceLocation LParenLoc = Tok.getLocation();
  PP.Lex(Tok);  
  if (Tok.is(tok::numeric_constant)) {
    Alignment = Actions.ActOnNumericConstant(Tok);
    if (Alignment.isInvalid)
      return;

    PP.Lex(Tok);
  } else if (Tok.is(tok::identifier)) {
    const IdentifierInfo *II = Tok.getIdentifierInfo();
    if (II == &PP.getIdentifierTable().get("show")) {
      Kind = Action::PPK_Show;
      PP.Lex(Tok);
    } else {
      if (II == &PP.getIdentifierTable().get("push")) {
        Kind = Action::PPK_Push;
      } else if (II == &PP.getIdentifierTable().get("pop")) {
        Kind = Action::PPK_Pop;
      } else {
        PP.Diag(Tok.getLocation(), diag::warn_pragma_pack_invalid_action);
        return;
      }        
      PP.Lex(Tok);
    
      if (Tok.is(tok::comma)) {
        PP.Lex(Tok);
        
        if (Tok.is(tok::numeric_constant)) {
          Alignment = Actions.ActOnNumericConstant(Tok);
          if (Alignment.isInvalid)
            return;

          PP.Lex(Tok);
        } else if (Tok.is(tok::identifier)) {
          Name = Tok.getIdentifierInfo();
          PP.Lex(Tok);
          
          if (Tok.is(tok::comma)) {
            PP.Lex(Tok);
            
            if (Tok.isNot(tok::numeric_constant)) {
              PP.Diag(Tok.getLocation(), diag::warn_pragma_pack_malformed, 
                      II->getName());
              return;
            }
            
            Alignment = Actions.ActOnNumericConstant(Tok);
            if (Alignment.isInvalid)
              return;

            PP.Lex(Tok);
          }
        } else {
          PP.Diag(Tok.getLocation(), diag::warn_pragma_pack_malformed, 
                  II->getName());
          return;
        }
      }
    }
  } 

  if (Tok.isNot(tok::r_paren)) {
    PP.Diag(Tok.getLocation(), diag::warn_pragma_pack_expected_rparen);
    return;
  }

  SourceLocation RParenLoc = Tok.getLocation();
  Actions.ActOnPragmaPack(Kind, Name, Alignment.Val, PackLoc, 
                          LParenLoc, RParenLoc);
}

