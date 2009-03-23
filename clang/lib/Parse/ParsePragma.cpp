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
#include "clang/Parse/ParseDiagnostic.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Parse/Action.h"
#include "clang/Parse/Parser.h"
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
    PP.Diag(Tok.getLocation(), diag::warn_pragma_expected_lparen) << "pack";
    return;
  }

  Action::PragmaPackKind Kind = Action::PPK_Default;
  IdentifierInfo *Name = 0;
  Action::OwningExprResult Alignment(Actions);
  SourceLocation LParenLoc = Tok.getLocation();
  PP.Lex(Tok);  
  if (Tok.is(tok::numeric_constant)) {
    Alignment = Actions.ActOnNumericConstant(Tok);
    if (Alignment.isInvalid())
      return;

    PP.Lex(Tok);
  } else if (Tok.is(tok::identifier)) {
    const IdentifierInfo *II = Tok.getIdentifierInfo();
    if (II->isStr("show")) {
      Kind = Action::PPK_Show;
      PP.Lex(Tok);
    } else {
      if (II->isStr("push")) {
        Kind = Action::PPK_Push;
      } else if (II->isStr("pop")) {
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
          if (Alignment.isInvalid())
            return;

          PP.Lex(Tok);
        } else if (Tok.is(tok::identifier)) {
          Name = Tok.getIdentifierInfo();
          PP.Lex(Tok);
          
          if (Tok.is(tok::comma)) {
            PP.Lex(Tok);
            
            if (Tok.isNot(tok::numeric_constant)) {
              PP.Diag(Tok.getLocation(), diag::warn_pragma_pack_malformed);
              return;
            }
            
            Alignment = Actions.ActOnNumericConstant(Tok);
            if (Alignment.isInvalid())
              return;

            PP.Lex(Tok);
          }
        } else {
          PP.Diag(Tok.getLocation(), diag::warn_pragma_pack_malformed);
          return;
        }
      }
    }
  }

  if (Tok.isNot(tok::r_paren)) {
    PP.Diag(Tok.getLocation(), diag::warn_pragma_expected_rparen) << "pack";
    return;
  }

  SourceLocation RParenLoc = Tok.getLocation();
  Actions.ActOnPragmaPack(Kind, Name, Alignment.release(), PackLoc,
                          LParenLoc, RParenLoc);
}

// #pragma unused(identifier)
void PragmaUnusedHandler::HandlePragma(Preprocessor &PP, Token &UnusedTok) {
  // FIXME: Should we be expanding macros here? My guess is no.
  SourceLocation UnusedLoc = UnusedTok.getLocation();
  
  // Lex the left '('.
  Token Tok;
  PP.Lex(Tok);
  if (Tok.isNot(tok::l_paren)) {
    PP.Diag(Tok.getLocation(), diag::warn_pragma_expected_lparen) << "unused";
    return;
  }
  SourceLocation LParenLoc = Tok.getLocation();
  
  // Lex the declaration reference(s).
  llvm::SmallVector<Action::ExprTy*, 5> Ex;
  SourceLocation RParenLoc;
  bool LexID = true;
  
  while (true) {
    PP.Lex(Tok);
    
    if (LexID) {
      if (Tok.is(tok::identifier)) {            
        Action::OwningExprResult Name = 
          Actions.ActOnIdentifierExpr(parser.CurScope, Tok.getLocation(),
                                      *Tok.getIdentifierInfo(), false);
      
        if (Name.isInvalid()) {
          if (!Ex.empty())
            Action::MultiExprArg Release(Actions, &Ex[0], Ex.size());
          return;
        }
        
        Ex.push_back(Name.release());        
        LexID = false;
        continue;
      }

      // Illegal token! Release the parsed expressions (if any) and emit
      // a warning.
      if (!Ex.empty())
        Action::MultiExprArg Release(Actions, &Ex[0], Ex.size());
      
      PP.Diag(Tok.getLocation(), diag::warn_pragma_unused_expected_var);
      return;
    }
    
    // We are execting a ')' or a ','.
    if (Tok.is(tok::comma)) {
      LexID = true;
      continue;
    }
    
    if (Tok.is(tok::r_paren)) {
      RParenLoc = Tok.getLocation();
      break;
    }
    
    // Illegal token! Release the parsed expressions (if any) and emit
    // a warning.
    if (!Ex.empty())
      Action::MultiExprArg Release(Actions, &Ex[0], Ex.size());
    
    PP.Diag(Tok.getLocation(), diag::warn_pragma_unused_expected_punc);
    return;
  }
  
  // Verify that we have a location for the right parenthesis.
  assert(RParenLoc.isValid() && "Valid '#pragma unused' must have ')'");
  assert(!Ex.empty() && "Valid '#pragma unused' must have arguments");

  // Perform the action to handle the pragma.    
  Actions.ActOnPragmaUnused(&Ex[0], Ex.size(), UnusedLoc, LParenLoc, RParenLoc);
}
