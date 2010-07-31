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
  PP.Lex(Tok);
  if (Tok.isNot(tok::eom)) {
    PP.Diag(Tok.getLocation(), diag::warn_pragma_extra_tokens_at_eol) << "pack";
    return;
  }

  Actions.ActOnPragmaPack(Kind, Name, Alignment.release(), PackLoc,
                          LParenLoc, RParenLoc);
}

// #pragma 'align' '=' {'native','natural','mac68k','power','reset'}
// #pragma 'options 'align' '=' {'native','natural','mac68k','power','reset'}
static void ParseAlignPragma(Action &Actions, Preprocessor &PP, Token &FirstTok,
                             bool IsOptions) {
  Token Tok;

  if (IsOptions) {
    PP.Lex(Tok);
    if (Tok.isNot(tok::identifier) ||
        !Tok.getIdentifierInfo()->isStr("align")) {
      PP.Diag(Tok.getLocation(), diag::warn_pragma_options_expected_align);
      return;
    }
  }

  PP.Lex(Tok);
  if (Tok.isNot(tok::equal)) {
    PP.Diag(Tok.getLocation(), diag::warn_pragma_align_expected_equal)
      << IsOptions;
    return;
  }

  PP.Lex(Tok);
  if (Tok.isNot(tok::identifier)) {
    PP.Diag(Tok.getLocation(), diag::warn_pragma_expected_identifier)
      << (IsOptions ? "options" : "align");
    return;
  }

  Action::PragmaOptionsAlignKind Kind = Action::POAK_Natural;
  const IdentifierInfo *II = Tok.getIdentifierInfo();
  if (II->isStr("native"))
    Kind = Action::POAK_Native;
  else if (II->isStr("natural"))
    Kind = Action::POAK_Natural;
  else if (II->isStr("packed"))
    Kind = Action::POAK_Packed;
  else if (II->isStr("power"))
    Kind = Action::POAK_Power;
  else if (II->isStr("mac68k"))
    Kind = Action::POAK_Mac68k;
  else if (II->isStr("reset"))
    Kind = Action::POAK_Reset;
  else {
    PP.Diag(Tok.getLocation(), diag::warn_pragma_align_invalid_option)
      << IsOptions;
    return;
  }

  SourceLocation KindLoc = Tok.getLocation();
  PP.Lex(Tok);
  if (Tok.isNot(tok::eom)) {
    PP.Diag(Tok.getLocation(), diag::warn_pragma_extra_tokens_at_eol)
      << (IsOptions ? "options" : "align");
    return;
  }

  Actions.ActOnPragmaOptionsAlign(Kind, FirstTok.getLocation(), KindLoc);
}

void PragmaAlignHandler::HandlePragma(Preprocessor &PP, Token &AlignTok) {
  ParseAlignPragma(Actions, PP, AlignTok, /*IsOptions=*/false);
}

void PragmaOptionsHandler::HandlePragma(Preprocessor &PP, Token &OptionsTok) {
  ParseAlignPragma(Actions, PP, OptionsTok, /*IsOptions=*/true);
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
  llvm::SmallVector<Token, 5> Identifiers;
  SourceLocation RParenLoc;
  bool LexID = true;

  while (true) {
    PP.Lex(Tok);

    if (LexID) {
      if (Tok.is(tok::identifier)) {
        Identifiers.push_back(Tok);
        LexID = false;
        continue;
      }

      // Illegal token!
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

    // Illegal token!
    PP.Diag(Tok.getLocation(), diag::warn_pragma_unused_expected_punc);
    return;
  }

  PP.Lex(Tok);
  if (Tok.isNot(tok::eom)) {
    PP.Diag(Tok.getLocation(), diag::warn_pragma_extra_tokens_at_eol) <<
        "unused";
    return;
  }

  // Verify that we have a location for the right parenthesis.
  assert(RParenLoc.isValid() && "Valid '#pragma unused' must have ')'");
  assert(!Identifiers.empty() && "Valid '#pragma unused' must have arguments");

  // Perform the action to handle the pragma.
  Actions.ActOnPragmaUnused(Identifiers.data(), Identifiers.size(),
                            parser.getCurScope(), UnusedLoc, LParenLoc, RParenLoc);
}

// #pragma weak identifier
// #pragma weak identifier '=' identifier
void PragmaWeakHandler::HandlePragma(Preprocessor &PP, Token &WeakTok) {
  // FIXME: Should we be expanding macros here? My guess is no.
  SourceLocation WeakLoc = WeakTok.getLocation();

  Token Tok;
  PP.Lex(Tok);
  if (Tok.isNot(tok::identifier)) {
    PP.Diag(Tok.getLocation(), diag::warn_pragma_expected_identifier) << "weak";
    return;
  }

  IdentifierInfo *WeakName = Tok.getIdentifierInfo(), *AliasName = 0;
  SourceLocation WeakNameLoc = Tok.getLocation(), AliasNameLoc;

  PP.Lex(Tok);
  if (Tok.is(tok::equal)) {
    PP.Lex(Tok);
    if (Tok.isNot(tok::identifier)) {
      PP.Diag(Tok.getLocation(), diag::warn_pragma_expected_identifier)
          << "weak";
      return;
    }
    AliasName = Tok.getIdentifierInfo();
    AliasNameLoc = Tok.getLocation();
    PP.Lex(Tok);
  }

  if (Tok.isNot(tok::eom)) {
    PP.Diag(Tok.getLocation(), diag::warn_pragma_extra_tokens_at_eol) << "weak";
    return;
  }

  if (AliasName) {
    Actions.ActOnPragmaWeakAlias(WeakName, AliasName, WeakLoc, WeakNameLoc,
                                 AliasNameLoc);
  } else {
    Actions.ActOnPragmaWeakID(WeakName, WeakLoc, WeakNameLoc);
  }
}
