//===--- Parse.cpp - C Language Family Parser -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the Parser interfaces.
//
//===----------------------------------------------------------------------===//

#include "clang/Parse/Parser.h"
#include "clang/Parse/Declarations.h"
#include "clang/Parse/Scope.h"
using namespace llvm;
using namespace clang;

Parser::Parser(Preprocessor &pp, ParserActions &actions)
  : PP(pp), Actions(actions), Diags(PP.getDiagnostics()) {
  // Create the global scope, install it as the current scope.
  CurScope = new Scope(0);
  Tok.SetKind(tok::eof);
  
  ParenCount = BracketCount = BraceCount = 0;
}

Parser::~Parser() {
  delete CurScope;
}


void Parser::Diag(SourceLocation Loc, unsigned DiagID,
                  const std::string &Msg) {
  Diags.Report(Loc, DiagID, Msg);
}

//===----------------------------------------------------------------------===//
// Error recovery.
//===----------------------------------------------------------------------===//

/// SkipUntil - Read tokens until we get to the specified token, then consume
/// it (unless DontConsume is false).  Because we cannot guarantee that the
/// token will ever occur, this skips to the next token, or to some likely
/// good stopping point.  If StopAtSemi is true, skipping will stop at a ';'
/// character.
/// 
/// If SkipUntil finds the specified token, it returns true, otherwise it
/// returns false.  
bool Parser::SkipUntil(tok::TokenKind T, bool StopAtSemi, bool DontConsume) {
  while (1) {
    // If we found the token, stop and return true.
    if (Tok.getKind() == T) {
      if (DontConsume) {
        // Noop, don't consume the token.
      } else if (isTokenParen()) {
        ConsumeParen();
      } else if (isTokenBracket()) {
        ConsumeBracket();
      } else if (isTokenBrace()) {
        ConsumeBrace();
      } else if (T == tok::string_literal) {
        ConsumeStringToken();
      } else {
        ConsumeToken();
      }
      return true;
    }
    
    switch (Tok.getKind()) {
    case tok::eof:
      // Ran out of tokens.
      return false;
      
    case tok::l_paren:
      // Recursively skip properly-nested parens.
      ConsumeParen();
      SkipUntil(tok::r_paren);
      break;
    case tok::l_square:
      // Recursively skip properly-nested square brackets.
      ConsumeBracket();
      SkipUntil(tok::r_square);
      break;
    case tok::l_brace:
      // Recursively skip properly-nested braces.
      ConsumeBrace();
      SkipUntil(tok::r_brace);
      break;
      
    // Okay, we found a ']' or '}' or ')', which we think should be balanced.
    // Since the user wasn't looking for this token (if they were, it would
    // already be handled), this isn't balanced.  If there is a LHS token at a
    // higher level, we will assume that this matches the unbalanced token
    // and return it.  Otherwise, this is a spurious RHS token, which we skip.
    case tok::r_paren:
      if (ParenCount) return false;  // Matches something.
      ConsumeParen();
      break;
    case tok::r_square:
      if (BracketCount) return false;  // Matches something.
      ConsumeBracket();
      break;
    case tok::r_brace:
      if (BraceCount) return false;  // Matches something.
      ConsumeBrace();
      break;
      
    case tok::string_literal:
      ConsumeStringToken();
      break;
    case tok::semi:
      if (StopAtSemi)
        return false;
      // FALL THROUGH.
    default:
      // Skip this token.
      ConsumeToken();
      break;
    }
  }  
}

//===----------------------------------------------------------------------===//
// C99 6.9: External Definitions.
//===----------------------------------------------------------------------===//

/// ParseTranslationUnit:
///       translation-unit: [C99 6.9]
///         external-declaration 
///         translation-unit external-declaration 
void Parser::ParseTranslationUnit() {

  if (Tok.getKind() == tok::eof)  // Empty source file is an extension.
    Diag(diag::ext_empty_source_file);
  
  while (Tok.getKind() != tok::eof)
    ParseExternalDeclaration();
}

/// ParseExternalDeclaration:
///       external-declaration: [C99 6.9]
///         function-definition        [TODO]
///         declaration                [TODO]
/// [EXT]   ';'
/// [GNU]   asm-definition             [TODO]
/// [GNU]   __extension__ external-declaration     [TODO]
/// [OBJC]  objc-class-definition      [TODO]
/// [OBJC]  objc-class-declaration     [TODO]
/// [OBJC]  objc-alias-declaration     [TODO]
/// [OBJC]  objc-protocol-definition   [TODO]
/// [OBJC]  objc-method-definition     [TODO]
/// [OBJC]  @end                       [TODO]
///
void Parser::ParseExternalDeclaration() {
  switch (Tok.getKind()) {
  case tok::semi:
    Diag(diag::ext_top_level_semi);
    ConsumeToken();
    break;
  default:
    // We can't tell whether this is a function-definition or declaration yet.
    ParseDeclarationOrFunctionDefinition();
    break;
  }
}

/// ParseDeclarationOrFunctionDefinition - Parse either a function-definition or
/// a declaration.  We can't tell which we have until we read up to the
/// compound-statement in function-definition.
///
///       function-definition: [C99 6.9.1]
///         declaration-specifiers[opt] declarator declaration-list[opt] 
///                 compound-statement                           [TODO]
///       declaration: [C99 6.7]
///         declaration-specifiers init-declarator-list[opt] ';' [TODO]
/// [!C99]  init-declarator-list ';'                             [TODO]
/// [OMP]   threadprivate-directive                              [TODO]
///
///       init-declarator-list: [C99 6.7]
///         init-declarator
///         init-declarator-list ',' init-declarator
///       init-declarator: [C99 6.7]
///         declarator
///         declarator '=' initializer
///
void Parser::ParseDeclarationOrFunctionDefinition() {
  // Parse the common declaration-specifiers piece.
  // NOTE: this can not be missing for C99 'declaration's.
  DeclSpec DS;
  ParseDeclarationSpecifiers(DS);

  // C99 6.7.2.3p6: Handle "struct-or-union identifier;", "enum { X };"
  if (Tok.getKind() == tok::semi)
    assert(0 && "Unimp!");
  
  
  // Parse the declarator.
  {
    Declarator DeclaratorInfo(DS, Declarator::FileContext);
    ParseDeclarator(DeclaratorInfo);

    // If the declarator was a function type... handle it.

    // must be: decl-spec[opt] declarator init-declarator-list
    // Parse declarator '=' initializer.
    if (Tok.getKind() == tok::equal)
      assert(0 && "cannot handle initializer yet!");
  }

  while (Tok.getKind() == tok::comma) {
    // Consume the comma.
    ConsumeToken();
    
    // Parse the declarator.
    Declarator DeclaratorInfo(DS, Declarator::FileContext);
    ParseDeclarator(DeclaratorInfo);
    
    // declarator '=' initializer
    if (Tok.getKind() == tok::equal)
      assert(0 && "cannot handle initializer yet!");
    
    
  }
  
  if (Tok.getKind() == tok::semi) {
    ConsumeToken();
  } else {
    Diag(Tok, diag::err_parse_error);
    // Skip to end of block or statement
    SkipUntil(tok::r_brace, true);
    if (Tok.getKind() == tok::semi)
      ConsumeToken();
  }
}

