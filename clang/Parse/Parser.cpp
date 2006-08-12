//===--- Parser.cpp - C Language Family Parser ----------------------------===//
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

/// MatchRHSPunctuation - For punctuation with a LHS and RHS (e.g. '['/']'),
/// this helper function matches and consumes the specified RHS token if
/// present.  If not present, it emits the specified diagnostic indicating
/// that the parser failed to match the RHS of the token at LHSLoc.  LHSName
/// should be the name of the unmatched LHS token.
void Parser::MatchRHSPunctuation(tok::TokenKind RHSTok, SourceLocation LHSLoc,
                                 const char *LHSName, unsigned DiagID) {
  
  if (Tok.getKind() == RHSTok) {
    ConsumeAnyToken();
  } else {
    Diag(Tok, DiagID);
    Diag(LHSLoc, diag::err_matching, LHSName);
    SkipUntil(RHSTok);
  }
}

/// ExpectAndConsume - The parser expects that 'ExpectedTok' is next in the
/// input.  If so, it is consumed and false is returned.
///
/// If the input is malformed, this emits the specified diagnostic.  Next, if
/// SkipToTok is specified, it calls SkipUntil(SkipToTok).  Finally, true is
/// returned.
bool Parser::ExpectAndConsume(tok::TokenKind ExpectedTok, unsigned DiagID,
                              tok::TokenKind SkipToTok) {
  if (Tok.getKind() == ExpectedTok) {
    ConsumeToken();
    return false;
  }
  
  Diag(Tok, DiagID);
  if (SkipToTok != tok::unknown)
    SkipUntil(SkipToTok);
  return true;
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
  // We always want this function to skip at least one token if the first token
  // isn't T and if not at EOF.
  bool isFirstTokenSkipped = true;
  while (1) {
    // If we found the token, stop and return true.
    if (Tok.getKind() == T) {
      if (DontConsume) {
        // Noop, don't consume the token.
      } else {
        ConsumeAnyToken();
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
      SkipUntil(tok::r_paren, false);
      break;
    case tok::l_square:
      // Recursively skip properly-nested square brackets.
      ConsumeBracket();
      SkipUntil(tok::r_square, false);
      break;
    case tok::l_brace:
      // Recursively skip properly-nested braces.
      ConsumeBrace();
      SkipUntil(tok::r_brace, false);
      break;
      
    // Okay, we found a ']' or '}' or ')', which we think should be balanced.
    // Since the user wasn't looking for this token (if they were, it would
    // already be handled), this isn't balanced.  If there is a LHS token at a
    // higher level, we will assume that this matches the unbalanced token
    // and return it.  Otherwise, this is a spurious RHS token, which we skip.
    case tok::r_paren:
      if (ParenCount && !isFirstTokenSkipped)
        return false;  // Matches something.
      ConsumeParen();
      break;
    case tok::r_square:
      if (BracketCount && !isFirstTokenSkipped)
        return false;  // Matches something.
      ConsumeBracket();
      break;
    case tok::r_brace:
      if (BraceCount && !isFirstTokenSkipped)
        return false;  // Matches something.
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
    isFirstTokenSkipped = false;
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
  DeclSpec DS;
  ParseDeclarationSpecifiers(DS);

  // C99 6.7.2.3p6: Handle "struct-or-union identifier;", "enum { X };"
  // declaration-specifiers init-declarator-list[opt] ';'
  if (Tok.getKind() == tok::semi)
    assert(0 && "Unimp!");   // FIXME: implement 'struct foo;'.
  
  
  // Parse the first declarator.
  Declarator DeclaratorInfo(DS, Declarator::FileContext);
  ParseDeclarator(DeclaratorInfo);
  // Error parsing the declarator?
  if (DeclaratorInfo.getIdentifier() == 0) {
    // If so, skip until the semi-colon or a }.
    SkipUntil(tok::r_brace, true);
    if (Tok.getKind() == tok::semi)
      ConsumeToken();
    return;
  }

  // If the declarator is the start of a function definition, handle it.
  if (Tok.getKind() == tok::equal ||  // int X()=  -> not a function def
      Tok.getKind() == tok::comma ||  // int X(),  -> not a function def
      Tok.getKind() == tok::semi ||   // int X();  -> not a function def
      Tok.getKind() == tok::kw_asm || // int X() __asm__ -> not a fn def
      Tok.getKind() == tok::kw___attribute) {// int X() __attr__ -> not a fn def
    // FALL THROUGH.
  } else if (DeclaratorInfo.isInnermostFunctionType() &&
             (Tok.getKind() == tok::l_brace ||  // int X() {}
              isDeclarationSpecifier())) {      // int X(f) int f; {}
    ParseFunctionDefinition(DeclaratorInfo);
    return;
  } else {
    if (DeclaratorInfo.isInnermostFunctionType())
      Diag(Tok, diag::err_expected_fn_body);
    else
      Diag(Tok, diag::err_expected_after_declarator);
    SkipUntil(tok::r_brace, true);
    if (Tok.getKind() == tok::semi)
      ConsumeToken();
    return;
  }

  // Parse the init-declarator-list for a normal declaration.
  ParseInitDeclaratorListAfterFirstDeclarator(DeclaratorInfo);
}

/// ParseFunctionDefinition - We parsed and verified that the specified
/// Declarator is well formed.  If this is a K&R-style function, read the
/// parameters declaration-list, then start the compound-statement.
///
///         declaration-specifiers[opt] declarator declaration-list[opt] 
///                 compound-statement                           [TODO]
///
void Parser::ParseFunctionDefinition(Declarator &D) {
  const DeclaratorTypeInfo &FnTypeInfo = D.getTypeObject(0);
  assert(FnTypeInfo.Kind == DeclaratorTypeInfo::Function &&
         "This isn't a function declarator!");
  
  // If this declaration was formed with a K&R-style identifier list for the
  // arguments, parse declarations for all of the args next.
  // int foo(a,b) int a; float b; {}
  if (!FnTypeInfo.Fun.hasPrototype && !FnTypeInfo.Fun.isEmpty) {
    // Read all the argument declarations.
    while (isDeclarationSpecifier())
      ParseDeclaration(Declarator::KNRTypeListContext);
    
    // Note, check that we got them all.
  } else {
    //if (isDeclarationSpecifier())
    //  Diag('k&r declspecs with prototype?');
    
    // FIXME: Install the arguments into the current scope.
  }

  // We should have an opening brace now.
  if (Tok.getKind() != tok::l_brace) {
    Diag(Tok, diag::err_expected_fn_body);

    // Skip over garbage, until we get to '{'.  Don't eat the '{'.
    SkipUntil(tok::l_brace, true, true);
    
    // If we didn't find the '{', bail out.
    if (Tok.getKind() != tok::l_brace)
      return;
  }
  
  ParseCompoundStatement();
}

