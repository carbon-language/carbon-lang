//===--- Initializer.cpp - Initializer Parsing ----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements initializer parsing as specified by C99 6.7.8.
//
//===----------------------------------------------------------------------===//

#include "clang/Parse/Parser.h"
#include "clang/Basic/Diagnostic.h"
using namespace llvm;
using namespace clang;


/// MayBeDesignationStart - Return true if this token might be the start of a
/// designator.
static bool MayBeDesignationStart(tok::TokenKind K) {
  switch (K) {
  default: return false;
  case tok::period:      // designator: '.' identifier
  case tok::l_square:    // designator: array-designator
  case tok::identifier:  // designation: identifier ':'
    return true;
  }
}

/// ParseInitializerWithPotentialDesignator - Parse the 'initializer' production
/// checking to see if the token stream starts with a designator.
///
///       designation:
///         designator-list '='
/// [GNU]   array-designator
/// [GNU]   identifier ':'
///
///       designator-list:
///         designator
///         designator-list designator
///
///       designator:
///         array-designator
///         '.' identifier
///
///       array-designator:
///         '[' constant-expression ']'
/// [GNU]   '[' constant-expression '...' constant-expression ']'
///
/// NOTE: [OBC] allows '[ objc-receiver objc-message-args ]' as an
/// initializer.  We need to consider this case when parsing array designators.
///
Parser::ExprResult Parser::ParseInitializerWithPotentialDesignator() {
  // Parse each designator in the designator list until we find an initializer.
  while (1) {
    switch (Tok.getKind()) {
    case tok::equal:
      // We read some number (at least one due to the grammar we implemented)
      // of designators and found an '=' sign.  The following tokens must be
      // the initializer.
      ConsumeToken();
      return ParseInitializer();
      
    default: {
      // We read some number (at least one due to the grammar we implemented)
      // of designators and found something that isn't an = or an initializer.
      // If we have exactly one array designator [TODO CHECK], this is the GNU
      // 'designation: array-designator' extension.  Otherwise, it is a parse
      // error.
      SourceLocation Loc = Tok.getLocation();
      ExprResult Init = ParseInitializer();
      if (Init.isInvalid) return Init;
      
      Diag(Tok, diag::ext_gnu_missing_equal_designator);
      return Init;
    }
    case tok::period:
      // designator: '.' identifier
      ConsumeToken();
      if (ExpectAndConsume(tok::identifier, diag::err_expected_ident))
        return ExprResult(true);
      break;
                         
    case tok::l_square: {
      // array-designator: '[' constant-expression ']'
      // array-designator: '[' constant-expression '...' constant-expression ']'
      SourceLocation StartLoc = ConsumeBracket();
      
      ExprResult Idx = ParseConstantExpression();
      if (Idx.isInvalid) {
        SkipUntil(tok::r_square);
        return Idx;
      }
      
      // Handle the gnu array range extension.
      if (Tok.getKind() == tok::ellipsis) {
        Diag(Tok, diag::ext_gnu_array_range);
        ConsumeToken();
        
        ExprResult RHS = ParseConstantExpression();
        if (RHS.isInvalid) {
          SkipUntil(tok::r_square);
          return RHS;
        }
      }
      
      MatchRHSPunctuation(tok::r_square, StartLoc);
      break;
    }
    case tok::identifier: {
      // Due to the GNU "designation: identifier ':'" extension, we don't know
      // whether something starting with an identifier is an
      // assignment-expression or if it is an old-style structure field
      // designator.
      // TODO: Check that this is the first designator.
      LexerToken Ident = Tok;
      ConsumeToken();
      
      // If this is the gross GNU extension, handle it now.
      if (Tok.getKind() == tok::colon) {
        Diag(Ident, diag::ext_gnu_old_style_field_designator);
        ConsumeToken();
        return ParseInitializer();
      }
      
      // Otherwise, we just consumed the first token of an expression.  Parse
      // the rest of it now.
      return ParseAssignmentExprWithLeadingIdentifier(Ident);
    }
    }
  }
}


/// ParseInitializer
///       initializer: [C99 6.7.8]
///         assignment-expression
///         '{' initializer-list '}'
///         '{' initializer-list ',' '}'
/// [GNU]   '{' '}'
///
///       initializer-list:
///         designation[opt] initializer
///         initializer-list ',' designation[opt] initializer
///
Parser::ExprResult Parser::ParseInitializer() {
  if (Tok.getKind() != tok::l_brace)
    return ParseAssignmentExpression();

  SourceLocation LBraceLoc = ConsumeBrace();
  
  // We support empty initializers, but tell the user that they aren't using
  // C99-clean code.
  if (Tok.getKind() == tok::r_brace)
    Diag(LBraceLoc, diag::ext_gnu_empty_initializer);
  else {
    while (1) {
      // Parse: designation[opt] initializer
      
      // If we know that this cannot be a designation, just parse the nested
      // initializer directly.
      ExprResult SubElt;
      if (!MayBeDesignationStart(Tok.getKind()))
        SubElt = ParseInitializer();
      else
        SubElt = ParseInitializerWithPotentialDesignator();
      
      // If we couldn't parse the subelement, bail out.
      if (SubElt.isInvalid) {
        SkipUntil(tok::r_brace);
        return SubElt;
      }
    
      // If we don't have a comma continued list, we're done.
      if (Tok.getKind() != tok::comma) break;
      ConsumeToken();
      
      // Handle trailing comma.
      if (Tok.getKind() == tok::r_brace) break;
    }    
  }
  
  // Match the '}'.
  MatchRHSPunctuation(tok::r_brace, LBraceLoc);
  return ExprResult(false);
}

