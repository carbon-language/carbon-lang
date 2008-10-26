//===--- ParseInit.cpp - Initializer Parsing ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements initializer parsing as specified by C99 6.7.8.
//
//===----------------------------------------------------------------------===//

#include "clang/Parse/Designator.h"
#include "clang/Parse/Parser.h"
#include "clang/Basic/Diagnostic.h"
#include "llvm/ADT/SmallString.h"
using namespace clang;


/// MayBeDesignationStart - Return true if this token might be the start of a
/// designator.  If we can tell it is impossible that it is a designator, return
/// false. 
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
/// initializer (because it is an expression).  We need to consider this case
/// when parsing array designators.
///
Parser::ExprResult Parser::
ParseInitializerWithPotentialDesignator(InitListDesignations &Designations,
                                        unsigned InitNum) {
  
  // If this is the old-style GNU extension:
  //   designation ::= identifier ':'
  // Handle it as a field designator.  Otherwise, this must be the start of a
  // normal expression.
  if (Tok.is(tok::identifier)) {
    if (NextToken().is(tok::colon)) {
      Diag(Tok, diag::ext_gnu_old_style_field_designator);

      Designation &D = Designations.CreateDesignation(InitNum);
      D.AddDesignator(Designator::getField(Tok.getIdentifierInfo()));
      ConsumeToken(); // Eat the identifier.
      
      assert(Tok.is(tok::colon) && "NextToken() not working properly!");
      ConsumeToken();
      return ParseInitializer();
    }
    
    // Otherwise, parse the assignment-expression.
    return ParseAssignmentExpression();
  }
  
  
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
      // When designation is empty, this can be '[' objc-message-expr ']'.  Note
      // that we also have the case of [4][foo bar], which is the gnu designator
      // extension + objc message send.
      SourceLocation StartLoc = ConsumeBracket();
      
      // If Objective-C is enabled and this is a typename or other identifier
      // receiver, parse this as a message send expression.
      if (getLang().ObjC1 && isTokObjCMessageIdentifierReceiver()) {
        // FIXME: Emit ext_gnu_missing_equal_designator for inits like
        // [4][foo bar].
        IdentifierInfo *Name = Tok.getIdentifierInfo();
        ConsumeToken();
        return ParseAssignmentExprWithObjCMessageExprStart(StartLoc, Name, 0);
      }
      
      // Note that we parse this as an assignment expression, not a constant
      // expression (allowing *=, =, etc) to handle the objc case.  Sema needs
      // to validate that the expression is a constant.
      ExprResult Idx = ParseAssignmentExpression();
      if (Idx.isInvalid) {
        SkipUntil(tok::r_square);
        return Idx;
      }
      
      // Given an expression, we could either have a designator (if the next
      // tokens are '...' or ']' or an objc message send.  If this is an objc
      // message send, handle it now.  An objc-message send is the start of 
      // an assignment-expression production.
      if (getLang().ObjC1 && Tok.isNot(tok::ellipsis) && 
          Tok.isNot(tok::r_square)) {
        // FIXME: Emit ext_gnu_missing_equal_designator for inits like
        // [4][foo bar].
        return ParseAssignmentExprWithObjCMessageExprStart(StartLoc, 0,Idx.Val);
      }
      
      // Handle the gnu array range extension.
      if (Tok.is(tok::ellipsis)) {
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
    }
  }
}


/// ParseBraceInitializer - Called when parsing an initializer that has a
/// leading open brace.
///
///       initializer: [C99 6.7.8]
///         '{' initializer-list '}'
///         '{' initializer-list ',' '}'
/// [GNU]   '{' '}'
///
///       initializer-list:
///         designation[opt] initializer
///         initializer-list ',' designation[opt] initializer
///
Parser::ExprResult Parser::ParseBraceInitializer() {
  SourceLocation LBraceLoc = ConsumeBrace();
  
  // We support empty initializers, but tell the user that they aren't using
  // C99-clean code.
  if (Tok.is(tok::r_brace)) {
    Diag(LBraceLoc, diag::ext_gnu_empty_initializer);
    // Match the '}'.
    return Actions.ActOnInitList(LBraceLoc, 0, 0, ConsumeBrace());
  }
  
  /// InitExprs - This is the actual list of expressions contained in the
  /// initializer.
  llvm::SmallVector<ExprTy*, 8> InitExprs;
  
  /// ExprDesignators - For each initializer, keep track of the designator that
  /// was specified for it, if any.
  InitListDesignations InitExprDesignations(Actions);

  bool InitExprsOk = true;
  
  while (1) {
    // Parse: designation[opt] initializer
    
    // If we know that this cannot be a designation, just parse the nested
    // initializer directly.
    ExprResult SubElt;
    if (!MayBeDesignationStart(Tok.getKind()))
      SubElt = ParseInitializer();
    else
      SubElt = ParseInitializerWithPotentialDesignator(InitExprDesignations,
                                                       InitExprs.size());

    // If we couldn't parse the subelement, bail out.
    if (!SubElt.isInvalid) {
      InitExprs.push_back(SubElt.Val);
    } else {
      InitExprsOk = false;
      
      // We have two ways to try to recover from this error: if the code looks
      // gramatically ok (i.e. we have a comma coming up) try to continue
      // parsing the rest of the initializer.  This allows us to emit
      // diagnostics for later elements that we find.  If we don't see a comma,
      // assume there is a parse error, and just skip to recover.
      if (Tok.isNot(tok::comma)) {
        SkipUntil(tok::r_brace, false, true);
        break;
      }
    }
      
    // If we don't have a comma continued list, we're done.
    if (Tok.isNot(tok::comma)) break;
    
    // TODO: save comma locations if some client cares.
    ConsumeToken();
    
    // Handle trailing comma.
    if (Tok.is(tok::r_brace)) break;
  }
  if (InitExprsOk && Tok.is(tok::r_brace))
    return Actions.ActOnInitList(LBraceLoc, &InitExprs[0], InitExprs.size(), 
                                 ConsumeBrace());
  
  // On error, delete any parsed subexpressions.
  for (unsigned i = 0, e = InitExprs.size(); i != e; ++i)
    Actions.DeleteExpr(InitExprs[i]);
  
  // Match the '}'.
  MatchRHSPunctuation(tok::r_brace, LBraceLoc);
  return ExprResult(true); // an error occurred.
}

