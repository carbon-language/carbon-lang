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
#include "clang/Parse/ParseDiagnostic.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"
using namespace clang;


/// MayBeDesignationStart - Return true if this token might be the start of a
/// designator.  If we can tell it is impossible that it is a designator, return
/// false.
static bool MayBeDesignationStart(tok::TokenKind K, Preprocessor &PP) {
  switch (K) {
  default: return false;
  case tok::period:      // designator: '.' identifier
  case tok::l_square:    // designator: array-designator
      return true;
  case tok::identifier:  // designation: identifier ':'
    return PP.LookAhead(0).is(tok::colon);
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
Parser::OwningExprResult Parser::ParseInitializerWithPotentialDesignator() {

  // If this is the old-style GNU extension:
  //   designation ::= identifier ':'
  // Handle it as a field designator.  Otherwise, this must be the start of a
  // normal expression.
  if (Tok.is(tok::identifier)) {
    const IdentifierInfo *FieldName = Tok.getIdentifierInfo();

    llvm::SmallString<256> NewSyntax;
    llvm::raw_svector_ostream(NewSyntax) << '.' << FieldName->getNameStr()
                                         << " = ";

    SourceLocation NameLoc = ConsumeToken(); // Eat the identifier.

    assert(Tok.is(tok::colon) && "MayBeDesignationStart not working properly!");
    SourceLocation ColonLoc = ConsumeToken();

    Diag(Tok, diag::ext_gnu_old_style_field_designator)
      << CodeModificationHint::CreateReplacement(SourceRange(NameLoc,
                                                             ColonLoc),
                                                 NewSyntax.str());

    Designation D;
    D.AddDesignator(Designator::getField(FieldName, SourceLocation(), NameLoc));
    return Actions.ActOnDesignatedInitializer(D, ColonLoc, true,
                                              ParseInitializer());
  }

  // Desig - This is initialized when we see our first designator.  We may have
  // an objc message send with no designator, so we don't want to create this
  // eagerly.
  Designation Desig;

  // Parse each designator in the designator list until we find an initializer.
  while (Tok.is(tok::period) || Tok.is(tok::l_square)) {
    if (Tok.is(tok::period)) {
      // designator: '.' identifier
      SourceLocation DotLoc = ConsumeToken();

      if (Tok.isNot(tok::identifier)) {
        Diag(Tok.getLocation(), diag::err_expected_field_designator);
        return ExprError();
      }

      Desig.AddDesignator(Designator::getField(Tok.getIdentifierInfo(), DotLoc,
                                               Tok.getLocation()));
      ConsumeToken(); // Eat the identifier.
      continue;
    }

    // We must have either an array designator now or an objc message send.
    assert(Tok.is(tok::l_square) && "Unexpected token!");

    // Handle the two forms of array designator:
    //   array-designator: '[' constant-expression ']'
    //   array-designator: '[' constant-expression '...' constant-expression ']'
    //
    // Also, we have to handle the case where the expression after the
    // designator an an objc message send: '[' objc-message-expr ']'.
    // Interesting cases are:
    //   [foo bar]         -> objc message send
    //   [foo]             -> array designator
    //   [foo ... bar]     -> array designator
    //   [4][foo bar]      -> obsolete GNU designation with objc message send.
    //
    SourceLocation StartLoc = ConsumeBracket();

    // If Objective-C is enabled and this is a typename or other identifier
    // receiver, parse this as a message send expression.
    if (getLang().ObjC1 && isTokObjCMessageIdentifierReceiver()) {
      // If we have exactly one array designator, this used the GNU
      // 'designation: array-designator' extension, otherwise there should be no
      // designators at all!
      if (Desig.getNumDesignators() == 1 &&
          (Desig.getDesignator(0).isArrayDesignator() ||
           Desig.getDesignator(0).isArrayRangeDesignator()))
        Diag(StartLoc, diag::ext_gnu_missing_equal_designator);
      else if (Desig.getNumDesignators() > 0)
        Diag(Tok, diag::err_expected_equal_designator);

      IdentifierInfo *Name = Tok.getIdentifierInfo();
      SourceLocation NameLoc = ConsumeToken();
      return ParseAssignmentExprWithObjCMessageExprStart(
                       StartLoc, NameLoc, Name, ExprArg(Actions));
    }

    // Note that we parse this as an assignment expression, not a constant
    // expression (allowing *=, =, etc) to handle the objc case.  Sema needs
    // to validate that the expression is a constant.
    OwningExprResult Idx(ParseAssignmentExpression());
    if (Idx.isInvalid()) {
      SkipUntil(tok::r_square);
      return move(Idx);
    }

    // Given an expression, we could either have a designator (if the next
    // tokens are '...' or ']' or an objc message send.  If this is an objc
    // message send, handle it now.  An objc-message send is the start of
    // an assignment-expression production.
    if (getLang().ObjC1 && Tok.isNot(tok::ellipsis) &&
        Tok.isNot(tok::r_square)) {

      // If we have exactly one array designator, this used the GNU
      // 'designation: array-designator' extension, otherwise there should be no
      // designators at all!
      if (Desig.getNumDesignators() == 1 &&
          (Desig.getDesignator(0).isArrayDesignator() ||
           Desig.getDesignator(0).isArrayRangeDesignator()))
        Diag(StartLoc, diag::ext_gnu_missing_equal_designator);
      else if (Desig.getNumDesignators() > 0)
        Diag(Tok, diag::err_expected_equal_designator);

      return ParseAssignmentExprWithObjCMessageExprStart(StartLoc,
                                                         SourceLocation(),
                                                         0, move(Idx));
    }

    // If this is a normal array designator, remember it.
    if (Tok.isNot(tok::ellipsis)) {
      Desig.AddDesignator(Designator::getArray(Idx.release(), StartLoc));
    } else {
      // Handle the gnu array range extension.
      Diag(Tok, diag::ext_gnu_array_range);
      SourceLocation EllipsisLoc = ConsumeToken();

      OwningExprResult RHS(ParseConstantExpression());
      if (RHS.isInvalid()) {
        SkipUntil(tok::r_square);
        return move(RHS);
      }
      Desig.AddDesignator(Designator::getArrayRange(Idx.release(),
                                                    RHS.release(),
                                                    StartLoc, EllipsisLoc));
    }

    SourceLocation EndLoc = MatchRHSPunctuation(tok::r_square, StartLoc);
    Desig.getDesignator(Desig.getNumDesignators() - 1).setRBracketLoc(EndLoc);
  }

  // Okay, we're done with the designator sequence.  We know that there must be
  // at least one designator, because the only case we can get into this method
  // without a designator is when we have an objc message send.  That case is
  // handled and returned from above.
  assert(!Desig.empty() && "Designator is empty?");

  // Handle a normal designator sequence end, which is an equal.
  if (Tok.is(tok::equal)) {
    SourceLocation EqualLoc = ConsumeToken();
    return Actions.ActOnDesignatedInitializer(Desig, EqualLoc, false,
                                              ParseInitializer());
  }

  // We read some number of designators and found something that isn't an = or
  // an initializer.  If we have exactly one array designator, this
  // is the GNU 'designation: array-designator' extension.  Otherwise, it is a
  // parse error.
  if (Desig.getNumDesignators() == 1 &&
      (Desig.getDesignator(0).isArrayDesignator() ||
       Desig.getDesignator(0).isArrayRangeDesignator())) {
    Diag(Tok, diag::ext_gnu_missing_equal_designator)
      << CodeModificationHint::CreateInsertion(Tok.getLocation(), "= ");
    return Actions.ActOnDesignatedInitializer(Desig, Tok.getLocation(),
                                              true, ParseInitializer());
  }

  Diag(Tok, diag::err_expected_equal_designator);
  return ExprError();
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
Parser::OwningExprResult Parser::ParseBraceInitializer() {
  SourceLocation LBraceLoc = ConsumeBrace();

  /// InitExprs - This is the actual list of expressions contained in the
  /// initializer.
  ExprVector InitExprs(Actions);

  if (Tok.is(tok::r_brace)) {
    // Empty initializers are a C++ feature and a GNU extension to C.
    if (!getLang().CPlusPlus)
      Diag(LBraceLoc, diag::ext_gnu_empty_initializer);
    // Match the '}'.
    return Actions.ActOnInitList(LBraceLoc, Action::MultiExprArg(Actions),
                                 ConsumeBrace());
  }

  bool InitExprsOk = true;

  while (1) {
    // Parse: designation[opt] initializer

    // If we know that this cannot be a designation, just parse the nested
    // initializer directly.
    OwningExprResult SubElt(Actions);
    if (MayBeDesignationStart(Tok.getKind(), PP))
      SubElt = ParseInitializerWithPotentialDesignator();
    else
      SubElt = ParseInitializer();

    // If we couldn't parse the subelement, bail out.
    if (!SubElt.isInvalid()) {
      InitExprs.push_back(SubElt.release());
    } else {
      InitExprsOk = false;

      // We have two ways to try to recover from this error: if the code looks
      // gramatically ok (i.e. we have a comma coming up) try to continue
      // parsing the rest of the initializer.  This allows us to emit
      // diagnostics for later elements that we find.  If we don't see a comma,
      // assume there is a parse error, and just skip to recover.
      // FIXME: This comment doesn't sound right. If there is a r_brace
      // immediately, it can't be an error, since there is no other way of
      // leaving this loop except through this if.
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
    return Actions.ActOnInitList(LBraceLoc, move_arg(InitExprs),
                                 ConsumeBrace());

  // Match the '}'.
  MatchRHSPunctuation(tok::r_brace, LBraceLoc);
  return ExprError(); // an error occurred.
}

